"""
===================
gprof_nn.validation
===================

This module defines functions to collect validation data from MRMS
and Kwajalein co-locations and GPROF retrievals.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from datetime import datetime
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from scipy.signal import convolve
from scipy.stats import binned_statistic_2d
from pansat.time import to_datetime64, TimeRange
from pansat.catalog.index import Index, find_matches
from pansat.utils import resample_data
from pansat.products.satellite.gpm import (
    l1c_r_gpm_gmi,
    l2a_gprof_gpm_gmi_v07a,
    l2a_gpm_dpr,
    l2b_gpm_cmb
)

from pyresample.geometry import SwathDefinition
from rich.progress import track
from satrain.metrics import (
    Bias,
    SMAPE,
    MAE,
    MSE,
    NRMSE,
    CorrelationCoef,
    PRCurve,
    SpectralCoherence
)
from tqdm import tqdm

from gprof_nn import sensors
from gprof_nn.coordinates import latlon_to_ecef
from gprof_nn.definitions import LIMITS
from gprof_nn.data.pretraining import simulate_tbs, SimulatorInput
from gprof_nn.data.training_data import decompress_and_load
from gprof_nn.data.retrieval import RetrievalFile
from gprof_nn.data.preprocessor import calculate_frozen_fraction
from gprof_nn.data.sim import SimFile
from gprof_nn.data.sim import apply_orographic_enhancement
from gprof_nn.utils import (
    calculate_interpolation_weights,
    interpolate,
    get_mask,
    calculate_smoothing_kernel
)
from gprof_nn.plotting import add_ticks, make_latlon_area
from gprof_nn import sensors
from gprof_nn.data.validation import CONUS


LOGGER = logging.getLogger(__name__)


def get_lon_lat_bins(lons: np.ndarray, lats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get longitude and latitude bins from center points.

    Args:
        lons: The array containing the longitudes.
        lats: The array containing the latitudes.

    Return:
        A tuple containing the longitude and latitude bins for the domain.
    """
    if lons.ndim == 2:
        lons = lons[0]

    if lats.ndim == 2:
        lats = lats[..., 0]

    lon_bins = np.zeros(lons.size + 1)
    lon_bins[1:-1] = 0.5 * (lons[1:] + lons[:-1])
    lon_bins[0] = lon_bins[1] - (lon_bins[2] - lon_bins[1])
    lon_bins[-1] = lon_bins[-2] + (lon_bins[-2] - lon_bins[-3])

    lat_bins = np.zeros(lats.size + 1)
    lat_bins[1:-1] = 0.5 * (lats[1:] + lats[:-1])
    lat_bins[0] = lat_bins[1] - (lat_bins[2] - lat_bins[1])
    lat_bins[-1] = lat_bins[-2] + (lat_bins[-2] - lat_bins[-3])

    return lon_bins, lat_bins


def get_timestamp(path: Path) -> datetime:
    """
    Get timestamp from collocation filename.

    Args:
        path: A Path object pointing to the collocation files or retrieval result files.

    Return:
        A Python datetime object representing the timestamp.
    """
    path = Path(path)
    parts = path.name.split("_")
    return datetime.strptime(parts[-1][:-3], "%Y%m%d%H%M%S")


def _hash_filenames(files: List[Path], length: int = 8) -> str:
    """
    Helper function to hash filenames.

    Args:
        files: A list containing the file paths.
        length: The length of the hash to product

    Return:
        A string representing the hash of the filenames in files.
    """
    h = hashlib.sha256()
    for f in sorted(map(str, files)):
        h.update(f.encode())
    return h.hexdigest()[:length]


class Evaluator:
    """
    The evaluator handles the evaluation of GPROF retrieval against collocation files.
    """
    def __init__(
            self,
            reference_files: Union[str, Path],
            retrieval_results: Dict[str, Union[str, Path]],
            precip_threshold: float = 1e-2
    ):
        """
        Args:
            reference_files: Path containing the reference files.
            retrieval_results: Dictionary mapping retrieval names to reference files.
        """
        reference_files = sorted(list(Path(reference_files).glob("**/*.nc")))
        self.reference_files = {}
        for path in reference_files:
            try:
                timestamp = get_timestamp(path)
                self.reference_files[timestamp] = path
            except ValueError as exc:
                raise exc
                continue

        self.retrieval_results = {}
        for name, path in retrieval_results.items():
            files = sorted(list(Path(path).glob("**/*.nc")))
            files += sorted(list(Path(path).glob("**/*.HDF5")))
            result_files = {}
            for path in files:
                try:
                    timestamp = get_timestamp(path)
                    result_files[timestamp] = path
                except ValueError:
                    continue
            self.retrieval_results[name] = result_files

        matched_times = set(self.reference_files.keys())
        for retrieval_times in self.retrieval_results.values():
            matched_times = matched_times.intersection(set(retrieval_times))
        self.matched_times = list(matched_times)
        self.precip_threshold = precip_threshold


    @property
    def total_precip(self):
        """
        Precipitation statistics for reference scenes.
        """
        ref_files = [self.reference_files[time] for time in self.matched_times]
        ref_files = ref_files

        file_hash = _hash_filenames(ref_files)
        stats = Path(".") / f"total_precip_{file_hash}.nc"

        if not stats.exists():
            total_precip = []
            for ref_file in tqdm(ref_files):
                with xr.open_dataset(ref_file, group="reference_data") as data:
                    precip = data.surface_precip.data
                    precip = precip[0 <= precip]
                    total_precip.append(precip.sum())
            total_precip = xr.Dataset({
                "total_precip": (("files",), np.array(total_precip))
            })
            total_precip.to_netcdf(stats)
            return total_precip
        return xr.load_dataset(stats)

    @property
    def center_coords(self):
        """
        Central coordinates of collocation scenes.
        """
        ref_files = [self.reference_files[time] for time in self.matched_times]
        ref_files = ref_files

        file_hash = _hash_filenames(ref_files)
        stats = Path(".") / f"center_coords_{file_hash}.nc"

        if not stats.exists():
            center_lons = []
            center_lats = []
            for ref_file in tqdm(ref_files):
                with xr.open_dataset(ref_file, group="reference_data") as data:
                    lons = data.longitude.data
                    lats = data.latitude.data
                    valid = np.isfinite(lons) * np.isfinite(lats)
                    center_lons.append(lons[valid].mean())
                    center_lats.append(lats[valid].mean())
            coords = xr.Dataset({
                "latitude": (("files",), np.array(center_lons)),
                "longitude": (("files",), np.array(center_lats))
            })
            coords.to_netcdf(stats)
            return coords
        return xr.load_dataset(stats)

    def get_reference_results(self, match_ind: int) -> xr.Dataset:
        """
        Load reference results for a given match index.
        """
        time = self.matched_times[match_ind]
        variables = [
            "surface_precip",
            "latitude",
            "longitude"
        ]
        with xr.open_dataset(self.reference_files[time], group="reference_data") as data:
            if "radar_quality_index" in data:
                variables.append("radar_quality_index")
            data = data[variables].compute()
        return data

    def get_reference_results_gridded(self, match_ind: int) -> xr.Dataset:
        """
        Load reference results for a given match index.
        """
        time = self.matched_times[match_ind]
        path = self.reference_files[time]
        path_gridded = path.parent.parent / "gridded" / path.name
        return xr.load_dataset(path_gridded, group="reference_data")

    def get_reference_results_gridded(self, match_ind: int) -> xr.Dataset:
        """
        Load reference results for a given match index.
        """
        time = self.matched_times[match_ind]
        ref_file = self.reference_files[time]
        ref_file = ref_file.parent.parent / "gridded" / ref_file.name
        return xr.load_dataset(ref_file, group="reference_data")

    def get_input_data_gridded(self, match_ind: int) -> xr.Dataset:
        """
        Load input data  for a given match index.
        """
        time = self.matched_times[match_ind]
        ref_file = self.reference_files[time]
        ref_file = ref_file.parent.parent / "gridded" / ref_file.name
        return xr.load_dataset(ref_file, group="input_data")

    def get_land_ocean_mask(self, match_ind: int) -> xr.Dataset:
        """
        Load land and ocean mask from reference data.
        """
        time = self.matched_times[match_ind]
        with xr.open_dataset(self.reference_files[time], group="input_data") as data:
            surface_type = data["surface_type"].data
            ocean_mask = (surface_type == 1) + ((13 <= surface_type) * (surface_type <= 13))
            land_mask = (2 < surface_type) * (surface_type < 8)
        return land_mask, ocean_mask

    def get_land_ocean_mask_gridded(self, match_ind: int) -> xr.Dataset:
        """
        Load land and ocean mask from reference data.
        """
        time = self.matched_times[match_ind]
        ref_file = self.reference_files[time]
        ref_file = ref_file.parent.parent / "gridded" / ref_file.name
        with xr.open_dataset(ref_file, group="input_data") as data:
            surface_type = data["surface_type"].data
            ocean_mask = (surface_type == 1)# + ((13 <= surface_type) * (surface_type <= 13))
            land_mask = (2 < surface_type) * (surface_type < 8)
        return land_mask, ocean_mask

    def get_retrieval_results(self, match_ind: int) -> xr.Dataset:
        time = self.matched_times[match_ind]
        results = {}
        for name, result_files in self.retrieval_results.items():
            results[name] = xr.load_dataset(result_files[time])
        return results

    def plot_results(
            self,
            match_ind: int,
            min_rqi: float = 0.9,
            bnds: Optional[Tuple[float, float, float, float]] = None,
            figsize: Optional[Tuple[float, float]] = None,
            print_metrics: bool = True,
            metrics_pos: Tuple[float, float] = (0.1, 0.9)
    ) -> plt.Figure:
        """
        Plot results for a given match.

        Args:
            match_ind: The index of the matched retrieval and reference files.
            min_rqi: Minimum required RQI to include pixels in calculation of retrieval metrics.
            bnds: Tuple defining the ROI to plot as (lon_min, lat_min, lon_max,lat_max).
            metrics_pos: Tuple containing the coordaintes of the metric box in axes-relative coordinates.

        Return:
            The matplotlib.Figure containing the retrieval results.
        """
        n_panels = len(self.retrieval_results) + 1

        if figsize is None:
            figsize = (n_panels * 8, 8)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, n_panels, height_ratios=[1.0, 0.1])
        norm = LogNorm(1e-2, 1e2)
        crs = ccrs.PlateCarree()

        time = self.matched_times[match_ind]
        reference = self.get_reference_results(match_ind)
        reference_gridded = self.get_reference_results_gridded(match_ind)
        results = self.get_retrieval_results(match_ind)

        if bnds is not None:
            lon_min, lat_min, lon_max, lat_max = bnds
            lon_mask = (
                (lon_min <= reference_gridded.longitude.data) *
                (reference_gridded.longitude.data <= lon_max)
            )
            lat_mask = (
                (lat_min <= reference_gridded.latitude.data) *
                (reference_gridded.latitude.data <= lat_max)
            )
            reference_gridded = reference_gridded[{
                "longitude": lon_mask,
                "latitude": lat_mask
            }]
        else:
            lat_mask = slice(0, None)
            lon_mask = slice(0, None)

        lons = reference.longitude.data
        lats = reference.latitude.data

        valid = 0 <= reference_gridded.surface_precip.data
        if "radar_quality_index" in reference_gridded:
            rqi = reference_gridded.radar_quality_index.data
            valid = valid * (0.9 < reference_gridded.radar_quality_index.data)
        else:
            rqi = np.ones_like(lons)
        lon_min = reference_gridded.longitude.data.min()
        lon_max = reference_gridded.longitude.data.max()
        lat_min = reference_gridded.latitude.data.min()
        lat_max = reference_gridded.latitude.data.max()
        lon_ticks = np.arange(np.round(lon_min), lon_max, 5.0)
        lat_ticks = np.arange(np.round(lat_min), lat_max, 5.0)

        area = make_latlon_area(
            lon_min, lat_min, lon_max, lat_max,
            reference_gridded.longitude.data.size,
            reference_gridded.latitude.data.size,
        )
        ext = area.area_extent
        ext = (ext[0], ext[2], ext[1], ext[3])

        ax = fig.add_subplot(gs[0, 0], projection=crs)
        sp_ref = reference_gridded.surface_precip.data
        m = ax.imshow(sp_ref, extent=ext, norm=norm, rasterized=True)
        ax.coastlines(color="grey")
        ax.set_title("(a) MRMS", loc="left")

        #if np.nanmin(rqi) < 0.9:
        #    rqi_levels = [0.9]
        #    cntr = ax.contour(
        #        rqi, levels=rqi_levels, linestyles=["--"], colors="white", extent=ext, origin="upper"
        #    )

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        add_ticks(ax, lon_ticks, lat_ticks, left=True, bottom=True)

        land_mask, ocean_mask = self.get_land_ocean_mask_gridded(match_ind)
        land_mask = land_mask[lat_mask][: , lon_mask]
        ocean_mask = ocean_mask[lat_mask][:, lon_mask]

        for ind, (name, res) in enumerate(results.items()):

            scan_inds = reference_gridded.scan_index
            pixel_inds = reference_gridded.pixel_index

            ax = fig.add_subplot(gs[0, ind + 1], projection=crs)
            var_names = [
                "surface_precip",
            ]
            if "latitude" in res.coords:
                res = res.drop_vars(("latitude", "longitude"))
            sp = res.surface_precip.data
            if "scan" in res.dims:
                res = res[var_names][{"scan": scan_inds, "pixel": pixel_inds}]
            else:
                res = res[var_names][{"scans": scan_inds, "pixels": pixel_inds}]
            invalid = pixel_inds.data < 0
            for var in var_names:
                if var in res:
                    res[var].data[invalid] = np.nan

            sp = res.surface_precip.data.copy()
            sp[sp < 0] = np.nan
            #sp = np.maximum(sp, 1e-3)

            m = ax.imshow(sp, norm=norm, extent=ext, rasterized=True)
            ax.contour(sp_ref, levels=[0.1, 1.0, 10.0], colors="w", extent=ext, origin="upper")

            # Metrics
            mask = np.isfinite(sp_ref) * np.isfinite(sp) * valid #land_mask * valid
            bias = np.mean(sp[mask] - sp_ref[mask]) / np.mean(sp_ref[mask]) * 100.0
            mse = np.mean((sp[mask] - sp_ref[mask]) ** 2)
            corr = np.corrcoef(sp[mask], sp_ref[mask])[0, 1]
            txt = f"Bias = {bias:.2f} %\nMSE = {mse:.2f} (mm h$^{{-1}}$)$^2$\nCorr. coef. = {corr:.2f}"

            print(name, " :: ", txt)
            if print_metrics:

                ax.text(
                    *metrics_pos, txt, transform=ax.transAxes, fontsize=20, color="black",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "black",
                        "alpha": 0.75
                    },
                    va="top",
                    ha="left"
                )

            add_ticks(ax, lon_ticks, lat_ticks, left=False, bottom=True)
            ax.coastlines(color="grey")
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)

            ax.set_title(f"({chr(ord('a') + ind + 1)}) {name}", loc="left")

        if len(results) < 3:
            cax = fig.add_subplot(gs[1, :])
        else:
            cax = fig.add_subplot(gs[1, 1:-1])

        plt.colorbar(m, label="Surface Precip [mm h$^{-1}$]", cax=cax, orientation="horizontal")
        fig.suptitle(time, fontsize=24)

        return fig


    def evaluate(
            self,
            lat_lims: Optional[Tuple[float, float]] = None,
            effective_resolution: bool = False
    ) -> None:
        """
        Iterates over scenes and calculates accuracy metrics for all retrievals.

        Args:
            lat_lims: Optional latitude limits to apply to the validation.
            effective_resolution: Set to 'True' to evaluate the effective resolution. This
                only works over dense reference data sources like MRMS.

        Return:
            A tuple of dictionaries ``results_land, results_ocean`` containing the validation results for ocean
            and land surfaces, respectively.
        """
        metric_classes = [
            Bias,
            SMAPE,
            MAE,
            MSE,
            CorrelationCoef,
            NRMSE,
        ]

        if effective_resolution:
            spectral_coherence = {
                name: SpectralCoherence(window_size=64) for name in self.retrieval_results.keys()
            }

        else:
            spectral_coherence = None

        metrics_land = {
            name: [cls() for cls in metric_classes] for name in self.retrieval_results.keys()
        }
        detection_metrics_land = {
            name: [PRCurve()] for name in self.retrieval_results.keys()
        }
        metrics_ocean = {
            name: [cls() for cls in metric_classes] for name in self.retrieval_results.keys()
        }
        detection_metrics_ocean = {
            name: [PRCurve()] for name in self.retrieval_results.keys()
        }

        desc = "Evaluating results"
        for match_ind in track(np.random.permutation(len(self.matched_times)), description=desc):

            try:

                reference = self.get_reference_results_gridded(match_ind)

                scan_inds = reference.scan_index
                pixel_inds = reference.pixel_index
                results_swath = self.get_retrieval_results(match_ind)
                results = {}
                for name, res in results_swath.items():
                    res = res.drop_vars(("latitude", "longitude"))
                    res = res[["surface_precip"]][{"scans": scan_inds, "pixels": pixel_inds}]
                    invalid = pixel_inds.data < 0
                    res["surface_precip"].data[invalid] = np.nan
                    results[name] = res

                sp_ref = reference.surface_precip.data
                valid_mask = 0 <= sp_ref
                if "radar_quality_index" in reference:
                    valid_mask *= (0.5 < reference.radar_quality_index.data)

                for res in results.values():
                    sp = res.surface_precip.data
                    valid_mask = valid_mask * (0 <= res.surface_precip.data)

                land_mask, ocean_mask = self.get_land_ocean_mask_gridded(match_ind)

                lats = reference.latitude.data
                if lat_lims is not None:
                    lat_min, lat_max = lat_lims
                    valid_lats = (lat_min <= lats) * (lats <= lat_max)
                    valid_mask *= valid_lats

                for name, res in results.items():

                    sp = res.surface_precip.data

                    if spectral_coherence is not None:
                        sp_ref_nan = sp_ref.copy()
                        sp_ref_nan[~valid_mask] = np.nan
                        spectral_coherence[name].update(sp, sp_ref_nan)

                    for metric in metrics_ocean[name]:
                        metric.update(sp[valid_mask * ocean_mask], sp_ref[valid_mask * ocean_mask])
                    for metric in metrics_land[name]:
                        metric.update(sp[valid_mask * land_mask], sp_ref[valid_mask * land_mask])

                    if "probability_of_precipitation" in res or "probability_of_precip" in res:
                        if "probability_of_precipitation" in res:
                            pop = res.probability_of_precipitation.data
                        else:
                            pop = res.probability_of_precip.data

                        for metric in detection_metrics_ocean[name]:
                            metric.update(pop[valid_mask * ocean_mask], self.precip_threshold <= sp_ref[valid_mask * ocean_mask])
                        for metric in detection_metrics_land[name]:
                            metric.update(pop[valid_mask * land_mask], self.precip_threshold <= sp_ref[valid_mask * land_mask])

            except Exception:
                LOGGER.exception(
                    f"Failed to process match {match_ind}."
                )

        if spectral_coherence is not None:
            res_sc = {
                name: [metric.compute()] for name, metric in spectral_coherence.items()
            }
        else:
            res_sc = {
                name: [] for name, metric in spectral_coherence.items()
            }

        results_land = {
            name: xr.merge(res_sc[name] + [metric.compute() for metric in metrics + detection_metrics_land[name]])
            for name, metrics in metrics_land.items()
        }
        results_ocean = {
            name: xr.merge(res_sc[name] + [metric.compute() for metric in metrics + detection_metrics_ocean[name]])
            for name, metrics in metrics_ocean.items()
        }

        return results_land, results_ocean




    def calculate_spatial_statistics(self, area) -> Dict[str, xr.Dataset]:
        from pansat.utils import resample_data
        from tqdm import tqdm

        lons, lats = area.get_lonlats()
        rqi_sum = np.zeros_like(lons)
        rqi_cts = np.zeros_like(lats)
        ocean_sum = np.zeros_like(lons)
        ocean_cts = np.zeros_like(lats)

        names = ["reference"] + list(self.retrieval_results.keys())
        stats = {
            name: {
                "sp_sum": np.zeros_like(lons),
                "sp_cts": np.zeros_like(lons),
                "pf_sum": np.zeros_like(lons),
                "pf_cts": np.zeros_like(lons),
            } for name in names
        }
        stats["reference"]["sp_nc_sum"] = np.zeros_like(lons)
        stats["reference"]["sp_nc_cts"] = np.zeros_like(lons)

        desc = "Evaluating results"
        for match_ind in track(np.random.permutation(len(self.matched_times)), description=desc):

            reference = self.get_reference_results(match_ind)
            results = self.get_retrieval_results(match_ind)
            sp_ref = reference.surface_precip.data

            land_mask, ocean_mask = self.get_land_ocean_mask(match_ind)

            reference["ocean_mask"] = (("scan", "pixel"), ocean_mask.astype(np.float32))
            reference = resample_data(
                reference[["radar_quality_index", "surface_precip", "ocean_mask", "gauge_correction_factor"]].compute(),
                area,
                radius_of_influence=15e3
            )
            rqi = reference.radar_quality_index.data
            rqi[rqi < 0] = np.nan
            rqi_sum += np.nan_to_num(rqi, nan=0.0, copy=True)
            rqi_cts += np.isfinite(rqi)

            gcf = reference.gauge_correction_factor.data

            om = reference.ocean_mask.data
            ocean_sum += np.nan_to_num(om, nan=0.0, copy=True)
            ocean_cts += np.isfinite(om)

            sp_ref = reference.surface_precip.data
            sp_ref[(sp_ref < 0) + (rqi < 0.99)] = np.nan
            stats["reference"]["sp_sum"] += np.nan_to_num(sp_ref, nan=0.0, copy=True)
            stats["reference"]["sp_cts"] += np.isfinite(sp_ref)
            stats["reference"]["sp_nc_sum"] += np.nan_to_num(sp_ref / gcf, nan=0.0, copy=True)
            stats["reference"]["sp_nc_cts"] += np.isfinite(sp_ref / gcf)

            pf = (1e-2 < sp_ref).astype(np.float32)
            pf[np.isnan(sp_ref)] = np.nan
            stats["reference"]["pf_sum"] += np.nan_to_num(pf, nan=0.0, copy=True)
            stats["reference"]["pf_cts"] += np.isfinite(pf)

            for name, res in results.items():
                if "probability_of_precipitation" in res:
                    pop_var = "probability_of_precipitation"
                else:
                    pop_var = "probability_of_precip"

                res = resample_data(res[["latitude", "longitude", "surface_precip", pop_var]], area, radius_of_influence=15e3)
                sp = res.surface_precip.data
                mask = np.isfinite(sp_ref) * (0.0 <= sp) * (0.99 < rqi)
                sp[~mask] = np.nan
                stats[name]["sp_sum"] += np.nan_to_num(sp, nan=0.0, copy=True)
                stats[name]["sp_cts"] += np.isfinite(sp)
                pop = res[pop_var].data
                pop = (0.1 < pop).astype(np.float32)
                pop[~mask] = np.nan
                stats[name]["pf_sum"] += np.nan_to_num(pop, nan=0.0, copy=True)
                stats[name]["pf_cts"] += np.isfinite(pop)

        results = {}
        for name, stats_n in stats.items():
            results[name] = xr.Dataset({
                "longitude": (("y", "x"), lons),
                "latitude": (("y", "x"), lats),
                "surface_precip": (("y", "x"), stats_n["sp_sum"] / stats_n["sp_cts"]),
                "precip_fraction": (("y", "x"), stats_n["pf_sum"] / stats_n["pf_cts"])
            })
        results["reference"]["surface_precip_nc"] = (
            ("y", "x"),
            stats["reference"]["sp_nc_sum"] / stats["reference"]["sp_nc_cts"]
        )
        results["reference"]["rqi"] = (("y", "x"), rqi_sum / rqi_cts)
        results["reference"]["ocean_mask"] = (("y", "x"), ocean_sum / ocean_cts)
        return results


def _get_sim_file_start_and_end_time(path: Path) -> Tuple[np.datetime64, np.datetime64]:
    """
    Extract start and end time from a sim file.
    """
    parts = Path(path).name.split('_')
    start_time = to_datetime64(datetime.strptime(parts[-2], "%Y%m%d%H%M%S"))
    end_time = to_datetime64(datetime.strptime(parts[-1][:-3], "%Y%m%d%H%M%S"))
    return start_time, end_time


class SimulatorEvaluator:
    """
    Evaluator class for evaluating GPROF V08 simulations.
    """
    def __init__(
            self,
            sim_file_path: Union[str, Path],
            collocation_path: Union[str, Path],
            target_product: "pansat.Product",
            target_sensor: "gprof_nn.sensors.Sensor"
    ):
        sim_files = sorted(
            list(Path(sim_file_path).glob("**/*.nc"))
        )
        sim_files_valid = []
        times_valid = []
        for sim_path in sim_files:
            try:
                times = _get_sim_file_start_and_end_time(sim_path)
            except Exception as exc:
                raise exc
                continue
            sim_files_valid.append(sim_path)
            times_valid.append(np.array(times))

        self.sim_files = np.stack(sim_files_valid)
        self.times = np.array(times_valid)

        self.sim_start_times = self.times[:, 0]
        self.sim_end_times = self.times[:, 1]
        self.colloc_files = sorted(list(Path(collocation_path).glob("**/*.nc")))
        self.target_product = target_product
        self.target_sensor = target_sensor

    def load_full_gmi_obs(self, time: np.datetime64) -> Tuple[Path, xr.Dataset]:
        """
        Load full GPM GMI observations for a given time.

        Args:
            time: The time for which to load the GPM L1C observations.

        Return:
            A xarray.Dataset containing the observation data.
        """
        from pansat.products.satellite.gpm import l1c_r_gpm_gmi
        rec = l1c_r_gpm_gmi.get(time)
        gmi_data = l1c_r_gpm_gmi.open(rec[0])
        return rec[0].local_path, gmi_data

    def get_gmi_file(self, index: int) -> Path:
        colloc_file = self.colloc_files[index]
        with xr.open_dataset(colloc_file, group="input_data") as colloc_data:
            colloc_data = colloc_data[
                ["observations_gprof", "latitude", "longitude", "scan_time"]
            ].compute()
        time = colloc_data.scan_time.mean().data
        recs = l1c_r_gpm_gmi.get(time)
        return recs[0].local_path

    def get_sim_file(self, time: np.datetime64) -> Optional[Path]:
        """
        Find sim file coverging a given time.

        Args:
            time: A numpy.datetime64 object defining the time.

        Returns:
            A path pointing to the sim file covering the given time or 'None' if no such file is available.
        """
        mask = (self.sim_start_times <= time) * (time <= self.sim_end_times)
        if not mask.any():
            return None
        ind = np.where(mask)[0][0]
        return self.sim_files[ind]


    def run_satformer(
            self,
            target_file: Path,
            gmi_file: Path,
            time_range: TimeRange
    ) -> xr.Dataset:
        """
        Simulate Tbs for matchup.

        Args:
            target_file: The  file containing the matchup.
            gmi_file: The GMI files containing the matchup.

        Return:
            A xarray.Dataset containing the simulated observations.
        """
        target_index = Index.index(self.target_product, [target_file])
        gmi_index = Index.index(l1c_r_gpm_gmi, [gmi_file]).subset(
            time_range=time_range
        )
        matches = find_matches(gmi_index, target_index)
        input_granule, target_granules = matches[-1]
        input_data = input_granule.open()
        lats = input_data.latitude_s1
        lons = input_data.latitude_s1

        results = simulate_tbs(
            "/gdata1/simon/gprof_v8/models/simulator_v2/gprof_nn_sim.pt",
            sensors.GMI,
            input_granule,
            self.target_sensor,
            target_granules,
            device="cuda:1"
        )
        return results

    def get_sf_input_loader(
            self,
            index: int
    ) -> xr.Dataset:
        """
        Get the input loader to load SatFormer input data.

        Args:
            index: The index identifying the matchup.
        """
        colloc_file = self.colloc_files[index]
        with xr.open_dataset(colloc_file, group="input_data") as colloc_data:
            colloc_data = colloc_data[
                ["observations_gprof", "latitude", "longitude", "scan_time"]
            ].compute()
        lons = colloc_data.longitude.data
        lats = colloc_data.latitude.data
        if lons.ndim < 2:
            lons, lats = np.meshgrid(lons, lats)

        time = colloc_data.scan_time.mean().data
        target_file = self.target_product.get(time)[0]
        gmi_file, gmi_obs = self.load_full_gmi_obs(time)
        gmi_obs = gmi_obs.rename(latitude_s1="latitude", longitude_s1="longitude")

        time_range = TimeRange(
            colloc_data.scan_time.min().data,
            colloc_data.scan_time.max().data
        )
        target_index = Index.index(self.target_product, [target_file])
        gmi_index = Index.index(l1c_r_gpm_gmi, [gmi_file]).subset(
            time_range=time_range
        )
        matches = find_matches(gmi_index, target_index)
        input_granule, target_granules = matches[-1]
        input_data = input_granule.open()
        lats = input_data.latitude_s1
        lons = input_data.latitude_s1

        input_loader = SimulatorInput(sensors.GMI, input_granule, self.target_sensor, target_granules)
        return input_loader


    def get_match(
            self,
            index,
            run_sf: bool = False,
            resample: bool = True
    ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, Optional[xr.Dataset]]:
        """
        Extract observation match.

        Args:
            index: The index of the collocation to match.
            run_sf: Set to True to run Satformer to get full swath simulations.
            resample: Whether to resample the data to the target sensor grid.

        Returns:
            A four-tuple containing collocation data, the sim-file data, the original GMI observations,
            and optionally the satformer results.
        """
        colloc_file = self.colloc_files[index]
        with xr.open_dataset(colloc_file, group="input_data") as colloc_data:
            colloc_data = colloc_data[
                ["observations_gprof", "earth_incidence_angle", "latitude", "longitude", "scan_time"]
            ].compute()
        lons = colloc_data.longitude.data
        lats = colloc_data.latitude.data
        if lons.ndim < 2:
            lons, lats = np.meshgrid(lons, lats)

        swath = SwathDefinition(lons=lons, lats=lats)
        time = colloc_data.scan_time.mean().data

        sim_file = self.get_sim_file(time)
        if sim_file is None:
            return None

        with xr.open_dataset(sim_file) as sim_data:
            sim_data = sim_data[[
                "simulated_brightness_temperatures",
                "brightness_temperature_biases",
                "satformer_tbs",
                "longitude",
                "latitude"
            ]].compute()

        sim_swath = SwathDefinition(sim_data.longitude.data, sim_data.latitude.data)

        if resample:

            inds = None
            if "angles" in sim_data:
                angles = sim_data.angles
                d_ang = np.abs(angles - np.abs(colloc_data.earth_incidence_angle[{"channel": 0}]))
                inds = d_ang.argmin("angles")

            sim_data = resample_data(sim_data, swath, radius_of_influence=20e3, new_dims=("scan", "pixel"))

            if inds is not None:
                sim_data = sim_data[{"angles": inds}]

        gmi_file, gmi_obs = self.load_full_gmi_obs(time)
        gmi_obs = gmi_obs.rename(latitude_s1="latitude", longitude_s1="longitude")
        tbs = xr.concat((gmi_obs.tbs_s1.rename(channels_s1="channels"), gmi_obs.tbs_s2.rename(channels_s2="channels")), dim="channels")
        gmi_obs["tbs"] = tbs
        if resample:
            gmi_obs = resample_data(gmi_obs[["tbs"]], swath, radius_of_influence=20e3)
        else:
            gmi_obs = gmi_obs[["tbs"]]

        if not run_sf:
            return colloc_data, sim_data, gmi_obs, None

        time_range = TimeRange(
            colloc_data.scan_time.min().data,
            colloc_data.scan_time.max().data
        )
        target_file = self.target_product.get(time)[0]
        results_satformer = self.run_satformer(target_file, gmi_file, time_range)
        if resample:
            results_satformer = resample_data(
                results_satformer,
                swath,
                radius_of_influence=20e3
            )

        return colloc_data, sim_data, gmi_obs, results_satformer



class PrecipDist:
    """
    Helper class to calculate precipitation distributions.
    """
    def __init__(
            self,
            resolution: float = 5.0,
            precip_threshold: float = 1e-2
    ):
        """
        Args:
            resolution: The resolution at which to collect the distribution.
            precip_threshold: Minimum precipiation for a pixel to be considered raining.
        """
        self.precip_threshold = precip_threshold

        precip_bins = np.logspace(-3, np.log10(200), 101)
        precip_bins[0] = 0.0
        self.precip_bins = precip_bins

        self.lon_bins = np.linspace(-180, 180, int(360 / resolution + 1))
        self.lat_bins = np.linspace(-90, 90, int(360 / resolution + 1))

        m = self.lon_bins.size - 1
        n = self.lat_bins.size - 1

        self.cts_precip = np.zeros(self.precip_bins.size - 1)
        self.acc = np.zeros((m, n))
        self.occ = np.zeros((m, n))
        self.cts = np.zeros((m, n))

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            precip: np.ndarray,
    ):
        """
        Collect precipitation statistics.

        Args:
            lons: An array containing the longitudes.
            lats: An array containingg the latitudes.
            precip: The precipitation values.
        """

        valid = (0.0 <= precip)
        self.acc += np.histogram2d(
            lats[valid],
            lons[valid],
            weights=precip[valid],
            bins=(self.lat_bins, self.lon_bins)
        )[0]
        self.occ += np.histogram2d(
            lats[valid],
            lons[valid],
            weights=self.precip_threshold <= precip[valid],
            bins=(self.lat_bins, self.lon_bins)
        )[0]
        self.cts += np.histogram2d(
            lats[valid],
            lons[valid],
            bins=(self.lat_bins, self.lon_bins)
        )[0]
        self.cts_precip += np.histogram(
            precip[valid],
            bins=self.precip_bins
        )[0]


    def compute(self) -> xr.Dataset:
        """
        Compute precipitation distributions.
        """

        lons = 0.5 * (self.lon_bins[:-1] + self.lon_bins[1:])
        lats = 0.5 * (self.lat_bins[:-1] + self.lat_bins[1:])

        d_bins = np.diff(self.precip_bins)
        pdf = self.cts_precip / self.cts_precip.sum() / d_bins
        surface_precip = 0.5 * (self.precip_bins[1:] + self.precip_bins[:-1])

        occurrence = self.occ / self.cts
        occurrence_zonal = self.occ.sum(-1) / self.cts.sum(-1)

        results = xr.Dataset({
            "longitude": (("longitude",), lons),
            "latitude": (("latitude",), lats),
            "surface_precip": (("surface_precip",), surface_precip),
            "surface_precip_global": (
                ("latitude", "longitude"), self.acc / self.cts
            ),
            "surface_precip_zonal": (
                ("latitude",), self.acc.sum(-1) / self.cts.sum(-1)
            ),
            "occurrence_global": (("latitude", "longitude"), occurrence),
            "occurrence_zonal": (("latitude",), occurrence_zonal),
            "surface_precip_dist": (
                ("surface_precip",), pdf
            ),
        })
        return results


def plot_zonal_means(
        results: Dict[str, xr.Dataset],
        title: str = "Zonal Means",
        smooth: Optional[int] = None,
        show_totals: bool = False,
        colors: Optional[Dict[str, str]] = None,
        linestyles: Optional[Dict[str, str]] = None,
) -> plt.Figure:

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 2, width_ratios=[1.0, 0.2])

    ax = fig.add_subplot(gs[0, 0])

    lats = next(iter(results.values())).latitude.data
    weights = np.cos(np.deg2rad(lats))
    #weights[40 < np.abs(lats)] = 0.0
    #
    if colors is None:
        colors = {name: f"C{ind}" for ind, name in enumerate(results)}

    if linestyles is None:
        linestyles = {name: f"-" for ind, name in enumerate(results)}

    handles = []
    totals = {}
    for name, res in results.items():
        spz = 24.0 * res.surface_precip_zonal

        if smooth is None:
            handles += ax.plot(spz, lats, label=name, ls=linestyles[name], color=colors[name])
        else:
            k = np.ones(smooth) / smooth
            spz_s = convolve(spz, k, mode="same")
            handles += ax.plot(spz_s, lats, label=name, ls=linestyles[name], color=colors[name])

        spg = res.surface_precip_global
        weights = np.cos(np.deg2rad(res.latitude))
        total = spg.weighted(weights).mean().data
        totals[name] = 24 * total

    ax.set_xlim(0, 24 * 0.4)
    ax.set_xlabel(r"Precipitation Rate [mm h$^{-1}$]")
    ax.set_ylabel(r"Latitude [$^\circ$ N]")
    ax.set_title(title)
    ax.set_ylim(-70, 70)

    if show_totals:
        txt = "Totals:\n" + "\n".join([f"{name}: {tot:.3} mm D$^{{-1}}$" for name, tot in totals.items()])
        transform = lax.transAxes
        lax.text(0.05, 0.25, txt, va="center", ha="left", fontsize=14)

        lax = fig.add_subplot(gs[0, 1])
        lax.set_axis_off()
        lax.legend(handles=handles, loc="upper center")
    else:
        lax = fig.add_subplot(gs[0, 1])
        lax.set_axis_off()
        lax.legend(handles=handles, loc="center left")

    return fig


def plot_global_dists(
        results: Dict[str, xr.Dataset],
        title: str = "Global Distributions",
) -> plt.Figure:
    """
    Plot Global Precipitation Distributions
    """
    n_rows = len(results)
    n_cols = 2
    gs = GridSpec(n_rows + 1, n_cols + 1, height_ratios=[1.0] * n_rows + [0.1], width_ratios=[0.05, 1.0, 1.0])
    fig = plt.figure(figsize=(20, n_rows * 5))
    crs = ccrs.PlateCarree()

    norm = Normalize(0, 12)
    diff_norm = Normalize(-2, 2)

    res_ref = next(iter(results.values()))
    sp_ref = 24.0 * res_ref.surface_precip_global.data
    lats = res_ref.latitude
    valid_cols = np.where(np.isfinite(sp_ref).any(1))[0]
    lat_min = lats[valid_cols[0]]
    lat_max = lats[valid_cols[-1]]

    name_ref = next(iter(results.keys()))

    lat_ticks = np.linspace(-60, 60, 5)
    lon_ticks = np.linspace(-180, 180, 10)

    for ind, (name, res) in enumerate(results.items()):

        ax = fig.add_subplot(gs[ind, 0])
        ax.set_axis_off()
        ax.text(0, 0, s=name, rotation=90, ha="center", va="center", fontsize=12)
        ax.set_ylim(-2, 2)

        spg = 24.0 * res.surface_precip_global
        ax = fig.add_subplot(gs[ind, 1], projection=crs)
        lons = res.longitude.data
        lats = res.latitude.data

        levels = np.arange(13)
        m_avg = ax.contourf(lons, lats, spg, norm=norm, cmap="Blues", rasterized=True, levels=levels, extend="both")
        ax.coastlines(color="grey")
        ax.set_ylim(lat_min, lat_max)

        add_ticks(ax, lons=lon_ticks, lats=lat_ticks, left=True, bottom=ind == len(results) - 1)

        if ind == 0:
            ax.set_title("Average Daily Precipitation")

        ax = fig.add_subplot(gs[ind, 2], projection=crs)
        if ind == 0:
            ax.set_axis_off()
            ax.set_title(f"Difference w.r.t {name_ref}")
            continue

        diff = spg - sp_ref
        lons = res.longitude.data
        lats = res.latitude.data

        levels = np.linspace(-2.0, 2.0, 21)[:-1] + 0.1
        m_diff = ax.contourf(lons, lats, diff, norm=diff_norm, cmap="coolwarm_r", rasterized=True, levels=levels, extend="both")
        ax.coastlines(color="grey")
        ax.set_ylim(lat_min, lat_max)

        add_ticks(ax, lons=lon_ticks, lats=lat_ticks, left=False, bottom=ind == len(results) - 1)

    cax = fig.add_subplot(gs[-1, 1])
    plt.colorbar(m_avg, label=r"Surface Precip [mm h$^{-1}$]", orientation="horizontal", cax=cax)

    cax = fig.add_subplot(gs[-1, 2])
    plt.colorbar(m_diff, label=r"Difference [mm h$^{-1}$]", orientation="horizontal", cax=cax)
    cax.xaxis.set_ticks(np.linspace(-1.8, 1.8, 10)[:-1] + 0.2)

    return fig


def plot_precip_pdfs(
        results: Dict[str, xr.Dataset],
        title: str = "Precipitation Distributions",
):
    """
    Plot Precipitation PDFs

    Args:
        results: Dictionary containing the precipitation distribution results.
        title: Title for the plot.

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    totals = {}
    for name, res in results.items():
        x = res.surface_precip.data
        y = res.surface_precip_dist.data
        ax.plot(x, y, label=name)
        total = 24 * np.trapz(x * y, x=x)
        totals[name] = total

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    txt = "\n".join([f"{name}: {tot:.3} mm D$^{{-1}}$" for name, tot in totals.items()])
    ax.text(1e-3, 1e-4, txt, va="top")


def plot_precip_vol_dist(
        results: Dict[str, xr.Dataset],
        title: str = "Precipitation Distributions"
):
    """
    Plot Precipitation PDFs

    Args:
        results: Dictionary containing the precipitation distribution results.
        title: Title for the plot.

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    totals = {}
    for name, res in results.items():
        x = res.surface_precip.data
        pdf = res.surface_precip_dist.data
        y = cumulative_trapezoid(x * pdf, x=x)
        ax.plot(x[1:], y / y[-1], label=name)

    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.legend()


def load_gprof_v7_results(collocation_file: Path) -> xr.Dataset:
    """
    Load GPROF V7 results.

    Args:
        collocation_file: A path object pointing to a collocation file.

    Return:
         The retrieval results as a xarray.Dataset
    """
    date = datetime.strptime(collocation_file.name.split("_")[-1], "%Y%m%d%H%M%S.nc")
    with xr.open_dataset(collocation_file, group="input_data") as data:
        scan_start = data.attrs["scan_start"]
        scan_end = data.attrs["scan_end"]
        gprof_recs = l2a_gprof_gpm_gmi_v07a.get(date)
        gprof_data = l2a_gprof_gpm_gmi_v07a.open(gprof_recs[0])[{"scans": slice(scan_start, scan_end)}]

        time_ref = l1c_r_gpm_gmi.get_temporal_coverage(data.attrs["gpm_input_file"])
        time_ret = gprof_recs[0].temporal_coverage
        assert time_ref.start == time_ret.start
        assert time_ref.end == time_ret.end

    gprof_data = gprof_data.rename(surface_precipitation="surface_precip").drop_vars(("latitude", "longitude"))
    scan_time = gprof_data.scan_time.mean().data
    time_diff = abs((scan_time.astype("datetime64[s]").item() - date).total_seconds())
    if 15 < time_diff // 60:
        raise ValueError(
            "GPROF V7 file not matching!"
        )
    sp = gprof_data.surface_precip.data
    sp[sp < 0] = np.nan
    return gprof_data


def load_gpm_cmb_results(collocation_file: Path) -> xr.Dataset:
    """
    Load GPM CMBresults.

    Args:
        collocation_file: A path object pointing to a collocation file.

    Return:
         The retrieval results as a xarray.Dataset
    """
    date = datetime.strptime(collocation_file.name.split("_")[-1], "%Y%m%d%H%M%S.nc")
    with xr.open_dataset(collocation_file, group="input_data") as data:
        scan_start = data.attrs["scan_start"]
        scan_end = data.attrs["scan_end"]
        gpm_recs = l2b_gpm_cmb.get(date)
        gpm_data = l2b_gpm_cmb.open(gpm_recs[0])

        lons = data.longitude.data
        lats = data.latitude.data

    swath = SwathDefinition(lons, lats)
    gpm_data = gpm_data[["latitude", "longitude", "estim_surf_precip_tot_rate"]]
    gpm_data = resample_data(gpm_data, swath, radius_of_influence=5e3).rename(
        estim_surf_precip_tot_rate="surface_precip"
    )
    sp = gpm_data.surface_precip.data
    sp[sp < 0] = np.nan
    return gpm_data
