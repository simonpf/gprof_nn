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
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from scipy.signal import convolve
from pansat.time import to_datetime64, TimeRange
from pansat.catalog.index import Index, find_matches
from pansat.utils import resample_data
from pansat.products.satellite.gpm import l1c_r_gpm_gmi
from pyresample.geometry import SwathDefinition
from rich.progress import track
from satrain.metrics import Bias, MSE, CorrelationCoef, PRCurve
from tqdm import tqdm

from gprof_nn import sensors
from gprof_nn.coordinates import latlon_to_ecef
from gprof_nn.definitions import LIMITS
from gprof_nn.data.pretraining import simulate_tbs
from gprof_nn.data.training_data import decompress_and_load
from gprof_nn.data.retrieval import RetrievalFile
from gprof_nn.data.sim import SimFile
from gprof_nn.data.sim import apply_orographic_enhancement
from gprof_nn.utils import (
    calculate_interpolation_weights,
    interpolate,
    get_mask,
    calculate_smoothing_kernel
)
from gprof_nn import sensors
from gprof_nn.data.validation import CONUS


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
        return xr.load_dataset(self.reference_files[time], group="reference_data")

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

    def get_retrieval_results(self, match_ind: int) -> xr.Dataset:
        time = self.matched_times[match_ind]
        results = {}
        for name, result_files in self.retrieval_results.items():
            results[name] = xr.load_dataset(result_files[time])
        return results

    def plot_results(self, match_ind: int, min_rqi: float = 1.0 ) -> plt.Figure:
        """
        Plot results for a given match.

        Args:
            match_ind: The index of the matched retrieval and reference files.

        Return:
            The matplotlib.Figure containing the retrieval results.
        """
        n_panels = len(self.retrieval_results) + 1
        fig = plt.figure(figsize=(n_panels * 4, 4))
        gs = GridSpec(1, n_panels + 1, width_ratios=[1.0] * n_panels + [0.075])
        norm = LogNorm(1e-2, 1e2)
        crs = ccrs.PlateCarree()

        time = self.matched_times[match_ind]
        reference = self.get_reference_results(match_ind)
        results = self.get_retrieval_results(match_ind)

        ax = fig.add_subplot(gs[0, 0], projection=crs)
        lons = reference.longitude.data
        lats = reference.latitude.data
        sp_ref = reference.surface_precip.data
        m = ax.pcolormesh(lons, lats, np.maximum(sp_ref, 1e-3), norm=norm)
        ax.coastlines(color="grey")
        valid = 0 <= sp_ref

        if "radar_quality_index" in reference:
            rqi = reference["radar_quality_index"].data
            levels = [0.5, 0.9, 0.999]
            colors = "w"
            linestyles = ["-", "--", ":"]
            ax.contour(lons, lats, rqi, levels=levels, colors=colors, linestyles=linestyles)
            valid *= (min_rqi <= rqi)

        valid_lons = lons[valid]
        valid_lats = lats[valid]
        lon_min = valid_lons.min()
        lon_max = valid_lons.max()
        lat_min = valid_lats.min()
        lat_max = valid_lats.max()

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        land_mask, ocean_mask = self.get_land_ocean_mask(match_ind)

        for ind, (name, res) in enumerate(results.items()):
            ax = fig.add_subplot(gs[0, ind + 1], projection=crs)
            lons = res.longitude.data
            lats = res.latitude.data
            sp = np.maximum(res.surface_precip.data, 1e-3)
            m = ax.pcolormesh(lons, lats, sp, norm=norm)

            mask = np.isfinite(sp_ref) * np.isfinite(sp) * valid
            bias = np.mean(sp[mask] - sp_ref[mask]) / np.mean(sp_ref[mask]) * 100.0
            mse = np.mean((sp[mask] - sp_ref[mask]) ** 2)
            corr = np.corrcoef(sp[mask], sp_ref[mask])[0, 1]
            txt = f"MSE = {mse:.2f}\nCorr. coef. = {corr:.2f}"
            ax.text(0.65, 0.8, txt, transform=ax.transAxes, fontsize=8, color="grey")

            ax.coastlines(color="grey")
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)

            ax.contour(lons, lats, ocean_mask.astype(np.float32), level=[0.5], colors="salmon")

        cax = fig.add_subplot(gs[0, -1])
        plt.colorbar(m, label="Surface Precip [mm h$^{-1}$]", cax=cax)

        fig.suptitle(time)


    def evaluate(self) -> None:
        """
        Iterates over scenes and calculates accuracy metrics for all retrievals.

        Return:
            A tuple of dictionaries ``results_land, results_ocean`` containing the validation results for ocean
            and land surfaces, respectively.
        """

        metrics_land = {
            name: [Bias(), MSE(), CorrelationCoef()] for name in self.retrieval_results.keys()
        }
        detection_metrics_land = {
            name: [PRCurve()] for name in self.retrieval_results.keys()
        }
        metrics_ocean = {
            name: [Bias(), MSE(), CorrelationCoef()] for name in self.retrieval_results.keys()
        }
        detection_metrics_ocean = {
            name: [PRCurve()] for name in self.retrieval_results.keys()
        }

        desc = "Evaluating results"
        for match_ind in track(np.random.permutation(len(self.matched_times)), description=desc):

            reference = self.get_reference_results(match_ind)
            results = self.get_retrieval_results(match_ind)
            sp_ref = reference.surface_precip.data


            valid_mask = 0 <= sp_ref
            if "radar_quality_index" in reference:
                valid_mask *= (0.999 < reference.radar_quality_index.data)

            for res in results.values():
                valid_mask = valid_mask * (0 <= res.surface_precip.data)

            land_mask, ocean_mask = self.get_land_ocean_mask(match_ind)

            lats = reference.latitude.data
            valid_mask = valid_mask * (lats < 40) * (sp_ref < 0.5)

            for name, res in results.items():
                sp = res.surface_precip.data
                for metric in metrics_ocean[name]:
                    metric.update(sp[valid_mask * ocean_mask], sp_ref[valid_mask * ocean_mask])
                for metric in metrics_land[name]:
                    metric.update(sp[valid_mask * land_mask], sp_ref[valid_mask * land_mask])

                if "probability_of_precipitation" in res:
                    pop = res.probability_of_precipitation.data
                else:
                    pop = res.probability_of_precip.data

                for metric in detection_metrics_ocean[name]:
                    metric.update(pop[valid_mask * ocean_mask], self.precip_threshold <= sp_ref[valid_mask * ocean_mask])
                for metric in detection_metrics_land[name]:
                    metric.update(pop[valid_mask * land_mask], self.precip_threshold <= sp_ref[valid_mask * land_mask])

        results_land = {
            name: xr.merge([metric.compute() for metric in metrics + detection_metrics_land[name]])
            for name, metrics in metrics_land.items()
        }
        results_ocean = {
            name: xr.merge([metric.compute() for metric in metrics + detection_metrics_ocean[name]])
            for name, metrics in metrics_ocean.items()
        }

        return results_land, results_ocean


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
    Evaluator class for evaluating simulations

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
        input_granule, target_granules = matches[0]
        input_data = input_granule.open()
        lats = input_data.latitude_s1
        lons = input_data.latitude_s1

        results = simulate_tbs(
            "/gdata1/simon/gprof_v8/models/simulator_final/gprof_nn_sim.pt",
            sensors.GMI,
            input_granule,
            self.target_sensor,
            target_granules,
            device="cuda:1"
        )
        return results

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
                ["observations_gprof", "latitude", "longitude", "scan_time"]
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
        if resample:
            sim_data = resample_data(sim_data, swath, radius_of_influence=20e3)

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
        totals: bool = False
) -> plt.Figure:

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    lats = next(iter(results.values())).latitude.data
    weights = np.cos(np.deg2rad(lats))
    #weights[40 < np.abs(lats)] = 0.0

    totals = {}
    for name, res in results.items():
        spz = 24.0 * res.surface_precip_zonal

        if smooth is None:
            ax.plot(spz, lats, label=name)
        else:
            k = np.ones(smooth) / smooth
            spz_s = convolve(spz, k, mode="same")
            ax.plot(spz_s, lats, label=name)

        valid = np.isfinite(spz)
        weights = np.ones_like(weights)
        total = (spz * weights)[valid].sum() / weights[valid].sum()
        totals[name] = total

    ax.legend()
    ax.set_xlim(0, 24 * 0.5)
    ax.set_xlabel(rf"Precipitation Rate [mm h$^{-1}$]")
    ax.set_ylabel(rf"Latitude [$^\circ$ N]")
    ax.set_title(title)
    ax.set_ylim(-70, 70)

    if totals:
        txt = "Totals:\n" + "\n".join([f"{name}: {tot:.3} mm D$^{{-1}}$" for name, tot in totals.items()])
        ax.text(24 * 0.25, -20, txt, va="top")

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
