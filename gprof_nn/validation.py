"""
===================
gprof_nn.validation
===================

This module defines functions to collect validation data from MRMS
and Kwajalein co-locations and GPROF retrievals.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import logging
from pathlib import Path

from h5py import File
from matplotlib.colors import to_rgba
import numpy as np
from pyresample import geometry, kd_tree
import xarray as xr
from rich.progress import track
from scipy.ndimage import rotate
from scipy.signal import convolve
import pandas as pd
from pykdtree.kdtree import KDTree

from gprof_nn.coordinates import latlon_to_ecef
from gprof_nn.data.training_data import decompress_and_load
from gprof_nn.data.retrieval import RetrievalFile
from gprof_nn.data.sim import SimFile
from gprof_nn.data.combined import GPMCMBFile
from gprof_nn.definitions import LIMITS
from gprof_nn.utils import (
    calculate_interpolation_weights,
    interpolate,
    get_mask
)
from gprof_nn.data.validation import CONUS


VALIDATION_VARIABLES = [
    "surface_precip",
    "surface_precip_avg",
    "frozen_precip",
    "pop",
    "precip_1st_tercile",
    "precip_2nd_tercile",
    "surface_type",
    "airmass_type"
]


LOGGER = logging.getLogger(__name__)


PRECIP_TYPES = {
    "Stratiform, warm": [1.0, 2.0],
    "Stratiform, cool": [10.0],
    "Snow": [3.0, 4.0],
    "Convective": [6.0],
    "Hail": [7.0],
    "Tropical/stratiform mix": [91.0],
    "Tropical/convective rain mix": [96.0]
}


def smooth_reference_field(surface_precip, angles, steps=11):
    """
    Smooth the reference precip field using rotating smoothing kernels.

    Rotating smoothing kernels are employed to account for the conical
    scanning of the sensor.

    Args:
        surface_precip: The refence precip field interpolated to a
            5km x 5km grid following the satellite swath.
        angles: The corresponding orientation angles of the footprints.
        steps: The number of kernels to use to approximate the changing
            viewing angles across the scan.
    """
    angles[angles > 90] -= 180
    angles[angles < -90] += 180

    #
    # Calculate smoothing kernels for different angles.
    #

    fwhm_a = 37.5 / 5.0
    w_a = int(fwhm_a) + 1
    fwhm_x = 37.5 / 5.0
    w_x = int(fwhm_x) + 1
    d_a = 2 * np.arange(-w_a, w_a + 1e-6).reshape(-1, 1) / fwhm_a
    d_x = 2 * np.arange(-w_x, w_x + 1e-6).reshape(1, -1) / fwhm_x
    k = np.exp(np.log(0.5) * (d_a ** 2 + d_x ** 2))
    k = k / k.sum()
    ks = []
    kernel_angles = np.linspace(-70, 70, steps)
    for angle in kernel_angles:
        # Need to inverse angle because y-axis points in
        # opposite direction.
        ks.append(rotate(k, -angle, order=1))

    cts = (surface_precip >= 0).astype(np.float32)
    sp = np.nan_to_num(surface_precip.copy(), 0.0)

    fields = []
    for k in ks:
        k = k / k.sum()
        counts = convolve(cts, k, mode="same", method="direct")
        smoothed = convolve(sp, k, mode="same", method="direct")
        smoothed = smoothed / counts
        smoothed[counts < 0.5] = np.nan
        fields.append(smoothed)
    fields = np.stack(fields, axis=-1)

    weights = calculate_interpolation_weights(angles, kernel_angles)
    smoothed = interpolate(fields, weights)
    smoothed[np.isnan(sp)] = np.nan
    return smoothed


class GPROFNN1DResults:
    """
    Data interface class to collect results from GPROF-NN retrieval.
    """
    def __init__(self, path):
        """
        Args:
            path: Path pointing to the root of the directory tree containing
                the retrieval results.
        """
        self.path = Path(path)
        files = self.path.glob("**/*.bin")
        self.granules = {}
        for filename in files:
            try:
                granule = int(str(filename).split("_")[-1].split(".")[0])
                self.granules[granule] = filename
            except ValueError:
                pass
            except IndexError:
                pass

    def __len__(self):
        return len(self.granules)

    @property
    def smooth(self):
        return False

    @property
    def group_name(self):
        return "gprof_nn_1d"

    def open_granule(self, granule):
        data = RetrievalFile(self.granules[granule]).to_xarray_dataset()
        return data


class GPROFNN3DResults(GPROFNN1DResults):
    """
    Data interface class to collect results from GPROF-NN retrieval.
    """
    def __init__(self, path, name=None):
        """
        Args:
            path: Path pointing to the root of the directory tree containing
                the retrieval results.
        """
        super().__init__(path)
        self.name = name

    @property
    def group_name(self):
        if self.name is None:
            return "gprof_nn_3d"
        else:
            return self.name


class GPROFResults:
    """
    Data interface class to collect results from GPROF V7 result files.
    """
    def __init__(self, path):
        """
        Args:
            path: Path pointing to the root of the directory tree containing
                the retrieval results.
        """
        self.path = Path(path)
        files = self.path.glob("**/*.nc")
        self.granules = {}
        for filename in files:
            try:
                granule = int(str(filename.name).split("_")[-1].split(".")[0])
                self.granules[granule] = filename
            except ValueError:
                pass
            except IndexError:
                pass

    def __len__(self):
        return len(self.granules)

    @property
    def smooth(self):
        return False

    @property
    def group_name(self):
        return "gprof_v7"

    def open_granule(self, granule):
        dataset = xr.load_dataset(self.granules[granule])
        return dataset


class GPROFLegacyResults:
    """
    Data interface class to collect results from GPROF V6 result files.
    """
    def __init__(self, path):
        """
        Args:
            path: Path pointing to the root of the directory tree containing
                the retrieval results.
        """
        self.path = Path(path)
        files = self.path.glob("**/*.HDF5")
        self.granules = {}
        for filename in files:
            try:
                granule = int(str(filename.name).split(".")[-3])
                self.granules[granule] = filename
            except ValueError:
                pass
            except IndexError:
                pass

    def __len__(self):
        return len(self.granules)

    @property
    def smooth(self):
        return False

    @property
    def group_name(self):
        return "gprof_v5"

    def open_granule(self, granule):
        with File(str(self.granules[granule]), "r") as data:
            data = data["S1"]
            latitude = data["Latitude"][:]
            longitude = data["Longitude"][:]
            surface_precip = data["surfacePrecipitation"][:]
            pop = data["probabilityOfPrecip"][:]
            frozen_precip = data["frozenPrecipitation"][:]
            precip_1st_tercile = data["precip1stTertial"][:]
            precip_2nd_tercile = data["precip2ndTertial"][:]
            surface_type = data["surfaceTypeIndex"][:]

        dims = ("scans", "pixels")
        dataset = xr.Dataset({
            "latitude": (dims, latitude),
            "longitude": (dims, longitude),
            "surface_precip": (dims, surface_precip),
            "pop": (dims, pop),
            "frozen_precip": (dims, frozen_precip),
            "precip_1st_tercile": (dims, precip_1st_tercile),
            "precip_2nd_tercile": (dims, precip_2nd_tercile),
            "surface_type": (dims, surface_type)
        })
        return dataset


class GPMCMBResults(GPROFLegacyResults):
    """
    Data interface class to collect results from GPROF V6 result files.
    """
    def __init__(self, path):
        """
        Args:
            path: Path pointing to the root of the directory tree containing
                the retrieval results.
        """
        self.path = Path(path)
        files = self.path.glob("**/*.HDF5")
        self.granules = {}
        for filename in files:
            try:
                granule = int(str(filename.name).split(".")[-3])
                self.granules[granule] = filename
            except ValueError:
                pass
            except IndexError:
                pass

    @property
    def smooth(self):
        return False

    def __len__(self):
        return len(self.granules)

    @property
    def group_name(self):
        return "combined"

    def open_granule(self, granule):
        input_file = GPMCMBFile(self.granules[granule])
        data = input_file.to_xarray_dataset(
            profiles=False,
            smooth=False,
        )
        data_smooth = input_file.to_xarray_dataset(
            profiles=False,
            smooth=True,
        )
        data["surface_precip_avg"] = (
            ("scans", "pixels"),
            data_smooth["surface_precip"].data
        )
        return data


class SimulatorFiles():
    """
    Interface class to read data from simulator files.
    """
    def __init__(self, path):
        """
        Args:
            path: Base path containing the simulator files.
        """
        self.path = Path(path)
        files = self.path.glob("**/*.sim")
        self.granules = {}
        for filename in files:
            try:
                granule = int(str(filename.name).split(".")[-2])
                self.granules[granule] = filename
            except ValueError:
                pass
            except IndexError:
                pass

    @property
    def smooth(self):
        return False

    def __len__(self):
        return len(self.granules)

    @property
    def group_name(self):
        return "simulator"

    def open_granule(self, granule):
        sim_data = SimFile(self.granules[granule]).to_xarray_dataset()
        return sim_data



class ResultCollector:
    """
    This class collects validation results from MRMS files and combines
    them with retrieval results from a selection of datasets.
    """
    def __init__(
            self,
            reference_path,
            datasets
    ):
        self.reference_path = Path(reference_path)
        self.datasets = datasets

        self.reference_files = list(self.reference_path.glob("**/*.nc"))


    def process_reference_file(self, filename, output_file):
        """
        Extract data from reference file and extract corresponding data from
        retrieval datasets.

        Results are written in NetCDF4 format to the given output file with
        each dataset stored in its own group.

        Args:
            filename: Path to the reference file containing the extract overpass
                data.
            output_file: Path to the output file to which to write the combined
                results.
        """
        granule = int(str(filename).split("_")[-1].split(".")[0])
        reference_data = xr.load_dataset(filename)

        precip_names = ["surface_precip",
                        "surface_precip_rr",
                        "surface_precip_rp",
                        "surface_precip_rc"]
        for variable in precip_names:
            if variable in reference_data.variables:
                surface_precip = reference_data[variable].data
                angles = reference_data.angles.data
                surface_precip_smoothed = smooth_reference_field(
                    surface_precip,
                    angles
                )
                reference_data[variable + "_avg"] = (
                    ("along_track", "across_track"),
                    surface_precip_smoothed
                )

        if "mask" in reference_data.variables:
            fields = []

            for values in PRECIP_TYPES.values():
                mask = reference_data["mask"]
                mask = np.stack([np.isclose(mask, v) for v in values]).any(axis=0)
                fields.append(smooth_reference_field(
                    mask,
                    angles
                ))
            fields = np.stack(fields)
            classes_smoothed = np.argmax(fields, axis=0)

            no_rain = (fields.max(axis=0) < 0.5)
            classes_smoothed[no_rain] = 0
            mask = reference_data["mask"].data
            classes_smoothed[mask < 0] = -1

            reference_data["mask_avg"] = (
                ("along_track", "across_track"),
                classes_smoothed
            )

        if "raining_fraction" in reference_data.variables:
            rf = reference_data["raining_fraction"].data
            angles = reference_data.angles.data
            rf_smoothed = smooth_reference_field(
                rf,
                angles
            )
            reference_data["raining_fraction_avg"] = (
                ("along_track", "across_track"),
                rf_smoothed
            )

        if "surface_precip_rp" in reference_data.variables:
            reference_data = reference_data.rename({
                "surface_precip_rc": "surface_precip",
                "surface_precip_rc_avg": "surface_precip_avg"
                })

        lats = reference_data.latitude.data
        lons = reference_data.longitude.data
        result_grid = geometry.SwathDefinition(lats=lats, lons=lons)

        for dataset in self.datasets:
            try:
                data = dataset.open_granule(granule)
                data_r = xr.Dataset({
                    "along_track": (("along_track",), reference_data.along_track.data),
                    "across_track": (("across_track",), reference_data.across_track.data),
                })
                def weighting_function(distance):
                    return np.exp(np.log(0.5) * (2.0  * (distance / 5e3)) ** 2)

                if dataset.group_name == "simulator":
                    lats = reference_data["latitude"].data.reshape(-1, 1)
                    lons = reference_data["longitude"].data.reshape(-1, 1)
                    coords = latlon_to_ecef(lons, lats)
                    coords = np.concatenate(coords, axis=1)

                    lats_sim = data["latitude"].data.reshape(-1, 1)
                    lons_sim = data["longitude"].data.reshape(-1, 1)
                    coords_sim = latlon_to_ecef(lons_sim, lats_sim)
                    coords_sim = np.concatenate(coords_sim, 1)

                    # Determine indices of matching L1C observations.
                    kdtree = KDTree(coords)
                    dists, indices = kdtree.query(coords_sim)

                    # Extract matching data
                    for variable in VALIDATION_VARIABLES:
                        if variable in data:
                            shape = reference_data["latitude"].data.shape
                            matched = np.zeros(shape, dtype=np.float32).ravel()
                            matched[:] = np.nan
                            matched[indices, ...] = data[variable]
                            matched[indices[dists > 5e3], ...] = np.nan
                            matched = matched.reshape(shape)
                            data_r[variable] = (("along_track", "across_track"), matched)

                else:
                    data_grid = geometry.SwathDefinition(
                        lats=data.latitude.data,
                        lons=data.longitude.data
                    )
                    resampling_info = kd_tree.get_neighbour_info(
                        data_grid,
                        result_grid,
                        7.5e3,
                        neighbours=8
                    )
                    valid_inputs, valid_outputs, indices, distances = resampling_info

                    resampling_info = kd_tree.get_neighbour_info(
                        data_grid,
                        result_grid,
                        7.5e3,
                        neighbours=1
                    )
                    valid_inputs_nn, valid_outputs_nn, indices_nn, distances_nn = resampling_info


                    for variable in VALIDATION_VARIABLES:
                        if variable in data.variables:
                            missing = np.nan
                            if data[variable].dtype not in [np.float32, np.float64]:
                                missing = -999
                                resampled = kd_tree.get_sample_from_neighbour_info(
                                    'nn', result_grid.shape, data[variable].data,
                                    valid_inputs_nn, valid_outputs_nn, indices_nn,
                                    fill_value=missing
                                )
                            else:
                                missing = np.nan
                                resampled = kd_tree.get_sample_from_neighbour_info(
                                    'custom', result_grid.shape, data[variable].data,
                                    valid_inputs, valid_outputs, indices,
                                    fill_value=missing, weight_funcs=weighting_function,
                                    distance_array=distances
                                )
                            data_r[variable] = (("along_track", "across_track"), resampled)

                    if dataset.smooth and "surface_precip" in data_r:
                        surface_precip = data_r["surface_precip"].data
                        angles = reference_data.angles.data
                        surface_precip_smoothed = smooth_reference_field(
                            surface_precip,
                            angles
                        )
                        reference_data["surface_precip_avg"] = (
                            ("along_track", "across_track"),
                            surface_precip_smoothed
                        )

                if "scan_time" in data:
                    time = data["scan_time"].mean()
                    data["time"] = (("time",), [time.data])
                if output_file.exists():
                    data_r.to_netcdf(output_file, group=dataset.group_name, mode="a")
                else:
                    data_r.to_netcdf(output_file, group=dataset.group_name)
            except KeyError as error:
                LOGGER.error(
                    "The following error was encountered while processing granule "
                    "'%s' of dataset '%s':\n %s",
                    granule,
                    dataset.group_name,
                    error)

        if output_file.exists():
            reference_data.to_netcdf(output_file, group="reference", mode="a")



    def run(self, output_path, start=None, end=None, n_processes=4):
        """
        Run collector.

        Args:
            output_path: Directory to which to write the result files.
            start: If given, only files after this data will be considered.
            end: If given, only files before this date will be considered.
            n_processes: The number of processes to use to collect the
                validation data.
        """
        output_path = Path(output_path)
        pool = ProcessPoolExecutor(max_workers=n_processes)

        tasks = []
        files = []

        for reference_file in self.reference_files:

            # Process only files in given range
            parts = reference_file.name.split("_")
            if len(parts) > 4:
                yearmonthday, hourminute = reference_file.name.split("_")[2:4]
                year = yearmonthday[:4]
                month = yearmonthday[4:6]
                day = yearmonthday[6:]
                hour = hourminute[:2]
                minute = hourminute[2:]
            else:
                yearmonthday = reference_file.name.split("_")[2]
                year = yearmonthday[:4]
                month = yearmonthday[4:6]
                day = yearmonthday[6:]
                hour = "00"
                minute = "00"
            date = np.datetime64(f"{year}-{month}-{day}T{hour}:{minute}:00")

            if start is not None and date < start:
                continue
            if end is not None and date >= end:
                continue

            output_filename = output_path / reference_file.name
            tasks.append(pool.submit(
                self.process_reference_file, reference_file, output_filename
            ))
            files.append(reference_file)

        for filename, task in track(list(zip(files, tasks))):
            try:
                task.result()
            except Exception as e:
                LOGGER.error(
                    "The following error was encountered when processing "
                    "file %s: \n %s", filename, e
                )

###############################################################################
# Utilities
###############################################################################

NAMES = {
    "gprof_nn_1d": "GPROF-NN 1D",
    "gprof_nn_3d": "GPROF-NN 3D",
    "gprof_v5": "GPROF V5",
    "gprof_v7": "GPROF V7",
    "simulator": "Simulator",
    "combined": "GPM-CMB"
}

REGIONS = {
    "C": [-102, 35, -91.6, 45.4],
    "NW": [-124, 38.6, -113.6, 49.0],
    "SW": [-113, 29, -102.6, 39.4],
    "NE": [-81.4, 38.6, -71, 49.0],
    "SE": [-91, 25, -80.6, 35.4],
    "KWAJ": [163.732, 4.71, 171.731, 12.71]
}

def open_reference_file(reference_path, granule):
    """
    Open the reference data file corresponding to a given granule
    number.

    Args:
        reference_path: Root of the directory tree containing the
            reference data.
        granule: The granule number.

    Return:
        An ``xarray.Dataset`` containing the loaded reference data
    """
    files = Path(reference_path).glob(f"**/*{granule}.nc")
    return xr.load_dataset(next(iter(files)))


def plot_granule(reference_path, granule, datasets, n_cols=3, height=4, width=4):
    """
    Plot overpass over reference data and corresponding retrieval results
    for a given granule.

    Args:
        reference_path: Path to the root of the directory tree containing
            the reference data.
        granule: The granule number as an iteger.
        dataset: List of dataset object providing access to the retrieval
            results.

    Return:
        The matplotlib Figure object containing the plotted data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LogNorm
    from matplotlib import cm
    import cartopy.crs as ccrs

    n = n_cols
    m = ((1 + len(datasets)) // n)
    if ((1 + len(datasets)) % n):
        m = m + 1
    f = plt.figure(figsize=(width * n + 1, height * m))

    gs = GridSpec(m, n + 1, width_ratios=[1.0] * n + [0.05])
    axs = np.array(
        [[f.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
          for j in range(n)]
         for i in range(m)]
    )
    axs = axs.ravel()

    precip_norm = LogNorm(1e-1, 1e2)
    cmap = copy(cm.get_cmap("plasma"))
    cmap.set_under("#FFFFFF10")

    ref_data = open_reference_file(reference_path, granule)
    ax = axs[0]

    lats = ref_data.latitude.data
    lons = ref_data.longitude.data
    xlims = [lons.min(), lons.max()]
    ylims = [lats.min(), lats.max()]

    surface_precip = ref_data["surface_precip"].data
    angles = ref_data.angles.data
    surface_precip_smoothed = smooth_reference_field(
        surface_precip,
        angles
    )
    sp = np.maximum(surface_precip_smoothed, 1e-4)
    ax.stock_img()
    m = ax.pcolormesh(lons, lats, sp, cmap=cmap, norm=precip_norm)
    ax.plot(lons[:, 0], lats[:, 0], ls="--", c="k")
    ax.plot(lons[:, -1], lats[:, -1], ls="--", c="k")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.coastlines()
    ax.set_title("(a) Reference", loc="left")

    for i, dataset in enumerate(datasets):
        ax = axs[i + 1]

        data = dataset.open_granule(granule)
        sp = np.maximum(data.surface_precip.data, 1e-4)
        lats = data.latitude.data
        lons = data.longitude.data

        ax.stock_img()
        ax.pcolormesh(lons, lats, sp, cmap=cmap, norm=precip_norm)
        ax.plot(lons[:, 0], lats[:, 0], ls="--", c="k")
        ax.plot(lons[:, -1], lats[:, -1], ls="--", c="k")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.coastlines()

        ax.set_title(f"({chr(ord('a') + i + 1)}) {dataset.group_name}", loc="left")

    for ax in axs[i + 2:]:
        ax.set_axis_off()

    ax = f.add_subplot(gs[:, -1])
    plt.colorbar(m, label="Surface precip. [mm/h]", cax=ax)
    return f

def calculate_scatter_plot(results, group, rqi_threshold=0.8):
    """
    Calculate normalized scatter plot for a given retrieval.

    Uses only observations over surface types 1 - 8 and 12 - 16 and
    that are not marked as snow by the radar.

    Args:
        results: Dict of xarray.Dataset containing the matched retrieved
            and reference precipitation for the different algorithms.
        group: The key to use to obtain the validation results form 'results'.
        rqi_threshold: Additional RQI threshold to select subsets of validation
            samples.

    Return:
        A tuple ``(bins, y)`` containing the precipitation bins ``bins`` and the
        corresponding conditional PDFs ``y``
    """
    bins = np.logspace(-2, 2, 101)

    sp_ref = results[group].surface_precip_avg.data
    sp = results[group].surface_precip.data
    valid = (sp_ref >= 0) * (sp >= 0)

    if "surface_type" in results[group]:
        surface_type = results[group].surface_type.data
        valid *= ((surface_type < 8) + ((surface_type > 11) * (surface_type < 17)))

    if "mask" in results[group].variables:
        mask = results[group].mask
        snow = np.isclose(mask, 3.0) + np.isclose(mask, 4.0)
        valid *= ~snow

    y, _, _ = np.histogram2d(
        sp_ref[valid],
        sp[valid],
        bins=bins,
        density=True
    )
    dx = np.diff(bins)
    x = 0.5 * (bins[1:] + bins[:-1])
    dx = dx.reshape(1, -1)
    y /= (y * dx).sum(axis=1, keepdims=True)

    return bins, y


def calculate_conditional_mean(results, group, rqi_threshold=0.8):
    """
    Calculate normalized scatter plot for a given retrieval.

    Uses only observations over surface types 1 - 8 and 12 - 16 and
    that are not marked as snow by the radar.

    Args:
        results: Dict of xarray.Dataset containing the matched retrieved
            and reference precipitation for the different algorithms.
        group: The key to use to obtain the validation results form 'results'.
        rqi_threshold: Additional RQI threshold to select subsets of validation
            samples.

    Return:
        A tuple ``(x, means)`` containing values of reference precipitation and
        the corresponding conditional mean.
    """
    bins = np.logspace(-2, 2, 101)

    sp_ref = results[group].surface_precip_avg.data
    sp = results[group].surface_precip.data
    valid = (sp_ref >= 0) * (sp >= 0)

    if "surface_type" in results[group]:
        surface_type = results[group].surface_type.data
        valid = ((surface_type < 8) + ((surface_type > 11) * (surface_type < 17)))

    mask = results[group].mask
    snow = np.isclose(mask, 3.0) + np.isclose(mask, 4.0)
    valid *= ~snow

    sp_ref = sp_ref[valid]
    sp = sp[valid]

    sums, _, = np.histogram(sp_ref, weights=sp, bins=bins)
    cts, _, = np.histogram(sp_ref, bins=bins)
    means = sums / cts
    x = 0.5 * (bins[1:] + bins[:-1])
    return x, means


def calculate_error_metrics(results, groups, rqi_threshold=0.8, region=None):
    """
    Calculate error metrics for validation data.

    Uses only observations over surface types 1 - 8 and 12 - 16 and
    that are not marked as snow by the radar.

    Args:
       results: Dictionary holding xr.Datasets with the results for the different retrievals.
       groups: Names of the retrievals for which to calculate error metrics.
       rqi_threshold: Optional additional rqi_threshold to filter co-locations.
    """
    bias = {}
    correlation = {}
    mse = {}
    mae = {}
    mape = {}
    far = {}
    pod = {}

    for group in groups:
        if group == "cmb":
            sp_ref = results[group].surface_precip_ref.data
        else:
            if "surface_precip_ref" not in results[group].variables:
                sp_ref = results[group].surface_precip_mrms.data
            else:
                sp_ref = results[group].surface_precip_ref.data
            sp_ref = results[group].surface_precip_avg.data
        sp = results[group].surface_precip.data

        valid = (sp_ref >= 0) * (sp >= 0)

        if "surface_type" in results[group]:
            surface_type = results[group].surface_type.data
            valid *= ((surface_type < 8) + ((surface_type > 11) * (surface_type < 17)))

        lats = results[group].latitude.data
        lons = results[group].longitude.data
        if region is not None:
            lon_0, lat_0, lon_1, lat_1 = REGIONS[region]
            valid *= ((lons >= lon_0) * (lons < lon_1) *
                    (lats >= lat_0) * (lats < lat_1))


        if "mask" in results[group]:
            mask = results[group].mask
            snow = np.isclose(mask, 3.0) + np.isclose(mask, 4.0)
            valid *= ~snow

        if "rqi" in results[group].variables:
            rqi = results[group].rqi.data
            valid *=  (rqi > rqi_threshold)

        sp = sp[valid]
        sp_ref = sp_ref[valid]

        bias[group] = 100 * np.mean(sp - sp_ref) / np.mean(sp_ref)
        mse[group] = np.mean((sp - sp_ref) ** 2)
        mae[group] = np.mean(np.abs(sp - sp_ref))

        ref = 0.5 * np.abs(sp) + np.abs(sp_ref)
        rel_err = np.abs((sp - sp_ref) / ref)

        mape[group] = np.mean(rel_err[ref > 1e-1]) * 100
        corr = np.corrcoef(x=sp_ref, y=sp)
        correlation[group] = corr[0, 1]

    data = {
        "Bias": list(bias.values()),
        "MAE": list(mae.values()),
        "MSE": list(mse.values()),
        "Correlation": list(correlation.values()),
        "SMAPE": list(mape.values()),
    }
    names = [NAMES[g] for g in groups]
    return pd.DataFrame(data, index=names)


def calculate_monthly_statistics(results, group, rqi_threshold=0.8, region=None):
    """
    Calculates monthly relative biases and correlations.

    Uses only observations over surface types 1 - 8 and 12 - 16 and
    that are not marked as snow by the radar.

    Args:
       results: Dictionary holding xr.Datasets with the results for the different retrievals.
       groups: Names of the retrievals for which to calculate error metrics.
       rqi_threshold: Optional additional rqi_threshold to filter co-locations.
    """
    sp_ref = results[group].surface_precip_avg.data
    sp = results[group].surface_precip.data
    valid = (sp_ref >= 0) * (sp >= 0)

    surface_type = results[group].surface_type.data
    valid *= ((surface_type < 8) + ((surface_type > 11) * (surface_type < 17)))

    mask = results[group].mask
    snow = np.isclose(mask, 3.0) + np.isclose(mask, 4.0)
    valid *= ~snow

    lats = results[group].latitude.data
    lons = results[group].longitude.data

    if region is not None:
        lon_0, lat_0, lon_1, lat_1 = REGIONS[region]
        valid *= ((lons >= lon_0) * (lons < lon_1) *
                  (lats >= lat_0) * (lats < lat_1))


    sp_ref = sp_ref[valid]
    sp = sp[valid]

    months = results[group].time.dt.month[valid]

    bins = np.linspace(0, 12, 13) - 0.5
    sums, _ = np.histogram(months, weights=sp_ref - sp, bins=bins)
    cts, _ = np.histogram(months, bins=bins)
    means = sums / cts

    corrs = []
    for i in range(1, 13):
        indices = months == i
        sp_m = sp[indices]
        sp_ref_m = sp_ref[indices]
        corrs.append(np.corrcoef(sp_m, sp_ref_m)[0, 1])

    months = np.arange(1, 13)
    return months, means, np.array(corrs)
    return pd.DataFrame(data, index=names)


def calculate_seasonal_cycles(results, group, rqi_threshold=0.8, region=None):
    """
    Calculates daily cycles.

    Uses only observations over surface types 1 - 8 and 12 - 16 and
    that are not marked as snow by the radar.

    Args:
       results: Dictionary holding xr.Datasets with the results for the different retrievals.
       groups: Names of the retrievals for which to calculate error metrics.
       rqi_threshold: Optional additional rqi_threshold to filter co-locations.
    """
    if group == "reference":
        sp_ref = results["gprof_nn_1d"].surface_precip_avg.data
        sp = results["gprof_nn_1d"].surface_precip_avg.data
        group = "gprof_nn_1d"
    else:
        sp_ref = results[group].surface_precip_avg.data
        sp = results[group].surface_precip.data
    valid = (sp_ref >= 0) * (sp >= 0)

    surface_type = results[group].surface_type.data
    valid *= ((surface_type < 8) + ((surface_type > 11) * (surface_type < 17)))

    mask = results[group].mask
    snow = np.isclose(mask, 3.0) + np.isclose(mask, 4.0)
    valid *= ~snow

    lats = results[group].latitude.data
    lons = results[group].longitude.data

    if region is not None:
        lon_0, lat_0, lon_1, lat_1 = REGIONS[region]
        valid *= ((lons >= lon_0) * (lons < lon_1) *
                  (lats >= lat_0) * (lats < lat_1))


    sp_ref = sp_ref[valid]
    sp = sp[valid]

    time = results[group].time[valid]
    months = time.dt.month.data

    bins = (np.linspace(0, 12, 13) + 0.5)
    sums, _ = np.histogram(months, weights=sp, bins=bins)
    cts, _ = np.histogram(months, bins=bins)
    means = sums / cts

    means = np.concatenate([means[-1:], means, means[:1]])
    means = 0.5  * (means[1:] + means[:-1])

    months = np.arange(1, 14)
    return months, means

def calculate_daily_cycles(results, group, rqi_threshold=0.8, region=None):
    """
    Calculates daily cycles.

    Uses only observations over surface types 1 - 8 and 12 - 16 and
    that are not marked as snow by the radar.

    Args:
       results: Dictionary holding xr.Datasets with the results for the different retrievals.
       groups: Names of the retrievals for which to calculate error metrics.
       rqi_threshold: Optional additional rqi_threshold to filter co-locations.
    """
    if group == "reference":
        sp_ref = results["gprof_nn_1d"].surface_precip_avg.data
        sp = results["gprof_nn_1d"].surface_precip_avg.data
        group = "gprof_nn_1d"
    else:
        sp_ref = results[group].surface_precip_avg.data
        sp = results[group].surface_precip.data
    valid = (sp_ref >= 0) * (sp >= 0)

    surface_type = results[group].surface_type.data
    valid *= ((surface_type < 8) + ((surface_type > 11) * (surface_type < 17)))

    mask = results[group].mask
    snow = np.isclose(mask, 3.0) + np.isclose(mask, 4.0)
    valid *= ~snow

    lats = results[group].latitude.data
    lons = results[group].longitude.data

    if region is not None:
        lon_0, lat_0, lon_1, lat_1 = REGIONS[region]
        valid *= ((lons >= lon_0) * (lons < lon_1) *
                  (lats >= lat_0) * (lats < lat_1))


    sp_ref = sp_ref[valid]
    sp = sp[valid]

    time = results[group].time[valid]
    time += (lons[valid] / 360 * 24 * 60 * 60).astype("timedelta64[s]")
    minutes = time.dt.hour.data + 60 * time.dt.minute.data

    bins = (np.linspace(0, 24, 25) - 0.5) * 60
    sums, _ = np.histogram(minutes, weights=sp, bins=bins)
    cts, _ = np.histogram(minutes, bins=bins)
    means = sums / cts

    means = np.concatenate([means[-1:], means, means[:1]])
    means = 0.5  * (means[1:] + means[:-1])
    hours = np.arange(0, 25, 1)

    return hours, means


def get_colors():
    """
    Return dictionary of plot colors for different retrievals.
    """
    c0 = to_rgba("C0")
    c0_d = np.array(c0).copy()
    c0_d[:3] *= 0.6
    c1 = to_rgba("C1")
    c1_d = np.array(c1).copy()
    c1_d[:3] *= 0.6
    c2 = to_rgba("C2")
    bar_palette = [c0, c0_d, c1, c1_d, c2]

    return {
        "gprof_v5": c0,
        "gprof_v7": c0_d,
        "gprof_nn_1d": c1,
        "gprof_nn_3d": c1_d,
        "combined": c2
    }
