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
from typing import Dict, Tuple, Optional

from h5py import File
from matplotlib.colors import to_rgba
import numpy as np
from pyresample import geometry, kd_tree
import xarray as xr
from rich.progress import track
from scipy.ndimage import rotate
from scipy.signal import convolve
from scipy.stats import binned_statistic_2d, binned_statistic, linregress
import pandas as pd
from pykdtree.kdtree import KDTree

from gprof_nn import sensors
from gprof_nn.coordinates import latlon_to_ecef
from gprof_nn.data.training_data import decompress_and_load
from gprof_nn.data.retrieval import RetrievalFile
from gprof_nn.data.sim import SimFile
from gprof_nn.data.combined import GPMCMBFile
from gprof_nn.definitions import LIMITS
from gprof_nn.data.sim import apply_orographic_enhancement
from gprof_nn.utils import (
    calculate_interpolation_weights,
    interpolate,
    get_mask,
    calculate_smoothing_kernel
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
    "Tropical/convective rain mix": [96.0],
    "Stratiform": [1.0, 2.0, 10.0, 91.0],
    "Convective": [6.0, 7.0, 96.0],
}

FOOTPRINTS = {
        "GMI": (12.5, 5),
        "AMSR2": (10.0, 10.0),
        "SSMIS": (38, 38),
        "TMIPO": (18, 30),
}


def smooth_reference_field(
        sensor,
        surface_precip,
        angles,
        steps=11,
        resolution=5
):
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

    kernels = []

    if isinstance(sensor, sensors.ConicalScanner):
        along_track, across_track = FOOTPRINTS[sensor.name]
        kernel_angles = np.linspace(-70, 70, steps)
        kernel = calculate_smoothing_kernel(
            along_track,
            across_track,
            res_a=resolution,
            res_x=resolution
            )
        for angle in kernel_angles:
            # Need to inverse angle because y-axis points in
            # opposite direction.
            kernels.append(rotate(kernel, -angle, order=1))
    elif isinstance(sensor, sensors.CrossTrackScanner):
        angles = np.clip(angles, -np.max(sensor.angles), np.max(sensor.angles))
        kernel_angles = np.linspace(1e-3, angles.max(), steps)
        angles = np.abs(angles)
        along_track = sensor.viewing_geometry.get_resolution_a(kernel_angles)
        across_track = sensor.viewing_geometry.get_resolution_x(kernel_angles)
        for fwhm_a, fwhm_x in zip(along_track, across_track):
            kernels.append(calculate_smoothing_kernel(
                fwhm_a / 1e3, fwhm_x / 1e3, res_a=resolution, res_x=resolution
            ))
    else:
        raise ValueError(
            "Sensor object must be either a 'ConicalScanner' or "
            "a 'CrossTrackScanner'."
        )

    cts = (surface_precip >= 0).astype(np.float32)
    sp = np.nan_to_num(surface_precip.copy(), 0.0)

    fields = []
    for kernel in kernels:
        kernel = kernel / kernel.sum()
        counts = convolve(cts, kernel, mode="same", method="direct")
        smoothed = convolve(sp, kernel, mode="same", method="direct")
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
        #data = RetrievalFile(self.granules[granule]).to_xarray_dataset()
        data = xr.load_dataset(self.granules[granule])
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


class GPROFNNHRResults:
    """
    Data interface class to collect results from GPROF-NN HR retrieval.
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
        return True

    @property
    def group_name(self):
        return "gprof_nn_hr"

    def open_granule(self, granule):
        data = xr.load_dataset(self.granules[granule])
        return data


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
            sensor,
            reference_path,
            datasets
    ):
        self.sensor = sensor
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
                    self.sensor,
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
                    self.sensor,
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
                self.sensor,
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
                        20e3,
                        neighbours=8
                    )
                    valid_inputs, valid_outputs, indices, distances = resampling_info

                    resampling_info = kd_tree.get_neighbour_info(
                        data_grid,
                        result_grid,
                        20e3,
                        neighbours=1
                    )
                    valid_inputs_nn, valid_outputs_nn, indices_nn, distances_nn = resampling_info


                    for variable in VALIDATION_VARIABLES:
                        if variable in data.variables:
                            missing = np.nan
                            if data[variable].dtype not in [np.float32, np.float64]:
                                missing = -999
                                v_data = data[variable].data
                                v_data[:, 0] = missing
                                v_data[:, -1] = missing
                                resampled = kd_tree.get_sample_from_neighbour_info(
                                    'nn', result_grid.shape, v_data,
                                    valid_inputs_nn, valid_outputs_nn, indices_nn,
                                    fill_value=missing
                                )
                            else:
                                missing = np.nan
                                v_data = data[variable].data
                                v_data[:, 0] = missing
                                v_data[:, -1] = missing
                                resampled = kd_tree.get_sample_from_neighbour_info(
                                    'nn', result_grid.shape, v_data,
                                    valid_inputs_nn, valid_outputs_nn, indices_nn,
                                    fill_value=missing
                                )
                            data_r[variable] = (("along_track", "across_track"), resampled)

                    if dataset.smooth and "surface_precip" in data_r:
                        surface_precip = data_r["surface_precip"].data
                        angles = reference_data.angles.data
                        surface_precip_smoothed = smooth_reference_field(
                            self.sensor,
                            surface_precip,
                            angles
                        )
                        data_r["surface_precip_avg"] = (
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
                LOGGER.exception(
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
                LOGGER.exception(
                    "The following error was encountered when processing "
                    "file %s: \n %s", filename, e
                )


def calculate_precip_contribution(
        results,
        precip_type=None,
        region=None,
        absolute=False,
        no_orographic=False,
        no_frozen=True,
        no_ocean=True,
        no_snow_sfc=True
):
    """
    Calculate contribution of precipitation type to total precipitation.
    Args:
        results: xarray.Dataset containing collocated validation data. The
            'surface_precip_ref' field will be used to calculate the
            precipitation.
        precip_type: Name of the precip type.
        absolute: If true the absolute contribution to the mean precipitation
            is returned.
        no_ocean: If 'True', precipitation over the ocean will be omitted
            from the analysis.
        no_orographic: If 'True', precipitation over mountains will be omitted
            from the analysis,
        no_froze: If 'True', precipitation classified as snow will be omitted
            from the analysis.
        no_snow_sfc: If 'True', precipitation over snow surface will be
            omitted from the calculation.

    Return:
        The fraction r (0 <= r <= 1) that the precipitation type contributed
        to overall precipitation.
    """
    surface_precip = results.surface_precip_ref.data
    mask = results.mask.data

    valid = surface_precip >= 0.0

    if "surface_type" in results:
        sfc = results.surface_type.data

        if no_ocean:
            valid *= (sfc > 1)

        if no_orographic:
            valid *= (sfc < 17)

        if no_snow_sfc:
            valid *= ((sfc < 8) + (sfc > 11))

    if "mask" in results:
        mask = results.mask
        if no_frozen:
            frozen = (
                np.isclose(mask, 3.0) +
                np.isclose(mask, 4.0) +
                np.isclose(mask, 7.0)
            )
            valid *= ~frozen

    if region is not None:
        lons = results.longitude
        lats = results.latitude
        lon_0, lat_0, lon_1, lat_1 = REGIONS[region]
        region_mask = ((lons >= lon_0) * (lons < lon_1) *
                       (lats >= lat_0) * (lats < lat_1))
        valid *= region_mask

    mean_precip_rate = surface_precip[valid].mean()

    rain_mask = (mask > 0)
    surface_precip = surface_precip[rain_mask * valid]
    mask = mask[rain_mask * valid]

    if precip_type is not None:
        type_mask = np.zeros_like(surface_precip, dtype=bool)
        for val in PRECIP_TYPES[precip_type]:
            type_mask += np.isclose(mask, val)
    else:
        type_mask = np.ones_like(surface_precip, dtype=bool)

    sp_t = surface_precip[type_mask].sum()
    sp_t /= surface_precip.sum()

    if absolute:
        sp_t *= mean_precip_rate

    return sp_t


###############################################################################
# Utilities
###############################################################################

NAMES = {
    "gprof_nn_1d": "GPROF-NN 1D",
    "gprof_nn_3d": "GPROF-NN 3D",
    "gprof_nn_hr": "GPROF-NN HR",
    "gprof_v5": "GPROF V5",
    "gprof_v7": "GPROF V7",
    "simulator": "Simulator",
    "combined": "GPM CMB",
    "combined_avg": "GPM CMB (Smoothed)"
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


def plot_granule(sensor, reference_path, granule, datasets, n_cols=3, height=4, width=4):
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
        sensor,
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
        valid = sp[sp >= 0]
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


def calculate_scatter_plot(
        results: Dict[str, xr.Dataset],
        group: str,
        rqi_threshold: int = 0.8,
        no_ocean: bool = True,
        no_orographic: bool = True,
        no_frozen: bool = True,
        no_snow_sfc: bool = True,
        fpavg: bool = False
):
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
        no_ocean: If 'True' precipitation over ocean surfaces is excluded from
            the analysis.
        no_orographic: If 'True' precipitation in mountain regions is excluded from
            the analysis.
        no_frozen: If 'True', precipitation classified as snow and hail will be
            excluded from the analysis.
        no_snow_sfc: If 'True', precipitation over snow surfaces will be excluded
            the analysis.
        fpavg: Whether to use footprint averaged reference data or not.

    Return:
        A tuple ``(bins, y)`` containing the precipitation bins ``bins`` and the
        corresponding conditional PDFs ``y``
    """
    bins = np.logspace(-2, 2, 101)

    if fpavg:
        sp_ref = results[group].surface_precip_ref_avg.data
    else:
        sp_ref = results[group].surface_precip_ref.data
    sp = results[group].surface_precip.data
    valid = (sp_ref >= 0) * (sp >= 0)

    if "surface_type" in results[group]:

        surface_type = results[group].surface_type.data
        sea_ice = (surface_type == 2) + (surface_type == 16)
        valid *= ~sea_ice

        if no_orographic:
            valid *= (surface_type < 17)
        if no_snow_sfc:
            valid *= ((surface_type < 8) + ((surface_type > 11)))
        if no_ocean:
            valid *= (surface_type > 1)

    if "mask" in results[group].variables and no_frozen:
        mask = results[group].mask
        frozen = (
            np.isclose(mask, 3.0) +
            np.isclose(mask, 4.0) +
            np.isclose(mask, 7.0)
        )
        valid *= ~frozen

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


def calculate_conditional_mean(
        results: Dict[str, xr.Dataset],
        group,
        rqi_threshold=0.8,
        no_ocean=True,
        no_orographic=True,
        no_frozen=True,
        no_snow_sfc=True,
        fpavg=False
):
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
        no_ocean: If 'True', pixels over ocean surfaces are excluded from the analysis
        no_orographic: If 'True', pixels over mountains are excluded from the analysis.
        no_frozen: If 'True' precipitation classified as frozen by MRMS is excluded
            from the analysis.
        no_snow_sfc: If 'True' precipitation over snow surfaces is excluded from
            the analysis.

    Return:
        A tuple ``(x, means)`` containing values of reference precipitation and
        the corresponding conditional mean.
    """
    bins = np.logspace(-2, 2, 101)

    if fpavg:
        sp_ref = results[group].surface_precip_ref_avg.data
    else:
        sp_ref = results[group].surface_precip_ref.data
    sp = results[group].surface_precip.data
    valid = (sp_ref >= 0) * (sp >= 0)

    if "surface_type" in results[group]:
        surface_type = results[group].surface_type.data

        sea_ice = (surface_type == 2) + (surface_type == 16)
        valid *= ~sea_ice

        if no_ocean:
            valid *= surface_type > 1

        if no_orographic:
            valid *= surface_type < 17

        if no_snow_sfc:
            valid *= ((surface_type < 8) + (surface_type > 11))

    if "mask" in results[group] and no_frozen:
        mask = results[group].mask
        frozen = (
            np.isclose(mask, 3.0) +
            np.isclose(mask, 4.0) +
            np.isclose(mask, 7.0)
        )
        valid *= ~frozen

    sp_ref = sp_ref[valid]
    sp = sp[valid]

    sums, _, = np.histogram(sp_ref, weights=sp, bins=bins)
    cts, _, = np.histogram(sp_ref, bins=bins)
    means = sums / cts
    x = 0.5 * (bins[1:] + bins[:-1])
    return x, means


def calculate_error_metrics(
        results,
        groups,
        rqi_threshold=0.8,
        region=None,
        ranges=None,
        fpa=False,
        no_orographic=False,
        no_ocean=True,
        no_frozen=True,
        no_snow_sfc=True
):
    """
    Calculate error metrics for validation data.

    Uses only observations over surface types 1 - 8 and 12 - 16 and
    that are not marked as snow by the radar.

    Args:
        results: Dictionary holding xr.Datasets with the results for the different retrievals.
        groups: Names of the retrievals for which to calculate error metrics.
        rqi_threshold: Optional additional rqi_threshold to filter co-locations.
        no_orographic: Whether or not to include precipitation over mountain surfaces.
        no_ocean: Whether or not to include precipitation over ocean.
        no_frozen: If 'True', precipitation classified as frozen will be excluded from
            the analysis.
        no_snow_sfc: If 'True', precipitation over snow surfaces will be excluded
            the analysis.

    Return:

        A pandas DataFrame containing the retrieval metrics.
    """
    bias = {}
    correlation = {}
    mse = {}
    mae = {}
    mape = {}
    far = {}
    pod = {}

    for group in groups:
        if fpa:
            sp_ref = results[group].surface_precip_ref_avg.data
        else:
            sp_ref = results[group].surface_precip_ref.data
        sp = results[group].surface_precip.data

        valid = (sp_ref >= 0) * (sp >= 0)

        if "surface_type" in results[group]:

            surface_type = results[group].surface_type.data
            sea_ice = (surface_type == 2) + (surface_type == 16)
            valid *= ~sea_ice

            if no_orographic:
                valid *= (surface_type < 17)
            if no_snow_sfc:
                valid *= ((surface_type < 8) + (surface_type > 11))
            if no_ocean:
                valid *= surface_type > 1

        if isinstance(ranges, tuple):
            rng = results[group].range.data
            valid *= (rng >= ranges[0])
            valid *= (rng <= ranges[1])
        elif ranges is not None:
            rng = results[group].range.data
            valid *= (rng <= ranges)

        lats = results[group].latitude.data
        lons = results[group].longitude.data
        if region is not None:
            lon_0, lat_0, lon_1, lat_1 = REGIONS[region]
            valid *= ((lons >= lon_0) * (lons < lon_1) *
                    (lats >= lat_0) * (lats < lat_1))

        if "mask" in results[group] and no_frozen:
            mask = results[group].mask
            frozen = (
                np.isclose(mask, 3.0) +
                np.isclose(mask, 4.0) +
                np.isclose(mask, 7.0)
            )
            valid *= ~frozen

        if "rqi" in results[group].variables:
            rqi = results[group].rqi.data
            valid *=  (rqi > rqi_threshold)

        sp = sp[valid]
        sp_ref = sp_ref[valid]

        bias[group] = np.mean(sp - sp_ref) / np.mean(sp_ref) * 100.0
        mse[group] = np.mean((sp - sp_ref) ** 2)
        mae[group] = np.mean(np.abs(sp - sp_ref))

        ref = 0.5 * (np.abs(sp) + np.abs(sp_ref))
        rel_err = np.abs((sp - sp_ref) / ref)

        mape[group] = np.mean(rel_err[sp_ref > 1e-1]) * 100
        corr = np.corrcoef(x=sp_ref, y=sp)
        correlation[group] = corr[0, 1]

    data = {
        "Bias": list(bias.values()),
        "MAE": list(mae.values()),
        "MSE": list(mse.values()),
        "Correlation coeff.": list(correlation.values()),
        "SMAPE": list(mape.values()),
    }
    names = [NAMES[g] for g in groups]
    return pd.DataFrame(data, index=names)


def calculate_explained_error(
        results,
        groups,
        rqi_threshold=0.8,
        region=None,
        ranges=None,
        fpa=False,
        no_orographic=False
):
    """
    Calculates the fraction of error explained by GPM CMB error.

    Uses only observations over surface types 1 - 8 and 12 - 16 and
    that are not marked as frozen by the radar.

    Args:
       results: Dictionary holding xr.Datasets with the results for the different retrievals.
       groups: Names of the retrievals for which to calculate error metrics.
       rqi_threshold: Optional additional rqi_threshold to filter co-locations.
       no_orographic: Whether or not to include precipitation over mountain surfaces.
    """
    var_exp = {}

    for group in groups:
        if fpa:
            sp_ref = results[group].surface_precip_ref_avg.data
        else:
            sp_ref = results[group].surface_precip_ref.data
        sp = results[group].surface_precip.data
        sp_cmb = results["combined"].surface_precip.data

        valid = (sp_ref >= 0) * (sp >= 0) * (sp_cmb >= 0.0)

        if "surface_type" in results[group]:
            surface_type = results[group].surface_type.data
            if no_orographic:
                valid *= ((surface_type < 8) + ((surface_type > 11) * (surface_type < 17)))
            else:
                valid *= ((surface_type < 8) + (surface_type > 11))


        if isinstance(ranges, tuple):
            rng = results[group].range.data
            valid *= (rng >= ranges[0])
            valid *= (rng <= ranges[1])
        elif ranges is not None:
            rng = results[group].range.data
            valid *= (rng <= ranges)

        lats = results[group].latitude.data
        lons = results[group].longitude.data
        if region is not None:
            lon_0, lat_0, lon_1, lat_1 = REGIONS[region]
            valid *= ((lons >= lon_0) * (lons < lon_1) *
                    (lats >= lat_0) * (lats < lat_1))


        if "mask" in results[group]:
            mask = results[group].mask
            frozen = np.isclose(mask, 3.0) + np.isclose(mask, 4.0)
            valid *= ~frozen

        if "rqi" in results[group].variables:
            rqi = results[group].rqi.data
            valid *=  (rqi > rqi_threshold)

        sp = sp[valid]
        sp_ref = sp_ref[valid]
        sp_cmb = sp_cmb[valid]

        _, _, r, *_ = linregress(sp_cmb - sp_ref, sp - sp_ref)
        var_exp[group] = r ** 2

    data = {
        "Explained variance": list(var_exp),
    }
    names = [NAMES[g] for g in groups]
    return pd.DataFrame(data, index=names)


def calculate_monthly_statistics(
        results,
        group,
        rqi_threshold=0.8,
        region=None,
        ranges=None
):
    """
    Calculates monthly relative biases and correlations.

    Uses only observations over surface types 1 - 8 and 12 - 16 and
    that are not marked as frozen by the radar.

    Args:
       results: Dictionary holding xr.Datasets with the results for the different retrievals.
       groups: Names of the retrievals for which to calculate error metrics.
       rqi_threshold: Optional additional rqi_threshold to filter co-locations.
    """
    sp_ref = results[group].surface_precip_ref.data
    sp = results[group].surface_precip.data
    valid = (sp_ref >= 0) * (sp >= 0)

    surface_type = results[group].surface_type.data
    valid *= ((surface_type < 8) + ((surface_type > 11) * (surface_type < 17)))

    mask = results[group].mask
    frozen = (np.isclose(mask, 3.0) + np.isclose(mask, 4.0) + np.isclose(mask, 7.0))
    valid *= ~frozen

    if ranges is not None:
        rng = results[group].range.data
        valid *= (rng <= ranges)

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


def calculate_seasonal_cycles(
        results,
        group,
        rqi_threshold=0.8,
        region=None,
        ranges=None,
        precip_type=None,
        no_ocean=True,
        no_orographic=True,
        no_frozen=True,
        no_snow_sfc=True
):
    """
    Calculates seasonal cycles.

    Args:
        results: Dictionary holding xr.Datasets with the results for the different retrievals.
        groups: Names of the retrievals for which to calculate error metrics.
        rqi_threshold: Optional additional rqi_threshold to filter co-locations.
        region: Name of a region to restrict the analysis to.
        ranges: Radar range limit to restrict the analysis to. This will only have
            an effect for the comparison against the Kwajalein data.
        precip_type: Precipitation type to restrict the analysis to.
        no_ocean: If 'True', measurements over ocean will be excluded.
        no_orographic: If 'True', measurements over mountains will be excluded.
        no_frozen: If 'True', precipitation classified as frozen is excluded from
            the analysis.
        no_snow_sfc: If 'True', precipitation over snow surfaces is excluded from
            the analysis.

    Return:
        A tuple ``(months, precip)`` containing the months and the corresponding
        normalized precipitation.
    """
    if group == "reference":
        sp_ref = results["gprof_nn_1d"].surface_precip_ref.data
        sp = results["gprof_nn_1d"].surface_precip_ref.data
        group = "gprof_nn_1d"
    else:
        sp_ref = results[group].surface_precip_ref.data
        sp = results[group].surface_precip.data
    valid = (sp_ref >= 0) * (sp >= 0)

    surface_type = results[group].surface_type.data

    if "surface_type" in results[group]:
        surface_type = results[group].surface_type.data
        if no_orographic:
            valid *= (surface_type < 17)
        if no_snow_sfc:
            valid *= ((surface_type < 8) + (surface_type > 11))
        if no_ocean:
            valid *= (surface_type > 1)

    mask = results[group].mask
    if no_frozen:
        frozen = (
            np.isclose(mask, 3.0) +
            np.isclose(mask, 4.0) +
            np.isclose(mask, 7.0)
        )
        valid *= ~frozen

    lats = results[group].latitude.data
    lons = results[group].longitude.data

    if region is not None:
        lon_0, lat_0, lon_1, lat_1 = REGIONS[region]
        valid *= ((lons >= lon_0) * (lons < lon_1) *
                  (lats >= lat_0) * (lats < lat_1))

    if ranges is not None:
        rng = results[group].range.data
        valid *= (rng <= ranges)

    sp_ref = sp_ref[valid]
    sp = sp[valid]
    mask = mask[valid]

    time = results[group].time[valid]
    months = time.dt.month.data.astype(float)

    bins = (np.linspace(0, 12, 13) + 0.5)

    mean_precip = binned_statistic(months, sp, bins=bins)[0]
    if precip_type is not None:
        tot_precip = binned_statistic(months, sp, bins=bins, statistic=np.sum)[0]
        t_mask = np.zeros_like(sp, dtype=bool)
        for val in PRECIP_TYPES[precip_type]:
            t_mask += np.isclose(mask, val)
        if np.any(t_mask):
            t_contrib = binned_statistic(months[t_mask],
                                         sp[t_mask],
                                         statistic=np.sum,
                                         bins=bins)[0]
            mean_precip *= t_contrib / tot_precip
        else:
            mean_precip *= np.nan

    mean_precip = np.concatenate([mean_precip[-1:], mean_precip, mean_precip[:1]])
    k = np.ones(3) / 3.0
    mean_precip = convolve(mean_precip, k, mode="valid")

    months = 0.5 * (bins[1:] + bins[:-1])
    return months, mean_precip


def calculate_diurnal_cycles(
        results,
        group,
        rqi_threshold=0.8,
        region=None,
        ranges=None,
        precip_type=None,
        no_ocean=True,
        no_orographic=True,
        no_frozen=True,
        no_snow_sfc=True,
):
    """
    Calculates diurnal cycles.

    Args:
        results: Dictionary holding xr.Datasets with the results for the different retrievals.
        groups: Names of the retrievals for which to calculate error metrics.
        rqi_threshold: Optional additional rqi_threshold to filter co-locations.
        region: Name of a region to restrict the analysis to.
        ranges: Upper limit for radar range to be included in the analysis.
        precip_type: If given, only the contribution of the given precipitation type
            will be calculated.
        no_ocean: If 'True', measurements over ocean will be excluded.
        no_orographic: If 'True', measurements over mountains will be excluded.
        no_frozen: If 'True', precipitation classified as frozen is excluded from
            the analysis.
        no_snow_sfc: If 'True', precipitation over snow surfaces is excluded from
            the analysis.

    Return:
        A tuple ``(hours, precip)`` containing the local time of day in
        ``hours`` and the corresponding mean precipitation rate in
       ``precip``.
    """
    if group == "reference":
        sp_ref = results["gprof_nn_1d"].surface_precip_ref_avg.data
        sp = results["gprof_nn_1d"].surface_precip_ref.data
        group = "gprof_nn_1d"
    else:
        sp_ref = results[group].surface_precip_ref_avg.data
        sp = results[group].surface_precip.data
    valid = (sp_ref >= 0) * (sp >= 0)

    if "surface_type" in results[group]:
        surface_type = results[group].surface_type.data
        if no_orographic:
            valid *= (surface_type < 17)
        if no_snow_sfc:
            valid *= ((surface_type < 8) + (surface_type > 11))
        if no_ocean:
            valid *= (surface_type > 1)


    mask = results[group].mask
    if no_frozen:
        frozen = (
            np.isclose(mask, 3.0) +
            np.isclose(mask, 4.0) +
            np.isclose(mask, 7.0)
        )
        valid *= ~frozen

    if ranges is not None:
        rng = results[group].range.data
        valid *= (rng <= ranges)

    lats = results[group].latitude.data
    lons = results[group].longitude.data

    if region is not None:
        lon_0, lat_0, lon_1, lat_1 = REGIONS[region]
        valid *= ((lons >= lon_0) * (lons < lon_1) *
                  (lats >= lat_0) * (lats < lat_1))

    sp_ref = sp_ref[valid]
    sp = sp[valid]
    mask = mask[valid]

    time = results[group].time[valid]
    time = time + (lons[valid] / 360 * 24 * 60 * 60).astype("timedelta64[s]")
    hours = time.dt.hour.data
    bins = np.linspace(0, 24, 25)

    mean_precip = binned_statistic(hours, sp, bins=bins)[0]

    if precip_type is not None:
        tot_precip = binned_statistic(hours, sp, bins=bins, statistic=np.sum)[0]
        t_mask = np.zeros_like(sp, dtype=bool)
        for val in PRECIP_TYPES[precip_type]:
            t_mask += np.isclose(mask, val)
        if np.any(t_mask):
            t_contrib = binned_statistic(hours[t_mask],
                                         sp[t_mask],
                                         statistic=np.sum,
                                         bins=bins)[0]
            mean_precip *= t_contrib / tot_precip
        else:
            mean_precip *= np.nan

    hours = bins[:-1]

    mean_precip = np.concatenate([mean_precip[-3:], mean_precip, mean_precip[:3]])
    k = np.exp(np.log(0.5) * ((np.ones(7) - 3) / 3) ** 2)
    k /= k.sum()
    mean_precip = convolve(mean_precip, k, mode="valid")

    return hours, mean_precip


def get_colors():
    """
    Return dictionary of plot colors for different retrievals.
    """
    c0 = to_rgba("C0")
    c0_d = np.array(c0).copy()
    c0_d[:3] *= 0.5
    c1 = to_rgba("C1")
    c1_d = np.array(c1).copy()
    c1_d[:3] *= 0.7
    c1_dd = np.array(c1).copy()
    c1_dd[:3] *= 0.4
    c2 = to_rgba("C2")
    bar_palette = [c0, c0_d, c1, c1_d, c2]

    return {
        "gprof_v5": c0,
        "gprof_v7": c0_d,
        "gprof_nn_1d": c1,
        "gprof_nn_3d": c1_d,
        "gprof_nn_hr": c1_dd,
        "combined": c2
    }

def extract_results_from_file(
        filename,
        groups,
        rqi_threshold,
        mask_group="gprof_v7",
        reference_group="reference"
):
    """
    Extract results from a combined collocation file.

    This utility function loads all valid pixels from a collocation
    files into an xarray dataset.

    Args:
        filename: Path to the collocation file containing the collocated
            retrieval results as a NetCDF4 file with a separate group
            for every retrieval.
        groups: The groups for which to extract the results.
        rqi_thresholds: If the 'reference' group has a 'radar_quality_index'
            variable (as is the case for MRMS collocations over CONUS), then
            only pixels with radar quality indices above this threshold will
            be extracted.
        mask_group: Only pixels where this group has valid retrieval results
            will be extracted.
        reference_group: The group whose 'surface_precip' values to as the
            the reference value.

    Return:
        An xarray.Dataset containing the loaded pixels as a 1D dataset.
    """
    ref = xr.load_dataset(filename, group="reference")
    time_ref, _ = xr.broadcast(ref.time, ref.surface_precip)
    time_ref = time_ref.data

    try:
        ref = xr.load_dataset(filename, group = reference_group)
    except OSError:
        LOGGER.error("File '%s' has no '%s' group.",
                     filename,
                     reference_group)
        return None
    sp_ref = ref.surface_precip.data
    if "surface_precip_avg" in ref:
        sp_ref_avg = ref.surface_precip_avg.data
    else:
        sp_ref_avg = ref.surface_precip.data
    if "radar_quality_index" in ref.variables:
        rqi = ref.radar_quality_index.data
    else:
        rqi = None

    if "mask" in ref.variables:
        mask = ref.mask.data
    else:
        mask = None

    ref = xr.load_dataset(filename, group="reference")
    lats = ref.latitude.data
    lons = ref.longitude.data

    results = {}

    try:
        gprof = xr.load_dataset(filename, group=mask_group)
        gprof_mask = gprof.surface_precip.data >= 0
        if rqi is None and "radar_quality_index" in gprof.variables:
            rqi = gprof.radar_quality_index.data
    except OSError:
        LOGGER.error("File '%s' has no '%s' group.", filename, mask_group)
        return None

    try:
        surface_types = xr.load_dataset(filename, group="gprof_nn_1d").surface_type.data
    except OSError:
        LOGGER.error("File '%s' has no '%s' group.", filename, "gprof_nn_3d")
        return None

    for group in groups:
        try:
            data = xr.load_dataset(filename, group=group)
            sp = data.surface_precip.data

            valid = (sp >= 0.0)
            valid = (sp_ref >= 0.0) * (sp >= 0.0) * (gprof_mask)
            if rqi is not None:
                valid = valid * (rqi >= rqi_threshold)

            along_track = data.along_track.data
            across_track = data.across_track.data
            across_track, _  = np.meshgrid(across_track, along_track)
            across_track = across_track[valid]

            samples = xr.Dataset(
                {
                    "surface_precip": (("samples",), sp[valid]),
                    "surface_precip_ref_avg": (("samples",), sp_ref_avg[valid]),
                    "surface_precip_ref": (("samples",), sp_ref[valid]),
                    "latitude": (("samples",), lats[valid]),
                    "longitude": (("samples",), lons[valid]),
                    "time": (("samples",), time_ref[valid]),
                    "across_track": (("samples",), across_track),
                }
            )
            if "surface_precip_avg" in data:
                samples["surface_precip_avg"] = (
                    ("samples", ), data.surface_precip_avg.data[valid]
                )
            else:
                samples["surface_precip_avg"] = (
                    ("samples",), data.surface_precip.data[valid]
                )
            if "pop" in data.variables:
                samples["pop"] = (("samples",), data.pop.data[valid])
            if rqi is not None:
                samples["rqi"] = (("samples",), rqi[valid])
            if mask is not None:
                samples["mask"] = (("samples",), mask[valid])
            samples["surface_type"] = (("samples"), surface_types[valid])
            if "airmass_type" in gprof.variables:
                samples["airmass_type"] = (("samples"), gprof.airmass_type.data[valid])
            if "range" in ref.variables:
                ranges, _ = xr.broadcast(ref.range, ref.surface_precip)
                samples["range"] = (("samples"), ranges.data[valid])
            results[group] = samples
        except OSError as e:
            LOGGER.error(
                    "The following error occurred during processing of "
                    " file '%s': \n %s",  filename, e
            )
            return None
    return results


def extract_results(
        path,
        groups,
        rqi_threshold,
        mask_group="gprof_v7",
        reference_group="reference"
):
    """
    Extracts collocation results from all collocations files in a given
    directory.

    Args:
        path: Directory containing the collocated validation resullts.
        groups: List of the names of the groups containing the retrieval results.
        rqi_threshold: A threshold for the minimum Radar Quality Index (RQI) of the radar
            measurements to be included in the results.

    Return:
        A dict mapping the group names of the retrieval products to datasets containing the
        retrieved precipitation 'surface_precip' and the ref precipitation as
        'surface_precip_ref'.
    """
    files = list(Path(path).glob("*.nc"))
    results = {}
    pool = ProcessPoolExecutor(max_workers=8)
    tasks = []
    for filename in files:
        args = [
            filename,
            groups,
            rqi_threshold
        ]
        kwargs = {
            "mask_group": mask_group,
            "reference_group": reference_group
        }
        tasks.append(pool.submit(extract_results_from_file, *args, **kwargs))

    for filename, task in zip(files, tasks):
        try:
            result = task.result()
            if result is not None:
                for k, stats in result.items():
                    results.setdefault(k, []).append(stats)

        except Exception as e:
            LOGGER.exception(e)

    for k in results:
        results[k] = xr.concat(results[k], "samples")
    pool.shutdown()
    return results


def gridded_stats(
        results: xr.Dataset,
        bins: Tuple[np.ndarray],
        min_samples: Optional[int] = None, fpa: bool = False,
        no_orographic=False,
        no_frozen=True,
        no_ocean=True,
        no_snow_sfc=True

):
    """
    Calculate correlation of retrieval and prediction over lat-lon grid.

    Args:
        results: An xr.Dataset containing the retrieved precipitation and
            the validation precipitation.
        bins: A tuple ``(lon_bins, lats_bins) containing the longitude
            and latitude bins.
        min_samples: The minimum numbers of samples in each bin.
        fpa: If True, the footprint-averaged reference precipitation will be
            used.
        no_orographic: Whether or not to include precipitation over mountain surfaces.
        no_ocean: Whether or not to include precipitation over ocean.
        no_frozen: If 'True', precipitation classified as frozen will be excluded from
            the analysis.
        no_snow_sfc: If 'True', precipitation over snow surfaces will be excluded
            the analysis.

    Return:
        A tuple ``(bias, mae, mse, corr)`` containing the bias, mean absolute
        error, mean-squared error and correlation for the given
    longitude-latitude.
    """
    sp = results.surface_precip
    if fpa:
        sp_ref = results.surface_precip_ref_avg
    else:
        sp_ref = results.surface_precip_ref

    valid = (sp >= 0.0) * (sp_ref >= 0.0)

    if "surface_type" in results:
        sfc = results.surface_type.data

        sea_ice = (sfc == 2) + (sfc == 16)
        valid *= ~sea_ice

        if no_ocean:
            valid *= (sfc > 1)

        if no_orographic:
            valid *= (sfc < 17)

        if no_snow_sfc:
            valid *= ((sfc < 8) + (sfc > 11))

    if "mask" in results:
        mask = results.mask
        if no_frozen:
            frozen = (
                np.isclose(mask, 3.0) +
                np.isclose(mask, 4.0) +
                np.isclose(mask, 7.0)
            )
            valid *= ~frozen

    sp = sp[valid]
    sp_ref = sp_ref[valid]
    lons = results.longitude.data[valid]
    lats = results.latitude.data[valid]

    xx = sp * sp
    xy = sp * sp_ref
    yy = sp_ref * sp_ref
    x = sp
    y = sp_ref

    mae = binned_statistic_2d(lons, lats, np.abs(x - y), bins=bins)[0]
    mse = binned_statistic_2d(lons, lats, (x - y) ** 2, bins=bins)[0]
    xx = binned_statistic_2d(lons, lats, xx, bins=bins)[0]
    xy = binned_statistic_2d(lons, lats, xy, bins=bins)[0]
    yy = binned_statistic_2d(lons, lats, yy, bins=bins)[0]
    x = binned_statistic_2d(lons, lats, x, bins=bins)[0]
    y = binned_statistic_2d(lons, lats, y, bins=bins)[0]
    n_samples = binned_statistic_2d(lons, lats, lons, "count", bins=bins)[0]

    if min_samples is not None:
        mae[n_samples < min_samples] = np.nan
        mse[n_samples < min_samples] = np.nan
        xx[n_samples < min_samples] = np.nan
        xy[n_samples < min_samples] = np.nan
        yy[n_samples < min_samples] = np.nan
        x[n_samples < min_samples] = np.nan
        y[n_samples < min_samples] = np.nan

    sigma_x = np.sqrt(xx - x ** 2)
    sigma_y = np.sqrt(yy - y ** 2)
    corr = (xy - x * y) / (sigma_x * sigma_y)
    return x - y, mae, mse, corr


def calculate_pr_curve(
        results: xr.Dataset,
        fpa: bool = False,
        no_orographic=False,
        no_frozen=True,
        no_ocean=True,
        no_snow_sfc=True

):
    """
    Calculate precision-recall curve for retrieval.

    Args:
        results: An xr.Dataset containing the retrieved precipitation and
            the validation precipitation.
        fpa: If True, the footprint-averaged reference precipitation will be
            used.
        no_orographic: Whether or not to include precipitation over mountain surfaces.
        no_ocean: Whether or not to include precipitation over ocean.
        no_frozen: If 'True', precipitation classified as frozen will be excluded from
            the analysis.
        no_snow_sfc: If 'True', precipitation over snow surfaces will be excluded
            the analysis.

    Return:
        A tuple ``(prec, rec, thresh)`` containing the precision values
        in ``prec``, the recall values in ``rec`` and the thresholds values
        in ``thresh``.
    """
    from sklearn.metrics import precision_recall_curve

    if "pop" in results:
        pop = results.pop
    else:
        pop = results.surface_precip
    if fpa:
        sp_ref = results.surface_precip_ref_avg
    else:
        sp_ref = results.surface_precip_ref

    valid = (pop >= 0.0) * (sp_ref >= 0.0)

    if "surface_type" in results:
        sfc = results.surface_type.data

        valid *= ((sfc != 2) * (sfc != 16))

        if no_ocean:
            valid *= (sfc > 1)

        if no_orographic:
            valid *= (sfc < 17)

        if no_snow_sfc:
            valid *= ((sfc < 8) + (sfc > 11))

    if "mask" in results:
        mask = results.mask
        if no_frozen:
            frozen = (
                np.isclose(mask, 3.0) +
                np.isclose(mask, 4.0) +
                np.isclose(mask, 7.0)
            )
            valid *= ~frozen

    pop = pop[valid]
    sp_ref = sp_ref[valid]
    prec, rec, threshs = precision_recall_curve(
        sp_ref > 1e-4,
        pop,
    )
    return prec, rec, threshs
