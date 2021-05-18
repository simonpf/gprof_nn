"""
=================
gprof_nn.data.sim
=================

This module contains functions to read and convert .sim files for GPROF
 v. 7.
"""
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from pykdtree.kdtree import KDTree
from netCDF4 import Dataset
from rich.progress import track
import xarray as xr

from gprof_nn.coordinates import latlon_to_ecef
from gprof_nn.data.preprocessor import PreprocessorFile, run_preprocessor
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.training_data import PROFILE_NAMES
from gprof_nn.data.mrms import MRMSMatchFile
from gprof_nn.utils import CONUS

ALL_TARGETS = [
    "surface_precip",
    "convective_precip",
    "cloud_water_content",
    "rain_water_content",
    "snow_water_content",
    "latent_heat",
]

LEVELS = np.concatenate([np.linspace(500.0, 1e4, 20), np.linspace(11e3, 18e3, 8)])

LOGGER = logging.getLogger(__name__)

###############################################################################
# Data types
###############################################################################

N_LAYERS = 28
N_FREQS = 15
DATE_TYPE = np.dtype(
    [
        ("year", "i4"),
        ("month", "i4"),
        ("day", "i4"),
        ("hour", "i4"),
        ("minute", "i4"),
        ("second", "i4"),
    ]
)

GMI_HEADER_TYPES = np.dtype(
    [
        ("satellite_code", "a5"),
        ("sensor", "a5"),
        ("frequencies", f"{N_FREQS}f4"),
        ("nominal_eia", f"{N_FREQS}f4"),
        ("start_pixel", "i4"),
        ("end_pixel", "i4"),
        ("start_scan", "i4"),
        ("end_scan", "i4"),
    ]
)

GMI_PIXEL_TYPES = np.dtype(
    [
        ("pixel_index", "i4"),
        ("scan_index", "i4"),
        ("data_source", "f4"),
        ("latitude", "f4"),
        ("longitude", "f4"),
        ("elevation", "f4"),
        ("scan_time", DATE_TYPE),
        ("surface_type", "i4"),
        ("surface_precip", "f4"),
        ("convective_precip", "f4"),
        ("emissivity", f"{N_FREQS}f4"),
        ("rain_water_content", f"{N_LAYERS}f4"),
        ("snow_water_content", f"{N_LAYERS}f4"),
        ("cloud_water_content", f"{N_LAYERS}f4"),
        ("latent_heat", f"{N_LAYERS}f4"),
        ("tbs_observed", f"{N_FREQS}f4"),
        ("tbs_simulated", f"{N_FREQS}f4"),
        ("d_tbs", f"{N_FREQS}f4"),
        ("tbs_bias", f"{N_FREQS}f4"),
    ]
)

###############################################################################
# GPROF GMI Simulation files
###############################################################################


class GMISimFile:
    """
    Interface class to read GPROF .sim files.
    """

    file_pattern = "GMI.dbsatTb.????????.??????.sim"

    @classmethod
    def find_files(cls, path, day=None):
        """
        Find all files that match the standard filename pattern for GMI
        sim files.
        """
        if day is None:
            pattern = cls.file_pattern
        else:
            pattern = f"GMI.dbsatTb.??????{day:02}.??????.sim"
        path = Path(path)
        return list(path.glob("**/" + pattern))

    def __init__(self, path):
        """
        Open .sim file.

        Args:
            path: Path to the .sim file to open.
        """
        self.path = path
        parts = str(path).split(".")
        self.granule = int(parts[-2])
        self.year = int(parts[-3][:4])
        self.month = int(parts[-3][4:6])
        self.day = int(parts[-3][6:])

        self.header = np.fromfile(self.path, GMI_HEADER_TYPES, count=1)
        offset = GMI_HEADER_TYPES.itemsize
        self.data = np.fromfile(self.path, GMI_PIXEL_TYPES, offset=offset)

    def match_targets(self, input_data, targets=None):
        """
        Match retrieval targets from .sim file to points in
        xarray dataset.

        Args:
            input_data: xarray dataset containing the input data from
                the preprocessor.
            targets: List of retrieval target variables to extract from
                the sim file.
        Return:
            The input dataset but with the requested retrieval targets added.
        """
        if targets is None:
            targets = ALL_TARGETS
        path_variables = [t for t in targets if "path" in t]
        for v in path_variables:
            profile_variable = v.replace("path", "content").replace("ice", "snow")
            if not profile_variable in targets:
                targets.append(profile_variable)
        targets = [t for t in targets if "path" not in t]

        n_scans = input_data.scans.size
        n_pixels = 221

        dx = 40
        i_c = 110
        ix_start = i_c - dx // 2
        ix_end = i_c + 1 + dx // 2

        lats_1c = input_data["latitude"][:, ix_start:ix_end].data.reshape(-1, 1)
        lons_1c = input_data["longitude"][:, ix_start:ix_end].data.reshape(-1, 1)
        coords_1c = latlon_to_ecef(lons_1c, lats_1c)
        coords_1c = np.concatenate(coords_1c, axis=1)

        lats = self.data["latitude"].reshape(-1, 1)
        lons = self.data["longitude"].reshape(-1, 1)
        coords_sim = latlon_to_ecef(lons, lats)
        coords_sim = np.concatenate(coords_sim, 1)

        kdtree = KDTree(coords_1c)
        dists, indices = kdtree.query(coords_sim)

        for t in targets:
            if t in PROFILE_NAMES:
                n = n_scans * (dx + 1)
                shape = (n_scans, dx + 1, 28)
                full_shape = (n_scans, n_pixels, 28)
                matched = np.zeros((n, 28))
            else:
                n = n_scans * (dx + 1)
                shape = (n_scans, dx + 1)
                full_shape = (n_scans, n_pixels)
                matched = np.zeros(n)

            matched[:] = np.nan
            matched[indices] = self.data[t]
            matched[indices][dists > 5e3] = np.nan
            matched = matched.reshape(shape)

            matched_full = np.zeros(full_shape, dtype=np.float32)
            matched_full[:] = np.nan
            matched_full[:, ix_start:ix_end] = matched

            if t in PROFILE_NAMES:
                input_data[t] = (("scans", "pixels", "levels"), matched_full)
                path = np.trapz(matched_full, x=LEVELS, axis=-1) * 1e-3
                path_name = t.replace("content", "path").replace("snow", "ice")
                input_data[path_name] = (("scans", "pixels"), path)
            else:
                input_data[t] = (("scans", "pixels"), matched_full)
        return input_data


###############################################################################
# Helper functions
###############################################################################


def _extract_scenes(data):
    """
    Extract 221 x 221 pixel wide scenes from dataset where
    ground truth surface precipitation rain rates are
    available.

    Args:
        xarray.Dataset containing the data from the preprocessor together
        with the matches surface precipitation from the .sim file.

    Return:
        New xarray.Dataset which containing 128x128 patches of input data
        and corresponding surface precipitation.
    """
    n = 221
    sp = data["surface_precip"].data

    if np.all(np.isnan(sp)):
        return xr.Dataset()

    i_start, i_end = np.where(np.any(~np.isnan(sp), axis=1))[0][[0, -1]]


    scenes = []
    i = i_start
    while i_start + n < i_end:
        subscene = data[{"scans": slice(i_start, i_start + n)}]
        sp = subscene["surface_precip"]
        if np.isfinite(sp.data).sum() < 500:
            continue
        scenes.append(subscene)
        i_start += n

    if scenes:
        return xr.concat(scenes, "samples")
    return xr.Dataset()


def _find_l1c_file(path, sim_file):
    """
    Find GPROF GMI L1C file corresponding to .sim file.

    Args:
        path: Path pointing to the root of the folder tree containing the
            L1C files.
        sim_files: GMISimFile for which to find the corresponding L1C
            file.

    Return:
        The corresponding L1C file.
    """
    year = sim_file.year - 2000
    month = sim_file.month
    day = sim_file.day
    path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"
    files = path.glob(f"1C-R*{sim_file.granule}*.HDF5")
    return next(iter(files))


def _load_era5_data(start_time, end_time, base_directory):
    """
    Loads ERA5 data matching the start and end time of a L1C
    file.

    Args:
        start_time: First scan time from L1C file.
        end_time: Last scan time from L1C file.
        base_directory: Root of the directory tree containing the
            ERA5 files.
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    year_start = start_time.year
    month_start = start_time.month
    day_start = start_time.day

    year_end = end_time.year
    month_end = end_time.month
    day_end = end_time.day

    file_expr = f"ERA5_{year_start}{month_start:02}{day_start:02}_surf.nc"
    file_start = list(Path(base_directory).glob(f"**/{file_expr}"))
    file_expr = f"ERA5_{year_end}{month_end:02}{day_end:02}_surf.nc"
    file_end = list(Path(base_directory).glob(f"**/{file_expr}"))

    files = list(set(file_start + file_end))
    return xr.concat([xr.load_dataset(f) for f in files], dim="time")


def _add_era5_precip(input_data, l1c_data, era5_data):
    """
    Adds total precipitation from ERA5 to data.

    Args:
        input_data: The preprocessor data to which the atmospheric data
            from the sim file has been added. Must contain "surface_precip"
            variable.
        l1c_data: The L1C data corresponding to input_data.
        era5_data: The era5 data covering the time range of observations
            in l1c_data.
    """
    l0 = era5_data[{"longitude": slice(0, 1)}].copy(deep=True)
    l0 = l0.assign_coords({"longitude": [360.0]})
    era5_data = xr.concat([era5_data, l0], "longitude")
    n_scans = l1c_data.scans.size
    n_pixels = l1c_data.pixels.size
    indices = input_data["surface_type"] == 2
    lats = xr.DataArray(input_data["latitude"].data[indices], dims="samples")
    lons = input_data["longitude"].data[indices]
    lons = np.where(lons < 0.0, lons + 360, lons)
    lons = xr.DataArray(lons, dims="samples")
    time = np.broadcast_to(
        l1c_data["scan_time"].data.reshape(-1, 1), (n_scans, n_pixels)
    )[indices]
    time = xr.DataArray(time, dims="samples")
    tp = era5_data["tp"].interp(
        {"latitude": lats, "longitude": lons, "time": time}, method="linear"
    )
    input_data["surface_precip"].data[indices] = tp.data


def process_sim_file(sim_filename, l1c_path, era5_path):
    """
    Extract 2D training scenes from sim file.

    This method reads a given sim file, matches it with the input data
    from the corresponding l1c and preprocessor files, adds ERA5 precip
    over sea ice and returns a dataset of 221 x 221 scenes for training
    of the GPROF-NN 2D algorithm.

    Args:
        sim_filename: Filename of the Sim file to process.
        l1c_path: Base path of the directory tree containing the L1C file.
        era5_path: Base path of the directory containing the ERA5 data.

    Return:
        xarray.Dataset containing the data from the sim file as multiple
        2D training scenes.
    """
    sim_file = GMISimFile(sim_filename)
    l1c_file = L1CFile.open_granule(sim_file.granule, l1c_path)

    LOGGER.info("Running preprocessor for sim file %s.", sim_filename)
    data_pp = run_preprocessor(l1c_file.filename)
    LOGGER.info("Matching retrieval targets for file %s.", sim_filename)
    sim_file.match_targets(data_pp)
    l1c_data = l1c_file.to_xarray_dataset()

    LOGGER.info("Adding ERA5 precip for file %s.", sim_filename)
    start_time = data_pp["scan_time"].data[0]
    end_time = data_pp["scan_time"].data[-1]
    era5_data = _load_era5_data(start_time, end_time, era5_path)
    _add_era5_precip(data_pp, l1c_data, era5_data)

    scenes = _extract_scenes(data_pp)
    return scenes


def process_mrms_file(mrms_filename, day, l1c_path):
    """
    Extract training data from MRMS-GMI match up files for given day.
    Matches the observations in the MRMS file with input data from the
    preprocessor, extracts observations over snow and returns a dataset
    of 2D training scenes for the GPROF-NN-2D algorithm.

    Args:
        mrms_filename: Filename of the MRMS file to process.
        day: The day of the month for which to extract data.
        l1c_path: Path to the root of the directory tree containing the L1C
             observations.
    """
    mrms_file = MRMSMatchFile(mrms_filename)
    l1c_files = L1CFile.find_files(
        mrms_file.scan_time[mrms_file.n_obs // 2], CONUS, l1c_path
    )
    scenes = []
    LOGGER.info("Found %s L1C file for MRMS file %s.",
                len(l1c_files),
                mrms_filename)
    for f in l1c_files:
        data_pp = run_preprocessor(f.filename)
        surface_type = data_pp["surface_type"]
        snow = (surface_type >= 8) * (surface_type < 11)
        if snow.sum() <= 0:
            continue

        LOGGER.info("Matching MRMS data for %s.",
                    f.filename)
        mrms_file.match_targets(data_pp)
        # Keep only obs over snow.
        data_pp["surface_precip"].data[~snow] = np.nan
        scenes.append(_extract_scenes(data_pp))

    return xr.concat(scenes, "samples")


###############################################################################
# File processor
###############################################################################


class SimFileProcessor:
    def __init__(
        self,
        output_file,
        sim_file_path=None,
        mrms_path=None,
        l1c_path=None,
        era5_path=None,
        n_workers=4,
        day=None,
    ):
        """
        Create retrieval driver.

        Args:
            path: The folder containing the input files.
            pattern: glob pattern to use to subselect input files.
            output_path: The path to which to write the retrieval
                 results
            input_class: The class to use to read and process the input files.
            n_workers: The number of worker processes to use.
            days: The days of each month to process.
        """

        self.output_file = output_file
        if sim_file_path is not None:
            self.sim_file_path = Path(sim_file_path)
        else:
            self.sim_file_path = None
        if mrms_path is not None:
            self.mrms_path = Path(mrms_path)
        else:
            self.mrms_path = None

        if l1c_path is None:
            raise ValueError(
                "The 'l1c_path' argument must be provided in order to process any"
                "sim files."
            )
        self.l1c_path = Path(l1c_path)

        if era5_path is None:
            raise ValueError(
                "The 'era5_path' argument must be provided in order to process any"
                "sim files."
            )
        self.era5_path = Path(era5_path)
        self.pool = ProcessPoolExecutor(max_workers=n_workers)

        if day is None:
            self.day = 1
        else:
            self.day = day

    def run(self):
        """
        Start the processing.

        This will start processing all suitable input files that have been found and
        stores the names of the produced result files in the ``processed`` attribute
        of the driver.
        """
        if self.sim_file_path is not None:
            sim_files = GMISimFile.find_files(self.sim_file_path, day=self.day)
        else:
            sim_files = []
        if self.mrms_path is not None:
            mrms_files = MRMSMatchFile.find_files(self.mrms_path)
        else:
            mrms_files = []

        tasks = []
        for f in sim_files:
            tasks.append(self.pool.submit(process_sim_file,
                                          f,
                                          self.l1c_path,
                                          self.era5_path))
        for f in mrms_files:
            tasks.append(self.pool.submit(process_mrms_file,
                                          f,
                                          self.day,
                                          self.l1c_path))

        datasets = []
        for t in track(tasks, description="Extracting data"):
            datasets.append(t.result())

        dataset = xr.concat(datasets, "samples")
        dataset.to_netcdf(self.output_file)
