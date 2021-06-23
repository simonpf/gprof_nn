"""
=================
gprof_nn.data.sim
=================

This module contains functions to read and convert .sim files for GPROF
 v. 7.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from pathlib import Path
import traceback
import sys

import numpy as np
import pandas as pd
import pyproj
from pykdtree.kdtree import KDTree
from netCDF4 import Dataset
from rich.progress import track
import xarray as xr

from gprof_nn.definitions import ALL_TARGETS, N_LAYERS, LEVELS
from gprof_nn.coordinates import latlon_to_ecef
from gprof_nn.data.preprocessor import PreprocessorFile, run_preprocessor
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.training_data import PROFILE_NAMES
from gprof_nn.data.mrms import (MRMSMatchFile,
                                get_surface_type_map,
                                get_surface_type_map_legacy)
from gprof_nn import sensors
from gprof_nn.utils import CONUS
from gprof_nn.logging import console

N_PIXELS_CENTER = 41

LOGGER = logging.getLogger(__name__)

###############################################################################
# Data types
###############################################################################

N_FREQS = 15
HEADER_TYPES = np.dtype(
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

###############################################################################
# GPROF GMI Simulation files
###############################################################################


class SimFile:
    """
    Interface class to read GPROF .sim files.
    """
    @classmethod
    def find_files(cls,
                   path,
                   sensor=sensors.GMI,
                   day=None):
        """
        Find all files that match the standard filename pattern for GMI
        sim files.
        """
        if day is None:
            pattern = sensor.FILE_PATTERN.format(day="??")
        else:
            pattern = sensor.FILE_PATTERN.format(day=f"{day:02}")
        path = Path(path)
        files = list(path.glob("**/????/" + pattern))
        if not files:
            files = list(path.glob("**/" + pattern))
        return files

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

        self.header = np.fromfile(self.path, HEADER_TYPES, count=1)
        offset = HEADER_TYPES.itemsize
        sensor = self.header["sensor"][0].decode().strip()
        try:
            sensor = getattr(sensors, sensor.upper())
        except AttributeError:
            raise Exception(
                f"The sensor {sensor} isn't currently supported."
            )
        self.sensor = sensor
        self.data = np.fromfile(self.path,
                                sensor.SIM_FILE_RECORD,
                                offset=offset)

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
            if profile_variable not in targets:
                targets.append(profile_variable)
        targets = [t for t in targets if "path" not in t]

        n_scans = input_data.scans.size
        n_pixels = 221
        dx = 40
        i_c = 110
        ix_start = i_c - dx // 2
        ix_end = i_c + 1 + dx // 2
        i_left = i_c - (N_PIXELS_CENTER // 2 + 1)
        i_right = i_c + (N_PIXELS_CENTER // 2)

        lats_1c = input_data["latitude"][:, ix_start:ix_end].data.reshape(-1, 1)
        lons_1c = input_data["longitude"][:, ix_start:ix_end].data.reshape(-1, 1)
        coords_1c = latlon_to_ecef(lons_1c, lats_1c)
        coords_1c = np.concatenate(coords_1c, axis=1)

        lats = self.data["latitude"].reshape(-1, 1)
        lons = self.data["longitude"].reshape(-1, 1)
        coords_sim = latlon_to_ecef(lons, lats)
        coords_sim = np.concatenate(coords_sim, 1)

        # Determine indices of matching L1C observations.
        kdtree = KDTree(coords_1c)
        dists, indices = kdtree.query(coords_sim)

        n_angles = 0
        if hasattr(self.sensor, "N_ANGLES"):
            n_angles = self.sensor.N_ANGLES

        # Extract matching data
        for t in targets:
            if t in PROFILE_NAMES:
                n = n_scans * (dx + 1)
                shape = (n_scans, dx + 1, 28)
                full_shape = (n_scans, n_pixels, 28)
                matched = np.zeros((n, 28), dtype=np.float32)
            else:
                n = n_scans * (dx + 1)
                if n_angles > 0:
                    shape = (n_scans, dx + 1, n_angles)
                    full_shape = (n_scans, n_pixels, n_angles)
                    matched = np.zeros((n, n_angles), dtype=np.float32)
                else:
                    shape = (n_scans, dx + 1)
                    full_shape = (n_scans, n_pixels)
                    matched = np.zeros(n, dtype=np.float32)

            matched[:] = np.nan
            matched[indices, ...] = self.data[t]
            matched[indices, ...][dists > 5e3] = np.nan
            matched = matched.reshape(shape)

            matched_full = np.zeros(full_shape, dtype=np.float32)
            matched_full[:] = np.nan
            matched_full[:, ix_start:ix_end] = matched

            if t in PROFILE_NAMES:
                data = matched_full[:, i_left:i_right]
                input_data[t] = (("scans", "pixels_center", "levels"),
                                 data.astype(np.float32))
                if "content" in t:
                    path = np.trapz(data, x=LEVELS, axis=-1) * 1e-3
                    path_name = t.replace("content", "path").replace("snow", "ice")
                    input_data[path_name] = (("scans", "pixels_center"), path)
            else:
                if t in ["surface_precip", "convective_precip"]:
                    dims = ("scans", "pixels")
                    if n_angles > 0:
                        dims = dims + ("angles",)
                    input_data[t] = (dims, matched_full.astype(np.float32))
                else:
                    input_data[t] = (("scans", "pixels_center"),
                                     matched_full[:, i_left:i_right].astype(np.float32))

        return input_data


ENHANCEMENT_FACTORS = {
    "ERA5": {
        (17, 0): 1.35683,
        (17, 1): 2.05213,
        (17, 2): 1.62242,
        (17, 3): 1.87049,
        (18, 0): 3.91369
    },
    "GANAL": {
        (17, 0): 1.58177,
        (17, 1): 1.81539,
        (18, 0): 3.91369,
    }
}

def apply_orographic_enhancement(data,
                                 kind="ERA5"):
    """
    Applies orographic enhancement factors to 'surface_precip' and
    'convective_precip' targets.

    Args:
        data: xarray.Dataset containing variables surface_precip,
            convective_precip, surface_type and airmass_type.
         kind: "ERA5" or "GANAL" depending on the source of ancillary data.

    Returns:
        None; Correct is applied in place.
    """
    kind = kind.upper()
    if kind not in ["ERA5", "GANAL"]:
        raise ValueError(
            "The kind argument to  must be 'ERA5' or 'GANAL'."
        )
    surface_types = data["surface_type"].data
    airmass_types = data["airmass_type"].data
    surface_precip = data["surface_precip"].data
    convective_precip = data["convective_precip"].data

    enh = np.ones(surface_precip.shape, dtype=np.float32)
    factors = ENHANCEMENT_FACTORS[kind]
    for st in [17, 18]:
        for at in range(4):
            key = (st, at)
            if not key in factors:
                continue
            indices = (surface_types == st) * (airmass_types == at)
            enh[indices] = factors[key]

    surface_precip *= enh
    convective_precip *= enh

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
        return None

    i_start, i_end = np.where(np.any(~np.isnan(sp), axis=1))[0][[0, -1]]


    scenes = []
    i = i_start
    while i_start + n < i_end:
        subscene = data[{"scans": slice(i_start, i_start + n)}]
        sp = subscene["surface_precip"]
        scenes.append(subscene)
        i_start += n

    if scenes:
        return xr.concat(scenes, "samples")
    return None

def _find_l1c_file(path, sim_file):
    """
    Find GPROF GMI L1C file corresponding to .sim file.

    Args:
        path: Path pointing to the root of the folder tree containing the
            L1C files.
        sim_files: SimFile for which to find the corresponding L1C
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

    surface_types = input_data["surface_type"].data
    indices = ((surface_types == 2) + (surface_types == 16))

    lats = xr.DataArray(input_data["latitude"].data[indices], dims="samples")
    lons = input_data["longitude"].data[indices]
    lons = np.where(lons < 0.0, lons + 360, lons)
    lons = xr.DataArray(lons, dims="samples")
    time = np.broadcast_to(
        l1c_data["scan_time"].data.reshape(-1, 1), (n_scans, n_pixels)
    )[indices]
    time = xr.DataArray(time, dims="samples")
    # Interpolate and convert to mm/h
    tp = era5_data["tp"].interp(
        {"latitude": lats, "longitude": lons, "time": time}, method="nearest"
    )
    input_data["surface_precip"].data[indices] = 1000.0 * tp.data
    cp = era5_data["cp"].interp(
        {"latitude": lats, "longitude": lons, "time": time}, method="nearest"
    )
    input_data["convective_precip"].data[indices] = 1000.0 * cp.data


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
    import gprof_nn.logging
    LOGGER.info("Starting processing sim file %s.", sim_filename)
    sim_file = SimFile(sim_filename)
    l1c_file = L1CFile.open_granule(sim_file.granule, l1c_path)

    LOGGER.info("Running preprocessor for sim file %s.", sim_filename)
    data_pp = run_preprocessor(l1c_file.filename)
    if data_pp is None:
        return None
    LOGGER.info("Matching retrieval targets for file %s.", sim_filename)
    sim_file.match_targets(data_pp)
    l1c_data = l1c_file.to_xarray_dataset()

    LOGGER.info("Adding ERA5 precip for file %s.", sim_filename)
    start_time = data_pp["scan_time"].data[0]
    end_time = data_pp["scan_time"].data[-1]
    LOGGER.info("Loading ERA5 data: %s %s", start_time, end_time)
    era5_data = _load_era5_data(start_time, end_time, era5_path)
    _add_era5_precip(data_pp, l1c_data, era5_data)
    LOGGER.info("Added era5 precip.")
    apply_orographic_enhancement(data_pp)

    surface_type = data_pp.variables["surface_type"].data
    snow = (surface_type >= 8) * (surface_type <= 11)
    for v in data_pp.variables:
        if v in ["surface_precip", "convective_precip"]:
            data_pp[v].data[snow] = np.nan

    return _extract_scenes(data_pp)


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
    import gprof_nn.logging
    LOGGER.info("Starting processing MRMS file %s.", mrms_filename)
    mrms_file = MRMSMatchFile(mrms_filename)

    indices = np.where(mrms_file.data["scan_time"][:, 2] == day)[0]
    if len(indices) <= 0:
        return None
    date = mrms_file.scan_time[indices[len(indices) // 2]]
    l1c_files = list(L1CFile.find_files(date, CONUS, l1c_path))

    scenes = []
    LOGGER.info("Found %s L1C file for MRMS file %s.",
                len(l1c_files),
                mrms_filename)
    for f in l1c_files:
        data_pp = run_preprocessor(f.filename)
        if data_pp is None:
            continue

        LOGGER.info("Matching MRMS data for %s.",
                    f.filename)
        mrms_file.match_targets(data_pp)
        surface_type = data_pp["surface_type"]
        snow = (surface_type >= 8) * (surface_type <= 11)
        if snow.sum() <= 0:
            continue
        # Keep only obs over snow.
        data_pp["surface_precip"].data[~snow] = np.nan
        data_pp["convective_precip"].data[~snow] = np.nan
        apply_orographic_enhancement(data_pp)

        new_scenes = _extract_scenes(data_pp)
        if new_scenes is not None:
            scenes.append(_extract_scenes(data_pp))

    if scenes:
        dataset = xr.concat(scenes, "samples")
        return add_targets(dataset)

    return None

def add_targets(data):
    """
    Helper function to ensure all target variables are present in
    dataset.
    """
    n_scans = data.scans.size
    n_pixels = data.pixels.size
    n_levels = 28

    for t in ALL_TARGETS:
        if not t in data.variables:
            if "content" in t or t == "latent_heat":
                d = np.zeros((n_scans, N_PIXELS_CENTER, n_levels),
                             dtype=np.float32)
                d[:] = np.nan
                data[t] = (("scans", "pixels_center", "levels"), d)
            else:
                if t in ["surface_precip", "convective_precip"]:
                    d = np.zeros((n_scans, n_pixels),
                                 dtype=np.float32)
                    d[:] = np.nan
                    data[t] = (("scans", "pixels"), d)
                else:
                    d = np.zeros((n_scans, N_PIXELS_CENTER),
                                 dtype=np.float32)
                    d[:] = np.nan
                    data[t] = (("scans", "pixels_center"), d)
    return data

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
                "The 'l1c_path' argument must be provided in order to process "
                "any sim files."
            )
        self.l1c_path = Path(l1c_path)

        if era5_path is None:
            raise ValueError(
                "The 'era5_path' argument must be provided in order to process"
                " any sim files."
            )
        self.era5_path = Path(era5_path)
        self.pool = ThreadPoolExecutor(max_workers=n_workers)

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
            sim_files = SimFile.find_files(self.sim_file_path, day=self.day)
        else:
            sim_files = []
        sim_files = np.random.permutation(sim_files)
        if self.mrms_path is not None:
            mrms_files = MRMSMatchFile.find_files(self.mrms_path)
        else:
            mrms_files = []
        mrms_files = np.random.permutation(mrms_files)

        n_sim_files = len(sim_files)
        print(f"Found {n_sim_files} .sim files.")
        n_mrms_files = len(mrms_files)
        print(f"Found {n_mrms_files} MRMS files.")
        i = 0

        # Submit tasks interleaving .sim and MRMS files.
        tasks = []
        while i < max(n_sim_files, n_mrms_files):
            if i < n_sim_files:
                sim_file = sim_files[i]
                tasks.append(self.pool.submit(process_sim_file,
                                              sim_file,
                                              self.l1c_path,
                                              self.era5_path))
            if i < n_mrms_files:
                mrms_file = mrms_files[i]
                tasks.append(self.pool.submit(process_mrms_file,
                                              mrms_file,
                                              self.day,
                                              self.l1c_path))
            i += 1

        n_datasets = len(tasks)
        n_chunks = 1
        chunk_size = n_datasets // n_chunks + 1
        datasets = []
        output_path = Path(self.output_file).parent
        output_file = Path(self.output_file).stem
        dataset_index = 0
        chunk_index = 0

        # Retrieve files and store them into 4 chunks.
        for t in track(tasks, description="Extracting data ..."):
            try:
                dataset = t.result()
            except Exception as e:
                LOGGER.warning(
                    "The follow error was encountered while collecting "
                    " results: %s", e
                )
                console.print_exception()
                dataset = None
            if dataset is not None:
                datasets.append(dataset)

            if len(datasets) >= chunk_size:
                chunk_index += 1
                filename = output_path / (output_file + f"_{dataset_index:02}.nc")
                print(f"Concatenating data file: {filename}")
                dataset = xr.concat(datasets, "samples")
                print(f"Writing file: {filename}")
                dataset.to_netcdf(filename)
                if chunk_index == (n_datasets % n_chunks):
                    chunk_size -= 1
                dataset_index += 1
                datasets = []

        dataset = xr.concat(datasets, "samples")
        filename = output_path / (output_file + f"_{dataset_index:02}.nc")
        print(f"Writing file: {filename}")
        dataset.to_netcdf(filename)
