"""
=================
gprof_nn.data.sim
=================

This module contains functions to read and convert .sim files for GPROF
 v. 7.
"""
from multiprocessing import Queue, Process, Manager
from pathlib import Path
import queue
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pyproj
from pykdtree.kdtree import KDTree
from netCDF4 import Dataset
from tqdm import tqdm
import xarray as xr

from gprof_nn.coordinates import latlon_to_ecef
from gprof_nn.data.preprocessor import PreprocessorFile
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.training_data import PROFILE_NAMES

ALL_TARGETS = [
    "surface_precip",
    "convective_precip",
    "cloud_water_content",
    "rain_water_content",
    "snow_water_content",
    "latent_heat"
]

LEVELS = np.concatenate([np.linspace(500.0, 1e4, 20),
                         np.linspace(11e3, 18e3, 8)])

###############################################################################
# Data types
###############################################################################

N_LAYERS = 28
N_FREQS = 15
DATE_TYPE = np.dtype(
    [("year", "i4"),
     ("month", "i4"),
     ("day", "i4"),
     ("hour", "i4"),
     ("minute", "i4"),
     ("second", "i4")]
)

GMI_HEADER_TYPES = np.dtype(
    [("satellite_code", "a5"),
     ("sensor", "a5"),
     ("frequencies", f"{N_FREQS}f4"),
     ("nominal_eia", f"{N_FREQS}f4"),
     ("start_pixel", "i4"),
     ("end_pixel", "i4"),
     ("start_scan", "i4"),
     ("end_scan", "i4"),
     ])

GMI_PIXEL_TYPES = np.dtype(
    [("pixel_index", "i4"),
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
     ("tbs_bias", f"{N_FREQS}f4")]
)

###############################################################################
# GPROF GMI Simulation files
###############################################################################

class GMISimFile:
    """
    Interface class to read GPROF .sim files.
    """
    def __init__(self, path):
        """
        Open .sim file.

        Args:
            path: Path to the .sim file to open.
        """
        self.path = path
        parts = str(path).split(".")
        self.granule = int(parts[-2])
        self.year  = int(parts[-3][:4])
        self.month = int(parts[-3][4:6])
        self.day = int(parts[-3][6:])

        self.header = np.fromfile(self.path,
                                  GMI_HEADER_TYPES,
                                  count=1)
        offset = GMI_HEADER_TYPES.itemsize
        self.data = np.fromfile(self.path,
                                GMI_PIXEL_TYPES,
                                offset=offset)

    def match_targets(self,
                      input_data,
                      targets=None):
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

            matched_full = np.zeros(full_shape,
                                    dtype=np.float32)
            matched_full[:] = np.nan
            matched_full[:, ix_start: ix_end] = matched

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
    Extract 128 x 128 pixel wide scenes from dataset where
    ground truth surface precipitation rain rates are
    available.

    Args:
        xarray.Dataset containing the data from the preprocessor together
        with the matches surface precipitation from the .sim file.

    Return:
        New xarray.Dataset which containing 128x128 patches of input data
        and corresponding surface precipitation.
    """

    n = 128

    sp = data["surface_precip"].data
    c_i = sp.shape[1] // 2
    i_start, i_end = np.where(np.any(~np.isnan(sp), axis=1))[0][[0, -1]]

    scenes = []
    i = i_start
    while i_start + n < i_end:
        scenes.append(data[{"scans": slice(i_start, i_start + 128),
                            "pixels": slice(c_i - 64, c_i + 64)}])
        i_start += n

    return xarray.concat(scenes, "samples")

def _run_preprocessor(sim_file,
                      l1c_file):
    """
    Run preprocessor on L1C GMI file.

    Args:
        sim_file: The GMISimFile object corresponding to the L1C file
             to process.
        l1c_file: Path of the L1C file for which to extract the input data
             using the preprocessor.

    Returns
        xarray.Dataset containing the retrieval input data for the given L1C file.
    """
    year = sim_file.year
    month = sim_file.month
    day = sim_file.day
    output_file = f"GMIERA5_{year:04}{month:02}{day:02}_{sim_file.granule:06}.pp"
    output_file = sim_file.path.parent / output_file


    if not output_file.exists():
        with NamedTemporaryFile(delete=False) as file:

            file.close()
            jobid = str(os.getpid()) + "_pp"
            prodtype = "CLIMATOLOGY"
            prepdir = "/qdata2/archive/ERA5"
            ancdir = "/qdata1/pbrown/gpm/ancillary"
            ingestdir = "/qdata1/pbrown/gpm/ppingest"

            subprocess.run(["gprof2020pp_GMI_L1C",
                            jobid,
                            prodtype,
                            str(l1c_file),
                            prepdir,
                            ancdir,
                            ingestdir,
                            str(file)])

            data = PreprocessorFile(file).to_xarray_dataset()
    else:
        data = PreprocessorFile(output_file).to_xarray_dataset()

    return data

def _write_results(output_file, data):
    """
    Write results to NetCDF4 file.

    If the file doesn't yet exist a new file will be created. If it
    does the data will be appended along the 'samples' dimension.

    Args:
        output_file: Path to the output file to store the data in.
        data: xarray.Dataset containing the data to store in the output file.
    """
    if not Path(output_file).exists():
        data.to_netcdf(path=output_file, unlimited_dims=["samples"])
    else:
        with Dataset(output_file, "a") as output_file:
            i = output_file.dimensions["samples"].size
            n = data["samples"].size
            for k in data:
                v = data[k]
                if v.dims.index("samples") == 0:
                    output_file.variables[k][i:i + n] = v.data


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

def _load_era5_data(start_time,
                    end_time,
                    base_directory):
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

def _add_era5_precip(input_data,
                     l1c_data,
                     era5_data):
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
    time = np.broadcast_to(l1c_data["scan_time"].data.reshape(-1, 1),
                           (n_scans, n_pixels))[indices]
    time = xr.DataArray(time, dims="samples")
    tp = era5_data["tp"].interp({"latitude": lats,
                                 "longitude": lons,
                                 "time": time},
                                method="linear")
    input_data["surface_precip"].data[indices] = tp.data

###############################################################################
# File processor
###############################################################################


class Worker(Process):
    """
    A worker process class for data processing of simulation files.
    """
    def __init__(self,
                 l1c_path,
                 output_file,
                 input_queue,
                 done_queue,
                 lock):
        """
        Create new worker.

        Args:
            output_path: The folder to which to write the retrieval results.
            input_queue: The queue from which the input files are taken
            done_queue: The queue onto which the processed files are placed.
            input_class: The class used to read and process the input data.
        """
        super().__init__()
        self.l1c_path = l1c_path
        self.output_file = output_file
        self.input_queue = input_queue
        self.done_queue = done_queue
        self.lock = lock

    def run(self):
        """
        Start the process.
        """
        while True:
            try:
                sim_file = GMISimFile(self.input_queue.get(False))

                l1c_file = _find_l1c_file(self.l1c_path, sim_file)

                data_pp = _run_preprocessor(sim_file, l1c_file)
                data = sim_file.match_surface_precip(data_pp)
                scenes = _extract_scenes(data)

                with self.lock:
                    _write_results(self.output_file, scenes)

                self.done_queue.put(sim_file)
            except queue.Empty:
                break

class SimFileProcessor:
    def __init__(self,
                 sim_file_path,
                 l1c_path,
                 output_file,
                 n_workers=4,
                 days=None):
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

        self.sim_file_path = Path(sim_file_path)
        self.l1c_path = Path(l1c_path)
        self.output_file = output_file
        self.done_queue = Queue()
        self.manager = Manager()
        self.lock = self.manager.Lock()

        self._fill_input_queue(days)

        self.workers = [Worker(self.l1c_path,
                               self.output_file,
                               self.input_queue,
                               self.done_queue,
                               self.lock) for i in range(n_workers)]
        self.processed = []

    def _fill_input_queue(self, days):
        """
        Scans the input folder for matching files and fills the input queue.
        """

        self.input_queue = Queue()

        if days is None:
            for f in self.sim_file_path.glob("**/*.sim"):
                print("putting on queue: ", f)
                self.input_queue.put(f)
        else:
            for d in days:
                for f in self.sim_file_path.glob(f"**/*{d:02}/*.sim"):
                    print("putting on queue: ", f)
                    self.input_queue.put(f)


    def run(self):
        """
        Start the processing.

        This will start processing all suitable input files that have been found and
        stores the names of the produced result files in the ``processed`` attribute
        of the driver.
        """
        if len(self.processed) > 0:
            print("This processor already ran.")
            return

        n_files = self.input_queue.qsize()
        [w.start() for w in self.workers]
        for i in tqdm(range(n_files)):
            self.processed.append(self.done_queue.get(True))
