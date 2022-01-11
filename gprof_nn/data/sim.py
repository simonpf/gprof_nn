"""
=================
gprof_nn.data.sim
=================

This module defines a class to read the simulator output files (*.sim) that
contain the atmospheric profiles and corresponding simulated brightness
temperatures.

The module also provides functionality to extract the training data for the
GPROF-NN algorithm from these files.
"""
from concurrent import futures
from datetime import datetime
import logging
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
from pykdtree.kdtree import KDTree
from rich.progress import Progress
import xarray as xr

import gprof_nn
from gprof_nn import sensors
from gprof_nn.definitions import N_LAYERS
from gprof_nn.definitions import (
    ALL_TARGETS,
    LEVELS,
    DATABASE_MONTHS,
    PROFILE_NAMES,
)

from gprof_nn.coordinates import latlon_to_ecef
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.mrms import MRMSMatchFile
from gprof_nn.data.preprocessor import run_preprocessor
from gprof_nn.data.utils import compressed_pixel_range, N_PIXELS_CENTER
from gprof_nn.logging import get_console
from gprof_nn.utils import CONUS


LOGGER = logging.getLogger(__name__)


###############################################################################
# Data types
###############################################################################

N_CHANS_MAX = 15
GENERIC_HEADER = np.dtype(
    [
        ("satellite_code", "a5"),
        ("sensor", "a5"),
    ]
)

###############################################################################
# GPROF GMI Simulation files
###############################################################################


class SimFile:
    """
    Interface class to read GPROF .sim files.

    The main purpose of this class is to provide and interface to read
    .sim files and convert them to 'xarray.Dataset' objects via the
    'to_xarray_dataset' method.

    Attributes:
        path: The path of the file
        granule: The GPM CO granule number to which this file corresponds.
        date: Date object specifying the day of the corresponding GPM orbit.
        sensor: Sensor object representing the sensor corresponding to
            this file.
        header: Numpy structured array containing the header data of the
            file.
        data: Numpy structured array containing raw data of the file.
    """

    @classmethod
    def find_files(cls, path, sensor=sensors.GMI, day=None):
        """
        Find all files that match the standard filename pattern for
        sim files for the given sensor.

        Args:
            path: Root of the directory tree in which to look for .sim
                files.
            sensor: The sensor for which to find .sim files.
            day: If given search is restricted to the given day within
                each month.

        Return:
            A list containing the found .sim files.
        """
        if day is None:
            pattern = sensor.sim_file_pattern.format(day="??")
        else:
            pattern = sensor.sim_file_pattern.format(day=f"{day:02}")
        path = Path(path)
        files = list(path.glob("**/????/" + pattern))
        if not files:
            files = list(path.glob("**/" + pattern))
        return files

    def __init__(self, path):
        """
        Open a .sim file.

        Args:
            path: Path to the .sim file to open.
        """
        self.path = path
        parts = str(path).split(".")
        self.granule = int(parts[-2])

        year = int(parts[-3][:4])
        month = int(parts[-3][4:6])
        day = int(parts[-3][6:])
        self.date = datetime(year, month, day)

        header = np.fromfile(self.path, GENERIC_HEADER, count=1)
        sensor = header["sensor"][0].decode().strip()
        try:
            sensor = getattr(sensors, sensor.upper())
        except AttributeError:
            raise ValueError(f"The sensor {sensor} isn't currently supported.")
        self.sensor = sensor
        self.header = np.fromfile(self.path, self.sensor.sim_file_header, count=1)
        offset = self.sensor.sim_file_header.itemsize
        self.data = np.fromfile(self.path, sensor.sim_file_record, offset=offset)

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
        for var in path_variables:
            profile_variable = var.replace("path", "content").replace("ice", "snow")
            if profile_variable not in targets:
                targets.append(profile_variable)
        targets = [t for t in targets if "path" not in t]

        n_scans = input_data.scans.size
        n_pixels = 221
        w_c = 40
        i_c = 110
        ix_start = i_c - w_c // 2
        ix_end = i_c + 1 + w_c // 2
        i_left, i_right = compressed_pixel_range()

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
        if self.sensor.n_angles > 1:
            n_angles = self.sensor.n_angles
        n_chans = self.sensor.n_chans

        if "tbs_simulated" in self.data.dtype.fields:
            if n_angles > 0:
                shape = (n_scans, w_c + 1, n_angles, n_chans)
                full_shape = (n_scans, n_pixels, n_angles, n_chans)
                matched = np.zeros((n_scans * (w_c + 1), n_angles, n_chans))
                dims = ("scans", "pixels_center", "angles", "channels")
            else:
                shape = (n_scans, w_c + 1, n_chans)
                full_shape = (n_scans, n_pixels, n_chans)
                matched = np.zeros((n_scans * (w_c + 1), n_chans))
                dims = ("scans", "pixels_center", "channels")
            matched[:] = np.nan
            assert np.all(indices[dists < 10e3] < matched.shape[0])
            indices = np.clip(indices, 0, matched.shape[0] - 1)

            tbs = self.data["tbs_simulated"]
            if isinstance(self.sensor, sensors.ConstellationScanner):
                tbs = tbs[..., self.sensor.gmi_channels]
            # tbs = tbs.reshape((-1,) + shape[2:])

            matched[indices, ...] = tbs
            matched[indices, ...][dists > 10e3] = np.nan
            matched = matched.reshape(shape)

            matched_full = np.zeros(full_shape, dtype=np.float32)
            matched_full[:] = np.nan
            matched_full[:, ix_start:ix_end] = matched

            input_data["simulated_brightness_temperatures"] = (
                dims,
                matched_full[:, i_left:i_right],
            )

        if "tbs_bias" in self.data.dtype.fields:
            shape = (n_scans, w_c + 1, n_chans)
            full_shape = (n_scans, n_pixels, n_chans)
            matched = np.zeros((n_scans * (w_c + 1), n_chans))

            matched[:] = np.nan

            biases = self.data["tbs_bias"]
            if isinstance(self.sensor, sensors.ConstellationScanner):
                biases = biases[..., self.sensor.gmi_channels]
            matched[indices, ...] = biases

            matched[indices, ...][dists > 10e3] = np.nan
            matched = matched.reshape(shape)

            matched_full = np.zeros(full_shape, dtype=np.float32)
            matched_full[:] = np.nan
            matched_full[:, ix_start:ix_end] = matched

            input_data["brightness_temperature_biases"] = (
                ("scans", "pixels_center", "channels"),
                matched_full[:, i_left:i_right],
            )

        # Extract matching data
        for target in targets:
            if target in PROFILE_NAMES:
                n = n_scans * (w_c + 1)
                shape = (n_scans, w_c + 1, 28)
                full_shape = (n_scans, n_pixels, 28)
                matched = np.zeros((n, 28), dtype=np.float32)
            else:
                n = n_scans * (w_c + 1)
                if n_angles > 0:
                    shape = (n_scans, w_c + 1, n_angles)
                    full_shape = (n_scans, n_pixels, n_angles)
                    matched = np.zeros((n, n_angles), dtype=np.float32)
                else:
                    shape = (n_scans, w_c + 1)
                    full_shape = (n_scans, n_pixels)
                    matched = np.zeros(n, dtype=np.float32)

            matched[:] = np.nan
            matched[indices, ...] = self.data[target]
            matched[indices, ...][dists > 5e3] = np.nan
            matched = matched.reshape(shape)

            matched_full = np.zeros(full_shape, dtype=np.float32)
            matched_full[:] = np.nan
            matched_full[:, ix_start:ix_end] = matched

            if target in PROFILE_NAMES:
                data = matched_full[:, i_left:i_right]
                input_data[target] = (
                    ("scans", "pixels_center", "levels"),
                    data.astype(np.float32),
                )
                if "content" in target:
                    path = np.trapz(data, x=LEVELS, axis=-1) * 1e-3
                    path_name = target.replace("content", "path").replace("snow", "ice")
                    input_data[path_name] = (("scans", "pixels_center"), path)
            else:
                if target in ["surface_precip", "convective_precip"]:
                    dims = ("scans", "pixels")
                    if n_angles > 0:
                        dims = dims + ("angles",)
                    input_data[target] = (dims, matched_full.astype(np.float32))
                else:
                    input_data[target] = (
                        ("scans", "pixels_center"),
                        matched_full[:, i_left:i_right].astype(np.float32),
                    )

        return input_data

    def to_xarray_dataset(self):
        """
        Return data in sim file as 'xarray.Dataset'.
        """
        results = {}
        dim_dict = {
            self.sensor.n_chans: "channels",
            N_LAYERS: "layers",
        }
        if isinstance(self.sensor, sensors.CrossTrackScanner):
            dim_dict[self.sensor.n_angles] = "angles"

        record_type = self.sensor.sim_file_record
        for key, _, *shape in record_type.descr:

            data = self.data[key]
            if key in [
                "emissivity",
                "tbs_observed",
                "tbs_simulated",
                "tbs_bias",
                "d_tbs",
            ]:
                if isinstance(self.sensor, sensors.ConstellationScanner):
                    data = data[..., self.sensor.gmi_channels]

            dims = ("samples",)
            if len(data.shape) > 1:
                dims = dims + tuple([dim_dict[s] for s in data.shape[1:]])

            results[key] = dims, self.data[key]

        dataset = xr.Dataset(results)
        return dataset


ENHANCEMENT_FACTORS = {
    "ERA5": {
        (17, 0): 1.35683,
        (17, 1): 2.05213,
        (17, 2): 1.62242,
        (17, 3): 1.87049,
        (18, 0): 3.91369,
    },
    "GANAL": {
        (17, 0): 1.58177,
        (17, 1): 1.81539,
        (18, 0): 3.91369,
    },
}


def apply_orographic_enhancement(data, kind="ERA5"):
    """
    Applies orographic enhancement factors to 'surface_precip' and
    'convective_precip' targets.

    Args:
        data: xarray.Dataset containing variables surface_precip,
            convective_precip, surface_type and airmass_type.
         kind: "ERA5" or "GANAL" depending on the source of ancillary data.

    Returns:
        None; Correction is applied in place.
    """
    kind = kind.upper()
    if kind not in ["ERA5", "GANAL"]:
        raise ValueError("The kind argument to  must be 'ERA5' or 'GANAL'.")
    surface_types = data["surface_type"].data
    airmass_types = data["airmass_type"].data
    surface_precip = data["surface_precip"].data
    convective_precip = data["convective_precip"].data

    enh = np.ones(surface_precip.shape, dtype=np.float32)
    factors = ENHANCEMENT_FACTORS[kind]
    for t_s in [17, 18]:
        for t_a in range(4):
            key = (t_s, t_a)
            if key not in factors:
                continue
            indices = (surface_types == t_s) * (airmass_types == t_a)
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
    surface_precip = data["surface_precip"].data

    if np.all(np.isnan(surface_precip)):
        return None

    i_start = 0
    i_end = data.scans.size

    scenes = []
    while i_start + n < i_end:
        subscene = data[{"scans": slice(i_start, i_start + n)}]
        surface_precip = subscene["surface_precip"].data
        if np.isfinite(surface_precip).sum() > 100:
            scenes.append(subscene)
            i_start += n
        else:
            i_start += n // 2

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
    year = sim_file.date.year - 2000
    month = sim_file.date.month
    day = sim_file.date.day
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
    l_0 = era5_data[{"longitude": slice(0, 1)}].copy(deep=True)
    l_0 = l_0.assign_coords({"longitude": [360.0]})
    era5_data = xr.concat([era5_data, l_0], "longitude")
    n_scans = l1c_data.scans.size
    n_pixels = l1c_data.pixels.size

    surface_types = input_data["surface_type"].data
    indices = (surface_types == 2) + (surface_types == 16)

    lats = xr.DataArray(input_data["latitude"].data[indices], dims="samples")
    lons = input_data["longitude"].data[indices]
    lons = np.where(lons < 0.0, lons + 360, lons)
    lons = xr.DataArray(lons, dims="samples")
    time = np.broadcast_to(
        l1c_data["scan_time"].data.reshape(-1, 1), (n_scans, n_pixels)
    )[indices]
    time = xr.DataArray(time, dims="samples")
    # Interpolate and convert to mm/h
    total_precip = era5_data["tp"].interp(
        {"latitude": lats, "longitude": lons, "time": time}, method="nearest"
    )
    if len(input_data.surface_precip.dims) > 2:
        total_precip = total_precip.data[..., np.newaxis]
    else:
        total_precip = total_precip.data
    input_data["surface_precip"].data[indices] = 1000.0 * total_precip

    convective_precip = era5_data["cp"].interp(
        {"latitude": lats, "longitude": lons, "time": time}, method="nearest"
    )
    if len(input_data.surface_precip.dims) > 2:
        convective_precip = convective_precip.data[..., np.newaxis]
    else:
        convective_precip = convective_precip.data
    input_data["convective_precip"].data[indices] = 1000.0 * convective_precip


def process_sim_file(sim_filename, sensor, configuration, era5_path, log_queue=None):
    """
    Extract 2D training scenes from sim file.

    This method reads a given sim file, matches it with the input data
    from the GMI L1C file and preprocessor files and reshapes the
    data to scenes of dimensions 221 x 221.

    For the GMI sensor also ERA5 precip over sea ice and sea ice edge are
    added to the surface precipitation.

    Surface and convective precipitation over mountains is set to NAN.
    This is also done for precipitation over sea ice for sensors that
    are not GMI.

    Args:
        sim_filename: Filename of the Sim file to process.
        sensor: The sensor for which the training data is extracted.
        l1c_path: Base path of the directory tree containing the L1C file.
        era5_path: Base path of the directory containing the ERA5 data.
        log_queue: Optional queue object to use for multi-process logging.

    Return:
        xarray.Dataset containing the data from the sim file as multiple
        2D training scenes.
    """
    import gprof_nn.logging

    if log_queue is not None:
        gprof_nn.logging.configure_queue_logging(log_queue)
    LOGGER.info("Processing sim file %s.", sim_filename)

    # Load sim file and corresponding GMI L1C file.
    sim_file = SimFile(sim_filename)
    l1c_file = L1CFile.open_granule(
        sim_file.granule, sensors.GMI.l1c_file_path, sensors.GMI
    )

    LOGGER.info("Running preprocessor for sim file %s.", sim_filename)
    data_pp = run_preprocessor(
        l1c_file.filename, sensor=sensor, configuration=configuration, robust=False
    )
    if data_pp is None:
        return None

    # If not dealing with GMI, brightness_temperature are only by-product
    # and therefore renamed.
    sensor = sim_file.sensor
    if sensor != sensors.GMI:
        data_pp = data_pp.rename(
            {
                "channels": "channels_gmi",
                "brightness_temperatures": "brightness_temperatures_gmi",
            }
        )

    # Match targets from sim file to preprocessor data.
    LOGGER.debug("Matching retrieval targets for file %s.", sim_filename)
    sim_file.match_targets(data_pp)
    l1c_data = l1c_file.to_xarray_dataset()

    # Orographic enhancement for types 17 and 18.
    apply_orographic_enhancement(data_pp)

    # Set surface_precip and convective_precip over snow surfaces to missing
    # since these are handled separately.
    surface_type = data_pp.variables["surface_type"].data
    snow = (surface_type >= 8) * (surface_type <= 11)
    for var in data_pp.variables:
        if var in ["surface_precip", "convective_precip"]:
            data_pp[var].data[snow] = np.nan

    # If we are dealing with GMI add precip from ERA5.
    if sensor == sensors.GMI:
        LOGGER.debug("Adding ERA5 precip for file %s.", sim_filename)
        start_time = data_pp["scan_time"].data[0]
        end_time = data_pp["scan_time"].data[-1]
        LOGGER.debug("Loading ERA5 data: %s %s", start_time, end_time)
        era5_data = _load_era5_data(start_time, end_time, era5_path)
        _add_era5_precip(data_pp, l1c_data, era5_data)
        LOGGER.debug("Added era5 precip.")
    # Else set to missing.
    else:
        sea_ice = (surface_type == 2) + (surface_type == 16)
        for var in ["surface_precip", "convective_precip"]:
            data_pp[var].data[sea_ice] = np.nan

    # Organize into scenes.
    data = _extract_scenes(data_pp)

    # Add source indicator.
    data["source"] = ("samples", np.zeros(data.samples.size, dtype=np.int8))
    return data


def process_mrms_file(sensor, mrms_filename, configuration, day, log_queue=None):
    """
    Extract training data from MRMS-GMI match up files for given day.
    Matches the observations in the MRMS file with input data from the
    preprocessor, extracts observations over snow and returns a dataset
    of 2D training scenes for the GPROF-NN-2D algorithm.

    Args:
        mrms_filename: Filename of the MRMS file to process.
        day: The day of the month for which to extract data.
    """
    import gprof_nn.logging

    if log_queue is not None:
        gprof_nn.logging.configure_queue_logging(log_queue)
    LOGGER.info("Processing MRMS file %s.", mrms_filename)
    mrms_file = MRMSMatchFile(mrms_filename)
    mrms_sensor = mrms_file.sensor

    indices = np.where(mrms_file.data["scan_time"][:, 2] == day)[0]
    if len(indices) <= 0:
        return None
    date = mrms_file.scan_time[indices[len(indices) // 2]]
    l1c_files = list(
        L1CFile.find_files(
            date, mrms_sensor.l1c_file_path, roi=CONUS, sensor=mrms_sensor
        )
    )

    scenes = []
    LOGGER.debug("Found %s L1C file for MRMS file %s.", len(l1c_files), mrms_filename)
    for file in l1c_files:
        # Extract scans over CONUS ans run preprocessor.
        _, f_roi = tempfile.mkstemp()
        try:
            file.extract_scans(CONUS, f_roi)
            data_pp = run_preprocessor(
                f_roi, configuration=configuration, sensor=sensor, robust=False
            )
        finally:
            Path(f_roi).unlink()
        if data_pp is None:
            continue
        if data_pp.channels.size > sensor.n_chans:
            data_pp = data_pp[{"channels": sensor.gmi_channels}]

        LOGGER.debug("Matching MRMS data for %s.", file.filename)
        mrms_file.match_targets(data_pp)
        surface_type = data_pp["surface_type"].data
        snow = (surface_type >= 8) * (surface_type <= 11)
        if snow.sum() <= 0:
            continue

        # Keep only obs over snow.
        data_pp["surface_precip"].data[~snow] = np.nan
        data_pp["convective_precip"].data[~snow] = np.nan
        apply_orographic_enhancement(data_pp)

        add_targets(data_pp, sensor)
        new_scenes = _extract_scenes(data_pp)
        if new_scenes is not None:
            scenes.append(new_scenes)

    if scenes:
        dataset = xr.concat(scenes, "samples")
        dataset["source"] = (("samples",), np.ones(dataset.samples.size, dtype=np.int8))
        dataset = extend_pixels(dataset)
        return dataset

    return None


def process_l1c_file(l1c_filename, sensor, configuration, era5_path, log_queue=None):
    """
    Match L1C files with ERA5 surface and convective precipitation for
    sea-ice and sea-ice-edge surfaces.

    Args:
        l1c_filename: Path to a L1C file which to match with ERA5 precip.
        sensor: Sensor class defining the sensor for which to process the
            L1C files.
        era5_path: Root of the directory tree containing the ERA5 data.
    """
    import gprof_nn.logging

    if log_queue is not None:
        gprof_nn.logging.configure_queue_logging(log_queue)
    LOGGER.info("Starting processing L1C file %s.", l1c_filename)

    data_pp = run_preprocessor(
        l1c_filename, sensor=sensor, configuration=configuration, robust=False
    )

    l1c_file = L1CFile(l1c_filename)
    if l1c_file.sensor != sensor:
        data_pp = data_pp[{"channels": sensor.gmi_channels}]

    if data_pp is None:
        return None
    data_pp = add_targets(data_pp, sensor)
    l1c_data = L1CFile(l1c_filename).to_xarray_dataset()

    start_time = data_pp["scan_time"].data[0]
    end_time = data_pp["scan_time"].data[-1]
    era5_data = _load_era5_data(start_time, end_time, era5_path)
    _add_era5_precip(data_pp, l1c_data, era5_data)
    apply_orographic_enhancement(data_pp)

    surface_type = data_pp["surface_type"].data
    sea_ice = (surface_type == 2) + (surface_type == 16)
    for v in ["surface_precip", "convective_precip"]:
        data_pp[v].data[~sea_ice] = np.nan

    if data_pp.channels.size > sensor.n_chans:
        data_pp = data_pp[{"channels": sensor.gmi_channels}]

    scenes = _extract_scenes(data_pp)
    scenes["source"] = (("samples",), 2 * np.ones(scenes.samples.size, dtype=np.int8))
    if scenes is not None:
        scenes = extend_pixels(scenes)
    return scenes


def extend_pixels(data, n_pixels=221):
    """
    Extends 'pixels' dimension of dataset to 'n_pixels'.

    Args:
        data: The 'xarray.Dataset' which should be extened to match
            the given size 'n_pixels' along the 'pixels' dimension.
        n_pixels: The desired extent of the 'pixels' dimension.

    Return:
        A new xarray dataset containing the variables of 'data' but with
        the 'pixels' dimensions extend to the given size.
    """
    if "pixels" in data.dims and data.pixels.size == 221:
        return data
    dimensions = dict(data.dims)
    dimensions["pixels"] = n_pixels
    data_new = dict(data.dims)
    data_new["pixels"] = np.arange(n_pixels)

    data_new = {}
    for name, var in data.variables.items():
        shape = tuple(dimensions[d] for d in var.dims)
        dims = var.dims
        var_new = np.zeros(shape, var.dtype)
        if var.dtype in [np.float32, np.float64]:
            var_new[:] = np.nan
        else:
            var_new[:] = -1
        data_new[name] = (dims, var_new)

    left = (n_pixels - data.pixels.size) // 2
    right = n_pixels - data.pixels.size - left

    data_new = xr.Dataset(data_new)
    data_new_sub = data_new[{"pixels": slice(left, -right)}]
    for n in data.variables:
        data_new_sub[n].data[:] = data[n].data[:]

    return data_new


def add_targets(dataset, sensor):
    """
    Helper function to ensure all target variables are present in
    dataset.
    """
    n_scans = dataset.scans.size
    n_pixels = dataset.pixels.size
    n_pixels_center = 41
    n_levels = 28

    for target in ALL_TARGETS:
        if target not in dataset.variables:
            if "content" in target or target == "latent_heat":
                data = np.zeros((n_scans, N_PIXELS_CENTER, n_levels), dtype=np.float32)
                data[:] = np.nan
                dataset[target] = (("scans", "pixels_center", "levels"), data)
            else:
                if target in ["surface_precip", "convective_precip"]:
                    data = np.zeros((n_scans, n_pixels), dtype=np.float32)
                    data[:] = np.nan
                    dataset[target] = (("scans", "pixels"), data)
                else:
                    data = np.zeros((n_scans, N_PIXELS_CENTER), dtype=np.float32)
                    data[:] = np.nan
                    dataset[target] = (("scans", "pixels_center"), data)

    if sensor != sensors.GMI:
        data = np.zeros((n_scans, n_pixels, 15), dtype=np.float32)
        data[:] = np.nan
        dataset["brightness_temperatures_gmi"] = (
            ("scans", "pixels", "channels_gmi"),
            data,
        )
    if sensor.n_angles > 1:
        n_angles = sensor.n_angles

        shape = (n_scans, n_pixels_center, n_angles, sensor.n_chans)
        data = np.zeros(shape, dtype=np.float32)
        data[:] = np.nan
        dataset["simulated_brightness_temperatures"] = (
            ("scans", "pixels_center", "angles", "channels"),
            data,
        )

        for var in ["surface_precip", "convective_precip"]:
            values = dataset[var].data
            new_shape = (n_scans, n_pixels, n_angles)
            values = np.broadcast_to(values[..., np.newaxis], new_shape)
            dataset[var] = (("scans", "pixels", "angles"), values.copy())

    else:
        shape = (n_scans, n_pixels_center, sensor.n_chans)
        data = np.zeros(shape, dtype=np.float32)
        data[:] = np.nan
        dataset["simulated_brightness_temperatures"] = (
            ("scans", "pixels_center", "channels"),
            data,
        )

    shape = (n_scans, n_pixels_center, sensor.n_chans)
    data = np.zeros(shape, dtype=np.float32)
    data[:] = np.nan
    dataset["brightness_temperature_biases"] = (
        ("scans", "pixels_center", "channels"),
        data,
    )
    return dataset


def add_brightness_temperatures(data, sensor):
    """
    Add brightness temperatures variables to dataset.

    Simulated observations from *.sim files for sensors other than GMI lack
    the 'brightness_temperature' variable. This function adds these as empty
    variables to enable merging with MRMS- and L1C-derived datasets.

    Args:
        data: 'xarray.Dataset' containing the matched data from the *.sim file.
        sensor: Sensor object representing the sensor for which the data is
            extracted.

    Return:
        The 'xarray.Dataset' with the added 'brighness_temperatures' variable.
    """
    if "brightness_temperatures" in data.variables.keys():
        return data
    n_samples = data.samples.size
    n_scans = data.scans.size
    n_pixels = data.pixels.size

    n_channels = sensor.n_chans
    shape = (n_samples, n_scans, n_pixels, n_channels)
    bts = np.zeros(shape, dtype=np.float32)
    bts[:] = np.nan
    data["brightness_temperatures"] = (("samples", "scans", "pixels", "channels"), bts)
    return data


###############################################################################
# File processor
###############################################################################


class SimFileProcessor:
    """
    Processor class that manages the extraction of GPROF training data. A
    single processor instance processes all *.sim, MRMRS matchup and L1C
    files for a given day from each month of the database period.
    """

    def __init__(
        self,
        output_file,
        sensor,
        configuration,
        era5_path=None,
        n_workers=4,
        day=None,
    ):
        """
        Create retrieval driver.

        Args:
            output_file: The file in which to store the extracted data.
            sensor: Sensor object defining the sensor for which to extract
                training data.
            era_5_path: Path to the root of the directory tree containing
                ERA5 data.
            n_workers: The number of worker processes to use.
            day: Day of the month for which to extract the data.
        """

        self.output_file = output_file
        self.sensor = sensor
        self.configuration = configuration

        if era5_path is None:
            raise ValueError(
                "The 'era5_path' argument must be provided in order to process"
                " any sim files."
            )
        self.era5_path = Path(era5_path)
        self.pool = futures.ProcessPoolExecutor(max_workers=n_workers)

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
        sim_file_path = self.sensor.sim_file_path
        sim_files = SimFile.find_files(sim_file_path, sensor=self.sensor, day=self.day)
        sim_files = np.random.permutation(sim_files)

        mrms_file_path = self.sensor.mrms_file_path
        if mrms_file_path is None:
            mrms_files = MRMSMatchFile.find_files(
                sensors.GMI.mrms_file_path, sensor=sensors.GMI
            )
        else:
            mrms_files = MRMSMatchFile.find_files(mrms_file_path, sensor=self.sensor)
        mrms_files = np.random.permutation(mrms_files)

        l1c_file_path = self.sensor.l1c_file_path
        l1c_files = []
        for year, month in DATABASE_MONTHS:
            try:
                date = datetime(year, month, self.day)
                l1c_files += L1CFile.find_files(date, l1c_file_path, sensor=self.sensor)
            except ValueError:
                pass

        # If no L1C files are found use GMI co-locations.
        if len(l1c_files) < 1:
            for year, month in DATABASE_MONTHS:
                try:
                    date = datetime(year, month, self.day)
                    l1c_file_path = sensors.GMI.l1c_file_path
                    l1c_files += L1CFile.find_files(
                        date, l1c_file_path, sensor=sensors.GMI
                    )
                except ValueError:
                    pass
        l1c_files = [f.filename for f in l1c_files]
        l1c_files = np.random.permutation(l1c_files)

        n_sim_files = len(sim_files)
        LOGGER.debug("Found %s SIM files.", n_sim_files)
        n_mrms_files = len(mrms_files)
        LOGGER.debug("Found %s MRMS files.", n_mrms_files)
        n_l1c_files = len(l1c_files)
        LOGGER.debug("Found %s L1C files.", n_l1c_files)
        i = 0

        # Submit tasks interleaving .sim and MRMS files.
        log_queue = gprof_nn.logging.get_log_queue()
        tasks = []
        while i < max(n_sim_files, n_mrms_files, n_l1c_files):
            if i < n_sim_files:
                sim_file = sim_files[i]
                tasks.append(
                    self.pool.submit(
                        process_sim_file,
                        sim_file,
                        self.sensor,
                        self.configuration,
                        self.era5_path,
                        log_queue=log_queue,
                    )
                )
            if i < n_mrms_files:
                mrms_file = mrms_files[i]
                tasks.append(
                    self.pool.submit(
                        process_mrms_file,
                        self.sensor,
                        mrms_file,
                        self.configuration,
                        self.day,
                        log_queue=log_queue,
                    )
                )
            if i < n_l1c_files:
                l1c_file = l1c_files[i]
                tasks.append(
                    self.pool.submit(
                        process_l1c_file,
                        l1c_file,
                        self.sensor,
                        self.configuration,
                        self.era5_path,
                        log_queue=log_queue,
                    )
                )
            i += 1

        datasets = []
        output_path = Path(self.output_file).parent
        output_file = Path(self.output_file).stem

        # Retrieve extracted observations and concatenate into
        # single dataset.

        n_tasks = len(tasks)
        n_chunks = 4
        chunk = 1

        with Progress(console=get_console()) as progress:
            gprof_nn.logging.set_log_level("INFO")
            pbar = progress.add_task("Extracting data:", total=len(tasks))
            for task in tasks:
                # Log messages from processes.
                task_done = False
                dataset = None
                while not task_done:
                    try:
                        gprof_nn.logging.log_messages()
                        dataset = task.result(timeout=1)
                        task_done = True
                    except futures.TimeoutError:
                        pass
                    except Exception as exc:
                        LOGGER.warning(
                            "The following error was encountered while "
                            "collecting results: %s",
                            exc,
                        )
                        get_console().print_exception()
                        task_done = True
                progress.advance(pbar)

                if dataset is not None:
                    dataset = add_brightness_temperatures(dataset, self.sensor)
                    datasets.append(dataset)
                    if len(datasets) > n_tasks // n_chunks:
                        dataset = xr.concat(datasets, "samples")
                        filename = output_path / (output_file + f"_{chunk:02}.nc")
                        dataset.attrs["sensor"] = self.sensor.name
                        dataset.to_netcdf(filename)
                        subprocess.run(["lz4", "-f", "--rm", filename], check=True)
                        LOGGER.info("Finished writing file: %s", filename)
                        datasets = []
                        chunk += 1

        # Store dataset with sensor name as attribute.
        dataset = xr.concat(datasets, "samples")
        filename = output_path / (output_file + f"_{chunk:02}.nc")
        dataset.attrs["sensor"] = self.sensor.name
        dataset.attrs["configuration"] = self.configuration
        LOGGER.info("Writing file: %s", filename)
        dataset.to_netcdf(filename)
        subprocess.run(["lz4", "-f", "--rm", filename], check=True)

        # Explicit clean up to avoid memory leak.
        del datasets
        del dataset
