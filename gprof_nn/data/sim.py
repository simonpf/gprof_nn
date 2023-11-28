"""
=================
gprof_nn.data.sim
=================

This module defines a class to read the output files of the GPROF
simulator (*.sim), which contain the atmospheric profiles and
corresponding simulated brightness temperatures.

The module also provides functionality to extract the training data
 for the GPROF-NN algorithm from these files.
"""
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import tempfile
from typing import Optional, Tuple

import click
import numpy as np
import pandas as pd
from pykdtree.kdtree import KDTree
from rich.progress import Progress
import xarray as xr

import gprof_nn
from gprof_nn import sensors
from gprof_nn.config import CONFIG
from gprof_nn.definitions import N_LAYERS
from gprof_nn.definitions import (
    ALL_TARGETS,
    LEVELS,
    DATABASE_MONTHS,
    PROFILE_NAMES,
    SEAICE_YEARS,
)

from gprof_nn.coordinates import latlon_to_ecef
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.era5 import load_era5_data, add_era5_precip
from gprof_nn.data.mrms import MRMSMatchFile
from gprof_nn.data.preprocessor import run_preprocessor
from gprof_nn.data.utils import (
    compressed_pixel_range,
    N_PIXELS_CENTER,
    save_scene
)
from gprof_nn.logging import get_console
from gprof_nn.utils import CONUS
from gprof_nn.sensors import Sensor


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


CHANNEL_INDICES = {
    "TMIPO": [0, 1, 2, 3, 4, 6, 7, 8, 9],
    "TMIPR": [0, 1, 2, 3, 4, 6, 7, 8, 9],
    "SSMI": [2, 3, 4, 6, 7, 8, 9],
    "SSMIS": [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14],
    "AMSR2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "AMSRE": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}


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
            day: Restricts the searth to given day of the month if
                given.

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
        self.granule = int(parts[-2][:6])

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
            input_data: ``xarray.Dataset`` containing the input data from
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
        i_left, i_right = compressed_pixel_range()
        cmpr = slice(i_left, i_right)
        w_c = i_right - i_left

        lats_1c = input_data["latitude"][:, cmpr].data.reshape(-1, 1)
        lons_1c = input_data["longitude"][:, cmpr].data.reshape(-1, 1)
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
                shape = (n_scans, w_c, n_angles, n_chans)
                full_shape = (n_scans, n_pixels, n_angles, n_chans)
                matched = np.zeros((n_scans * w_c, n_angles, n_chans))
                dims = ("scans", "pixels_center", "angles", "channels")
            else:
                shape = (n_scans, w_c, n_chans)
                full_shape = (n_scans, n_pixels, n_chans)
                matched = np.zeros((n_scans * w_c, n_chans))
                dims = ("scans", "pixels_center", "channels")
            matched[:] = np.nan
            assert np.all(indices[dists < 10e3] < matched.shape[0])
            indices = np.clip(indices, 0, matched.shape[0] - 1)

            tbs = self.data["tbs_simulated"]
            if self.sensor.sensor_name in CHANNEL_INDICES:
                ch_inds = CHANNEL_INDICES[self.sensor.sensor_name]
                tbs = tbs[..., ch_inds]
            # tbs = tbs.reshape((-1,) + shape[2:])

            matched[indices, ...] = tbs
            matched[indices, ...][dists > 10e3] = np.nan
            matched = matched.reshape(shape)

            matched_full = np.zeros(full_shape, dtype=np.float32)
            matched_full[:] = np.nan
            matched_full[:, cmpr] = matched

            input_data["simulated_brightness_temperatures"] = (
                dims,
                matched_full[:, cmpr],
            )

        if "tbs_bias" in self.data.dtype.fields:
            shape = (n_scans, w_c, n_chans)
            full_shape = (n_scans, n_pixels, n_chans)
            matched = np.zeros((n_scans * w_c, n_chans))

            matched[:] = np.nan

            biases = self.data["tbs_bias"]
            if self.sensor.sensor_name in CHANNEL_INDICES:
                ch_inds = CHANNEL_INDICES[self.sensor.sensor_name]
                biases = biases[..., ch_inds]
            matched[indices, ...] = biases

            matched[indices, ...][dists > 10e3] = np.nan
            matched = matched.reshape(shape)

            matched_full = np.zeros(full_shape, dtype=np.float32)
            matched_full[:] = np.nan
            matched_full[:, cmpr] = matched

            input_data["brightness_temperature_biases"] = (
                ("scans", "pixels_center", "channels"),
                matched_full[:, cmpr],
            )

        # Extract matching data
        for target in targets:
            if target in PROFILE_NAMES:
                n = n_scans * w_c
                shape = (n_scans, w_c, 28)
                full_shape = (n_scans, n_pixels, 28)
                matched = np.zeros((n, 28), dtype=np.float32)
            else:
                n = n_scans * w_c
                if n_angles > 0:
                    shape = (n_scans, w_c, n_angles)
                    full_shape = (n_scans, n_pixels, n_angles)
                    matched = np.zeros((n, n_angles), dtype=np.float32)
                else:
                    shape = (n_scans, w_c)
                    full_shape = (n_scans, n_pixels)
                    matched = np.zeros(n, dtype=np.float32)

            matched[:] = np.nan
            matched[indices, ...] = self.data[target]
            matched[indices, ...][dists > 5e3] = np.nan
            matched = matched.reshape(shape)

            matched_full = np.zeros(full_shape, dtype=np.float32)
            matched_full[:] = np.nan
            matched_full[:, cmpr] = matched

            if target in PROFILE_NAMES:
                data = matched_full[:, cmpr]
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
                        matched_full[:, cmpr].astype(np.float32),
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
                if self.sensor.sensor_name in CHANNEL_INDICES:
                    ch_inds = CHANNEL_INDICES[self.sensor.sensor_name]
                    data = data[..., ch_inds]

            dims = ("samples",)
            if len(data.shape) > 1:
                dims = dims + tuple([dim_dict[s] for s in data.shape[1:]])

            results[key] = dims, data

        dataset = xr.Dataset(results)

        year = dataset["scan_time"].data["year"] - 1970
        month = dataset["scan_time"].data["month"] - 1
        day = dataset["scan_time"].data["day"] - 1
        hour = dataset["scan_time"].data["hour"]
        minute = dataset["scan_time"].data["minute"]
        second = dataset["scan_time"].data["second"]
        dates = (
            year.astype("datetime64[Y]")
            + month.astype("timedelta64[M]")
            + day.astype("timedelta64[D]")
            + hour.astype("timedelta64[h]")
            + minute.astype("timedelta64[m]")
            + second.astype("timedelta64[s]")
        )
        dataset["scan_time"] = (("samples",), dates.astype("datetime64[ns]"))

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
    airlifting_index = data["airlifting_index"].data
    surface_precip = data["surface_precip"].data
    convective_precip = data["convective_precip"].data

    enh = np.ones(surface_precip.shape, dtype=np.float32)
    factors = ENHANCEMENT_FACTORS[kind]
    for t_s in [17, 18]:
        for t_a in range(4):
            key = (t_s, t_a)
            if key not in factors:
                continue
            indices = (surface_types == t_s) * (airlifting_index == t_a)
            enh[indices] = factors[key]

    surface_precip *= enh
    convective_precip *= enh


###############################################################################
# Helper functions
###############################################################################


def _extract_scenes(data, min_valid=20):
    """
    Extract 221 x 221 pixel wide scenes from dataset where
    ground truth surface precipitation rain rates are
    available.

    Args:
        data: xarray.Dataset containing the data from the preprocessor
            together matched reference data.
        min_valid: The minimum number of pixels with valid surface
            precipitation.

    Return:
        New xarray.Dataset which containing 128x128 patches of input data
        and corresponding surface precipitation.
    """
    n = 221
    surface_precip = data["surface_precip"].data

    n_scans = data.scans.size
    n_pixels = data.pixels.size

    if np.all(np.isnan(surface_precip)):
        return None

    valid = np.stack(np.where(surface_precip >= 0.0), -1)
    valid_inds = list(np.random.permutation(valid.shape[0]))

    scenes = []

    while len(valid_inds) > 0:

        ind = np.random.choice(valid_inds)
        scan_cntr, pixel_cntr = valid[ind]

        scan_start = min(max(scan_cntr - n // 2, 0), n_scans - n)
        scan_end = scan_start + 221
        pixel_start = min(max(pixel_cntr - n // 2, 0), n_pixels - n)
        pixel_end = pixel_start + 221

        subscene = data[
            {
                "scans": slice(scan_start, scan_end),
                "pixels": slice(pixel_start, pixel_end),
            }
        ]
        surface_precip = subscene["surface_precip"].data
        if np.isfinite(surface_precip).sum() > 20:
            scenes.append(subscene)
            covered = (
                (valid[..., 0] >= scan_start) * (valid[..., 0] < scan_end) *
                (valid[..., 1] >= pixel_start) * (valid[..., 1] < pixel_end)
            )
            covered = {ind for ind in valid_inds if covered[ind]}
            valid_inds = [ind for ind in valid_inds if not ind in covered]
        else:
            valid_inds.remove(ind)

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


def collocate_targets(
        sim_filename,
        sensor,
        era5_path,
        subset=None,
        log_queue=None
):
    """
    This method collocates retrieval input data from the preprocessor
    with retrieval targets from a sim file.

    For GMI also ERA5 precip over sea ice and sea ice edge are
    added to the surface precipitation.

    Surface and convective precipitation over mountains is set to NAN.
    This is also done for precipitation over sea ice for sensors that
    are not GMI.

    Args:
        sim_filename: Filename of the Sim file to process.
        sensor: The sensor for which the training data is extracted.
        era5_path: Base path of the directory containing the ERA5 data.
        subset: An optional SubsetConfig object limiting that will be
            used to mask samples not to be used in the training.
        log_queue: Optional queue object to use for multi-process logging.

    Return:
        An xarray.Dataset containing GMI preprocessor data collocated with
        reference data from the given SIM file.
    """
    import gprof_nn.logging

    if log_queue is not None:
        gprof_nn.logging.configure_queue_logging(log_queue)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Processing sim file %s.", sim_filename)

    # Load sim file and corresponding GMI L1C file.
    sim_file = SimFile(sim_filename)
    l1c_file = L1CFile.open_granule(
        sim_file.granule, sensors.GMI.l1c_file_path, sensors.GMI
    )

    LOGGER.info("Running preprocessor for sim file %s.", sim_filename)
    data_pp = run_preprocessor(
        l1c_file.filename, sensor=sensor, robust=False
    )
    data_pp = data_pp.drop_vars(
        [
            "earth_incidence_angle",
            "sunglint_angle",
            "quality_flag",
            "wet_bulb_temperature",
            "lapse_rate",
        ]
    )
    if isinstance(sensor, sensors.CrossTrackScanner):
        data_pp["earth_incidence_angle"] = (
            ("scans", "pixels"),
            np.ones_like(data_pp.two_meter_temperature.data),
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
        era5_data = load_era5_data(start_time, end_time)
        add_era5_precip(data_pp, era5_data)
        LOGGER.debug("Added era5 precip.")

    # Else set to missing.
    else:
        sea_ice = (surface_type == 2) + (surface_type == 16)
        for var in ["surface_precip", "convective_precip"]:
            data_pp[var].data[sea_ice] = np.nan

    if subset is not None:
        subset.mask_surface_precip(data_pp)

    return data_pp


def write_training_samples_1d(
        dataset: xr.Dataset,
        output_path: Path,
) -> None:
    """
    Write training data in GPROF-NN 1D format.

    Args:
        dataset: An 'xarray.Dataset' containing collocated input
            observations and reference data.
        output_path: Path to which the training data will be written.
    """
    subset = {}
    dataset = dataset[{"pixels": slice(*compressed_pixel_range())}]
    mask = np.isfinite(dataset.surface_precip.data)

    for var in dataset.variables:
        arr = dataset[var]
        if arr.data.ndim < 2:
            arr_data = np.broadcast_to(arr.data[..., None], mask.shape)
        else:
            arr_data = arr.data

        subset[var] = ((("samples",) + arr.dims[2:]), arr_data[mask])

    subset = xr.Dataset(subset)
    start_time = pd.to_datetime(dataset.scan_time.data[0].item())
    start_time = start_time.strftime("%Y%m%d%H%M%S")
    end_time = pd.to_datetime(dataset.scan_time.data[-1].item())
    end_time = end_time.strftime("%Y%m%d%H%M%S")
    filename = f"sim_{start_time}_{end_time}.nc"

    save_scene(subset, output_path / filename)


def write_training_samples_3d(
        dataset,
        output_path,
        min_valid=20
):
    """
    Write training data in GPROF-NN 3D format.

    Args:
        dataset:
        output_path:
        min_valid: The minimum number of valid surface precipitation
            pixels for a scene to be stored.

    """
    mask = np.any(np.isfinite(dataset.surface_precip.data), 1)
    valid_scans = np.where(mask)[0]
    n_scans = dataset.scans.size

    encodings = {
        name: {"zlib": True} for name in dataset.variables
    }

    while len(valid_scans) > 0:

        ind = np.random.randint(0, len(valid_scans))
        scan_start = min(max(valid_scans[ind] - 110, 0), n_scans - 221)
        scan_end = scan_start + 221

        scene = dataset[{"scans": slice(scan_start, scan_end)}]
        start_time = pd.to_datetime(scene.scan_time.data[0].item())
        start_time = start_time.strftime("%Y%m%d%H%M%S")
        end_time = pd.to_datetime(scene.scan_time.data[-1].item())
        end_time = end_time.strftime("%Y%m%d%H%M%S")
        filename = f"sim_{start_time}_{end_time}.nc"

        valid_pixels = (scene.surface_precip.data >= 0.0).sum()
        if valid_pixels > min_valid:
            #scene.to_netcdf(output_path / filename, encoding=encodings)
            save_scene(scene, output_path / filename)

        within_scene = (valid_scans >= scan_start) * (valid_scans < scan_end)
        if within_scene.sum() == 0:
            break
        valid_scans = valid_scans[~within_scene]


def process_sim_file(
        sensor: Sensor,
        sim_file: Path,
        era5_path: Optional[Path],
        output_path_1d: Path,
        output_path_3d: Path,
) -> None:
    """
    Extract training data from a single .sim-file and write output to
    given folders.

    Args:
        sensor: A sensor object representing the sensor for which to extract
            the training data.
        sim_file: Path to the .sim file to process.
        era5_path: Path to the ERA5 data archive. This is required for adding
            precip over sea ice surfaces.
        output_path_1d: Path pointing to the folder to which the 1D training data
            should be written.
        output_path_3d: Path pointing to the folder to which the 3D training data
            should be written.
    """
    data = collocate_targets(sim_file, sensor, era5_path)
    if output_path_1d is not None:
        write_training_samples_1d(data, output_path_1d)
    if output_path_3d is not None:
        write_training_samples_3d(data, output_path_3d)


def process_files(
        sensor: Sensor,
        path: Path,
        start_time: np.datetime64,
        end_time: np.datetime64,
        output_path_1d: Path,
        output_path_3d: Path,
        n_processes: int = 1
) -> None:
    """
    Parallel processing of all .sim files within a given time range.

    Args:
        sensor: A sensor object representing the sensor for which to extract
            the training data.
        sim_file_path: Path to the folder containing the sim files from which to
            extract the training data.
        start_time: Start time of the time interval limiting the sim files from
            which training scenes will be extracted.
        end_time: End time of the time interval limiting the sim files from
            which training scenes will be extracted.
        output_path_1d: Path pointing to the folder to which the 1D training data
            should be written.
        output_path_3d: Path pointing to the folder to which the 3D training data
            should be written.
    """
    sim_files = sorted(list(path.glob("**/*.sim")))
    files = []
    for path in sim_files:
        date = path.stem.split(".")[-2]
        date = datetime.strptime(date, "%Y%m%d")
        if (date >= start_time) and (date < end_time):
            files.append(path)

    LOGGER.info(f"""
    Found {len(files)} files to process in time range {start_time} - {end_time}.
    """)

    pool = futures.ProcessPoolExecutor(max_workers=n_processes)
    tasks = []
    for path in files:
        tasks.append(
            pool.submit(
                process_sim_file,
                sensor,
                path,
                CONFIG.data.era5_path,
                output_path_1d,
                output_path_3d
            )
        )
        tasks[-1].file = path

    for task in futures.as_completed(tasks):
        try:
            task.result()
            LOGGER.info(f"""
            Finished processing file {task.file}.
            """)
        except Exception as exc:
            LOGGER.exception(
                "The following error was encountered when processing file %s:"
                "%s.",
                task.file,
                exc
            )


@click.argument("sensor")
@click.argument("sim_file_path")
@click.argument("start_time")
@click.argument("end_time")
@click.argument("output_1d")
@click.argument("output_3d")
@click.option("-n" ,"--n_processes", default=1)
def cli(
        sensor: Sensor,
        sim_file_path: Path,
        start_time: np.datetime64,
        end_time: np.datetime64,
        output_1d: Path,
        output_3d: Path,
        n_processes: int = 1
) -> None:
    """
    This function implements the command line interface for extracting
    training data from sim files.

    Args:
        sensor: A sensor object representing the sensor for which to extract
            the training data.
        sim_file_path: Path to the folder containing the sim files from which to
            extract the training data.
        start_time: Start time of the time interval limiting the sim files from
            which training scenes will be extracted.
        end_time: End time of the time interval limiting the sim files from
            which training scenes will be extracted.
        output_1d: Path pointing to the folder to which the 1D training data
            should be written.
        output_3d: Path pointing to the folder to which the 3D training data
            should be written.
    """
    from gprof_nn import sensors
    from gprof_nn.data.sim import process_files

    # Check sensor
    sensor_obj = getattr(sensors, sensor.strip().upper(), None)
    if sensor_obj is None:
        LOGGER.error("The sensor '%s' is not known.", sensor)
        return 1
    sensor = sensor_obj

    sim_file_path = Path(sim_file_path)
    if not sim_file_path.exists() or not sim_file_path.is_dir():
        LOGGER.error("The 'sim_file_path' argument must point to a directory.")
        return 1

    output_path_1d = Path(output_1d)
    if not output_path_1d.exists() or not output_path_1d.is_dir():
        LOGGER.error("The 'output_1d' argument must point to a directory.")
        return 1

    output_path_3d = Path(output_3d)
    if not output_path_3d.exists() or not output_path_3d.is_dir():
        LOGGER.error("The 'output_3d' argument must point to a directory.")
        return 1

    try:
        start_time = np.datetime64(start_time)
    except ValueError:
        LOGGER.error(
            "Coud not parse 'start_time' argument as numpy.datetime64 object. "
            "Please make sure that the start time is provided in the right "
            "format."
        )
        return 1

    try:
        end_time = np.datetime64(end_time)
    except ValueError:
        LOGGER.error(
            "Coud not parse 'end_time' argument as numpy.datetime64 object. "
            "Please make sure that the start time is provided in the right "
            "format."
        )
        return 1

    process_files(
        sensor,
        sim_file_path,
        start_time,
        end_time,
        output_path_1d,
        output_path_3d,
        n_processes=n_processes
    )



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


def get_l1c_files_for_seaice(sensor, day):
    """
    Finds sensors L1C files that should be used to extract
    ERA5 collocations.

    The function first checks whether there is a specific SEAICE year
    defined for the given sensor in ``gprof_nn.definitions``. If that
    is not the case it will look for L1C files for the current database
    period.

    If the above doesn't produce any L1C files, then GMI collocations
    with ERA5 are used.

    Args:
        sensor: Sensor for which the data is to be extracted.

    Return:
        List of L1C filenames to process.
    """
    # Collect L1C files to process.
    l1c_file_path = sensor.l1c_file_path
    l1c_files = []

    # Get L1C for specific year ...
    if sensor.name in SEAICE_YEARS:
        year = SEAICE_YEARS[sensor.name]
        for month in range(1, 13):
            try:
                date = datetime(year, month, day)
                l1c_files += list(L1CFile.find_files(
                    date, l1c_file_path, sensor=sensor
                ))
            except ValueError:
                pass
    else:
        for year, month in DATABASE_MONTHS:
            try:
                date = datetime(year, month, day)
                l1c_files += list(L1CFile.find_files(
                    date, l1c_file_path, sensor=sensor
                ))
            except ValueError:
                pass

    # If no L1C files are found use GMI co-locations.
    if len(l1c_files) < 1:
        for year, month in DATABASE_MONTHS:
            try:
                date = datetime(year, month, day)
                l1c_file_path = sensors.GMI.l1c_file_path
                l1c_files += list(L1CFile.find_files(
                    date, l1c_file_path, sensor=sensors.GMI
                ))
            except ValueError:
                pass
    l1c_files = [f.filename for f in l1c_files]
    l1c_files = np.random.permutation(l1c_files)
    return l1c_files


@dataclass
class SubsetConfig:
    tcwv_bounds: Optional[Tuple[float, float]] = None
    t2m_bounds: Optional[Tuple[float, float]] = None
    ocean_only: bool = False
    land_only: bool = False
    surface_types: Optional[Tuple[float, float]] = None


    def mask_surface_precip(self, dataset):
        """
        Sets surface precip in given dataset to nan for samples
        outside certain ancillary data bounds.

        Args:
            dataset: An xarray.Dataset containing a 'surface_precip' field
                and GPROF anciallary data.
        """
        surface_precip = dataset.surface_precip.data

        if self.tcwv_bounds is not None:
            tcwv_min, tcwv_max = self.tcwv_bounds
            tcwv = dataset.total_column_water_vapor.data
            valid = (tcwv >= tcwv_min) * (tcwv <= tcwv_max)
            surface_precip[~valid] = np.nan

        if self.t2m_bounds is not None:
            t2m_min, t2m_max = self.t2m_bounds
            t2m = dataset.two_meter_temperature.data
            valid = (t2m >= t2m_min) * (t2m <= t2m_max)
            surface_precip[~valid] = np.nan

        if self.ocean_only:
            ocean_frac = dataset.ocean_fraction
            valid = ocean_frac == 100
            surface_precip[~valid] = np.nan

        if self.land_only:
            land_frac = dataset.land_fraction
            valid = land_frac == 100
            surface_precip[~valid] = np.nan

        if self.surface_types is not None:
            valid = np.zeros_like(surface_precip, dtype=bool)
            for surface_type in self.surface_types:
                valid += dataset.surface_type.data == surface_type
            surface_precip[~valid] = np.nan


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
        subset=None
    ):
        """
        Create retrieval driver.

        Args:
            output_file: The file in which to store the extracted data.
            sensor: Sensor object defining the sensor for which to extract
                training data.
            era5_path: Path to the root of the directory tree containing
                ERA5 data.
            n_workers: The number of worker processes to use.
            day: Day of the month for which to extract the data.
            subset: A SubsetConfig object specifying a subset of the
                database to extract.
        """
        self.output_file = output_file
        self.sensor = sensor
        self.configuration = configuration

        self.era5_path = era5_path
        if self.era5_path is not None:
            self.era5_path = Path(self.era5_path)

        self.pool = futures.ProcessPoolExecutor(max_workers=n_workers)

        if day is None:
            self.day = 1
        else:
            self.day = day

        if subset is None:
            subset = SubsetConfig()
        self.subset = subset

    def run(self):
        """
        Start the processing.

        This will start processing all suitable input files that have been found and
        stores the names of the produced result files in the ``processed`` attribute
        of the driver.
        """
        # Collect simulator files to process.
        sim_file_path = self.sensor.sim_file_path
        if self.sensor.sim_file_path is not None:
            sim_files = SimFile.find_files(
                sim_file_path,
                sensor=self.sensor,
                day=self.day
            )
            sim_files = np.random.permutation(sim_files)
        else:
            sim_files = []

        # Collect MRMS files to process.
        if self.sensor.mrms_file_path is not None:
            mrms_file_path = self.sensor.mrms_file_path
            if mrms_file_path is None:
                mrms_files = MRMSMatchFile.find_files(
                    sensors.GMI.mrms_file_path, sensor=sensors.GMI
                )
            else:
                if hasattr(self.sensor, "mrms_sensor"):
                    mrms_sensor = self.sensor.mrms_sensor
                else:
                    mrms_sensor = self.sensor
                mrms_files = MRMSMatchFile.find_files(
                    mrms_file_path,
                    sensor=mrms_sensor
                )
            mrms_files = np.random.permutation(mrms_files)
        else:
            mrms_files = []

        # Collect L1C files to process.
        l1c_file_path = self.sensor.l1c_file_path
        if self.era5_path is not None:
            l1c_files = get_l1c_files_for_seaice(self.sensor, self.day)[:100]
        else:
            l1c_files = []

        n_sim_files = len(sim_files)
        LOGGER.info("Found %s SIM files.", n_sim_files)
        n_mrms_files = len(mrms_files)
        LOGGER.info("Found %s MRMS files.", n_mrms_files)
        n_l1c_files = len(l1c_files)
        LOGGER.info("Found %s L1C files.", n_l1c_files)
        i = 0

        # Submit tasks interleaving .sim and MRMS files.
        log_queue = gprof_nn.logging.get_log_queue()
        tasks = []
        files = []
        while i < max(n_sim_files, n_mrms_files, n_l1c_files):
            if i < n_sim_files:
                sim_file = sim_files[i]
                files.append(sim_file)
                tasks.append(
                    self.pool.submit(
                        process_sim_file,
                        sim_file,
                        self.sensor,
                        self.configuration,
                        self.era5_path,
                        self.subset,
                        log_queue=log_queue,
                    )
                )
            if i < n_mrms_files:
                mrms_file = mrms_files[i]
                files.append(mrms_file)
                if hasattr(self.sensor, "mrms_sensor"):
                    sensor = self.sensor.mrms_sensor
                else:
                    sensor = self.sensor
                tasks.append(
                    self.pool.submit(
                        process_mrms_file,
                        sensor,
                        mrms_file,
                        self.configuration,
                        self.day,
                        log_queue=log_queue,
                    )
                )
            if i < n_l1c_files:
                l1c_file = l1c_files[i]
                files.append(l1c_file)
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
            pbar = progress.add_task("Extracting data:", total=len(tasks))
            for task, filename in zip(tasks, files):
                # Log messages from processes.
                task_done = False
                dataset = None
                while not task_done:
                    try:
                        gprof_nn.logging.log_messages()
                        dataset = task.result()
                        task_done = True
                    except futures.TimeoutError:
                        pass
                    except Exception as exc:
                        LOGGER.error(
                            "The following error was encountered while "
                            "processing file %s results: %s",
                            str(filename),
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

                        encodings = {}
                        for var in dataset:
                            encodings[var] = {"zlib": True}
                            if dataset[var].dtype == np.float64:
                                encodings[var]["dtype"] = "float32"
                        dataset.to_netcdf(filename, encoding=encodings)
                        # subprocess.run(["lz4", "-f", "--rm", filename], check=True)
                        LOGGER.info("Finished writing file: %s", filename)
                        datasets = []
                        chunk += 1

        if len(datasets) > 0:
            # Store dataset with sensor name as attribute.
            dataset = xr.concat(datasets, "samples")
            filename = output_path / (output_file + f"_{chunk:02}.nc")
            dataset.attrs["sensor"] = self.sensor.name
            dataset.attrs["configuration"] = self.configuration
            LOGGER.info("Writing file: %s", filename)

            encodings = {}
            for var in dataset:
                encodings[var] = {"zlib": True}
                if dataset[var].dtype == np.float64:
                    encodings[var]["dtype"] = "float32"
            dataset.to_netcdf(filename, encoding=encodings)

        # Explicit clean up to avoid memory leak.
        del datasets
        del dataset
