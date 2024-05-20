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
from copy import copy
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
from gprof_nn.definitions import (
    ALL_TARGETS,
    DATABASE_MONTHS,
    DATA_SPLIT,
    LEVELS,
    N_LAYERS,
    PROFILE_TARGETS,
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
    save_scene,
    write_training_samples_1d,
    write_training_samples_3d
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
        if isinstance(self.sensor, sensors.ConicalScanner):
            n_chans = 15

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
            if target in PROFILE_TARGETS:
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

            if target in PROFILE_TARGETS:
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
                if target in [
                        "surface_precip",
                        "surface_precip_combined",
                        "convective_precip"
                ]:
                    dims = ("scans", "pixels")
                    if n_angles > 0:
                        dims = dims + ("angles",)
                    input_data[target] = (dims, matched_full.astype(np.float32))
                else:
                    input_data[target] = (
                        ("scans", "pixels_center"),
                        matched_full[:, cmpr].astype(np.float32),
                    )

        if n_angles > 0:
            input_data["angles"] = (("angles",), self.header["viewing_angles"][0])

        return input_data

    def to_xarray_dataset(self):
        """
        Return data in sim file as 'xarray.Dataset'.
        """
        results = {}
        dim_dict = {
            len(self.header["frequencies"][0]): "channels",
            N_LAYERS: "layers",
        }
        if isinstance(self.sensor, sensors.CrossTrackScanner):
            dim_dict[self.sensor.n_angles] = "angles"

        record_type = self.sensor.sim_file_record
        for key, _, *shape in record_type.descr:

            data = self.data[key]
            dims = ("samples",)
            if len(data.shape) > 1:
                dims = dims + tuple([dim_dict[s] for s in data.shape[1:]])

            results[key] = dims, data

        if isinstance(self.sensor, sensors.CrossTrackScanner):
            results["angles"] = (("angles",), self.header["viewing_angles"][0])

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


def apply_orographic_enhancement(sensor, data, kind="ERA5"):
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
    surface_types = data["surface_type"].data
    airlifting_index = data["airlifting_index"].data
    surface_precip = data["surface_precip"].data
    convective_precip = data["convective_precip"].data

    enh = np.ones(surface_precip.shape, dtype=np.float32)

    factors = sensor.orographic_enhancement
    types = ((17, 1), (17, 2), (17, 3), (17, 4), (18, 1))
    for ind, (t_s, t_a) in enumerate(types):
        indices = (surface_types == t_s) * (airlifting_index == t_a)
        enh[indices] = sensor.orographic_enhancement[ind]

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
        sim_filename: Path,
        sensor: sensors.Sensor,
        era5_path: Path,
        subset: Optional["SubsetConfig"] = None,
        log_queue: Optional["Queue"] = None,
        include_cmb_precip: bool = False
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
        data_pp = data_pp.rename({"channels": "channels_gmi"})

    # Match targets from sim file to preprocessor data.
    LOGGER.debug("Matching retrieval targets for file %s.", sim_filename)
    targets = copy(ALL_TARGETS)
    if include_cmb_precip:
        targets.append("surface_precip_combined")
    sim_file.match_targets(data_pp, targets=targets)
    l1c_data = l1c_file.to_xarray_dataset()

    # Orographic enhancement for types 17 and 18.
    apply_orographic_enhancement(sensor, data_pp)

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

    data_pp.attrs["sensor"] = sim_file.sensor.name
    data_pp.attrs["source"] = "sim"

    return data_pp


def process_sim_file(
        sensor: Sensor,
        sim_file: Path,
        era5_path: Optional[Path],
        output_path_1d: Path,
        output_path_3d: Path,
        include_cmb_precip: bool = False,
        lonlat_bounds: Optional[Tuple[float, float, float, float]] = None
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
        include_cmb_precip: Flag to trigger include of surface precip derived solely
             from cmb.
        lonlat_bounds: Optional coordinate tuple ``(lon_ll, lat_ll, lon_ur, lat_ur)``
            containing the longitude and latitude coordinates of the lower-left corner
            (``lon_ll`` and ``lat_ll``) followed by the longitude and latitude coordinates
            of the upper right corner (``lon_ur``, ``lat_ur``).

            a rectangular bounding box to constrain the training data extracted.
    """
    data = collocate_targets(
        sim_file,
        sensor,
        era5_path,
        include_cmb_precip=include_cmb_precip
    )

    if lonlat_bounds is not None:
        lon_ll, lat_ll, lon_ur, lat_ur = lonlat_bounds
        lons = data.longitude.data
        lats = data.latitude.data
        valid = ((lats >= lat_ll) * (lats <= lat_ur))
        # Box expands across date line :(
        if lon_ur < lon_ll:
            valid *= ~((lon_ur < lons) * (lons < lon_ll))
        else:
            valid *= ((lon_ll <= lons) * (lons <= lon_ur))
        data.surface_precip.data[~valid] = np.nan

    if output_path_1d is not None:
        write_training_samples_1d(output_path_1d, "sim", data)
    if output_path_3d is not None:
        write_training_samples_3d(output_path_3d, "sim", data, n_scans=221, n_pixels=221)


def process_files(
        sensor: Sensor,
        path: Path,
        output_path_1d: Path,
        output_path_3d: Path,
        n_processes: int = 1,
        start_time: Optional[np.datetime64] = None,
        end_time: Optional[np.datetime64] = None,
        split: Optional[str] = None,
        include_cmb_precip: bool = False,
        lonlat_bounds: Optional[Tuple[float, float, float, float]] = None
) -> None:
    """
    Parallel processing of all .sim files within a given time range.

    Args:
        sensor: A sensor object representing the sensor for which to extract
            the training data.
        path: Path to the folder containing the sim files from which to
            extract the training data.
        output_path_1d: Path pointing to the folder to which the 1D training data
            should be written.
        output_path_3d: Path pointing to the folder to which the 3D training data
            should be written.
        start_time: Start time of the time interval limiting the sim files from
            which training scenes will be extracted.
        end_time: End time of the time interval limiting the sim files from
            which training scenes will be extracted.
        split: Optional string specifying which split of the data to extract.
            Must be one of 'training', 'validation', 'test'.
        include_cmb_precip: Flag to trigger include of surface precip derived solely
             from cmb.
        lonlat_bounds: Optional coordinate tuple ``(lon_ll, lat_ll, lon_ur, lat_ur)``
            containing the longitude and latitude coordinates of the lower-left corner
            (``lon_ll`` and ``lat_ll``) followed by the longitude and latitude coordinates
            of the upper right corner (``lon_ur``, ``lat_ur``).
    """
    sim_files = sorted(list(path.glob("**/*.sim")))
    files = []
    for path in sim_files:
        date = path.stem.split(".")[-2]
        try:
            date = datetime.strptime(date, "%Y%m%d")
        except ValueError:
            LOGGER.warning(
                "Ignoring file not matching expected sim file name patter: %s",
                path
            )
            continue
        if start_time is not None and date < start_time:
            continue
        if end_time is not None and date > end_time:
            continue
        if split is not None:
            days = DATA_SPLIT[split.lower()]
            if not date.day in days:
                continue
        files.append(path)

    if start_time is None and split is None:
        LOGGER.info("Found %s files to process.", len(files))
    elif split is not None:
        LOGGER.info("Found %s files to process for split '%s'.", len(files), split)
    else:
        LOGGER.info(
            "Found  files to process for split '%s' in time range %s - %s.",
            len(files),
            split,
            start_time,
            end_time
        )

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
                output_path_3d,
                include_cmb_precip=include_cmb_precip,
                lonlat_bounds=lonlat_bounds
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
@click.argument("split", type=click.Choice(['training', 'validation', 'test']))
@click.argument("output_1d")
@click.argument("output_3d")
@click.option(
    "--start_time",
    default=None,
    help="Optional start time to limit the .sim files from which training data will be extracted.",
    metavar="YYYY-mm-ddTHH:MM:SS"
)
@click.option(
    "--end_time",
    default=None,
    help="Optional end time to limit the .sim files from which training data will be extracted.",
    metavar="YYYY-mm-ddTHH:MM:SS"
)
@click.option("-n" ,"--n_processes", default=1)
@click.option(
    "--include_cmb_precip",
    is_flag=True,
    default=False,
    help="If set, non-MIRS-augmented CMB-only precipitation will be included in the training data."
)
@click.option(
    "--bounds",
    default=None,
    help=(
        "Optional bounds lon_min,lat_min,lon_max,lat_max consisting of four comma-separated floating "
        "point values specifying the longitude and latitude coordinates of the lower left corner"
        "(lon_min, lat_min) and upper right corner (lon_max, lat_max) of a rectangular bounding box "
        "to which to restrict the training data. If lon_min > lon_max, the bounding box is chosen so"
        " that it wraps around the date line."
    ),
    metavar="lon_min,lat_min,lon_max,lat_max"

)
def cli(sensor: Sensor,
        sim_file_path: Path,
        split: str,
        output_1d: Path,
        output_3d: Path,
        start_time: Optional[np.datetime64] = None,
        end_time: Optional[np.datetime64] = None,
        n_processes: int = 1,
        include_cmb_precip: bool = False,
        bounds: Tuple[float, float, float, float] = None
) -> None:
    """
    \b
    Extract GPROF-NN training/validation/test data for the sensor SENSOR from sim
    files located in SIM_FILE_PATH and store extracted data in OUTPUT_1D and
    OUTPUT_3D
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

    if start_time is not None:
        try:
            start_time = np.datetime64(start_time)
        except ValueError:
            LOGGER.error(
                "Coud not parse 'start_time' argument as numpy.datetime64 object. "
                "Please make sure that the start time is provided in the right "
                "format."
            )
            return 1

    if end_time is not None:
        try:
            end_time = np.datetime64(end_time)
        except ValueError:
            LOGGER.error(
                "Coud not parse 'end_time' argument as numpy.datetime64 object. "
                "Please make sure that the start time is provided in the right "
                "format."
            )
            return 1

    if bounds is not None:
        bnds = bounds.split(",")
        if not len(bnds) == 4:
            LOGGER.error(
                "If provided 'bounds' should be a string containing the longitude and latitude "
                "coordinates of the lower-left corner followed by the coordinates of the upper "
                "right corner separated by commas."
            )
        lon_ll = float(bnds[0])
        lat_ll = float(bnds[1])
        lon_ur = float(bnds[2])
        lat_ur = float(bnds[3])
        lonlat_bounds = (lon_ll, lat_ll, lon_ur, lat_ur)
    else:
        lonlat_bounds = None


    process_files(
        sensor,
        sim_file_path,
        output_path_1d,
        output_path_3d,
        split=split,
        start_time=start_time,
        end_time=end_time,
        n_processes=n_processes,
        include_cmb_precip=include_cmb_precip,
        lonlat_bounds=lonlat_bounds
    )



