"""
=========================
gprof_nn.data.pretraining
=========================

The module provides functionality to extract data for unsupervised pre-training
of the GPROF-NN T model.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple
import os

import click
import numpy as np
import pandas as pd
from pansat import Granule, TimeRange
from pansat.catalog import Index
from pansat.catalog.index import find_matches
from pansat.granule import merge_granules
from pansat.products import Product
from pansat.products.satellite.gpm import (
    l1c_gpm_gmi,
    l1c_npp_atms,
    l1c_gcomw1_amsr2
)
from pansat.utils import resample_data
from pyresample.geometry import SwathDefinition
from rich.progress import Progress, track
import torch
import xarray as xr

from gprof_nn.sensors import Sensor
from gprof_nn.data import preprocessor
from gprof_nn.data.utils import save_scene, extract_scenes
from gprof_nn.data.l1c import L1CFile
from gprof_nn.logging import (
    configure_queue_logging,
    log_messages,
    get_log_queue,
    get_console
)


LOGGER = logging.getLogger(__name__)


PRODUCTS = {
    "gmi": (l1c_gpm_gmi,),
    "atms": (l1c_npp_atms,),
    "amsr2": (l1c_gcomw1_amsr2,)
}

POLARIZATIONS = {
    "H": 0,
    "QH": 1,
    "V": 2,
    "QV": 2,
}


BEAM_WIDTHS = {
    "gmi": [1.75, 1.75, 1.0, 1.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    "atms": [5.2, 5.2, 2.2, 1.1, 1.1, 1.1, 1.1, 1.1],
    "amsr2": [1.2, 1.2, 0.65, 0.65, 0.75, 0.75, 0.35, 0.35, 0.15, 0.15, 0.15, 0.15],
}


CHANNEL_REGEXP = re.compile("([\d\.\s\+\/-]*)\s*GHz\s*(\w*)-Pol")


SEM_A = 6_378_137.0
SEM_B = 6_356_752.0
ECC2 = 1.0 - (SEM_B ** 2 / SEM_A ** 2)


def lla_to_ecef(coords_lla: np.ndarray):
    """
    Converts latitude-longitude-altitude (LLA) coordinates to
    earth-centric earth-fixed coordinates (ECEF)

    Params:
        coords_lla: A numpy.ndarray containing the three coordinates oriented along the last axis.

    Return:
        coords_ecef: An array of the same shape as 'coords_lla' but containing the x, y, and z
             coordinates along the last axis.
    """
    lon = np.radians(coords_lla[..., 0])
    lat = np.radians(coords_lla[..., 1])
    alt = coords_lla[..., 2]

    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)

    x = (roc + alt) * np.cos(lat) * np.cos(lon)
    y = (roc + alt) * np.cos(lat) * np.sin(lon)
    z = (roc * (1 - ECC2) + alt) * np.sin(lat)

    return np.stack((x, y, z), -1)


def extract_scans(granule: Granule, dest: Path, min_scans: Optional[int] = None) -> Path:
    """
    Extract and write scans from L1C file into a separate file.

    Args:
        granule: A pansat granule specifying a subset of an orbit.
        dest: A directory to which the extracted scans will be written.
        min_scans: A minimum number of scans to extract.

    Return:
        The path of the file containing the extracted scans.
    """
    scan_start, scan_end = granule.primary_index_range
    n_scans = scan_end - scan_start
    if min_scans is not None and n_scans < min_scans:
        scan_c = (scan_end + scan_start) // 2
        scan_start = scan_c - min_scans // 2
        scan_end = scan_start + min_scans
    l1c_path = granule.file_record.local_path
    l1c_file = L1CFile(granule.file_record.local_path)
    output_filename = dest / l1c_path.name
    l1c_file.extract_scan_range(scan_start, scan_end, output_filename)
    return output_filename


def run_preprocessor(gpm_granule: Granule) -> xr.Dataset:
    """
    Run preprocessor on a GPM granule.

    Args:
        gpm_granule: A pansat granule identifying a subset of an orbit
            of GPM L1C files.

    Return:
        An xarray.Dataset containing the results from the preprocessor.
    """
    old_dir = os.getcwd()

    try:
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            l1c_file = extract_scans(gpm_granule, tmp, min_scans=128)
            os.chdir(tmp)
            sensor = L1CFile(l1c_file).sensor
            preprocessor_data = preprocessor.run_preprocessor(
                l1c_file, sensor, robust=False
            )
    finally:
        os.chdir(old_dir)

    preprocessor_data = preprocessor_data.rename({
        "scans": "scan",
        "pixels": "pixel",
        "channels": "channel_gprof",
        "brightness_temperatures": "observations_gprof",
        "earth_incidence_angle": "earth_incidence_angle_gprof"
    })
    invalid = preprocessor_data.observations_gprof.data < 0
    preprocessor_data.observations_gprof.data[invalid] = np.nan

    return preprocessor_data


def calculate_angles(
        fp_lons: np.ndarray,
        fp_lats: np.ndarray,
        sensor_lons: np.ndarray,
        sensor_lats: np.ndarray,
        sensor_alts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate zenith and azimuth angles describing the observations geometry.

    Args:
        fp_lons: Array containing the longitude coordinates of the observation
            footprints.
        fp_lats: Array containing the latitude coordinates of the observation
            footprints.
        sensor_lons: The longitude coordinates of the sensor.
        sensor_lats: The latitude coordinates of the sensor.
        sensor_alts: The altitude coordinates of the sensor.

    Return:
        A tuple ``zenith, azimuth`` containing the zenith and azimuth coordinates
        of all lines of sights.
    """
    sensor_lla = np.stack((sensor_lons, sensor_lats, sensor_alts), -1)
    sensor_ecef = lla_to_ecef(sensor_lla)

    fp_lla = np.stack((fp_lons, fp_lats, np.zeros_like(fp_lons)), -1)
    fp_ecef = lla_to_ecef(fp_lla)
    local_up = fp_ecef / np.linalg.norm(fp_ecef, axis=-1, keepdims=True)
    fp_east = fp_lla.copy()
    fp_east[..., 0] = 0.1
    fp_east = lla_to_ecef(fp_east)
    fp_east = fp_east / np.linalg.norm(fp_east, axis=-1, keepdims=True)
    local_north = np.cross(local_up, fp_east)

    sensor_ecef = np.broadcast_to(sensor_ecef[..., None, :], fp_lla.shape)
    los = sensor_ecef - fp_ecef
    zenith = np.arccos((local_up * los).sum(-1) / np.linalg.norm(los, axis=-1))
    proj = los - local_up * (local_up * los).sum(-1, keepdims=True)

    azimuth = np.arccos((local_north * proj).sum(-1) / np.linalg.norm(proj, axis=-1))
    mask = np.isclose(np.linalg.norm(proj, axis=-1), 0.0)
    azimuth[mask] = 0.0

    return np.rad2deg(zenith), np.rad2deg(azimuth)


def calculate_obs_properties(
        preprocessor_data: xr.Dataset,
        granule: Granule,
        radius_of_influence: float = 5e3,
) -> xr.Dataset:
    """
    Extract observations and corresponding meta data from granule.

    Args:
        preprocessor_data: The preprocessor data to which to resample all
            observaitons.
        granule:


    """
    lons = preprocessor_data.longitude.data
    lats = preprocessor_data.latitude.data
    swath = SwathDefinition(lats=lats, lons=lons)

    observations = []
    meta_data = []

    l1c_file = L1CFile(granule.file_record.local_path)
    sensor = l1c_file.sensor.name.lower()

    granule_data = granule.open()
    if "latitude" in granule_data:
        pass

    else:
        swath_ind = 1
        while f"latitude_s{swath_ind}" in granule_data:

            freqs = []
            offsets = []
            pols = []

            for match in CHANNEL_REGEXP.findall(granule_data[f"tbs_s{swath_ind}"].attrs["LongName"]):
                freq, pol = match
                freq = freq.replace("/", "")
                if freq.find("+-") > 0:
                    freq, offs = freq.split("+-")
                    freqs.append(float(freq))
                    offsets.append(float(offs))
                else:
                    freqs.append(float(freq))
                    offsets.append(0.0)
                pols.append(POLARIZATIONS[pol])

            swath_data = granule_data[[
                f"longitude_s{swath_ind}",
                f"latitude_s{swath_ind}",
                f"tbs_s{swath_ind}",
                f"channels_s{swath_ind}"
            ]]

            fp_lons = swath_data[f"longitude_s{swath_ind}"].data
            fp_lats = swath_data[f"latitude_s{swath_ind}"].data
            sensor_lons = granule_data["spacecraft_longitude"].data
            sensor_lats = granule_data["spacecraft_latitude"].data
            sensor_alt = granule_data["spacecraft_altitude"].data * 1e3
            zenith, azimuth = calculate_angles(fp_lons, fp_lats, sensor_lons, sensor_lats, sensor_alt)
            sensor_alt = np.broadcast_to(sensor_alt[..., None], zenith.shape)

            swath_data = swath_data.rename({
                f"longitude_s{swath_ind}": "longitude",
                f"latitude_s{swath_ind}": "latitude"
            })
            swath_data["sensor_alt"] = (("scans", "pixels"), sensor_alt)
            swath_data["zenith"] = (("scans", "pixels"), zenith)
            swath_data["azimuth"] = (("scans", "pixels"), azimuth)

            swath_data_r = resample_data(
                swath_data,
                swath,
                radius_of_influence=radius_of_influence,
            )
            sensor_alt = swath_data_r.sensor_alt.data
            zenith = swath_data_r.zenith.data
            azimuth = swath_data_r.azimuth.data

            for chan_ind in range(swath_data_r[f"channels_s{swath_ind}"].size):
                observations.append(swath_data_r[f"tbs_s{swath_ind}"].data[..., chan_ind])
                meta = np.stack((
                    freqs[chan_ind] * np.ones_like(observations[-1]),
                    offsets[chan_ind] * np.ones_like(observations[-1]),
                    pols[chan_ind] * np.ones_like(observations[-1]),
                    BEAM_WIDTHS[sensor][chan_ind] * np.ones_like(observations[-1]),
                    sensor_alt,
                    zenith,
                    azimuth
                ))
                meta_data.append(meta)

            swath_ind += 1

        observations = np.stack(observations)
        meta_data = np.stack(meta_data)
        return xr.Dataset({
            "observations": (("channels", "scans", "pixels"), observations),
            "meta_data": (("channels", "meta", "scans", "pixels"), meta_data)
        })




def extract_pretraining_scenes(
        input_sensor: Sensor,
        target_sensor: Sensor,
        match: Tuple[Granule, Tuple[Granule]],
        output_path: Path,
        scene_size: Tuple[int, int],
        radius_of_influence: float
) -> None:
    input_granule, target_granules = match
    target_granules = merge_granules(sorted(list(target_granules)))
    for target_granule in target_granules:
        input_data = run_preprocessor(input_granule)
        input_obs = calculate_obs_properties(input_data, input_granule, radius_of_influence=radius_of_influence)
        target_obs = calculate_obs_properties(input_data, target_granule, radius_of_influence=radius_of_influence)

        training_data = xr.Dataset({
            "input_observations": input_obs.observations.rename(channels="input_channels"),
            "input_meta_data": input_obs.meta_data.rename(channels="input_channels"),
            "target_observations": target_obs.observations.rename(channels="target_channels"),
            "target_meta_data": target_obs.meta_data.rename(channels="target_channels"),
        })

        tbs = training_data.input_observations.data
        tbs[tbs < 0] = np.nan
        valid = np.isfinite(tbs).any(0)
        tbs = training_data.target_observations.data
        tbs[tbs < 0] = np.nan
        valid *= np.isfinite(tbs).any(0)
        training_data["valid"] = (("scans", "pixels"), np.zeros_like(valid, dtype="float32"))
        training_data.valid.data[~valid] = np.nan

        scenes = extract_scenes(
            training_data,
            n_scans=64,
            n_pixels=64,
            overlapping=True,
            min_valid=50,
            reference_var="valid",
        )
        LOGGER.info(
            "Extracted %s training scenes from %s.",
            len(scenes),
            input_granule
        )

        scene_ind = 0
        for scene in scenes:
            scene = scene.drop_vars(["valid"])
            start_time = target_granule.time_range.start
            start_str = start_time.strftime("%Y%m%d%H%M%S")
            end_time = target_granule.time_range.end
            end_str = end_time.strftime("%Y%m%d%H%M%S")
            output_filename = f"{input_sensor.name.lower()}_{target_sensor.name.lower()}_{start_str}_{end_str}_{scene_ind:04}.nc"
            encodings = {
                "input_observations": {"dtype": "float32", "zlib": True},
                "input_meta_data": {"dtype": "float32", "zlib": True},
                "target_observations": {"dtype": "float32", "zlib": True},
                "target_meta_data": {"dtype": "float32", "zlib": True},
            }
            scene.to_netcdf(output_path / output_filename)
            scene_ind += 1



class InputLoader:
    def __init__(self, inputs: List[Any], radius_of_influence: float = 100e3):
        self.inputs = inputs
        self.radius_of_influence = radius_of_influence

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> int:
        return self.load_data(index)

    def load_data(self, ind: int) -> Tuple[Dict[str, torch.Tensor], str, xr.Dataset]:

        input_granule, target_granules = self.inputs[ind]
        target_granule = sorted(list(target_granules))[0]

        input_data = run_preprocessor(input_granule)
        input_obs = calculate_obs_properties(input_data, input_granule, radius_of_influence=self.radius_of_influence)
        target_obs = calculate_obs_properties(input_data, target_granule, radius_of_influence=self.radius_of_influence)

        training_data = xr.Dataset({
            "latitude": input_data.latitude,
            "longitude": input_data.longitude,
            "input_observations": input_obs.observations.rename(channels="input_channels"),
            "input_meta_data": input_obs.meta_data.rename(channels="input_channels"),
            "target_observations": target_obs.observations.rename(channels="target_channels"),
            "target_meta_data": target_obs.meta_data.rename(channels="target_channels"),
        })
        tbs = training_data.input_observations.data
        tbs[tbs < 0] = np.nan
        n_seq_in = tbs.shape[0]
        tbs = training_data.target_observations.data
        tbs[tbs < 0] = np.nan
        n_seq_out = tbs.shape[0]

        input_data = {
            "input_observations": torch.tensor(training_data.input_observations.data)[None, None],
            "input_meta": torch.tensor(training_data.input_meta_data.data)[None].transpose(1, 2),
            "output_meta": torch.tensor(training_data.target_meta_data.data)[None].transpose(1, 2),
        }

        filename = "match_" + target_granule.time_range.start.strftime("%Y%m%d%H%M%s") + ".nc"

        return input_data, filename, training_data


def extract_samples(
        input_sensor: Sensor,
        target_sensor: Sensor,
        start_time: np.datetime64,
        end_time: np.datetime64,
        output_path: Path,
        scene_size: Tuple[int, int] = (64, 64),
        radius_of_influence: float = 20e3
) -> None:
    """
    Extract pretraining sensors.
    """

    input_products = PRODUCTS[input_sensor.name.lower()]
    target_products = PRODUCTS[target_sensor.name.lower()]
    for input_product in input_products:
        for target_product in target_products:
            input_recs = input_product.get(TimeRange(start_time, end_time))
            input_index = Index.index(input_product, input_recs)
            target_recs = target_product.get(TimeRange(start_time, end_time))
            target_index = Index.index(target_product, target_recs)
            matches = find_matches(input_index, target_index, np.timedelta64(15, "m"))
            for match in matches:
                extract_pretraining_scenes(
                    input_sensor,
                    target_sensor,
                    match,
                    output_path,
                    scene_size=scene_size,
                    radius_of_influence=radius_of_influence
                )



def process_l1c_file(
    sensor: Sensor,
    l1c_file: Path,
    output_path: Path,
    log_queue: Optional[multiprocessing.Queue] = None
) -> None:
    """
    Extract pretraining data from a single L1C file.

    Args:
        sensor: A gprof_nn.sensors.Sensor object specifying the sensor object
            for which to extract pretraining data.
        l1c_file: Path to the L1C file to process.
        output_path: The output path to which write the extracted scenes.
    """
    if log_queue is not None:
        import gprof_nn.logging
        configure_queue_logging(log_queue)
        LOGGER = logging.getLogger(__name__)

    output_path = Path(output_path)
    data_pp = run_preprocessor(
        l1c_file,
        sensor,
        robust=False
    )
    scenes = extract_scenes(
        data_pp,
        n_scans=128,
        n_pixels=64,
        overlapping=True,
        min_valid=50,
        reference_var="brightness_temperatures"
    )

    for scene in scenes:
        start_time = pd.to_datetime(scene.scan_time.data[0].item())
        start_time = start_time.strftime("%Y%m%d%H%M%S")
        end_time = pd.to_datetime(scene.scan_time.data[-1].item())
        end_time = end_time.strftime("%Y%m%d%H%M%S")
        filename = f"pre_{sensor.name.lower()}_{start_time}_{end_time}.nc"
        save_scene(scene, output_path / filename)


def process_l1c_files(
        sensor: Sensor,
        l1c_path: Path,
        start_time: np.datetime64,
        end_time: np.datetime64,
        output_path: Path,
        n_processes=4
) -> None:
    """
    Process multiple L1C files.

    Args:
        sensor: A gprof_nn.sensors.Sensor object specifying the sensor object
            for which to extract pretraining data.
        l1c_path: Path pointing to the root of the directory tree that contains
            L1C files for the sensor.
        start_time: A numpy.datetime64 object defining the start time of the
            time interval from which to extract pretraining samples.
        end_time: A numpy.datetime64 object defining the end time of the
            time interval from which to extract pretraining samples.
        output_path: The output path to which write the extracted scenes.
    """
    LOGGER = logging.getLogger(__name__)
    l1c_files = sorted(list(l1c_path.glob(f"**/{sensor.l1c_file_prefix}*.HDF5")))
    files = []
    for path in l1c_files:
        date_str = path.name.split(".")[4]
        date = datetime.strptime(date_str[:16], "%Y%m%d-S%H%M%S")
        date = pd.to_datetime(date)
        if (date >= start_time) and (date < end_time):
            files.append(path)

    pool = ProcessPoolExecutor(max_workers=n_processes)
    log_queue = get_log_queue()
    tasks = []
    for path in files:
        tasks.append(
            pool.submit(
                process_l1c_file,
                sensor,
                path,
                output_path,
                log_queue=log_queue
            )
        )
        tasks[-1].file = path

    with Progress(console=get_console()) as progress:
        pbar = progress.add_task(
            "Extracting pretraining data:",
            total=len(tasks)
        )
        for task in as_completed(tasks):
            log_messages()
            try:
                task.result()
                LOGGER.info(
                    f"Finished processing file %s.",
                    task.file
                )
            except Exception as exc:
                LOGGER.exception(
                    "The following error was encountered when processing file %s:"
                    "%s.",
                    task.file,
                    exc
                )
            progress.advance(pbar)


@click.argument("input_sensor")
@click.argument("target_sensor")
@click.argument("year", type=int)
@click.argument("month", type=int)
@click.argument("days", nargs=-1, type=int)
@click.argument("output_path")
@click.option("--n_processes", default=None, type=int)
@click.option("--scene_size", type=tuple, default=(64, 64))
@click.option("--radius_of_influence", default=100e3)
def cli(
        input_sensor: Sensor,
        target_sensor: Sensor,
        year: int,
        month: int,
        days: int,
        output_path: Path,
        n_processes: int,
        scene_size: Tuple[int, int] = (64, 64),
        radius_of_influence: float = 100e3
) -> None:
    """
    Extract pretraining data for SATFORMER training.

    Args:
        input_sensor: The name of the input sensor.
        target_sensor: The name of the target sensor.
        year: The year for which to extract the training data.
        month: The month for which to extract the training data.
        days: A list of the days of the month for which to extract the training data.
        output_path: The path to which to write the training data.
        n_processes: The number of processes to use for parallel processing
    """
    from gprof_nn import sensors

    # Check sensors
    input_sensor_obj = getattr(sensors, input_sensor.strip().upper(), None)
    if input_sensor_obj is None:
        LOGGER.error("The sensor '%s' is not known.", sensor)
        return 1
    input_sensor = input_sensor_obj
    target_sensor_obj = getattr(sensors, target_sensor.strip().upper(), None)
    if target_sensor_obj is None:
        LOGGER.error("The sensor '%s' is not known.", sensor)
        return 1
    target_sensor = target_sensor_obj

    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))

    output_path = Path(output_path)
    if not output_path.exists() or not output_path.is_dir():
        LOGGER.error("The 'output' argument must point to a directory.")
        return 1

    if n_processes is None:
        for day in track(days):
            start_time = datetime(year, month, day)
            end_time = datetime(year, month, day + 1)
            extract_samples(
                input_sensor,
                target_sensor,
                start_time,
                end_time,
                output_path=output_path,
                scene_size=scene_size,
                radius_of_influence=radius_of_influence
            )
    else:
        pool = ProcessPoolExecutor(max_workers=n_processes)
        tasks = []
        for day in days:
            start_time = datetime(year, month, day)
            end_time = datetime(year, month, day + 1)
            tasks.append(
                pool.submit(
                    extract_samples,
                    input_sensor,
                    target_sensor,
                    start_time,
                    end_time,
                    output_path=output_path,
                    scene_size=scene_size,
                    radius_of_influence=radius_of_influence
                )
            )


        with Progress() as progress:
            task_progress = progress.add_task(
                "[green]Running tasks...",
                total=len(days)
            )
            for task in tasks:
                task.result()
