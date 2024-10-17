"""
=========================
gprof_nn.data.pretraining
=========================
This module provides functionality to extract observation collocations between various sensors
of the GPM constellation and extract training samples suitable for training an observation
translator model.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
import re
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
    l1c_noaa20_atms,
    l1c_gcomw1_amsr2,
)
from pansat.utils import resample_data
from pyresample.geometry import SwathDefinition
from rich.progress import Progress, track
import torch
import xarray as xr

from gprof_nn.sensors import Sensor
from gprof_nn.data.utils import (
    save_scene,
    extract_scenes,
    run_preprocessor,
    upsample_data,
    add_cpcir_data,
    calculate_obs_properties,
    mask_invalid_values
)
from gprof_nn.data.l1c import L1CFile
from gprof_nn.logging import (
    configure_queue_logging,
    log_messages,
    get_log_queue,
    get_console
)


LOGGER = logging.getLogger(__name__)


# pansat products for each sensor.
PRODUCTS = {
    "gmi": (l1c_gpm_gmi,),
    "atms": (l1c_npp_atms, l1c_noaa20_atms),
    "amsr2": (l1c_gcomw1_amsr2,)
}

UPSAMPLING_FACTORS = {
    "gmi": (3, 1),
    "atms": (3, 3,),
    "amsr2": (1, 1)
}
RADIUS_OF_INFLUENCE = {
    "gmi": 20e3,
    "atms": 100e3,
    "amsr2": 10e3
}


def extract_pretraining_scenes(
        input_sensor: Sensor,
        target_sensor: Sensor,
        match: Tuple[Granule, Tuple[Granule]],
        output_path: Path,
        scene_size: Tuple[int, int],
) -> None:
    """
    Extract training scenes from a match-up of two GPM sensors.

    Args:
        input_sensor: A sensor object representing the sensor from which to extract the input data.
        target_sensor: A sensor object representing the sensor from which to extract the output data.
        match: A match object specifying a collocation of observations from the two sensors.
        output_path: The path to which to write the extracted training scenes.
        scene_size: The size of the training scenes.
    """
    input_granule, target_granules = match
    target_granules = merge_granules(sorted(list(target_granules)))
    for target_granule in target_granules:

        input_data = run_preprocessor(input_granule)
        mask_invalid_values(input_data)

        for var in input_data:
            if np.issubdtype(input_data[var].data.dtype, np.floating):
                invalid = input_data[var].data < -1_000
                input_data[var].data[invalid] = np.nan

        upsampling_factors = UPSAMPLING_FACTORS[input_sensor.name.lower()]
        if max(upsampling_factors) > 1:
            input_data = upsample_data(input_data, upsampling_factors)
        input_data = add_cpcir_data(input_data)

        rof_in = RADIUS_OF_INFLUENCE[input_sensor.name.lower()]
        rof_targ = RADIUS_OF_INFLUENCE[target_sensor.name.lower()]
        input_obs = calculate_obs_properties(input_data, input_granule, radius_of_influence=rof_in)
        target_obs = calculate_obs_properties(input_data, target_granule, radius_of_influence=rof_targ)


        training_data = xr.Dataset({
            "input_observations": input_obs.observations.rename(channels="input_channels"),
            "input_meta_data": input_obs.meta_data.rename(channels="input_channels"),
            "target_observations": target_obs.observations.rename(channels="target_channels"),
            "target_meta_data": target_obs.meta_data.rename(channels="target_channels"),
            "two_meter_temperature": input_data.two_meter_temperature,
            "total_column_water_vapor": input_data.total_column_water_vapor,
            "leaf_area_index": input_data.leaf_area_index,
            "land_fraction": input_data.land_fraction,
            "ice_fraction": input_data.ice_fraction,
            "elevation": input_data.elevation,
            "ir_observations": input_data.ir_observations,
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
            n_scans=128,
            n_pixels=128,
            overlapping=False,
            min_valid=(128 * 128) / 2.0,
            reference_var="valid",
        )
        LOGGER.info(
            "Extracted %s training scenes from %s.",
            len(scenes),
            input_granule
        )

        uint16_max = 2 ** 16 - 1
        encodings = {
            "input_observations": {"dtype": "uint16", "zlib": True, "scale_factor": 0.01, "_FillValue": uint16_max},
            "input_meta_data": {"dtype": "uint16", "zlib": True, "scale_factor": 0.01, "_FillValue": uint16_max},
            "target_observations": {"dtype": "uint16", "zlib": True, "scale_factor": 0.01, "_FillValue": uint16_max},
            "target_meta_data": {"dtype": "uint16", "zlib": True, "scale_factor": 0.01, "_FillValue": uint16_max},
            "two_meter_temperature": {"dtype": "uint16", "zlib": True, "scale_factor": 0.1, "_FillValue": uint16_max},
            "total_column_water_vapor": {"dtype": "float32", "zlib": True},
            "leaf_area_index": {"dtype": "float32", "zlib": True},
            "land_fraction": {"dtype": "int8", "zlib": True},
            "ice_fraction": {"dtype": "int8", "zlib": True},
            "elevation": {"dtype": "uint16", "zlib": True, "scale_factor": 0.5, "_FillValue": uint16_max},
            "ir_observations": {"dtype": "uint16", "zlib": True, "scale_factor": 0.01, "_FillValue": uint16_max},
        }

        scene_ind = 0
        for scene in scenes:
            scene = scene.drop_vars(["valid"])
            start_time = target_granule.time_range.start
            start_str = start_time.strftime("%Y%m%d%H%M%S")
            end_time = target_granule.time_range.end
            end_str = end_time.strftime("%Y%m%d%H%M%S")
            output_filename = f"{input_sensor.name.lower()}_{target_sensor.name.lower()}_{start_str}_{end_str}_{scene_ind:04}.nc"
            scene.to_netcdf(output_path / output_filename, encoding=encodings)
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
        mask = np.all(np.isnan(tbs), axis=(1, 2))
        tbs = training_data.target_observations.data
        tbs[tbs < 0] = np.nan
        n_seq_out = tbs.shape[0]

        input_data = {
            "observations": torch.tensor(training_data.input_observations.data)[None, None],
            "input_observation_mask": torch.tensor(mask, dtype=torch.bool)[None],
            "input_observation_props": torch.tensor(training_data.input_meta_data.data)[None].transpose(1, 2),
            "dropped_observation_props": torch.tensor(training_data.input_meta_data.data)[11:][None].transpose(1, 2),
            "output_observation_props": torch.tensor(training_data.target_meta_data.data)[None].transpose(1, 2),
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
) -> None:
    """
    Extract pretraining samples.

    Args:
        input_sensor: The sensor from which to extract the input data.
        target_sensor: The sensor from which to extract the target data.
        start_time: The start of the time period for which to extract training
            samples.
        end_time: The end of the data extraction period.
        output_path: The path to which to write the extracted samples.
        scene_size: The size of the training scenes to extract.
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
def cli(
        input_sensor: Sensor,
        target_sensor: Sensor,
        year: int,
        month: int,
        days: int,
        output_path: Path,
        n_processes: int,
        scene_size: Tuple[int, int] = (64, 64),
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
            )
    else:
        pool = ProcessPoolExecutor(max_workers=n_processes)
        tasks = []
        for day in days:
            start_time = datetime(year, month, day)
            end_time = datetime(year, month, day)
            tasks.append(
                pool.submit(
                    extract_samples,
                    input_sensor,
                    target_sensor,
                    start_time,
                    end_time,
                    output_path=output_path,
                    scene_size=scene_size,
                )
            )


        with Progress() as progress:
            task_progress = progress.add_task(
                "[green]Running tasks...",
                total=len(days)
            )
            for task in tasks:
                task.result()
