"""
=========================
gprof_nn.data.pretraining
=========================

Functionality to extract pretraining data.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from rich.progress import Progress

from gprof_nn.sensors import Sensor
from gprof_nn.data.preprocessor import run_preprocessor
from gprof_nn.data.utils import save_scene, extract_scenes
from gprof_nn.logging import (
    configure_queue_logging,
    log_messages,
    get_log_queue,
    get_console
)


LOGGER = logging.getLogger(__name__)


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
        overlapping=False,
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


@click.argument("sensor")
@click.argument("l1c_file_path")
@click.argument("start_time")
@click.argument("end_time")
@click.argument("output_path")
@click.option("--n_processes", default=8)
def cli(
        sensor: Sensor,
        l1c_file_path: Path,
        start_time: np.datetime64,
        end_time: np.datetime64,
        output_path: Path,
        n_processes: int
) -> None:
    """
    Extract pretraining data from L1C files.

    Args:
        sensor: A sensor object representing the sensor for which to extract
            the training data.
        l1c_file_path: Path to the folder containing the L1c files from which to
            extract the pretraining training data.
        start_time: Start time of the time interval limiting the L1C files from
            which training scenes will be extracted.
        end_time: End time of the time interval limiting the L1C files from
            which training scenes will be extracted.
        output_path: Path pointing to the folder to which the pretraining files
            should be written.
    """
    from gprof_nn import sensors

    # Check sensor
    sensor_obj = getattr(sensors, sensor.strip().upper(), None)
    if sensor_obj is None:
        LOGGER.error("The sensor '%s' is not known.", sensor)
        return 1
    sensor = sensor_obj

    l1c_file_path = Path(l1c_file_path)
    if not l1c_file_path.exists() or not l1c_file_path.is_dir():
        LOGGER.error("The 'l1c_file_path' argument must point to a directory.")
        return 1

    output_path = Path(output_path)
    if not output_path.exists() or not output_path.is_dir():
        LOGGER.error("The 'output' argument must point to a directory.")
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

    process_l1c_files(
        sensor,
        l1c_file_path,
        start_time,
        end_time,
        output_path,
        n_processes=n_processes
    )
    return 0
