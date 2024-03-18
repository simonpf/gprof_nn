"""
gprof_nn.data.era5
==================

This module provides functionality for accessing ERA5 data.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
from multiprocessing import Queue
from pathlib import Path
from typing import Union, Optional

import click
import numpy as np
import pandas as pd
from rich.progress import Progress
import xarray as xr

from gprof_nn.config import CONFIG
from gprof_nn import sensors
from gprof_nn.definitions import DATA_SPLIT
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.preprocessor import run_preprocessor
from gprof_nn.logging import get_console, log_messages
from gprof_nn.data.utils import (
    write_training_samples_1d,
    write_training_samples_3d
)


LOGGER = logging.getLogger(__name__)


def load_era5_data(
        start_time: Union[datetime, np.datetime64],
        end_time: Union[datetime, np.datetime64]
) -> xr.Dataset:
    """
    Loads ERA5 data matching the start and end time of a L1C
    file.

    This assumes that ERA5 data is available in hourly files and sorted
    into folder 'YYYYmm/ERA5_YYYYmmdd_surf.nc' where Y is the year, m
    the month and d the day.

    Args:
        start_time: First scan time from L1C file.
        end_time: Last scan time from L1C file.

    Rerturn:
        An xarray.Dataset containing the loaded ERA5 data.
    """
    base_directory = CONFIG.data.era5_path
    if base_directory is None:
        raise RuntimeError(
            "The 'data.era5_path' must be set to load ER5 data."
        )

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    year_start = start_time.year
    month_start = start_time.month
    day_start = start_time.day

    year_end = end_time.year
    month_end = end_time.month
    day_end = end_time.day

    file_start = (
        base_directory /
        f"{year_start:04}{month_start:02}" /
        f"ERA5_{year_start:04}{month_start:02}{day_start:02}_surf.nc"
    )
    file_end = (
        base_directory /
        f"{year_end:04}{month_end:02}" /
        f"ERA5_{year_end:04}{month_end:02}{day_end:02}_surf.nc"
    )

    data_start = xr.load_dataset(file_start)
    if file_start == file_end:
        return data_start

    data_end = xr.load_dataset(file_end)
    return xr.concat([data_start, data_end], dim="time")


def add_era5_precip(input_data, era5_data):
    """
    Adds total precipitation from ERA5 to preprocessor data.

    Args:
        input_data: The preprocessor data to which the atmospheric data
            from the sim file has been added. Must contain "surface_precip"
            variable.
        era5_data: The era5 data covering the time range of observations
            in input data.
    """
    l_0 = era5_data[{"longitude": slice(0, 1)}].copy(deep=True)
    l_0 = l_0.assign_coords({"longitude": [360.0]})
    era5_data = xr.concat([era5_data, l_0], "longitude")
    n_scans = input_data.scans.size
    n_pixels = input_data.pixels.size

    surface_types = input_data["surface_type"].data
    indices = (surface_types == 2) + (surface_types == 16)

    lats = xr.DataArray(input_data["latitude"].data[indices], dims="samples")
    lons = input_data["longitude"].data[indices]
    lons = np.where(lons < 0.0, lons + 360, lons)
    lons = xr.DataArray(lons, dims="samples")
    time = np.broadcast_to(
        input_data["scan_time"].data.reshape(-1, 1), (n_scans, n_pixels)
    )[indices]
    time = xr.DataArray(time, dims="samples")
    # Interpolate and convert to mm/h
    total_precip = era5_data["tp"].interp(
        {"latitude": lats, "longitude": lons, "time": time}, method="nearest"
    )

    if not "surface_precip" in input_data:
        surface_precip = np.nan * np.ones((n_scans, n_pixels), np.float32)
        convective_precip = np.nan * np.ones((n_scans, n_pixels), np.float32)
        input_data["surface_precip"] = (("scans", "pixels"), surface_precip)
        input_data["convective_precip"] = (("scans", "pixels"), convective_precip)

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


def process_l1c_file(
        sensor: sensors.Sensor,
        l1c_file: Path,
        output_path_1d: Optional[Path] = None,
        output_path_3d: Optional[Path] = None,
        log_queue=None
):
    """
    Match L1C file with ERA5 surface and convective precipitation for
    sea-ice and sea-ice-edge surfaces and extract training samples.

    Args:
        sensor: Sensor object defining the sensor for which to process the
            L1C files.
        l1c_file: Path to the L1C file which to match with ERA5 precip.
        output_path_1d: Path pointing to the folder to which to write
            the training samples for the GPROF-NN 1D retrieval.
        output_path_3d: Path pointing to the folder to which to write
            the training samples for the GPROF-NN 3D retrieval.
    """
    import gprof_nn.logging

    if log_queue is not None:
        gprof_nn.logging.configure_queue_logging(log_queue)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Starting processing L1C file %s.", l1c_file)

    data_pp = run_preprocessor(
        l1c_file, sensor=sensor, robust=False
    )

    # Drop unneeded variables.
    drop = ["sunglint_angle", "quality_flag", "wet_bulb_temperature", "lapse_rate"]
    data_pp = data_pp.drop_vars(drop)

    start_time = data_pp["scan_time"].data[0]
    end_time = data_pp["scan_time"].data[-1]
    era5_data = load_era5_data(start_time, end_time)
    add_era5_precip(data_pp, era5_data)

    data_pp.attrs["source"] = "era5"
    if output_path_1d is not None:
        write_training_samples_1d(
            output_path_1d,
            "mrms",
            data_pp,
        )
    if output_path_3d is not None:
        n_pixels = 64
        n_scans = 128
        write_training_samples_3d(
            output_path_3d,
            "mrms",
            data_pp,
            n_scans=n_scans,
            n_pixels=n_pixels,
            overlapping=True,
            min_valid=50,
            reference_var="surface_precip"
        )


def process_l1c_files(
        sensor: sensors.Sensor,
        l1c_path: Path,
        start_time: np.datetime64,
        end_time: np.datetime64,
        output_path_1d: Optional[Path] = None,
        output_path_3d: Optional[Path] = None,
        split: str = None,
        n_processes: int = 4,
        log_queue: Optional[Queue] = None
):
    """
    Extract ERA5 training samples for all L1C files in a given time interval.

    Args:
        sensor: Sensor object defining the sensor for which to process the
            L1C files.
        l1c_file: Path to the L1C file which to match with ERA5 precip.
        output_path_1d: Path pointing to the folder to which to write
            the training samples for the GPROF-NN 1D retrieval.
        output_path_3d: Path pointing to the folder to which to write
            the training samples for the GPROF-NN 3D retrieval.
        split: An optional string specifying whether to extract only data from
            one of the three data splits ['training', 'validation', 'test'].
        n_processes: The number of processes to use for parallel
            processing.
        log_queue: Queue to use for logging from sub-processes.
    """
    import gprof_nn.logging

    if log_queue is not None:
        gprof_nn.logging.configure_queue_logging(log_queue)
    LOGGER = logging.getLogger(__name__)

    time = start_time
    l1c_files = []

    LOGGER.info("Looking for files in %s.", l1c_path)
    while time < end_time:

        l1c_files_day = L1CFile.find_files(time, l1c_path, sensor=sensor)
        # Check if day of month should be skipped.
        if split is not None:
            days = DATA_SPLIT[split]
            l1c_files_split = []
            for l1c_file in l1c_files:
                time = L1CFile(l1c_file).start_time
                day_of_month = int(
                    (time - time.astype("datetime64[M]")).astype("timedelta64[D]").astype("int64")
                )
                if day_of_month + 1 in days:
                    l1c_files_split.append(l1c_file)
            l1c_files_day = l1c_files_split

        l1c_files += l1c_files_day
        time += np.timedelta64(24 * 60 * 60, "s")

    LOGGER.info("Found %s L1C files to process", len(l1c_files))


    pool = ProcessPoolExecutor(max_workers=n_processes)
    tasks = []
    for l1c_file in l1c_files:
        tasks.append(
            pool.submit(
                process_l1c_file,
                sensor,
                l1c_file.path,
                output_path_1d,
                output_path_3d,
            )
        )
        tasks[-1].l1c_file = l1c_file

    with Progress(console=get_console()) as progress:
        pbar = progress.add_task(
            "Extracting ERA5 collocations:",
            total=len(tasks)
        )
        for task in as_completed(tasks):
            log_messages()
            try:
                task.result()
                LOGGER.info(f"""
                Finished processing file {task.l1c_file}.
                """)
            except Exception as exc:
                LOGGER.exception(
                    "The following error was encountered when processing file %s:"
                    "%s.",
                    task.l1c_file,
                    exc
                )
            progress.advance(pbar)


@click.argument("sensor")
@click.argument("l1c_file_path")
@click.argument("start_time")
@click.argument("end_time")
@click.argument("output_1d")
@click.argument("output_3d")
@click.option("--n_processes", default=4)
def cli(
        sensor: sensors.Sensor,
        l1c_file_path: Path,
        start_time: np.datetime64,
        end_time: np.datetime64,
        output_1d: Path,
        output_3d: Path,
        n_processes: int = 4
) -> None:
    """
    Extract training data from L1C/ERA5 collocations.

    Args:
        sensor: A sensor object representing the sensor for which to extract
            the training data.
        l1c_file_path: Path to the folder containing the L1C files from which to
            extract the training data.
        start_time: Start time of the time interval limiting the l1c files from
            which training scenes will be extracted.
        end_time: End time of the time interval limiting the L1C files from
            which training scenes will be extracted.
        output_1d: Path pointing to the folder to which the 1D training data
            should be written.
        output_3d: Path pointing to the folder to which the 3D training data
            should be written.
        n_processes: The number of processes to use for data extraction.
    """
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

    process_l1c_files(
        sensor,
        l1c_file_path,
        start_time,
        end_time,
        output_path_1d,
        output_path_3d,
        n_processes=n_processes
    )
