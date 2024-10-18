"""
======================
gprof_nn.data.cloudsat
======================

This module provides functionality to extract collocations between GPM sensors
and CloudSat.
"""
from calendar import monthrange
from datetime import datetime
import logging
from pathlib import Path
from typing import Tuple

import click

import numpy as np
from pansat import Granule, TimeRange
from pansat.granule import merge_granules
from pansat.catalog import Index
from pansat.catalog.index import find_matches
from pansat.products.satellite.gpm import (
    l1c_gpm_gmi, l1c_npp_atms, l1c_noaa20_atms, l1c_gcomw1_amsr2
)
from pansat.products.satellite.cloudsat import (
    l2c_precip_column,
    l2c_rain_profile,
    l2c_snow_profile
)
from pyresample.geometry import SwathDefinition
from pansat.utils import resample_data
from rich.progress import track

from gprof_nn.sensors import Sensor
from gprof_nn.data.utils import (
    run_preprocessor,
    upsample_data,
    add_cpcir_data,
    calculate_obs_properties,
    extract_scenes,
    mask_invalid_values
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


def extract_cloudsat_scenes(
        sensor: Sensor,
        match: Tuple[Granule, Tuple[Granule]],
        output_path: Path,
        scene_size: Tuple[int, int],
) -> None:
    """
    Extract training scenes between a GPM sensor and CloudSat observations.

    Args:
        sensor: A sensor object representing the GPM sensor.
        match: A match object specifying the match between the GPM sensor and the CloudSat
            retrievals.
        output_path: The path to which to write the extracted training scenes.
        scene_size: The size of the training scenes to extract.
    """
    input_granule, target_granules = match
    target_granules = merge_granules(sorted(list(target_granules)))

    for target_granule in target_granules:
        input_data = run_preprocessor(input_granule)
        for var in input_data:
            if np.issubdtype(input_data[var].data.dtype, np.floating):
                invalid = input_data[var].data < -1_000
                input_data[var].data[invalid] = np.nan

        upsampling_factors = UPSAMPLING_FACTORS[sensor.name.lower()]
        if max(upsampling_factors) > 1:
            input_data = upsample_data(input_data, upsampling_factors)
        input_data = add_cpcir_data(input_data)

        lons = input_data.longitude.data
        lats = input_data.latitude.data
        swath = SwathDefinition(lats=lats, lons=lons)

        rof_in = RADIUS_OF_INFLUENCE[sensor.name.lower()]
        input_obs = calculate_obs_properties(input_data, input_granule, radius_of_influence=rof_in)

        cs_data = target_granule.open().reset_coords("time")

        precip_column_rec = l2c_precip_column.get(target_granule.time_range)
        if len(precip_column_rec) == 0:
            raise ValueError(
                "No 2C-PRECIP-COLUMN record for granule %s.", target_granule
            )
        precip_column_data = l2c_precip_column.open(precip_column_rec[0])[{
            "rays": slice(*target_granule.primary_index_range)
        }].reset_index("rays").reset_coords("time")
        cs_data["precip_flag"] = precip_column_data["precip_flag"]

        snow_profile_rec = l2c_snow_profile.get(target_granule.time_range)
        if len(snow_profile_rec) == 0:
            raise ValueError(
                "No 2C-SNOW-PROFILE record for granule %s.", target_granule
            )
        snow_profile_data = l2c_snow_profile.open(snow_profile_rec[0])[{
            "rays": slice(*target_granule.primary_index_range)
        }].reset_index("rays").reset_coords("time")
        cs_data["surface_precip_snow"] = snow_profile_data["surface_precip"]

        levels = np.concatenate([0.5 + np.arange(20) * 0.5, np.arange(10.5, 18.0)])
        profiles_interp = {}
        profiles = ["cloud_liquid_water_content", "rain_water_content", "snow_water_content"]
        for ray in range(cs_data.rays.size):
            cs_data_r = cs_data[{"rays": ray}]
            z = cs_data_r.height.data
            for profile in profiles:
                prf = np.flip(cs_data_r[profile].data).astype("float32")
                profiles_interp.setdefault(profile, []).append(
                    np.interp(levels * 1e3, np.flip(z), prf)
                )
        profiles = {name: np.stack(arrays) for name, arrays in profiles_interp.items()}
        for name, array in profiles.items():
            cs_data[name] = (("rays", "levels"), array)
        cs_data["levels"] = (("levels",), levels)
        cs_data = cs_data.drop_dims("bins")

        cs_data_r = resample_data(cs_data.transpose("rays", "levels"), swath, new_dims=(("scans", "pixels")))

        input_data["surface_precip"] = (
            ("scans", "pixels"),
            cs_data_r["surface_precip"].data
        )
        sp = input_data.surface_precip.data
        sp[sp < 0] = np.nan

        input_data["surface_precip_snow"] = (
            ("scans", "pixels"),
            cs_data_r["surface_precip_snow"].data
        )
        sp = input_data.surface_precip_snow.data
        sp[sp < 0] = np.nan

        for profile in profiles:
            input_data[profile] = (
                ("scans", "pixels", "levels"),
                cs_data_r[profile].bfill("levels").data
            )
            prf = input_data[profile].data
            prf[prf < 0] = np.nan
            path_name = profile.replace('content', 'path')
            input_data[path_name] = input_data[profile].integrate("levels")

        input_data["levels"] = (("levels",), levels)
        input_data["scan_time_cloudsat"] = cs_data_r["time"]

        time_diff = input_data.scan_time - input_data.scan_time_cloudsat
        valid = (
            (np.abs(time_diff.data) < np.timedelta64(15, "m")) *
            (np.isfinite(input_data.surface_precip.data) + np.isfinite(input_data.surface_precip_snow.data))
        )
        valid_field = np.ones_like(valid, dtype=np.float32)
        valid_field[~valid] = np.nan
        input_data["valid"] = (("scans", "pixels"), valid_field)

        input_data["input_observations"] = input_obs.observations.rename({"channels": "all_channels"})
        input_data["input_meta_data"] = input_obs.meta_data.rename({"channels": "all_channels"})
        mask_invalid_values(input_data)

        scenes = extract_scenes(
            input_data,
            n_scans=128,
            n_pixels=128,
            overlapping=True,
            min_valid=100,
            reference_var="valid",
            offset=50
        )
        LOGGER.info(
            "Extracted %s training scenes from %s.",
            len(scenes),
            input_granule
        )

        uint16_max = 2 ** 16 - 1
        encodings = {
            "scan_time": {"zlib": True},
            "scan_time_cloudsat": {"zlib": True},
            "brightness_temperatures": {"dtype": "uint16", "zlib": True, "scale_factor": 0.01, "_FillValue": uint16_max},
            "input_observations": {"dtype": "uint16", "zlib": True, "scale_factor": 0.01, "_FillValue": uint16_max},
            "input_meta_data": {"dtype": "uint16", "zlib": True, "scale_factor": 0.01, "_FillValue": uint16_max},
            "two_meter_temperature": {"dtype": "uint16", "zlib": True, "scale_factor": 0.1, "_FillValue": uint16_max},
            "total_column_water_vapor": {"dtype": "float32", "zlib": True},
            "leaf_area_index": {"dtype": "float32", "zlib": True},
            "land_fraction": {"dtype": "int8", "zlib": True, "_FillValue": -1},
            "ice_fraction": {"dtype": "int8", "zlib": True, "_FillValue": -1},
            "elevation": {"dtype": "uint16", "zlib": True, "scale_factor": 0.5, "_FillValue": uint16_max},
            "ir_observations": {"dtype": "uint16", "zlib": True, "scale_factor": 0.01, "_FillValue": uint16_max},
        }
        for var in [
                "rain_water_content",
                "snow_water_content",
                "cloud_liquid_water_content",
                "rain_water_path",
                "snow_water_path",
                "cloud_liquid_water_path",
                "surface_precip",
        ]:
            encodings[var] = {"dtype": "float32", "zlib": True}

        scene_ind = 0
        for scene in scenes:
            scene = scene.drop_vars("valid")
            start_time = target_granule.time_range.start
            start_str = start_time.strftime("%Y%m%d%H%M%S")
            end_time = target_granule.time_range.end
            end_str = end_time.strftime("%Y%m%d%H%M%S")
            output_filename = f"{sensor.name.lower()}_cloudsat_{start_str}_{end_str}_{scene_ind:04}.nc"
            scene.to_netcdf(output_path / output_filename, encoding=encodings)
            scene_ind += 1


def extract_samples(
        sensor: Sensor,
        start_time: np.datetime64,
        end_time: np.datetime64,
        output_path: Path,
        scene_size: Tuple[int, int] = (64, 64),
) -> None:
    """
    Extract GPM-CloudSat training scenes.

    Args:
        sensor: The GPM sensor for which to extract training scenes.
        start_time: The begining of the time period for which to extract training data.
        end_time: The end of the time period for which to extract training data.
        output_path: The path to which to write the extracted training scenes.
        scene_size: The size of the training scenes to extract.
    """
    input_products = PRODUCTS[sensor.name.lower()]
    target_product = l2c_rain_profile
    for input_product in input_products:
            input_recs = input_product.get(TimeRange(start_time, end_time))
            input_index = Index.index(input_product, input_recs)
            target_recs = target_product.get(TimeRange(start_time, end_time))
            target_index = Index.index(target_product, target_recs)
            matches = find_matches(input_index, target_index, np.timedelta64(15, "m"))
            for match in matches:
                try:
                    extract_cloudsat_scenes(
                        sensor,
                        match,
                        output_path,
                        scene_size=scene_size,
                    )
                except Exception:
                    LOGGER.exception(
                        "Encountered an error processing granule %s.",
                        match[0]
                    )

@click.argument("sensor")
@click.argument("year", type=int)
@click.argument("month", type=int)
@click.argument("days", nargs=-1, type=int)
@click.argument("output_path")
@click.option("--n_processes", default=None, type=int)
@click.option("--scene_size", type=tuple, default=(64, 64))
def cli(
        sensor: Sensor,
        year: int,
        month: int,
        days: int,
        output_path: Path,
        n_processes: int,
        scene_size: Tuple[int, int] = (64, 64),
) -> None:
    """
    Extract CloudSat scenes data for SATFORMER training.

    Args:
        sensor: The name of the GPM sensor.
        year: The year for which to extract the training data.
        month: The month for which to extract the training data.
        days: A list of the days of the month for which to extract the training data.
        output_path: The path to which to write the training data.
        n_processes: The number of processes to use for parallel processing
    """
    from gprof_nn import sensors

    # Check sensors
    sensor_obj = getattr(sensors, sensor.strip().upper(), None)
    if sensor_obj is None:
        LOGGER.error("The sensor '%s' is not known.", sensor)
        return 1
    sensor = sensor_obj

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
                sensor,
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
                    start_time,
                    end_time,
                    output_path=output_path,
                    scene_size=scene_size,
                )
            )


        with Progress() as progress:
            task_progress = progress.add_task(
                "[green]Extracting training data:",
                total=len(days)
            )
            for task in tasks:
                task.result()
