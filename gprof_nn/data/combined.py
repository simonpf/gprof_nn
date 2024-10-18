"""
======================
gprof_nn.data.combined
======================

This module provides functionality to extract collocations between GPM sensors
and GPM CMB.
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
from pansat.products.satellite.gpm import (
    l2b_corra2022_gpm_dprgmi_v07a,
    l2b_corra2022_gpm_dprgmi_v07b,
    l2b_corra2022_gpm_dprgmi_v07c,
)
from pyresample.geometry import SwathDefinition
from pansat.utils import resample_data
from rich.progress import track

from gprof_nn.sensors import Sensor
from gprof_nn.data.utils import (
    PANSAT_PRODUCTS,
    UPSAMPLING_FACTORS,
    RADIUS_OF_INFLUENCE,
    run_preprocessor,
    upsample_data,
    add_cpcir_data,
    calculate_obs_properties,
    extract_scenes,
    mask_invalid_values
)


LOGGER = logging.getLogger(__name__)


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


def extract_cmb_scenes(
        sensor: Sensor,
        match: Tuple[Granule, Tuple[Granule]],
        output_path: Path,
        scene_size: Tuple[int, int],
) -> None:
    """
    Extract training scenes between a GPM sensor and GPM CMB retrievals.

    Args:
        sensor: A sensor object representing the GPM sensor.
        match: A match object specifying the match between the GPM sensor and the GPM CMB
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

        cmb_data = target_granule.open()[[
            "estim_surf_precip_tot_rate",
            "precip_tot_water_cont",
            "precip_liq_water_cont",
            "cloud_ice_water_cont",
            "cloud_liq_water_cont",
            "scan_time"
        ]].rename({
            "estim_surf_precip_tot_rate": "surface_precip",
            "precip_tot_water_cont": "total_water_content",
            "precip_liq_water_cont": "rain_water_content",
            "cloud_ice_water_cont": "ice_water_content",
            "cloud_liq_water_cont": "cloud_liquid_water_content",
            "scan_time": "scan_time_cmb"
        })

        profiles = [
            "total_water_content",
            "rain_water_content",
            "ice_water_content",
            "cloud_liquid_water_content"
        ]
        for var in profiles:
            var_data = cmb_data[var]
            var_data = var_data.where(var_data > -9000)
            if "vertical_bins" in var_data.dims:
                var_data = var_data.ffill("vertical_bins")
            var_data.data[:] = var_data.data[..., ::-1]
            cmb_data[var] = var_data.astype(np.float32)

        swc = cmb_data.total_water_content - cmb_data.rain_water_content
        cmb_data["snow_water_content"] = swc
        cmb_data = cmb_data.drop_vars("total_water_content")

        cmb_data["snow_water_path"] = swc.integrate("vertical_bins")
        cmb_data["rain_water_path"] = cmb_data["rain_water_content"].integrate("vertical_bins")
        cmb_data["ice_water_path"] = cmb_data["ice_water_content"].integrate("vertical_bins")
        cmb_data["cloud_liquid_water_path"] = cmb_data["cloud_liquid_water_content"].integrate("vertical_bins")

        target_levels = np.concatenate([0.5 * np.arange(20) + 0.25, 10.5 + np.arange(8)])
        cmb_data = cmb_data.interp(vertical_bins=target_levels).reset_coords(("scan_time_cmb"))
        cmb_data["scan_time_cmb"] = cmb_data["scan_time_cmb"].astype(np.int64)
        cmb_data_r = resample_data(
            cmb_data,
            swath,
            radius_of_influence=5e3,
            new_dims=(("scans", "pixels"))
        )
        cmb_data_r["scan_time_cmb"] = cmb_data_r["scan_time_cmb"].astype("datetime64[ns]")

        input_data["surface_precip"] = (
            ("scans", "pixels"),
            cmb_data_r["surface_precip"].data
        )

        sp = input_data.surface_precip.data
        sp[sp < 0] = np.nan
        input_data["scan_time_cmb"] = cmb_data_r["scan_time_cmb"]

        profiles = [
            "rain_water_content",
            "snow_water_content",
            "ice_water_content",
            "cloud_liquid_water_content"
        ]
        for profile in profiles:
            input_data[profile] = cmb_data_r[profile]
            path_name = profile.replace("content", "path")
            input_data[path_name] = cmb_data_r[path_name]

        time_diff = np.abs(input_data.scan_time - input_data.scan_time_cmb)
        input_data.surface_precip.data[time_diff.data > np.timedelta64(15, "m")] = np.nan

        input_data["input_observations"] = input_obs.observations.rename({"channels": "all_channels"})
        input_data["input_meta_data"] = input_obs.meta_data.rename({"channels": "all_channels"})
        mask_invalid_values(input_data)

        scenes = extract_scenes(
            input_data,
            n_scans=128,
            n_pixels=128,
            overlapping=True,
            min_valid=100,
            reference_var="surface_precip",
            offset=32
        )
        LOGGER.info(
            "Extracted %s training scenes from %s.",
            len(scenes),
            input_granule
        )

        uint16_max = 2 ** 16 - 1
        encodings = {
            "scan_time": {"zlib": True},
            "scan_time_cmb": {"zlib": True},
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
                "ice_water_content",
                "cloud_liquid_water_content",
                "rain_water_path",
                "snow_water_path",
                "ice_water_path",
                "cloud_liquid_water_path",
                "surface_precip",
        ]:
            encodings[var] = {"dtype": "float32", "zlib": True}

        scene_ind = 0
        for scene in scenes:
            start_time = target_granule.time_range.start
            start_str = start_time.strftime("%Y%m%d%H%M%S")
            end_time = target_granule.time_range.end
            end_str = end_time.strftime("%Y%m%d%H%M%S")
            output_filename = f"{sensor.name.lower()}_cmb_{start_str}_{end_str}_{scene_ind:04}.nc"
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
    Extract GPM-CMB training scenes.

    Args:
        sensor: The GPM sensor for which to extract training scenes.
        start_time: The begining of the time period for which to extract training data.
        end_time: The end of the time period for which to extract training data.
        output_path: The path to which to write the extracted training scenes.
        scene_size: The size of the training scenes to extract.
    """
    input_products = PANSAT_PRODUCTS[sensor.name.lower()]
    target_product = l2b_corra2022_gpm_dprgmi_v07a
    for input_product in input_products:
            input_recs = input_product.get(TimeRange(start_time, end_time))
            input_index = Index.index(input_product, input_recs)
            target_recs = target_product.get(TimeRange(start_time, end_time))
            target_index = Index.index(target_product, target_recs)
            matches = find_matches(input_index, target_index, np.timedelta64(15, "m"))
            for match in matches:
                extract_cmb_scenes(
                    sensor,
                    match,
                    output_path,
                    scene_size=scene_size,
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
    Extract CMB scenes data for SATFORMER training.

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
