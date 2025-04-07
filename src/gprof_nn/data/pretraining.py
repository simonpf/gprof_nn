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
from datetime import datetime, timedelta
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
    l1c_r_gpm_gmi,
    l1c_npp_atms,
    l1c_noaa20_atms,
    l1c_gcomw1_amsr2,
    l1c_xcal2016v_noaa19_mhs_v07a,
    l1c_xcal2016v_noaa18_mhs_v07a,
    l1c_xcal2019v_metopc_mhs_v07a,
)
from pansat.utils import resample_data
from pytorch_retrieve import load_model
from pytorch_retrieve.inference import run_inference
from pytorch_retrieve.config import (
    OutputConfig,
    RetrievalOutputConfig,
    InferenceConfig
)
from pyresample.geometry import SwathDefinition
from rich.progress import Progress, track
import torch
import xarray as xr

from gprof_nn.definitions import ANCILLARY_VARIABLES
from gprof_nn.sensors import Sensor
from gprof_nn.data.utils import (
    save_scene,
    extract_scenes,
    run_preprocessor,
    upsample_data,
    add_cpcir_data,
    calculate_obs_properties,
    mask_invalid_values,
    RADIUS_OF_INFLUENCE,
    UPSAMPLING_FACTORS,
    PANSAT_PRODUCTS
)
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.training_data import transform_observations_satformer
from gprof_nn.logging import (
    configure_queue_logging,
    log_messages,
    get_log_queue,
    get_console
)


LOGGER = logging.getLogger(__name__)


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
        input_data = mask_invalid_values(input_data)

        upsampling_factors = UPSAMPLING_FACTORS[input_sensor.name.lower()]
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
            "ir_observations": input_data.ir_observations,
        })
        for var in ANCILLARY_VARIABLES:
            training_data[var] = input_data[var]

        tbs = training_data.input_observations.data
        tbs[tbs < 0] = np.nan
        valid = np.isfinite(tbs).any(0)
        tbs = training_data.target_observations.data
        tbs[tbs < 0] = np.nan
        valid *= np.isfinite(tbs).any(0)
        training_data["valid"] = (("scans", "pixels"), np.zeros_like(valid, dtype="float32"))

        scan_time_input = input_obs.scan_time
        scan_time_target = input_obs.scan_time
        time_diff = scan_time_input - scan_time_target
        valid *= np.abs(time_diff.data) < np.timedelta64(15, "m")

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
            "land_fraction": {"dtype": "int8", "zlib": True, "_FillValue": -1},
            "ice_fraction": {"dtype": "int8", "zlib": True, "_FillValue": -1},
            "elevation": {"dtype": "uint16", "zlib": True, "scale_factor": 0.5, "_FillValue": uint16_max},
            "ir_observations": {"dtype": "uint16", "zlib": True, "scale_factor": 0.01, "_FillValue": uint16_max},
        }

        for var in training_data:
            print(var, training_data[var].dtype)

        scene_ind = 0
        for scene in scenes:
            scene = scene.drop_vars(["valid"])
            meta = scene["input_meta_data"].data
            meta[meta < 0] = np.nan
            meta = scene["target_meta_data"].data
            meta[meta < 0] = np.nan
            start_time = target_granule.time_range.start
            start_str = start_time.strftime("%Y%m%d%H%M%S")
            end_time = target_granule.time_range.end
            end_str = end_time.strftime("%Y%m%d%H%M%S")
            output_filename = f"{input_sensor.name.lower()}_{target_sensor.name.lower()}_{start_str}_{end_str}_{scene_ind:04}.nc"
            scene.to_netcdf(output_path / output_filename, encoding=encodings)
            scene_ind += 1


class InputLoader:
    """
    Inputloader class to run a GPROF-NN Simulator model on GPM match-ups.
    """
    def __init__(
            self,
            inputs: List[Any],
            radius_of_influence: float = 100e3
    ):
        """
        Args:
            inputs: A match tuple containing the matched granules from two sensors of the GPM
                constellation.
        """
        self.inputs = inputs

    def __len__(self) -> int:
        """
        The number of match-ups to process.
        """
        return len(self.inputs)

    def __getitem__(self, index: int) -> int:
        """
        Return retrieval input data.
        """
        return self.load_data(index)

    def load_data(self, ind: int) -> Tuple[Dict[str, torch.Tensor], str, xr.Dataset]:
        """
        Load data from match-up.

        Args:
             ind: The index of the match-up to load the data from.

        Return:
            A tuple ``x, aux, filename`` returning a dictionary containing the retrieval input ``x``,
            auxiliary data ``aux``, and a output filename ``filename``.
        """
        inpt_granule, target_granules = self.inputs[ind]
        target_granule = sorted(list(target_granules))[0]

        inpt_file = L1CFile(inpt_granule.file_record.local_path)
        inpt_sensor = inpt_file.sensor
        targ_file = L1CFile(target_granule.file_record.local_path)
        targ_sensor = targ_file.sensor

        inpt_data = run_preprocessor(inpt_granule)
        upsampling_factors = UPSAMPLING_FACTORS[inpt_sensor.name.lower()]
        inpt_data = upsample_data(inpt_data, upsampling_factors)
        inpt_data = add_cpcir_data(inpt_data)
        roi_inpt = RADIUS_OF_INFLUENCE[inpt_sensor.name.lower()]
        inpt_observations = calculate_obs_properties(
            inpt_data,
            inpt_granule,
            radius_of_influence=roi_inpt
        )
        roi_targ = RADIUS_OF_INFLUENCE[targ_sensor.name.lower()]
        target_observations = calculate_obs_properties(
            inpt_data,
            target_granule,
            radius_of_influence=roi_targ
        )

        lons = inpt_data.longitude.data
        valid = np.isfinite(inpt_data.longitude.data)

        inpt_obs = inpt_observations.observations.data
        inpt_meta = inpt_observations.meta_data.data

        obs_in = []
        meta_in = []
        for ind, obs in enumerate(inpt_obs):
            obs[..., ~valid] = np.nan
            valid = np.isfinite(obs)
            mean = np.mean(obs[valid])
            std = np.std(obs[valid])
            obs_n = (obs - mean) / std
            obs = np.stack([
                np.ones_like(obs_n) * mean,
                np.ones_like(obs_n) * std,
                obs_n
            ])
            obs_in.append(torch.tensor(obs))
            meta = inpt_meta[ind]
            meta[..., ~valid] = np.nan
            meta_in.append(torch.tensor(inpt_meta[ind]))


        obs_in = torch.stack(obs_in, 1)[None]
        meta_in = torch.stack(meta_in, 1)[None]
        obs_in_mask = torch.isnan(obs_in).all(1).all(-1).all(-1)

        inpt = {
            "observations": obs_in,
            "input_observation_props": meta_in,
            "input_observation_mask": obs_in_mask,
        }

        anc_vars = [
            "two_meter_temperature",
            "total_column_water_vapor",
            "leaf_area_index",
            "land_fraction",
            "ice_fraction",
            "elevation",
            "ir_observations",
        ]
        for anc_var in anc_vars:
            anc_data = torch.tensor(inpt_data[anc_var].data).to(dtype=torch.float32)
            if anc_data.dim() < 3:
                anc_data = anc_data[None]
            anc_data = anc_data[None, :, None]
            anc_mask = torch.isnan(anc_data).all()[None, None]
            inpt[anc_var] = anc_data
            inpt[anc_var + "_mask"] = anc_mask

        props = torch.tensor(target_observations["meta_data"].data)[None]
        inpt["output_observation_props"] = props.transpose(1, 2)

        training_data = xr.Dataset({
            "latitude": inpt_data.latitude,
            "longitude": inpt_data.longitude,
            "input_observations": inpt_observations.observations.rename(channels="input_channels"),
            "input_meta_data": inpt_observations.meta_data.rename(channels="input_channels"),
            "target_observations": target_observations.observations.rename(channels="target_channels"),
            "target_meta_data": target_observations.meta_data.rename(channels="target_channels"),
        })

        filename = "match_" + target_granule.time_range.start.strftime("%Y%m%d%H%M%s") + ".nc"

        return inpt, filename, training_data


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

    input_products = PANSAT_PRODUCTS[input_sensor.name.lower()]
    target_products = PANSAT_PRODUCTS[target_sensor.name.lower()]
    for input_product in input_products:
        for target_product in target_products:
            input_recs = input_product.get(TimeRange(start_time, end_time))
            input_index = Index.index(input_product, input_recs)
            target_recs = target_product.get(TimeRange(start_time, end_time))
            target_index = Index.index(target_product, target_recs)
            matches = find_matches(input_index, target_index, np.timedelta64(15, "m"))
            for match in matches:
                try:
                    extract_pretraining_scenes(
                        input_sensor,
                        target_sensor,
                        match,
                        output_path,
                        scene_size=scene_size,
                    )
                except Exception:
                    LOGGER.exception(
                        "Encountered an error when extracting training data for match %s",
                        match[0]
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
        LOGGER.error("The sensor '%s' is not known.", input_sensor)
        return 1
    input_sensor = input_sensor_obj
    target_sensor_obj = getattr(sensors, target_sensor.strip().upper(), None)
    if target_sensor_obj is None:
        LOGGER.error("The sensor '%s' is not known.", target_sensor)
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
            end_time = start_time + timedelta(days=1)
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
            end_time = start_time + timedelta(days=1)
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


class SimulatorInput():
    def __init__(
            self,
            input_sensor,
            input_granule: Granule,
            target_sensor,
            target_granules: List[Granule]):
        self.input_sensor = input_sensor
        self.input_granule = input_granule
        self.target_sensor = target_sensor
        self.target_granules = merge_granules(sorted(list(target_granules)))

    def __len__(self):
        return 1

    def __iter__(self):
        for target_granule in self.target_granules:
            yield self.load_input_data(target_granule)


    def load_input_data(self, target_granule):

        input_granule = self.input_granule
        input_data = run_preprocessor(input_granule)
        input_data = mask_invalid_values(input_data)

        upsampling_factors = UPSAMPLING_FACTORS[self.input_sensor.name.lower()]
        input_data = upsample_data(input_data, upsampling_factors)
        input_data = add_cpcir_data(input_data)

        rof_in = RADIUS_OF_INFLUENCE[self.input_sensor.name.lower()]
        rof_targ = RADIUS_OF_INFLUENCE[self.target_sensor.name.lower()]
        input_obs = calculate_obs_properties(input_data, input_granule, radius_of_influence=rof_in)
        target_obs = calculate_obs_properties(input_data, target_granule, radius_of_influence=rof_targ)


        data = xr.Dataset({
            "input_observations": input_obs.observations.rename(channels="input_channels"),
            "input_meta_data": input_obs.meta_data.rename(channels="input_channels"),
            "target_observations": target_obs.observations.rename(channels="target_channels"),
            "target_meta_data": target_obs.meta_data.rename(channels="target_channels"),
            "ir_observations": input_data.ir_observations,
        })
        for var in ANCILLARY_VARIABLES:
            data[var] = input_data[var]

        tbs = data.input_observations.data
        tbs[tbs < 0] = np.nan
        valid = np.isfinite(tbs).any(0)
        tbs = data.target_observations.data
        tbs[tbs < 0] = np.nan
        valid *= np.isfinite(tbs).any(0)
        data["valid"] = (("scans", "pixels"), np.zeros_like(valid, dtype="float32"))

        scan_time_input = input_obs.scan_time
        scan_time_target = input_obs.scan_time
        time_diff = scan_time_input - scan_time_target
        valid *= np.abs(time_diff.data) < np.timedelta64(15, "m")

        data.valid.data[~valid] = np.nan

        n_chans_in = data.input_channels.size
        n_chans_out = data.target_channels.size

        obs_in = []
        meta_in = []
        for input_ind in range(n_chans_in):
            obs = data.input_observations.data[input_ind]
            meta = data.input_meta_data.data[input_ind]
            obs, meta = transform_observations_satformer(obs, meta)
            obs_in.append(torch.tensor(obs.astype(np.float32)))
            meta_in.append(torch.tensor(meta.astype(np.float32)))


        obs_out = []
        meta_out = []
        for output_ind in range(n_chans_out):
            obs_out.append(torch.tensor(data.target_observations.data[[output_ind]]))
            meta_out.append(torch.tensor(data.target_meta_data.data[output_ind]))

        inpt = {
            "observations": torch.stack(obs_in, 1)[None],
            "input_observation_props": torch.stack(meta_in, 1)[None],
            "output_observation_props": torch.stack(meta_out, 1)[None],
        }

        for anc_var in ANCILLARY_VARIABLES + ["ir_observations"]:
            anc_data = torch.tensor(data[anc_var].data).to(dtype=torch.float32)
            if anc_data.ndim < 3:
                anc_data = anc_data[None]
            if anc_data.ndim < 4:
                anc_data = anc_data[:, None]

            n_chans, n_seq, n_y, n_x = anc_data.shape
            anc_mask = torch.isnan(anc_data).all()[None]
            inpt[anc_var] = anc_data[None]
            inpt[anc_var + "_mask"] = anc_mask[None]

        mask = torch.isnan(inpt["observations"]).all(0).all(-1).all(-1)
        inpt["input_observation_mask"] = mask

        return inpt, {}, "results.nc"


def simulate_tbs(
        model_path: Path,
        input_sensor,
        input_granule,
        target_sensor,
        target_granules,
        device: Optional[str] = None

):
    input_loader = SimulatorInput(input_sensor, input_granule, target_sensor, target_granules)
    model = load_model(model_path).eval()

    expected_value = RetrievalOutputConfig(model.output_config["output_observations"], "ExpectedValue", {})
    random_sample = RetrievalOutputConfig(model.output_config["output_observations"], "RandomSample", {"n_samples": 1})
    retrieval_output = {
        "output_observations": {
            "output_observations": expected_value,
            "output_observations_rand": random_sample
        }
    }

    mask_vars = [
        "input_observation_mask",
    ]
    for var in ANCILLARY_VARIABLES:
        mask_vars.append(f"{var}_mask")
    mask_vars.append("ir_observations_mask")

    inference_config = InferenceConfig(
        tile_size=128,
        spatial_overlap=32,
        retrieval_output=retrieval_output,
        batch_size=1,
        exclude_from_tiling=mask_vars
    )

    if device is None:
        device = "cuda:0"

    results = run_inference(
        model,
        input_loader,
        inference_config,
        device=device,
    )
    results = results[0]
    return results


class GPROFNNHRInputLoader:
    """
    Inputloader for the experimental GPROF-NN HR retrieval.
    """
    def __init__(
            self,
            path: str | Path | List[str | Path],
    ):
        """
        Args:
            path: A path object pointing to a specific input file or a folder
                hierarchy containing multiple input files.
        """
        # Determine input files.
        if isinstance(path, list):
            self.input_files = [Path(fle) for fle in path]
        else:
            path = Path(path)
            if path.is_dir():
                input_files = sorted(list(path.glob("**/*.HDF5")))
                self.input_files = input_files
            else:
                self.input_files = [path]


    def __len__(self) -> int:
        """
        The number of files to process.
        """
        return len(self.input_files)


    def load_input_data(self, path: Path) -> Dict[str, torch.Tensor]:
        """
        Load retrieval input data.

        Args:
            path: A path object pointing to the file from which to load the input data.

        Return:
            A dictionary mapping the names of the retrieval inputs ('brightness_temperatures',
            'earth_incidence_angles', 'ancillary_data') for tensor containing the corresponding data.
        """

        l1c_file = L1CFile(path)
        sensor = l1c_file.sensor
        data_pp = run_preprocessor(path, sensor)

        upsampling_factors = UPSAMPLING_FACTORS[sensor.name.lower()]
        input_data = upsample_data(data_pp, upsampling_factors)
        input_data = add_cpcir_data(input_data)

        rof_in = RADIUS_OF_INFLUENCE[sensor.name.lower()]
        rec = FileRecord(path, product=PANSAT_PRODUCTS[sensor.name.lower()][0])
        granule = Granule(rec, rec.temporal_coverage, None)
        rof_in = RADIUS_OF_INFLUENCE["gmi"]
        input_obs = calculate_obs_properties(input_data, granule, radius_of_influence=rof_in)
        observations = torch.tensor(input_obs.observations.data)
        input_observation_props = torch.tensor(input_obs.meta_data.data).transpose(0, 1)[None]

        obs_in = []
        for ind, obs in enumerate(input_obs.observations.data):
            valid = obs >= 0.0
            obs[..., ~valid] = np.nan
            mean = np.mean(obs[valid])
            std = np.std(obs[valid])
            obs_n = (obs - mean) / std
            obs = np.stack([
                np.ones_like(obs_n) * mean,
                np.ones_like(obs_n) * std,
                obs_n
            ])
            obs_in.append(torch.tensor(obs))

            input_observation_props[..., ind, torch.tensor(~valid)] = np.nan

        obs_in = torch.stack(obs_in, 1)[None]
        obs_in_mask = torch.isnan(obs_in).all(1).all(-1).all(-1)

        inpt = {
            "observations": obs_in,
            "input_observation_props": input_observation_props,
            "input_observation_mask": obs_in_mask,
        }

        anc_vars = [
            "two_meter_temperature",
            "total_column_water_vapor",
            "leaf_area_index",
            "land_fraction",
            "ice_fraction",
            "elevation",
            "ir_observations",
        ]
        for anc_var in anc_vars:
            anc_data = torch.tensor(input_data[anc_var].data).to(dtype=torch.float32)
            if anc_data.dim() < 3:
                anc_data = anc_data[None]
            anc_data = anc_data[None, :, None]
            anc_mask = torch.isnan(anc_data).all()[None, None]
            inpt[anc_var] = anc_data
            inpt[anc_var + "_mask"] = anc_mask



        aux = {
            "scan_time": input_data.scan_time.data,
            "longitude": input_data.longitude.data,
            "latitude": input_data.latitude.data,
            "total_column_water_vapor": input_data.total_column_water_vapor.data,
            "two_meter_temperature": input_data.two_meter_temperature.data,
            #"convective_fraction": input_data.convective_precipitation.data,
            #"moisture_convergence": input_data.moisture_convergence.data,
            "leaf_area_index": input_data.leaf_area_index.data,
            #"snow_depth": input_data.snow_depth.data,
            #"orographic_wind": input_data.orographic_wind.data,
            #"wind_speed_10m": input_data["10m_wind"].data,
            #"mountain_index": input_data.mountain_type.data,
            "land_fraction": input_data.land_fraction.data,
            "ice_fraction": input_data.ice_fraction.data,
            "elevation": input_data.elevation.data
        }
        return inpt, aux


    def __getitem__(self, ind: int):
        input_data, aux = self.load_input_data(self.input_files[ind])
        return input_data, aux, self.input_files[ind].name

    def __iter__(self):
        for path in self.input_files:
            input_data, aux = self.load_input_data(path)
            yield input_data, aux, path.name

    def finalize_results(
            self,
            results: Dict[str, torch.Tensor],
            aux: Dict[str, np.ndarray],
            filename: str
    ) -> Tuple[xr.Dataset, str]:
        """
        Combines retrieval results with auxiliary data into orbit-based retrieval
        result files. This method does is called as part of the inference method
        provided by pytorch_retrieve.

        Args:
            results: A dictionary mapping retrieval output names to tensor containing
                corresponding results.
            aux: A dictionary containing auxiliary data passed along from the
                retrieval input.
            filename: The filename of the input file.

        Return:
            A tuple ``(results, filename)`` containing the retrieval results as
            xarray.Dataset in ``results`` and the filename to use to store the
            results in ``filename``.
        """
        lons = aux["longitude"]
        lats = aux["latitude"]
        shape = lons.shape

        if lons.ndim == 2:
            dims = ("scans", "pixels", "levels")
        else:
            dims = ("samples", "levels")

        output = xr.Dataset()
        for name, data in aux.items():
            data = data.squeeze()
            if data.ndim > 2 and data.shape[-1] != 28:
                data = data.transpose((1, 2, 0))
            dims_v = dims[:data.ndim]

            output[name] = (dims_v, data)

        if lons.ndim == 2:
            dims = ("scans", "pixels", "levels")
        else:
            dims = ("samples", "levels")
        for var, tensor in results.items():

            # Discard dummy dimensions.
            tensor = tensor[0].squeeze()


            if var == "surface_precip_terciles":
                tensor = torch.permute(tensor, (1, 2, 0))
                if lons.ndim < 2:
                    dims_v = ("samples",)
                else:
                    dims_v = ("scans", "pixels")
                output["surface_precip_1st_tercile"] = (
                    dims_v, tensor[..., 0].numpy()
                )
                output["surface_precip_1st_tercile"].encoding = {"dtype": "float32", "zlib": True}
                output["surface_precip_2nd_tercile"] = (
                    dims_v,
                    tensor[..., 1].numpy()
                )
                output["surface_precip_2nd_tercile"].encoding = {"dtype": "float32", "zlib": True}

            else:
                if tensor.dim() > 2:
                    tensor = tensor.squeeze()
                    tensor = torch.permute(tensor, (1, 2, 0))
                dims_v = dims[:tensor.dim()]

                tensor = tensor.numpy()
                if "valid_input" in aux:
                    tensor[~aux["valid_input"]] = -9999.9

                output[var] = (dims_v, tensor)
                # Use compressiong to keep file size reasonable.
                output[var].encoding = {"dtype": "float32", "zlib": True}


        # Quick and dirty way to transform 1C filename to 2A filename
        output_filename = (
            filename.replace("1C-R", "2A")
            .replace("1C", "2A")
            .replace("pp", "nc")
            .replace("HDF5", "nc")
        )

        # Return outputs as xr.Dataset and filename to use to save data.
        return output, output_filename
