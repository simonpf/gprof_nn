"""
gprof_nn.data.finetuning
========================

Implements functionality to extract finetune datasets for the GPROF-NN retrievals.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from functools import cache
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import click
from filelock import FileLock
import numpy as np
import torch
from torch import nn
import xarray as xr

from pansat import Granule, TimeRange
from pansat.catalog import Index
from pansat.catalog.index import find_matches
from pansat.utils import resample_data

from pyresample.geometry import SwathDefinition

from pytorch_retrieve import load_model
from pytorch_retrieve.inference import run_inference

from gprof_nn.utils import to_datetime64
from gprof_nn import sensors
from gprof_nn.definitions import ALL_TARGETS
from gprof_nn.retrieval import GPROFNNInputLoader
from gprof_nn.definitions import ANCILLARY_VARIABLES
from gprof_nn.data.utils import (
    PANSAT_PRODUCTS,
    RADIUS_OF_INFLUENCE,
    extract_scans,
    extract_scenes,
    run_preprocessor,
    write_training_samples_1d,
    write_training_samples_3d,

)


LOGGER = logging.getLogger(__name__)


@cache
def get_retrieval_results(path: Path) -> Dict[int, Path]:
    """
    Create a mapping of orbit number to corresponding retrieval result files.

    Args:
         path: A path object pointing to the directory containing the retrieval results.

    Return:
        A mapping of orbit numbers to the corresponding result file.
    """
    result_files = path.glob("**/*.nc")
    orbit_to_files = {}
    for path in result_files:
        try:
            orbit_num = int(path.name.split(".")[-3])
            orbit_to_files[orbit_num] = path
        except ValueError:
            pass
    LOGGER.info(
        "Found %s retrieval files.", len(orbit_to_files)
    )
    return orbit_to_files


def run_retrieval(
        gpm_granule: Granule,
        retrieval_model: nn.Module,
        device: str = "cuda:0"
) -> xr.Dataset:
    """
    Run retrieval on a GPM granule.

    Args:
        gpm_granule: A pansat granule identifying a subset of an orbit
            of GPM L1C files.
        retrieval_model: The GPROF-NN retrieval model to run the retrieval on.
        device: The name of the device to run the retrieval on.

    Return:
        An xarray.Dataset containing the results from the preprocessor.
    """
    from gprof_nn.data.l1c import L1CFile
    old_dir = os.getcwd()

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        l1c_file = extract_scans(gpm_granule, tmp, min_scans=256)
        input_loader = GPROFNNInputLoader([l1c_file])
        lock = FileLock(f"{device}.lock")
        with lock:
            results = run_inference(
                retrieval_model,
                input_loader,
                retrieval_model.inference_config,
                device=device,
            )
            results = results[0]
    return results


def process_match(
        target_sensor: sensors.Sensor,
        target_granule: Granule,
        reference_sensor: sensors.Sensor,
        reference_granules: List[Granule],
        retrieval_model: Optional[nn.Module],
        retrieval_path: Optional[nn.Module],
        output_path_1d: Path,
        output_path_3d: Path,
        climate_only: bool = False
) -> None:
    """
    Process granule.

    Args:
        target_sensor: The sensor for which to extract the training data.
        target_granule: The target-sensor granule defining the match.
        reference_sensor: The sensor from which to extract the reference data.
        reference_granules: The reference-sensor granules defining the match.
        retrieval_model: The retrieval model to use to generate the reference retrievals.
        output_path_1d: The path to which to write the GPROF-NN 1D training data.
        output_path_3d: The path to which to write the GPROF-NN 3D training data.
        climate_only: Set to True to use only CLIM preprocessor.
    """
    if len(reference_granules) == 0:
        raise ValueError(
            "Empty match."
        )

    for reference_granule in reference_granules:

        if retrieval_model is not None:
            retrieval_results = run_retrieval(reference_granule, retrieval_model)
        else:
            orbit_num = int(reference_granule.file_record.filename.split(".")[-3])
            all_results = get_retrieval_results(retrieval_path)
            retrieval_results = xr.load_dataset(all_results[orbit_num])

        retrieval_results = retrieval_results.drop_vars(["surface_precip"])
        retrieval_results = retrieval_results.rename(
            latent_heating="latent_heat",
            surface_precip_rand="surface_precip"
        )[
            ALL_TARGETS + ["scan_time", "longitude", "latitude"]
        ]

        if climate_only or np.random.rand() > 0.5:
            settings = {
                "prodtype": "CLIMATOLOGY",
                "prepdir": "/qdata2/archive/ERA5/"
            }
        else:
            settings = {
                "prodtype": "STANDARD",
                "prepdir": "/qdata1/pbrown/gpm/modelprep/GANALV7/"
            }
        input_data = run_preprocessor(target_granule, settings=settings)
        swath = SwathDefinition(lons=input_data.longitude.data, lats=input_data.latitude.data)
        retrieval_results = resample_data(
            retrieval_results,
            swath,
            new_dims=("scans", "pixels"),
            radius_of_influence=RADIUS_OF_INFLUENCE[reference_sensor.name.lower()]
        )
        scan_time = retrieval_results.scan_time
        input_data = xr.merge([input_data, retrieval_results.drop_vars(["scan_time"])])

        time_diff = retrieval_results.scan_time - input_data.scan_time

        time_diff.to_netcdf("time_diff.nc")
        input_data.to_netcdf("input.nc")
        diff = time_diff.data.astype("timedelta64[m]")

        valid = (np.abs(time_diff) < np.timedelta64(15, "m")) * np.isfinite(input_data.surface_precip)
        valid = valid.data.astype(np.float32)
        valid[valid < 1.0] = np.nan
        input_data["valid"] = (("scans", "pixels"), valid)

        ref_name = reference_sensor.name.lower()
        targ_name = target_sensor.name.lower()
        prefix = f"{targ_name}_{ref_name}"

        input_data["source"] = "finetuning"

        if output_path_3d is not None:
            write_training_samples_3d(
                output_path_3d,
                prefix,
                input_data,
                n_scans=128,
                n_pixels=64,
                min_valid=10,
                reference_var="valid"
            )
        if output_path_1d is not None:
            write_training_samples_1d(
                output_path_1d,
                prefix,
                input_data,
                reference_var="valid"
            )


def extract_finetuning_samples(
        reference_sensor: sensors.Sensor,
        retrieval_model: Optional[Path],
        retrieval_path: Optional[Path],
        target_sensor: sensors.Sensor,
        year: int,
        month: int,
        day: int,
        output_path_1d: Path,
        output_path_3d: Path,
        climate_only: bool = False
) -> None:
    """
    Extract training samples to fine-tune GPROF-NN retrievals.

    Args:
        reference_sensor: The sensor to use to generate the retrieval reference data from.
        retrieval_model: The retrieval model to use.
        year: Integer defining the year.
        month: Integer defining the month.
        day: Integer defining the day.
        output_path_1d: The path to write the 1D training data to.
        output_path_3d: The path to write the 3D training data to.
        climate_only: Set to True to use only CLIM preprocessor.
    """
    if retrieval_path is None and retrieval_model is None:
        raise ValueError(
            "One of 'retrieval_path' or 'retrieval_model' must not be None."
        )

    ref_prods = PANSAT_PRODUCTS[reference_sensor.name.lower()][:1]
    targ_prods = PANSAT_PRODUCTS[target_sensor.name.lower()]

    if retrieval_model is not None:
        retrieval_model = load_model(retrieval_model)
    if retrieval_path is not None:
        retrieval_path = Path(retrieval_path)

    start_time = datetime(year, month, day)
    end_time = start_time + timedelta(hours=23, minutes=59)
    time_range = TimeRange(
        start_time,
        end_time
    )
    for ref_prod in ref_prods:
        for targ_prod in targ_prods:
            ref_recs = ref_prod.get(time_range)
            LOGGER.info(
                "Found %s files for reference product %s.",
                len(ref_recs),
                ref_prod.name
            )
            ref_index = Index.index(ref_prod, ref_recs)
            LOGGER.info("Getting target input files.")
            targ_recs = targ_prod.get(time_range)
            LOGGER.info(
                "Found %s files for target product %s.",
                len(targ_recs),
                targ_prod.name
            )
            targ_index = Index.index(targ_prod, targ_recs)
            matches = find_matches(targ_index, ref_index, np.timedelta64(15, "m"))

            LOGGER.info(
                "Found %s collocations for reference product %s and target product %s.",
                len(matches), ref_prod, targ_prod
            )

            for targ_granule, reference_granules in matches:
                try:
                    process_match(
                        target_sensor,
                        targ_granule,
                        reference_sensor,
                        reference_granules,
                        retrieval_model,
                        retrieval_path,
                        output_path_1d,
                        output_path_3d,
                        climate_only=climate_only
                    )
                except Exception:
                    LOGGER.exception(
                        "Encountered an error when processing match %s.",
                        targ_granule
                    )


@click.argument("reference_sensor", type=str)
@click.argument("target_sensor", type=str)
@click.argument("year", type=int)
@click.argument("month", type=int)
@click.argument("days", type=int, nargs=-1)
@click.argument("output_path_1d", type=str)
@click.argument("output_path_3d", type=str)
@click.option("--n_processes", type=int, default=1, help="Number of processes to use to parallelize data extraction.")
@click.option("--retrieval_model", type=str, default=None, help="Reference retrieval model.")
@click.option("--retrieval_path", type=str, default=None, help="Path containing reference retrieval results.")
@click.option("--climate_only", is_flag=True, help="Use only CLIM preprocessor.")
def cli(
        reference_sensor: str,
        target_sensor: str,
        year: int,
        month: int,
        days: List[int],
        output_path_1d: str,
        output_path_3d: str,
        n_processes: int = 1,
        retrieval_model: Optional[str] = None,
        retrieval_path: Optional[str] = None,
        climate_only: bool = False
) -> None:
    """
    Extract samples to fine-tune GPROF retrievals for TARGET_SENSOR using retrieval from
    REFERENCE_SENSOR.
    """
    reference_sensor = getattr(sensors, reference_sensor.upper())
    target_sensor = getattr(sensors, target_sensor.upper())

    if retrieval_model is not None:
        retrieval_model = Path(retrieval_model)
        if not retrieval_model.exists():
            LOGGER.error(
                "'retrieval model' argument must point to an existing retrieval model."
            )
            return 1
    if retrieval_path is not None:
        retrieval_path = Path(retrieval_path)
        if not retrieval_path.exists():
            LOGGER.error(
                "'retrieval path' argument must point to a directory containing retrieval results."
            )
            return 1

    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))

    output_path_1d = Path(output_path_1d)
    if not output_path_1d.exists():
        LOGGER.error(
            "'output_path_1d' must point to an existing directory."
        )
        return 1

    output_path_3d = Path(output_path_3d)
    if not output_path_3d.exists():
        LOGGER.error(
            "'output_path_1d' must point to an existing directory."
        )
        return 1

    if n_processes <= 1:
        for day in days:
            extract_finetuning_samples(
                reference_sensor,
                retrieval_model,
                retrieval_path,
                target_sensor,
                year,
                month,
                day,
                output_path_1d,
                output_path_3d,
                climate_only=climate_only
            )
    else:
        pool = ProcessPoolExecutor(max_workers=n_processes)
        tasks = []
        for day in days:
            task = pool.submit(
                extract_finetuning_samples,
                reference_sensor,
                retrieval_model,
                retrieval_path,
                target_sensor,
                year,
                month,
                day,
                output_path_1d,
                output_path_3d,
                climate_only=climate_only
            )
            tasks.append(task)
        for day, task in zip(days, tasks):
            try:
                task.result()
            except Exception:
                LOGGER.exception(
                    "Encountered the following error when processing %s/%s/%s.",
                    year, month, day
                )
