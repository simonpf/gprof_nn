"""
gprof_nn.data.finetuning
========================

Implements functionality to extract finetune datasets for the GPROF-NN
retrievals.
"""
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

import click
import numpy as np
import xarray as xr

from gprof_nn.utils import to_datetime64
from gprof_nn.definitions import ANCILLARY_VARIABLES
from gprof_nn.data.utils import (
    write_training_samples_1d,
    write_training_samples_3d
)


LOGGER = logging.getLogger(__name__)


def extract_finetuning_samples(
        collocation_path: Path,
        output_path_1d: Path,
        output_path_3d: Path,
        start_time: Optional[np.datetime64] = None,
        end_time: Optional[np.datetime64] = None,
) -> None:
    """
    Extract training samples for fine-tuning the GPROF-NN retrievals.

    Args:
        collocation_path: A path object pointing to the directory containing the extracted
            collocations.
        output_path_1d: A path object pointing to the path to which to write the training
            samples for the GPROF-NN 1D retrieval.
        output_path_3d: A path object pointing to the path to which to write the training
            samples for the GPROF-NN 3D retrieval.
        start_time: An optional start time to limit the collocations files considered.
        end_time: An optional end time to limit the collocation files considered.
    """
    collocation_files = sorted(list(Path(collocation_path).glob("*.nc")))
    valid_files = []
    for collocation_file in collocation_files:
        date_str = collocation_file.stem.split("_")[-1]
        date = to_datetime64(
            datetime.strptime(date_str, "%Y%m%d%H%M%S")
        )
        if start_time is not None and date < start_time:
            continue
        if end_time is not None and date > start_time:
            continue
        valid_files.append(collocation_file)

    LOGGER.info(
        "Found %s collocation files in range %s - %s.",
        len(valid_files),
        start_time,
        end_time
    )

    input_variables = (
        ANCILLARY_VARIABLES +
        [
            "tbs_mw_gprof",
            "earth_incidence_angle",
            "surface_type",
            "airlifting_index",
            "mountain_type",
            "scan_time",
            "latitude",
            "longitude"
        ]
    )

    reference_variables = ["surface_precip"]


    for path in valid_files:

        with xr.open_dataset(path, group="input_data") as data:
            input_data = data[input_variables].rename(
                tbs_mw_gprof="brightness_temperatures"
            )

        with xr.open_dataset(path, group="reference_data") as data:
            ref_data = data[reference_variables].rename(
                surface_precip="surface_precip_combined"
            ).drop(["latitude", "longitude"])

        data = xr.merge(
            [input_data, ref_data]
        )

        write_training_samples_1d(
            output_path_1d,
            "cmb",
            data,
            reference_var="surface_precip_combined"

        )
        write_training_samples_3d(
            output_path_3d,
            "cmb",
            data,
            n_scans=128,
            n_pixels=64,
            overlapping=True,
            reference_var="surface_precip_combined"
        )


@click.argument("collocation_path", type=str)
@click.argument("output_path_1d", type=str)
@click.argument("output_path_3d", type=str)
@click.option("--start_time", type=str, default=None)
@click.option("--end_time", type=str, default=None)
def cli(
        collocation_path: str,
        output_path_1d: str,
        output_path_3d: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
) -> None:
    """
    Extract training samples for fine-tuning the GPROF-NN retrievals.

    Args:
        collocation_path: A path object pointing to the directory containing the extracted
            collocations.
        output_path_1d: A path object pointing to the path to which to write the training
            samples for the GPROF-NN 1D retrieval.
        output_path_3d: A path object pointing to the path to which to write the training
            samples for the GPROF-NN 3D retrieval.
        start_time: An optional start time to limit the collocations files considered.
        end_time: An optional end time to limit the collocation files considered.
    """
    collocation_path = Path(collocation_path)
    output_path_1d = Path(output_path_1d)
    output_path_3d = Path(output_path_3d)

    if start_time is not None:
        try:
            start_time = np.datetime64(start_time)
        except RuntimeError:
            LOGGER.error(
                "Couldn't parse start time '%s' as np.datetime64 value."
            )
            return 1

    if end_time is not None:
        try:
            end_time = np.datetime64(end_time)
        except RuntimeError:
            LOGGER.error(
                "Couldn't parse end time '%s' as np.datetime64 value."
            )
            return 1

    extract_finetuning_samples(
        collocation_path,
        output_path_1d,
        output_path_3d,
        start_time=start_time,
        end_time=end_time
    )
