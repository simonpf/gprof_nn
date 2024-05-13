"""
==================
gprof_nn.retrieval
==================

This module contains classes and functionality that drive the execution
of the retrieval.
"""
import logging
import math
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
import re
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import xarray as xr

import torch
from torch import nn
import numpy as np
from pansat import Granule
import pandas as pd

from gprof_nn import sensors
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.preprocessor import PreprocessorFile, run_preprocessor
from gprof_nn.definitions import ANCILLARY_VARIABLES, ALL_TARGETS
from gprof_nn.data.training_data import (
    EIA_GMI,
    load_tbs_1d_gmi,
    load_tbs_1d_xtrack_sim,
    load_tbs_1d_conical_sim,
    load_tbs_1d_xtrack_other,
    load_tbs_1d_conical_other,
    load_ancillary_data_1d,
    load_training_data_3d_gmi,
    load_training_data_3d_xtrack_sim,
    load_training_data_3d_conical_sim,
    load_training_data_3d_other,
)


LOGGER = logging.getLogger(__name__)



def load_input_data_preprocessor(
        preprocessor_file: Path
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """
    Load retrieval input data from preprocessor file.

    Args:
        preprocessor_file: A path pointing to a preprocessor file.

    Return:
        A pair of dictionaries: The first one containing the input tensors
        'brightness_temperatures', 'viewing_angles', and 'ancillary_data',
        the second containing auxilliary data to store in the retrieval output.
    """
    file_pp = PreprocessorFile(preprocessor_file)
    data_pp = file_pp.to_xarray_dataset()

    tbs = data_pp.brightness_temperatures.data.astype(np.float32)
    tbs[tbs < 0] = np.nan
    tbs_full = np.nan * np.zeros((tbs.shape[:2] + (15,)), dtype=np.float32)
    tbs_full[..., file_pp.sensor.gprof_channels] = tbs
    tbs_full = np.transpose(tbs_full, (2, 0, 1))

    viewing_angles = data_pp.earth_incidence_angle.data.astype(np.float32)
    angs_full = np.nan * np.zeros(viewing_angles.shape[:2] + (15,), dtype=np.float32)
    if viewing_angles.ndim == 2:
        viewing_angles = viewing_angles[..., None]
    angs_full[..., file_pp.sensor.gprof_channels] = viewing_angles
    angs_full = np.transpose(angs_full, (2, 0, 1))

    anc = torch.tensor(np.stack(
        [data_pp[anc_var].data.astype("float32") for anc_var in ANCILLARY_VARIABLES]
    ))

    input_data = {
        "brightness_temperatures": tbs_full,
        "viewing_angles": angs_full,
        "ancillary_data": anc
    }
    aux = {
        "scan_time": data_pp.scan_time.data,
        "longitude": data_pp.longitude.data,
        "latitude": data_pp.latitude.data,
    }
    return input_data, aux


def load_input_data_l1c(
        l1c_file: Path,
        needs_ancillary: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Load retrieval input data from a L1C file.

    Args:
        l1c_file: A path pointing to a L1C file.

    Return:
        A dictionary containing the input tensors 'brightness_temperatures',
        'viewing_angles', and 'ancillary_data'.
    """
    sensor = L1CFile(l1c_file).sensor
    if needs_ancillary:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pp_path = tmp_path / l1c_file.with_suffix(".pp").name
            run_preprocessor(l1c_file, sensor, output_file=pp_path)
            return load_input_data_preprocessor(pp_path)

    l1c_data = L1CFile(l1c_file).to_xarray_dataset()
    tbs = torch.tensor(
        np.transpose(l1c_data.brightness_temperatures.data.astype(np.float32), (2, 0, 1))
    )
    aux = {
        "scan_time": l1c_data.scan_time.data,
        "latitude": l1c_data.latitude.data,
        "longitude": l1c_data.longitude.data,
    }
    return {"brightness_temperatures": tbs}, aux


def load_input_data_training_1d(
        training_file: Path
) -> Dict[str, torch.Tensor]:
    """
    Load retrieval input data from a 1D training file.

    Args:
        training_file: A path object pointing to the training file from which
            to load the input data.

    Return:
        A dictionary containing the input tensors 'brightness_temperatures',
        'viewing_angles', and 'ancillary_data'.
    """
    rng = np.random.default_rng(42)

    with xr.open_dataset(training_file) as data:
        sensor = sensors.get_sensor(data.attrs["sensor"])

        if sensor == sensors.GMI:
            tbs = load_tbs_1d_gmi(data)
            anc = load_ancillary_data_1d(data)
            angs = torch.tensor(np.broadcast_to(EIA_GMI.astype("float32"), tbs.shape))

        elif isinstance(sensor, sensors.CrossTrackScanner):
            if data.attrs["source"] == "sim":
                angles = data["angles"].data
                angs = rng.uniform(
                    angles.min(),
                    angles.max(),
                    size=data.samples.size,
                ).astype(np.float32)
                tbs = load_tbs_1d_xtrack_sim(data, angs, sensor)
                angs = torch.tensor(angs)
                angs = torch.tensor(np.broadcast_to(angs[..., None], tbs.shape))
            else:
                tbs, angs = load_tbs_1d_xtrack_other(data, sensor)
            anc = load_ancillary_data_1d(data)

        elif isinstance(sensor, sensors.ConicalScanner):

            if data.source == "sim":
                tbs = load_tbs_1d_conical_sim(data, sensor)
                angs = torch.tensor(np.broadcast_to(EIA_GMI.astype("float32"), tbs.shape))
            else:
                tbs, angs = load_tbs_1d_conical_other(data, sensor)
            anc = load_ancillary_data_1d(data)

        input_data = {
            "brightness_temperatures": tbs,
            "ancillary_data": anc,
            "viewing_angles": angs
        }
        aux = {
            "longitude": data.longitude.data,
            "latitude": data.latitude.data,
        }
        return input_data, aux


def load_input_data_training_3d(
        training_file: Path
) -> Dict[str, torch.Tensor]:
    """
    Load retrieval input data from a 3D training file.

    Args:
        training_file: A path object pointing to the training file from which
            to load the input data.

    Return:
        A dictionary containing the input tensors 'brightness_temperatures',
        'viewing_angles', and 'ancillary_data'.
    """
    rng = np.random.default_rng(42)

    with xr.open_dataset(training_file) as scene:
        sensor = scene.attrs["sensor"]
        sensor = getattr(sensors, sensor)

        if sensor == sensors.GMI:
            input_data, targets = load_training_data_3d_gmi(
                scene,
                targets=ALL_TARGETS,
                augment=None,
                rng=rng
            )
        elif isinstance(sensor, sensors.CrossTrackScanner):
            if scene.source == "sim":
                input_data, targets = load_training_data_3d_xtrack_sim(
                    sensor,
                    scene,
                    targets=ALL_TARGETS,
                    augment=None,
                    rng=rng
                )
            else:
                input_data, targets = load_training_data_3d_other(
                    sensor,
                    scene,
                    targets=ALL_TARGETS,
                    augment=None,
                    rng=rng
                )
        elif isinstance(sensor, sensors.ConicalScanner):
            if scene.source == "sim":
                input_data, targets = load_training_data_3d_conical_sim(
                    sensor,
                    scene,
                    targets=ALL_TARGETS,
                    augment=None,
                    rng=rng
                )
            else:
                input_data, targets = load_training_data_3d_other(
                    sensor,
                    scene,
                    targets=ALL_TARGETS,
                    augment=None,
                    rng=rng
                )

        aux = {
            "longitude": scene.longitude.data,
            "latitude": scene.latitude.data,
        }
        for name, target_data in targets.items():
            aux[name + "_ref"] = target_data
        return input_data, aux
    raise RuntimeError(
        "Invalid sensor/scene combination in training file %s.",
        training_file
    )


def determine_input_format(path: Path) -> str:
    """
    Determine input format from input file.

    Args:
        path: A path object pointing to the file for which to determine the input format.

    Return:
        A string representing the input format: 'preprocessor', 'l1c', 'training_1d', 'training_3d'.

    """
    if path.suffix == ".pp":
        return "preprocessor"
    if path.suffix == ".HDF5":
        return "l1c"
    if path.suffix == ".nc":
        with xr.open_dataset(path) as input_data:
            if "scans" in input_data.dims:
                return "training_3d"
            return "training_1d"
    raise RuntimeError(
        f"Encountered an input file with suffix {path.suffix}, which is currently not supported."
    )


class GPROFNNInputLoader:
    def __init__(
            self,
            path: str | Path | List[str | Path],
            input_format: Optional[str] = None,
            config: str = "3d",
            needs_ancillary: bool = True
    ):

        # Determine input files.
        if isinstance(path, list):
            self.input_files = [Path(fle) for fle in path]
        else:
            path = Path(path)
            if path.is_dir():
                input_files = sorted(list(path.glob("**/*.nc")))
                input_files += sorted(list(path.glob("**/*.HDF5")))
                input_files += sorted(list(path.glob("**/*.pp")))
                self.input_files = input_files
            else:
                self.input_files = [path]

        self.needs_ancillary = needs_ancillary
        config = config.lower()
        if not config in ['1d', '3d']:
            raise ValueError(
                "Config must be '1d' for GPROF-NN 1D retrievals or '3d' for GPROF-NN 3D retrievals."
            )
        self.config = config

        self.input_format = input_format


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
            'viewing_angles', 'ancillary_data') for tensor containing the corresponding data.
        """
        if self.input_format is None:
            input_format = determine_input_format(path)

        if input_format == "preprocessor":
            input_data, aux = load_input_data_preprocessor(path)
        elif input_format == "l1c":
            input_data, aux = load_input_data_l1c(path)
        elif input_format == "training_1d":
            input_data, aux = load_input_data_training_1d(path)
        elif input_format == "training_3d":
            input_data, aux = load_input_data_training_3d(path)
        else:
            raise ValueError(
                f"Encountered unknown input format '{input_format}'."
            )

        if self.config == "1d":
            input_data = {
                name: torch.permute(tensor, (1, 2, 0)).reshape((-1, 15)) if tensor.ndim == 3 else tensor
                for name, tensor in input_data.items()
            }

        return input_data, aux

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

        shape = (data.scans.size, data.pixels.size)
        dims = ("levels", "scans", "pixels")

        results = xr.Dataset()
        for name, data in aux:
            results[var] = (dims_v, tensor.numpy())

        for var, tensor in results.items():

            # Discard dummy dimensions.
            tensor = tensor.squeeze()
            if self.config.lower() == "1d":
                tensor = tensor.reshape(shape + tensor.shape[1:])
                if tensor.dim() > 2:
                    tensor = torch.permute(tensor, (2, 0, 1))

            if var == "surface_precip_terciles":
                results["surface_precip_1st_tercile"] = (
                    ("scans", "pixels"), tensor[0].numpy()
                )
                results["surface_precip_1st_tercile"].encoding = {"dtype": "float32", "zlib": True}
                results["surface_precip_2nd_tercile"] = (
                    ("scans", "pixels"),
                    tensor[1].numpy()
                )
                results["surface_precip_2nd_tercile"].encoding = {"dtype": "float32", "zlib": True}
            else:
                dims_v = dims[-tensor.dim():]
                results[var] = (dims_v, tensor.numpy())
                # Use compressiong to keep file size reasonable.
                results[var].encoding = {"dtype": "float32", "zlib": True}

        # Quick and dirty way to transform 1C filename to 2A filename
        output_filename = (
            filename.replace("1C-R", "2A")
            .replace("1C", "2A")
            .replace("HDF5", "nc")
        )

        # Return results as xr.Dataset and filename to use to save data.
        return data, output_filename
