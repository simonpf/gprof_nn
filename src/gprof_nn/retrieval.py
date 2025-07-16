"""
==================
gprof_nn.retrieval
==================

This module contains classes and functionality that drive the execution
of the retrieval.
"""
from functools import cache
import logging
import math
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
import re
from typing import Dict, List, Optional, Union, Tuple

import click
import hdf5plugin
import numpy as np
import xarray as xr

import torch
from torch import nn
import numpy as np
import pandas as pd
from pytorch_retrieve import load_model
from pytorch_retrieve.architectures import MLP, EncoderDecoder
from pytorch_retrieve.inference import InferenceRunner, SequentialInferenceRunner

try:
    from pansat import Granule, FileRecord
except ImportError:
    pass

from gprof_nn.logging import enable_file_logging
from gprof_nn import sensors
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.utils import UPSAMPLING_FACTORS
from gprof_nn.config import CONFIG
from gprof_nn.download import download_model
from gprof_nn.data import preprocessor
from gprof_nn.data.preprocessor import PreprocessorFile, run_preprocessor
from gprof_nn.definitions import ANCILLARY_VARIABLES, ALL_TARGETS, ALL_OUTPUTS
from gprof_nn.data.training_data import (
    EIA_GMI,
    load_tbs_1d_gmi,
    load_tbs_1d_xtrack_sim,
    load_tbs_1d_conical_sim,
    load_tbs_1d_xtrack_other,
    load_tbs_1d_conical_other,
    load_ancillary_data,
    load_training_data_3d_gmi,
    load_training_data_3d_xtrack_sim,
    load_training_data_3d_conical_sim,
    load_training_data_3d_other,
    determine_ancillary_config,
    load_ancillary_data
)


LOGGER = logging.getLogger(__name__)


def get_model(sensor) -> nn.Module:
    """
    Download and load the retrieval model for a given sensor.

    Args:
        sensor: The sensor object representing the sensor.

    Return:
        The loaded retrieval model.
    """
    LOGGER.debug("Loading standard retrieval model for sensor '%s'.", sensor.name)
    return load_model(download_model(sensor))


@cache
def load_scaling_factors(sensor: sensors.Sensor) -> xr.Dataset:
    """
    Load scaling factors for given sensor.

    Args:
        sensor: The sensor object for which to load the scaling factors.

    Return:
        A xarray.Dataset containing the loaded scaling factors.
    """
    filename = f"{sensor.platform.name.upper()}_GPROFV8_pixel_adj.nc"
    data = xr.load_dataset(Path(__file__).parent / "files" / filename)
    return data


def calculate_quality_flag_and_pixel_status(
        sensor: sensors.Sensor,
        tbs_full,
        input_data: xr.Dataset
) -> np.ndarray:
    """
    Calculate the pixel status based on input data.

    Args:
        sensor: The sensor for which the retrieval is performed.
        tbs_full: The full brightness temperature array that will be input to the retrieval.
        input_data: An xarray.Dataset containing the rerieval input data.
    """
    missing_tbs = tbs_full[sensor.gprof_channel_indices]
    any_missing = np.any(np.isnan(missing_tbs), axis=0)
    all_missing = np.all(np.isnan(missing_tbs), axis=0)

    lons = input_data.longitude.data
    lats = input_data.latitude.data
    invalid_coords = ~(
        (-180.0 <= lons) * (lons <= 180.0) *
        (-90.0 <= lats) * (lats <= 90.0)
    )

    tbs_out_of_range = np.any(
        (tbs_full[sensor.gprof_channel_indices] > 350.0) + (tbs_full[sensor.gprof_channel_indices] < 0.0),
        axis=0
    )

    if "snow_mask" in input_data:
        snow_mask = input_data.snow_mask.data
        snow_depth = input_data.snow_depth.data
        ice_fraction = input_data.ice_fraction.data

        snow_or_ice = np.zeros_like(any_missing)
        snow_or_ice[0.0 < snow_mask] = True
        snow_or_ice[0.0 < snow_depth] = True
        snow_or_ice[0.0 < ice_fraction] = True
    else:
        snow_or_ice = np.zeros_like(any_missing)

    status = -99.0 * np.ones_like(any_missing)
    qflag = -99.0 * np.ones_like(any_missing)

    all_good = (~any_missing) * (~invalid_coords) * (~snow_or_ice)
    qflag[snow_or_ice * ~all_missing] = 2
    qflag[any_missing * ~all_missing] = 1
    qflag[all_good] = 0

    status[all_good] = 0
    status[invalid_coords] = 1
    status[tbs_out_of_range] = 2

    return qflag, status


def load_input_data_preprocessor(
        preprocessor_file: Path,
        high_res: bool = False,
        ancillary_config: Optional[str] = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """
    Load retrieval input data from preprocessor file.

    Args:
        preprocessor_file: A path pointing to a preprocessor file.
        ancillary_config: A string specifying the ancillary data configuration to load.

    Return:
        A pair of dictionaries: The first one containing the input tensors
        'brightness_temperatures', 'earth_incidence_angles', and 'ancillary_data',
        the second containing auxilliary data to store in the retrieval output.
    """
    file_pp = PreprocessorFile(preprocessor_file)
    data_pp = file_pp.to_xarray_dataset()
    sensor = file_pp.sensor

    upsampling_factors = UPSAMPLING_FACTORS.get(sensor.name.lower(), (1, 1))
    if high_res and max(upsampling_factors) > 1:
        data_pp = upsample_data(data_pp, upsampling_factors)

    # Brightness temperatures
    tbs = data_pp.brightness_temperatures.data.astype(np.float32)
    tbs[tbs < 0] = np.nan
    if tbs.shape[-1] < 15:
        tbs_full = np.nan * np.zeros((tbs.shape[:2] + (15,)), dtype=np.float32)
        tbs_full[..., sensor.gprof_channel_indices] = tbs
    else:
        tbs_full = tbs.astype(np.float32)
    tbs_full = np.transpose(tbs_full, (2, 0, 1))
    tbs_full[tbs_full < 0] = np.nan

    # Earth incidence angles
    angs_full = np.nan * np.ones_like(tbs_full)
    chan_inds = list(sensor.gprof_channels.keys())
    if isinstance(sensor, sensors.CrossTrackScanner):
        eia = data_pp.earth_incidence_angle.data.astype(np.float32).copy()
        eia[eia < -100] = np.nan
        angs_full[chan_inds] = eia[None]
    else:
        angs_full[chan_inds] = torch.tensor(sensor.earth_incidence_angle[..., None, None], dtype=torch.float32)

    # Ancillary data
    if ancillary_config is None:
        ancillary_config = determine_ancillary_config(data_pp)
    anc = load_ancillary_data(data_pp, ancillary_config, stack_dim=0)

    qflag, status = calculate_quality_flag_and_pixel_status(sensor, tbs_full, data_pp)

    input_data = {
        "brightness_temperatures": tbs_full,
        "earth_incidence_angles": angs_full,
        "ancillary_data": anc
    }

    aux = {
        "sensor": sensor,
        "ancillary_config": ancillary_config,
        "pixel_status": status,
        "quality_flag": qflag,
        "scan_time": data_pp.scan_time.data,
        "longitude": data_pp.longitude.data,
        "latitude": data_pp.latitude.data,
        "total_column_water_vapor": data_pp.total_column_water_vapor.data,
        "two_meter_temperature": data_pp.two_meter_temperature.data,
        "convective_precipitation": data_pp.convective_precipitation.data,
        "moisture_convergence": data_pp.moisture_convergence.data,
        "leaf_area_index": data_pp.leaf_area_index.data,
        "snow_depth": data_pp.snow_depth.data,
        "orographic_wind": data_pp.orographic_wind.data,
        "wind_speed_10m": data_pp["10m_wind"].data,
        "mountain_index": data_pp.mountain_type.data,
        "land_fraction": data_pp.land_fraction.data,
        "ice_fraction": data_pp.ice_fraction.data,
        "elevation": data_pp.elevation.data,
        "snow_mask": data_pp.snow_mask.data,
        "preprocessor_file": file_pp
    }
    return input_data, aux


def load_input_data_l1c(
        l1c_file: Path,
        ancillary_config: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Load retrieval input data from a L1C file.

    Args:
        l1c_file: A path pointing to a L1C file.

    Return:
        A dictionary containing the input tensors 'brightness_temperatures',
        'earth_incidence_angles', and 'ancillary_data'.
    """
    sensor = L1CFile(l1c_file).sensor

    if preprocessor.is_available(sensor):
        try:
            with TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                pp_path = tmp_path / l1c_file.with_suffix(".pp").name
                prodtype = {
                    "STD": "STANDARD",
                    "CLI": "CLIMATOLOGY",
                    "NRT": "NRT"
                }
                settings = {"prodtype": prodtype.get(ancillary_config, "CLIMATOLOGY")}
                if ancillary_config in ["STD", "NRT"]:
                    settings["prepdir"] = "/qdata1/pbrown/gpm/modelprep/GANALV7/"
                else:
                    settings["prepdir"] = "/qdata2/archive/ERA5/"
                run_preprocessor(
                    l1c_file,
                    sensor,
                    output_file=pp_path,
                    settings=settings
                )
                pp_data = load_input_data_preprocessor(pp_path, ancillary_config=ancillary_config)
                return pp_data
        except RuntimeError:
            LOGGER.warning(
                "Encountered and error running the preprocessor. Running retrieval without ancillary "
                "data."
            )
    else:
        LOGGER.warning(
            "No preprocessor found on the current system. Running retrieval without ancillary data."
        )

    l1c_data = L1CFile(l1c_file).to_xarray_dataset()
    tbs = np.transpose(l1c_data.brightness_temperatures.data.astype(np.float32), (2, 0, 1))
    tbs[tbs < 0] = np.nan
    tbs[tbs > 350.0] = np.nan

    if tbs.shape[0] == 15:
        tbs_full = tbs
    else:
        tbs_full = np.nan * np.zeros((15,) + tbs.shape[1:], dtype=np.float32)
        tbs_full[sensor.gprof_channel_indices] = tbs

    anc = np.nan * np.zeros((14,) + tbs.shape[1:])

    angs_full = np.nan * np.ones_like(tbs_full)
    chan_inds = list(sensor.gprof_channels.keys())
    if isinstance(sensor, sensors.CrossTrackScanner):
        eia = l1c_data.earth_incidence_angle.data.astype(np.float32).copy()
        eia[eia < -100] = np.nan
        angs_full[chan_inds] = eia[None]
    else:
        angs_full[chan_inds] = sensor.earth_incidence_angle[..., None, None].astype(np.float32)

    inpt = {
        "brightness_temperatures": tbs_full,
        "ancillary_data": anc,
        "earth_incidence_angles": angs_full
    }

    qflag, status = calculate_quality_flag_and_pixel_status(sensor, tbs_full, l1c_data)
    missing = np.nan * np.zeros_like(tbs_full[0])
    aux = {
        "sensor": sensor,
        "pixel_status": status,
        "quality_flag": qflag,
        "scan_time": l1c_data.scan_time.data,
        "longitude": l1c_data.longitude.data,
        "latitude": l1c_data.latitude.data,
        "total_column_water_vapor": missing,
        "two_meter_temperature": missing,
        "convective_precipitation": missing,
        "moisture_convergence": missing,
        "leaf_area_index": missing,
        "snow_depth": missing,
        "orographic_wind": missing,
        "wind_speed_10m": missing,
        "mountain_index": missing,
        "land_fraction": missing,
        "ice_fraction": missing,
        "elevation": missing,
        "snow_mask": missing,
    }

    return inpt, aux


def load_input_data_training_1d(
        training_file: Path,
        ancillary_config: str
) -> Dict[str, torch.Tensor]:
    """
    Load retrieval input data from a 1D training file.

    Args:
        training_file: A path object pointing to the training file from which
            to load the input data.
        ancillary_config: String specifying the ancillary data configuration tol oad.

    Return:
        A dictionary containing the input tensors 'brightness_temperatures',
        'earth_incidence_angles', and 'ancillary_data'.
    """
    rng = np.random.default_rng(42)

    with xr.open_dataset(training_file) as data:
        sensor = sensors.get_sensor(data.attrs["sensor"])

        if sensor == sensors.GMI:
            tbs = load_tbs_1d_gmi(data)
            anc = load_ancillary_data(data, configuration=ancillary_config, stack_dim=1)
            angs = torch.tensor(np.broadcast_to(EIA_GMI.astype("float32"), tbs.shape))
        elif isinstance(sensor, sensors.CrossTrackScanner):
            if data.attrs["source"] == "sim":
                angles = data["angles"].data
                angs = rng.uniform(
                    angles.min(),
                    angles.max(),
                    size=data.samples.size,
                ).astype(np.float32)
                tbs, _ = load_tbs_1d_xtrack_sim(data, angs, sensor, targets=[])
                angs = torch.tensor(angs)
                angs = torch.tensor(np.broadcast_to(angs[..., None], tbs.shape))
            else:
                tbs, angs = load_tbs_1d_xtrack_other(data, sensor)
            anc = load_ancillary_data(data, configuration="CLI", stack_dim=1)
        elif isinstance(sensor, sensors.ConstellationScanner):
            if data.source == "sim":
                tbs = load_tbs_1d_conical_sim(data, sensor)
                angs = torch.tensor(
                    np.broadcast_to(EIA_GMI.astype("float32"), tbs.shape)
                )
            else:
                tbs, angs = load_tbs_1d_conical_other(data, sensor)
            anc = load_ancillary_data(data, configuration="CLI", stack_dim=1)

        input_data = {
            "brightness_temperatures": tbs,
            "earth_incidence_angles": angs,
            "ancillary_data": anc,
        }

        qflag, status = calculate_quality_flag_and_pixel_status(sensor, tbs, data)

        aux = {
            "sensor": sensor,
            "pixel_status": status,
            "quality_flag": qflag,
            "longitude": data.longitude.data,
            "latitude": data.latitude.data,
        }
        return input_data, aux


def load_input_data_training_3d(
        training_file: Path,
        ancillary_config: str
) -> Dict[str, torch.Tensor]:
    """
    Load retrieval input data from a 3D training file.

    Args:
        training_file: A path object pointing to the training file from which
            to load the input data.
        ancillary_config: A string specifying the ancillary data configuration to load.

    Return:
        A dictionary containing the input tensors 'brightness_temperatures',
        'earth_incidence_angles', and 'ancillary_data'.
    """
    rng = np.random.default_rng(42)

    with xr.open_dataset(training_file) as scene:
        sensor = scene.attrs["sensor"]
        sensor = getattr(sensors, sensor)

        targets_aux = ALL_TARGETS + ["longitude", "latitude"]

        if sensor == sensors.GMI:
            input_data, targets = load_training_data_3d_gmi(
                scene,
                targets=targets_aux,
                augment=None,
                rng=rng
            )
        elif isinstance(sensor, sensors.CrossTrackScanner):
            if scene.source == "sim":
                input_data, targets = load_training_data_3d_xtrack_sim(
                    sensor,
                    scene,
                    targets=targets_aux,
                    augment=None,
                    rng=rng
                )
            else:
                input_data, targets = load_training_data_3d_other(
                    sensor,
                    scene,
                    targets=targets_aux,
                    augment=None,
                    rng=rng
                )
        elif isinstance(sensor, sensors.ConstellationScanner):
            if scene.source == "sim":
                input_data, targets = load_training_data_3d_conical_sim(
                    sensor,
                    scene,
                    targets=targets_aux,
                    augment=None,
                    rng=rng
                )
            else:
                input_data, targets = load_training_data_3d_other(
                    sensor,
                    scene,
                    targets=targets_aux,
                    augment=None,
                    rng=rng
                )

        tbs_full = input_data["brightness_temperatures"]
        #qflag, status = calculate_quality_flag_and_pixel_status(sensor, tbs_full, scene)

        aux = {
            "sensor": sensor,
            #"quality_flag": qflag,
            #"pixel_status": status,
            "longitude": targets.pop("longitude").numpy(),
            "latitude": targets.pop("latitude").numpy(),
        }
        for name, target_data in targets.items():
            aux[name + "_ref"] = target_data.numpy()
        return input_data, aux

    raise RuntimeError(
        "Invalid sensor/scene combination in training file %s.",
        training_file
    )


def load_input_data_collocations(
        collocation_file: Path,
        ancillary_config: str  = "NONE"
) -> Dict[str, np.ndarray]:
    """
    Load retrieval input data from a SPEED collocation file.

    Args:
        collocation_file: A path object pointing to the training file from which
            to load the input data.

    Return:
        A dictionary containing the input tensors 'brightness_temperatures',
        'earth_incidence_angles', and 'ancillary_data'.
    """
    sensor = sensors.get_sensor(collocation_file.name.split("_")[-2].upper())

    with xr.open_dataset(collocation_file, group="input_data") as scene:

        scene = scene.transpose("scan", "pixel", "channel", "channel_gprof", ...)

        tbs = np.transpose(scene.observations_gprof.data.astype(np.float32), (2, 0, 1))
        if tbs.shape[0] < 15:
            tbs_full = np.nan * np.zeros((15,) + tbs.shape[1:])
            tbs_full[sensor.gprof_channel_indices] = tbs
        else:
            tbs_full = tbs
        tbs_full[tbs_full < 0] = np.nan

        # Earth incidence angles
        angs_full = np.nan * np.ones_like(tbs_full)
        chan_inds = list(sensor.gprof_channels.keys())
        if isinstance(sensor, sensors.CrossTrackScanner):
            eia = scene.earth_incidence_angle_gprof.data.astype(np.float32).copy()
            eia[eia < -100] = np.nan
            angs_full[chan_inds] = eia[None]
        else:
            angs_full[chan_inds] = sensor.earth_incidence_angle[..., None, None].astype(np.float32)

        # Ancillary data
        anc = load_ancillary_data(scene, ancillary_config, stack_dim=0)

        input_data = {
            "brightness_temperatures": tbs_full,
            "ancillary_data": anc,
            "earth_incidence_angles": angs_full
        }

        qflag, status = calculate_quality_flag_and_pixel_status(sensor, tbs_full, scene)
        aux = {
            "sensor": sensor,
            "ancillary_config": ancillary_config,
            "pixel_status": status,
            "quality_flag": qflag,
            "longitude": scene.longitude.data,
            "latitude": scene.latitude.data
        }
        return input_data, aux


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
        if path.name.startswith("cmb_") or path.name.startswith("mrms_"):
            return "collocations"
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
            path: Union[str, Path, List[Union[str, Path]]],
            input_format: Optional[str] = None,
            config: str = "3d",
            ancillary_config: str = "CLI",
            output_format: str = "NETCDF",
            bias_correction: bool = True
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

        config = config.lower()
        if not config in ['1d', '3d']:
            raise ValueError(
                "Config must be '1d' for GPROF-NN 1D retrievals or '3d' for GPROF-NN 3D retrievals."
            )
        self.config = config

        self.input_format = input_format
        self.ancillary_config = ancillary_config
        self.output_format = output_format.upper()
        self.bias_correction = bias_correction


    def __len__(self) -> int:
        """
        The number of files to process.
        """
        return len(self.input_files)

    def infer_sensor(self) -> sensors.Sensor:
        """
        Infer sensor from input file.
        """
        inpt = self.input_files[0]
        if inpt.suffix == ".HDF5":
            return L1CFile(inpt).sensor
        elif inpt.suffix == ".pp":
            return PreprocessorFile(inpt).sensor
        elif inpt.suffix == ".nc":
            with xr.open_dataset(inpt) as smpl:
                return sensors.get_sensor(smpl.attrs["sensor"])
        else:
            raise ValueError(
                "Failed to infer sensor from input file '%s'.",
                inpt
            )

    def load_input_data(self, path: Path) -> Dict[str, torch.Tensor]:
        """
        Load retrieval input data.

        Args:
            path: A path object pointing to the file from which to load the input data.

        Return:
            A dictionary mapping the names of the retrieval inputs ('brightness_temperatures',
            'earth_incidence_angles', 'ancillary_data') for tensor containing the corresponding data.
        """
        if self.input_format is None:
            input_format = determine_input_format(path)

        if input_format == "preprocessor":
            LOGGER.debug("Loading input data from input file '%s' in preprocessor format.", path)
            input_data, aux = load_input_data_preprocessor(path, ancillary_config=self.ancillary_config)
        elif input_format == "l1c":
            LOGGER.debug("Loading input data from input file '%s' in L1C format.", path)
            input_data, aux = load_input_data_l1c(path, ancillary_config=self.ancillary_config)
        elif input_format == "training_1d":
            LOGGER.debug("Loading input data from input file '%s' in 1D training data format.", path)
            input_data, aux = load_input_data_training_1d(path, ancillary_config=self.ancillary_config)
        elif input_format == "training_3d":
            LOGGER.debug("Loading input data from input file '%s' in 3D training data format.", path)
            input_data, aux = load_input_data_training_3d(path, ancillary_config=self.ancillary_config)
        elif input_format == "collocations":
            LOGGER.debug("Loading input data from input file '%s' in SPEED collocation format.", path)
            input_data, aux = load_input_data_collocations(path, ancillary_config=self.ancillary_config)
        else:
            raise ValueError(
                f"Encountered unknown input format '{input_format}'."
            )

        input_data = {
            name: (torch.tensor(data) if isinstance(data, np.ndarray) else data)[None]
            for name, data in input_data.items()
        }

        if self.config == "1d":
            input_data = {
                name: torch.permute(tensor, (0, 2, 3, 1)).reshape((-1, tensor.shape[1]))
                if tensor.ndim == 4 else tensor
                for name, tensor in input_data.items()
            }
        aux["output_format"] = self.output_format

        return input_data, aux

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
            filename: str,
            output_path: Path,
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

        sensor = aux.pop("sensor")
        ancillary_config = aux.pop("ancillary_config", "NONE")

        if lons.ndim == 2:
            dims = ("scans", "pixels", "levels")
        else:
            dims = ("samples", "levels")

        output = xr.Dataset()

        preprocessor_file = aux.pop("preprocessor_file", None)
        output_format = aux.pop("output_format", "NETCDF")

        # Copy relevant input data.
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
            tensor = tensor.cpu().float().squeeze()
            if self.config.lower() == "1d":
                tensor = tensor.reshape(shape + tensor.shape[1:])

            if var == "surface_precip_terciles":
                if self.config == "3d":
                    tensor = torch.permute(tensor, (1, 2, 0))
                if lons.ndim < 2:
                    dims_v = ("samples",)
                else:
                    dims_v = ("scans", "pixels")
                output["surface_precip_1st_tercile"] = (
                    dims_v, np.maximum(tensor[..., 0].numpy(), 0.0)
                )
                output["surface_precip_1st_tercile"].encoding = {"dtype": "float32", "zlib": True}
                output["surface_precip_2nd_tercile"] = (
                    dims_v,
                    np.maximum(tensor[..., 1].numpy(), 0.0)
                )
                output["surface_precip_2nd_tercile"].encoding = {"dtype": "float32", "zlib": True}

            else:
                if self.config == "3d" and tensor.dim() > 2:
                    tensor = tensor.squeeze()
                    tensor = torch.permute(tensor, (1, 2, 0))

                dims_v = dims[:tensor.dim()]
                tensor = tensor.numpy()
                if var != "latent_heating":
                    tensor = np.maximum(tensor, 0.0)

                output[var] = (dims_v, tensor)
                # Use compressiong to keep file size reasonable.
                output[var].encoding = {"dtype": "float32", "zlib": True}

        # Apply bias correction

        if self.bias_correction:

            try:
                adjustment_factors = load_scaling_factors(sensor)
            except FileNotFoundError:
                adjustment_factors = None
                LOGGER.warning(
                    "No bias adjustment factors for sensor %s (%s)",
                    sensor.name,
                    sensor.platform.name
                )

            if adjustment_factors is not None:

                land_fraction = output.land_fraction.data
                ice_fraction = output.ice_fraction.data
                snow_mask = output.snow_mask.data
                scaling = np.ones_like(output["surface_precip"].data)

                ocean_mask = (land_fraction <= 2) * (ice_fraction == 0)
                ocean_scaling = adjustment_factors.ocean_bias.data * adjustment_factors.ocean_adj.data
                ocean_scaling = np.broadcast_to(ocean_scaling[None], scaling.shape)
                scaling[ocean_mask] = ocean_scaling[ocean_mask]
                mean_ocean_scaling = ocean_scaling[ocean_mask].mean()

                landrain_mask = (95 < land_fraction) * (snow_mask == 0)
                landrain_scaling = adjustment_factors.landrain_bias.data * adjustment_factors.landrain_adj.data
                landrain_scaling = np.broadcast_to(landrain_scaling[None], scaling.shape)
                scaling[landrain_mask] = landrain_scaling[landrain_mask]
                mean_land_scaling = landrain_scaling[landrain_mask].mean()

                LOGGER.debug(
                    "Applying bias correction (Ocean = %s, Land = %s)",
                    mean_ocean_scaling,
                    mean_land_scaling
                )

                output["surface_precip"].data *= scaling
                output["surface_precip_1st_tercile"].data *= scaling
                output["surface_precip_2nd_tercile"].data *= scaling
        else:
            LOGGER.debug("Skipping bias correction.")

        qflag = aux["quality_flag"]
        for name in ALL_OUTPUTS:
            if name in output:
                output[name].data[qflag < 0] = np.nan

        for var in output:
            var_data = output[var].data
            if np.issubdtype(var_data.dtype, np.floating):
                output[var].data = np.nan_to_num(output[var].data, nan=-9999.9)
            else:
                output[var].data = np.nan_to_num(output[var].data, nan=-99)

        output.attrs["ancillary_config"] = ancillary_config

        LOGGER.debug("Successfully processed %s scans.", output.scans.size)

        if output_format.upper() == "NETCDF":
            # Quick and dirty way to transform 1C filename to 2A filename
            if output_path is None:
                return output

            if output_path.is_dir():
                output_filename = (
                    filename.replace("1C-R", "2A")
                    .replace("1C", "2A")
                    .replace("pp", "nc")
                    .replace("HDF5", "nc")
                )
            else:
                output_filename = output_path.name

            LOGGER.debug("Writing retrieval results in NetCDF format to '%s'.", output_filename)
            # Return outputs as xr.Dataset and filename to use to save data.
            return output, output_filename


        # Output format is binary
        LOGGER.debug("Writing retrieval results in binary format to '%s'.", output_path)
        suffix = {
            "STD": "STD",
            "CLI": "CLIM",
            "NRT": "NRT"
        }.get(self.ancillary_config, "")
        return preprocessor_file.write_retrieval_results(output_path, output, suffix=suffix)


def run_retrieval(
        input_path: Path,
        output_path: Optional[Path] = None,
        device: str = "cpu",
        dtype: str = "float32",
        ancillary_config: Optional[str] = None,
        output_format: str = "NETCDF",
        n_input_loaders: int = 1,
        retrieval_model: Optional[str] = None,
        batch_size: Optional[int] = None,
        no_profiles: bool = False
) -> Union[List[xr.Dataset], List[Path]]:
    """
    Run GPROF-NN retrieval.

    Args:
        input_path: A path object pointing to a single input file or a directory tree
            containing multiple input files.
        output_path: If given, output files will written to the given directory.
        device: A string identifying the torch device to run the retrieval on.
        dtype: A string identifying the floating point type to use to run the
            retrieval.
        ancillary_config: Optional ancillary configuration to use to run the retrieval.
            Shoule be one of ['NONE', 'NRT', 'STD', 'CLI']. NOTE: This only has
            an effect if run on preprocessor files or if the GPROF preprocessor
            is available.
        output_format: The output format to use for the output files. Should be one
            of ['NETCDF', 'BINARY']
        n_input_loaders: The number of processes to use to load the input data.
        retrieval_model: Optional path to an existing retrieval model. If not given
            the default retrieval for each sensor is used.
        no_profiles: Set to 'True' to disable calculation of profiles.

    Return:
        If no 'output_path' is given, will return a list of xarray.Datasets containing
        the retrieval results for all files found in the 'input_path'. If 'output_path'
        is given, will return a list of path objects pointing to the files containing
        the results for each input file found in 'input_path'.
    """
    if retrieval_model is None:
        input_loader = GPROFNNInputLoader(
            input_path,
            config="3d",
            ancillary_config=ancillary_config,
            output_format=output_format
        )
        sensor = input_loader.infer_sensor()
        model = get_model(sensor).eval()
    else:
        model = load_model(retrieval_model).eval()
        if isinstance(model, MLP):
            config = "1d"
        elif isinstance(model, EncoderDecoder):
            config = "3d"
        else:
            raise ValueError(
                f"Encountered unsupported model type '{type(model)}'.",
            )

        input_loader = GPROFNNInputLoader(
            input_path,
            config=config,
            ancillary_config=ancillary_config,
            output_format=output_format
        )

    if batch_size is not None:
        model.inference_config.batch_size = batch_size

    if no_profiles:
        profiles = [
            "snow_water_content",
            "rain_water_content",
            "cloud_water_content",
            "latent_heat"
        ]
        iconf = model.inference_config
        retrieval_output = {
            name: outputs for name, outputs in iconf.retrieval_output.items()
            if not name in profiles
        }
        model.inference_config.retrieval_output = retrieval_output
        for prof in profiles:
            model.heads.pop(prof)
    inference_config = model.inference_config

    if output_path is not None:
        output_path = Path(output_path)

    device = torch.device(device)
    dtype = getattr(torch, dtype)

    if n_input_loaders > 1:
        runner = InferenceRunner(
            model,
            input_loader,
            inference_config,
            n_input_loaders=n_input_loaders,
        )
    else:
        runner = SequentialInferenceRunner(
            model,
            input_loader,
            inference_config,
        )
    return runner.run(output_path=output_path, device=device, dtype=dtype)


@click.argument("input_path", type=str)
@click.option(
    "--output_path",
    type=str,
    metavar="PATH",
    default=None,
    help=(
        "A directory to which write the retrieval results to."
    )
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help=(
        "The torch device on which to perform inference, i.e., 'cpu', 'cuda', etc."
    )
)
@click.option(
    "--dtype",
    type=str,
    default="float32",
    help=(
        "The floating point type to use for inference."
    )
)
@click.option(
    "--ancillary_config",
    type=str,
    default="CLI",
    help=(
        "The ancillary data configuration to use for inference."
    )
)
@click.option(
    "--output_format",
    type=str,
    default="netcdf",
    help=(
        "The format to use to store the retrieval results. Shoule be 'netcdf' for NetCDF4 format"
        " (default) or 'binary' for GPROF binary format."
    )
)
@click.option(
    "--n_input_loaders",
    type=int,
    default=1,
    help=(
        "The number of processes to use to load the input data."
    )
)
@click.option(
    "--retrieval_model",
    type=str,
    help="Path pointing to a model file to use for the retrieval."
)
@click.option(
    "--no_profiles",
    is_flag=True
)
@click.option(
    "--log_file",
    type=str,
    help="Log retrieval progress to the given file."
)
@click.option(
    "--no_bias_correction",
    is_flag=True
)
def cli(
        input_path: Path,
        output_path: Optional[Path] = None,
        device: str = "cpu",
        dtype: str = "float32",
        ancillary_config: str = "CLI",
        output_format: str = "NETCDF",
        n_input_loaders: int = 1,
        retrieval_model: Optional[str] = None,
        no_profiles: bool = False,
        log_file: Optional[str] = None,
        no_bias_correction: bool = False
) -> None:
    """
    Run the GPROF-NN retrieval on a single input file or a folder of input files located at INPUT_PATH and write the results to the current working directory.
    """
    if log_file is not None:
        enable_file_logging(log_file)

    if retrieval_model is None:
        input_loader = GPROFNNInputLoader(
            input_path,
            config="3d",
            ancillary_config=ancillary_config,
            output_format=output_format,
            bias_correction=not no_bias_correction
        )
        sensor = input_loader.infer_sensor()
        model = get_model(sensor)
    else:
        try:
            LOGGER.debug("Loading retrieval model %s.", retrieval_model)
            model = load_model(retrieval_model).eval()
        except Exception:
            LOGGER.exception(
                "Encountered the following error when trying to load the model from "
                " file '%s'.",
                model
            )
            return 1

        if isinstance(model, MLP):
            config = "1d"
        elif isinstance(model, EncoderDecoder):
            config = "3d"
        else:
            LOGGER.error(
                "Encountered unsupported model type '%s'.",
                type(model)
            )
            return 1

        input_loader = GPROFNNInputLoader(
            input_path,
            config=config,
            ancillary_config=ancillary_config,
            output_format=output_format
        )

    if no_profiles:
        profiles = [
            "snow_water_content",
            "rain_water_content",
            "cloud_water_content",
            "latent_heat"
        ]
        iconf = model.inference_config
        retrieval_output = {
            name: outputs for name, outputs in iconf.retrieval_output.items()
            if not name in profiles
        }
        model.inference_config.retrieval_output = retrieval_output
        for prof in profiles:
            model.heads.pop(prof)

    inference_config = model.inference_config

    if output_path is None:
        output_path = Path(".")
    else:
        output_path = Path(output_path)

    device = torch.device(device)
    dtype = getattr(torch, dtype)

    if n_input_loaders > 1:
        runner = InferenceRunner(
            model,
            input_loader,
            inference_config,
            n_input_loaders=n_input_loaders,
        )
    else:
        runner = SequentialInferenceRunner(
            model,
            input_loader,
            inference_config,
        )
    runner.run(output_path=output_path, device=device, dtype=dtype)
