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

import click
import hdf5plugin
import numpy as np
import xarray as xr

import torch
from torch import nn
import numpy as np
from pansat import Granule, FileRecord
import pandas as pd
from pytorch_retrieve import load_model
from pytorch_retrieve.architectures import MLP, EncoderDecoder
from pytorch_retrieve.inference import InferenceRunner

import gprof_nn.logging
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
    load_ancillary_data,
    load_training_data_3d_gmi,
    load_training_data_3d_xtrack_sim,
    load_training_data_3d_conical_sim,
    load_training_data_3d_other,
    determine_ancillary_config,
    load_ancillary_data
)
from gprof_nn.data.utils import (
    upsample_data,
    add_cpcir_data,
    calculate_obs_properties,
    PANSAT_PRODUCTS,
    RADIUS_OF_INFLUENCE,
    UPSAMPLING_FACTORS
)


LOGGER = logging.getLogger(__name__)



def load_input_data_preprocessor(
        preprocessor_file: Path,
        high_res: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """
    Load retrieval input data from preprocessor file.

    Args:
        preprocessor_file: A path pointing to a preprocessor file.

    Return:
        A pair of dictionaries: The first one containing the input tensors
        'brightness_temperatures', 'earth_incidence_angles', and 'ancillary_data',
        the second containing auxilliary data to store in the retrieval output.
    """
    file_pp = PreprocessorFile(preprocessor_file)
    data_pp = file_pp.to_xarray_dataset()
    sensor = file_pp.sensor

    upsampling_factors = UPSAMPLING_FACTORS[sensor.name.lower()]
    if high_res and max(upsampling_factors) > 1:
        data_pp = upsample_data(data_pp, upsampling_factors)

    tbs = data_pp.brightness_temperatures.data.astype(np.float32)
    tbs[tbs < 0] = np.nan
    if tbs.shape[-1] < 15:
        tbs_full = np.nan * np.zeros((tbs.shape[:2] + (15,)), dtype=np.float32)
        tbs_full[..., sensor.gprof_channel_indices] = tbs
    else:
        tbs_full = tbs.astype(np.float32)
    tbs_full = np.transpose(tbs_full, (2, 0, 1))

    eia = data_pp.earth_incidence_angle.data.astype(np.float32)
    if eia.ndim == 2:
        eia = eia[..., None]

    if eia.shape[-1] < 15:
        angs_full = np.nan * np.zeros(eia.shape[:2] + (15,), dtype=np.float32)
        angs_full[..., sensor.gprof_channel_indices] = eia
    else:
        angs_full = eia.astype(np.float32)
    angs_full = np.transpose(angs_full, (2, 0, 1))

    cfg = determine_ancillary_config(data_pp)
    anc = load_ancillary_data(data_pp, cfg, stack_dim=0)

    tbs_full[tbs_full < 0] = np.nan

    input_data = {
        "brightness_temperatures": tbs_full,
        "earth_incidence_angles": angs_full,
        "ancillary_data": anc
    }
    aux = {
        "valid_input": (np.isfinite(tbs_full)).any(0),
        "scan_time": data_pp.scan_time.data,
        "longitude": data_pp.longitude.data,
        "latitude": data_pp.latitude.data,
        "total_column_water_vapor": data_pp.total_column_water_vapor.data,
        "two_meter_temperature": data_pp.two_meter_temperature.data,
        "convective_fraction": data_pp.convective_precipitation.data,
        "moisture_convergence": data_pp.moisture_convergence.data,
        "leaf_area_index": data_pp.leaf_area_index.data,
        "snow_depth": data_pp.snow_depth.data,
        "orographic_wind": data_pp.orographic_wind.data,
        "wind_speed_10m": data_pp["10m_wind"].data,
        "mountain_index": data_pp.mountain_type.data,
        "land_fraction": data_pp.land_fraction.data,
        "ice_fraction": data_pp.ice_fraction.data,
        "elevation": data_pp.elevation.data
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
        'earth_incidence_angles', and 'ancillary_data'.
    """
    sensor = L1CFile(l1c_file).sensor
    if needs_ancillary:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pp_path = tmp_path / l1c_file.with_suffix(".pp").name
            run_preprocessor(l1c_file, sensor, output_file=pp_path)
            pp_data = load_input_data_preprocessor(pp_path)
            return pp_data

    l1c_data = L1CFile(l1c_file).to_xarray_dataset()
    tbs = torch.tensor(
        np.transpose(l1c_data.brightness_temperatures.data.astype(np.float32), (2, 0, 1))
    )
    tbs[tbs < 0] = np.nan
    valid = torch.isfinite(tbs).any(0).numpy()
    aux = {
        "valid_input": valid,
        "scan_time": l1c_data.scan_time.data,
        "latitude": l1c_data.latitude.data,
        "longitude": l1c_data.longitude.data,
    }
    return {"brightness_temperatures": tbs}, aux


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
        valid  = torch.isfinite(tbs).any(0).numpy()
        aux = {
            "valid_input": valid,
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

        tbs = input_data["brightness_temperatures"].numpy()
        valid = torch.isfinit(tbs).any(0).numpy()
        aux = {
            "valid_input": valid,
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
        collocation_file: Path
) -> Dict[str, torch.Tensor]:
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

        tbs = torch.tensor(np.transpose(scene.observations_gprof.data.astype(np.float32), (2, 0, 1)))
        tbs_full = torch.nan * torch.zeros((15,) + tbs.shape[1:])
        tbs_full[sensor.gprof_channel_indices] = tbs
        tbs_full[tbs_full < 0] = np.nan

        valid = torch.isfinite(tbs_full).any(0)

        anc = []
        for anc_var in ANCILLARY_VARIABLES:
            anc.append(torch.tensor(scene[anc_var].data.astype(np.float32)))
        anc = torch.stack(anc)
        valid = valid * (anc > -9_000).all(0)

        eia = torch.tensor(scene.earth_incidence_angle_gprof.data.astype(np.float32))
        if eia.ndim == 2:
            eia = eia[None]
        eia_full = torch.nan * torch.zeros((15,) + tbs.shape[1:])
        eia_full[:] = eia

        input_data = {
            "brightness_temperatures": tbs_full,
            "ancillary_data": anc,
            "earth_incidence_angles": eia_full
        }
        aux = {
            "valid_input": valid,
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
            path: str | Path | List[str | Path],
            input_format: Optional[str] = None,
            config: str = "3d",
            needs_ancillary: bool = True,
            ancillary_config: Optional[str] = None
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
        if ancillary_config is None:
            ancillary_config = "CLI"
        self.ancillary_config = ancillary_config


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
        if self.input_format is None:
            input_format = determine_input_format(path)

        if input_format == "preprocessor":
            input_data, aux = load_input_data_preprocessor(path, self.ancillary_config)
        elif input_format == "l1c":
            input_data, aux = load_input_data_l1c(path)
        elif input_format == "training_1d":
            input_data, aux = load_input_data_training_1d(path, self.ancillary_config)
        elif input_format == "training_3d":
            input_data, aux = load_input_data_training_3d(path, self.ancillary_config)
        elif input_format == "collocations":
            input_data, aux = load_input_data_collocations(path)
        else:
            raise ValueError(
                f"Encountered unknown input format '{input_format}'."
            )

        input_data = {
            name: torch.tensor(data)[None] for name, data in input_data.items()
        }

        if self.config == "1d":
            input_data = {
                name: torch.permute(tensor, (0, 2, 3, 1)).reshape((-1, tensor.shape[1]))
                if tensor.ndim == 4 else tensor
                for name, tensor in input_data.items()
            }

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
            tensor = tensor.squeeze()
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
                    dims_v, tensor[..., 0].numpy()
                )
                output["surface_precip_1st_tercile"].encoding = {"dtype": "float32", "zlib": True}
                output["surface_precip_2nd_tercile"] = (
                    dims_v,
                    tensor[..., 1].numpy()
                )
                output["surface_precip_2nd_tercile"].encoding = {"dtype": "float32", "zlib": True}

            else:
                if self.config == "3d" and tensor.dim() > 2:
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


class GPROFNNHRInputLoader:
    def __init__(
            self,
            path: str | Path | List[str | Path],
    ):

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
            print(ind, np.nanmin(obs), np.nanmax(obs))
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


@click.argument("retrieval_model", type=str,)
@click.argument("input_path", type=str)
@click.option(
    "--output_path",
    type=str,
    metavar="PATH",
    default=None,
    help=(
        "An optional destination to which to write the inference results."
    )
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help=(
        "The device on which to perform inference."
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
    default=None,
    help=(
        "The ancillary data configuration to use for inference."
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
def cli(
        retrieval_model: str,
        input_path: Path,
        output_path: Optional[Path] = None,
        device: str = "cpu",
        dtype: str = "float32",
        ancillary_config: Optional[str] = None,
        n_input_loaders: int = 1,
) -> None:
    """
    Run GPROF-NN retrieval using the retrieval model RETRIEVAL_MODEL on all input
    files located in INPUT_PATH.
    """
    try:
        model = load_model(retrieval_model).eval()
    except Exception:
        LOGGER.exception(
            "Encountered the following error when trying to load the model from "
            " file '%s'.",
            model
        )
        return 1

    inference_config = model.inference_config

    if isinstance(model, MLP):
        config = "1d"
    elif isinstance(model, EncoderDecoder):
        config = "3d"
    else:
        config = "hr"

    if config == "hr":
        input_loader = GPROFNNHRInputLoader(input_path, ancillary_config=ancillary_config)
    else:
        input_loader = GPROFNNInputLoader(input_path, config=config)

    if output_path is None:
        output_path = Path(".")
    else:
        output_path = Path(output_path)

    device = torch.device(device)
    dtype = getattr(torch, dtype)

    runner = InferenceRunner(
        model,
        input_loader,
        inference_config,
        n_input_loaders=n_input_loaders,
    )
    runner.run(output_path=output_path, device=device, dtype=dtype)
