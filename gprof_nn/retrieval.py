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

import numpy as np
import xarray as xr

import torch
from torch import nn
import pandas as pd

from gprof_nn import sensors
from gprof_nn.definitions import PROFILE_NAMES, ALL_TARGETS
from gprof_nn.data import get_profile_clusters
from gprof_nn.data.training_data import (
    GPROF_NN_1D_Dataset,
    GPROF_NN_3D_Dataset,
    decompress_and_load,
    _THRESHOLDS,
)
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.preprocessor import PreprocessorFile, run_preprocessor
from gprof_nn.data.utils import load_variable


LOGGER = logging.getLogger(__name__)


###############################################################################
# Helper functions.
###############################################################################


def expand_tbs(tbs):
    """
    Helper functions to expand GMI observations to the 15 channels.

    The GMI preprocessor as well as the simulator all produce observation
    data with 15 channels for GMI with two of them containing only missing
    values. Since the GPROF-NN networks expect 15 channel as input, data
    that comes directly from a L1C file must extended accordingly.

    Args:
        tbs: An array containing 13 brightness temperatures of GMI
            oriented along its last axis.

    Return:
        Array containing the same observations but with two empty
        chanels added at indices 5 and 12.
    """
    tbs_e = np.zeros(tbs.shape[:-1] + (15,), dtype=np.float32)
    tbs_e[..., :5] = tbs[..., :5]
    tbs_e[..., 5] = np.nan
    tbs_e[..., 6:12] = tbs[..., 5:11]
    tbs_e[..., 12] = np.nan
    tbs_e[..., 13:] = tbs[..., 11:]
    return tbs_e


def calculate_padding_dimensions(t):
    """
    Calculate list of PyTorch padding values to extend the spatial
    dimension of input tensor to multiples of 32.

    Args:
        t: The ``torch.Tensor`` to pad.

    Return
        A tuple ``(p_l_n, p_r_n, p_l_m, p_r_m)`` containing the
        left and right padding  for the second to last dimension
        (``p_l_m, p_r_m``) and for the last dimension (``p_l_n, p_r_n``).
    """
    shape = t.shape

    n = shape[-1]
    d_n = math.ceil(n / 32) * 32 - n
    p_l_n = d_n // 2
    p_r_n = d_n - p_l_n

    m = shape[-2]
    d_m = math.ceil(m / 32) * 32 - m
    p_l_m = d_m // 2
    p_r_m = d_m - p_l_m
    return (p_l_n, p_r_n, p_l_m, p_r_m)


def combine_input_data_1d(dataset, sensor):
    """
    Combine retrieval input data into input matrix for the single-pixel
    retrieval.

    Args:
        dataset: ``xarray.Dataset`` containing the input variables.
        v_tbs: Name of the variable to load the brightness temperatures
            from.
        sensor: The sensor object representing the sensor from which the
            data stems.

    Return:
        Rank-2 input tensor containing the input data with features oriented
        along  axis 1.
    """
    n_chans = sensor.n_chans

    tbs = dataset["brightness_temperatures"].data.copy()
    # Input from L1C file has only 13 channels.
    if sensor == sensors.GMI and tbs.shape[-1] < n_chans:
        tbs = expand_tbs(tbs)
    tbs = tbs.reshape(-1, n_chans)
    invalid = (tbs > 500.0) + (tbs < 0.0)
    tbs[invalid] = np.nan
    features = [tbs]

    if "two_meter_temperature" in dataset.variables:
        t2m = load_variable(dataset, "two_meter_temperature")
        t2m = t2m.reshape(-1, 1)
        tcwv = load_variable(dataset, "total_column_water_vapor")
        tcwv = tcwv.reshape(-1, 1)

        st = dataset["surface_type"].data.ravel()
        n_types = 18
        st_1h = np.zeros((st.shape[0], n_types), dtype=np.float32)
        for j in range(18):
            st_1h[:, j][st == j + 1] = 1.0

        at = np.maximum(dataset["airmass_type"].data, 0.0)
        at = at.ravel()
        n_types = 4
        at_1h = np.zeros((st.shape[0], n_types), dtype=np.float32)
        for j in range(4):
            at_1h[:, j][at == j] = 1.0

        features += [t2m, tcwv, st_1h, at_1h]

    if isinstance(sensor, sensors.CrossTrackScanner):
        va = dataset["earth_incidence_angle"].data
        features.insert(1, va.reshape(-1, 1))

    x = np.concatenate(features, axis=1)
    x[:, :n_chans][x[:, :n_chans] < 0] = np.nan
    return x


def combine_input_data_3d(dataset, sensor, v_tbs="brightness_temperatures"):
    """
    Combine retrieval input data into input tensor format for convolutional
    retrieval.

    Args:
        dataset: ``xarray.Dataset`` containing the input variables.
        v_tbs: Name of the variable to load the brightness temperatures
            from.
        sensor: The sensor object representing the sensor from which the
            data stems.
        v_tbs: Name of the variable to load as brightness temperatures.

    Return:
        Rank-4 input tensor containing the input data with features oriented
        along axis 1.
    """
    n_chans = sensor.n_chans

    tbs = dataset[v_tbs][:].data
    if tbs.shape[-1] < n_chans:
        tbs = expand_tbs(tbs)

    invalid = (tbs > 500.0) + (tbs < 0.0)
    tbs[invalid] = np.nan

    features = [tbs]
    if "two_meter_temperature" in dataset:
        # 2m temperature
        t2m = load_variable(dataset, "two_meter_temperature")[..., np.newaxis]
        # Total precipitable water.
        tcwv = load_variable(dataset, "total_column_water_vapor")[..., np.newaxis]

        # Surface type
        st = dataset["surface_type"][:].data
        n_types = 18
        shape = tbs.shape[:-1]
        st_1h = np.zeros(shape + (n_types,), dtype=np.float32)
        for i in range(n_types):
            indices = st == (i + 1)
            st_1h[indices, i] = 1.0
        # Airmass type
        # Airmass type is defined slightly different from surface type in
        # that there is a 0 type.
        am = dataset["airmass_type"][:].data
        n_types = 4
        am_1h = np.zeros(shape + (n_types,), dtype=np.float32)
        for i in range(n_types):
            indices = am == i
            am_1h[indices, i] = 1.0
        am_1h[am < 0, 0] = 1.0

        features += [t2m, tcwv, st_1h, am_1h]

    if isinstance(sensor, sensors.CrossTrackScanner):
        va = dataset["earth_incidence_angle"].data
        features.insert(1, va[..., np.newaxis])

    input_data = np.concatenate(features, axis=-1)
    input_data = input_data.astype(np.float32)
    if input_data.ndim < 4:
        input_data = np.expand_dims(input_data, 0)
    input_data = np.transpose(input_data, (0, 3, 1, 2))
    return input_data


def run_preprocessor_l1c(l1c_file, output_file):
    """
    Run preprocessor on L1C file.

    Args:
        l1c_file: Path pointing to the L1C file on which to run the
            preprocessor.
        output_file: The file to which to write the results.

    Raises:
        subprocess.CalledProcessError when running the preprocessor fails.
    """
    sensor = L1CFile(l1c_file).sensor
    try:
        LOGGER.info("Running preprocessor for input file %s.", l1c_file)
        run_preprocessor(l1c_file, sensor, output_file=output_file, robust=False)
    except subprocess.CalledProcessError as e:
        LOGGER.error(
            (
                "Running the preprocessor for file %s failed with the "
                "following error failed.\n%s\n%s"
            ),
            l1c_file,
            e.stdout,
            e.stderr,
        )
        raise e


###############################################################################
# Retrieval driver
###############################################################################


GPROF_BINARY = "GPROF_BINARY"
L1C = "L1C"
NETCDF = "NETCDF"


class RetrievalDriver:
    """
    The ``RetrievalDriver`` class implements the logic for running the GPROF-NN
    retrieval using different neural network models and writing output to
    different formats.
    """

    def __init__(
        self,
        input_file,
        model,
        output_file=None,
        device="cpu",
        compress=True,
        preserve_structure=False,
        sensor=None,
    ):
        """
        Create retrieval driver.

        Args:
            input_file: Path to the preprocessor or NetCDF file containing the
                input data for the retrieval.
            model: The neural network to use for the retrieval.
            output_file: If given and output format is not 'GPROF_BINARY' the
                retrieval results will be written to this file.
            device: String identifying the device to run the retrieval on.
                "cpu" to run on CPU or "cuda:i" to run on the ith GPU.
            compress: If set to ``True`` NetCDF output will be gzipped.
            preserve_structure: Special option that will ensure that the
                spatial structure is conserved even for the 1D retrieval.
            sensor: Optional sensor argument to provide to the NetCDF loader.
        """
        input_file = Path(input_file)
        self.input_file = input_file
        self.model = model
        self.compress = compress

        # Determine input format.
        suffix = input_file.suffix
        if suffix in [".pp", ".bin"]:
            self.input_format = GPROF_BINARY
        elif suffix.endswith("HDF5"):
            self.input_format = L1C
        else:
            self.input_format = NETCDF

        # Determine output format.
        if output_file is None or Path(output_file).is_dir():
            self.output_format = self.input_format
        else:
            if self.input_format in [NETCDF, L1C]:
                self.output_format = NETCDF
            else:
                output_file = Path(output_file)
                if output_file.suffix.lower() == ".bin":
                    self.output_format = GPROF_BINARY
                else:
                    self.output_format = NETCDF

        output_suffix = ".BIN"
        if self.output_format == NETCDF:
            output_suffix = ".nc"

        # Determine output filename.
        if output_file is None:
            if self.output_format == GPROF_BINARY:
                self.output_file = self.input_file.parent
            else:
                if suffix != ".gz":
                    self.output_file = Path(
                        self.input_file.name.replace(suffix, output_suffix)
                    )
                else:
                    self.output_file = input_file.name
                if self.output_file == self.input_file:
                    raise ValueError(
                        f"The provided input and output file ('{input_file}') "
                        "are the same, which could cause data loss. Aborting."
                    )
        elif Path(output_file).is_dir():
            if self.output_format == GPROF_BINARY:
                self.output_file = Path(output_file)
            else:
                if suffix != ".gz":
                    self.output_file = Path(output_file) / self.input_file.name.replace(
                        suffix, output_suffix
                    )
                else:
                    self.output_file = Path(output_file) / self.input_file.name
        else:
            self.output_file = output_file

        self.device = device
        self.preserve_structure = preserve_structure
        self.sensor = sensor

    def _load_input_data(self):
        """
        Load retrieval input data.

        Return:
            If the input data was successfully loaded the input data
            object is returned. ``None`` otherwise.
        """
        # Load input data.
        if self.input_format in [GPROF_BINARY, L1C]:
            input_data = self.model.preprocessor_class(
                self.input_file, self.model.normalizer
            )
        else:
            loader_class = self.model.netcdf_class
            if loader_class == NetcdfLoader1D and self.preserve_structure:
                loader_class = NetcdfLoader1DFull
            input_data = loader_class(
                self.input_file, normalizer=self.model.normalizer, sensor=self.sensor
            )
        return input_data

    def _run(self, xrnn, input_data):
        """
        Batch-wise processing of retrieval input.

        Args:
            xrnn: A quantnn neural-network model.
            input_data: Iterable ``input_data`` object providing access to
                batched input data.

        Return:
            A dataset containing the concatenated retrieval results for all
            batches.
        """
        means = {}
        precip_1st_tercile = []
        precip_2nd_tercile = []
        pop = []
        samples = []

        device = self.device

        with torch.no_grad():
            xrnn.model.to(device)
            for i in range(len(input_data)):
                x = input_data[i]
                x = x.float().to(device)
                y_pred = xrnn.predict(x)
                if not isinstance(y_pred, dict):
                    y_pred = {"surface_precip": y_pred}

                y_mean = xrnn.posterior_mean(y_pred=y_pred)
                for k, y in y_pred.items():
                    means.setdefault(k, []).append(y_mean[k].cpu())
                    if k == "surface_precip":
                        t = xrnn.posterior_quantiles(
                            y_pred=y, quantiles=[0.333, 0.667], key=k
                        )
                        precip_1st_tercile.append(t[:, 0].cpu())
                        precip_2nd_tercile.append(t[:, 1].cpu())
                        p = xrnn.probability_larger_than(y_pred=y, y=1e-4, key=k)
                        pop.append(p.cpu())
                        s = xrnn.sample_posterior(
                            y_pred=y, n_samples=1, key="surface_precip"
                        )[:, 0]
                        samples.append(s.cpu())

        data = {}
        for k in means:
            y = np.concatenate([t.numpy() for t in means[k]])
            data[k] = (input_data.dimensions[k], y)

        if len(precip_1st_tercile) > 0:
            dims = input_data.dimensions["surface_precip"]
            data["precip_1st_tercile"] = (
                dims,
                np.concatenate([t.numpy() for t in precip_1st_tercile]),
            )
            data["precip_2nd_tercile"] = (
                dims,
                np.concatenate([t.numpy() for t in precip_2nd_tercile]),
            )
            pop = np.concatenate([t.numpy() for t in pop])
            data["pop"] = (dims, pop)
            data["most_likely_precip"] = data["surface_precip"]
            data["precip_flag"] = (dims, pop > 0.5)
            samples = np.concatenate([t.numpy() for t in samples])
            data["surface_precip_samples"] = (dims, samples)
        data = xr.Dataset(data)

        xrnn.model.cpu()

        return input_data.finalize(data)

    def run(self):
        """
        Run retrieval and store results to file.

        Return:
            Name of the output file that the results have been written
            to.
        """
        input_data = self._load_input_data()
        if input_data is None:
            return None

        LOGGER.info("Running retrieval for file '%s'.", self.input_file)
        results = self._run(self.model, input_data)

        # Make sure output folder exists.
        folder = Path(self.output_file).parent
        folder.mkdir(parents=True, exist_ok=True)

        ancillary_data = get_profile_clusters()
        if self.output_format == GPROF_BINARY:
            return input_data.write_retrieval_results(
                self.output_file,
                results,
                ancillary_data=ancillary_data,
                suffix=self.model.suffix,
            )

        # Output format is NetCDF.
        # Include some inputs in result.
        if hasattr(input_data, "data"):
            variables = [
                "latitude",
                "longitude",
                "total_column_water_vapor",
                "two_meter_temperature",
                "surface_type",
                "airmass_type",
            ]
            for var in variables:
                if var in input_data.data.variables:
                    results[var] = input_data.data[var]

        LOGGER.info("Writing retrieval results to file '%s'.", self.output_file)
        results.to_netcdf(self.output_file)
        if self.compress:
            LOGGER.info("Compressing file '%s'.", self.output_file)
            subprocess.run(["gzip", "-f", self.output_file], check=True)
            output_file = self.output_file.with_suffix(".nc.gz")
        else:
            output_file = self.output_file

        return output_file


class RetrievalGradientDriver(RetrievalDriver):
    """
    Specialization of ``RetrievalDriver`` that retrieves only surface precipitation
    and its gradients with respect to the input variables.
    """

    N_CHANNELS = 15

    def __init__(self, input_file, model, output_file=None, compress=True):
        """
        Create retrieval driver.

        Args:
            input_file: Path to the preprocessor or NetCDF file containing the
                input data for the retrieval.
            model: The neural network to use for the retrieval.
            output_file: If given and output format is not 'GPROF_BINARY' the
                retrieval results will be written to this file.
            compress: If set to ``True`` NetCDF output will be gzipped.
        """
        ancillary_data = get_profile_clusters()
        super().__init__(input_file, model, output_file=output_file, compress=compress)

    def _run(self, xrnn, input_data):
        """
        Batch-wise processing of retrieval input.

        Args:
            xrnn: A quantnn neural-network model.
            input_data: Iterable ``input_data`` object providing access to
                batched input data.

        Return:
            A dataset containing the concatenated retrieval results for all
            batches.
        """
        means = {}
        gradients = {}
        precip_1st_tercile = []
        precip_2nd_tercile = []
        pop = []
        samples = []

        device = next(iter(xrnn.model.parameters())).device
        for i in range(len(input_data)):
            x = input_data[i]
            x = x.float().to(device)
            x.requires_grad = True
            y_pred = xrnn.predict(x)
            if not isinstance(y_pred, dict):
                y_pred = {"surface_precip": y_pred}

            y_mean = xrnn.posterior_mean(y_pred=y_pred)
            grads = {}
            for k in y_pred:
                if k == "surface_precip":
                    xrnn.model.zero_grad()
                    dims = list(range(2, y_mean[k].ndim))
                    y_mean_sum = y_mean[k].sum(dims)
                    y_mean_sum.backward(torch.ones_like(y_mean_sum))
                    grads[k] = x.grad

            for k, y in y_pred.items():
                means.setdefault(k, []).append(y_mean[k].detach().cpu())
                if k in grads:
                    gradients.setdefault(k, []).append(grads[k].detach().cpu())
                if k == "surface_precip":
                    y = y.detach()
                    t = xrnn.posterior_quantiles(
                        y_pred=y, quantiles=[0.333, 0.667], key=k
                    )
                    precip_1st_tercile.append(t[:, 0].cpu())
                    precip_2nd_tercile.append(t[:, 1].cpu())
                    p = xrnn.probability_larger_than(y_pred=y, y=1e-4, key=k)
                    pop.append(p.cpu())
                    s = xrnn.sample_posterior(
                        y_pred=y, n_samples=1, key="surface_precip"
                    )[:, 0]
                    samples.append(s.cpu())

        dims = input_data.scalar_dimensions
        dims_p = input_data.profile_dimensions

        data = {}
        for k in means:
            y = np.concatenate([t.detach().numpy() for t in means[k]])
            if k in PROFILE_NAMES:
                data[k] = (dims_p, y)
            else:
                data[k] = (dims, y)

        for k in gradients:
            y = np.concatenate([t.numpy() for t in gradients[k]])
            if k in PROFILE_NAMES:
                data[k + "_grad"] = (dims_p + ("inputs",), y)
            else:
                data[k + "_grad"] = (dims + ("inputs",), y)

        if len(precip_1st_tercile) > 0:
            dims = input_data.dimensions["surface_precip"]
            data["precip_1st_tercile"] = (
                dims,
                np.concatenate([t.numpy() for t in precip_1st_tercile]),
            )
            data["precip_2nd_tercile"] = (
                dims,
                np.concatenate([t.numpy() for t in precip_2nd_tercile]),
            )
            pop = np.concatenate([t.numpy() for t in pop])
            data["pop"] = (dims, pop)
            data["most_likely_precip"] = data["surface_precip"]
            data["precip_flag"] = (dims, pop > 0.5)
            samples = np.concatenate([t.numpy() for t in samples])
            data["surface_precip_samples"] = (dims, samples)

        data = xr.Dataset(data)
        return input_data.finalize(data)


###############################################################################
# Netcdf Format
###############################################################################


class NetcdfLoader1D(GPROF_NN_1D_Dataset):
    """
    Data loader for running the GPROF-NN 1D retrieval on input data
    in NetCDF data format.
    """

    def __init__(self, filename, normalizer, batch_size=16 * 1024, sensor=None):
        """
        Create loader for input data in NetCDF format that provides input
        data for the GPROF-NN 1D retrieval.

        Args:
            filename: The name of the NetCDF file containing the input data.
            normalizer: The normalizer object to use to normalize the input
                data.
            batch_size: How many observations to combine into a single
                input batch.
        """
        targets = ALL_TARGETS + ["latitude", "longitude"]
        GPROF_NN_1D_Dataset.__init__(
            self,
            filename,
            targets=targets,
            normalizer=normalizer,
            batch_size=batch_size,
            sensor=sensor,
            shuffle=False,
            augment=False,
        )
        self.n_samples = len(self)
        self.scalar_dimensions = ("samples",)
        self.profile_dimensions = ("samples", "layers")
        self.dimensions = {
            t: ("samples", "layers") if t in PROFILE_NAMES else ("samples")
            for t in ALL_TARGETS
        }
        self.data = self.to_xarray_dataset()

    def __getitem__(self, i):
        """
        Return batch of input data.

        Args:
            The batch index.

        Return:
            PyTorch tensor containing the batch of input data.
        """
        i_start = i * self.batch_size
        i_end = i_start + self.batch_size
        x = torch.tensor(self.x[i_start:i_end])

        return x

    def finalize(self, data):
        """
        Reshape retrieval results into shape of input data.
        """
        invalid = np.all(self.x[:, : self.sensor.n_chans] <= -1.5, axis=-1)
        for v in ALL_TARGETS:
            data[v].data[invalid] = np.nan

        variables = [target for target in ALL_TARGETS if target in data.variables]
        for var in variables:
            data[var + "_true"] = self.data[var]
        data["latitude"] = self.data["latitude"]
        data["longitude"] = self.data["longitude"]
        if "earth_incidence_angle" in self.data.variables:
            data["earth_incidence_angle"] = self.data["earth_incidence_angle"]

        return data


# Kept for backwards compatibility.
# TODO: Remove before release.
NetcdfLoader0D = NetcdfLoader1D


class NetcdfLoader3D(GPROF_NN_3D_Dataset):
    """
    Data loader for running the GPROF-NN 3D retrieval on input data
    in NetCDF data format.
    """

    def __init__(self, filename, normalizer, batch_size=32, sensor=None):
        """
        Create loader for input data in NetCDF format that provides input
        data for the GPROF-NN 3D retrieval.

        Args:
            filename: The name of the NetCDF file containing the input data.
            normalizer: The normalizer object to use to normalize the input
                data.
            batch_size: How many observations to combine into a single
                input batch.
            sensor: Sensor object to use to load the data. This can be used to
                apply a specific correction to the input data.
        """
        targets = ALL_TARGETS + ["latitude", "longitude"]
        super().__init__(
            filename,
            targets=targets,
            normalizer=normalizer,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            input_dimensions=None,
            sensor=sensor,
        )
        self.n_samples = len(self)
        self.scalar_dimensions = ("samples",)
        self.profile_dimensions = ("samples", "layers")
        dimensions = {}
        for t in ALL_TARGETS:
            if t in PROFILE_NAMES:
                dimensions[t] = ("samples", "layers", "scans", "pixels")
            else:
                dimensions[t] = ("samples", "scans", "pixels")
        self.dimensions = dimensions
        self.data = self.to_xarray_dataset()
        self.padding = calculate_padding_dimensions(self.x[0])

    def __getitem__(self, i):
        """
        Return batch of input data.

        Args:
            The batch index.

        Return:
            PyTorch tensor containing the batch of input data.
        """
        i_start = i * self.batch_size
        i_end = i_start + self.batch_size
        x = torch.tensor(self.x[i_start:i_end])
        x = torch.nn.functional.pad(x, self.padding, mode="replicate")
        return x

    def finalize(self, data):
        """
        Reshape retrieval results into shape of input data.
        """
        dims = {}
        n_pixels = data.pixels.size
        n_scans = data.scans.size
        data = data[
            {
                "pixels": slice(self.padding[0], n_pixels - self.padding[1]),
                "scans": slice(self.padding[2], n_scans - self.padding[3]),
            }
        ]
        if "layers" in data.dims:
            dims = ["samples", "scans", "pixels", "layers"]
            data = data.transpose(*dims)

        vars = [target for target in ALL_TARGETS if target in data.variables]
        for var in vars:
            data[var + "_true"] = self.data[var]
        data["latitude"] = self.data["latitude"]
        data["longitude"] = self.data["longitude"]
        if "earth_incidence_angle" in self.data.variables:
            data["earth_incidence_angle"] = self.data["earth_incidence_angle"]

        invalid = np.all(self.x[:, : self.sensor.n_chans] <= -1.5, axis=1)
        for var in vars:
            data[var].data[invalid] = np.nan

        return data


# Kept for backwards compatibility.
# TODO: Remove before release.
NetcdfLoader2D = NetcdfLoader3D


class NetcdfLoader1DFull(NetcdfLoader3D):
    """
    Special data loader for the 1D retrieval that retains the spatial
    structure of the data.
    """

    def __init__(self, filename, normalizer, batch_size=32, sensor=None):
        """
        Create loader for input data in NetCDF format that provides input
        data for the GPROF-NN 3D retrieval.

        Args:
            filename: The name of the NetCDF file containing the input data.
            normalizer: The normalizer object to use to normalize the input
                data.
            batch_size: How many observations to combine into a single
                input batch.
        """
        super().__init__(filename, normalizer, batch_size=batch_size, sensor=sensor)
        self.scalar_dimensions = ("samples",)
        self.profile_dimensions = ("samples", "layers")
        self.dimensions = {
            t: ("samples", "layers") if t in PROFILE_NAMES else ("samples")
            for t in ALL_TARGETS
        }

    def __getitem__(self, i):
        """
        Return batch of input data.

        Args:
            The batch index.

        Return:
            PyTorch tensor containing the batch of input data.
        """
        i_start = i * self.batch_size
        i_end = i_start + self.batch_size
        x = torch.tensor(self.x[i_start:i_end])
        x = x.permute(0, 2, 3, 1).flatten(0, 2)
        return x

    def finalize(self, data):
        """
        Reshape retrieval results into shape of input data.
        """
        pixels = self.data.pixels.data
        scans = self.data.scans.data
        n_samples = data.samples.size / len(pixels) / len(scans)
        samples = np.arange(n_samples)
        index = pd.MultiIndex.from_product(
            (samples, scans, pixels), names=("new_samples", "scans", "pixels")
        )

        # Reproduce scene dimensions
        data = data.assign(samples=index).unstack("samples")
        data = data.rename({"new_samples": "samples"})

        if "layers" in data.dims:
            dims = ["samples", "scans", "pixels", "layers"]
            data = data.transpose(*dims)

        vars = [target for target in ALL_TARGETS if target in data.variables]
        for var in vars:
            data[var + "_true"] = self.data[var]
        data["latitude"] = self.data["latitude"]
        data["longitude"] = self.data["longitude"]

        invalid = np.all(self.x[:, : self.sensor.n_chans] <= -1.5, axis=1)
        for var in vars:
            data[var].data[invalid] = np.nan

        return data


###############################################################################
# Preprocessor format
###############################################################################


class ObservationLoader1D:
    """
    Base class for interfaces that load retrieval input data from a L1C
    file or a preprocessor file.
    """

    def __init__(self, filename, file_class, normalizer, batch_size=1024 * 8):
        """
        Create observation loader.

        Args:
            filename: Path to the preprocessor or L1C file from which to
                load the input data for the retrieval.
            file_class: The class to use to load the observations.
            normalizer: The normalizer object to use to normalize the input
                data.
            scans_per_batch: How scans should be combined into a single
                batch.
        """
        filename = Path(filename)
        self.filename = filename
        input_file = file_class(filename)
        self.sensor = input_file.sensor
        self.data = input_file.to_xarray_dataset()
        self.normalizer = normalizer
        self.n_scans = self.data.scans.size
        self.n_pixels = self.data.pixels.size
        self.scans_per_batch = batch_size // self.n_pixels

        self.scalar_dimensions = ("samples",)
        self.profile_dimensions = ("samples", "layers")
        self.dimensions = {
            t: ("samples", "layers") if t in PROFILE_NAMES else ("samples",)
            for t in ALL_TARGETS
        }

        x = combine_input_data_1d(self.data, self.sensor)
        self.x = torch.tensor(self.normalizer(x))

    def __len__(self):
        """
        The number of batches in the preprocessor file.
        """
        return math.ceil(self.n_scans / self.scans_per_batch)

    def __getitem__(self, i):
        """
        Return batch of retrieval inputs as PyTorch tensor.

        Args:
            i: The index of the batch.

        Return:
            PyTorch Tensor ``x`` containing the normalized inputs.
        """
        i_start = i * self.scans_per_batch * self.n_pixels
        i_end = i_start + self.scans_per_batch * self.n_pixels
        return self.x[i_start:i_end]

    def finalize(self, data):
        """
        Transform retrieval results back into format of the input data.
        Recreates the scan and pixel dimensions that are lost due to
        the batch-wise processing of the retrieval.

        Args:
            data: 'xarray.Dataset' containing the retrieval results
                produced by evaluating the GPROF-NN model on the input
                data from the loader.

        Return:
            The data in 'data' modified to match the format of the input
            data.
        """
        scans = np.arange(data.samples.size // self.n_pixels)
        pixels = np.arange(self.n_pixels)
        names = ("scans", "pixels")
        index = pd.MultiIndex.from_product((scans, pixels), names=names)
        data = data.assign(samples=index).unstack("samples")
        if "layers" in data.dims:
            dims = ["scans", "pixels", "layers"]
            data = data.transpose(*dims)

        tbs = self.data["brightness_temperatures"].data
        invalid = np.all(tbs < 0, axis=-1)
        for var in ALL_TARGETS:
            if var in data.variables:
                data[var].data[invalid] = np.nan

                # Replace small values with zero.
                if var in _THRESHOLDS:
                    thresh = _THRESHOLDS[var]
                    if thresh > 0:
                        small = data[var].data < thresh
                        data[var].data[small] = 0.0

        return data

    def write_retrieval_results(
        self, output_path, results, ancillary_data=None, suffix=None
    ):
        """
        Write retrieval results to file.

        Args:
            output_path: The folder to which to write the output.
            results: ``xarray.Dataset`` containing the retrieval results.
            ancillary_data: The folder containing the profile clusters.
            suffix: Suffix to append to algorithm name in filename.

        Return:
            The filename of the retrieval output file.
        """
        preprocessor_file = PreprocessorFile(self.filename)
        return preprocessor_file.write_retrieval_results(
            output_path, results, ancillary_data=ancillary_data, suffix=suffix
        )


class PreprocessorLoader1D(ObservationLoader1D):
    """
    Interface class to load retrieval input data for retrieval models
    that require ancillary data from the preprocessor.

    If the provided file is an L1C file (i.e. comes in HDF5 format), this
    loader will try to run the preprocessor on it in order to convert it
    to a preprocessor file.
    """

    def __init__(self, filename, normalizer, batch_size=1024 * 16):
        """
        Create preprocessor loader.

        Args:
            filename: Path to the preprocessor file from which to load the
                input data.
            normalizer: The normalizer object to use to normalize the input
                data.
            batch_size: How many pixels should be processed simultaneously
                in a single batch.
        """
        suffix = filename.suffix
        if suffix.endswith("HDF5"):
            with TemporaryDirectory() as tmp:
                output_file = Path(tmp) / "input.pp"
                run_preprocessor_l1c(filename, output_file)
                super().__init__(
                    output_file, PreprocessorFile, normalizer, batch_size=batch_size
                )
        else:
            LOGGER.info("Loading preprocessor input data from %s.", filename)
            super().__init__(
                filename, PreprocessorFile, normalizer, batch_size=batch_size
            )


# Kept for backwards compatibility.
# TODO: Remove before release.
PreprocessorLoader0D = PreprocessorLoader1D


class L1CLoader1D(ObservationLoader1D):
    """
    Interface class to load GPROF-NN 1D retrieval input from L1C files.
    """

    def __init__(self, filename, normalizer, batch_size=1024 * 16):
        """
        Create L1C loader.

        Args:
            filename: Path to the L1C file from which to load the
                input data.
            normalizer: The normalizer object to use to normalize the input
                data.
            batch_size: How many pixels should be processed simultaneously
                in a single batch.
        """
        super().__init__(filename, L1CFile, normalizer, batch_size=batch_size)


L1CLoader0D = L1CLoader1D


class ObservationLoader3D:
    """
    Base class for the data loaders that load retrieval input data for
    the GPROF-NN 3D retrieval from preprocessor or L1C files.
    """

    def __init__(self, filename, file_class, normalizer):
        """
        Create observation loader.

        Args:
            filename: Path to the preprocessor file from which to load the
                input data.
            file_class: The file class to use to load the input data.
            normalizer: The normalizer object to use to normalize the input
                data.
        """
        self.filename = filename
        self.normalizer = normalizer

        input_file = file_class(filename)
        self.data = input_file.to_xarray_dataset()
        tcwv = self.data["total_column_water_vapor"]
        self.n_scans = self.data.scans.size
        self.n_pixels = self.data.pixels.size

        input_data = combine_input_data_3d(self.data, input_file.sensor)
        self.input_data = self.normalizer(input_data)
        self.padding = calculate_padding_dimensions(input_data)

        self.scalar_dimensions = ("samples", "scans", "pixels")
        self.profile_dimensions = ("samples", "layers", "scans", "pixels")
        self.dimensions = {
            t: ("samples", "layers", "scans", "pixels")
            if t in PROFILE_NAMES
            else ("samples", "scans", "pixels")
            for t in ALL_TARGETS
        }

    def __len__(self):
        """
        The number of batches in the input file.
        """
        return 1

    def __getitem__(self, i):
        """
        Return batch of retrieval inputs as PyTorch tensor.

        Args:
            i: The index of the batch.

        Return:
            PyTorch Tensor ``x`` containing the normalized inputs.
        """
        x = torch.tensor(self.input_data)
        return nn.functional.pad(x, self.padding, mode="replicate")

    def finalize(self, data):
        """
        Reshape retrieval results into shape of input data.
        """
        n_scans = data.scans.size
        n_pixels = data.pixels.size
        data = data[
            {
                "samples": 0,
                "scans": slice(self.padding[2], n_scans - self.padding[3]),
                "pixels": slice(self.padding[0], n_pixels - self.padding[1]),
            }
        ]

        if "layers" in data.dims:
            dims = ["scans", "pixels", "layers"]
            data = data.transpose(*dims)
        return data

    def write_retrieval_results(
        self, output_path, results, ancillary_data=None, suffix=None
    ):
        """
        Write retrieval results to file.

        Args:
            output_path: The folder to which to write the output.
            results: ``xarray.Dataset`` containing the retrieval results.
            ancillary_data: The folder containing the profile clusters.
            suffix: Suffix to append to algorithm name in filename.

        Return:
            The filename of the retrieval output file.
        """
        preprocessor_file = PreprocessorFile(self.filename)
        return preprocessor_file.write_retrieval_results(
            output_path, results, ancillary_data=ancillary_data, suffix=suffix
        )


class PreprocessorLoader3D(ObservationLoader3D):
    """
    Interface class to load retrieval input for the GPROF-NN 3D retrieval
    form preprocessor files.
    """

    def __init__(self, filename, normalizer):
        """
        Create preprocessor loader.

        Args:
            filename: Path to the preprocessor file from which to load the
                input data.
            normalizer: The normalizer object to use to normalize the input
                data.
        """
        suffix = filename.suffix
        if suffix.endswith("HDF5"):
            with TemporaryDirectory() as tmp:
                output_file = Path(tmp) / "input.pp"
                run_preprocessor_l1c(filename, output_file)
                super().__init__(output_file, PreprocessorFile, normalizer)
        else:
            LOGGER.info("Loading preprocessor input data from %s.", filename)
            super().__init__(filename, PreprocessorFile, normalizer)


# Kept for backwards compatibility.
# TODO: Remove before release.
PreprocessorLoader2D = PreprocessorLoader3D


class L1CLoader3D(ObservationLoader3D):
    """
    Interface class to load retrieval input for the GPROF-NN 3D retrieval
    form L1C files.
    """

    def __init__(self, filename, normalizer):
        """
        Create preprocessor loader.

        Args:
            filename: Path to the preprocessor file from which to load the
                input data.
            normalizer: The normalizer object to use to normalize the input
                data.
        """
        super().__init__(filename, L1CFile, normalizer)


# Kept for backwards compatibility.
# TODO: Remove before release.
L1CLoader2D = L1CLoader3D


###############################################################################
# Simulator format
###############################################################################


class SimulatorLoader:
    """
    The 'SimulatorLoader' class loads and prepares the input for a simulator
    network from a training data netCDF file. The predicted simulated
    brightness temperatures and biases are then combined with the input data
    to produce the results.
    """

    def __init__(self, filename, normalizer, batch_size=8, **kwargs):
        """
        Args:
            filename: Path to the netCDF file from which to load the input.
            normalizer: Normalizer object to use to normalize the retrieval
                inputs.
            batch_size: The batch size to use for the processing.
        """
        super().__init__()
        self.filename = filename
        self.dataset = decompress_and_load(self.filename)
        self.normalizer = normalizer
        self.batch_size = batch_size

        sensor_name = self.dataset.attrs["sensor"]
        sensor = getattr(sensors, sensor_name, None)
        if sensor is None:
            raise ValueError(f"Sensor {sensor_name} isn't yet supported.")
        self.sensor = sensor

        self._load_data()
        self.n_samples = self.input_data.shape[0]

        if sensor.n_angles > 1:
            dims_tbs = ("samples", "angles", "channels", "scans", "pixels")
            dims_bias = ("samples", "channels", "scans", "pixels")
            self.dimensions = {
                f"simulated_brightness_temperatures_{j}": dims_tbs
                for j in range(sensor.n_chans)
            }
            for j in range(sensor.n_chans):
                self.dimensions[f"brightness_temperature_biases_{j}"] = dims_bias
        else:
            dims = ("samples", "channels", "scans", "pixels")
            self.dimensions = {
                f"simulated_brightness_temperatures_{j}": dims
                for j in range(sensor.n_chans)
            }
            for j in range(sensor.n_chans):
                self.dimensions[f"brightness_temperature_biases_{j}"] = dims

    def _load_data(self):
        """
        Load data from training data NetCDF format into 'input' data
        attribute.
        """
        dataset = self.dataset[{"samples": self.dataset.source == 0}]
        if self.sensor == sensors.GMI:
            input_data = combine_input_data_3d(
                dataset, self.sensor, v_tbs="brightness_temperatures"
            )
        else:
            input_data = combine_input_data_3d(
                dataset, sensors.GMI, v_tbs="brightness_temperatures_gmi"
            )
        self.input_data = self.normalizer(input_data)
        self.padding = calculate_padding_dimensions(input_data)

    def __len__(self):
        n = self.input_data.shape[0]
        if n % self.batch_size > 0:
            return n // self.batch_size + 1
        return n // self.batch_size

    def __getitem__(self, i):
        """
        Return batch of input data.

        Args:
            The batch index.

        Return:
            PyTorch tensor containing the batch of input data.
        """
        i_start = i * self.batch_size
        i_end = i_start + self.batch_size
        x = torch.tensor(self.input_data[i_start:i_end])
        return torch.nn.functional.pad(x, self.padding, mode="replicate")

    def finalize(self, data):
        """
        Copy predicted Tbs and biases into input file and return.
        """
        data = data[
            {
                "scans": slice(self.padding[0], -self.padding[1]),
                "pixels": slice(self.padding[2], -self.padding[3]),
            }
        ]

        n_samples = self.dataset.samples.size
        n_scans = data.scans.size
        n_pixels = data.pixels.size
        n_channels = self.sensor.n_chans

        data = data.transpose("samples", "scans", "pixels", "angles", "channels")

        if self.sensor.n_angles > 1:
            dims = ("samples", "scans", "pixels", "angles", "channels")
            n_angles = data.angles.size
            self.dataset["simulated_brightness_temperatures"] = (
                dims,
                np.nan
                * np.zeros(
                    (n_samples, n_scans, n_pixels, n_angles, n_channels),
                    dtype=np.float32,
                ),
            )
            self.dataset["brightness_temperature_biases"] = (
                ("samples", "scans", "pixels", "channels"),
                np.nan
                * np.zeros(
                    (n_samples, n_scans, n_pixels, n_channels), dtype=np.float32
                ),
            )
        else:
            dims = ("samples", "scans", "pixels", "channels")
            self.dataset["simulated_brightness_temperatures"] = (
                dims,
                np.nan
                * np.zeros(
                    (n_samples, n_scans, n_pixels, n_channels), dtype=np.float32
                ),
            )
            self.dataset["brightness_temperature_biases"] = (
                ("samples", "scans", "pixels", "channels"),
                np.nan
                * np.zeros(
                    (n_samples, n_scans, n_pixels, n_channels), dtype=np.float32
                ),
            )
        index = 0
        for i in range(n_samples):
            if self.dataset.source[i] == 0:
                v = self.dataset["simulated_brightness_temperatures"].data
                for j in range(n_channels):
                    v_in = data[f"simulated_brightness_temperatures_{j}"].data
                    v[i, ..., j] = v_in[index, ..., 0]

                v = self.dataset["brightness_temperature_biases"]
                for j in range(n_channels):
                    v_in = data[f"brightness_temperature_biases_{j}"].data
                    v[i, :, :, j] = v_in[index, ..., 0]

                index += 1
        return self.dataset
