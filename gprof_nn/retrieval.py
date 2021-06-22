"""
==================
gprof_nn.retrieval
==================

This module contains classes and functionality that drive the execution
of the retrieval.
"""
import math
from pathlib import Path

import numpy as np
import xarray as xr

import torch
from torch import nn
import pandas as pd

from gprof_nn.definitions import PROFILE_NAMES
from gprof_nn.data.training_data import (GPROF0DDataset,
                                         GPROF0DDataset)
from gprof_nn.data.preprocessor import PreprocessorFile


###############################################################################
# Helper functions.
###############################################################################


def calculate_padding_dimensions(t):
    """
    Calculate list of padding values to extend second-to-last and last
    dimensions of the tensor to multiples of 32.

    Args:
        t: The ``torch.Tensor`` to pad.

    Return
        A tuple ``(p_l_m, p_r_m, p_l_n, p_r_n)`` containing the
        left and right padding  for the second to
        last dimension (``p_l_m, p_r_m``) and for the last dimension
        (``p_l_n, p_r_n``).
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


def combine_input_data_2d(dataset):
    """
    Combine retrieval input data into input tensor format for convolutional
    retrieval.

    Args:
         ``xarray.Dataset`` with the input variables.

    Return:
        Rank-4 input tensor containing the input data with feature oriented
        along the first axis.
    """
    bts = dataset["brightness_temperatures"][:].data
    invalid = (bts > 500.0) + (bts < 0.0)
    bts[invalid] = np.nan
    # 2m temperature
    t2m = dataset["two_meter_temperature"][:].data[..., np.newaxis]
    # Total precipitable water.
    tcwv = dataset["total_column_water_vapor"][:].data[..., np.newaxis]
    # Surface type
    st = dataset["surface_type"][:].data
    n_types = 18
    shape = bts.shape[:-1]
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

    input_data = np.concatenate([bts, t2m, tcwv, st_1h, am_1h], axis=-1)
    input_data = input_data.astype(np.float32)
    if input_data.ndim < 4:
        input_data = np.expand_dims(input_data, 0)
    input_data = np.transpose(input_data, (0, 3, 1, 2))
    return input_data


###############################################################################
# Retrieval driver
###############################################################################


PREPROCESSOR = "preprocessor"
TRAINING_DATA = "training_data"

class RetrievalDriver:
    """
    Helper class that implements the logic to run the GPROF-NN retrieval.
    """
    N_CHANNELS = 15

    def __init__(self,
                 input_file,
                 normalizer,
                 model,
                 ancillary_data=None,
                 output_file=None):
        """
        Args:
            input_file: Path to the file containing the input data for the
                 retrieval.
            normalizer_file: Path to the file containing the normalizer.
            model: The neural network to use for the retrieval.



        """
        from gprof_nn.data.preprocessor import PreprocessorFile
        input_file = Path(input_file)
        self.input_file = input_file
        self.normalizer = normalizer
        self.model = model
        self.ancillary_data = ancillary_data

        suffix = input_file.suffix
        if suffix.endswith("pp"):
            self.format = PREPROCESSOR
        else:
            self.format = TRAINING_DATA

        if self.format == PREPROCESSOR:
            self.input_data = model.preprocessor_class(input_file,
                                                       self.normalizer)
        else:
            self.input_data = model.netcdf_class(
                input_file,
                normalizer=normalizer
            )

        output_suffix = ".BIN"
        if self.format == TRAINING_DATA:
            output_suffix = ".nc"

        if output_file is None:
            self.output_file = Path(
                self.input_file.name.replace(suffix, output_suffix)
            )
        elif Path(output_file).is_dir():
            self.output_file = (
                Path(output_file) /
                self.input_file.name.replace(suffix, output_suffix)
                )
        else:
            self.output_file = output_file

    def _run(self, xrnn):
        """
        Batch-wise processing of retrieval input.

        Args:
            xrnn: A quantnn neural-network model.

        Return:
            A dataset containing the concatenated retrieval results for all
            batches.
        """
        means = {}
        precip_1st_tercile = []
        precip_3rd_tercile = []
        pop = []

        with torch.no_grad():
            device = next(iter(xrnn.model.parameters())).device
            for i in range(len(self.input_data)):
                x = self.input_data[i]
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
                        precip_3rd_tercile.append(t[:, 1].cpu())
                        p = xrnn.probability_larger_than(y_pred=y,
                                                         y=1e-4,
                                                         key=k)
                        pop.append(p.cpu())

        dims = self.input_data.scalar_dimensions
        dims_p = self.input_data.profile_dimensions

        data = {}
        for k in means:
            y = np.concatenate([t.numpy() for t in means[k]])
            if k in PROFILE_NAMES:
                data[k] = (dims_p, y)
            else:
                data[k] = (dims, y)


        data["precip_1st_tercile"] = (
            dims,
            np.concatenate([t.numpy() for t in precip_1st_tercile])
        )
        data["precip_3rd_tercile"] = (
            dims,
            np.concatenate([t.numpy() for t in precip_3rd_tercile])
        )
        pop = np.concatenate([t.numpy() for t in pop])
        data["pop"] = (dims, pop)
        data["most_likely_precip"] = data["surface_precip"]
        data["precip_flag"] = (dims, pop > 0.5)
        data = xr.Dataset(data)

        return self.input_data.finalize(data)

    def run_retrieval(self):
        """
        Run retrieval and store results to file.

        Return:
            Name of the output file that the results have been written
            to.
        """
        results = self._run(self.model)
        if self.format == PREPROCESSOR:
            return self.input_data.write_retrieval_results(
                self.output_file.parent,
                results,
                ancillary_data=self.ancillary_data
            )
        else:
            results.to_netcdf(self.output_file)
            return self.output_file


###############################################################################
# Netcdf Format
###############################################################################


class NetcdfLoader:

    def __init__(self):
        self.n_samples = 0
        self.batch_size = 1

    def __len__(self):
        """
        The number of batches in the dataset.
        """
        return math.ceil(self.n_samples / self.batch_size)

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
        return x


class NetcdfLoader0D(NetcdfLoader):
    """
    Data loader for running the GPROF-NN 0D retrieval on input data
    in NetCDF data format.
    """
    def __init__(self,
                 filename,
                 normalizer,
                 batch_size=16 * 1024):
        super().__init__()
        self.filename = filename
        self.normalizer = normalizer
        self.batch_size = batch_size

        self._load_data()
        self.n_samples = self.input_data.shape[0]

        self.dimensions = ("samples", "pixels")
        self.profile_dimensions = ("samples", "pixels")

        self.scalar_dimensions = ("samples")
        self.profile_dimensions = ("samples", "layers")

    def _load_data(self):
        """
        Load data from training data NetCDF format into the
        objects 'input_data' attribute.
        """
        dataset = xr.open_dataset(self.filename)

        #
        # Load data into input vector
        #

        bts = dataset["brightness_temperatures"][:].data
        invalid = (bts > 500.0) + (bts < 0.0)
        bts[invalid] = np.nan
        # 2m temperature
        t2m = dataset["two_meter_temperature"][:].data[..., np.newaxis]
        # Total precipitable water.
        tcwv = dataset["total_column_water_vapor"][:].data[..., np.newaxis]
        # Surface type
        st = dataset["surface_type"][:].data
        n_types = 18
        shape = bts.shape[:3]
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

        input_data = np.concatenate([bts, t2m, tcwv, st_1h, am_1h], axis=-1)
        input_data = input_data.astype(np.float32)
        input_data = self.normalizer(input_data.reshape(-1, 39))
        self.input_data = input_data

        if input_data.ndim > 2:
            self.kind = "standard"
        else:
            self.kind = "bin"

    def finalize(self, data):
        """
        Reshape retrieval results into shape of input data.
        """
        if self.kind == "standard":
            samples = np.arange(data.samples // (221 * 221))
            scans = np.arange(221)
            pixels = np.arange(221)
            names = ("samples_t", "scans", "pixels")
            index = pd.MultiIndex.from_product((samples, scans, pixels),
                                               names=names)
            data = data.rename_dims({"samples_t": "samples"})
            data = data.assign(samples=index).unstack('samples')
        return data


class NetcdfLoader2D(NetcdfLoader):
    """
    Data loader for running the GPROF-NN 2D retrieval on input data
    in NetCDF data format.
    """
    def __init__(self,
                 filename,
                 normalizer,
                 batch_size=8):
        super().__init__()
        self.filename = filename
        self.normalizer = normalizer
        self.batch_size = batch_size

        self._load_data()
        self.n_samples = self.input_data.shape[0]

        self.scalar_dimensions = ("samples", "scans", "pixels")
        self.profile_dimensions = ("samples", "layers", "scans", "pixels")

    def _load_data(self):
        """
        Load data from training data NetCDF format into 'input' data
        attribute.
        """
        dataset = xr.open_dataset(self.filename)
        input_data = combine_input_data_2d(dataset)
        self.input_data = input_data
        self.padding = calculate_padding_dimensions(input_data)

    def __getitem__(self, i):
        """
        Return batch of input data.

        Args:
            The batch index.

        Return:
            PyTorch tensor containing the batch of input data.
        """
        x = super().__getitem__(i)
        return torch.nn.functional.pad(x, self.padding, mode="replicate")


    def finalize(self, data):
        """
        Reshape retrieval results into shape of input data.
        """
        data = data[{
            "scans": slice(self.padding[0], -self.padding[1]),
            "pixels": slice(self.padding[2], -self.padding[3])
        }]
        return data.transpose("samples", "scans", "pixels", "layers")


###############################################################################
# Netcdf Format
###############################################################################


class PreprocessorLoader0D:
    """
    Interface class to run the GPROF-NN retrieval on preprocessor files.
    """
    def __init__(self,
                 filename,
                 normalizer,
                 batch_size=1024 * 8):
        """
        Create preprocessor loader.

        Args:
            filename: Path to the preprocessor file from which to load the
                input data.
            normalizer: The normalizer object to use to normalize the input
                data.
            scans_per_batch: How scans should be combined into a single
                batch.
        """
        self.filename = filename
        preprocessor_file = PreprocessorFile(filename)
        self.data = preprocessor_file.to_xarray_dataset()
        self.normalizer = normalizer
        self.n_scans = self.data.scans.size
        self.n_pixels = self.data.pixels.size
        self.scans_per_batch = batch_size // 221

        self.scalar_dimensions = ("samples")
        self.profile_dimensions = ("samples", "layers")

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
        i_start = i * self.scans_per_batch
        i_end = min(i_start + self.scans_per_batch,
                    self.n_scans)

        n = (i_end - i_start) * self.data.pixels.size
        x = np.zeros((n, 39), dtype=np.float32)

        tbs = self.data["brightness_temperatures"].data[i_start:i_end]
        tbs = tbs.reshape(-1, 15)
        t2m = self.data["two_meter_temperature"].data[i_start:i_end]
        t2m = t2m.reshape(-1)
        tcwv = self.data["total_column_water_vapor"].data[i_start:i_end]
        tcwv = tcwv.reshape(-1)
        st = self.data["surface_type"].data[i_start:i_end]
        st = st.reshape(-1)
        at = np.maximum(self.data["airmass_type"].data[i_start:i_end], 0.0)
        at = at.reshape(-1)

        x[:, :15] = tbs
        x[:, :15][x[:, :15] < 0] = np.nan

        x[:, 15] = t2m
        x[:, 15][x[:, 15] < 0] = np.nan

        x[:, 16] = tcwv
        x[:, 16][x[:, 16] < 0] = np.nan

        for i in range(18):
            x[:, 17 + i][st == i + 1] = 1.0

        for i in range(4):
            x[:, 35 + i][at == i] = 1.0

        x = self.normalizer(x)
        return torch.tensor(x)

    def finalize(self, data):
        """
        Reshape retrieval results into shape of input data.
        """
        scans = np.arange(data.samples.size // 221)
        pixels = np.arange(221)
        names = ("scans", "pixels")
        index = pd.MultiIndex.from_product((scans, pixels),
                                           names=names)
        data = data.assign(samples=index).unstack('samples')
        return data.transpose("scans", "pixels", "layers")


    def write_retrieval_results(self,
                                output_path,
                                results,
                                ancillary_data=None):
        """
        Write retrieval results to file.

        Args:
            output_path: The folder to which to write the output.
            results: ``xarray.Dataset`` containing the retrieval results.
            ancillary_data: The folder containing the profile clusters.

        Return:
            The filename of the retrieval output file.
        """
        preprocessor_file = PreprocessorFile(self.filename)
        return preprocessor_file.write_retrieval_results(
            output_path,
            results,
            ancillary_data=ancillary_data
        )


class PreprocessorLoader2D:
    """
    Interface class to run the GPROF-NN 2D retrieval on preprocessor files.
    """
    def __init__(self,
                 filename,
                 normalizer):
        """
        Create preprocessor loader.

        Args:
            filename: Path to the preprocessor file from which to load the
                input data.
            normalizer: The normalizer object to use to normalize the input
                data.
            scans_per_batch: How scans should be combined into a single
                batch.
        """
        self.filename = filename
        preprocessor_file = PreprocessorFile(filename)
        self.data = preprocessor_file.to_xarray_dataset()
        self.normalizer = normalizer
        self.n_scans = self.data.scans.size
        self.n_pixels = self.data.pixels.size

        input_data = combine_input_data_2d(self.data)
        self.input_data = self.normalizer(input_data)
        self.padding = calculate_padding_dimensions(input_data)

        self.scalar_dimensions = ("samples", "scans", "pixels")
        self.profile_dimensions = ("samples", "layers", "scans", "pixels")

    def __len__(self):
        """
        The number of batches in the preprocessor file.
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
        data = data[{
            "samples": 0,
            "scans": slice(self.padding[2], -self.padding[3]),
            "pixels": slice(self.padding[0], -self.padding[1])
        }]
        return data.transpose("scans", "pixels", "layers")

    def write_retrieval_results(self,
                                output_path,
                                results,
                                ancillary_data=None):
        """
        Write retrieval results to file.

        Args:
            output_path: The folder to which to write the output.
            results: ``xarray.Dataset`` containing the retrieval results.
            ancillary_data: The folder containing the profile clusters.

        Return:
            The filename of the retrieval output file.
        """
        preprocessor_file = PreprocessorFile(self.filename)
        return preprocessor_file.write_retrieval_results(
            output_path,
            results,
            ancillary_data=ancillary_data
        )
