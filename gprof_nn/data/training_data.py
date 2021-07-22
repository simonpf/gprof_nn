"""
===========================
gprof_nn.data.training_data
===========================

This module defines the dataset classes that provide access to
the training data for the GPROF-NN retrievals.
"""
import math
import logging
import os
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from netCDF4 import Dataset

from quantnn.normalizer import MinMaxNormalizer

from gprof_nn import sensors
from gprof_nn.data.preprocessor import PreprocessorFile
from gprof_nn.augmentation import M, N, extract_domain, get_transformation_coordinates

LOGGER = logging.getLogger(__name__)
_DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")

_THRESHOLDS = {
    "surface_precip": 1e-4,
    "convective_precip": 1e-4,
    "rain_water_path": 1e-4,
    "ice_water_path": 1e-4,
    "cloud_water_path": 1e-4,
    "total_column_water_vapor": 1e0,
    "rain_water_content": 1e-5,
    "cloud_water_content": 1e-5,
    "snow_water_content": 1e-5,
    "latent_heat": -99999,
}


def write_preprocessor_file(input_data, output_file, template=None):
    """
    Extract sample from training data file and write to preprocessor format.

    Args:
        input_data: Path to a NetCDF4 file containing the training or test
            data or xarray.Dataset containing the data to write to a
            preprocessor file.
        output_file: Path of the file to write the output to.
        template: Template preprocessor file use to determine the orbit header
             information. If not provided this data will be filled with dummy
             values.
    """
    if not isinstance(input_data, xr.Dataset):
        data = xr.open_dataset(input_data)
    else:
        data = input_data
    n_pixels = data.pixels.size
    n_scans = data.scans.size
    if hasattr(data, "samples"):
        n_scenes = data.samples.size
        dim_offset = 1
    else:
        n_scenes = 1
        dim_offset = 0

    c = math.ceil(n_scenes / (n_pixels * 256))
    if c > 256:
        raise ValueError(
            "The dataset contains too many observations to be savely "
            " converted to a preprocessor file."
        )
    n_scans_r = n_scans * n_scenes
    n_pixels_r = n_pixels

    new_dataset = {
        "scans": np.arange(n_scans_r),
        "pixels": np.arange(n_pixels_r),
        "channels": np.arange(15),
    }
    dims = ("scans", "pixels", "channels")
    for k in data:
        if k == "scan_time":
            da = data[k]
            new_dataset[k] = (("scans",), da.data.ravel()[:n_scans_r])
        else:
            da = data[k]
            new_shape = (n_scans_r, n_pixels_r) + da.shape[(2 + dim_offset) :]
            new_shape = new_shape[: len(da.data.shape) - dim_offset]
            dims = da.dims[dim_offset:]
            if "pixels_center" in dims:
                continue
            new_dataset[k] = (dims, da.data.reshape(new_shape))

    if "nominal_eia" in data.attrs:
        new_dataset["earth_incidence_angle"] = (
            ("scans", "pixels", "channels"),
            np.broadcast_to(
                data.attrs["nominal_eia"].reshape(1, 1, -1), (n_scans_r, n_pixels_r, 15)
            ),
        )

    new_data = xr.Dataset(new_dataset)
    PreprocessorFile.write(output_file, new_data, template=template)


###############################################################################
# GPROF-NN 0D
###############################################################################


class Dataset0DBase:
    """
    Base class for batched datasets providing generic implementations of batch
    access and shuffling.
    """

    def __init__(self):
        seed = int.from_bytes(os.urandom(4), "big") + os.getpid()
        self._rng = np.random.default_rng(seed)

    def _shuffle(self):
        if not self._shuffled:
            LOGGER.info("Shuffling dataset %s.", self.filename.name)
            indices = self._rng.permutation(self.x.shape[0])
            self.x = self.x[indices, :]
            if isinstance(self.y, dict):
                self.y = {k: self.y[k][indices] for k in self.y}
            else:
                self.y = self.y[indices]
            self._shuffled = True

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        if i >= len(self):
            LOGGER.info("Finished iterating through dataset %s.", self.filename.name)
            raise IndexError()
        if i == 0:
            if self.shuffle:
                self._shuffle()
            if self.transform_zeros:
                self._transform_zeros()

        self._shuffled = False
        if self.batch_size is None:
            if isinstance(self.y, dict):
                return (
                    torch.tensor(self.x[[i], :]),
                    {k: torch.tensor(self.y[k][[i]]) for k in self.y},
                )

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)

        x = torch.tensor(self.x[i_start:i_end, :])
        if isinstance(self.y, dict):
            y = {k: torch.tensor(self.y[k][i_start:i_end]) for k in self.y}
        else:
            y = torch.tensor(self.y[i_start:i_end])

        return x, y

    def __len__(self):
        """
        The number of samples in the dataset.
        """
        if self.batch_size:
            n = self.x.shape[0] // self.batch_size
            if (self.x.shape[0] % self.batch_size) > 0:
                n = n + 1
            return n
        else:
            return self.x.shape[0]


class GPROF0DDataset(Dataset0DBase):
    """
    Dataset class providing an interface for the single-pixel GPROF-NN 0D
    retrieval algorithm.

    Attributes:
        x: Rank-2 tensor containing the input data with
           samples along first dimension.
        y: The target values
        filename: The filename from which the data is loaded.
        targets: List of names of target variables.
        batch_size: The size of data batches returned by __getitem__ method.
        normalizer: The normalizer used to normalize the data.
        shuffle: Whether or not the ordering of the data is shuffled.
        augment: Whether or not high-frequency observations are randomly set to
            missing to simulate observations at the edge of the swath.
    """

    def __init__(
        self,
        filename,
        targets=None,
        normalize=True,
        normalizer=None,
        transform_zeros=True,
        batch_size=512,
        shuffle=True,
        augment=True,
        sensor=None,
        permute=None,
        equalizer=None,
    ):
        """
        Create GPROF 0D dataset.

        Args:
            filename: Path to the NetCDF file containing the training data to
                load.
            targets: String or list of strings specifying the names of the
                variables to use as retrieval targets.
            normalize: Whether or not to normalize the input data.
            normalizer: Normalizer object  or class to use to normalize the
                 input data. If normalizer is a class object this object will
                 be initialized with the training input data. If 'None' a
                 ``quantnn.normalizer.MinMaxNormalizer`` will be used and
                 initialized with the loaded data.
            transform_zeros: Whether or not to replace very small values with
                random values.
            batch_size: Number of samples in each training batch.
            shuffle: Whether or not to shuffle the training data.
            augment: Whether or not to randomly mask high-frequency channels
                and to randomly permute ancillary data.
            sensor: Sensor object corresponding to the training data. Only
                necessary if the sensor cannot be inferred from the
                corresponding sensor attribute of the dataset file.
            permute: If not ``None`` the input feature corresponding to the
                given index will be permuted in order to break correlation
                between input and output.
        """
        super().__init__()
        self.filename = Path(filename)

        if targets is None:
            targets = ["surface_precip"]
        self.targets = targets
        self.transform_zeros = transform_zeros
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        if sensor is None:
            self.sensor = sensors.GMI
        else:
            self.sensor = sensor

        x, y = self.sensor.load_training_data_0d(
            filename, self.targets, self.augment, self._rng,
            equalizer=equalizer
        )
        self.x = x
        self.y = y
        LOGGER.info("Loaded %s samples from %s", self.x.shape[0], self.filename.name)

        indices_1h = list(range(self.sensor.n_inputs - 22, self.sensor.n_inputs))
        if normalizer is None:
            self.normalizer = MinMaxNormalizer(self.x, exclude_indices=indices_1h)
        elif isinstance(normalizer, type):
            self.normalizer = normalizer(self.x, exclude_indices=indices_1h)
        else:
            self.normalizer = normalizer

        self.normalize = normalize
        if normalize:
            self.x = self.normalizer(self.x)

        if transform_zeros:
            self._transform_zeros()

        if permute is not None:
            n_features = self.sensor.n_freqs + 2
            if isinstance(self.sensor, sensors.CrossTrackScanner):
                n_features += 1
            if permute < n_features:
                self.x[:, permute] = self._rng.permutation(self.x[:, permute])
            elif permute == n_features:
                self.x[:, -24:-4] = self._rng.permutation(self.x[:, -24:-4])
            else:
                self.x[:, -4:] = self._rng.permutation(self.x[:, -4:])

        self.x = self.x.astype(np.float32)
        if isinstance(self.y, dict):
            self.y = {k: self.y[k].astype(np.float32) for k in self.y}
        else:
            self.y = self.y.astype(np.float32)

        self._shuffled = False
        if self.shuffle:
            self._shuffle()

    def __repr__(self):
        return f"GPROF0DDataset({self.filename.name}, n_batches={len(self)})"

    def __str__(self):
        return f"GPROF0DDataset({self.filename.name}, n_batches={len(self)})"

    def _transform_zeros(self):
        """
        Transforms target values that are zero to small, non-zero values.
        """
        if isinstance(self.y, dict):
            y = self.y
        else:
            y = {self.targets: self.y}
        for k, y_k in y.items():
            threshold = _THRESHOLDS[k]
            indices = (y_k <= threshold) * (y_k >= -threshold)
            if indices.sum() > 0:
                t_l = np.log10(threshold)
                y_k[indices] = 10 ** self._rng.uniform(t_l - 4, t_l, indices.sum())

    def _load_data(self):
        """
        Loads the data from the file into the ``x`` and ``y`` attributes.
        """

    def save(self, filename):
        """
        Store dataset as NetCDF file.

        Args:
            filename: The name of the file to which to write the dataset.
        """
        if self.normalize:
            x = self.normalizer.invert(self.x)
        else:
            x = self.x

        n_samples = x.shape[0]
        n_pixels = 221
        n_scans = n_samples // 221
        n_levels = 28
        if (n_samples % 221) > 0:
            n_scans += 1

        bts = np.zeros((1, n_scans, n_pixels, 15), dtype=np.float32)
        t2m = np.zeros((1, n_scans, n_pixels), dtype=np.float32)
        tcwv = np.zeros((1, n_scans, n_pixels), dtype=np.float32)
        st = np.zeros((1, n_scans, n_pixels), dtype=np.float32)
        at = np.zeros((1, n_scans, n_pixels), dtype=np.float32)

        bts.reshape((-1, 15))[:n_samples] = x[:, :15]
        t2m.ravel()[:n_samples] = x[:, 15]
        tcwv.ravel()[:n_samples] = x[:, 16]
        st.ravel()[:n_samples] = np.where(x[:, 17 : 17 + 18])[1]
        at.ravel()[:n_samples] = np.where(x[:, 17 + 18 : 17 + 22])[1]

        dataset = xr.open_dataset(self.filename)

        dims = ("samples", "scans", "pixels", "channel")
        new_dataset = {
            "brightness_temperatures": (dims, bts),
            "two_meter_temperature": (dims[:-1], t2m),
            "total_column_water_vapor": (dims[:-1], tcwv),
            "surface_type": (dims[:-1], st),
            "airmass_type": (dims[:-1], at),
        }
        dims = ("samples", "scans", "pixels", "levels")
        if isinstance(self.y, dict):
            for k, v in self.y.items():
                shape = (1, n_scans, n_pixels, n_levels)
                n_dims = v.ndim
                data = np.zeros(shape[: 2 + n_dims], dtype=np.float32)
                if n_dims == 2:
                    data.reshape(-1, 28)[:n_samples] = v
                else:
                    data.ravel()[:n_samples] = v
                new_dataset[k] = (dims[: 2 + n_dims], data)
        else:
            shape = (1, n_scans, n_pixels)
            data = np.zeros(shape)
            data.reshape(-1)[:n_samples] = self.y
            new_dataset[k] = (dims[:-1], data)
            new_dataset["surface_precip"] = (
                (
                    "samples",
                    "scans",
                    "pixels",
                ),
                self.y[np.newaxis, :],
            )
        new_dataset = xr.Dataset(new_dataset)
        new_dataset.attrs = dataset.attrs
        new_dataset.to_netcdf(filename)


class TrainingObsDataset0D(GPROF0DDataset):
    """
    Special training dataset that serves only the simulated brightness
    temperatures and ancillary data in order to train an observation
    noise model.
    """

    def __init__(
        self,
        filename,
        batch_size=512,
        normalize=True,
        normalizer_x=None,
        normalizer_y=None,
        shuffle=True,
        augment=True,
        sensor=None,
    ):
        """
        Args:
            filename: Path to the NetCDF file containing the training data to
                load.
            batch_size: Number of samples in each training batch.
            normalize: Whether or not to normalize the input data.
            normalizer_x: Normalizer to use to normalize the input data.
            normalizer_y: Normalizer to use to normalizer the target data.
            shuffle: Whether or not to shuffle the training data.
            augment: Whether or not to randomly mask high-frequency channels
                and to randomly permute ancillary data.
            sensor: Sensor object defining the sensor from which the training
                data stems.
        """
        super().__init__(
            filename,
            normalize=False,
            transform_zeros=False,
            targets=[],
            batch_size=batch_size,
            shuffle=shuffle,
            augment=augment,
            sensor=sensor,
        )
        self.normalize = normalize
        self.normalizer_x = normalizer_x
        self.normalizer_y = normalizer_y

        # Extract observations and ancillary data.
        self.y = self.x[:, : sensor.n_freqs]
        features = []
        if isinstance(sensor, sensors.CrossTrackScanner):
            features += [self.x[:, [sensor.n_freqs]]]
        features += [self.x[:, -22:-4]]
        self.x = np.concatenate(features, axis=1)

        valid = np.all(np.isfinite(self.y), axis=-1)
        valid *= np.sum(self.x[:, 1:], axis=-1) > 0
        valid *= (self.x[:, 5] >= -1.0) * (self.x[:, 5] <= 1.0)
        self.y = self.y[valid]
        self.x = self.x[valid]

        # Normalize ancillary data and observations
        # independently.
        if self.normalize:
            if self.normalizer_x is None:
                indices_1h = list(range(1, 19))
                self.normalizer_x = MinMaxNormalizer(self.x, exclude_indices=indices_1h)
            self.x = self.normalizer_x(self.x)
            if self.normalizer_y is None:
                self.normalizer_y = MinMaxNormalizer(self.y)
            self.y = self.normalizer_y(self.y)

    def __repr__(self):
        s = f"TrainingObsDataset0D({self.filename.name}, " f"n_batches={len(self)})"
        return s

    def __str__(self):
        s = f"TrainingObsDataset0D({self.filename.name}, " f"n_batches={len(self)})"
        return s


def _replace_randomly(x, p, rng=None):
    """
    Randomly replaces a fraction of the elements in the tensor with another
    randomly sampled value.

    Args:
        x: The input tensor in which to replace some values by random
            permutations.
        p: The probability with which to replace any value along the first
            dimension in x.

    Returns:
         None, augmentation is performed in place.
    """
    if rng is None:
        indices = np.random.rand(x.shape[0]) > (1.0 - p)
        indices_r = np.random.permutation(x.shape[0])[: indices.sum()]
    else:
        indices = rng.random(x.shape[0]) > (1.0 - p)
        indices_r = rng.permutation(x.shape[0])[: indices.sum()]
    replacements = x[indices_r, ...]
    x[indices] = replacements


def _expand_pixels(data):
    """
    Expand target data array that only contain data for central pixels.

    Args:
        data: Array containing data of a retrieval target variable.

    Return:
        The input data expanded to the full GMI swath along the third
        dimension.
    """
    if len(data.shape) <= 2 or data.shape[2] == 221:
        return data
    new_shape = list(data.shape)
    new_shape[2] = 221

    i_start = (221 - data.shape[2]) // 2

    data_new = np.zeros(new_shape, dtype=data.dtype)
    data_new[:] = np.nan
    data_new[:, :, i_start:-i_start] = data
    return data_new


###############################################################################
# GPROF-NN 2D
###############################################################################


class SimulatorDataset:
    """
    Dataset to train a simulator network to predict simulated brightness
    temperatures and brightness temperature biases.
    """
    def __init__(
        self,
        filename,
        normalize=True,
        batch_size=32,
        normalizer=None,
        shuffle=True,
        augment=True,
    ):
        """
        Args:
            filename: Path to the NetCDF file containing the training data.
            normalize: Whether or not to normalize the input data.
            batch_size: Number of samples in each training batch.
            normalizer: The normalizer used to normalize the data.
            shuffle: Whether or not to shuffle the training data.
            augment: Whether or not to randomly mask high-frequency channels
                and to randomly permute ancillary data.
        """
        self.filename = Path(filename)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        seed = int.from_bytes(os.urandom(4), "big") + os.getpid()
        self._rng = np.random.default_rng(seed)
        self._load_data()

        indices_1h = list(range(17, 39))
        if normalizer is None:
            self.normalizer = MinMaxNormalizer(self.x, exclude_indices=indices_1h)
        elif isinstance(normalizer, type):
            self.normalizer = normalizer(self.x, exclude_indices=indices_1h)
        else:
            self.normalizer = normalizer

        self.normalize = normalize
        if normalize:
            self.x = self.normalizer(self.x)

        self.x = self.x.astype(np.float32)
        self.y = {k: self.y[k].astype(np.float32) for k in self.y}

        self._shuffled = False
        if self.shuffle:
            self._shuffle()

    def __repr__(self):
        return f"SimulatorDataset({self.filename.name}, n_batches={len(self)})"

    def __str__(self):
        return self.__repr__()

    def _load_data(self):
        """
        Loads the data from the file into ``x`` and ``y`` attributes.
        """
        with xr.open_dataset(self.filename) as dataset:

            # Load only simulated data.
            dataset = dataset[{"samples": dataset.source == 0}]
            variables = dataset.variables

            # Brightness temperatures
            n = dataset.samples.size

            x = np.zeros((n, 39, M, N))
            y = {}

            for i in range(n):

                if self.augment:
                    p_x_o = 2.0 * self._rng.random() - 1.0
                    p_x_i = 2.0 * self._rng.random() - 1.0
                    p_y = 2.0 * self._rng.random() - 1.0
                else:
                    p_x_o = 0.0
                    p_x_i = 0.0
                    p_y = 0.0

                coords = get_transformation_coordinates(p_x_i, p_x_o, p_y)

                #
                # Simulator input
                #

                tbs = dataset["brightness_temperatures_gmi"][i][:].data
                tbs = extract_domain(tbs, p_x_i, p_x_o, p_y, coords=coords)
                tbs = np.transpose(tbs, (2, 0, 1))

                invalid = (tbs > 500.0) + (tbs < 0.0)
                tbs[invalid] = np.nan

                # Simulate missing high-frequency channels
                if self.augment:
                    r = self._rng.random()
                    n_p = self._rng.integers(10, 30)
                    if r > 0.9:
                        tbs[10:15, :, :n_p] = np.nan

                t2m = dataset["two_meter_temperature"][i][:].data
                t2m = extract_domain(t2m, p_x_i, p_x_o, p_y, coords=coords)
                t2m = t2m[np.newaxis, ...]
                t2m[t2m < 0] = np.nan

                tcwv = dataset["total_column_water_vapor"][i][:].data
                tcwv = extract_domain(tcwv, p_x_i, p_x_o, p_y, coords=coords)
                tcwv = tcwv[np.newaxis, ...]
                tcwv[tcwv < 0] = np.nan

                st = dataset["surface_type"][i][:].data
                st = extract_domain(st, p_x_i, p_x_o, p_y, coords=coords, order=0)
                st_1h = np.zeros((18,) + st.shape, dtype=np.float32)
                for j in range(18):
                    st_1h[j, st == (j + 1)] = 1.0

                at = dataset["airmass_type"][i][:].data
                at = extract_domain(at, p_x_i, p_x_o, p_y, coords=coords, order=0)
                at_1h = np.zeros((4,) + st.shape, dtype=np.float32)
                for j in range(4):
                    at_1h[j, np.maximum(at, 0) == j] = 1.0

                x[i] = np.concatenate([tbs, t2m, tcwv, st_1h, at_1h], axis=0)

                dims = (dataset.angles.size, dataset.channels.size)

                #
                # Simulator output
                #

                targets = [
                    "simulated_brightness_temperatures",
                    "brightness_temperature_biases"
                ]

                for k in targets:
                    y_k_r = _expand_pixels(dataset[k].data[i][:][np.newaxis, ...])
                    y_k = y.setdefault(
                        k,
                        np.zeros((n, ) + dims[-(y_k_r.ndim - 3):] + (M, N), dtype=np.float32),
                    )
                    y_k_i = extract_domain(
                        y_k_r[0], p_x_i, p_x_o, p_y, coords=coords
                    )
                    np.nan_to_num(y_k_i, copy=False, nan=-9999)

                    dims_sp = tuple(range(2))
                    dims_t = tuple(range(2, y_k_i.ndim))
                    y_k[i] = np.transpose(y_k_i, dims_t + dims_sp)

                # Also flip data if requested.
                if self.augment:
                    r = self._rng.random()
                    if r > 0.5:
                        x[i] = np.flip(x[i], -2)
                        for k in targets:
                            y[k][i] = np.flip(y[k][i], -2)

                    r = self._rng.random()
                    if r > 0.5:
                        x[i] = np.flip(x[i], -1)
                        for k in targets:
                            y[k][i] = np.flip(y[k][i], -1)
        self.x = x
        self.y = y

    def _shuffle(self):
        if not self._shuffled:
            LOGGER.info("Shuffling dataset %s.", self.filename.name)
            indices = self._rng.permutation(self.x.shape[0])
            self.x = self.x[indices, :]
            if isinstance(self.y, dict):
                self.y = {k: self.y[k][indices] for k in self.y}
            else:
                self.y = self.y[indices]
            self._shuffled = True

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        if i >= len(self):
            LOGGER.info("Finished iterating through dataset %s.", self.filename.name)
            raise IndexError()
        if i == 0:
            self._shuffle()

        self._shuffled = False
        if self.batch_size is None:
            if isinstance(self.y, dict):
                return (
                    torch.tensor(self.x[[i], :]),
                    {k: torch.tensor(self.y[k][[i]]) for k in self.y},
                )

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)

        x = torch.tensor(self.x[i_start:i_end, :])
        if isinstance(self.y, dict):
            y = {k: torch.tensor(self.y[k][i_start:i_end]) for k in self.y}
        else:
            y = torch.tensor(self.y[i_start:i_end])

        return x, y

    def __len__(self):
        """
        The number of samples in the dataset.
        """
        if self.batch_size:
            n = self.x.shape[0] // self.batch_size
            if self.x.shape[0] % self.batch_size > 0:
                n = n + 1
            return n
        else:
            return self.x.shape[0]


class GPROF2DDataset:
    """
    Dataset class providing an interface for the convolutional GPROF-NN 2D
    retrieval algorithm.

    Attributes:
        x: Rank-4 tensor containing the input data with
           samples along first dimension and channels along second.
        y: The target values
        filename: The filename from which the data is loaded.
        target: The name of the variable(s) used as retrieval target(s).
        batch_size: The size of data batches returned by __getitem__ method.
        normalizer: The normalizer used to normalize the data.
        shuffle: Whether or not the ordering of the data is shuffled.
        augment: Whether or not data augmentation is applied.
    """

    def __init__(
        self,
        filename,
        target="surface_precip",
        normalize=True,
        transform_zeros=True,
        batch_size=32,
        normalizer=None,
        shuffle=True,
        augment=True,
    ):
        """
        Load GPROF 2D data.

        Args:
            filename: Path to the NetCDF file containing the training data to
                load.
            target: String or list of strings specifying the names of the
                variables to use as retrieval targets.
            normalize: Whether or not to normalize the input data.
            transform_zeros: Whether or not to replace very small
                values with random values.
            batch_size: Number of samples in each training batch.
            shuffle: Whether or not to shuffle the training data.
            augment: Whether or not to randomly mask high-frequency channels
                and to randomly permute ancillary data.
        """
        self.filename = Path(filename)
        self.target = target
        self.transform_zeros = transform_zeros
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        seed = int.from_bytes(os.urandom(4), "big") + os.getpid()
        self._rng = np.random.default_rng(seed)

        self._load_data()

        indices_1h = list(range(17, 39))
        if normalizer is None:
            self.normalizer = MinMaxNormalizer(self.x, exclude_indices=indices_1h)
        elif isinstance(normalizer, type):
            self.normalizer = normalizer(self.x, exclude_indices=indices_1h)
        else:
            self.normalizer = normalizer

        self.normalize = normalize
        if normalize:
            self.x = self.normalizer(self.x)

        if transform_zeros:
            self._transform_zeros()

        self.x = self.x.astype(np.float32)
        if isinstance(self.y, dict):
            self.y = {k: self.y[k].astype(np.float32) for k in self.y}
        else:
            self.y = self.y.astype(np.float32)

        self._shuffled = False
        if self.shuffle:
            self._shuffle()

    def __repr__(self):
        return f"GPROF2DDataset({self.filename.name}, n_batches={len(self)})"

    def __str__(self):
        return self.__repr__()

    def _transform_zeros(self):
        """
        Transforms target values that are zero to small, non-zero values.
        """
        if isinstance(self.y, dict):
            y = self.y
        else:
            y = {self.target: self.y}
        for k, y_k in y.items():
            threshold = _THRESHOLDS[k]
            indices = (y_k <= threshold) * (y_k >= -threshold)
            if indices.sum() > 0:
                t_l = np.log10(threshold)
                y_k[indices] = 10 ** self._rng.uniform(t_l - 4, t_l, indices.sum())

    def _load_data(self):
        """
        Loads the data from the file into the ``x`` and ``y`` attributes.
        """
        with Dataset(self.filename, "r") as dataset:

            variables = dataset.variables

            #
            # Input data
            #

            # Brightness temperatures
            n = dataset.dimensions["samples"].size

            x = np.zeros((n, 39, M, N))
            if isinstance(self.target, list):
                y = {}
            else:
                y = np.zeros(
                    (
                        n,
                        M,
                        N,
                    )
                    + dataset[self.target][0].shape[3:],
                    dtype=np.float32,
                )
            for i in range(n):
                if self.augment:
                    p_x_o = 2.0 * self._rng.random() - 1.0
                    p_x_i = 2.0 * self._rng.random() - 1.0
                    p_y = 2.0 * self._rng.random() - 1.0
                else:
                    p_x_o = 0.0
                    p_x_i = 0.0
                    p_y = 0.0

                coords = get_transformation_coordinates(p_x_i, p_x_o, p_y)

                tbs = dataset["brightness_temperatures"][i][:]
                tbs = extract_domain(tbs, p_x_i, p_x_o, p_y, coords=coords)
                tbs = np.transpose(tbs, (2, 0, 1))

                invalid = (tbs > 500.0) + (tbs < 0.0)
                tbs[invalid] = np.nan

                # Simulate missing high-frequency channels
                if self.augment:
                    r = self._rng.random()
                    n_p = self._rng.integers(10, 30)
                    if r > 0.80:
                        tbs[:, 10:15, :n_p] = np.nan

                t2m = variables["two_meter_temperature"][i][:]
                t2m = extract_domain(t2m, p_x_i, p_x_o, p_y, coords=coords)
                t2m = t2m[np.newaxis, ...]
                t2m[t2m < 0] = np.nan

                tcwv = variables["total_column_water_vapor"][i][:]
                tcwv = extract_domain(tcwv, p_x_i, p_x_o, p_y, coords=coords)
                tcwv = tcwv[np.newaxis, ...]
                tcwv[tcwv < 0] = np.nan

                st = dataset["surface_type"][i][:]
                st = extract_domain(st, p_x_i, p_x_o, p_y, coords=coords, order=0)
                st_1h = np.zeros((18,) + st.shape, dtype=np.float32)
                for j in range(18):
                    st_1h[j, st == (j + 1)] = 1.0

                at = dataset["airmass_type"][i][:]
                at = extract_domain(at, p_x_i, p_x_o, p_y, coords=coords, order=0)
                at_1h = np.zeros((4,) + st.shape, dtype=np.float32)
                for j in range(4):
                    at_1h[j, np.maximum(at, 0) == j] = 1.0

                x[i] = np.concatenate([tbs, t2m, tcwv, st_1h, at_1h], axis=0)

                dims = (n, 28)
                if isinstance(self.target, list):
                    for k in self.target:
                        y_k_r = _expand_pixels(dataset[k][i][:][np.newaxis, ...])
                        y_k = y.setdefault(
                            k,
                            np.zeros(dims[: y_k_r.ndim - 2] + (M, N), dtype=np.float32),
                        )
                        y_k_i = extract_domain(
                            y_k_r[0], p_x_i, p_x_o, p_y, coords=coords
                        )
                        np.nan_to_num(y_k_i, copy=False, nan=-9999)
                        if k == "latent_heat":
                            y_k_i[y_k_i < -400] = -9999
                        else:
                            y_k_i[y_k_i < 0] = -9999
                        if y_k_i.ndim > 2:
                            y_k[i] = np.transpose(y_k_i, (2, 0, 1))
                        else:
                            y_k[i] = y_k_i

                else:
                    y_r = _expand_pixels(dataset[self.target][i][:][np.newaxis, ...])
                    y_i = extract_domain(y_r[0], p_x_i, p_x_o, p_y, coords=coords)
                    np.nan_to_num(y_i, copy=False, nan=-9999)
                    if self.target == "latent_heat":
                        y_i[y_i < -400] = -9999
                    else:
                        y_i[y_i < 0] = -9999
                    if y_i.ndim > 2:
                        y[i] = np.transpose(y_i, (2, 0, 1))
                    else:
                        y[i] = y_i

                # Also flip data if requested.
                if self.augment:
                    r = self._rng.random()
                    if r > 0.5:
                        x[i] = np.flip(x[i], -2)
                        if isinstance(self.target, list):
                            for k in self.target:
                                y[k][i] = np.flip(y[k][i], -2)
                        else:
                            y[i] = np.flip(y[i], -2)

                    r = self._rng.random()
                    if r > 0.5:
                        x[i] = np.flip(x[i], -1)
                        if isinstance(self.target, list):
                            for k in self.target:
                                y[k][i] = np.flip(y[k][i], -1)
                        else:
                            y[i] = np.flip(y[i], -1)
        self.x = x
        self.y = y

    def _shuffle(self):
        if not self._shuffled:
            LOGGER.info("Shuffling dataset %s.", self.filename.name)
            indices = self._rng.permutation(self.x.shape[0])
            self.x = self.x[indices, :]
            if isinstance(self.y, dict):
                self.y = {k: self.y[k][indices] for k in self.y}
            else:
                self.y = self.y[indices]
            self._shuffled = True

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        if i >= len(self):
            LOGGER.info("Finished iterating through dataset %s.", self.filename.name)
            raise IndexError()
        if i == 0:
            self._shuffle()
            if self.transform_zeros:
                self._transform_zeros()

        self._shuffled = False
        if self.batch_size is None:
            if isinstance(self.y, dict):
                return (
                    torch.tensor(self.x[[i], :]),
                    {k: torch.tensor(self.y[k][[i]]) for k in self.y},
                )

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)

        x = torch.tensor(self.x[i_start:i_end, :])
        if isinstance(self.y, dict):
            y = {k: torch.tensor(self.y[k][i_start:i_end]) for k in self.y}
        else:
            y = torch.tensor(self.y[i_start:i_end])

        return x, y

    def __len__(self):
        """
        The number of samples in the dataset.
        """
        if self.batch_size:
            n = self.x.shape[0] // self.batch_size
            if self.x.shape[0] % self.batch_size > 0:
                n = n + 1
            return n
        else:
            return self.x.shape[0]
