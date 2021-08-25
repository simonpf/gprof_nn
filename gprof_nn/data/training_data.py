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

from scipy.signal import convolve
import numpy as np
import torch
import xarray as xr
from netCDF4 import Dataset

from quantnn.normalizer import MinMaxNormalizer

from gprof_nn import sensors
from gprof_nn.utils import apply_limits
from gprof_nn.data.utils import load_variable, decompress_scene, remap_scene
from gprof_nn.definitions import MASKED_OUTPUT, LIMITS
from gprof_nn.data.utils import expand_pixels
from gprof_nn.data.preprocessor import PreprocessorFile
from gprof_nn.augmentation import (
    extract_domain,
    get_transformation_coordinates
)

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
    Extract samples from training dataset and write to a preprocessor
    file.

    This functions serves as an interface between the GPROF-NN training
    data and the GPROF legacy algorithm as it can be used to create
    preprocessor files with the observations. These can then be used
    as input for the legacy GPROF.

    Note: If the input isn't organized into scenes with dimensions
    'scans' and 'pixels' the number of samples that will be written to
    the file will be the largest multiple of 256 that is smaller than
    or equal to the original number of samples. This means that up to
    255 samples may be lost when writing them to a preprocessor file.

    Args:
        input_data: Path to a NetCDF4 file containing the training or test
            data or 'xarray.Dataset containing the data to write to a
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

    if "pixels" not in data.dims or "scans" not in data.dims:
        n_pixels = 256
        n_scans = data.samples.size // n_pixels
        n_scenes = 1
        dim_offset = -1
    else:
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
            dims = ("scans", "pixels") + da.dims[2 + dim_offset :]
            if "pixels_center" in dims:
                continue
            n_elems = np.prod(new_shape)
            elements = da.data.ravel()[:n_elems]
            new_dataset[k] = (dims, elements.reshape(new_shape))

    if "nominal_eia" in data.attrs:
        new_dataset["earth_incidence_angle"] = (
            ("scans", "pixels", "channels"),
            np.broadcast_to(
                data.attrs["nominal_eia"].reshape(1, 1, -1), (n_scans_r, n_pixels_r, 15)
            ),
        )

    sensor = getattr(sensors, data.attrs["sensor"])
    new_data = xr.Dataset(new_dataset)
    PreprocessorFile.write(output_file, new_data, sensor, template=template)


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


class GPROF_NN_0D_Dataset(Dataset0DBase):
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

        with xr.open_dataset(self.filename) as dataset:
            if "sensor" not in dataset.attrs:
                raise Exception(f"Provided dataset lacks 'sensor' attribute.")
            sensor_name = dataset.attrs["sensor"]
            sensor = getattr(sensors, sensor_name, None)
            if sensor is None:
                raise Exception(f"Sensor {sensor_name} isn't supported yet")
        self.sensor = sensor

        x, y = self.sensor.load_training_data_0d(
            filename, self.targets, self.augment, self._rng, equalizer=equalizer
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
            n_features = self.sensor.n_chans + 2
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
        return f"GPROF_NN_0D_Dataset({self.filename.name}, n_batches={len(self)})"

    def __str__(self):
        return f"GPROF_NN_0D_Dataset({self.filename.name}, n_batches={len(self)})"

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

    def to_xarray_dataset(self,
                          mask=None,
                          batch=None):
        """
        Convert training data to xarray dataset.

        Args:
            mask: A mask to select samples to include in the dataset.

        Return:
            An 'xarray.Dataset' containing the training data but converted
            back to the original format.
        """
        if batch is None:
            x = self.x
            y = self.y
        else:
            x, y = batch

        if mask is None:
            mask = slice(0, None)

        if self.normalize:
            x = self.normalizer.invert(x[mask])
        else:
            x = x[mask]
        sensor = self.sensor

        n_samples = x.shape[0]
        n_levels = 28

        tbs = x[:, : sensor.n_chans]
        if sensor.n_angles > 1:
            eia = x[:, self.n_chans]
        else:
            eia = None
        t2m = x[:, -24]
        tcwv = x[:, -23]
        st = np.zeros(n_samples, dtype=np.int32)
        i, j = np.where(x[:, -22:-4])
        st[i] = j + 1
        at = np.zeros(n_samples, dtype=np.int32) + 1
        i, j = np.where(x[:, -4:])
        at[i] = j

        dims = ("samples", "channels")
        new_dataset = {
            "brightness_temperatures": (dims, tbs),
            "two_meter_temperature": (dims[:-1], t2m),
            "total_column_water_vapor": (dims[:-1], tcwv),
            "surface_type": (dims[:-1], st),
            "airmass_type": (dims[:-1], at),
        }
        if eia is not None:
            new_dataset["earth_incidence_angle"] = (dims[:2], eia)

        dims = ("samples", "levels")
        for k, v in y.items():
            n_dims = v.ndim
            new_dataset[k] = (dims[:n_dims], v)

        new_dataset = xr.Dataset(new_dataset)
        with xr.open_dataset(self.filename) as dataset:
            new_dataset.attrs = dataset.attrs
        return new_dataset

    def save(self, filename):
        """
        Store dataset as NetCDF file.

        Args:
            filename: The name of the file to which to write the dataset.
        """
        new_dataset = self.to_xarray_dataset()
        new_dataset.to_netcdf(filename)


class TrainingObsDataset0D(GPROF_NN_0D_Dataset):
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
        self.y = self.x[:, : sensor.n_chans]
        features = []
        if isinstance(sensor, sensors.CrossTrackScanner):
            features += [self.x[:, [sensor.n_chans]]]
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


###############################################################################
# GPROF-NN 2D
###############################################################################


class GPROF_NN_2D_Dataset:
    """
    Base class for GPROF-NN 2D-retrieval training data in which training
    samples consist of 2D scenes of input data and corresponding target
    fields.

    Objects of this class act as an iterator over batches in the training
    data set.
    """

    def __init__(
        self,
        filename,
        targets=None,
        batch_size=32,
        normalize=True,
        normalizer=None,
        transform_zeros=True,
        shuffle=True,
        augment=True,
    ):
        """
        Args:
            filename: Path of the NetCDF file containing the training data.
            sensor: The sensor object to use to load the data.
            targets: List of the targets to load from the data.
            batch_size: The size of batches in the training data.
            normalize: Whether or not to noramlize the input data.
            normalizer: Normalizer object to use to normalize the input
                data. May alternatively be a normalizer class that will
                be used to instantiate a new normalizer object with the loaded
                input data. If 'None', a new ``quantnn.normalizer.MinMaxNormalizer``
                will be created with the loaded input data.
            transform_zeros: Whether or not to transform target values that are
                zero to small random values.
            shuffle: Whether or not to shuffle the data.
            augment: Whether or not to augment the training data.
        """
        self.filename = Path(filename)
        if targets is None:
            self.targets = ["surface_precip"]
        else:
            self.targets = targets

        self.transform_zeros = transform_zeros
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        seed = int.from_bytes(os.urandom(4), "big") + os.getpid()
        self._rng = np.random.default_rng(seed)

        sensor = xr.open_dataset(filename).attrs["sensor"]
        sensor = getattr(sensors, sensor)
        x, y = sensor.load_training_data_2d(filename, self.targets, augment, self._rng)
        self.sensor = sensor

        indices_1h = list(range(17, 39))
        if normalizer is None:
            self.normalizer = MinMaxNormalizer(x, exclude_indices=indices_1h)
        elif isinstance(normalizer, type):
            self.normalizer = normalizer(x, exclude_indices=indices_1h)
        else:
            self.normalizer = normalizer

        self.normalize = normalize
        if normalize:
            x = self.normalizer(x)

        self.x = x
        self.y = y

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
        return f"GPROF_NN_2D_Dataset({self.filename.name}, n_batches={len(self)})"

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

    def _shuffle(self):
        if not self._shuffled and self.shuffle:
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

    def to_xarray_dataset(self, mask=None):
        """
        Convert training data to xarray dataset.

        Args:
            mask: A mask to select samples to include in the dataset.

        Return:
            An 'xarray.Dataset' containing the training data but converted
            back to the original format.
        """
        if mask is None:
            mask = slice(0, None)
        if self.normalize:
            x = self.normalizer.invert(self.x[mask])
        else:
            x = self.x[mask]
        sensor = self.sensor

        n_samples = x.shape[0]
        n_levels = 28

        tbs = np.transpose(x[:, : sensor.n_chans], (0, 2, 3, 1))
        if sensor.n_angles > 1:
            eia = x[:, self.n_chans]
        else:
            eia = None
        t2m = x[:, -24]
        tcwv = x[:, -23]

        st = np.zeros(t2m.shape, dtype=np.int32)
        for i in range(18):
            mask = x[:, -22 + i] == 1
            st[mask] = i + 1

        at = np.zeros(t2m.shape, dtype=np.int32)
        for i in range(4):
            mask = x[:, -4 + i] == 1
            at[mask] = i

        dims = ("samples", "scans", "pixels", "channels")
        new_dataset = {
            "brightness_temperatures": (dims, tbs),
            "two_meter_temperature": (dims[:-1], t2m),
            "total_column_water_vapor": (dims[:-1], tcwv),
            "surface_type": (dims[:-1], st),
            "airmass_type": (dims[:-1], at),
        }
        if eia is not None:
            new_dataset["earth_incidence_angle"] = (dims[:2], eia)

        dims = ("samples", "scans", "pixels", "levels")
        for k, v in self.y.items():
            n_dims = v.ndim
            if n_dims > 3:
                v = np.transpose(v, (0, 2, 3, 1))
            new_dataset[k] = (dims[:n_dims], v)

        new_dataset = xr.Dataset(new_dataset)
        with xr.open_dataset(self.filename) as dataset:
            new_dataset.attrs = dataset.attrs
        return new_dataset

    def save(self, filename):
        """
        Store dataset as NetCDF file.

        Args:
            filename: The name of the file to which to write the dataset.
        """
        new_dataset = self.to_xarray_dataset()
        new_dataset.to_netcdf(filename)


class SimulatorDataset(GPROF_NN_2D_Dataset):
    """
    Dataset to train a simulator network to predict simulated brightness
    temperatures and brightness temperature biases.
    """

    def __init__(
        self,
        filename,
        batch_size=32,
        normalize=True,
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
        targets = ["simulated_brightness_temperatures", "brightness_temperature_biases"]
        self.transform_zeros = False
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        seed = int.from_bytes(os.urandom(4), "big") + os.getpid()
        self._rng = np.random.default_rng(seed)

        dataset = xr.open_dataset(filename)
        dataset = dataset[{"samples": dataset.source == 0}]
        x, y = self.load_training_data_2d(dataset, targets, augment, self._rng)
        indices_1h = list(range(17, 39))
        if normalizer is None:
            self.normalizer = MinMaxNormalizer(x, exclude_indices=indices_1h)
        elif isinstance(normalizer, type):
            self.normalizer = normalizer(x, exclude_indices=indices_1h)
        else:
            self.normalizer = normalizer

        self.normalize = normalize
        if normalize:
            x = self.normalizer(x)

        self.x = x
        self.y = y

        self.x = self.x.astype(np.float32)
        if isinstance(self.y, dict):
            self.y = {k: self.y[k].astype(np.float32) for k in self.y}
        else:
            self.y = self.y.astype(np.float32)

        self._shuffled = False
        if self.shuffle:
            self._shuffle()

    def load_training_data_2d(self, dataset, targets, augment, rng):
        """
        Load data for training a simulator data.

        This function is different from the standard loading function in that
        the input observations are always the GMI observations.

        Args:
            dataset: The 'xarray.Dataset' from which to load the training
                data.
            targets: List of the targets to load.
            augment: Whether or not to augment the training data.
            rng: 'numpy.random.Generator' to use to generate random numbers.

        Return:

            Tuple ``(x, y)`` containing the training input ``x`` and a
            dictionary of target data ``y``.
        """
        sensor = getattr(sensors, dataset.attrs["sensor"])

        #
        # Input data
        #

        # Brightness temperatures
        n = dataset.samples.size

        x = []
        y = {}

        vs = ["latitude", "longitude"]
        if sensor != sensors.GMI:
            vs += ["brightness_temperatures_gmi"]

        for i in range(n):

            scene = decompress_scene(dataset[{"samples": i}], targets + vs)

            if augment:
                p_x_o = rng.random()
                p_x_i = rng.random()
                p_y = rng.random()
            else:
                p_x_o = 0.0
                p_x_i = 0.0
                p_y = 0.0

            lats = scene.latitude.data
            lons = scene.longitude.data
            coords = get_transformation_coordinates(
                lats, lons, sensor.viewing_geometry, 96, 128, p_x_i, p_x_o, p_y
            )

            scene = remap_scene(scene, coords, targets + vs)

            #
            # Input data
            #

            if sensor == sensors.GMI:
                tbs = sensor.load_brightness_temperatures(scene)
            else:
                tbs = load_variable(scene, "brightness_temperatures_gmi")
            tbs = np.transpose(tbs, (2, 0, 1))
            if augment:
                r = rng.random()
                n_p = rng.integers(10, 30)
                if r > 0.80:
                    tbs[10:15, :, :n_p] = np.nan
            t2m = sensor.load_two_meter_temperature(scene)[np.newaxis]
            tcwv = sensor.load_total_column_water_vapor(scene)[np.newaxis]
            st = sensor.load_surface_type(scene)
            st = np.transpose(st, (2, 0, 1))
            am = sensor.load_airmass_type(scene)
            am = np.transpose(am, (2, 0, 1))
            x.append(np.concatenate([tbs, t2m, tcwv, st, am], axis=0))

            #
            # Output data
            #

            for t in targets:
                y_t = sensor.load_target(scene, t, None)
                y_t = np.nan_to_num(y_t, nan=MASKED_OUTPUT)
                dims_sp = tuple(range(2))
                dims_t = tuple(range(2, y_t.ndim))

                y.setdefault(t, []).append(np.transpose(y_t, dims_t + dims_sp))

            # Also flip data if requested.
            if augment:
                r = rng.random()
                if r > 0.5:
                    x[i] = np.flip(x[i], -2)
                    for k in targets:
                        y[k][i] = np.flip(y[k][i], -2)

                r = rng.random()
                if r > 0.5:
                    x[i] = np.flip(x[i], -1)
                    for k in targets:
                        y[k][i] = np.flip(y[k][i], -1)

        x = np.stack(x)
        for k in targets:
            y[k] = np.stack(y[k])

        return x, y
