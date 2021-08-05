"""
================
gprof_nn.sensors
================

This module defines different sensor classes to represent the
 various sensors of the GPM constellation.
"""
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
from scipy.signal import convolve
import xarray as xr

from gprof_nn.definitions import N_LAYERS, LIMITS
from gprof_nn.data.utils import (load_variable,
                                 decompress_scene,
                                 remap_scene)
from gprof_nn.data import types

from gprof_nn.utils import (apply_limits,
                            calculate_interpolation_weights,
                            interpolate)
from gprof_nn.data.utils import expand_pixels
from gprof_nn.augmentation import (
    GMI_GEOMETRY,
    MHS_GEOMETRY,
    get_transformation_coordinates,
    extract_domain,
)

DATE_TYPE = np.dtype(
    [
        ("year", "i4"),
        ("month", "i4"),
        ("day", "i4"),
        ("hour", "i4"),
        ("minute", "i4"),
        ("second", "i4"),
    ]
)

MASKED_OUTPUT = -9999

###############################################################################
# Helper functions
###############################################################################


_BIAS_SCALES_GMI = 1.0 / np.cos(np.deg2rad([52.8, 49.19, 49.19, 49.19, 49.19]))


def calculate_smoothing_kernel(res_x_source,
                               res_a_source,
                               res_x_target,
                               res_a_target,
                               size):
    """
    Calculate smoothing kernel to smooth a field observed at
    the resolution of a source sensor to that of a given target
    sensor.
    """
    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    y, x = np.meshgrid(x, x, indexing="ij")
    x = x * res_x_source / res_x_target
    y = y * res_a_source / res_a_target
    w = np.exp(np.log(0.5) * (x ** 2 + y ** 2))
    w = w / w.sum()
    return w


def calculate_smoothing_kernels(sensor,
                                geometry):
    """
    Calculate smoothing kernels to average variables given on the GMI swath
    to a lower resolution.

    Args:
        sensor: Sensor object representing the sensor to which to smooth the
             variables.
        geometry: Viewing geometry of the sensor.

    Return:
        A list of 2D Gaussian convolution kernels.
    """
    res_x_source = 14.4e3
    res_a_source = 8.6e3
    angles = sensor.angles
    res_x_target = geometry.get_resolution_x(angles)
    res_a_target = geometry.get_resolution_a()

    kernels = []
    for res in res_x_target:
        k = calculate_smoothing_kernel(res_x_source,
                                       res_a_source,
                                       res,
                                       res_a_target,
                                       size=11)
        kernels.append(k)
    return kernels


def smooth_gmi_field(field,
                     kernels):
    """
    Smooth a variable using precomputed kernels.

    Args:
        field: The variable which to smooth. The first two dimensions
            are assumed to correspond to scans and pixels, respectively.
        kernels: List of kernels to use for the smoothing.

    Args:
        A new field with an new dimension at index 2 corresponding to the
        smoothed fields for each of the kernels in 'kernels'.
    """
    mask = np.isfinite(field)
    mask_f = mask.astype(np.float32)
    field = np.nan_to_num(field, nan=0.0)
    smoothed = []
    for k in kernels:
        if field.ndim > k.ndim:
            shape = k.shape + (1,) * (field.ndim - k.ndim)
            k = k.reshape(shape)
        field_s = convolve(field, k, mode="same")
        counts = convolve(mask_f, k, mode="same")
        field_s[~mask] = np.nan
        smoothed.append(field_s / counts)

    return np.stack(smoothed, axis=2)


def calculate_bias_scaling(angles):
    """
    Calculate the scaling factor of the simulation biases for MHS.

    Args:
        angles: Numpy array containing the viewing angles for which
            to calculate the bias correction.

    Return:
        The scaling factors for the bias correction for the given viewing
        angles.
    """
    if angles.shape[angles.ndim - 1] > 1:
        angles = angles[..., np.newaxis]
    scales = 1.0 / np.cos(np.deg2rad(angles))
    return scales / _BIAS_SCALES_GMI


###############################################################################
# Sensor classes
###############################################################################


class Sensor(ABC):
    """
    The sensor class defines the abstract properties and methods that
    each sensor class must implement.
    """
    def __init__(self,
                 name,
                 n_freqs):
        from gprof_nn.data import preprocessor
        self._name = name
        self._n_freqs = n_freqs
        self._preprocessor_orbit_header = types.get_preprocessor_orbit_header(
            n_freqs
        )
        self._preprocessor_pixel_record = types.get_preprocessor_pixel_record(
            n_freqs,
            types.CONICAL
        )

    @property
    def name(self):
        """
        The name of the sensor.
        """
        return self._name

    @property
    def sensor_id(self):
        """
        String that uniquely identifies the sensor.
        """
        return self._name

    @abstractproperty
    def n_inputs(self):
        """
        The number of input features for the GPORF-NN retrieval.
        """

    def n_freqs(self):
        """
        The number of frequencies or channels of the sensor.
        """
        return self._n_freqs

    @abstractproperty
    def bin_file_header(self):
        """
        Numpy dtype defining the binary structure of the header
        of binned, non-clustered database files.
        """

    @abstractproperty
    def get_bin_file_record(self, surface_type):
        """
        Args:
            surface_type: The surface type corresponding to the
                bin file.

        Return:
            Numpy dtype defining the binary record structure of binned,
            non-clustered database files.
        """

    @abstractproperty
    def l1c_file_prefix(self):
        """
        The prefix used for L1C files of this sensor.
        """

    @abstractproperty
    def l1c_file_path(self):
        """
        The default file path on the CSU system.
        """

    @abstractproperty
    def mrms_file_path(self):
        """
        The default path for MRMS match-up files on the CSU system.
        """

    @abstractproperty
    def mrms_file_record(self):
        """
        Numpy dtype defining the binary record structure for MRMS match
        up files.
        """

    @abstractproperty
    def sim_file_pattern(self):
        """
        Glob pattern to identify sim files.
        """

    @abstractproperty
    def sim_file_path(self):
        """
        The default sim-file path on the CSU system.
        """

    @abstractproperty
    def sim_file_header(self):
        """
        Numpy dtype defining the binary structure of the header
        of simulator files.
        """

    @abstractproperty
    def sim_file_record(self):
        """
        Numpy dtype defining the binary record structure of simulator
        files.
        """

    @property
    def preprocessor_orbit_header(self):
        """
        Numpy dtype defining the binary structure of the header of
        preprocessor files.
        """
        return self._preprocessor_orbit_header

    @property
    def preprocessor_pixel_record(self):
        """
        Numpy dtype defining the binary record structure of preprocessor
        files.
        """
        return self._preprocessor_pixel_record

    @abstractmethod
    def load_data_0d(self, filename):
        """
        Load input data for GPROF-NN 0D algorithm from NetCDF file.

        Args:
            filename: Path of the file from which to load the data.
            targets: List of target names to load.
            augment: Flag indicating whether or not to augment the training
                data.
            rng: Numpy random number generator to use for the data
                augmentation.

        Return:
            Rank-2 tensor ``x`` with input features oriented along its
            last dimension.
        """
        pass

    @abstractmethod
    def load_training_data_0d(self, filename, targets, augment, rng):
        """
        Load training data for GPROF-NN 0D algorithm from NetCDF file.

        Args:
            filename: Path of the file from which to load the data.
            targets: List of target names to load.
            augment: Flag indicating whether or not to augment the training
                data.
            rng: Numpy random number generator to use for the data
                augmentation.

        Return:
            Tuple ``(x, y)`` consisting of rank-2 tensor ``x`` with input
            features oriented along the last dimension and a dictionary
            ``y`` containing the values of the retrieval targets for all
            inputs in ``x``.
        """
        pass

    def load_data_2d(self, filename, targets):
        """
        Load input data for GPROF-NN 2D algorithm from NetCDF file.

        Args:
            filename: Path of the file from which to load the data.
            targets: List of target names to load.
            augment: Flag indicating whether or not to augment the training
                data.
            rng: Numpy random number generator to use for the data
                augmentation.

        Return:
            Rank-4 tensor ``x`` with input features oriented along
            the second dimension and along- and across-track
            dimensions along the third and fourth, respectively.
        """

    def load_training_data_2d(self, filename, targets):
        """
        Load training data for GPROF-NN 2D algorithm from NetCDF file.

        Args:
            filename: Path of the file from which to load the data.
            targets: List of target names to load.
            augment: Flag indicating whether or not to augment the training
                data.
            rng: Numpy random number generator to use for the data
                augmentation.

        Return:
            Tuple ``(x, y)`` containing the Rank-4 tensor ``x`` with input
            features oriented along the second dimensions and the along and
            across track dimensions along the third and fourth dimension,
            respectively.
        """

    def load_brightness_temperatures(self, data, angles=None, mask=None):
        pass

    def load_total_column_water_vapor(self, data, mask=None):
        """
        Load total column water vapor from the given dataset and replace
        invalid values.
        """
        return load_variable(data, "total_column_water_vapor", mask=mask)

    def load_two_meter_temperature(self, data, mask=None):
        """
        Load two-meter temperature from the given dataset and replace invalid
        values.
        """
        return load_variable(data, "two_meter_temperature", mask=mask)

    def load_viewing_angle(self, data, mask=None):
        """
        Load viewing angle from the given dataset and replace invalid values.
        """
        return load_variable(data, "load_viewing_angle", mask=mask)

    def load_surface_type(self, data, mask=None):
        """
        Load surface type from dataset and convert to 1-hot encoding.
        """
        st = data["surface_type"].data
        if mask is not None:
            st = st[mask]
        n_types = 18
        shape = st.shape + (n_types,)
        st_1h = np.zeros(shape, dtype=np.float32)
        for i in range(n_types):
            mask = st == i + 1
            st_1h[mask, i] = 1.0
        return st_1h

    def load_airmass_type(self, data, mask=None):
        """
        Load airmass type from dataset and convert to 1-hot encoding.
        """
        am = data["airmass_type"].data
        if mask is not None:
            am = am[mask]
        am = np.maximum(am, 0)
        n_types = 4
        shape = am.shape + (n_types,)
        am_1h = np.zeros(shape, dtype=np.float32)
        for i in range(n_types):
            mask = am == i
            am_1h[mask, i] = 1.0
        return am_1h

    def load_target(self, data, name, mask=None):
        """
        Load and, if necessary, expand target variable.
        """
        v = load_variable(data, name, mask=mask)
        return v


class ConicalScanner(Sensor):
    """
    Base class for conically-scanning sensors.
    """

    def __init__(
        self,
        name,
        n_freqs,
        l1c_prefix,
        l1c_file_path,
        mrms_file_path,
        sim_file_pattern,
        sim_file_path,
    ):
        super().__init__(name, n_freqs)
        self._n_freqs = n_freqs

        self._bin_file_header = np.dtype(
            [
                ("satellite_code", "a5"),
                ("sensor", "a5"),
                ("frequencies", f"{n_freqs}f4"),
                ("nominal_eia", f"{n_freqs}f4"),
            ]
        )
        self._bin_file_record = np.dtype(
            [
                ("dataset_number", "i4"),
                ("latitude", "f4"),
                ("longitude", "f4"),
                ("scan_time", "i4", (6,)),
                ("surface_precip", np.float32),
                ("convective_precip", np.float32),
                ("brightness_temperatures", "f4", (n_freqs,)),
                ("delta_tb", "f4", (n_freqs,)),
                ("rain_water_path", np.float32),
                ("cloud_water_path", np.float32),
                ("ice_water_path", np.float32),
                ("total_column_water_vapor", np.float32),
                ("two_meter_temperature", np.float32),
                ("rain_water_content", "f4", (N_LAYERS,)),
                ("cloud_water_content", "f4", (N_LAYERS,)),
                ("snow_water_content", "f4", (N_LAYERS,)),
                ("latent_heat", "f4", (N_LAYERS,)),
            ]
        )
        self._l1c_file_prefix = l1c_prefix
        self._l1c_file_path = l1c_file_path

        self._mrms_file_path = mrms_file_path
        self._mrms_file_record = np.dtype(
            [
                ("latitude", "f4"),
                ("longitude", "f4"),
                ("scan_time", f"5i4"),
                ("quality_flag", f"f4"),
                ("surface_precip", "f4"),
                ("surface_rain", "f4"),
                ("convective_rain", "f4"),
                ("stratiform_rain", "f4"),
                ("snow", "f4"),
                ("quality_index", "f4"),
                ("gauge_fraction", "f4"),
                ("standard_deviation", "f4"),
                ("n_stratiform", "i4"),
                ("n_convective", "i4"),
                ("n_rain", "i4"),
                ("n_snow", "i4"),
                ("fraction_missing", "f4"),
                ("brightness_temperatures", f"{n_freqs}f4"),
            ]
        )

        self._sim_file_pattern = sim_file_pattern
        self._sim_file_path = sim_file_path
        self._sim_file_header = np.dtype(
            [
                ("satellite_code", "a5"),
                ("sensor", "a5"),
                ("frequencies", f"{n_freqs}f4"),
                ("nominal_eia", f"{n_freqs}f4"),
                ("start_pixel", "i4"),
                ("end_pixel", "i4"),
                ("start_scan", "i4"),
                ("end_scan", "i4"),
            ]
        )
        self._sim_file_record = np.dtype(
            [
                ("pixel_index", "i4"),
                ("scan_index", "i4"),
                ("data_source", "f4"),
                ("latitude", "f4"),
                ("longitude", "f4"),
                ("elevation", "f4"),
                ("scan_time", DATE_TYPE),
                ("surface_type", "i4"),
                ("surface_precip", "f4"),
                ("convective_precip", "f4"),
                ("emissivity", f"{n_freqs}f4"),
                ("rain_water_content", f"{N_LAYERS}f4"),
                ("snow_water_content", f"{N_LAYERS}f4"),
                ("cloud_water_content", f"{N_LAYERS}f4"),
                ("latent_heat", f"{N_LAYERS}f4"),
                ("tbs_observed", f"{n_freqs}f4"),
                ("tbs_simulated", f"{n_freqs}f4"),
                ("d_tbs", f"{n_freqs}f4"),
                ("tbs_bias", f"{n_freqs}f4"),
            ]
        )


    @property
    def name(self):
        return self._name

    @property
    def n_inputs(self):
        """
        The number of input features for the GPORF-NN retrieval.
        """
        return self.n_freqs + 2 + 18 + 4

    @property
    def n_freqs(self):
        return self._n_freqs

    @property
    def bin_file_header(self):
        return self._bin_file_header

    def get_bin_file_record(self, surface_type):
        return self._bin_file_record

    @property
    def sim_file_header(self):
        return self._sim_file_header

    @property
    def sim_file_record(self):
        return self._sim_file_record

    @property
    def l1c_file_prefix(self):
        return self._l1c_file_prefix

    @property
    def l1c_file_path(self):
        return self._l1c_file_path

    @property
    def mrms_file_path(self):
        return self._mrms_file_path

    @property
    def mrms_file_record(self):
        return self._mrms_file_record

    @property
    def sim_file_pattern(self):
        return self._sim_file_pattern

    @property
    def sim_file_path(self):
        return self._sim_file_path

    def load_brightness_temperatures(self, data, angles=None, mask=None):
        return load_variable(data, "brightness_temperatures", mask=mask)

    def load_data_0d(self, filename):
        """
        Load all samples in training data as input for GPROF-NN 0D retrieval.

        Args:
            filename: The filename of the NetCDF file containing the training
                data.

        Return:
            Numpy array with two dimensions containing all samples in the give
            file as input for the GPROF-NN 0D retrieval.
        """
        x = []
        with xr.open_dataset(filename) as dataset:
            n_samples = dataset.samples.size
            for i in range(n_samples):
                scene = dataset[{"samples": i}]
                tbs = self.load_brightness_temperatures(scene)
                tbs = tbs.reshape(-1, self.n_freqs)
                t2m = self.load_two_meter_temperature(scene)
                t2m = t2m.reshape(-1, 1)
                tcwv = self.load_total_column_water_vapor(scene)
                mask = np.ones(tcwv.shape, dtype=np.bool)
                tcwv = tcwv.reshape(-1, 1)
                st = self.load_surface_type(scene, mask)
                am = self.load_airmass_type(scene, mask)
                x.append(np.concatenate([tbs, t2m, tcwv, st, am], axis=1))
        return np.concatenate(x, axis=0)

    def load_training_data_0d(self, filename, targets, augment, rng, equalizer=None):
        """
        Load training data for GPROF-NN 0D retrieval. This function will
        only load pixels that with a finite surface precip value in order
        to avoid training on samples that don't provide any information to
        the 0D retrieval.

        Output values that may be missing for a given pixel are masked using
        the 'MASKED_OUTPUT' value.

        Args:
            filename: The filename of the NetCDF file containing the training
                data.
            targets: List of the targets to load.
            augment: Whether or not to augment the training data.
            rng: Numpy random number generator to use for augmentation.
            equalizer: Not used.

        Return:
            Tuple ``(x, y)`` containing the un-batched, un-shuffled training
            data as it is contained in the given NetCDF file.
        """
        x = []
        y = {}

        with xr.open_dataset(filename) as dataset:
            n_samples = dataset.samples.size
            for i in range(n_samples):

                if "surface_precip" not in targets:
                    ts = targets + ["surface_precip"]
                else:
                    ts = targets
                scene = decompress_scene(dataset[{"samples": i}], ts)

                #
                # Input data
                #

                # Select only samples that have a finite surface precip value.
                sp = self.load_target(scene, "surface_precip")
                valid = sp >= 0

                tbs = self.load_brightness_temperatures(scene, mask=valid)
                if augment:
                    r = rng.random(tbs.shape[0])
                    tbs[r > 0.9, 10:15] = np.nan
                t2m = self.load_two_meter_temperature(scene, valid)[..., np.newaxis]
                tcwv = self.load_total_column_water_vapor(scene, valid)
                tcwv = tcwv[..., np.newaxis]
                st = self.load_surface_type(scene, valid)
                am = self.load_airmass_type(scene, valid)
                x.append(np.concatenate([tbs, t2m, tcwv, st, am], axis=1))

                #
                # Output data
                #

                for t in targets:
                    y_t = self.load_target(scene, t, valid)
                    y_t = np.nan_to_num(y_t, nan=MASKED_OUTPUT)
                    y.setdefault(t, []).append(y_t)
        x = np.concatenate(x)
        y = {k: np.concatenate(y[k]) for k in y}
        return x, y

    def load_data_2d(self, filename):
        """
        Load all scenes in training data as input for GPROF-NN 0D retrieval.

        Args:
            filename: The filename of the NetCDF file containing the training
                data.

        Return:
            Numpy array with with shape
            ``(n_samples, n_features, height, widht)`` containing all scenes
            from the training data file.
        """
        with xr.open_dataset(self.filename) as dataset:
            #
            # Input data
            #

            # Brightness temperatures
            n = dataset.samples.size

            x = []

            for i in range(n):
                scene = dataset[{"samples": i}]

                tbs = self.load_brightness_temperatures(scene)
                tbs = np.transpose(tbs, (2, 0, 1))
                t2m = self.load_two_meter_temperature(dataset)[np.newaxis]
                tcwv = self.load_two_meter_temperature(dataset)[np.newaxis]
                st = self.load_surface_type(dataset)
                st = np.transpose(st, (2, 0, 1))
                am = self.load_airmass_type(dataset)
                am = np.transpose(am, (2, 0, 1))
                x.append(np.concatenate([tbs, t2m, tcwv, st, am], axis=0))

            return np.stack(x)

    def load_training_data_2d(self, dataset, targets, augment, rng):
        """
        Load training data for GPROF-NN 2D retrieval. This function extracts
        scenes of 128 x 96 pixels from each training data sample and arranges
        the data to match the expected format of the GPROF-NN 2D retrieval.

        Args:
            filename: The filename of the NetCDF file containing the training
                data.
            targets: List of the targets to load.
            augment: Whether or not to augment the training data.
            rng: Numpy random number generator to use for augmentation.

        Return:
            Tuple ``(x, y)`` containing the un-batched, un-shuffled training
            data as it is contained in the given NetCDF file.
        """
        x = []
        y = {}

        with xr.open_dataset(dataset) as dataset:
            n = dataset.samples.size
            for i in range(n):
                scene = decompress_scene(dataset[{"samples": i}], targets)

                if augment:
                        p_x_o = rng.random()
                        p_x_i = rng.random()
                        p_y = rng.random()
                else:
                        p_x_o = 0.0
                        p_x_i = 0.0
                        p_y = 0.0

                coords = get_transformation_coordinates(
                        GMI_GEOMETRY, 96, 128, p_x_i, p_x_o, p_y
                   )

                scene = remap_scene(dataset[{"samples": i}], coords, targets)

                #
                # Input data
                #

                tbs = self.load_brightness_temperatures(scene)
                tbs = np.transpose(tbs, (2, 0, 1))
                if augment:
                    r = rng.random()
                    n_p = rng.integers(10, 30)
                    if r > 0.80:
                        tbs[10:15, :, :n_p] = np.nan
                t2m = self.load_two_meter_temperature(scene)[np.newaxis]
                tcwv = self.load_total_column_water_vapor(scene)[np.newaxis]
                st = self.load_surface_type(scene)
                st = np.transpose(st, (2, 0, 1))
                am = self.load_airmass_type(scene)
                am = np.transpose(am, (2, 0, 1))
                x.append(np.concatenate([tbs, t2m, tcwv, st, am], axis=0))

                #
                # Output data
                #

                for t in targets:
                    y_t = self.load_target(scene, t)
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


class CrossTrackScanner(Sensor):
    """
    Base class for cross-track-scanning sensors.
    """

    def __init__(
        self,
        name,
        angles,
        nedt,
        n_freqs,
        l1c_prefix,
        l1c_file_path,
        mrms_file_path,
        sim_file_pattern,
        sim_file_path,
    ):
        super().__init__(name, n_freqs)

        self._preprocessor_orbit_header = types.get_preprocessor_orbit_header(
            n_freqs
        )
        self._preprocessor_pixel_record = types.get_preprocessor_pixel_record(
            n_freqs,
            types.XTRACK
        )
        self._name = name
        self._angles = angles
        self.nedt = nedt
        n_angles = angles.size
        self._n_freqs = n_freqs
        self.kernels = calculate_smoothing_kernels(self,
                                                   MHS_GEOMETRY)

        self._bin_file_header = np.dtype(
            [
                ("satellite_code", "a5"),
                ("sensor", "a5"),
                ("frequencies", "f4", (n_freqs,)),
                ("nominal_eia", "f4", (n_angles,)),
            ]
        )
        self._l1c_file_prefix = l1c_prefix
        self._l1c_file_path = l1c_file_path

        self._mrms_file_path = mrms_file_path
        self._mrms_file_record = np.dtype(
            [
                ("datasetnum", "i4"),
                ("latitude", "f4"),
                ("longitude", "f4"),
                ("orbitnum", "i4"),
                ("n_pixels", "i4"),
                ("n_scans", "i4"),
                ("scan_time", f"5i4"),
                ("skin_temperature", f"i4"),
                ("total_column_water_vapor", f"i4"),
                ("surface_type", f"i4"),
                ("quality_flag", f"f4"),
                ("two_meter_temperature", "f4"),
                ("wet_bulb_temperature", "f4"),
                ("lapse_rate", "f4"),
                ("surface_precip", "f4"),
                ("surface_rain", "f4"),
                ("convective_rain", "f4"),
                ("stratiform_rain", "f4"),
                ("snow", "f4"),
                ("quality_index", "f4"),
                ("gauge_fraction", "f4"),
                ("standard_deviation", "f4"),
                ("n_stratiform", "i4"),
                ("n_convective", "i4"),
                ("n_rain", "i4"),
                ("n_snow", "i4"),
                ("fraction_missing", "f4"),
                ("brightness_temperatures", f"{n_freqs}f4"),
            ]
        )
        self._sim_file_pattern = sim_file_pattern
        self._sim_file_path = sim_file_path
        self._sim_file_header = np.dtype(
            [
                ("satellite_code", "a5"),
                ("sensor", "a5"),
                ("frequencies", f"{n_freqs}f4"),
                ("viewing_angles", f"{n_angles}f4"),
                ("start_pixel", "i4"),
                ("end_pixel", "i4"),
                ("start_scan", "i4"),
                ("end_scan", "i4"),
            ]
        )
        self._sim_file_record = np.dtype(
            [
                ("pixel_index", "i4"),
                ("scan_index", "i4"),
                ("latitude", "f4"),
                ("longitude", "f4"),
                ("elevation", "f4"),
                ("scan_time", DATE_TYPE),
                ("surface_type", "i4"),
                ("surface_precip", f"{n_angles}f4"),
                ("convective_precip", f"{n_angles}f4"),
                ("emissivity", f"{n_angles * n_freqs}f4"),
                ("rain_water_content", f"{N_LAYERS}f4"),
                ("snow_water_content", f"{N_LAYERS}f4"),
                ("cloud_water_content", f"{N_LAYERS}f4"),
                ("latent_heat", f"{N_LAYERS}f4"),
                ("tbs_simulated", f"{n_angles * n_freqs}f4"),
                ("tbs_bias", f"{n_freqs}f4"),
            ]
        )

    @property
    def n_inputs(self):
        """
        The number of input features for the GPORF-NN retrieval.
        """
        return self.n_freqs + 3 + 18 + 4

    @property
    def angles(self):
        return self._angles

    @property
    def n_angles(self):
        return self._angles.size

    @property
    def n_freqs(self):
        return self._n_freqs

    @property
    def bin_file_header(self):
        return self._bin_file_header

    def get_bin_file_record(self, surface_type):
        if surface_type in [2, 8, 9, 10, 11, 16]:
            return np.dtype(
                [
                    ("dataset_number", "i4"),
                    ("latitude", "f4"),
                    ("longitude", "f4"),
                    ("surface_precip", "f4"),
                    ("convective_precip", "f4"),
                    ("pixel_position", "i4"),
                    ("brightness_temperatures", "f4", (self.n_freqs,)),
                    ("rain_water_path", np.float32),
                    ("cloud_water_path", np.float32),
                    ("ice_water_path", np.float32),
                    ("total_column_water_vapor", np.float32),
                    ("two_meter_temperature", np.float32),
                    ("rain_water_content", "f4", (N_LAYERS,)),
                    ("cloud_water_content", "f4", (N_LAYERS,)),
                    ("snow_water_content", "f4", (N_LAYERS,)),
                    ("latent_heat", "f4", (N_LAYERS,)),
                ]
            )
        else:
            return np.dtype(
                [
                    ("dataset_number", "i4"),
                    ("latitude", "f4"),
                    ("longitude", "f4"),
                    ("surface_precip", "f4", (self.n_angles,)),
                    ("convective_precip", "f4", (self.n_angles,)),
                    ("brightness_temperatures", "f4", (self.n_angles, self.n_freqs)),
                    ("rain_water_path", np.float32),
                    ("cloud_water_path", np.float32),
                    ("ice_water_path", np.float32),
                    ("total_column_water_vapor", np.float32),
                    ("two_meter_temperature", np.float32),
                    ("rain_water_content", "f4", (N_LAYERS,)),
                    ("cloud_water_content", "f4", (N_LAYERS,)),
                    ("snow_water_content", "f4", (N_LAYERS,)),
                    ("latent_heat", "f4", (N_LAYERS,)),
                ]
            )

    @property
    def l1c_file_prefix(self):
        return self._l1c_file_prefix

    @property
    def l1c_file_path(self):
        return self._l1c_file_path

    @property
    def mrms_file_path(self):
        return self._mrms_file_path

    @property
    def mrms_file_record(self):
        return self._mrms_file_record

    @property
    def sim_file_pattern(self):
        return self._sim_file_pattern

    @property
    def sim_file_path(self):
        return self._sim_file_path

    @property
    def sim_file_header(self):
        return self._sim_file_header

    @property
    def sim_file_record(self):
        return self._sim_file_record

    @property
    def preprocessor_orbit_header(self):
        return self._preprocessor_orbit_header

    @property
    def preprocessor_pixel_record(self):
        return self._preprocessor_pixel_record

    def load_brightness_temperatures(self, data, weights, mask=None):
        """
        Load and bias-correct simulated brightness temperatures.

        Args:
            data: 'xarray.Dataset' containing from which to load the
                brightness temperatures.
            weights: Interpolation weight to interpolate the observations
                to specific angles.
            mask: A mask to subset the samples to load.

        Return:
            Array containing the interpolated and bias corrected simulated
            brightness temperatures.
        """
        if data.source == 0:
            tbs_sim = load_variable(
                data, "simulated_brightness_temperatures", mask=mask
            )

            bias = load_variable(data, "brightness_temperature_biases")
            bias = smooth_gmi_field(bias, self.kernels)
            if mask is not None:
                bias = bias[mask]

            tbs = interpolate(tbs_sim - bias, weights)
        else:
            tbs = load_variable(data, "brightness_temperatures", mask=mask)
        return tbs

    def load_earth_incidence_angle(self, data, mask=None):
        return load_variable(data, "earth_incidence_angle", mask=mask)

    def load_target(self, data, name, weights, mask=None):
        """
        Load and, if necessary, expand target variable.
        """
        v = load_variable(data, name, mask=mask)
        if name in ["surface_precip", "convective_precip"]:
            if data.source == 0:
                v = interpolate(v, weights)
            else:
                v = v[..., 0]
        return v

    def load_data_0d(self, filename):
        """
        Load all samples in training data as input for GPROF-NN 0D retrieval.

        Args:
            filename: The filename of the NetCDF file containing the training
                data.

        Return:
            Numpy array with two dimensions containing all samples in the give
            file as input for the GPROF-NN 0D retrieval.
        """
        with xr.open_dataset(filename) as dataset:
            n_samples = dataset.samples.size

            x = []

            for i in range(n_samples):
                vs = [
                    "simulated_brightness_temperatures",
                    "brightness_temperature_biases",
                    "earth_incidence_angle"
                ]
                scene = decompress_scene(dataset[{"samples": i}], vs)
                source = dataset.source[i]

                n_scans = scene.scans.size
                n_pixels = scene.pixels.size

                mask = np.ones((n_scans, n_pixels)).astype(np.bool)

                if source == 0:
                    eia = np.arange(n_scans * n_pixels)
                    eia = eia.reshape(n_scans, n_pixels)
                    eia = self.angles[eia % self.n_angles]
                    eia = eia[mask]
                    weights = calculate_interpolation_weights(eia, self.angles)
                    eia = eia[..., np.newaxis]
                else:
                    weights = None
                    eia = load_variable(scene, "earth_incidence_angle", mask=mask)
                    eia = eia[..., :1]

                tbs = self.load_brightness_temperatures(scene, weights, mask=mask)
                t2m = self.load_two_meter_temperature(scene, mask=mask)
                t2m = t2m[..., np.newaxis]
                tcwv = self.load_two_meter_temperature(scene, mask=mask)
                tcwv = tcwv[..., np.newaxis]
                st = self.load_surface_type(scene, mask=mask)
                am = self.load_airmass_type(scene, mask=mask)

                x.append(np.concatenate([tbs, eia, t2m, tcwv, st, am], axis=1))

        x = np.concatenate(x, axis=0)
        return x

    def load_training_data_0d(self, filename, targets, augment, rng, equalizer=None):
        """
        Load training data for GPROF-NN 0D retrieval. This function will
        only load pixels that with a finite surface precip value in order
        to avoid training on samples that don't provide any information to
        the 0D retrieval.

        Earth incidence angles are sampled uniformly from the range
        ``[-angle_max, angle_max]``, where ``angle_max`` is calculated by
        adding half the distance between largest and second-largest angle
        to the largest angle in the sensors ``angles`` attribute.

        Output values that may be missing for a given pixel are masked using
        the 'MASKED_OUTPUT' value.

        Args:
            filename: The filename of the NetCDF file containing the training
                data.
            targets: List of the targets to load.
            augment: Whether or not to augment the training data.
            rng: Numpy random number generator to use for augmentation.
            equalizer: Not used.

        Return:
            Tuple ``(x, y)`` containing the un-batched, un-shuffled training
            data as it is contained in the given NetCDF file.
        """
        angles_max = self.angles[0] + 0.5 * (self.angles[0] - self.angles[1])
        angles_min = -self.angles[0] + 0.5 * (self.angles[0] - self.angles[1])

        x = []
        y = {}

        with xr.open_dataset(filename) as dataset:

            n_samples = dataset.samples.size

            # Iterate over samples in training data (scenes of size 221 x 221)
            # and extract only pixels for which surface precip is defined.

            vs = [
                "simulated_brightness_temperatures",
                "brightness_temperature_biases",
                "earth_incidence_angle",
            ]
            if "surface_precip" not in targets:
                vs += ["surface_precip"]

            for i in range(n_samples):
                scene = decompress_scene(dataset[{"samples": i}], targets + vs)
                source = dataset.source[i]

                sp = scene["surface_precip"].data
                mask = np.all(sp >= 0, axis=-1)

                if source == 0:
                    eia = rng.uniform(angles_min, angles_max, size=mask.sum())
                    weights = calculate_interpolation_weights(eia, self.angles)
                else:
                    weights = None
                    tbs = scene["brightness_temperatures"].data
                    mask = mask * np.all((tbs > 0) * (tbs < 500))
                    eia = load_variable(scene, "earth_incidence_angle", mask=mask)[
                        ..., 0
                    ]

                #
                # Input data
                #

                tbs = self.load_brightness_temperatures(scene, weights, mask=mask)
                eia = eia[..., np.newaxis]
                t2m = self.load_two_meter_temperature(scene, mask=mask)
                t2m = t2m[..., np.newaxis]
                tcwv = self.load_total_column_water_vapor(scene, mask=mask)
                tcwv = tcwv[..., np.newaxis]
                st = self.load_surface_type(scene, mask=mask)
                am = self.load_airmass_type(scene, mask=mask)

                x.append(np.concatenate([tbs, eia, t2m, tcwv, st, am], axis=1))

                #
                # Output data
                #

                for t in targets:
                    y_t = self.load_target(scene, t, weights, mask=mask)
                    y_t = np.nan_to_num(y_t, nan=MASKED_OUTPUT)
                    y.setdefault(t, []).append(y_t)

        x = np.concatenate(x, axis=0)
        y = {t: np.concatenate(y[t], axis=0) for t in y}

        return x, y

    def load_data_2d(self, filename, targets, augment, rng):
        pass

    def _load_training_data_2d_sim(self, scene, targets, augment, rng):
        """
        Load training data for scene extracted from a sim file. Since
        these scenes are located on the GMI swath, they need to remapped
        to match

        Args:
            scene: 'xarray.Dataset' containing the training data sample
                to load.
            targets: List of the retrieval targets to load as output data.
            augment: Whether or not to augment the training data.
            rng: 'numpy.random.Generator' to use to generate random numbers.

        Returns:
            Tuple ``x, y`` containing one sample of training data for the
            GPROF-NN 2D retrieval.
        """
        if augment:
            p_x_o = rng.random()
            p_x_i = rng.random()
            p_y = rng.random()
        else:
            p_x_o = 0.0
            p_x_i = 0.0
            p_y = 0.0

        width = 32
        height = 128

        coords = get_transformation_coordinates(
            MHS_GEOMETRY, width, height, p_x_i, p_x_o, p_y
        )

        vs = ["simulated_brightness_temperatures", "brightness_temperature_biases"]
        scene = remap_scene(scene, coords, targets + vs)

        center = MHS_GEOMETRY.get_window_center(p_x_o, width, height)
        j_start = int(center[1, 0, 0] - width // 2)
        j_end = int(center[1, 0, 0] + width // 2)
        eia = MHS_GEOMETRY.get_earth_incidence_angles()
        eia = eia[j_start:j_end]
        weights = calculate_interpolation_weights(eia, self.angles)
        eia = np.repeat(eia.reshape(1, -1), height, axis=0)
        weights = np.repeat(weights.reshape(1, -1, self.n_angles), height, axis=0)

        tbs = self.load_brightness_temperatures(scene, weights)
        tbs = np.transpose(tbs, (2, 0, 1))
        eia = eia[np.newaxis]
        t2m = self.load_two_meter_temperature(scene)[np.newaxis]
        tcwv = self.load_total_column_water_vapor(scene)[np.newaxis]
        st = self.load_surface_type(scene)
        st = np.transpose(st, (2, 0, 1))
        am = self.load_airmass_type(scene)
        am = np.transpose(am, (2, 0, 1))
        x = np.concatenate([tbs, eia, t2m, tcwv, st, am], axis=0)

        #
        # Output data
        #

        y = {}

        for t in targets:
            y_t = self.load_target(scene, t, weights)
            y_t = np.nan_to_num(y_t, nan=MASKED_OUTPUT)

            dims_sp = tuple(range(2))
            dims_t = tuple(range(2, y_t.ndim))

            y[t] = np.transpose(y_t, dims_t + dims_sp)

        # Also flip data if requested.
        if augment:
            r = rng.random()
            if r > 0.5:
                x = np.flip(x, -2)
                for k in targets:
                    y[k] = np.flip(y[k], -2)

            r = rng.random()
            if r > 0.5:
                x = np.flip(x, -1)
                for k in targets:
                    y[k] = np.flip(y[k], -1)
        return x, y

    def _load_training_data_2d_other(self, scene, targets, augment, rng):
        """
        Load training data for sea ice or snow surfaces. These observations
        were extracted directly from L1C files and are on the original MHS
        grid.

        Args:
            scene: 'xarray.Dataset' containing the training data sample
                to load.
            targets: List of the retrieval targets to load as output data.
            augment: Whether or not to augment the training data.
            rng: 'numpy.random.Generator' to use to generate random numbers.

        Returns:
            Tuple ``x, y`` containing one sample of training data for the
            GPROF-NN 2D retrieval.
        """
        if augment:
            p_x = rng.random()
        else:
            p_x = 0.5

        c = MHS_GEOMETRY.get_window_center(p_x, 32, 128)
        i = c[0, 0, 0]
        j = c[0, 0, 0]

        width = 32
        height = 128

        i_start = int(i - height // 2)
        i_end = int(i + height // 2)
        j_start = int(j - width // 2)
        j_end = int(j + width // 2)

        tbs = self.load_brightness_temperatures(scene, None)
        tbs = tbs[i_start:i_end, j_start:j_end]
        tbs = np.transpose(tbs, (2, 0, 1))

        eia = self.load_earth_incidence_angle(scene)
        eia = eia[np.newaxis, i_start:i_end, j_start:j_end, 0]

        t2m = self.load_two_meter_temperature(scene)
        t2m = t2m[np.newaxis, i_start:i_end, j_start:j_end]

        tcwv = self.load_total_column_water_vapor(scene)
        tcwv = tcwv[np.newaxis, i_start:i_end, j_start:j_end]

        st = self.load_surface_type(scene)
        st = np.transpose(st[i_start:i_end, j_start:j_end], (2, 0, 1))
        am = self.load_airmass_type(scene)
        am = np.transpose(am[i_start:i_end, j_start:j_end], (2, 0, 1))

        x = np.concatenate([tbs, eia, t2m, tcwv, st, am], axis=0)

        #
        # Output data
        #
        y = {}

        for t in targets:
            y_t = self.load_target(scene, t, None)
            y_t = y_t[i_start:i_end, j_start:j_end]
            y_t = np.nan_to_num(y_t, nan=MASKED_OUTPUT)

            dims_sp = tuple(range(2))
            dims_t = tuple(range(2, y_t.ndim))

            y[t] = np.transpose(y_t, dims_t + dims_sp)

        # Also flip data if requested.
        if augment:
            r = rng.random()
            if r > 0.5:
                x = np.flip(x, -2)
                for k in targets:
                    y[k] = np.flip(y[k], -2)

            r = rng.random()
            if r > 0.5:
                x = np.flip(x, -1)
                for k in targets:
                    y[k] = np.flip(y[k], -1)

        return x, y

    def load_training_data_2d(self, dataset, targets, augment, rng):

        if isinstance(dataset, (str, Path)):
            dataset = xr.open_dataset(dataset)

        # Brightness temperatures
        n = dataset.samples.size

        x = []
        y = []

        vs = [
            "simulated_brightness_temperatures",
            "brightness_temperature_biases",
            "earth_incidence_angle",
            "source",
        ]
        if "surface_precip" not in targets:
            vs += ["surface_precip"]

        for i in range(n):
            scene = decompress_scene(dataset[{"samples": i}], targets + vs)
            source = scene.source
            if source == 0:
                x_i, y_i = self._load_training_data_2d_sim(scene, targets, augment, rng)
            else:
                x_i, y_i = self._load_training_data_2d_other(
                    scene, targets, augment, rng
                )
            x.append(x_i)
            y.append(y_i)

        x = np.stack(x)
        y = {k: np.stack([y_i[k] for y_i in y]) for k in y[0]}

        return x, y


GMI = ConicalScanner(
    "GMI",
    15,
    "1C-R.GPM.GMI",
    "/pdata4/archive/GPM/1CR_GMI",
    "/pdata4/veljko/GMI2MRMS_match2019/db_mrms4GMI/",
    "GMI.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7"
)

MHS_ANGLES = np.array(
    [59.798, 53.311, 46.095, 39.222, 32.562, 26.043, 19.619, 13.257, 6.934, 0.0]
)

MHS_NEDT = np.array([1.0, 1.0, 4.0, 2.0, 2.0])


MHS = CrossTrackScanner(
    "MHS",
    MHS_ANGLES,
    MHS_NEDT,
    5,
    "1C.*.MHS.",
    "/pdata4/archive/GPM/1C_NOAA19",
    "/pdata4/veljko/MHS2MRMS_match2019/monthly_2021/",
    "MHS.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7x",
)
