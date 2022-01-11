"""
================
gprof_nn.sensors
================

This module defines different sensor classes to represent the
 various sensors of the GPM constellation.

This module provides sensor objects for all satellites of the
GPM constellation. Most sensors exist in two or three variants:

  1. A generic variant that is named directly after the sensor
     ('GMI', 'MHS', ...) that can be used whenever it is
     irrelevant to which specific sensor instance is referred,
     i.e. for reading simulator or bin files.
  2. A specific variant named using the naming scheme
     {sensor}_{platform}. The should be used when specific
     instance of the sensor plays a role, e.g. training of
     a retrieval model.
  3. A specific variant named using the naming scheme
     {sensor}_{platform}_C. This sensor also applies a
     quantile-matching correction to the training data.
"""
from abc import ABC, abstractmethod, abstractproperty
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import convolve
import xarray as xr

from gprof_nn.data import types
from gprof_nn.definitions import N_LAYERS, LIMITS
from gprof_nn.data.utils import load_variable, decompress_scene, remap_scene
from gprof_nn.data.cdf import CdfCorrection
from gprof_nn.utils import (
    apply_limits,
    get_mask,
    calculate_interpolation_weights,
    interpolate,
)
from gprof_nn.data.utils import expand_pixels
from gprof_nn.augmentation import (
    Conical,
    CrossTrack,
    get_transformation_coordinates,
    extract_domain,
    SCANS_PER_SAMPLE,
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


DATA_FOLDER = Path(__file__).parent / "files"


###############################################################################
# Helper functions
###############################################################################


_BIAS_SCALES_GMI = 1.0 / np.cos(np.deg2rad([52.8, 49.19, 49.19, 49.19, 49.19]))


def calculate_smoothing_kernel(
    res_x_source, res_a_source, res_x_target, res_a_target, size
):
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


def calculate_smoothing_kernels(sensor):
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
    geometry = sensor.viewing_geometry
    res_x_source = 14.4e3
    res_a_source = 8.6e3
    angles = sensor.angles
    res_x_target = geometry.get_resolution_x(angles)
    res_a_target = geometry.get_resolution_a()

    kernels = []
    for res in res_x_target:
        k = calculate_smoothing_kernel(
            res_x_source, res_a_source, res, res_a_target, size=11
        )
        kernels.append(k)
    return kernels


def smooth_gmi_field(field, kernels):
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

    def __init__(self, kind, name, channels, angles, platform, viewing_geometry):
        self.kind = kind
        self._name = name
        self._channels = channels
        self._angles = angles
        n_chans = len(channels)
        n_angles = len(angles)
        self._n_chans = n_chans
        self._n_angles = n_angles
        self.platform = platform
        self.viewing_geometry = viewing_geometry

        # Bin file types
        self._bin_file_header = types.get_bin_file_header(n_chans, n_angles, kind)

        # Preprocessor types
        self._preprocessor_orbit_header = types.get_preprocessor_orbit_header(
            n_chans, self.kind
        )
        self._preprocessor_pixel_record = types.get_preprocessor_pixel_record(
            n_chans, self.kind
        )

        # Sim file types
        self._sim_file_header = types.get_sim_file_header(n_chans, n_angles, kind)
        self._sim_file_record = types.get_sim_file_record(
            n_chans, n_angles, N_LAYERS, kind
        )

        # MRMS types
        self._mrms_file_record = types.get_mrms_file_record(n_chans, n_angles, kind)

    def __eq__(self, other):
        return self.name == other.name and self.platform == other.platform

    @property
    def name(self):
        """
        The name of the sensor.
        """
        return self._name

    @property
    def sensor_name(self):
        """
        The name of the sensor only.
        """
        return self._name

    @property
    def sensor_id(self):
        """
        The name of the sensor only.
        """
        return self._name

    @property
    def full_name(self):
        """
        A combination of sensor and platform name.
        """
        platform = self.platform.name.upper().replace("-", "")
        sensor = self.name.upper()
        return f"{sensor}_{platform}"

    @property
    def channels(self):
        """
        List containing channel frequencies and polarizations.
        """
        return self._n_chans

    @property
    def n_chans(self):
        """
        The number of channels of the sensor
        """
        return self._n_chans

    @property
    def angles(self):
        """
        The list of earth incidence angles that are simulated for this
        sensor.
        """
        return self._angles

    @property
    def n_angles(self):
        """
        The list of earth incidence angles that are simulated for this
        sensor.
        """
        return self._n_angles

    @property
    def angle_bins(self):
        """
        Angle bin boundaries to map the viewing angles of the real observations
        to the discrete angles of the simulations.
        """
        angle_bins = np.zeros(self.angles.size + 1)
        angle_bins[1:-1] = 0.5 * (self.angles[1:] + self.angles[:-1])
        angle_bins[0] = 2.0 * angle_bins[1] - angle_bins[2]
        angle_bins[-1] = 2.0 * angle_bins[-2] - angle_bins[-3]
        return angle_bins

    @property
    def n_inputs(self):
        """
        The number of frequencies or channels of the sensor.
        """
        n_inputs = self.n_freqs + 24
        if self.kind == types.XTRACK:
            n_inputs += 1
        return n_inputs

    @property
    def bin_file_header(self):
        """
        Numpy dtype defining the binary structure of the header
        of binned, non-clustered database files.
        """
        return self._bin_file_header

    def get_bin_file_record(self, surface_type):
        """
        Args:
            surface_type: The surface type corresponding to the
                bin file.

        Return:
            Numpy dtype defining the binary record structure of binned,
            non-clustered database files.
        """
        return types.get_bin_file_record(
            self.n_chans, self.n_angles, N_LAYERS, surface_type, self.kind
        )

    @property
    def l1c_file_prefix(self):
        """
        The prefix used for L1C files of this sensor.
        """
        return self.platform.l1c_file_prefix

    @property
    def l1c_file_path(self):
        """
        The default file path on the CSU system.
        """
        return self.platform.l1c_file_path

    @abstractproperty
    def mrms_file_path(self):
        """
        The default path for MRMS match-up files on the CSU system.
        """

    @property
    def mrms_file_record(self):
        """
        Numpy dtype defining the binary record structure for MRMS match
        up files.
        """
        return self._mrms_file_record

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

    @property
    def sim_file_header(self):
        """
        Numpy dtype defining the binary structure of the header
        of simulator files.
        """
        return self._sim_file_header

    @property
    def sim_file_record(self):
        """
        Numpy dtype defining the binary record structure of simulator
        files.
        """
        return self._sim_file_record

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
    def load_training_data_1d(self, dataset, targets, augment, rng):
        """
        Load training data for GPROF-NN 1D algorithm from NetCDF file.

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

    def load_training_data_3d(self, filename, targets):
        """
        Load training data for GPROF-NN 3D algorithm from NetCDF file.

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
        channels,
        angles,
        platform,
        viewing_geometry,
        mrms_file_path,
        sim_file_pattern,
        sim_file_path,
    ):
        super().__init__(
            types.CONICAL, name, channels, angles, platform, viewing_geometry
        )
        self._n_angles = 1
        n_chans = len(channels)

        self._mrms_file_path = mrms_file_path
        self._sim_file_pattern = sim_file_pattern
        self._sim_file_path = sim_file_path

    def __repr__(self):
        return f"ConicalScanner(name={self.name}, platform={self.platform.name})"

    @property
    def name(self):
        return self._name

    @property
    def n_inputs(self):
        """
        The number of input features for the GPORF-NN retrieval.
        """
        return self.n_chans + 2 + 18 + 4

    @property
    def mrms_file_path(self):
        return self._mrms_file_path

    @property
    def sim_file_pattern(self):
        return self._sim_file_pattern

    @property
    def sim_file_path(self):
        return self._sim_file_path

    def load_brightness_temperatures(self, data, angles=None, mask=None):
        return load_variable(data, "brightness_temperatures", mask=mask)

    def _load_scene_1d(self, scene, targets, augment, rng):
        """
        Helper function to parallelize loading of 1D training data.
        """
        pass
        if "surface_precip" not in targets:
            ts = targets + ["surface_precip"]
        else:
            ts = targets
        scene = decompress_scene(scene, ts)

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
        x = np.concatenate([tbs, t2m, tcwv, st, am], axis=1)

        #
        # Output data
        #

        y = {}
        for t in targets:
            y_t = self.load_target(scene, t, valid)
            y_t = np.nan_to_num(y_t, nan=MASKED_OUTPUT)
            y[t] = y_t
        return x, y

    def load_training_data_1d(self, dataset, targets, augment, rng, n_workers=1):
        """
        Load training data for GPROF-NN 1D retrieval. This function will
        only load pixels that with a finite surface precip value in order
        to avoid training on samples that don't provide any information to
        the 1D retrieval.

        Output values that may be missing for a given pixel are masked using
        the 'MASKED_OUTPUT' value.

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

        if isinstance(dataset, (str, Path)):
            dataset = xr.open_dataset(dataset)
            loaded = True
        else:
            loaded = False

        n_scenes = dataset.samples.size

        # Multi-process loading.
        if n_workers > 1:
            pool = ProcessPoolExecutor(max_workers=n_workers)
            # Distribute tasks to workers
            tasks = []
            for i in range(n_scenes):
                scene = dataset[{"samples": i}]
                tasks.append(pool.submit(
                    ConicalScanner._load_scene_1d,
                    self, scene, targets,
                    augment,  rng
                ))

            # Collect results
            for task in tasks:
                x_i, y_i = task.result()
                x.append(x_i)
                for target in targets:
                    y.setdefault(target, []).append(y_i[target])
        # Single-process loading.
        else:
            for i in range(n_scenes):
                scene = dataset[{"samples": i}]
                x_i, y_i = self._load_scene_1d(
                    scene,
                    targets,
                    augment,
                    rng)
                x.append(x_i)
                for target in targets:
                    y.setdefault(target, []).append(y_i[target])

        x = np.concatenate(x, axis=0)
        for k in targets:
            y[k] = np.concatenate(y[k], axis=0)

        if loaded:
            dataset.close()

        return x, y

    def _load_scene_3d(self, scene, targets, augment, variables, rng, width, height):
        """
        Helper function for parallelized loading of training samples.
        """
        scene = decompress_scene(scene, targets + variables)

        if augment:
            p_x_o = rng.random()
            p_x_i = rng.random()
            p_y = rng.random()
        else:
            p_x_o = 0.5
            p_x_i = 0.5
            p_y = 0.5

        lats = scene.latitude.data
        lons = scene.longitude.data
        coords = get_transformation_coordinates(
            lats, lons, self.viewing_geometry, width, height, p_x_i, p_x_o, p_y
        )

        scene = remap_scene(scene, coords, targets)

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

        x = np.concatenate([tbs, t2m, tcwv, st, am], axis=0)

        #
        # Output data
        #
        y = {}

        for t in targets:
            y_t = self.load_target(scene, t)
            y_t = np.nan_to_num(y_t, nan=MASKED_OUTPUT)

            dims_sp = tuple(range(2))
            dims_t = tuple(range(2, y_t.ndim))

            y[t] = np.transpose(y_t, dims_t + dims_sp).astype(np.float32)

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

    def load_training_data_3d(
            self, dataset, targets, augment, rng, width=96, height=128, n_workers=1
    ):
        """
        Load training data for GPROF-NN 3D retrieval. This function extracts
        scenes of 128 x 96 pixels from each training data sample and arranges
        the data to match the expected format of the GPROF-NN 3D retrieval.

        Args:
            filename: The filename of the NetCDF file containing the training
                data.
            targets: List of the targets to load.
            augment: Whether or not to augment the training data.
            rng: Numpy random number generator to use for augmentation.
            width: The width of each input image.
            height: The height of each input image.
            n_workers: If larger than 1, a process pool with that many workers
                 will be used to load the data in parallel.

        Return:
            Tuple ``(x, y)`` containing the un-batched, un-shuffled training
            data as it is contained in the given NetCDF file.
        """
        x = []
        y = {}

        vs = ["latitude", "longitude"]

        if isinstance(dataset, (str, Path)):
            dataset = xr.open_dataset(dataset)
            loaded = True
        else:
            loaded = False
        n_scenes = dataset.samples.size

        # Multi-process loading.
        if n_workers > 1:
            pool = ProcessPoolExecutor(max_workers=n_workers)
            # Distribute tasks to workers
            tasks = []
            for i in range(n_scenes):
                scene = dataset[{"samples": i}]
                tasks.append(pool.submit(
                    ConicalScanner._load_scene_3d,
                    self, scene, targets,
                    augment, vs, rng,
                    width, height
                ))

            # Collect results
            for task in tasks:
                x_i, y_i = task.result()
                x.append(x_i)
                for target in targets:
                    y.setdefault(target, []).append(y_i[target])
        # Single-process loading.
        else:
            for i in range(n_scenes):
                scene = dataset[{"samples": i}]
                if scene.source == 0:
                    n = 1
                else:
                    n = 2
                for j in range(n):
                    x_i, y_i = self._load_scene_3d(
                        scene,
                        targets,
                        augment,
                        vs,
                        rng,
                        width,
                        height)
                    x.append(x_i)
                    for target in targets:
                        y.setdefault(target, []).append(y_i[target])

        x = np.stack(x)
        for k in targets:
            y[k] = np.stack(y[k])

        if loaded:
            dataset.close()

        return x, y


class ConstellationScanner(ConicalScanner):
    """
    This class represents conically-scanning sensors that are for which
    observations can only be simulated.
    """
    def __init__(
        self,
        name,
        channels,
        nedt,
        angles,
        platform,
        viewing_geometry,
        mrms_file_path,
        sim_file_pattern,
        sim_file_path,
        gmi_channels,
        correction=None
    ):
        super().__init__(
            name, channels, angles, platform, viewing_geometry,
            mrms_file_path, sim_file_pattern, sim_file_path
        )
        self.nedt = nedt
        self.gmi_channels=gmi_channels
        self.correction = correction
        self.apply_biases = True

    def load_brightness_temperatures(self, data, mask=None):
        """
        Load and bias-correct simulated brightness temperatures.

        Args:
            data: 'xarray.Dataset' containing from which to load the
                brightness temperatures.
            mask: A mask to subset the samples to load.

        Return:
            Array containing the interpolated and bias corrected simulated
            brightness temperatures.
        """
        if data.source == 0:
            tbs_sim = load_variable(
                data, "simulated_brightness_temperatures", mask=mask
            )

            if self.apply_biases:
                bias = load_variable(data, "brightness_temperature_biases")
                # Apply scaling of biases

            if mask is not None:
                bias = bias[mask]
            tbs = tbs_sim - bias

        else:
            tbs = load_variable(data, "brightness_temperatures", mask=mask)
        return tbs

    def load_training_data_1d(self, dataset, targets, augment, rng):
        """
        Load training data for GPROF-NN 1D retrieval. This function will
        only load pixels that with a finite surface precip value in order
        to avoid training on samples that don't provide any information to
        the 1D retrieval.

        Output values that may be missing for a given pixel are masked using
        the 'MASKED_OUTPUT' value.

        Args:
            dataset:: 'xarray.Dattaset' containing the raw training data.
            targets: List of the targets to load.
            augment: Whether or not to augment the training data.
            rng: Numpy random number generator to use for augmentation.

        Return:
            Tuple ``(x, y)`` containing the un-batched, un-shuffled training
            data as it is contained in the given NetCDF file.
        """
        x = []
        y = {}

        if isinstance(dataset, (str, Path)):
            dataset = xr.open_dataset(dataset)
            loaded = True
        else:
            loaded = False

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
            mask = sp >= 0

            if source == 0:
                tbs = scene["simulated_brightness_temperatures"].data
                mask_tbs = get_mask(
                    tbs, *LIMITS["simulated_brightness_temperatures"]
                )
                biases = scene["brightness_temperature_biases"].data
                mask_biases = get_mask(
                    biases, *LIMITS["brightness_temperature_biases"]
                )
                mask = (
                    mask
                    * np.all(mask_tbs, axis=-1)
                    * np.all(mask_biases, axis=-1)
                )
            else:
                tbs = scene["brightness_temperatures"].data
                mask = mask * np.all((tbs > 0) * (tbs < 500), axis=-1)

            #
            # Input data
            #

            tbs = self.load_brightness_temperatures(scene, mask=mask)
            t2m = self.load_two_meter_temperature(scene, mask=mask)
            tcwv = self.load_total_column_water_vapor(scene, mask=mask)
            st = scene.surface_type.data[mask]

            if self.correction is not None:
                tbs = self.correction(self, st, None, tcwv, tbs, augment=augment)

            if augment and self.nedt is not None:
                noise = rng.normal(size=tbs.shape)
                for i in range(noise.shape[-1]):
                    noise[..., i] *= self.nedt[i]
                tbs = tbs + noise

            st = self.load_surface_type(scene, mask=mask)
            t2m = t2m[..., np.newaxis]
            tcwv = tcwv[..., np.newaxis]
            am = self.load_airmass_type(scene, mask=mask)

            x.append(np.concatenate([tbs, t2m, tcwv, st, am], axis=1))

            #
            # Output data
            #

            for t in targets:
                y_t = self.load_target(scene, t, mask=mask)
                y_t = np.nan_to_num(y_t, nan=MASKED_OUTPUT)
                y.setdefault(t, []).append(y_t)

        x = np.concatenate(x, axis=0)
        y = {t: np.concatenate(y[t], axis=0) for t in y}

        if loaded:
            dataset.close()

        return x, y

    def _load_training_data_3d_sim(
        self, scene, targets, augment, rng, width=32, height=128
    ):
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
            width: The width of each input image.
            height: The height of each input image.

        Returns:
            Tuple ``x, y`` containing one sample of training data for the
            GPROF-NN 3D retrieval.
        """
        if augment:
            p_x_i = rng.random()
            p_y = rng.random()
        else:
            p_x_i = 0.5
            p_y = 0.5
            p_x_o = 0.5

        lats = scene.latitude.data
        lons = scene.longitude.data

        alt = self.viewing_geometry.altitude
        if augment:
            alt_new = rng.uniform(0.9, 1.1) * alt
        else:
            alt_new = alt
        self.viewing_geometry.altitude = alt_new
        coords = get_transformation_coordinates(
            lats, lons, self.viewing_geometry, width, height, p_x_i, p_x_o, p_y
        )
        self.viewing_geometry.altitude = alt

        vs = ["simulated_brightness_temperatures", "brightness_temperature_biases"]
        scene = remap_scene(scene, coords, targets + vs)

        tbs = self.load_brightness_temperatures(scene)
        t2m = self.load_two_meter_temperature(scene)
        tcwv = self.load_total_column_water_vapor(scene)
        st = scene.surface_type.data

        if self.correction is not None:
            tbs = self.correction(self, st, eia, tcwv, tbs, augment=augment)

        if augment:
            if self.nedt is not None:
                noise = rng.normal(size=tbs.shape)
                for i in range(noise.shape[-1]):
                    noise[..., i] *= self.nedt[i]
                tbs += noise

            # Apply modeling error caused by simulator.
            if self.modeling_error is not None:
                noise = rng.normal(size=self.n_chans)
                for i, n in enumerate(noise):
                    tbs[..., i] += self.modeling_error[i] * n

            r = rng.random()
            if r > 0.80:
                n_p = rng.integers(6, 16)
                tbs[:, :n_p] = np.nan
            r = rng.random()
            if r > 0.80:
                n_p = rng.integers(6, 16)
                tbs[:, -n_p:] = np.nan

        tcwv = tcwv[np.newaxis]
        t2m = t2m[np.newaxis]
        st = self.load_surface_type(scene)
        am = self.load_airmass_type(scene)

        tbs = np.transpose(tbs, (2, 0, 1))
        st = np.transpose(st, (2, 0, 1))
        am = np.transpose(am, (2, 0, 1))

        x = np.concatenate([tbs, t2m, tcwv, st, am], axis=0)

        #
        # Output data
        #

        y = {}

        for t in targets:
            y_t = self.load_target(scene, t)
            y_t = np.nan_to_num(y_t, nan=MASKED_OUTPUT)

            dims_sp = tuple(range(2))
            dims_t = tuple(range(2, y_t.ndim))

            y[t] = np.transpose(y_t, dims_t + dims_sp)

        # Also flip data if requested.
        if augment:
            r = rng.random()
            if r > 0.5:
                x = np.flip(x, -2)
                x[self.n_chans] *= -1.0
                for k in targets:
                    y[k] = np.flip(y[k], -2)

            r = rng.random()
            if r > 0.5:
                x = np.flip(x, -1)
                for k in targets:
                    y[k] = np.flip(y[k], -1)
        return x, y

    def _load_training_data_3d_other(
        self, scene, targets, augment, rng, width=32, height=128
    ):
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
            width: The width of each input image.
            height: The height of each input image.

        Returns:
            Tuple ``x, y`` containing one sample of training data for the
            GPROF-NN 3D retrieval.
        """
        if augment:
            p_x = rng.random()
            p_y = rng.random()
        else:
            p_x = 0.5
            p_y = 0.5

        n_scans = SCANS_PER_SAMPLE
        n_pixels = self.viewing_geometry.pixels_per_scan

        i = height // 2 + (n_scans - height) * p_y
        j = (SCANS_PER_SAMPLE - n_pixels + width) // 2 + (n_pixels - width) * p_x

        i_start = int(i - height // 2)
        i_end = int(i + height // 2)
        j_start = int(j - width // 2)
        j_end = int(j + width // 2)

        tbs = self.load_brightness_temperatures(scene, None)
        tbs = tbs[i_start:i_end, j_start:j_end]
        tbs = np.transpose(tbs, (2, 0, 1))

        r = rng.random()
        if r > 0.80:
            n_p = rng.integers(6, 16)
            tbs[:, :n_p] = np.nan
        r = rng.random()
        if r > 0.80:
            n_p = rng.integers(6, 16)
            tbs[:, -n_p:] = np.nan

        t2m = self.load_two_meter_temperature(scene)
        t2m = t2m[np.newaxis, i_start:i_end, j_start:j_end]

        tcwv = self.load_total_column_water_vapor(scene)
        tcwv = tcwv[np.newaxis, i_start:i_end, j_start:j_end]

        st = self.load_surface_type(scene)
        st = np.transpose(st[i_start:i_end, j_start:j_end], (2, 0, 1))
        am = self.load_airmass_type(scene)
        am = np.transpose(am[i_start:i_end, j_start:j_end], (2, 0, 1))

        x = np.concatenate([tbs, t2m, tcwv, st, am], axis=0)

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

    def load_training_data_3d(
        self, dataset, targets, augment, rng, width=32, height=128
    ):
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
            "latitude",
            "longitude",
        ]
        if "surface_precip" not in targets:
            vs += ["surface_precip"]

        for i in range(n):
            scene = decompress_scene(dataset[{"samples": i}], targets + vs)
            source = scene.source
            if source == 0:
                x_i, y_i = self._load_training_data_3d_sim(
                    scene, targets, augment, rng, width=width, height=height
                )
            else:
                x_i, y_i = self._load_training_data_3d_other(
                    scene, targets, augment, rng, width=width, height=height
                )
            x.append(x_i)
            y.append(y_i)

        x = np.stack(x)
        y = {k: np.stack([y_i[k] for y_i in y]) for k in y[0]}

        return x, y


class CrossTrackScanner(Sensor):
    """
    Base class for cross-track-scanning sensors.
    """
    def __init__(
            self,
            name,
            channels,
            nedt,
            angles,
            platform,
            viewing_geometry,
            mrms_file_path,
            sim_file_pattern,
            sim_file_path,
            gmi_channels,
            correction=None,
            modeling_error=None
    ):
        super().__init__(
            types.XTRACK, "MHS", channels, angles, platform, viewing_geometry
        )
        self.nedt = nedt
        n_chans = len(channels)
        n_angles = len(angles)
        self.kernels = calculate_smoothing_kernels(self)

        self.gmi_channels = np.array(gmi_channels)
        gmi_angles = GMI.angles[self.gmi_channels]
        self.bias_scales = (np.cos(np.deg2rad(gmi_angles).reshape(1, -1)) /
                            np.cos(np.deg2rad(self.angles).reshape(-1, 1)))
        self.apply_biases = True
        self.bias_scaling = True

        self._mrms_file_path = mrms_file_path
        self._sim_file_pattern = sim_file_pattern
        self._sim_file_path = sim_file_path

        if correction is None:
            self._correction = None
        else:
            self._correction = correction
        # Convert modeling error to list of channel-wise modeling errors.
        if isinstance(modeling_error, int):
            modeling_error = [modelling_error] * n_chans
        self.modeling_error = modeling_error

    def __repr__(self):
        return f"CrossTrackScanner(name={self.name}, " f"platform={self.platform.name})"

    @property
    def correction(self):
        if self._correction is not None:
            if not isinstance(self._correction, CdfCorrection):
                self._correction = CdfCorrection(self._correction)
        return self._correction

    @property
    def n_inputs(self):
        """
        The number of input features for the GPORF-NN retrieval.
        """
        return self.n_chans + 3 + 18 + 4

    @property
    def mrms_file_path(self):
        return self._mrms_file_path

    @property
    def sim_file_pattern(self):
        return self._sim_file_pattern

    @property
    def sim_file_path(self):
        return self._sim_file_path

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

            if self.apply_biases:
                bias = load_variable(data, "brightness_temperature_biases")
                bias = smooth_gmi_field(bias, self.kernels)
                # Apply scaling of biases
                if self.bias_scaling:
                    shape = [1] * (bias.ndim - 2) + [10, 5]
                    bias = bias * self.bias_scales.reshape(shape)

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

    def load_training_data_1d(self, dataset, targets, augment, rng):
        """
        Load training data for GPROF-NN 1D retrieval. This function will
        only load pixels that with a finite surface precip value in order
        to avoid training on samples that don't provide any information to
        the 1D retrieval.

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

        Return:
            Tuple ``(x, y)`` containing the un-batched, un-shuffled training
            data as it is contained in the given NetCDF file.
        """
        angles_max = self.angles[0] + 0.5 * (self.angles[0] - self.angles[1])
        angles_min = -angles_max

        x = []
        y = {}

        if isinstance(dataset, (str, Path)):
            dataset = xr.open_dataset(dataset)
            loaded = True
        else:
            loaded = False

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
                tbs = scene["simulated_brightness_temperatures"].data
                mask_tbs = get_mask(
                    tbs, *LIMITS["simulated_brightness_temperatures"]
                )
                biases = scene["brightness_temperature_biases"].data
                mask_biases = get_mask(
                    biases, *LIMITS["brightness_temperature_biases"]
                )
                mask = (
                    mask
                    * np.all(mask_tbs, axis=(-2, -1))
                    * np.all(mask_biases, axis=-1)
                )
                if augment:
                    eia = rng.uniform(angles_min, angles_max, size=mask.sum())
                else:
                    angles = self.viewing_geometry.get_earth_incidence_angles()
                    indices = np.arange(mask.sum()) % angles.size
                    eia = angles[indices]

                weights = calculate_interpolation_weights(np.abs(eia), self.angles)
            else:
                weights = None
                tbs = scene["brightness_temperatures"].data
                mask = mask * np.all((tbs > 0) * (tbs < 500), axis=-1)
                eia = load_variable(scene, "earth_incidence_angle", mask=mask)[
                    ..., 0
                ]

            #
            # Input data
            #

            tbs = self.load_brightness_temperatures(scene, weights, mask=mask)
            t2m = self.load_two_meter_temperature(scene, mask=mask)
            tcwv = self.load_total_column_water_vapor(scene, mask=mask)
            st = scene.surface_type.data[mask]

            if self.correction is not None:
                tbs = self.correction(self, st, eia, tcwv, tbs, augment=augment)

            if augment and self.nedt is not None:
                noise = rng.normal(size=tbs.shape)
                for i in range(noise.shape[-1]):
                    noise[..., i] *= self.nedt[i]
                tbs = tbs + noise

            st = self.load_surface_type(scene, mask=mask)
            eia = eia[..., np.newaxis]
            t2m = t2m[..., np.newaxis]
            tcwv = tcwv[..., np.newaxis]
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

        if loaded:
            dataset.close()

        return x, y

    def _load_training_data_3d_sim(
        self, scene, targets, augment, rng, width=32, height=128
    ):
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
            width: The width of each input image.
            height: The height of each input image.

        Returns:
            Tuple ``x, y`` containing one sample of training data for the
            GPROF-NN 3D retrieval.
        """
        if augment:
            p_x_i = rng.random()
            p_y = rng.random()
        else:
            p_x_i = 0.5
            p_y = 0.5

        p_x_o = rng.random()

        lats = scene.latitude.data
        lons = scene.longitude.data

        alt = self.viewing_geometry.altitude
        if augment:
            alt_new = rng.uniform(0.9, 1.1) * alt
        else:
            alt_new = alt
        self.viewing_geometry.altitude = alt_new
        coords = get_transformation_coordinates(
            lats, lons, self.viewing_geometry, width, height, p_x_i, p_x_o, p_y
        )
        self.viewing_geometry.altitude = alt

        vs = ["simulated_brightness_temperatures", "brightness_temperature_biases"]
        scene = remap_scene(scene, coords, targets + vs)

        center = self.viewing_geometry.get_window_center(p_x_o, width, height)
        j_start = int(center[1, 0, 0] - width // 2)
        j_end = int(center[1, 0, 0] + width // 2)
        eia = self.viewing_geometry.get_earth_incidence_angles()
        eia = eia[j_start:j_end]
        weights = calculate_interpolation_weights(np.abs(eia), self.angles)
        eia = np.repeat(eia.reshape(1, -1), height, axis=0)
        weights = np.repeat(weights.reshape(1, -1, self.n_angles), height, axis=0)

        tbs = self.load_brightness_temperatures(scene, weights)
        t2m = self.load_two_meter_temperature(scene)
        tcwv = self.load_total_column_water_vapor(scene)
        st = scene.surface_type.data

        if self.correction is not None:
            tbs = self.correction(self, st, eia, tcwv, tbs, augment=augment)

        if augment:
            if self.nedt is not None:
                noise = rng.normal(size=tbs.shape)
                for i in range(noise.shape[-1]):
                    noise[..., i] *= self.nedt[i]
                tbs += noise

            # Apply modeling error caused by simulator.
            if self.modeling_error is not None:
                noise = rng.normal(size=self.n_chans)
                for i, n in enumerate(noise):
                    tbs[..., i] += self.modeling_error[i] * n

            r = rng.random()
            if r > 0.80:
                n_p = rng.integers(6, 16)
                tbs[:, :n_p] = np.nan
            r = rng.random()
            if r > 0.80:
                n_p = rng.integers(6, 16)
                tbs[:, -n_p:] = np.nan

        eia = eia[np.newaxis]
        if augment:
            eia = eia + rng.uniform(-0.5, 0.5, size=eia.shape)
        tcwv = tcwv[np.newaxis]
        t2m = t2m[np.newaxis]
        st = self.load_surface_type(scene)
        am = self.load_airmass_type(scene)

        tbs = np.transpose(tbs, (2, 0, 1))
        st = np.transpose(st, (2, 0, 1))
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
                x[self.n_chans] *= -1.0
                for k in targets:
                    y[k] = np.flip(y[k], -2)

            r = rng.random()
            if r > 0.5:
                x = np.flip(x, -1)
                for k in targets:
                    y[k] = np.flip(y[k], -1)
        return x, y

    def _load_training_data_3d_other(
        self, scene, targets, augment, rng, width=32, height=128
    ):
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
            width: The width of each input image.
            height: The height of each input image.

        Returns:
            Tuple ``x, y`` containing one sample of training data for the
            GPROF-NN 3D retrieval.
        """
        if augment:
            p_x = rng.random()
            p_y = rng.random()
        else:
            p_x = 0.5
            p_y = 0.5

        n_scans = SCANS_PER_SAMPLE
        n_pixels = self.viewing_geometry.pixels_per_scan

        i = height // 2 + (n_scans - height) * p_y
        j = (SCANS_PER_SAMPLE - n_pixels + width) // 2 + (n_pixels - width) * p_x

        i_start = int(i - height // 2)
        i_end = int(i + height // 2)
        j_start = int(j - width // 2)
        j_end = int(j + width // 2)

        tbs = self.load_brightness_temperatures(scene, None)
        tbs = tbs[i_start:i_end, j_start:j_end]
        tbs = np.transpose(tbs, (2, 0, 1))

        r = rng.random()
        if r > 0.80:
            n_p = rng.integers(6, 16)
            tbs[:, :n_p] = np.nan
        r = rng.random()
        if r > 0.80:
            n_p = rng.integers(6, 16)
            tbs[:, -n_p:] = np.nan

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

    def load_training_data_3d(
        self, dataset, targets, augment, rng, width=32, height=128
    ):
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
            "latitude",
            "longitude",
        ]
        if "surface_precip" not in targets:
            vs += ["surface_precip"]

        for i in range(n):
            scene = decompress_scene(dataset[{"samples": i}], targets + vs)
            source = scene.source
            if source == 0:
                x_i, y_i = self._load_training_data_3d_sim(
                    scene, targets, augment, rng, width=width, height=height
                )
            else:
                x_i, y_i = self._load_training_data_3d_other(
                    scene, targets, augment, rng, width=width, height=height
                )
            x.append(x_i)
            y.append(y_i)

        x = np.stack(x)
        y = {k: np.stack([y_i[k] for y_i in y]) for k in y[0]}

        return x, y


###############################################################################
# Platforms
###############################################################################


class Platform:
    """
    The 'Platform' class represents the satellite that a specifc sensor
    is flown on. It is used to hold information that is specifc to that
    platform such as the data path and the prefix of L1C files.
    """
    def __init__(
            self,
            name,
            l1c_file_path,
            l1c_file_prefix
    ):
        self.name = name
        self.l1c_file_path = l1c_file_path
        self.l1c_file_prefix = l1c_file_prefix

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Platform(name={self.name})"


TRMM = Platform("TRMM", "/pdata4/archive/GPM/1C_TMI/", "1C.TRMM.TMI")
NOAA19 = Platform("NOAA19", "/pdata4/archive/GPM/1C_NOAA19/", "1C.NOAA19.MHS")
GPM = Platform("GPM-CO", "/pdata4/archive/GPM/1CR_GMI/", "1C-R.GPM.GMI")

###############################################################################
# GMI
###############################################################################

GMI_CHANNELS = [
    (10.6, "V"),
    (10.6, "H"),
    (18.7, "V"),
    (18.7, "H"),
    (23.0, "V"),
    (np.nan, ""),
    (37.0, "V"),
    (37.0, "H"),
    (89.0, "V"),
    (89.0, "H"),
    (166.0, "V"),
    (166.0, "H"),
    (np.nan, ""),
    (186.0, "V"),
    (190.0, "H"),
]

GMI_ANGLES = np.array([
    52.8,
    52.8,
    52.8,
    52.8,
    52.8,
    -9999.9,
    52.8,
    52.8,
    52.8,
    52.8,
    49.19,
    49.19,
    -9999.9,
    49.19,
    49.19,
])


GMI_VIEWING_GEOMETRY = Conical(
    altitude=455e3,
    earth_incidence_angle=53.0,
    scan_range=140.0,
    pixels_per_scan=221,
    scan_offset=13.4e3,
)


GMI = ConicalScanner(
    "GMI",
    GMI_CHANNELS,
    GMI_ANGLES,
    GPM,
    GMI_VIEWING_GEOMETRY,
    "/pdata4/veljko/GMI2MRMS_match2019/db_mrms4GMI/",
    "GMI.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7",
)

###############################################################################
# MHS
###############################################################################

MHS_ANGLES = np.array(
    [59.798, 53.311, 46.095, 39.222, 32.562, 26.043, 19.619, 13.257, 6.934, 0.0]
)

MHS_CHANNELS = [
    (89.0, "V"),
    (150.0, "H"),
    (184.0, "V"),
    (186.0, "V"),
    (190.0, "H"),
]

MHS_NEDT = np.array([0.3, 0.5, 3.8, 0.7, 0.3])

MHS_VIEWING_GEOMETRY = CrossTrack(
    altitude=855e3, scan_range=2.0 * 49.5, pixels_per_scan=90, scan_offset=17e3
)

MHS_GMI_CHANNELS = [8, 11, 13, 13, 14]

MHS = CrossTrackScanner(
    "MHS",
    MHS_CHANNELS,
    MHS_NEDT,
    MHS_ANGLES,
    NOAA19,
    MHS_VIEWING_GEOMETRY,
    "/pdata4/veljko/MHS2MRMS_match2019/monthly_2021/",
    "MHS.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7x",
    MHS_GMI_CHANNELS,
)

MHS_NOAA19 = CrossTrackScanner(
    "MHS",
    MHS_CHANNELS,
    MHS_NEDT,
    MHS_ANGLES,
    NOAA19,
    MHS_VIEWING_GEOMETRY,
    "/pdata4/veljko/MHS2MRMS_match2019/monthly_2021/",
    "MHS.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7x",
    MHS_GMI_CHANNELS,
    correction=DATA_FOLDER / "corrections_mhs_noaa19.nc",
    modeling_error=[3.0, 2.0, 2.0, 2.0, 2.0]
)

MHS_NOAA19_FULL = CrossTrackScanner(
    "MHS",
    MHS_CHANNELS,
    MHS_NEDT,
    MHS_ANGLES,
    NOAA19,
    MHS_VIEWING_GEOMETRY,
    "/pdata4/veljko/MHS2MRMS_match2019/monthly_2021/",
    "MHS.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7x",
    MHS_GMI_CHANNELS,
    correction=DATA_FOLDER / "corrections_mhs_noaa19_full.nc",
    modeling_error=[3.0, 2.0, 2.0, 2.0, 2.0]
)


def get_sensor(sensor, platform=None):
    if platform is not None:
        platform = platform.upper().replace("-", "")
        key = f"{sensor.upper()}_{platform.upper()}"
        try:
            return globals()[key]
        except KeyError:
            pass
    key = sensor.upper()
    return globals()[key]

###############################################################################
# TMI
###############################################################################

TMI_CHANNELS = [
    (10.65, "V"),
    (10.65, "H"),
    (19.35, "V"),
    (19.35, "H"),
    (21.3, "V"),
    (37.0, "V"),
    (37.0, "H"),
    (85.5, "V"),
    (85.5, "H"),
]

TMI_ANGLES = np.array([
    52.8,
    52.8,
    52.8,
    52.8,
    52.8,
    52.8,
    52.8,
    52.8,
    52.8,
])

TMI_NEDT = np.array([
    0.63,
 	0.54,
 	0.50,
 	0.47,
 	0.71,
 	0.36,
 	0.31,
 	0.52,
 	0.93
])

TMI_GMI_CHANNELS = [0, 1, 2, 3, 4, 6, 7, 8, 9]


TMIPR_VIEWING_GEOMETRY = Conical(
    altitude=350e3,
    earth_incidence_angle=53.0,
    scan_range=130.0,
    pixels_per_scan=208,
    scan_offset=13.4e3,
)

TMIPO_VIEWING_GEOMETRY = Conical(
    altitude=400e3,
    earth_incidence_angle=53.0,
    scan_range=130.0,
    pixels_per_scan=208,
    scan_offset=13.4e3,
)


TMIPR = ConstellationScanner(
    "TMI",
    TMI_CHANNELS,
    TMI_NEDT,
    TMI_ANGLES,
    TRMM,
    TMIPR_VIEWING_GEOMETRY,
    None,
    "TMIPR.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7_tmi",
    TMI_GMI_CHANNELS
)


TMIPO = ConstellationScanner(
    "TMI",
    TMI_CHANNELS,
    TMI_NEDT,
    TMI_ANGLES,
    TRMM,
    TMIPO_VIEWING_GEOMETRY,
    None,
    "TMIPO.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7_tmi",
    TMI_GMI_CHANNELS
)

TMI = TMIPR
