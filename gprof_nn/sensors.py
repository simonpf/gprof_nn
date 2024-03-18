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
     sensor}_{platform}_C. This sensor also applies a
     quantile-matching correction to the training data.
"""
from abc import ABC, abstractmethod, abstractproperty
from concurrent.futures import ProcessPoolExecutor
from copy import copy
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
    w = np.exp(np.log(0.5) * (x**2 + y**2))
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
    res_x_source = 4.5e3
    res_a_source = 13.5e3
    angles = sensor.angles
    res_x_target = geometry.get_resolution_x(angles)
    res_a_target = geometry.get_resolution_a(angles)

    kernels = []
    for res_a, res_x in zip(res_a_target, res_x_target):
        k = calculate_smoothing_kernel(
            res_x_source, res_a_source, res_x, res_a, size=11
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


def drop_inputs_from_sample(x, probability, sensor, rng):
    """
    Drop inputs from network input.

    Args:
        x: The input sample with channels/features along the first axis..
        probability: The probability with which to drop a channels.
        sensor: The sensor for which the data is extracted.
        rng: Random generator object to use.
    """
    scalar_inputs = sensor.n_chans + 2
    if isinstance(sensor, sensors.CrossTrackScanner):
        scalar_input += 1
    # Iterate of scalar variable and drop randomly
    for i_ch in range(scalar_input):
        r = rng.uniform()
        if r <= probability:
            x[i] = np.nan

    # Drop surface type
    r = rng.uniform()
    if r <= probability:
        x[scalar_input : scalar_inputs + 8] = np.nan

    # Drop airmass type
    r = rng.uniform()
    if r <= probability:
        x[scalar_input + 8 : scalar_inputs + 12] = np.nan


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
        self._latitude_ratios = None
        self.platform = platform
        self.viewing_geometry = viewing_geometry
        self.missing_channels = None
        self.l1c_version = ""
        self.use_simulated_tbs = False
        self.delta_tbs = None

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
        n_inputs = self.n_freqs + 9
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

    @l1c_file_path.setter
    def l1c_file_path(self, path):
        """
        The default file path on the CSU system.
        """
        self.platform.l1c_file_path = path

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

    @mrms_file_record.setter
    def mrms_file_record(self, record):
        """
        Numpy dtype defining the binary record structure for MRMS match
        up files.
        """
        self._mrms_file_record = record

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

    @property
    def latitude_ratios(self):
        """
        Latitude ratios to use for resampling of training data.
        """
        if self._latitude_ratios is not None:
            if not isinstance(self._latitude_ratios, np.ndarray):
                self._latitude_ratios = np.load(self._latitude_ratios)
        return self._latitude_ratios

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
        return load_variable(data, "viewing_angle", mask=mask)

    def load_land_fraction(self, data, mask=None):
        """
        Load land fraction from dataset.
        """
        return load_variable(data, "land_fraction", mask=mask)

    def load_ice_fraction(self, data, mask=None):
        """
        Load ice fraction from dataset
        """
        return load_variable(data, "ice_fraction", mask=mask)

    def load_snow_depth(self, data, mask=None):
        """
        Load ice fraction from dataset
        """
        return load_variable(data, "snow_depth", mask=mask)

    def load_leaf_area_index(self, data, mask=None):
        """
        Load ice fraction from dataset
        """
        return load_variable(data, "leaf_area_index", mask=mask)

    def load_orographic_wind(self, data, mask=None):
        """
        Load orographic uplift.
        """
        return load_variable(data, "orographic_wind", mask=mask)

    def load_moisture_convergence(self, data, mask=None):
        """
        Load moisutre convergence
        """
        return load_variable(data, "moisture_convergence", mask=mask)

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
        return self.n_chans + 9

    @property
    def mrms_file_path(self):
        return self._mrms_file_path

    @mrms_file_path.setter
    def mrms_file_path(self, path):
        self._mrms_file_path = path

    @property
    def sim_file_pattern(self):
        return self._sim_file_pattern

    @property
    def sim_file_path(self):
        return self._sim_file_path

    @sim_file_path.setter
    def sim_file_path(self, path):
        self._sim_file_path = path

    def load_brightness_temperatures(self, data, angles=None, mask=None):
        if self.use_simulated_tbs:
            return load_variable(data, "simulated_brightness_temperatures", mask=mask)
        return load_variable(data, "brightness_temperatures", mask=mask)

    def load_scene(
        self, scene, targets, augment, variables, rng, width, height, drop_inputs=None
    ):
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
            p_y = rng.random()

        lats = scene.latitude.data
        lons = scene.longitude.data
        if augment:
            alt_old = copy(self.viewing_geometry.altitude)
            so_old = copy(self.viewing_geometry.scan_offset)
            self.viewing_geometry.altitude = alt_old + np.random.uniform(-10, 10)
            self.viewing_geometry.scan_offset = so_old + np.random.uniform(-0.2, 0.2)
        coords = get_transformation_coordinates(
            lats, lons, self.viewing_geometry, width, height, p_x_i, p_x_o, p_y
        )
        if augment:
            self.viewing_geometry.altitude = alt_old
            self.viewing_geometry.scan_offset = so_old

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
        land_frac = self.load_land_fraction(scene)[None]
        ice_frac = self.load_ice_fraction(scene)[None]
        snow_depth = self.load_snow_depth(scene)[None]
        leaf_area_index = self.load_leaf_area_index(scene)[ None]
        orographic_wind = self.load_orographic_wind(scene)[ None]
        moisture_conv = self.load_moisture_convergence(scene)[None]

        x = np.concatenate([
            tbs,
            t2m,
            tcwv,
            land_frac,
            ice_frac,
            snow_depth,
            leaf_area_index,
            orographic_wind,
            moisture_conv
        ], axis=0)

        #
        # Output data
        #
        y = {}

        for t in targets:
            y_t = self.load_target(scene, t)

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

        if drop_inputs is not None:
            drop_inputs_from_sample(x, drop_inputs, self, rng)

        return x, y

    def load_training_data_3d(
        self,
        dataset,
        targets,
        augment,
        rng,
        width=96,
        height=128,
        n_workers=1,
        drop_inputs=None,
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
            drop_inputs: A probability with which to set all inputs randomly
                to a missing value.

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
                tasks.append(
                    pool.submit(
                        ConicalScanner._load_scene_3d,
                        self,
                        scene,
                        targets,
                        augment,
                        vs,
                        rng,
                        width,
                        height,
                        drop_inputs=drop_inputs,
                    )
                )

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
                if not augment or scene.source == 0:
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
                        height,
                        drop_inputs=drop_inputs,
                    )
                    x.append(x_i)
                    for target in targets:
                        y.setdefault(target, []).append(y_i[target])

        x = np.stack(x)
        for k in targets:
            y[k] = np.stack(y[k])

        if loaded:
            dataset.close()

        if drop_inputs is not None:
            drop_inputs_from_sample(x, drop_inputs, self, rng)

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
        correction=None,
        modeling_error=None,
        latitude_ratios=None,
    ):
        super().__init__(
            name,
            channels,
            angles,
            platform,
            viewing_geometry,
            mrms_file_path,
            sim_file_pattern,
            sim_file_path,
        )
        self.nedt = nedt
        self.gmi_channels = gmi_channels

        if correction is None:
            self._correction = None
        else:
            self._correction = correction

        self.modeling_error = modeling_error

        self._latitude_ratios = latitude_ratios

        # MRMS types
        self._mrms_file_record = types.get_mrms_file_record(
            self.n_chans, self.n_angles, types.CONICAL_CONST
        )

    @property
    def correction(self):
        if self._correction is not None:
            if not isinstance(self._correction, CdfCorrection):
                self._correction = CdfCorrection(self._correction)
        return self._correction

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

            if not self.use_simulated_tbs:
                bias = load_variable(data, "brightness_temperature_biases")
                if mask is not None:
                    bias = bias[mask]
                tbs = tbs_sim - bias

        else:
            tbs = load_variable(data, "brightness_temperatures", mask=mask)
        return tbs

    def load_training_data_1d(
        self, dataset, targets, augment, rng, indices=None, drop_inputs=None
    ):
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
            indices: List of scene indices from which to load the data. This
                can be used to resample training scenes.
            drop_inputs: A probability with which to set all inputs randomly
                to a missing value.

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

        vs = ["simulated_brightness_temperatures", "brightness_temperature_biases"]
        if "surface_precip" not in targets:
            vs += ["surface_precip"]

        if indices is None:
            indices = range(n_samples)

        for sample_index in indices:
            scene = decompress_scene(dataset[{"samples": sample_index}], targets + vs)
            source = dataset.source[sample_index]

            sp = scene["surface_precip"].data
            mask = sp >= 0

            if source == 0:
                tbs = scene["simulated_brightness_temperatures"].data
                mask_tbs = get_mask(tbs, *LIMITS["simulated_brightness_temperatures"])
                biases = scene["brightness_temperature_biases"].data
                mask_biases = get_mask(biases, *LIMITS["brightness_temperature_biases"])
                mask = mask * np.all(mask_tbs, axis=-1) * np.all(mask_biases, axis=-1)
            else:
                tbs = scene["brightness_temperatures"].data
                mask = mask * np.any((tbs > 0) * (tbs < 500), axis=-1)

            #
            # Input data
            #

            tbs = self.load_brightness_temperatures(scene, mask=mask)
            t2m = self.load_two_meter_temperature(scene, mask=mask)
            tcwv = self.load_total_column_water_vapor(scene, mask=mask)
            st = scene.surface_type.data[mask]

            if self.correction is not None:
                tbs = self.correction(rng, self, tbs, st, None, tcwv, augment=augment)

            # Add thermal noise to simulated observations.
            if augment and self.nedt is not None:
                noise = rng.normal(size=tbs.shape)
                for i in range(noise.shape[-1]):
                    noise[..., i] *= self.nedt[i]
                tbs = tbs + noise

            # Randomly set missing channels to missing.
            if self.missing_channels is not None:
                p_thresh = 1 / len(self.missing_channels)
                for channel in self.missing_channels:
                    tbs_c = tbs[..., channel]
                    p = rng.uniform(size=tbs_c.shape)
                    tbs_c[p <= p_thresh] = np.nan

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

        if drop_inputs is not None:
            drop_inputs_from_sample(x, drop_inputs, self, rng)

        return x, y

    def _load_training_data_3d_sim(
        self, scene, targets, augment, rng, width=32, height=128, drop_inputs=None
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
            drop_inputs: A probability with which to set all inputs randomly
                to a missing value.

        Returns:
            Tuple ``x, y`` containing one sample of training data for the
            GPROF-NN 3D retrieval.
        """
        if augment:
            p_x_i = rng.random()
            p_y = rng.random()
            p_x_o = rng.random()
        else:
            p_x_i = 0.5
            p_x_o = 0.5
            p_y = rng.random()

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
            tbs = self.correction(rng, self, tbs, st, None, tcwv, augment=augment)

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

            # Simulate missing obs at edge of swath.
            # This may not be not necessary for concical scanners.
            r = rng.random()
            if r > 0.80:
                n_p = rng.integers(6, 16)
                tbs[:, :n_p] = np.nan
            r = rng.random()
            if r > 0.80:
                n_p = rng.integers(6, 16)
                tbs[:, -n_p:] = np.nan

            # Randomly set missing channels to missing.
            if augment:
                p = rng.uniform()
                if p > 0.75:
                    if self.missing_channels is not None:
                        p_thresh = 0.5
                        for channel in self.missing_channels:
                            p = rng.uniform()
                            if p <= p_thresh:
                                tbs[..., channel] = np.nan

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
                for k in targets:
                    y[k] = np.flip(y[k], -2)

            r = rng.random()
            if r > 0.5:
                x = np.flip(x, -1)
                for k in targets:
                    y[k] = np.flip(y[k], -1)

        if drop_inputs is not None:
            drop_inputs_from_sample(x, drop_inputs, self, rng)

        return x, y

    def _load_training_data_3d_other(
        self, scene, targets, augment, rng, width=32, height=128, drop_inputs=None
    ):
        """
        Load training data for sea ice or snow surfaces. These observations
        were extracted directly from L1C files and  correspond to the original
        viewing geometry.

        Args:
            scene: 'xarray.Dataset' containing the training data sample
                to load.
            targets: List of the retrieval targets to load as output data.
            augment: Whether or not to augment the training data.
            rng: 'numpy.random.Generator' to use to generate random numbers.
            width: The width of each input image.
            height: The height of each input image.
            drop_inputs: A probability with which to set all inputs randomly
                to a missing value.

        Returns:
            Tuple ``x, y`` containing one sample of training data for the
            GPROF-NN 3D retrieval.
        """
        p_x = rng.random()
        p_y = rng.random()

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

        # Randomly set missing channels to missing.
        if augment:
            p = rng.uniform()
            if p > 0.75:
                if self.missing_channels is not None:
                    p_thresh = 0.5
                    for channel in self.missing_channels:
                        p = rng.uniform()
                        if p <= p_thresh:
                            tbs[..., channel] = np.nan

        # Move channel to first dim.
        tbs = np.transpose(tbs, (2, 0, 1))

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

        if drop_inputs is not None:
            drop_inputs_from_sample(x, drop_inputs, self, rng)

        return x, y

    def load_training_data_3d(
        self,
        dataset,
        targets,
        augment,
        rng,
        width=32,
        height=128,
        indices=None,
        drop_inputs=None,
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
            "source",
            "latitude",
            "longitude",
        ]
        if "surface_precip" not in targets:
            vs += ["surface_precip"]

        if indices is None:
            indices = range(n)
        for sample_index in indices:
            scene = decompress_scene(dataset[{"samples": sample_index}], targets + vs)
            source = scene.source
            if source == 0:
                x_i, y_i = self._load_training_data_3d_sim(
                    scene, targets, augment, rng, width=width, height=height, drop_inputs=drop_inputs
                )
            else:
                x_i, y_i = self._load_training_data_3d_other(
                    scene, targets, augment, rng, width=width, height=height, drop_inputs=drop_inputs
                )
            x.append(x_i)
            y.append(y_i)

        x = np.stack(x)
        y = {k: np.stack([y_i[k] for y_i in y]) for k in y[0]}

        if drop_inputs is not None:
            drop_inputs_from_sample(x, drop_inputs, self, rng)

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
        modeling_error=None,
    ):
        super().__init__(
            types.XTRACK, name, channels, angles, platform, viewing_geometry
        )
        self.nedt = nedt
        n_chans = len(channels)
        n_angles = len(angles)
        self.kernels = calculate_smoothing_kernels(self)

        self.gmi_channels = np.array(gmi_channels)
        gmi_angles = GMI.angles[self.gmi_channels]
        self.bias_scales = np.cos(np.deg2rad(gmi_angles).reshape(1, -1)) / np.cos(
            np.deg2rad(self.angles).reshape(-1, 1)
        )
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
            modeling_error = [modeling_error] * n_chans
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

    @mrms_file_path.setter
    def mrms_file_path(self, path):
        self._mrms_file_path = path

    @property
    def sim_file_pattern(self):
        return self._sim_file_pattern

    @property
    def sim_file_path(self):
        return self._sim_file_path

    @sim_file_path.setter
    def sim_file_path(self, path):
        self._sim_file_path = path

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

            if not self.use_simulated_tbs:
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

    def load_training_data_1d(
        self, dataset, targets, augment, rng, indices=None, drop_inputs=None
    ):
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
            indices: A list of scene indices to use to restrict the
                training data to.
            drop_inputs: A probability with which to drop inputs.

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

        if indices is None:
            indices = range(n_samples)

        for sample_index in indices:
            scene = decompress_scene(dataset[{"samples": sample_index}], targets + vs)
            source = dataset.source[sample_index]

            sp = scene["surface_precip"].data
            mask = np.all(sp >= 0, axis=-1)

            if source == 0:
                tbs = scene["simulated_brightness_temperatures"].data
                mask_tbs = get_mask(tbs, *LIMITS["simulated_brightness_temperatures"])
                biases = scene["brightness_temperature_biases"].data
                mask_biases = get_mask(biases, *LIMITS["brightness_temperature_biases"])
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
                eia = load_variable(scene, "earth_incidence_angle", mask=mask)

            #
            # Input data
            #

            tbs = self.load_brightness_temperatures(scene, weights, mask=mask)
            t2m = self.load_two_meter_temperature(scene, mask=mask)
            tcwv = self.load_total_column_water_vapor(scene, mask=mask)
            st = scene.surface_type.data[mask]

            if self.correction is not None:
                tbs = self.correction(rng, self, tbs, st, eia, tcwv, augment=augment)

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

            if drop_inputs is not None:
                drop_inputs_from_sample(x, drop_inputs, self, rng)

        x = np.concatenate(x, axis=0)
        y = {t: np.concatenate(y[t], axis=0) for t in y}

        if loaded:
            dataset.close()

        return x, y

    def _load_training_data_3d_sim(
        self,
        scene,
        targets,
        augment,
        rng,
        width=32,
        height=128,
        drop_inputs=None,
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
            p_y = rng.random()

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

        center = self.viewing_geometry.get_window_center(p_x_o, width)
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
            tbs = self.correction(rng, self, tbs, st, eia, tcwv, augment=augment)

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

        if drop_inputs is not None:
            drop_inputs_from_sample(x, drop_inputs, self, rng)

        return x, y

    def _load_training_data_3d_other(
        self,
        scene,
        targets,
        augment,
        rng,
        width=32,
        height=128,
        drop_inputs=None,
    ):
        """
        Load training data for sea ice or snow surfaces. These observations
        were extracted directly from L1C files and correspond to the original
        viewing geometry.

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
        p_x = rng.random()
        p_y = rng.random()

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
        eia = eia[np.newaxis, i_start:i_end, j_start:j_end]

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

        if drop_inputs is not None:
            drop_inputs(x, drop_self, self, rng)

        return x, y

    def load_training_data_3d(
        self, dataset, targets, augment, rng, width=32, height=128, drop_inputs=None
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
                    scene,
                    targets,
                    augment,
                    rng,
                    width=width,
                    height=height,
                    drop_inputs=drop_inputs,
                )
            else:
                x_i, y_i = self._load_training_data_3d_other(
                    scene,
                    targets,
                    augment,
                    rng,
                    width=width,
                    height=height,
                    drop_inputs=drop_inputs,
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

    def __init__(self, name, l1c_file_path, l1c_file_prefix):
        self.name = name
        self.l1c_file_path = l1c_file_path
        self.l1c_file_prefix = l1c_file_prefix

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Platform(name={self.name})"


TRMM = Platform("TRMM", "/pdata4/archive/GPM/1C_TMI_ITE/", "1C.TRMM.TMI")
NOAA19 = Platform("NOAA19", "/pdata4/archive/GPM/1C_NOAA19_V7/", "1C.NOAA19.MHS")
NPP = Platform("NPP", "/pdata4/archive/GPM/1C_ATMS_ITE/", "1C.NPP.ATMS")
GPM = Platform("GPM-CO", "/pdata4/archive/GPM/1CR_GMI_V7/", "1C-R.GPM.GMI")
F15 = Platform("F15", "/pdata4/archive/GPM/1C_F15_ITE/", "1C.F15.SSMI")
F17 = Platform("F17", "/pdata4/archive/GPM/1C_F17_ITE/", "1C.F17.SSMIS")
GCOMW1 = Platform("GCOM-W1", "/pdata4/archive/GPM/1C_AMSR2_V7/", "1C.GCOMW1.AMSR2")
AQUA = Platform("AQUA", "/pdata4/archive/GPM/1C_AMSRE/", "1C.AQUA.AMSRE")

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


GMI_ANGLES = np.array(
    [
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
    ]
)


GMI_VIEWING_GEOMETRY = Conical(
    altitude=444e3,
    earth_incidence_angle=53.0,
    scan_range=140.0,
    pixels_per_scan=221,
    scan_offset=13.2e3,
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
    [59.498, 53.311, 46.095, 39.222, 32.562, 26.043, 19.619, 13.257, 6.934, 0.0]
)

MHS_CHANNELS = [
    (89.0, "V"),
    (150.0, "H"),
    (184.0, "V"),
    (186.0, "V"),
    (190.0, "H"),
]

MHS_NEDT = np.array([0.3, 0.5, 0.5, 0.5, 0.5])

MHS_VIEWING_GEOMETRY = CrossTrack(
    altitude=855e3,
    scan_range=2.0 * 49.5,
    pixels_per_scan=90,
    scan_offset=17e3,
    beam_width=1.1,
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
    "/qdata1/pbrown/dbaseV7/simV7x_mhs",
    MHS_GMI_CHANNELS,
    correction=DATA_FOLDER / "corrections_mhs.nc",
    modeling_error=[3.0, 2.0, 2.0, 2.0, 2.0],
)

MHS_GPROF = CrossTrackScanner(
    "MHS",
    MHS_CHANNELS,
    MHS_NEDT,
    MHS_ANGLES,
    NOAA19,
    MHS_VIEWING_GEOMETRY,
    "/pdata4/veljko/MHS2MRMS_match2019/monthly_2021/",
    "MHS.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7x_mhs",
    MHS_GMI_CHANNELS,
    correction=DATA_FOLDER / "corrections_mhs_gprof.nc",
    modeling_error=[3.0, 2.0, 2.0, 2.0, 2.0],
)

MHS_NO_CORRECTION = CrossTrackScanner(
    "MHS",
    MHS_CHANNELS,
    MHS_NEDT,
    MHS_ANGLES,
    NOAA19,
    MHS_VIEWING_GEOMETRY,
    "/pdata4/veljko/MHS2MRMS_match2019/monthly_2021/",
    "MHS.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7x_mhs",
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
    "/qdata1/pbrown/dbaseV7/simV7x_mhs",
    MHS_GMI_CHANNELS,
    correction=DATA_FOLDER / "corrections_mhs_noaa19.nc",
    modeling_error=[3.0, 2.0, 2.0, 2.0, 2.0],
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
    "/qdata1/pbrown/dbaseV7/simV7x_mhs",
    MHS_GMI_CHANNELS,
    correction=DATA_FOLDER / "corrections_mhs_noaa19_full.nc",
    modeling_error=[3.0, 2.0, 2.0, 2.0, 2.0],
)


def get_sensor(sensor, platform=None, date=None):
    if platform is not None:
        platform = platform.upper().replace("-", "")
        key = f"{sensor.upper()}_{platform.upper()}"
        if key not in globals():
            key = sensor.upper()
    else:
        key = sensor.upper()
    sensor = globals()[key]
    if sensor == TMI:
        if date is not None:
            if date > np.datetime64("2001-08-22T00:00:00"):
                sensor = TMIPO
            else:
                sensor = TMIPR

    return sensor


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

TMI_ANGLES = np.array(
    [
        52.8,
        52.8,
        52.8,
        52.8,
        52.8,
        52.8,
        52.8,
        52.8,
        52.8,
    ]
)

TMI_NEDT = np.array([0.63, 0.54, 0.50, 0.47, 0.71, 0.36, 0.31, 0.52, 0.93])

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

TMIPR_MODELING_ERROR = np.sqrt(np.array([1.1, 2.4, 1.0, 2.6, 0.9, 2.0, 7.1, 2.4, 6.1]))

TMIPR_NC = ConstellationScanner(
    "TMIPR",
    TMI_CHANNELS,
    TMI_NEDT,
    TMI_ANGLES,
    TRMM,
    TMIPR_VIEWING_GEOMETRY,
    None,
    "TMIPR.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7_tmi",
    TMI_GMI_CHANNELS,
)

TMIPR = ConstellationScanner(
    "TMIPR",
    TMI_CHANNELS,
    TMI_NEDT,
    TMI_ANGLES,
    TRMM,
    TMIPR_VIEWING_GEOMETRY,
    None,
    "TMIPR.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7_tmi",
    TMI_GMI_CHANNELS,
    correction=DATA_FOLDER / "corrections_tmipr.nc",
    modeling_error=TMIPR_MODELING_ERROR,
    latitude_ratios=DATA_FOLDER / "latitude_ratios_tmipr.npy",
)


TMIPO = ConstellationScanner(
    "TMIPO",
    TMI_CHANNELS,
    TMI_NEDT,
    TMI_ANGLES,
    TRMM,
    TMIPO_VIEWING_GEOMETRY,
    None,
    "TMIPO.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7_tmi",
    TMI_GMI_CHANNELS,
    correction=DATA_FOLDER / "corrections_tmipo.nc",
    latitude_ratios=DATA_FOLDER / "latitude_ratios_tmipo.npy",
)

TMI = TMIPR

###############################################################################
# SSMI
###############################################################################

SSMI_CHANNELS = [
    (19.35, "V"),
    (19.35, "H"),
    (22.235, "V"),
    (37.0, "H"),
    (37.0, "V"),
    (85.5, "H"),
    (85.5, "V"),
]

SSMI_ANGLES = np.array(
    [
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
    ]
)

SSMI_NEDT = np.array(
    [
        0.63,
        0.54,
        0.50,
        0.47,
        0.71,
        0.36,
        0.31,
    ]
)

SSMI_MODELING_ERROR = np.sqrt(
    np.array(
        [
            1.0,
            1.5,
            0.5,
            1.0,
            2.0,
            1.5,
            2.0,
        ]
    )
)

SSMI_GMI_CHANNELS = [2, 3, 4, 6, 7, 8, 9]

SSMI_VIEWING_GEOMETRY = Conical(
    altitude=833e3,
    earth_incidence_angle=53.0,
    scan_range=102.4,
    pixels_per_scan=128,
    scan_offset=12.5e3,
)

SSMI = ConstellationScanner(
    "SSMI",
    SSMI_CHANNELS,
    SSMI_NEDT,
    SSMI_ANGLES,
    F15,
    SSMI_VIEWING_GEOMETRY,
    None,
    "SSMI.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7_ssmi",
    SSMI_GMI_CHANNELS,
    modeling_error=SSMI_MODELING_ERROR,
    correction=DATA_FOLDER / "corrections_ssmi.nc",
)

SSMI_F08 = ConstellationScanner(
    "SSMI",
    SSMI_CHANNELS,
    SSMI_NEDT,
    SSMI_ANGLES,
    F15,
    SSMI_VIEWING_GEOMETRY,
    None,
    "SSMI.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7_ssmi",
    SSMI_GMI_CHANNELS,
    modeling_error=SSMI_MODELING_ERROR,
    correction=DATA_FOLDER / "corrections_ssmi.nc",
)
SSMI_F08.missing_channels = [5, 6]

###############################################################################
# SSMIS
###############################################################################

SSMIS_CHANNELS = [
    (19.35, "V"),
    (19.35, "H"),
    (22.235, "V"),
    (37.0, "V"),
    (37.0, "H"),
    (91.655, "V"),
    (91.655, "H"),
    (150, "H"),
    (189, "H"),
    (186, "H"),
    (184, "H"),
]

SSMIS_ANGLES = np.array(
    [
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
    ]
)

SSMIS_NEDT = np.array(
    [
        0.7,
        0.7,
        0.7,
        0.5,
        0.5,
        0.9,
        0.9,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
)

SSMIS_MODELING_ERROR = np.sqrt(
    np.array(
        [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )
)

SSMIS_GMI_CHANNELS = [2, 3, 4, 6, 7, 8, 9, 11, 14, 14, 13]


SSMIS_VIEWING_GEOMETRY = Conical(
    altitude=880e3,
    earth_incidence_angle=53.0,
    scan_range=140,
    pixels_per_scan=180,
    scan_offset=12.5e3,
)

SSMIS = ConstellationScanner(
    "SSMIS",
    SSMIS_CHANNELS,
    SSMIS_NEDT,
    SSMIS_ANGLES,
    F17,
    SSMIS_VIEWING_GEOMETRY,
    "/pdata4/veljko/SSMIS2MRMS_match2019/monthly_2021/",
    "SSMIS.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7_ssmis",
    SSMIS_GMI_CHANNELS,
    modeling_error=SSMIS_MODELING_ERROR,
    correction=DATA_FOLDER / "corrections_ssmis.nc",
)

SSMIS.missing_channels = [3, 7, 8, 9, 10]

###############################################################################
# AMSR2
###############################################################################

AMSR2_CHANNELS = [
    (10.65, "V"),
    (10.65, "H"),
    (18.7, "V"),
    (18.7, "H"),
    (23.8, "V"),
    (23.8, "H"),
    (36.5, "V"),
    (36.5, "H"),
    (89, "V"),
    (89, "H"),
]

AMSR2_ANGLES = np.array(
    [
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
        53.1,
    ]
)

AMSR2_NEDT = np.array([0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.7, 0.7, 1.2, 1.4])

AMSR2_MODELING_ERROR = np.sqrt(
    np.array(
        [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )
)

AMSR2_GMI_CHANNELS = [0, 1, 2, 3, 4, 4, 6, 7, 8, 9]

AMSR2_VIEWING_GEOMETRY = Conical(
    altitude=700e3,
    earth_incidence_angle=55.0,
    scan_range=140,
    pixels_per_scan=180,
    scan_offset=12.5e3,
)

AMSR2 = ConstellationScanner(
    "AMSR2",
    AMSR2_CHANNELS,
    AMSR2_NEDT,
    AMSR2_ANGLES,
    GCOMW1,
    AMSR2_VIEWING_GEOMETRY,
    "/pdata4/veljko/AMSR22MRMS_match2019/monthly_2021/",
    "AMSR2.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7_amsr2",
    AMSR2_GMI_CHANNELS,
    modeling_error=AMSR2_MODELING_ERROR,
    correction=DATA_FOLDER / "corrections_amsr2.nc",
)


###############################################################################
# AMSR-E
###############################################################################

AMSRE_CHANNELS = [
    (10.65, "V"),
    (10.65, "H"),
    (18.7, "V"),
    (18.7, "H"),
    (23.8, "V"),
    (23.8, "H"),
    (36.5, "V"),
    (36.5, "H"),
    (89, "V"),
    (89, "H"),
]

AMSRE_ANGLES = np.array(
    [
        55.0,
        55.0,
        55.0,
        55.0,
        55.0,
        55.0,
        55.0,
        55.0,
        55.0,
        55.0,
    ]
)

AMSRE_NEDT = np.array([0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1.1, 1.1])

AMSRE_MODELING_ERROR = np.sqrt(
    np.array(
        [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )
)

AMSRE_GMI_CHANNELS = [
    0,
    1,
    2,
    3,
    4,
    4,
    5,
    6,
    7,
    8,
]

AMSRE_VIEWING_GEOMETRY = Conical(
    altitude=703e3,
    earth_incidence_angle=55.0,
    scan_range=140,
    pixels_per_scan=180,
    scan_offset=12.5e3,
)

AMSRE = ConstellationScanner(
    "AMSRE",
    AMSRE_CHANNELS,
    AMSRE_NEDT,
    AMSRE_ANGLES,
    AQUA,
    AMSRE_VIEWING_GEOMETRY,
    "/pdata4/veljko/AMSR22MRMS_match2019/monthly_2021/",
    "AMSRE.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7_amsre",
    AMSRE_GMI_CHANNELS,
    modeling_error=AMSRE_MODELING_ERROR,
    correction=DATA_FOLDER / "corrections_amsre.nc",
)


AMSRE.mrms_sensor = AMSR2

###############################################################################
# ATMS
###############################################################################

ATMS_ANGLES = np.array(
    [59.498, 53.311, 46.095, 39.222, 32.562, 26.043, 19.619, 13.257, 6.934, 0.0]
)

ATMS_CHANNELS = [
    (88.2, "V"),
    (165.5, "H"),
    (184.0, "V"),
    (186.0, "V"),
    (190.0, "H"),
]

ATMS_NEDT = np.array([0.3, 0.5, 0.5, 0.5, 0.5])

ATMS_VIEWING_GEOMETRY = CrossTrack(
    altitude=855e3,
    scan_range=2.0 * 52.725,
    pixels_per_scan=96,
    scan_offset=16e3,
    beam_width=1.5,
)

ATMS_GMI_CHANNELS = [8, 11, 13, 13, 14]

ATMS = CrossTrackScanner(
    "ATMS",
    ATMS_CHANNELS,
    ATMS_NEDT,
    ATMS_ANGLES,
    NPP,
    ATMS_VIEWING_GEOMETRY,
    "/pdata4/veljko/MHS2MRMS_match2019/monthly_2021/",
    "ATMS.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7x_atms",
    ATMS_GMI_CHANNELS,
    correction=DATA_FOLDER / "corrections_atms.nc",
    modeling_error=[3.0, 2.0, 2.0, 2.0, 2.0],
)
ATMS.mrms_sensor = MHS
