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


GMI canonical channel slots

Channel index | Frequency
---------------------------
            0 |       10V
            1 |       10H
            2 |       19V
            3 |       19H
            4 |       22V
            5 |       22H
            6 |       37V
            7 |       37H
            8 |       89V
            9 |       89H
           10 |      150V
           11 |      150H
           12 |     183/1
           13 |     183/3
           14 |     183/7
"""
from abc import ABC, abstractmethod, abstractproperty
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import logging
from pathlib import Path

import numpy as np
import toml
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
import gprof_nn.augmentation
from gprof_nn.augmentation import (
    Conical,
    CrossTrack,
    get_transformation_coordinates,
    extract_domain,
    SCANS_PER_SAMPLE,
)


DATA_FOLDER = Path(__file__).parent / "files"


LOGGER = logging.getLogger(__file__)


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

    def __init__(
            self,
            kind,
            name,
            platform,
            viewing_geometry,
            gprof_channels,
            frequencies,
            offsets,
            polarization,
            orographic_enhancement
    ):
        self.kind = kind
        self._name = name
        self.platform = platform
        self.viewing_geometry = viewing_geometry
        self.gprof_channels = gprof_channels
        self.frequencies = frequencies
        self.offsets = offsets
        self.polarization = polarization
        self.orographic_enhancement = orographic_enhancement

        self.n_chans = len(self.gprof_channels)
        self.n_angles = 10 if isinstance(self, CrossTrackScanner) else 1

        # Bin file types
        self._bin_file_header = types.get_bin_file_header(self.n_chans, self.n_angles, kind)

        # Preprocessor types
        self._preprocessor_orbit_header = types.get_preprocessor_orbit_header(
            self.n_chans, self.kind
        )
        self._preprocessor_pixel_record = types.get_preprocessor_pixel_record(
            self.n_chans, self.kind
        )

        # Sim file types
        self._sim_file_header = types.get_sim_file_header(self.n_chans, self.n_angles, kind)
        self._sim_file_record = types.get_sim_file_record(
            self.n_chans, self.n_angles, N_LAYERS, kind
        )

        # MRMS types
        self._mrms_file_record = types.get_mrms_file_record(self.n_chans, self.n_angles, kind)

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
    def gprof_channel_indices(self) -> np.ndarray:
        return np.array([int(ind) for ind in self.gprof_channels.keys()])

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


class ConicalScanner(Sensor):
    """
    Base class for conically-scanning sensors.
    """

    def __init__(
            self,
            name,
            platform,
            viewing_geometry,
            gprof_channels,
            frequencies,
            offsets,
            polarization,
            orographic_enhancement
    ):
        super().__init__(
            types.CONICAL, name, platform, viewing_geometry, gprof_channels,
            frequencies, offsets, polarization, orographic_enhancement
        )
        self._n_angles = 1


    def __repr__(self):
        return f"ConicalScanner(name={self.name}, platform={self.platform.name})"

    @property
    def name(self):
        return self._name


class CrossTrackScanner(Sensor):
    """
    Base class for cross-track-scanning sensors.
    """
    def __init__(
            self,
            name,
            platform,
            viewing_geometry,
            gprof_channels,
            frequencies,
            offsets,
            polarization,
            orographic_enhancement
    ):
        super().__init__(
            types.XTRACK, name, platform, viewing_geometry, gprof_channels,
            frequencies, offsets, polarization, orographic_enhancement
        )

    def __repr__(self):
        return f"CrossTrackScanner(name={self.name}, " f"platform={self.platform.name})"



class ConstellationScanner(Sensor):
    """
    This class represents conically-scanning sensors that are for which
    observations can only be simulated.
    """

    def __init__(
            self,
            name,
            platform,
            viewing_geometry,
            gprof_channels,
            frequencies,
            offsets,
            polarization,
            orographic_enhancement
    ):
        super().__init__(
            types.CONICAL_CONST,
            name,
            platform,
            viewing_geometry,
            gprof_channels,
            frequencies,
            offsets,
            polarization,
            orographic_enhancement
        )


    def __repr__(self):
        return f"ConstellationScanner(name={self.name}, " f"platform={self.platform.name})"

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


def parse_sensor(sensor_file: Path) -> Sensor:
    """
    Parse sensor object from sensor config file.

    Args:
        sensor_file: A Path object pointing to the sensor file to parse.

    Return:
        A sensor object with the properties defined in 'sensor_file'.
    """
    sensor_config = toml.loads(open(sensor_file).read())
    sensor_name = sensor_file.stem.upper()

    platform = sensor_config.pop("platform", None)
    if platform is None:
        raise RuntimeError(
            f"Sensor defined in '{sensor_file}' is lacking the 'platform' entry.",
        )
    platform = Platform(**platform)

    viewing_geometry = sensor_config.pop("viewing_geometry", None)
    if viewing_geometry is None:
        raise RuntimeError(
            f"Sensor defined in '{sensor_file}' is lacking the 'viewing_geometry' entry.",
        )
    kind = viewing_geometry.pop("kind", None)
    if kind is None:
        raise RuntimeError(
            f"'viewing_geometry' entry of sensor file {sensor_file} is lacking "
            f"the 'kind' entry."
        )
    try:
        geometry_class = getattr(gprof_nn.augmentation, kind)
    except AttributeError:
        raise RuntimeError(
            f"The viewing geometry class '{geometry_class}' is not known."
        )
    viewing_geometry = geometry_class(**viewing_geometry)

    sensor = sensor_config.pop("sensor", None)
    if sensor is None:
        raise RuntimeError(
            f"Sensor defined in '{sensor_file}' is lacking the 'sensor' entry.",
        )
    kind = sensor.pop("kind", None)
    if kind is None:
        raise RuntimeError(
            f"'sensor' entry of sensor file {sensor_file} is lacking "
            f"the 'kind' entry."
        )
    try:
        sensor_class = globals().get(kind)
    except KeyError:
        raise RuntimeError(
            f"The sensor class '{sensor_class}' is not known."
        )

    name = sensor_file.stem.upper()
    args = [
        "gprof_channels",
        "frequencies",
        "offsets",
        "polarization",
        "orographic_enhancement"
    ]
    args = [sensor[arg] for arg in args]
    sensor = sensor_class(name.split('_')[0], platform, viewing_geometry, *args)
    return sensor


sensor_files = (Path(__file__).parent / "sensors").glob("*toml")
for sensor_file in sensor_files:
    sensor_config = toml.loads(open(sensor_file).read())
    sensor_name = sensor_file.stem.upper()
    try:
        sensor = parse_sensor(sensor_file)
        name = sensor_file.stem.upper()
        globals()[name] = sensor
    except Exception:
        LOGGER.exception(
            "The following error was encountered when trying to parse sensor "
            "config file %s.",
            sensor_file
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
