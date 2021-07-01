"""
================
gprof_nn.sensors
================

This module defines different sensor classes and objects to represent the
 various sensors of the GPM constellation.
"""
from abc import ABC, abstractmethod, abstractproperty

from netCDF4 import Dataset
import numpy as np
import xarray as xr

from gprof_nn.definitions import N_LAYERS


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


###############################################################################
# Helper functions
###############################################################################


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
# Sensor classes
###############################################################################


class Sensor(ABC):
    """
    The sensor class defines the abstract properties and methods that
    each sensor class must implement.
    """
    @abstractproperty
    def name(self):
        """
        The name of the sensor.
        """

    @abstractproperty
    def n_inputs(self):
        """
        The number of input features for the GPORF-NN retrieval.
        """

    @abstractproperty
    def n_freqs(self):
        """
        The number of frequencies or channels of the sensor.
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
        Numpy dtype defining the binary record structure of simulator
        files.
        """

    @abstractproperty
    def sim_file_record(self):
        """
        Numpy dtype defining the binary record structure of simulator
        files.
        """

    @abstractproperty
    def preprocessor(self):
        """
        The name of the preprocessor executable for this sensor.
        """

    @abstractproperty
    def preprocessor(self):
        """
        The name of the preprocessor executable for this sensor.
        """

    @abstractproperty
    def preprocessor_orbit_header(self):
        """
        Numpy dtype defining the binary structure of the header of
        preprocessor files.
        """

    @abstractproperty
    def preprocessor_file_record(self):
        """
        Numpy dtype defining the binary record structure of preprocessor
        files.
        """

    @abstractmethod
    def load_data_0d(self,
                     filename):
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
    def load_training_data_0d(self,
                              filename,
                              targets,
                              augment,
                              rng):
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


class ConicalScanner(Sensor):
    """
    Base class for conically-scanning sensors.
    """
    def __init__(self,
                 name,
                 n_freqs,
                 l1c_prefix,
                 l1c_file_path,
                 mrms_file_path,
                 sim_file_pattern,
                 sim_file_path,
                 preprocessor):

        self._name = name
        self._n_freqs = n_freqs
        self._l1c_file_prefix = l1c_prefix
        self._l1c_file_path = l1c_file_path

        self._mrms_file_path = mrms_file_path
        self._mrms_file_record =  np.dtype([
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
        ])

        self._sim_file_patthern = sim_file_pattern
        self._sim_file_path = sim_file_path
        self._sim_file_header = np.dtype([
            ("satellite_code", "a5"),
            ("sensor", "a5"),
            ("frequencies", f"{n_freqs}f4"),
            ("nominal_eia", f"{n_freqs}f4"),
            ("start_pixel", "i4"),
            ("end_pixel", "i4"),
            ("start_scan", "i4"),
            ("end_scan", "i4"),
        ])
        self._sim_file_record = np.dtype([
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
        ])
        self._preprocessor = preprocessor

        self._preprocessor_orbit_header = np.dtype([
            ("satellite", "a12"),
            ("sensor", "a12"),
            ("preprocessor", "a12"),
            ("profile_database_file", "a128"),
            ("radiometer_file", "a128"),
            ("calibration_file", "a128"),
            ("granule_number", "i"),
            ("number_of_scans", "i"),
            ("number_of_pixels", "i"),
            ("n_channels", "i"),
            ("frequencies", f"{n_freqs}f4"),
            ("comment", "a40"),
        ])
        self._preprocessor_file_record = np.dtype([
            ("latitude", "f4"),
            ("longitude", "f4"),
            ("brightness_temperatures", f"{n_freqs}f4"),
            ("earth_incidence_angle", f"{n_freqs}f4"),
            ("wet_bulb_temperature", "f4"),
            ("lapse_rate", "f4"),
            ("total_column_water_vapor", "f4"),
            ("surface_temperature", "f4"),
            ("two_meter_temperature", "f4"),
            ("quality_flag", "i"),
            ("sunglint_angle", "i1"),
            ("surface_type", "i1"),
            ("airmass_type", "i2"),
        ])

    @property
    def name(self):
        return _name

    @abstractproperty
    def n_inputs(self):
        """
        The number of input features for the GPORF-NN retrieval.
        """
        return self.n_freqs + 2 + 18 + 4

    @property
    def n_freqs(self):
        return self.n_freqs

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
        return self._sim_file_record

    @property
    def sim_file_record(self):
        return self._sim_file_record

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def preprocessor_orbit_header(self):
        return self._preprocessor_orbit_header

    @property
    def preprocessor_file_record(self):
        return self._preprocessor_file_record

    def load_data_0d(self,
                     filename,
                     targets,
                     augment,
                     rng):

        pass

    def load_training_data_0d(self,
                              filename,
                              targets,
                              augment,
                              rng):
        with Dataset(filename, "r") as dataset:

            variables = dataset.variables

            #
            # Input data
            #

            # Brightness temperatures
            sp = dataset["surface_precip"][:]
            valid = (sp >= 0)
            n = valid.sum()

            bts = dataset["brightness_temperatures"][:][valid]

            invalid = (bts > 500.0) + (bts < 0.0)
            bts[invalid] = np.nan

            # Simulate missing high-frequency channels
            if augment:
                r = rng.random(bts.shape[0])
                bts[r > 0.9, 10:15] = np.nan

            # 2m temperature, values less than 0 must be missing.
            t2m = variables["two_meter_temperature"][:][valid].reshape(-1, 1)
            t2m[t2m < 0] = np.nan

            # Total precitable water, values less than 0 are missing.
            tcwv = variables["total_column_water_vapor"][:][valid].reshape(-1, 1)
            # Surface type
            st = variables["surface_type"][:][valid]

            n_types = 18
            st_1h = np.zeros((n, n_types), dtype=np.float32)
            st_1h[np.arange(n), st.ravel() - 1] = 1.0
            # Airmass type
            # Airmass type is defined slightly different from surface type in
            # that there is a 0 type.
            am = variables["airmass_type"][:][valid]
            n_types = 4
            am_1h = np.zeros((n, n_types), dtype=np.float32)
            am_1h[np.arange(n), np.maximum(am.ravel(), 0)] = 1.0

            x = np.concatenate([bts, t2m, tcwv, st_1h, am_1h], axis=1)

            #
            # Output data
            #

            n = dataset.dimensions["samples"].size
            y = {}
            for t in targets:
                y_t = variables[t][:].filled(np.nan)
                y_t = _expand_pixels(y_t)[valid]
                np.nan_to_num(y_t, copy=False, nan=-9999)
                if t == "latent_heat":
                    y_t[y_t < -400] = -9999
                else:
                    y_t[y_t < 0] = -9999
                y[t] = y_t
        return x, y

    def load_data_2d(self, filename, targets):
        pass

    def load_training_data_2d(self, filename, targets):
        pass


class CrossTrackScanner(Sensor):
    """
    Base class for conically-scanning sensors.
    """
    def __init__(self,
                 name,
                 angles,
                 n_freqs,
                 l1c_prefix,
                 l1c_file_path,
                 mrms_file_path,
                 sim_file_pattern,
                 sim_file_path,
                 preprocessor):
        self._name = name
        self._angles = angles
        n_angles = angles.size
        self._n_freqs = n_freqs
        self._l1c_file_prefix = l1c_prefix
        self._l1c_file_path = l1c_file_path

        self._mrms_file_path = mrms_file_path
        self._mrms_file_record =  np.dtype([
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
        ])
        self._sim_file_pattern = sim_file_pattern
        self._sim_file_path = sim_file_path
        self._sim_file_header = np.dtype([
            ("satellite_code", "a5"),
            ("sensor", "a5"),
            ("frequencies", f"{n_freqs}f4"),
            ("viewing_angles", f"{n_angles}f4"),
            ("start_pixel", "i4"),
            ("end_pixel", "i4"),
            ("start_scan", "i4"),
            ("end_scan", "i4"),
        ])
        self._sim_file_record = np.dtype([
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
        ])
        self._preprocessor = preprocessor

        self._preprocessor_orbit_header = np.dtype([
            ("satellite", "a12"),
            ("sensor", "a12"),
            ("preprocessor", "a12"),
            ("profile_database_file", "a128"),
            ("radiometer_file", "a128"),
            ("calibration_file", "a128"),
            ("granule_number", "i"),
            ("number_of_scans", "i"),
            ("number_of_pixels", "i"),
            ("n_channels", "i"),
            ("frequencies", f"{n_freqs}f4"),
            ("comment", "a40"),
        ])
        self._preprocessor_file_record = np.dtype([
            ("latitude", "f4"),
            ("longitude", "f4"),
            ("brightness_temperatures", f"{n_freqs}f4"),
            ("earth_incidence_angle", f"f4"),
            ("wet_bulb_temperature", "f4"),
            ("lapse_rate", "f4"),
            ("total_column_water_vapor", "f4"),
            ("surface_temperature", "f4"),
            ("two_meter_temperature", "f4"),
            ("quality_flag", "i"),
            ("sunglint_angle", "i1"),
            ("surface_type", "i1"),
            ("airmass_type", "i2"),
        ])

    @property
    def name(self):
        return self._name

    @abstractproperty
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
    def preprocessor(self):
        return self._preprocessor

    @property
    def preprocessor_orbit_header(self):
        return self._preprocessor_orbit_header

    @property
    def preprocessor_file_record(self):
        return self._preprocessor_file_record

    def load_data_0d(self,
                     filename):

        with xr.open_dataset(filename) as dataset:

            #
            # Input data
            #

            n_samples = dataset.samples.size

            x = []
            y = {}

            n_angles = self.n_angles

            for i in range(n_samples):

                scene = dataset[{"samples": i}]
                source = dataset.source[i]

                n_scans = scene.scans.size
                n_pixels = scene.pixels.size

                if source == 0:

                    sp = scene["surface_precip"].data
                    np.nan_to_num(sp, nan=-9999, copy=False)

                    angles = np.arange(n_scans * n_pixels)
                    angles = angles.reshape(n_scans, n_pixels)
                    angles = angles % self.n_angles

                    inds_l = np.trunc(angles).astype(np.int32)
                    inds_r = np.ceil(angles).astype(np.int32)
                    f = angles - inds_r

                    # Interpolate brightness temperatures.
                    bts = scene["simulated_brightness_temperatures"]
                    bts = _expand_pixels(bts.data[np.newaxis])[0]
                    np.nan_to_num(bts, nan=-9999, copy=False)

                    bts_i = np.zeros((n_scans, n_pixels, self.n_freqs),
                                     dtype=np.float32)
                    for i in range(self.n_angles):
                        mask = inds_l == i
                        bts_i[mask] += f[mask, np.newaxis] * bts[mask, :, i]
                        mask = inds_r == i
                        bts_i[mask] += (1 - f[mask, np.newaxis]) * bts[mask, :, i]
                    bts = bts_i

                    invalid = (bts > 500.0) + (bts < 0.0)
                    bts[invalid] = np.nan
                    bts = bts.reshape(-1, self.n_freqs)

                    vas = self.angles[angles]
                    vas = vas.reshape(-1, 1)

                else:

                    sp = scene["surface_precip"].data
                    np.nan_to_num(sp, nan=-9999, copy=False)

                    # Interpolate brightness temperatures.
                    bts = scene["brightness_temperatures"].data[:, :]
                    np.nan_to_num(bts, nan=-9999, copy=False)

                    invalid = (bts > 500.0) + (bts < 0.0)
                    bts[invalid] = np.nan
                    bts = bts.reshape(-1, self.n_freqs)

                    vas = scene["earth_incidence_angle"].data[:, :, 0]
                    vas = vas.reshape(-1, 1)
                    np.nan_to_num(vas, nan=-9999, copy=False)

                # 2m temperature, values less than 0 must be missing.
                t2m = scene["two_meter_temperature"].data
                t2m = t2m.reshape(-1, 1)
                t2m[t2m < 0] = np.nan

                # Total precitable water, values less than 0 are missing.
                tcwv = scene["total_column_water_vapor"].data
                tcwv = tcwv.reshape(-1, 1)

                # Surface type
                st = scene["surface_type"].data
                n_types = 18
                st_1h = np.zeros((n_scans, n_pixels, n_types),
                                 dtype=np.float32)
                for i in range(n_types):
                    mask = st == i + 1
                    st_1h[mask] = 1.0
                st_1h = st_1h.reshape(-1, n_types)

                # Airmass type
                # Airmass type is defined slightly different from surface
                # type in that there is a 0 type.
                am = scene["airmass_type"].data
                n_types = 4
                am_1h = np.zeros((n_scans, n_pixels, n_types),
                                 dtype=np.float32)
                for i in range(n_types):
                    mask = np.maximum(am, 0) == i
                    am_1h[mask] = 1.0
                am_1h = am_1h.reshape(-1, n_types)

                x += [np.concatenate(
                    [bts, vas, t2m, tcwv, st_1h, am_1h],
                    axis=1
                )]

        x = np.concatenate(x, axis=0)
        return x

    def load_training_data_0d(self,
                              filename,
                              targets,
                              augment,
                              rng):

        with xr.open_dataset(filename) as dataset:

            #
            # Input data
            #

            n_samples = dataset.samples.size

            x = []
            y = {}

            n_angles = self.n_angles

            for i in range(n_samples):

                scene = dataset[{"samples": i}]
                source = dataset.source[i]

                if source == 0:

                    sp = scene["surface_precip"].data
                    np.nan_to_num(sp, nan=-9999, copy=False)
                    valid = np.all(sp >= 0, axis=-1)
                    n = valid.sum()

                    angles = rng.uniform(0.0, n_angles - 1, n)
                    inds_l = np.trunc(angles).astype(np.int32)
                    inds_r = np.ceil(angles).astype(np.int32)
                    f = angles - inds_r
                    inds = rng.random(angles.size) < 0.5
                    angles[inds] *= -1

                    # Interpolate brightness temperatures.
                    bts = scene["simulated_brightness_temperatures"]
                    np.nan_to_num(bts, nan=-9999, copy=False)
                    bts = _expand_pixels(bts.data[np.newaxis])[0][valid]

                    bts_i = np.zeros((n, self.n_freqs), dtype=np.float32)
                    for i in range(self.n_angles):
                        mask = inds_l == i
                        bts_i[mask] += f[mask, np.newaxis] * bts[mask, :, i]
                        mask = inds_r == i
                        bts_i[mask] += (1 - f[mask, np.newaxis]) * bts[mask, :, i]
                    bts = bts_i
                    vas = (f * self.angles[inds_l] + (1.0 - f)
                           * self.angles[inds_r])
                    vas = vas.reshape(-1, 1)

                    invalid = (bts > 500.0) + (bts < 0.0)
                    bts[invalid] = np.nan

                    # 2m temperature, values less than 0 must be missing.
                    t2m = scene["two_meter_temperature"].data[valid]
                    t2m = t2m[..., np.newaxis]
                    t2m[t2m < 0] = np.nan

                    # Total precitable water, values less than 0 are missing.
                    tcwv = scene["total_column_water_vapor"].data[valid]
                    tcwv = tcwv[..., np.newaxis]
                    # Surface type
                    st = scene["surface_type"].data[valid]

                    n_types = 18
                    st_1h = np.zeros((n, n_types), dtype=np.float32)
                    for i in range(n_types):
                        mask = st == i + 1
                        st_1h[mask] = 1.0
                    st_1h = st_1h.reshape(-1, n_types)
                    # Airmass type
                    # Airmass type is defined slightly different from surface
                    # type in that there is a 0 type.
                    am = scene["airmass_type"].data[valid]
                    n_types = 4
                    am_1h = np.zeros((n, n_types), dtype=np.float32)
                    for i in range(n_types):
                        mask = np.maximum(am, 0) == i
                        am_1h[mask] = 1.0
                    am_1h = am_1h.reshape(-1, n_types)

                    x += [np.concatenate(
                        [bts, vas, t2m, tcwv, st_1h, am_1h],
                        axis=1
                    )]

                    for t in targets:
                        if t in ["surface_precip", "convective_precip"]:
                            y_t = scene[t].data[valid]
                            y_i = np.zeros((n,), dtype=np.float32)
                            for j in range(self.n_angles):
                                mask = inds_l == i
                                y_i[mask] += f[mask] * y_t[mask, j]
                                mask = inds_r == i
                                y_i[mask] += (1 - f[mask]) * y_t[mask, j]
                            y_t = y_i
                        else:
                            y_t = scene[t].data
                            y_t = _expand_pixels(y_t[np.newaxis])[0][valid]

                        np.nan_to_num(y_t, copy=False, nan=-9999)
                        if t == "latent_heat":
                            y_t[y_t < -400] = -9999
                        else:
                            y_t[y_t < 0] = -9999
                        y.setdefault(t, []).append(y_t)

                else:

                    sp = scene["surface_precip"].data
                    np.nan_to_num(sp, nan=-9999, copy=False)
                    valid = np.all(sp >= 0, axis=-1)
                    n = valid.sum()

                    # Interpolate brightness temperatures.
                    bts = scene["brightness_temperatures"].data[valid, :]
                    vas = scene["earth_incidence_angle"].data[valid, ..., :1]

                    invalid = (bts > 500.0) + (bts < 0.0)
                    bts[invalid] = np.nan

                    # 2m temperature, values less than 0 must be missing.
                    t2m = scene["two_meter_temperature"].data[valid]
                    t2m = t2m[..., np.newaxis]
                    t2m[t2m < 0] = np.nan

                    # Total precitable water, values less than 0 are missing.
                    tcwv = scene["total_column_water_vapor"].data[valid]
                    tcwv = tcwv[..., np.newaxis]
                    # Surface type
                    st = scene["surface_type"].data[valid]

                    n_types = 18
                    st_1h = np.zeros((n, n_types), dtype=np.float32)
                    for i in range(n_types):
                        mask = st == i + 1
                        st_1h[mask] = 1.0
                    st_1h = st_1h.reshape(-1, n_types)

                    # Airmass type
                    # Airmass type is defined slightly different from surface
                    # type in that there is a 0 type.
                    am = scene["airmass_type"].data[valid]
                    n_types = 4
                    am_1h = np.zeros((n, n_types), dtype=np.float32)
                    for i in range(n_types):
                        mask = np.maximum(am, 0) == i
                        am_1h[mask] = 1.0
                    am_1h = am_1h.reshape(-1, n_types)

                    x += [np.concatenate(
                        [bts, vas, t2m, tcwv, st_1h, am_1h],
                        axis=1
                    )]

                    for t in targets:
                        if t in ["surface_precip", "convective_precip"]:
                            y_t = scene[t].data[valid, 0]
                        else:
                            y_t = scene[t].data
                            y_t = _expand_pixels(y_t[np.newaxis])[0][valid]

                        np.nan_to_num(y_t, copy=False, nan=-9999)
                        if t == "latent_heat":
                            y_t[y_t < -400] = -9999
                        else:
                            y_t[y_t < 0] = -9999
                        y.setdefault(t, []).append(y_t)

        x = np.concatenate(x, axis=0)
        for k in y:
            y[k] = np.concatenate(y[k], axis=0)

        return x, y

    def load_data_2d(self, filename, targets):
        pass

    def load_training_data_2d(self, filename, targets):
        pass

GMI = ConicalScanner(
    "GMI",
    15,
    "1C-R.GPM.GMI",
    "/pdata4/archive/GPM/1CR_GMI",
    "/pdata4/veljko/GMI2MRMS_match2019/db_mrms4GMI/",
    "GMI.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7",
    "gprof2020pp_GMI_L1C"
)

MHS_ANGLES = np.array([
    59.498, 53.311, 46.095, 39.222, 32.562, 26.043,
    19.619, 13.257,  6.934,  0.0
])
MHS = CrossTrackScanner(
    "MHS",
    MHS_ANGLES,
    5,
    "1C.*.MHS.",
    "/pdata4/archive/GPM/1C_NOAA19",
    "/pdata4/veljko/MHS2MRMS_match2019/monthly_2021/",
    "MHS.dbsatTb.??????{day}.??????.sim",
    "/qdata1/pbrown/dbaseV7/simV7x",
    "gprof2020pp_MHS_L1C" 
)
