"""
================
gprof_nn.sensors
================

This module defines different sensor classes and objects to represent the
 various sensors of the GPM constellation.
"""
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import xarray as xr

from gprof_nn.definitions import N_LAYERS
from gprof_nn.data.utils import expand_pixels
from gprof_nn.augmentation import (GMI_GEOMETRY,
                                   MHS_GEOMETRY,
                                   get_transformation_coordinates,
                                   extract_domain)


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


_BIAS_SCALES_GMI = 1.0 / np.cos(np.deg2rad(
    [52.8, 49.19, 49.19, 49.19, 49.19]
))


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

        self._bin_file_header = np.dtype(
            [
                ("satellite_code", "a5"),
                ("sensor", "a5"),
                ("frequencies", f"{n_freqs}f4"),
                ("nominal_eia", f"{n_freqs}f4")
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
                ("latent_heat", "f4", (N_LAYERS,))
            ]
        )
        self._l1c_file_prefix = l1c_prefix
        self._l1c_file_path = l1c_file_path

        self._mrms_file_path = mrms_file_path
        self._mrms_file_record = np.dtype([
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

        self._sim_file_pattern = sim_file_pattern
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
                     rng,
                     equalizer=None):

        pass

    def load_training_data_0d(self,
                              filename,
                              targets,
                              augment,
                              rng,
                              equalizer=None):
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
            tcwv = variables["total_column_water_vapor"][:][valid]
            tcwv = tcwv.reshape(-1, 1)
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
                y_t = expand_pixels(y_t)[valid]
                np.nan_to_num(y_t, copy=False, nan=-9999)
                if t == "latent_heat":
                    y_t[y_t < -400] = -9999
                else:
                    y_t[y_t < 0] = -9999
                y[t] = y_t
        return x, y

    def load_data_2d(self,
                     filename,
                     targets,
                     augment,
                     rng):
        with xr.open_dataset(self.filename) as dataset:
            #
            # Input data
            #

            # Brightness temperatures
            n = dataset.samples.size

            x = []
            y = {}

            for i in range(n):
                if augment:
                    p_x_o = rng.random()
                    p_x_i = rng.random()
                    p_y = rng.random()
                else:
                    p_x_o = 0.0
                    p_x_i = 0.0
                    p_y = 0.0

                coords = get_transformation_coordinates(GMI_GEOMETRY,
                                                        96, 128,
                                                        p_x_i, p_x_o, p_y)

                tbs = dataset["brightness_temperatures"][i][:]
                invalid = (tbs > 500.0) + (tbs < 0.0)
                tbs[invalid] = np.nan

                tbs = extract_domain(tbs, coords)
                tbs = np.transpose(tbs, (2, 0, 1))

                # Simulate missing high-frequency channels
                if augment:
                    r = rng.random()
                    n_p = rng.integers(10, 30)
                    if r > 0.80:
                        tbs[:, 10:15, :n_p] = np.nan

                t2m = dataset["two_meter_temperature"][i][:]
                t2m[t2m < 0] = np.nan
                t2m = extract_domain(t2m, coords)
                t2m = t2m[np.newaxis, ...]

                tcwv = dataset["total_column_water_vapor"][i][:]
                tcwv[tcwv < 0] = np.nan
                tcwv = extract_domain(tcwv, coords)
                tcwv = tcwv[np.newaxis, ...]

                st = dataset["surface_type"][i][:]
                st = extract_domain(st, coords, order=0)
                st_1h = np.zeros((18,) + st.shape, dtype=np.float32)
                for j in range(18):
                    st_1h[j, st == (j + 1)] = 1.0

                at = dataset["airmass_type"][i][:]
                at = extract_domain(at, coords=coords, order=0)
                at_1h = np.zeros((4,) + st.shape, dtype=np.float32)
                for j in range(4):
                    at_1h[j, np.maximum(at, 0) == j] = 1.0

                x += [np.concatenate([tbs, t2m, tcwv, st_1h, at_1h], axis=0)]

                #
                # Output data
                #

                for k in targets:
                    # Expand and reproject data.
                    y_k_r = expand_pixels(dataset[k][i][:][np.newaxis, ...])
                    y_k_r[y_k_r <= -999] = np.nan
                    y_k_r = extract_domain(
                        y_k_r[0], coords,
                    )

                    y_k = y.setdefault(k, [])
                    np.nan_to_num(y_k_r, copy=False, nan=-9999)

                    dims_sp = tuple(range(2))
                    dims_t = tuple(range(2, y_k_r.ndim))
                    y_k += [np.transpose(y_k_r, dims_t + dims_sp)]

                # Also flip data if requested.
                if self.augment:
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

    def load_training_data_2d(self,
                              dataset,
                              targets,
                              augment,
                              rng):
        if isinstance(dataset, (str, Path)):
            dataset = xr.open_dataset(dataset)

        #
        # Input data
        #

        # Brightness temperatures
        n = dataset.samples.size

        x = []
        y = {}

        for i in range(n):
            if augment:
                p_x_o = rng.random()
                p_x_i = rng.random()
                p_y = rng.random()
            else:
                p_x_o = 0.0
                p_x_i = 0.0
                p_y = 0.0

            coords = get_transformation_coordinates(GMI_GEOMETRY,
                                                    96, 128,
                                                    p_x_i, p_x_o, p_y)

            tbs = dataset["brightness_temperatures"][i].data
            invalid = (tbs > 500.0) + (tbs < 0.0)
            tbs[invalid] = np.nan
            tbs = extract_domain(tbs, coords)
            tbs = np.transpose(tbs, (2, 0, 1))


            # Simulate missing high-frequency channels
            if augment:
                r = rng.random()
                n_p = rng.integers(10, 30)
                if r > 0.80:
                    tbs[:, 10:15, :n_p] = np.nan

            t2m = dataset["two_meter_temperature"][i].data
            t2m[t2m < 0] = np.nan
            t2m = extract_domain(t2m, coords)
            t2m = t2m[np.newaxis, ...]

            tcwv = dataset["total_column_water_vapor"][i].data
            tcwv[tcwv < 0] = np.nan
            tcwv = extract_domain(tcwv, coords)
            tcwv = tcwv[np.newaxis, ...]

            st = dataset["surface_type"][i].data
            st = extract_domain(st, coords, order=0)
            st_1h = np.zeros((18,) + st.shape, dtype=np.float32)
            for j in range(18):
                st_1h[j, st == (j + 1)] = 1.0

            at = dataset["airmass_type"][i].data
            at = extract_domain(at, coords, order=0)
            at_1h = np.zeros((4,) + st.shape, dtype=np.float32)
            for j in range(4):
                at_1h[j, np.maximum(at, 0) == j] = 1.0

            x += [np.concatenate([tbs, t2m, tcwv, st_1h, at_1h], axis=0)]

            #
            # Output data
            #

            for k in targets:
                # Expand and reproject data.
                y_k_r = expand_pixels(dataset[k][i].data[np.newaxis, ...])
                y_k_r[y_k_r <= -999] = np.nan
                y_k_r = extract_domain(
                    y_k_r[0], coords,
                )

                y_k = y.setdefault(k, [])
                np.nan_to_num(y_k_r, copy=False, nan=-9999)

                dims_sp = tuple(range(2))
                dims_t = tuple(range(2, y_k_r.ndim))
                y_k += [np.transpose(y_k_r, dims_t + dims_sp)]

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
    def __init__(self,
                 name,
                 angles,
                 nedt,
                 n_freqs,
                 l1c_prefix,
                 l1c_file_path,
                 mrms_file_path,
                 sim_file_pattern,
                 sim_file_path,
                 preprocessor):
        self._name = name
        self._angles = angles
        self.nedt = nedt
        n_angles = angles.size
        self._n_freqs = n_freqs

        self._bin_file_header = np.dtype(
            [
                ("satellite_code", "a5"),
                ("sensor", "a5"),
                ("frequencies", "f4", (n_freqs,)),
                ("nominal_eia", "f4", (n_angles,))
            ]
        )
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
                    ("latent_heat", "f4", (N_LAYERS,))
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
                    (
                        "brightness_temperatures",
                        "f4",
                        (self.n_angles, self.n_freqs)
                    ),
                    ("rain_water_path", np.float32),
                    ("cloud_water_path", np.float32),
                    ("ice_water_path", np.float32),
                    ("total_column_water_vapor", np.float32),
                    ("two_meter_temperature", np.float32),
                    ("rain_water_content", "f4", (N_LAYERS,)),
                    ("cloud_water_content", "f4", (N_LAYERS,)),
                    ("snow_water_content", "f4", (N_LAYERS,)),
                    ("latent_heat", "f4", (N_LAYERS,))
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
            angles_sim = self.angles
            angles_sim[0] = 60.0
            angles_sim[-1] = 0.0

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
                    f = inds_r - angles

                    # Interpolate brightness temperatures.
                    bts = scene["simulated_brightness_temperatures"]
                    bts = expand_pixels(bts.data[np.newaxis])[0]
                    np.nan_to_num(bts, nan=-9999, copy=False)

                    bts_i = np.zeros((n_scans, n_pixels, self.n_freqs),
                                     dtype=np.float32)
                    for j in range(self.n_angles):
                        mask = inds_l == j
                        bts_i[mask] += f[mask, np.newaxis] * bts[mask, j]
                        mask = inds_r == j
                        bts_i[mask] += (1 - f[mask, np.newaxis]) * bts[mask, j]
                    bts = bts_i

                    invalid = (bts > 500.0) + (bts < 0.0)
                    bts[invalid] = np.nan
                    bts = bts.reshape(-1, self.n_freqs)

                    vas = angles_sim[angles]
                    vas = vas.reshape(-1, 1)

                    bias_scales = calculate_bias_scaling(vas)
                    bias = scene["brightness_temperature_biases"]
                    bias = expand_pixels(bias.data[np.newaxis])[0]
                    bias = bias.reshape(-1, self.n_freqs)

                    mask = bias > -1000
                    #bias = bias_scales * bias
                    bts[mask] = bts[mask] - bias[mask]
                    bts += np.random.normal(size=bts.shape) * self.nedt[np.newaxis, ...]

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


                # 2m temperature, values less than 0 must be missing.
                t2m = scene["two_meter_temperature"].data
                t2m = t2m.reshape(-1, 1)
                np.nan_to_num(t2m, nan=-9999, copy=False)
                t2m[t2m < 0] = np.nan

                # Total precitable water, values less than 0 are missing.
                tcwv = scene["total_column_water_vapor"].data
                tcwv = tcwv.reshape(-1, 1)

                # Surface type
                st = np.maximum(scene["surface_type"].data, 1)
                n_types = 18
                st_1h = np.zeros((n_scans, n_pixels, n_types),
                                 dtype=np.float32)
                for j in range(n_types):
                    mask = st == j + 1
                    st_1h[mask, j] = 1.0
                st_1h = st_1h.reshape(-1, n_types)

                # Airmass type
                # Airmass type is defined slightly different from surface
                # type in that there is a 0 type.
                am = scene["airmass_type"].data
                n_types = 4
                am_1h = np.zeros((n_scans, n_pixels, n_types),
                                 dtype=np.float32)
                for j in range(n_types):
                    mask = np.maximum(am, 0) == j
                    am_1h[mask, j] = 1.0
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
                              rng,
                              equalizer=None):

        angles_sim = self.angles
        angles_sim[0] = 60.0
        angles_sim[-1] = 0.0


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

                    bts = scene["simulated_brightness_temperatures"]
                    bts = expand_pixels(bts.data[np.newaxis])[0]
                    np.nan_to_num(bts, nan=-9999, copy=False)

                    sp = scene["surface_precip"].data
                    st = scene["surface_type"].data
                    np.nan_to_num(sp, nan=-9999, copy=False)
                    valid = np.all(sp >= 0, axis=-1) * (st > 0)
                    n = valid.sum()

                    angles = rng.uniform(0.0, n_angles - 1, n)
                    inds_l = np.trunc(angles).astype(np.int32)
                    inds_r = np.ceil(angles).astype(np.int32)
                    f = inds_r - angles

                    # Interpolate brightness temperatures.
                    bts = bts[valid]
                    bts_i = np.zeros((n, self.n_freqs), dtype=np.float32)
                    for j in range(self.n_angles):
                        mask = inds_l == j
                        bts_i[mask] += f[mask, np.newaxis] * bts[mask, j]
                        mask = inds_r == j
                        bts_i[mask] += (1 - f[mask, np.newaxis]) * bts[mask, j]
                    bts = bts_i

                    vas = (f * angles_sim[inds_l] + (1.0 - f)
                           * angles_sim[inds_r])
                    vas = vas.reshape(-1, 1)

                    # Also sample negative viewing angles
                    inds = rng.random(vas.size) < 0.5
                    vas[inds] *= -1

                    # Calculate corrected biases.
                    bias_scales = calculate_bias_scaling(vas)
                    bias = scene["brightness_temperature_biases"]
                    bias = expand_pixels(bias.data[np.newaxis])[0][valid]

                    mask = bias > -1000
                    #bias = bias_scales * bias
                    #bts[mask] = bts[mask] - bias[mask]

                    #bts += (rng.standard_normal(size=bts.shape)
                    #        * self.nedt[np.newaxis, ...])

                    invalid = (bts > 500.0) + (bts < 0.0)
                    bts[invalid] = np.nan

                    # 2m temperature, values less than 0 must be missing.
                    t2m = scene["two_meter_temperature"].data[valid]
                    t2m = t2m[..., np.newaxis]
                    t2m[t2m < 0] = np.nan

                    # Total precitable water, values less than 0 are missing.
                    tcwv = scene["total_column_water_vapor"].data[valid]
                    tcwv = tcwv[..., np.newaxis]
                    tcwv[tcwv < 0] = np.nan

                    # Surface type
                    st = scene["surface_type"].data[valid]

                    n_types = 18
                    st_1h = np.zeros((n, n_types), dtype=np.float32)
                    for i in range(n_types):
                        mask = st == i + 1
                        st_1h[mask, i] = 1.0
                    st_1h = st_1h.reshape(-1, n_types)
                    # Airmass type
                    # Airmass type is defined slightly different from surface
                    # type in that there is a 0 type.
                    am = scene["airmass_type"].data[valid]
                    n_types = 4
                    am_1h = np.zeros((n, n_types), dtype=np.float32)
                    for i in range(n_types):
                        mask = np.maximum(am, 0) == i
                        am_1h[mask, i] = 1.0
                    am_1h = am_1h.reshape(-1, n_types)

                    if equalizer is not None:
                        bts = equalizer(bts, vas[:, 0], st,
                                        noise_factor=1.0,
                                        rng=rng)

                    x += [np.concatenate(
                        [bts, vas, t2m, tcwv, st_1h, am_1h],
                        axis=1
                    )]

                    for t in targets:
                        # Interpolate surface and convective precip.
                        if t in ["surface_precip", "convective_precip"]:
                            y_t = scene[t].data[valid]
                            y_i = np.zeros((n,), dtype=np.float32)
                            for j in range(self.n_angles):
                                mask = inds_l == j
                                y_i[mask] += f[mask] * y_t[mask, j]
                                mask = inds_r == j
                                y_i[mask] += (1 - f[mask]) * y_t[mask, j]
                            y_t = y_i
                        else:
                            y_t = scene[t].data
                            y_t = expand_pixels(y_t[np.newaxis])[0][valid]

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

                    bts = scene["brightness_temperatures"].data[valid, :]
                    vas = scene["earth_incidence_angle"].data[valid, ..., :1]
                    vas[vas < -100] = np.nan

                    invalid = (bts > 500.0) + (bts < 0.0)
                    bts[invalid] = np.nan

                    # 2m temperature, values less than 0 must be missing.
                    t2m = scene["two_meter_temperature"].data[valid]
                    t2m = t2m[..., np.newaxis]
                    t2m[t2m < 0] = np.nan

                    # Total precitable water, values less than 0 are missing.
                    tcwv = scene["total_column_water_vapor"].data[valid]
                    tcwv = tcwv[..., np.newaxis]
                    tcwv[tcwv < 0] = np.nan

                    # Surface type
                    st = scene["surface_type"].data[valid]
                    n_types = 18
                    st_1h = np.zeros((n, n_types), dtype=np.float32)
                    for j in range(n_types):
                        mask = st == j + 1
                        st_1h[mask, j] = 1.0
                    st_1h = st_1h.reshape(-1, n_types)

                    # Airmass type
                    # Airmass type is defined slightly different from surface
                    # type in that there is a 0 type.
                    am = scene["airmass_type"].data[valid]
                    n_types = 4
                    am_1h = np.zeros((n, n_types), dtype=np.float32)
                    for j in range(n_types):
                        mask = np.maximum(am, 0) == j
                        am_1h[mask, j] = 1.0
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
                            y_t = expand_pixels(y_t[np.newaxis])[0][valid]

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

    def load_data_2d(self,
                     filename,
                     targets,
                     augment,
                     rng):
        pass



    def _load_training_data_2d_sim(self,
                                   data,
                                   targets,
                                   augment,
                                   rng):
        # Brightness temperatures
        if augment:
            p_x_o = rng.random()
            p_x_i = rng.random()
            p_y = rng.random()
        else:
            p_x_o = 0.5
            p_x_i = 0.5
            p_y = 0.5

        height = 128
        width = 32

        coords = get_transformation_coordinates(MHS_GEOMETRY,
                                                width, height,
                                                p_x_i, p_x_o, p_y)

        weights = MHS_GEOMETRY.get_interpolation_weights(self.angles)
        weights = np.repeat(weights[np.newaxis, ...], height, axis=0)
        center = MHS_GEOMETRY.get_window_center(p_x_o, width, height)
        j_start = int(center[1, 0, 0] - width // 2)
        j_end = int(center[1, 0, 0] + width // 2)
        weights = weights[:, j_start:j_end, :, np.newaxis]

        tbs = data["simulated_brightness_temperatures"].data
        invalid = (tbs > 500.0) + (tbs < 0.0)
        tbs[invalid] = np.nan
        tbs = extract_domain(tbs, coords)

        biases = data["brightness_temperature_biases"].data
        invalid = (biases < -999)
        biases[invalid] = np.nan
        biases = extract_domain(biases, coords)
        biases = np.expand_dims(biases, axis=-2)

        tbs = tbs - biases
        tbs = np.sum(tbs * weights, axis=-2)
        tbs = np.transpose(tbs, (2, 0, 1))

        # Simulate missing high-frequency channels
        if augment:
            r = rng.random()
            n_p = rng.integers(10, 30)
            if r > 0.80:
                tbs[:, 10:15, :n_p] = np.nan

        t2m = data["two_meter_temperature"].data
        t2m[t2m < 0] = np.nan
        t2m = extract_domain(t2m, coords)
        t2m = t2m[np.newaxis, ...]

        tcwv = data["total_column_water_vapor"].data
        tcwv[tcwv < 0] = np.nan
        tcwv = extract_domain(tcwv, coords)
        tcwv = tcwv[np.newaxis, ...]

        st = data["surface_type"].data
        st = extract_domain(st, coords, order=0)
        st_1h = np.zeros((18,) + st.shape, dtype=np.float32)
        for j in range(18):
            st_1h[j, st == (j + 1)] = 1.0

        at = data["airmass_type"].data
        at = extract_domain(at, coords, order=0)
        at_1h = np.zeros((4,) + st.shape, dtype=np.float32)
        for j in range(4):
            at_1h[j, np.maximum(at, 0) == j] = 1.0

        x = np.concatenate([tbs, t2m, tcwv, st_1h, at_1h], axis=0)

        #
        # Output data
        #

        y = {}

        for k in targets:
            # Expand and reproject data.
            y_k_r = expand_pixels(data[k].data[np.newaxis, ...])
            y_k_r[y_k_r < -999] = np.nan
            y_k_r = extract_domain(
                y_k_r[0], coords,
            )
            np.nan_to_num(y_k_r, copy=False, nan=-9999)

            dims_sp = tuple(range(2))
            dims_t = tuple(range(2, y_k_r.ndim))
            y[k] = np.transpose(y_k_r, dims_t + dims_sp)

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

    def _load_training_data_2d_other(self,
                                     data,
                                     targets,
                                     augment,
                                     rng):
        # Brightness temperatures
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

        tbs = data["brightness_temperatures"].data[i_start:i_end, j_start:j_end]
        tbs = np.transpose(tbs, (2, 0, 1))
        invalid = (tbs > 500.0) + (tbs < 0.0)
        tbs[invalid] = np.nan

        t2m = data["two_meter_temperature"].data[i_start:i_end, j_start:j_end]
        t2m = t2m[np.newaxis, ...]
        t2m[t2m < 0] = np.nan

        tcwv = data["total_column_water_vapor"].data[i_start:i_end, j_start:j_end]
        tcwv = tcwv[np.newaxis, ...]
        tcwv[tcwv < 0] = np.nan

        st = data["surface_type"].data[i_start:i_end, j_start:j_end]
        st_1h = np.zeros((18,) + st.shape, dtype=np.float32)
        for j in range(18):
            st_1h[j, st == (j + 1)] = 1.0

        at = data["airmass_type"].data[i_start:i_end, j_start:j_end]
        at_1h = np.zeros((4,) + st.shape, dtype=np.float32)
        for j in range(4):
            at_1h[j, np.maximum(at, 0) == j] = 1.0

        x = np.concatenate([tbs, t2m, tcwv, st_1h, at_1h], axis=0)

        #
        # Output data
        #
        y = {}

        for k in targets:
            # Expand and reproject data.
            y_k_r = expand_pixels(data[k].data[np.newaxis, ...])
            y_k_r = y_k_r[0, i_start:i_end, j_start:j_end]
            y_k_r[y_k_r < -999] = np.nan
            np.nan_to_num(y_k_r, copy=False, nan=-9999)

            dims_sp = tuple(range(2))
            dims_t = tuple(range(2, y_k_r.ndim))
            y[k] = np.transpose(y_k_r, dims_t + dims_sp)

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

    def load_training_data_2d(self,
                              dataset,
                              targets,
                              augment,
                              rng):

        if isinstance(dataset, (str, Path)):
            dataset = xr.open_dataset(dataset)

        # Brightness temperatures
        n = dataset.samples.size
        sources = dataset.source

        x = []
        y = []

        for i in range(n):
            if sources[i] == 0:
                x_i, y_i = self._load_training_data_2d_sim(dataset[{"samples": i}],
                                                           targets,
                                                           augment,
                                                           rng)
            else:
                x_i, y_i = self._load_training_data_2d_other(dataset[{"samples": i}],
                                                             targets,
                                                             augment,
                                                             rng)
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
    "/qdata1/pbrown/dbaseV7/simV7",
    "gprof2020pp_GMI_L1C"
)

MHS_ANGLES = np.array([
    59.798, 53.311, 46.095, 39.222, 32.562, 26.043,
    19.619, 13.257,  6.934,  0.0
])

MHS_NEDT = np.array([
    5.0, 5.0, 5.0, 5.0, 5.0
])


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
    "gprof2020pp_MHS_L1C"
)
