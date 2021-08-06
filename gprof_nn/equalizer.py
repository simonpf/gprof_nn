"""
==================
gprof_nn.equalizer
==================

This module provides equalizer classes that are used to correct simulated
observations to matched real ones.
"""
from pathlib import Path
import pickle

import numpy as np
import xarray as xr

class Equalizer():
    """
    Base class for equalizers defining functions to load an save an
    equalizer.
    """
    @staticmethod
    def load(filename):
        """
        Load equalizer from file.

        Args:
            filename: The path to the file containing the saved equalizer.

        Returns:
            The loaded Equalizer object
        """
        with open(filename, "rb") as file:
            return pickle.load(file)


    def save(self, filename):
        """
        Store equalizer to file.

        Args:
            filename: The file to which to store the equalizer.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)


class QuantileEqualizer(Equalizer):
    """
    Equalizer that corrects observations by matching the quantiles of
    brightness temperature distributions of each channel.

    """
    def __init__(self,
                 sensor,
                 source_data,
                 target_data):
        """
        Args:
            sensor: Sensor object identifying the sensor from which the
                observations stem.
            source_data: Path to the NetCDF file containing the training data
                statistics.
            target_data: Path to the NetCDF file containing the observation data
                statistics.
        """
        super().__init__()

        self.sensor = sensor
        source_data = xr.load_dataset(source_data)
        target_data = xr.load_dataset(target_data)

        tb_bins = np.linspace(100, 310, 211)
        self.tb_bins = tb_bins

        angles = sensor.angles
        self.angle_bins = np.zeros(angles.size + 1)
        self.angle_bins[1:-1] = 0.5 * (angles[1:] + angles[:-1])
        self.angle_bins[0] = 2.0 * self.angle_bins[1] - self.angle_bins[2]
        self.angle_bins[-1] = 2.0 * self.angle_bins[-2] - self.angle_bins[-3]

        n_channels = source_data.channels.size
        tb_values = 0.5 * (tb_bins[1:] + tb_bins[:-1])
        if "angles" in source_data.dims:
            n_angles = source_data.angles.size
            self.has_angles = True
            self.biases = np.zeros((18, n_channels, n_angles, tb_bins.size - 1))

        else:
            self.has_angles = False
            self.biases = np.zeros(18, n_channels, tb_bins.size - 1)

        for i in range(18):
            for j in range(n_channels):
                if self.has_angles:
                    for k in range(n_angles):
                        tb_bins = source_data.brightness_temperature_bins.data
                        tbs_source = source_data.brightness_temperatures
                        cdf_source = tbs_source.data[i, j, k]
                        cdf_source /= cdf_source.sum(keepdims=True)
                        cdf_source = np.cumsum(cdf_source)
                        tbs_target = target_data.brightness_temperatures
                        cdf_target = tbs_target.data[i, j, k]
                        cdf_target /= cdf_target.sum(keepdims=True)
                        cdf_target = np.cumsum(cdf_target)
                        q_source = np.interp(tb_values, tb_bins, cdf_source)
                        offsets = np.interp(q_source, cdf_target, tb_bins)
                        offsets -= tb_values
                        self.biases[i, j, k] = offsets
                    for k in range(n_angles):
                        if np.all(~np.isfinite(self.biases[i, j, k])):
                            if k < n_angles // 2:
                                self.biases[i, j, k] = self.biases[i, j, k + 1]
                            else:
                                self.biases[i, j, k] = self.biases[i, j, k - 1]

                else:
                    tb_bins = source_data.brightness_temperature_bins.data
                    tbs_source = source_data.brightness_temperatures
                    cdf_source = tbs_source.data[i, j]
                    cdf_source /= cdf_source.sum(keepdims=True)
                    cdf_source = np.cumsum(cdf_source)
                    tbs_target = target_data.brightness_temperatures
                    cdf_target = tbs_target.data[i, j]
                    cdf_target /= cdf_target.sum(keepdims=True)
                    cdf_target = np.cumsum(cdf_target)
                    q_source = np.interp(tb_values, tb_bins, cdf_source)
                    offsets = np.interp(q_source, cdf_target, tb_bins)
                    offsets -= tb_values
                    self.biases[i, j] = offsets

    def __call__(self,
                 tbs,
                 eia,
                 surface_type,
                 noise_factor=0.0,
                 rng=None):
        """
        Apply correction to observations.

        Args:
            tbs: Array containing the brightness temperature to correct.
            eia: Array containing the corresponding earth incidence angles.
            surface_type: Array containing the corresponding surface type.
            noise_factor: Factor that the correction offset is multiplied
                with and used as standard deviation to add noise to the
                observations.
            rng: Optional random number generator to use to generate
                the noise.
        """
        if rng is None:
            rng = np.random.default_rng()

        tbs_c = tbs.copy()

        inds_st = surface_type - 1

        for i in range(self.sensor.n_chans):
            inds_tb = np.digitize(tbs[..., i], self.tb_bins[1:-1])
            if self.has_angles:
                inds_a = np.digitize(np.abs(eia), self.angle_bins[1:-1])
                b = self.biases[inds_st, i, inds_a, inds_tb]
                tbs_c[:, i] += b
            else:
                b = self.biases[inds_st, i, inds_tb]
                tbs_c[:, i] += b
            if noise_factor > 0.0:
                tbs_c[:, i] += (noise_factor * b *
                                rng.standard_normal(size=tbs_c.shape[0]))
        return tbs_c


class ConditionalEqualizer(Equalizer):
    """
    Equalizer that corrects observations by matching the quantiles of
    brightness temperature distributions of each channel.

    """
    def __init__(self,
                 sensor,
                 channel_index,
                 source_data,
                 target_data):
        """
        Args:
            sensor: Sensor object identifying the sensor from which the
                observations stem.
            source_data: Path to the NetCDF file containing the training data
                statistics.
            target_data: Path to the NetCDF file containing the observation data
                statistics.
        """
        super().__init__()

        self.channel_index = channel_index
        self.sensor = sensor
        source_data = xr.load_dataset(source_data)
        target_data = xr.load_dataset(target_data)

        tb_bins = np.linspace(100, 310, 211)
        self.tb_bins = tb_bins

        angles = sensor.angles
        self.angle_bins = np.zeros(angles.size + 1)
        self.angle_bins[1:-1] = 0.5 * (angles[1:] + angles[:-1])
        self.angle_bins[0] = 2.0 * self.angle_bins[1] - self.angle_bins[2]
        self.angle_bins[-1] = 2.0 * self.angle_bins[-2] - self.angle_bins[-3]

        n_channels = source_data.channels.size
        tb_values = 0.5 * (tb_bins[1:] + tb_bins[:-1])
        if "angles" in source_data.dims:
            n_angles = source_data.angles.size
            self.has_angles = True
            self.biases = np.zeros((18, n_channels, n_angles, tb_bins.size - 1))

        else:
            self.has_angles = False
            self.biases = np.zeros(18, n_channels, tb_bins.size - 1)

        for i in range(18):
            for j in range(n_channels):
                if self.has_angles:
                    for k in range(n_angles):
                        indices = {
                            "surface_type_bins": i,
                            "channels": j,
                            "angles": k
                        }
                        offsets = self._calculate_offsets(
                            tb_values,
                            source_data[indices]["conditional_brightness_temperatures"],
                            target_data[indices]["conditional_brightness_temperatures"],
                        )
                        self.biases[i, j, k] = offsets
                    for k in range(n_angles):
                        if np.all(~np.isfinite(self.biases[i, j, k])):
                            if k < n_angles // 2:
                                self.biases[i, j, k] = self.biases[i, j, k + 1]
                            else:
                                self.biases[i, j, k] = self.biases[i, j, k - 1]

                else:
                    indices = {
                        "surface_type_bins": i,
                        "channels": j,
                    }
                    offsets = self._calculate_offsets(
                        tb_values,
                        source_data[indices]["conditional_brightness_temperatures"],
                        target_data[indices]["conditional_brightness_temperatures"],
                    )
                    self.biases[i, j] = offsets

    def _calculate_offsets(
            self,
            tb_values,
            source_dist,
            target_dist
    ):
        x_source = source_dist.brightness_temperature_bins.data

        cdf_source = np.cumsum(source_dist.data.sum(-1))
        cdf_source /= cdf_source[-1]
        cdf_target = np.cumsum(target_dist.data.sum(-1))
        cdf_target /= cdf_target[-1]

        pdf_source = (source_dist.data /
                      source_dist.data.sum(axis=-1, keepdims=True))
        pdf_target = (target_dist.data /
                      target_dist.data.sum(axis=-1, keepdims=True))
        means_source = np.trapz(x_source * pdf_source, x=x_source, axis=-1)
        means_target = np.trapz(x_source * pdf_target, x=x_source, axis=-1)


        # Interpolate tb_values to closest non-nan conditional mean.
        inds = np.isfinite(pdf_source).all(-1)
        if inds.sum() == 0:
            offsets = np.zeros(tb_values.size)
            offsets[:] = np.nan
            return offsets
        means_source = np.interp(tb_values,
                                 x_source[inds],
                                 means_source[inds])
        assert np.all(np.isfinite(means_source))

        # Calculate quantile fractions  of conditional channels
        qs_source = np.interp(tb_values, x_source, cdf_source)

        inds = np.isfinite(pdf_target).all(-1)
        if inds.sum() == 0:
            offsets = np.zeros(tb_values.size)
            offsets[:] = np.nan
            return offsets
        means_target = np.interp(qs_source,
                                 cdf_target[inds],
                                 means_target[inds])
        return means_target - means_source


    def __call__(self,
                 tbs,
                 eia,
                 surface_type,
                 noise_factor=0.0,
                 rng=None):
        """
        Apply correction to observations.

        Args:
            tbs: Array containing the brightness temperature to correct.
            eia: Array containing the corresponding earth incidence angles.
            surface_type: Array containing the corresponding surface type.
            noise_factor: Factor that the correction offset is multiplied
                with and used as standard deviation to add noise to the
                observations.
            rng: Optional random number generator to use to generate
                the noise.
        """
        if rng is None:
            rng = np.random.default_rng()

        tbs_c = tbs.copy()

        inds_st = surface_type - 1
        inds_tb = np.digitize(tbs[..., self.channel_index], self.tb_bins[1:-1])

        for i in range(self.sensor.n_chans):
            if self.has_angles:
                inds_a = np.digitize(eia, self.angle_bins[1:-1])
                b = self.biases[inds_st, i, inds_a, inds_tb]
                tbs_c[:, i] += b
            else:
                b = self.biases[inds_st, i, inds_tb]
                tbs_c[:, i] += b
            if noise_factor > 0.0:
                tbs_c[:, i] += (noise_factor * b *
                                rng.standard_normal(size=tbs_c.shape[0]))
        return tbs_c
