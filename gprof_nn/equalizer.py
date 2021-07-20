"""
==================
gprof_nn.equalizer
==================

This module provides equalizer classes that are used to correct simulated
observations to matched real ones.
"""
from pathlib import Path

import numpy as np
import xarray as xr

class QuantileEqualizer():
    def __init__(self,
                 sensor,
                 source_data,
                 target_data):

        self.sensor = sensor
        source_data = xr.load_dataset(source_data)
        target_data = xr.load_dataset(target_data)

        tb_bins = np.linspace(100, 400, 31)
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
                    self.biases[i, j, k] = offsets


    def __call__(self, tbs, eia, surface_type):

        tbs_c = tbs.copy()

        inds_st = surface_type - 1

        for i in range(self.n_freqs):
            if self.has_angles:
                inds_a = np.digitize(self.angle_bins[1:-1], eia)
                inds_tb = np.digitize(self.tb_bins[1:-1], tbs[..., i])
                tbs_c[:, i] += self.biases[inds_st, inds_a, inds_tb]
            else:
                inds_tb = np.digitize(self.tb_bins[1:-1], tbs[..., i])
                tbs_c[:, i] += self.biases[inds_st, inds_tb]

        return tbs_c



