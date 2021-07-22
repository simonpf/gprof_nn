"""
Tests for the 'gprof_nn.equalizer' module.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn.equalizer import (ConditionalEqualizer,
                                QuantileEqualizer)
from gprof_nn import sensors

def test_quantile_equalizer_save_load(tmpdir):
    """
    Tests saving and loading of equalizers.
    """
    data_path = Path(__file__).parent / "data" / "statistics"
    source_file = data_path / "training_data_statistics_mhs.nc"
    target_file = data_path / "observation_statistics_mhs.nc"
    equalizer = QuantileEqualizer(sensors.MHS,
                                  source_file,
                                  target_file)
    equalizer.save(tmpdir / "test.pckl")
    equalizer_2 = QuantileEqualizer.load(tmpdir / "test.pckl")
    assert np.all(np.isclose(equalizer.biases ,
                             equalizer_2.biases))


def test_quantile_equalizer_mhs():
    """
    Ensure that QuantileEqualizer successfully equalizes the quantiles
    of the distribution.
    """
    data_path = Path(__file__).parent / "data" / "statistics"
    source_file = data_path / "training_data_statistics_mhs.nc"
    target_file = data_path / "observation_statistics_mhs.nc"

    equalizer = QuantileEqualizer(sensors.MHS,
                                  source_file,
                                  target_file)

    tb_bins = np.linspace(0, 400, 81)
    tb_values = 0.5 * (tb_bins[1:] + tb_bins[:-1])

    source_data = xr.load_dataset(source_file)
    target_data = xr.load_dataset(target_file)
    tb_bins = source_data.brightness_temperature_bins.data
    tbs_source = source_data.brightness_temperatures.data
    cdf_source = tbs_source[0, 1, 1]
    cdf_source /= cdf_source.sum(keepdims=True)
    cdf_source = np.cumsum(cdf_source)
    tbs_target = target_data.brightness_temperatures.data
    cdf_target = tbs_target[0, 1, 1]
    cdf_target /= cdf_target.sum(keepdims=True)
    cdf_target = np.cumsum(cdf_target)

    qs_source = np.interp(tb_values, tb_bins, cdf_source)
    tbs_corrected = equalizer(tb_values.reshape(-1, 1).repeat(5, axis=1),
                              sensors.MHS.angles[0],
                              1,
                              noise_factor=0.0)
    qs_target = np.interp(tbs_corrected[:, 1], tb_bins, cdf_target)

    assert np.all(np.isfinite(equalizer.biases))
    assert np.all(np.isclose(qs_source, qs_target))

    tbs_corrected = equalizer(tb_values.reshape(-1, 1).repeat(5, axis=1),
                              sensors.MHS.angles[0],
                              1,
                              noise_factor=1.0)
    qs_target = np.interp(tbs_corrected[:, 1], tb_bins, cdf_target)
    assert ~np.all(np.isclose(qs_source, qs_target))


def test_conditional_equalizer_mhs():
    """
    Ensure that QuantileEqualizer successfully equalizes the quantiles
    of the distribution.
    """
    data_path = Path(__file__).parent / "data" / "statistics"
    source_file = data_path / "training_data_statistics_mhs.nc"
    target_file = data_path / "observation_statistics_mhs.nc"

    equalizer = ConditionalEqualizer(sensors.MHS,
                                     1,
                                     source_file,
                                     target_file)

    tb_bins = np.linspace(0, 400, 81)
    tb_values = 0.5 * (tb_bins[1:] + tb_bins[:-1])

    source_data = xr.load_dataset(source_file)
    target_data = xr.load_dataset(target_file)
    tb_bins = source_data.brightness_temperature_bins.data
    tbs_source = source_data.brightness_temperatures.data
    cdf_source = tbs_source[0, 1, 1]
    cdf_source /= cdf_source.sum(keepdims=True)
    cdf_source = np.cumsum(cdf_source)
    tbs_target = target_data.brightness_temperatures.data
    cdf_target = tbs_target[0, 1, 1]
    cdf_target /= cdf_target.sum(keepdims=True)
    cdf_target = np.cumsum(cdf_target)

    qs_source = np.interp(tb_values, tb_bins, cdf_source)
    tbs_corrected = equalizer(tb_values.reshape(-1, 1).repeat(5, axis=1),
                              sensors.MHS.angles[1],
                              1,
                              noise_factor=0.0)
    qs_target = np.interp(tbs_corrected[:, 1], tb_bins, cdf_target)

    print(qs_source[20:-20], qs_target[20:-20])

    inds = (qs_source > 1e-5) * (qs_source < 0.9999)

    assert np.all(np.isfinite(equalizer.biases))
    assert np.all(np.isclose(qs_source[inds], qs_target[inds]))
    tbs_corrected = equalizer(tb_values.reshape(-1, 1).repeat(5, axis=1),
                              sensors.MHS.angles[1],
                              1,
                              noise_factor=1.0)
    qs_target = np.interp(tbs_corrected[:, 1], tb_bins, cdf_target)
    assert ~np.all(np.isclose(qs_source, qs_target))
