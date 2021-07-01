"""
Tests for the data loading function of the sensor classes.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn import sensors

TEST_FILE_MHS = "gprof_nn_mhs_era5_5.nc"


def test_load_data_mhs():

    path = Path(__file__).parent
    input_file = path / "data" / TEST_FILE_MHS
    input_data = xr.open_dataset(input_file)

    sensor = sensors.MHS
    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x = sensor.load_data_0d(input_file)

    # Make sure all observation angles are withing expected limits.
    valid = np.abs(x[:, 5] > -1000)
    assert np.all(np.abs(x[valid, 5]) <= sensor.angles[0] + 1.0)
    assert np.all(np.abs(x[valid, 5]) >= sensor.angles[-1])

    st = x[:, 8:26] > 0
    np.all(np.isclose(st.sum(axis=1), 1.0))

    at = x[:, 26:30] > 0
    np.all(np.isclose(at.sum(axis=1), 1.0))


def test_load_training_data_mhs(tmp_path):

    path = Path(__file__).parent
    input_file = path / "data" / TEST_FILE_MHS
    input_data = xr.open_dataset(input_file)

    for i in range(10):
        input_data.simulated_brightness_temperatures.data[:, :, :, i] = i
        input_data.brightness_temperatures.data[:, :, :, i] = i
        input_data.surface_precip[:, :, :, i] = i

    input_data.to_netcdf(tmp_path / "te")

    sensor = sensors.MHS
    sensor.angles = np.arange(10)

    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x, y = sensor.load_training_data_0d(input_file,
                                        targets,
                                        False,
                                        rng)

    # Assert all targets are loaded
    sp_ref = input_data.surface_precip.data
    sp_ref = sp_ref[np.all(sp_ref >= 0, axis=-1), :]

    assert all(t in y for t in targets)

    sp = y["surface_precip"]
    assert np.all(sp_ref.max(axis=-1) >= sp)
    assert np.all(sp_ref.min(axis=-1) <= sp)


    # Make sure all observation angles are withing expected limits.
    assert np.all(np.abs(x[:, 5]) <= sensor.angles[0] + 1.0)
    assert np.all(np.abs(x[:, 5]) >= sensor.angles[-1])

    st = x[:, 8:26] > 0
    np.all(np.isclose(st.sum(axis=1), 1.0))

    at = x[:, 26:30] > 0
    np.all(np.isclose(at.sum(axis=1), 1.0))

def test_load_training_data_mhs():

    path = Path(__file__).parent
    input_file = path / "data" / TEST_FILE_MHS
    input_data = xr.open_dataset(input_file)

    sensor = sensors.MHS

    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x, y = sensor.load_training_data_0d(input_file,
                                        targets,
                                        False,
                                        rng)

    # Assert all targets are loaded
    sp_ref = input_data.surface_precip.data
    sp_ref = sp_ref[np.all(sp_ref >= 0, axis=-1), :]

    assert all(t in y for t in targets)

    sp = y["surface_precip"]
    assert np.all(sp_ref.max(axis=-1) >= sp)
    assert np.all(sp_ref.min(axis=-1) <= sp)


    # Make sure all observation angles are withing expected limits.
    assert np.all(np.abs(x[:, 5]) <= sensor.angles[0] + 1.0)
    assert np.all(np.abs(x[:, 5]) >= sensor.angles[-1])

    st = x[:, 8:26] > 0
    np.all(np.isclose(st.sum(axis=1), 1.0))

    at = x[:, 26:30] > 0
    np.all(np.isclose(at.sum(axis=1), 1.0))


def test_interpolation_mhs(tmp_path):
    """
    Ensure that interpolation of surface precipitation
    works.
    """
    path = Path(__file__).parent
    input_file = path / "data" / TEST_FILE_MHS
    input_data = xr.open_dataset(input_file)

    input_data = input_data.sel({"samples": input_data.source == 0})

    for i in range(10):
        input_data.surface_precip[:, :, :, i] = i

    input_data.to_netcdf(tmp_path / "test.nc")

    sensor = sensors.MHS
    sensor._angles = np.arange(10)

    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x, y = sensor.load_training_data_0d(tmp_path / "test.nc",
                                        targets,
                                        False,
                                        rng)

    # Assert all targets are loaded
    sp = y["surface_precip"]
    va = np.abs(x[:, 5])

    inds = (sp > 1.0) * (sp < 8.0)
    assert np.all(np.isclose(va[inds], sp[inds]))
