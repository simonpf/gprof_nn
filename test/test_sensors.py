"""
Tests for the data loading function of the sensor classes.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.training_data import decompress_and_load

TEST_FILE_GMI = Path("gmi") / "gprof_nn_gmi_era5.nc.gz"
TEST_FILE_MHS = Path("mhs") / "gprof_nn_mhs_era5.nc.gz"
TEST_FILE_TMI = Path("tmi") / "gprof_nn_tmi_era5.nc.gz"


DATA_PATH = get_test_data_path()


def test_calculate_smoothing_kernel():
    """
    Ensure that 'calculate_smoothing_kernel' returns kernels with the right
    FWHM.
    """
    k = sensors.calculate_smoothing_kernel(1, 1, 2, 2, 11)
    c = k[5, 5]
    c2 = k[5, 3]
    assert np.isclose(c2 / c, 0.5)
    c = k[5, 5]
    c2 = k[3, 5]
    assert np.isclose(c2 / c, 0.5)


def test_calculate_smoothing_kernels():
    """
    Ensure that 'calculate_smoothing_kernels' returns one kernel for each
    viewing angle and that the kernels have the expected shape.
    """
    kernels = sensors.calculate_smoothing_kernels(sensors.MHS)
    assert len(kernels) == sensors.MHS.n_angles
    assert kernels[0].shape == (11, 11)


def test_smooth_gmi_field():
    """
    Ensure that smoothing a GMI field inserts the smoothed field along the
    right axis.
    """
    field = np.zeros((32, 32, 4))
    field[15, 15] = 1.0

    kernels = sensors.calculate_smoothing_kernels(sensors.MHS)
    kernels = [kernels[0]] * 10
    field_s = sensors.smooth_gmi_field(field, kernels)

    assert field_s.shape[2] == sensors.MHS.n_angles
    assert np.all(np.isclose(field_s[:, :, 0], field_s[:, :, 1], atol=1e-3))


def test_load_training_data_1d_gmi():
    """
    Ensure that loading the training data for GMI produces realistic
    values.
    """
    input_file = DATA_PATH / TEST_FILE_GMI
    input_data = decompress_and_load(input_file)

    sensor = sensors.GMI

    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x, y = sensor.load_training_data_1d(input_data, targets, False, rng)

    # TB ranges
    assert np.all(x[:, :5] > 20)
    assert np.all(x[:, :5] < 500)
    # Two-meter temperature
    assert np.all(x[:, 15] > 200)
    assert np.all(x[:, 15] < 400)
    # TCWV
    assert np.all(x[:, 16] >= 0)
    assert np.all(x[:, 16] < 100)

    # Assert all targets are loaded
    assert all(t in y for t in targets)

    # Ensure that loaded surface precip is within the range given
    # of the surface precip observed for the different angles.
    sp_ref = input_data.surface_precip.data
    mask = sp_ref >= 0
    sp_ref = sp_ref[mask]
    sp = y["surface_precip"]
    mask = np.isfinite(sp)
    assert np.all(sp_ref[mask].max(axis=-1) >= sp[mask])
    assert np.all(sp_ref[mask].min(axis=-1) <= sp[mask])

    st = x[:, -22:-4] > 0
    assert np.all(np.isclose(st.sum(axis=1), 1.0))

    at = x[:, -4:] > 0
    assert np.all(np.isclose(at.sum(axis=1), 1.0))


def test_load_training_data_3d_gmi():
    """
    Ensure that loading the training data for GMI produces realistic
    values.
    """
    input_file = DATA_PATH / TEST_FILE_GMI
    input_data = decompress_and_load(input_file)

    sensor = sensors.GMI

    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x, y = sensor.load_training_data_3d(input_data, targets, False, rng)

    # TB ranges
    assert np.all(x[:, :5] > 20)
    assert np.all(x[:, :5] < 500)
    # Two-meter temperature
    assert np.all(x[:, 15] > 200)
    assert np.all(x[:, 15] < 400)
    # TCWV
    assert np.all(x[:, 16] >= 0)
    assert np.all(x[:, 16] < 100)

    # Assert all targets are loaded
    assert all(t in y for t in targets)

    # Ensure that loaded surface precip is within the range given
    # of the surface precip observed for the different angles.
    sp = y["surface_precip"]
    mask = sp >= 0
    assert np.all(sp[mask] < 500)

    tbs = x[:, :15]
    valid = tbs >= 0
    assert np.all((tbs[valid] > 20) * (tbs[valid] < 400))

    st = x[:, -22:-4] > 0
    assert np.all(np.isclose(st.sum(axis=1), 1.0))

    at = x[:, -4:] > 0
    assert np.all(np.isclose(at.sum(axis=1), 1.0))


def test_load_training_data_1d_mhs():
    """
    Ensure that loading the training data for MHS produces realistic
    values.
    """
    input_file = DATA_PATH / TEST_FILE_MHS
    input_data = decompress_and_load(input_file)

    sensor = sensors.MHS

    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x, y = sensor.load_training_data_1d(input_data, targets, False, rng)

    # TB ranges
    assert np.all(x[:, :5] > 20)
    assert np.all(x[:, :5] < 500)
    # Earth incidence angles
    assert np.all(x[:, 5] > -65)
    assert np.all(x[:, 5] < 65)
    # Two-meter temperature
    assert np.all(x[:, 6] > 200)
    assert np.all(x[:, 6] < 400)
    # TCWV
    assert np.all(x[:, 7] >= 0)
    assert np.all(x[:, 7] < 100)

    # Assert all targets are loaded
    assert all(t in y for t in targets)


def test_load_training_data_3d_mhs():
    """
    Ensure that loading the training data for MHS produces realistic
    values.
    """
    input_file = DATA_PATH / TEST_FILE_MHS
    input_data = decompress_and_load(input_file)

    sensor = sensors.MHS

    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x, y = sensor.load_training_data_3d(input_data, targets, False, rng)

    # Two-meter temperature
    valid = np.isfinite(x[:, 6])
    assert np.all(x[:, 6][valid] > 150)
    assert np.all(x[:, 6][valid] < 400)
    # TCWV
    valid = np.isfinite(x[:, 7])
    assert np.all(x[:, 7][valid] >= 0)
    assert np.all(x[:, 7][valid] < 100)

    # Assert all targets are loaded
    assert all(t in y for t in targets)

    # Ensure that loaded surface precip is within the range given
    # of the surface precip observed for the different angles.
    sp = y["surface_precip"]
    mask = sp >= 0
    assert np.all(sp[mask] < 500)

    tbs = x[:, :5]
    valid = tbs >= 0
    assert np.all((tbs[valid] > 20) * (tbs[valid] < 400))


def test_interpolation_mhs(tmp_path):
    """
    Ensure that interpolation of surface precipitation
    works.
    """
    input_file = DATA_PATH / TEST_FILE_MHS
    input_data = decompress_and_load(input_file)

    input_data = input_data.sel({"samples": input_data.source == 0})

    for i in range(10):
        input_data.surface_precip[:, :, :, i] = i

    input_data.to_netcdf(tmp_path / "test.nc")

    sensor = sensors.MHS
    sensor._angles = np.arange(10)

    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x, y = sensor.load_training_data_1d(tmp_path / "test.nc", targets, False, rng)

    # Assert all targets are loaded
    sp = y["surface_precip"]
    va = np.abs(x[:, 5])

    inds = (sp > 1.0) * (sp < 8.0)
    assert np.all(np.isclose(va[inds], sp[inds]))


def test_load_training_data_1d_tmi():
    """
    Ensure that loading the training data for TMI produces realistic
    values.
    """
    DATA_PATH = Path(__file__).parent / "data"
    input_file = DATA_PATH / TEST_FILE_TMI
    input_data = decompress_and_load(input_file)

    sensor = sensors.TMI

    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x, y = sensor.load_training_data_1d(input_data, targets, False, rng)

    # TB ranges
    assert np.all(x[:, :9] > 20)
    assert np.all(x[:, :9] < 500)
    # Two-meter temperature
    assert np.all(x[:, 9] > 200)
    assert np.all(x[:, 9] < 400)
    # TCWV
    assert np.all(x[:, 10] >= 0)
    assert np.all(x[:, 10] < 100)

    # Assert all targets are loaded
    assert all(t in y for t in targets)


def test_load_training_data_3d_tmi():
    """
    Ensure that loading the training data for TMI produces realistic
    values.
    """
    DATA_PATH = Path(__file__).parent / "data"
    input_file = DATA_PATH / TEST_FILE_TMI
    input_data = decompress_and_load(input_file)

    sensor = sensors.TMI

    targets = ["surface_precip", "rain_water_content"]
    rng = np.random.default_rng()

    x, y = sensor.load_training_data_3d(input_data, targets, False, rng)

    # Two-meter temperature
    valid = np.isfinite(x[:, 6])
    assert np.all(x[:, 9][valid] > 150)
    assert np.all(x[:, 9][valid] < 400)
    # TCWV
    valid = np.isfinite(x[:, 7])
    assert np.all(x[:, 10][valid] >= 0)
    assert np.all(x[:, 10][valid] < 100)

    # Assert all targets are loaded
    assert all(t in y for t in targets)

    # Ensure that loaded surface precip is within the range given
    # of the surface precip observed for the different angles.
    sp = y["surface_precip"]
    mask = sp >= 0
    assert np.all(sp[mask] < 500)

    tbs = x[:, :5]
    valid = tbs >= 0
    assert np.all((tbs[valid] > 20) * (tbs[valid] < 400))

