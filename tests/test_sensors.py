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


