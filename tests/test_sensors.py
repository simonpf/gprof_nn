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
