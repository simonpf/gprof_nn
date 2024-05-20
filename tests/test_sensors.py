"""
Tests for the data loading function of the sensor classes.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.training_data import decompress_and_load


def test_sensor_parsing():
    """
    Ensure that parsing of sensors works as expected.
    """
    assert hasattr(sensors, "GMI")
    assert hasattr(sensors, "TMI")
    assert hasattr(sensors, "AMSR2")
    assert hasattr(sensors, "TMS")
    assert hasattr(sensors, "MHS")
