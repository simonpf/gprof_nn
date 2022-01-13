"""
Tests for the gprof_nn.data.validation module.
"""
import numpy as np

from gprof_nn import sensors
from gprof_nn.data.validation import ValidationData


def test_get_granules():
    """
    Test listing of granules.
    """
    validation_data = ValidationData(sensors.GMI)
    granules = validation_data.get_granules(2016, 10)
    assert 15199 in granules


def test_open_granule():
    """
    Test listing of granules.
    """
    validation_data = ValidationData(sensors.GMI)
    data = validation_data.open_granule(2016, 10, 15199)

    sp = data.precip_rate.data
    sp = sp[sp >= 0]
    assert np.all(sp <= 500)

    lats = data.latitude.data
    assert np.all((lats >= 0) * (lats <= 60))

    lons = data.longitude.data
    assert np.all((lons >= -180) * (lons <= 0))
