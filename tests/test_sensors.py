"""
Tests for the data loading function of the sensor classes.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.training_data import decompress_and_load

try:
    import pansat
    HAS_PANSAT = True
except ImportError:
    HAS_PANSAT = False


def test_sensor_parsing():
    """
    Ensure that parsing of sensors works as expected.
    """
    assert hasattr(sensors, "GMI")
    assert hasattr(sensors, "TMI")
    assert hasattr(sensors, "AMSR2")
    assert hasattr(sensors, "TMS")
    assert hasattr(sensors, "MHS")


@pytest.mark.skipif(not HAS_PANSAT, reason="Needs pansat.")
def test_pansat_products():
    """
    Ensure that pansat products of sensors are set correctly.
    """
    from pansat.products import Product

    gmi_prod = sensors.GMI.pansat_products[0]
    assert isinstance(gmi_prod, Product)

    tmi_prod = sensors.TMI.pansat_products[0]
    assert isinstance(tmi_prod, Product)

    atms_prod = sensors.ATMS.pansat_products[0]
    assert isinstance(atms_prod, Product)

    mhs_prod = sensors.MHS.pansat_products[0]
    assert isinstance(mhs_prod, Product)

    amsub_prod = sensors.AMSUB.pansat_products[0]
    assert isinstance(amsub_prod, Product)

    ssmi_prod = sensors.SSMI.pansat_products[0]
    assert isinstance(ssmi_prod, Product)

    ssmis_prod = sensors.SSMIS.pansat_products[0]
    assert isinstance(ssmis_prod, Product)

    mwi_prod = sensors.MWI.pansat_products[0]
    assert isinstance(mwi_prod, Product)
