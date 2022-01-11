"""
Tests for the loading of surface maps for the GPROF-NN data processing.
"""
from datetime import datetime

import pytest
import numpy as np

from gprof_nn.data.surface import (read_land_mask,
                                   read_autosnow,
                                   read_emissivity_classes)
from gprof_nn.data.preprocessor import has_preprocessor


HAS_PREPROCESSOR = has_preprocessor()


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_read_land_mask():
    """
    Test reading of land mask.
    """
    mask = read_land_mask("GMI")
    assert mask.mask.shape == (180 * 32, 360 * 32)

    mask = read_land_mask("MHS")
    assert mask.mask.shape == (180 * 16, 360 * 16)

    # Ensure point in North Atlantic is classified as Ocean.
    m = mask.interp({"longitude": -46.0, "latitude": 35.0})
    assert np.isclose(m.mask.data, 0)

    # Ensure point in Africa is classified as land.
    m = mask.interp({"longitude": 0.0, "latitude": 20.0})
    assert np.all(m.mask.data > 0)


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_read_autosnow():
    """
    Test reading of autosnow files.
    """
    autosnow = read_autosnow("2021-01-01T00:00:00")

    # Ensure no snow around equator
    autosnow_eq = autosnow.interp({"latitude": 0.0, "longitude": 0.0}, "nearest")
    assert np.all(autosnow_eq.snow.data == 0)


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_read_emissivity_classes():
    """
    Test reading of emissivity classes.
    """
    data = read_emissivity_classes()

    # Ensure point in North Atlantic is classified as Ocean.
    data_i = data.interp({"longitude": -46.0, "latitude": 35.0})
    assert np.all(np.isclose(data_i.emissivity.data, 0))

    # Ensure point in Africa is classified as land.
    data_i = data.interp({"longitude": 0.0, "latitude": 20.0})
    assert np.all(data_i.emissivity.data > 0)
