"""
Tests for the loading of surface maps for the GPROF-NN data processing.
"""
from datetime import datetime

import pytest
import numpy as np

from gprof_nn.data.surface import read_landmask
from gprof_nn.data.preprocessor import has_preprocessor


HAS_PREPROCESSOR = has_preprocessor()


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_read_landmask():
    """
    Test reading of land mask.
    """
    mask = read_landmask("GMI")
    assert mask.mask.shape == (360 * 32, 180 * 32)

    mask = read_landmask("MHS")
    assert mask.mask.shape == (360 * 16, 180 * 16)

    # Ensure point in North Atlantic is classified as Ocean.
    m = mask.interp({"longitude": -46.0, "latitude": 35.0})
    assert np.isclose(m.mask.data, 0)

    # Ensure point in Africa is classified as land.
    m = mask.interp({"longitude": 0.0, "latitude": 20.0})
    assert np.all(m.mask.data > 0)
