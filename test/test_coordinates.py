"""
Test for the coordinates submodule.
"""
import numpy as np
from gprof_nn.coordinates import latlon_to_ecef

def test_latlon_to_ecef():

    lons = [0, 90, 180, 270, 360]
    lats = [0, 0, 0, 0, 0]

    x, y, z = latlon_to_ecef(lons, lats)

    assert np.all(np.isclose(x[[1, 3]], 0.0))
    assert np.all(np.isclose(y[[0, 2, 4]], 0.0))
    assert np.all(np.isclose(z, 0.0))
