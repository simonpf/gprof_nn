"""
Tests for the data augmentation methods in gprof_nn.augmentation.
"""
import numpy as np
from gprof_nn.augmentation import (get_center_pixels,
                                   get_transformation_coordinates,
                                   offset_x,
                                   M,
                                   N)

def test_center_pixels():
    c_o, c_i = get_center_pixels(-1.0, -1.0)
    assert np.isclose(c_o, N // 2)
    assert np.isclose(c_i, 130 - N // 2)

    c_o, c_i = get_center_pixels(0.0, 0.0)
    assert np.isclose(c_o, 110)
    assert np.isclose(c_i, 110)

    c_o, c_i = get_center_pixels(1.0, 1.0)
    assert np.isclose(c_o, 220 - N // 2)
    assert np.isclose(c_i, 90 + N // 2)

def test_offset_x():
    o_x = offset_x(0.0, 0.0)
    assert np.all(np.isclose(o_x, np.arange(N) - N // 2))


def test_transformation_coordinates():

    c = get_transformation_coordinates(0.0, 0.0, -1.0)
    assert np.isclose(c[0].min(), 0.0)
    assert np.isclose(c[0].max(), M - 1.0)
    assert np.isclose(c[1].min(), 110 - N // 2)
    assert np.isclose(c[1].max(), 110 + N // 2 - 1)

    c = get_transformation_coordinates(0.0, 0.0, 1.0)
    assert np.isclose(c[0].min(), 221 - M)
    assert np.isclose(c[0].max(), 220)
    assert np.isclose(c[1].min(), 110 - N // 2)
    assert np.isclose(c[1].max(), 110 + N // 2 - 1)