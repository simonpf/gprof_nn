"""
Tests for the data augmentation methods in gprof_nn.augmentation.
"""
import numpy as np
from gprof_nn import sensors
from gprof_nn.augmentation import (GMI_GEOMETRY,
                                   MHS_GEOMETRY,
                                   get_center_pixels,
                                   get_transformation_coordinates,
                                   offset_x,
                                   get_center_pixel_input,
                                   M,
                                   N)


def test_gmi_geometry():
    """
    Assert that coordinate transformation function for GMI viewing
    geometry are reversible.
    """
    i = np.arange(0, 221, 10)
    j = np.arange(0, 221, 10)
    ij = np.stack(np.meshgrid(i, j))
    xy = GMI_GEOMETRY.pixel_coordinates_to_euclidean(ij)
    ij_r = GMI_GEOMETRY.euclidean_to_pixel_coordinates(xy)
    assert np.all(np.isclose(ij, ij_r))


def test_mhs_geometry():
    """
    Assert that coordinate transformation function for GMI viewing
    geometry are reversible.
    """
    i = np.arange(0, 90, 10)
    j = np.arange(0, 90, 10)
    ij = np.stack(np.meshgrid(i, j))
    xy = MHS_GEOMETRY.pixel_coordinates_to_euclidean(ij)
    ij_r = MHS_GEOMETRY.euclidean_to_pixel_coordinates(xy)
    assert np.all(np.isclose(ij, ij_r))


def test_interpolation_weights():
    """
    Ensure that all interpolation weights are positive and sum to 1.
    """
    weights = MHS_GEOMETRY.get_interpolation_weights(sensors.MHS.angles)
    assert np.all(weights.sum(-1) == 1.0)
    assert np.all(weights >= 0)


def test_inputer_center():
    """
    Ensures that the calculated window always contains the center of
    the GMI swath.
    """
    l = get_center_pixel_input(0.0, 64)
    assert l + 32 == 110
    r = get_center_pixel_input(1.0, 64)
    assert r - 32 == 110


def test_transformation_coordinates():
    """
    Ensure that transformation coordinates correspond to identity
    mapping for when input and output window are located at the
    center of the swath.
    """
    c = get_transformation_coordinates(GMI_GEOMETRY,
                                       64,
                                       64,
                                       0.5,
                                       0.5,
                                       0.5)
    print(c[1, 0, :])
    assert np.all(np.isclose(c[1, 0, :], np.arange(110 - 32, 110 + 32)))
