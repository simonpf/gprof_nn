"""
Tests for the data augmentation methods in gprof_nn.augmentation.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.augmentation import (
    Swath,
    get_center_pixels,
    get_transformation_coordinates,
    get_center_pixel_input,
    M,
    N
)
from gprof_nn.data.training_data import decompress_and_load


DATA_PATH = get_test_data_path()


def test_gmi_geometry():
    """
    Assert that coordinate transformation function for GMI viewing
    geometry are reversible.
    """
    i = np.arange(0, 221, 10)
    j = np.arange(0, 221, 10)
    ij = np.stack(np.meshgrid(i, j))
    geometry = sensors.GMI.viewing_geometry
    xy = geometry.pixel_coordinates_to_euclidean(ij)
    ij_r = geometry.euclidean_to_pixel_coordinates(xy)
    assert np.all(np.isclose(ij, ij_r))


def test_mhs_geometry():
    """
    Assert that coordinate transformation function for GMI viewing
    geometry are reversible.
    """
    i = np.arange(0, 90, 10)
    j = np.arange(0, 90, 10)
    ij = np.stack(np.meshgrid(i, j))
    geometry = sensors.MHS.viewing_geometry
    xy = geometry.pixel_coordinates_to_euclidean(ij)
    ij_r = geometry.euclidean_to_pixel_coordinates(xy)
    assert np.all(np.isclose(ij, ij_r))


def test_swath_geometry():
    """
    Assert that coordinate transformation function for GMI viewing
    geometry are reversible.
    """
    input_file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc.gz"
    input_data = decompress_and_load(input_file)

    lats = input_data.latitude.data[0]
    lons = input_data.longitude.data[0]

    i = np.arange(0, 221, 10)
    j = np.arange(0, 221, 10)
    ij = np.stack(np.meshgrid(i, j))

    swath = Swath(lats, lons)

    xy = swath.pixel_coordinates_to_euclidean(ij)
    ij_r = swath.euclidean_to_pixel_coordinates(xy)
    assert np.all(np.isclose(ij, ij_r))



def test_interpolation_weights():
    """
    Ensure that all interpolation weights are positive and sum to 1.
    """
    geometry = sensors.MHS.viewing_geometry
    weights = geometry.get_interpolation_weights(sensors.MHS.angles)
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
    input_file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc.gz"
    input_data = decompress_and_load(input_file)

    lats = input_data.latitude.data[0]
    lons = input_data.longitude.data[0]
    geometry = sensors.GMI.viewing_geometry
    c = get_transformation_coordinates(
        lats, lons, geometry, 64,
        64, 0.5, 0.5, 0.5
    )
    assert np.all(np.isclose(c[1, 0, :], np.arange(110 - 32, 110 + 32), atol=2.0))
