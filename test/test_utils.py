"""
Tests for the ``gprof_nn.utils`` module.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn.augmentation import get_transformation_coordinates
from gprof_nn.data import get_test_data_path
from gprof_nn.sensors import GMI_VIEWING_GEOMETRY
from gprof_nn.utils import (apply_limits,
                            get_mask,
                            calculate_interpolation_weights,
                            interpolate)
from gprof_nn.data.utils import (load_variable,
                                 decompress_scene,
                                 remap_scene)
from gprof_nn.data.training_data import decompress_and_load


DATA_PATH = get_test_data_path()


def test_apply_limits():
    """
    Ensure that upper and lower bounds are applied correctly.
    """
    x = np.random.normal(size=(10, 10))

    x_l = apply_limits(x, 0.0, None)
    x_l = x_l[np.isfinite(x_l)]
    assert np.all(x_l >= 0.0)

    x_r = apply_limits(x, None, 0.0)
    x_r = x_r[np.isfinite(x_r)]
    assert np.all(x_r <= 0.0)

    x = apply_limits(x, 0.0, 0.0)
    x = x[np.isfinite(x)]
    assert x.size == 0


def test_get_mask():
    """
    Ensure that values extracted with mask are within given limits.
    """
    x = np.random.normal(size=(10, 10))

    mask = get_mask(x, 0.0, None)
    x_l = x[mask]
    assert np.all(x_l >= 0.0)

    mask = get_mask(x, None, 0.0)
    x_r = x[mask]
    assert np.all(x_r <= 0.0)

    mask = get_mask(x, 0.0, 0.0)
    x = x[mask]
    assert x.size == 0


def test_calculate_interpolation_weights():
    """
    Ensure that calculating interpolation weights for the grid values
    itself produces a diagonal matrix of weights.

    Also ensure that weights always sum to one across last dimension.
    """
    grid = np.arange(0, 11)
    weights = calculate_interpolation_weights(grid, grid)

    assert np.all(np.isclose(weights.diagonal(), 1.0))

    values = np.random.uniform(0, 10, size=(10, 10))
    weights = calculate_interpolation_weights(values, grid)
    assert np.all(np.isclose(np.sum(weights, 2), 1.0))


def test_interpolation():
    """
    Ensure that calculating interpolation weights for the grid values
    itself produces a diagonal matrix of weights.
    """
    grid = np.arange(0, 11)
    weights = calculate_interpolation_weights(grid, grid)
    y = interpolate(np.repeat(grid.reshape(1, -1), 11, 0), weights)
    assert np.all(np.isclose(grid, y))

    values = np.random.uniform(0, 10, size=(10))
    weights = calculate_interpolation_weights(values, grid)
    y = interpolate(np.repeat(grid.reshape(1, -1), 10, 0), weights)
    assert np.all(np.isclose(y, values))


def test_load_variable():
    """
    Ensure that loading a variable correctly replaces invalid value and
    conserves shape when used without mask.

    Also ensure that masking works.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = decompress_and_load(input_file)
    sp = load_variable(dataset, "surface_precip")

    expected_shape = (dataset.samples.size,
                      dataset.scans.size,
                      dataset.pixels.size)
    assert sp.shape == expected_shape

    sp = sp[np.isfinite(sp)]
    assert np.all((sp >= 0.0) * (sp < 500))

    sp = load_variable(dataset, "surface_precip")
    mask = sp > 10
    sp = load_variable(dataset, "surface_precip", mask)
    sp = sp[np.isfinite(sp)]
    assert np.all((sp > 10.0) * (sp < 500))


def test_decompress_scene():
    """
    Ensure that loading a variable correctly replaces invalid value and
    conserves shape when used without mask.

    Also ensure that masking works.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    scene = decompress_and_load(input_file)[{"samples": 1}]

    scene_d = decompress_scene(scene, ["surface_precip",
                                       "rain_water_content",
                                       "rain_water_path"])

    assert "pixels" in scene_d.rain_water_content.dims

    # Over ocean all pixels where IWP is defines should also
    # have a valid surface precip value.
    rwp = scene_d.rain_water_path.data
    sp = scene_d.surface_precip.data
    st = scene_d.surface_type
    inds = (st == 1) * (rwp >= 0.0)
    assert np.all(sp[inds] >= 0.0)

