"""
Tests for the ``gprof_nn.utils`` module.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn.augmentation import (get_transformation_coordinates,
                                   GMI_GEOMETRY)
from gprof_nn.utils import (apply_limits,
                            get_mask,
                            calculate_interpolation_weights,
                            interpolate)
from gprof_nn.data.utils import (load_variable,
                                 decompress_scene,
                                 remap_scene)

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
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"
    dataset = xr.open_dataset(input_file)
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
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"
    scene = xr.open_dataset(input_file)[{"samples": 1}]

    scene_d = decompress_scene(scene, ["rain_water_content"])

    assert "pixels" in scene_d.rain_water_content.dims


def test_remap_scene():
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"
    scene = xr.open_dataset(input_file)[{"samples": 1}]
    coords = get_transformation_coordinates(
        GMI_GEOMETRY, 221, 221, 0.5, 0.5, 0.5
    )
    scene_r = remap_scene(scene, coords, ["surface_precip"])

    sp_r = scene_r.surface_precip.data
    mask = np.isfinite(sp_r)
    sp_r = sp_r[mask]
    sp = scene.surface_precip.data
    sp = sp[mask]

    assert np.all(np.isclose(sp, sp_r))
