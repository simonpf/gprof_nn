"""
Tests for the ``gprof_nn.utils`` module.
"""
import numpy as np

from gprof_nn.utils import (apply_limits,
                            calculate_interpolation_weights,
                            interpolate)


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
