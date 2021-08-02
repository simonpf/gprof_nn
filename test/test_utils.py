"""
Tests for the ``gprof_nn.utils`` module.
"""
import numpy as np

from gprof_nn.utils import apply_limits


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
