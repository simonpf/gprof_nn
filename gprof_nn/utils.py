"""
==============
gprof_nn.utils
==============

Collection of utility attributes and functions.
"""
import numpy as np
import xarray as xr

from gprof_nn.definitions import SURFACE_TYPE_NAMES

# Rectangular bounding box around continental united states (CONUS).
CONUS = (-130, 20, -60.0, 55)


def surface_type_to_name(surface_index):
    """
    Transform surface type index to name.

    Args:
        surface_index: The integer surface type code.
    Return:
        String representation of the surface type.
    """
    return SURFACE_TYPE_NAMES[int(surface_index) - 1]


def load_retrieval_results(files_gprof, files_gprof_nn_1d, files_reference):
    """
    Load and combine retrieval results from different algorithm.

    Args:
        files_gprof: List of files containing the results from the legacy GPROF
             algorithm.
        files_gprof_nn_1d: List of files containing the results from the
             GPROF-NN 1D algorithm.
        files_reference: List of files containing the reference results.

    Returns:
        'xarray.Dataset' containing the reference data combined with the
        retrieval results.
    """
    results_gprof = []
    results_gprof_nn_1d = []
    reference = []

    for f in files_reference:
        reference.append(xr.load_dataset(f))
    reference = xr.concat(reference, "samples")

    for f in files_gprof:
        results_gprof.append(xr.load_dataset(f))
    if files_gprof:
        results_gprof = xr.concat(results_gprof, "samples")
        for v in reference.variables:
            if not v in results_gprof.variables:
                continue
            data = reference[v].data
            reference[v + "_gprof"] = results_gprof[v]

    for f in files_gprof_nn_1d:
        results_gprof_nn_1d.append(xr.load_dataset(f))
    if files_gprof_nn_1d:
        results_gprof_nn_1d = xr.concat(results_gprof_nn_1d, "samples")
        for v in reference.variables:
            if not v in results_gprof_nn_1d.variables:
                continue
            data = reference[v].data
            reference[v + "_gprof_nn_1d"] = results_gprof_nn_1d[v]

    return reference


def apply_limits(v, v_min, v_max):
    """
    Apply limits to variable.

    Args:
        v: A numpy array containing the variable values.
        v_min: A lower bound below which values of the variable
            should be considered physically unsound or 'None'.
        v_max: An upper bound above which values of the variable should
            be considered physically unsound or 'None'.

    Return:
        A copy of 'v' with value below the lower and above the upper
        bound set to 'NAN'.
    """
    if v_min is None and v_max is None:
        return v
    v = v.copy()
    if v_min is not None:
        mask = v < v_min
        v[mask] = np.nan
    if v_max is not None:
        mask = v > v_max
        v[mask] = np.nan
    return v


def get_mask(v, v_min, v_max):
    """
    Return boolean mask identifying valid values.

    Args:

        v: A numpy array containing the variable values.
        v_min: A lower bound below which values of the variable
            should be considered physically unsound or 'None'.
        v_max: An upper bound above which values of the variable should
            be considered physically unsound or 'None'.

    Return:
        Bool array of same shape as v containing 'True' where the values
        of 'v' are within the range ``[v_min, v_max]``.
    """
    mask = np.ones(v.shape, dtype=np.bool)
    if v_min is not None:
        mask = mask * (v >= v_min)
    if v_max is not None:
        mask = mask * (v <= v_max)
    return mask


def calculate_interpolation_weights(angles, angle_grid):
    """
    Calculate interpolation weights for angle-dependent variables.

    Args:
        args: The angles to which to interpolate the variables.
        angle_grid: Array containing the angle grid on which the
            variables are calculated.

    Return:
        An array with one more dimension than 'angles' containing the
        interpolation weights that can be used to linearly interpolate
        variables to the given angles.
    """
    weights = np.zeros(angles.shape + (angle_grid.size,), np.float32)
    indices = np.digitize(angles, angle_grid)

    for i in range(angle_grid.size - 1):
        mask = (indices - 1) == i
        weights[mask, i] = (angle_grid[i + 1] - angles[mask]) / (
            angle_grid[i + 1] - angle_grid[i]
        )
        weights[mask, i + 1] = (angles[mask] - angle_grid[i]) / (
            angle_grid[i + 1] - angle_grid[i]
        )
    weights[indices == 0] = 0.0
    weights[indices == 0, 0] = 1.0
    weights[indices >= angle_grid.size] = 0.0
    weights[indices >= angle_grid.size, -1] = 1.0

    return weights


def interpolate(variable, weights):
    """
    Interpolate variable using precalculated weights.

    Args:
        variable: Array containing the variable values to interpolate.
        weights: Weight array pre-calculated using
        'calcualte_interpolation_weights'. Dimensions must match the
        first dimensions of 'variable'.

    Return:
        The values in variable interpolated using the
        precalculated weights.
    """
    if weights.shape != variable.shape[: weights.ndim]:
        raise ValueError(
            "Provided weights don't match the shape of value array to " "interpolate."
        )
    shape = variable.shape[: weights.ndim] + (1,) * (variable.ndim - weights.ndim)
    weights_r = weights.reshape(shape)
    return np.sum(variable * weights_r, axis=weights.ndim - 1)

def bootstrap_mean(data, n_samples=10, weights=None):
    """
    Calculate mean and standard deviation using boostrapping.

    Args:
        data: 1D array containing the samples of which to calculate mean
            and standard deviation.
        n_samples: The number of bootstrap samples to perform.
        weights: If provided used to calculate a weighted mean of
            the results.

    Return:
        Tuple ``(mu, std)`` containing the estimated mean ``mu`` and
        corresponding standard deivation ``std``.
    """
    stats = []
    for i in range(n_samples):
        indices = np.random.randint(0, data.size, size=data.size)
        if weights is None:
            stats.append(np.mean(data[indices]))
        else:
            ws = weights[indices]
            samples = data[indices]
            stats.append(np.sum(samples * ws) / ws.sum())
    data_r = np.stack(stats)
    mu = data_r.mean()
    std = data_r.std()
    return mu, std
