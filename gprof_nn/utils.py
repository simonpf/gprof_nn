"""
==============
gprof_nn.utils
==============

Collection of utility attributes and functions.
"""
import numpy as np
import xarray as xr

# Rectangular bounding box around continental united states (CONUS).
CONUS = (-130, 20, -60.0, 55)

SURFACE_TYPE_NAMES = [
    "Ocean",
    "Sea-Ice",
    "Vegetation 1",
    "Vegetation 2",
    "Vegetation 3",
    "Vegetation 4",
    "Vegetation 5",
    "Snow 1",
    "Snow 2",
    "Snow 3",
    "Snow 4",
    "Standing Water",
    "Land Coast",
    "Mixed land/ocean o. water",
    "Ocean or water Coast",
    "Sea-ice edge",
    "Mountain Rain",
    "Mountain Snow",
]


def surface_type_to_name(surface_index):
    """
    Transform surface type index to name.

    Args:
        surface_index: The integer surface type code.
    Return:
        String representation of the surface type.
    """
    return SURFACE_TYPE_NAMES[int(surface_index) - 1]


def load_retrieval_results(files_gprof,
                           files_gprof_nn_0d,
                           files_reference):
    """
    Load and combine retrieval results from different algorithm.

    Args:
        files_gprof: List of files containing the results from the legacy GPROF
             algorithm.
        files_gprof_nn_0d: List of files containing the results from the
             GPROF-NN 0D algorithm.
        files_reference: List of files containing the reference results.

    Returns:
        'xarray.Dataset' containing the reference data combined with the
        retrieval results.
    """
    results_gprof = []
    results_gprof_nn_0d = []
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

    for f in files_gprof_nn_0d:
        results_gprof_nn_0d.append(xr.load_dataset(f))
    if files_gprof_nn_0d:
        results_gprof_nn_0d = xr.concat(results_gprof_nn_0d, "samples")
        for v in reference.variables:
            if not v in results_gprof_nn_0d.variables:
                continue
            data = reference[v].data
            reference[v + "_gprof_nn_0d"] = results_gprof_nn_0d[v]

    return reference


def apply_limits(v,
                 v_min,
                 v_max):
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


def calculate_interpolation_weights(angles,
                                    angle_grid):
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
        weights[mask, i] = ((angle_grid[i + 1] - angles[mask]) /
                            (angle_grid[i + 1] - angle_grid[i]))
        weights[mask, i + 1] = ((angles[mask] - angle_grid[i]) /
                                (angle_grid[i + 1] - angle_grid[i]))
    weights[indices == 0] = 0.0
    weights[indices == 0, 0] = 1.0
    weights[indices >= angle_grid.size] = 0.0
    weights[indices >= angle_grid.size, -1] = 1.0

    return weights


def interpolate(variable,
                weights):
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
    if weights.shape != variable.shape[:weights.ndim]:
        raise ValueError(
            "Provided weights don't match the shape of value array to "
            "interpolate."
        )
    shape = (variable.shape[:weights.ndim]
             + (1,) * (variable.ndim - weights.ndim))
    weights = weights.reshape(shape)
    return np.sum(variable * weights, axis=weights.ndim - 1)


