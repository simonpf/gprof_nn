"""
===================
gprof_nn.data.utils
===================

Functions that are shared across multiple sub modules of
the ``gprof_nn.data`` module.
"""
import numpy as np
import xarray as xr

from gprof_nn.augmentation import extract_domain
from gprof_nn.definitions import LIMITS
from gprof_nn.utils import apply_limits

CENTER = 110
N_PIXELS_CENTER = 41


def compressed_pixel_range():
    """
    Calculates the start and end indices of the compressed storage
    of profile variables.

    Return:
        Tuple ``(i_start, i_end)`` containing the start index ``i_start``
        and end index ``i_end``
    """
    i_start = CENTER - (N_PIXELS_CENTER // 2 + 1)
    i_end = CENTER + (N_PIXELS_CENTER // 2)
    return i_start, i_end


def expand_pixels(data, axis=2):
    """
    Expand target data array that only contain data for central pixels.

    Args:
        data: Array containing data of a retrieval target variable.

    Return:
        The input data expanded to the full GMI swath along the third
        dimension.
    """
    if len(data.shape) <= axis or data.shape[axis] == 221:
        return data
    new_shape = list(data.shape)
    new_shape[axis] = 221
    i_start = CENTER - (N_PIXELS_CENTER // 2 + 1)
    i_end = CENTER + (N_PIXELS_CENTER // 2)
    data_new = np.zeros(new_shape, dtype=np.float32)
    data_new[:] = np.nan

    selection = [slice(0, None)] * data_new.ndim
    selection[axis] = slice(i_start, i_end)
    data_new[tuple(selection)] = data
    return data_new


def load_variable(data, variable, mask=None):
    """
    Loads a variable from the given dataset
    and replaces invalid values.

    Args:
        data: ``xarray.Dataset`` containing the training data.
        variable: The name of the variable to load.
        mask: A mask to subset values in the training data.

    Return:
        An array containing the two-meter temperature from the
        dataset.
    """
    if variable == "scan_time":
        var = data["scan_time"].data.astype(np.int64)
        shape = (data.scans.size, data.pixels.size)
        var = np.broadcast_to(var[..., np.newaxis], shape)
        v = var
    else:
        v = data[variable].data

    if mask is not None:
        v = v[mask]
    if variable in LIMITS:
        v_min, v_max = LIMITS[variable]
    else:
        v_min = -999
        v_max = None
    v = apply_limits(v, v_min, v_max)
    return v


def decompress_scene(scene, targets):
    """
    Decompresses compressed variables in scene.

    Args:
        scene: 'xarray.Dataset' containing the training data scene.
        targets: List of the targets to remap.

    Returns:
        New 'xarray.Dataset' with all compressed variables decompressed.
    """
    variables = [
        "brightness_temperatures",
        "two_meter_temperature",
        "total_column_water_vapor",
        "surface_type",
        "airmass_type",
        "source",
    ] + targets

    data = {}
    for v in variables:
        if "pixels_center" in scene[v].dims:
            data_r = expand_pixels(scene[v].data, axis=1)
            dims = scene[v].dims
            dims = [d if d != "pixels_center" else "pixels" for d in dims]
            data[v] = (dims, data_r)
        else:
            data[v] = (scene[v].dims, scene[v].data)

    return xr.Dataset(data)


def remap_scene(scene, coords, targets):
    """
    Perform viewing geometry correction to a 2D training data
    scene.

    Args:
        scene: 'xarray.Dataset' containing the training data scene.
        coords: Precomputed coordinates to use for remapping.
        targets: List of the targets to remap.

    Returns:
        New 'xarray.Dataset' containing the remapped data.
    """
    variables = [
        "brightness_temperatures",
        "two_meter_temperature",
        "total_column_water_vapor",
        "surface_type",
        "airmass_type",
        "source",
    ] + targets

    data = {}
    dims = ("scans", "pixels")
    for v in variables:
        if "scans" in scene[v].dims:
            data_v = scene[v].data
            if v in ["scan_time"]:
                data_r = extract_domain(data_v, coords, order=0)
            elif v in ["surface_type", "airmass_type", "scan_time"]:
                data_r = extract_domain(data_v, coords, order=0)
                data_r = data_r.astype(np.int32)
            else:
                if v in LIMITS:
                    data_v = apply_limits(data_v, *LIMITS[v])
                data_r = extract_domain(data_v, coords, order=1)
                data_r = data_r.astype(np.float32)
            data[v] = (dims + scene[v].dims[2:], data_r)
        else:
            data[v] = (scene[v].dims, scene[v].data)
    return xr.Dataset(data)


def upsample_scans(array, axis=0):
    """
    Upsample array by a factor of 3 using linear interpolation.

    Args:
        array: The array to upsample.
        axis: The dimension along which to upsample the array.

    Return:
        A new array containign the upsampled data from 'array'.
    """
    n_dims = array.ndim

    new_shape = list(array.shape)
    new_shape[axis] = 3 * (new_shape[axis] - 1) + 1
    array_3 = np.zeros(new_shape, dtype=array.dtype)

    indices_l = [slice(0, None)] * n_dims
    indices_l[axis] = slice(0, -1)
    indices_r = [slice(0, None)] * n_dims
    indices_r[axis] = slice(1, None)
    array_l = array[tuple(indices_l)]
    array_r = array[tuple(indices_r)]
    diff = array_r - array_l

    indices = [slice(0, None)] * n_dims
    indices[axis] = slice(0, None, 3)
    array_3[tuple(indices)] = array
    indices[axis] = slice(1, None, 3)
    array_3[tuple(indices)] = array_l + 1 / 3 * diff#2 / 3 * array_l + 1 / 3 * array_r
    indices[axis] = slice(2, None, 3)
    array_3[tuple(indices)] = array_l + 2 / 3 * diff#1 / 3 * array_l + 2 / 3 * array_r

    return array_3
