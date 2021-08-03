"""
===================
gprof_nn.data.utils
===================

Functions that are shared across multiple sub modules of
the ``gprof_nn.data`` module.
"""
import numpy as np

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
    data_new = np.zeros(new_shape, dtype=data.dtype)
    data_new[:] = np.nan

    selection = [slice(0, None)] * data_new.ndim
    selection[axis] = slice(i_start, i_end)
    data_new[selection] = data
    return data_new
