"""
==============
gprof_nn.utils
==============

Collection of utility attributes and functions.
"""
from copy import copy

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
    v = v.copy().astype(np.float32)
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


def great_circle_distance(lats_1, lons_1, lats_2, lons_2):
    """
    Approximate distance between locations on earth.

    Uses haversine formulat with an earth radius of 6 371 km.

    Args:
        lats_1: Latitude coordinates of starting points.
        lons_1: Longitude coordinates of starting points.
        lats_2: Latitude coordinates of target points.
        lons_2: Longitude coordinates of target points.

    Return:
        The distance between the points described by the input
        arrays in m.
    """
    lats_1 = np.deg2rad(lats_1)
    lons_1 = np.deg2rad(lons_1)
    lats_2 = np.deg2rad(lats_2)
    lons_2 = np.deg2rad(lons_2)

    d_lons = lons_2 - lons_1
    d_lats = lats_2 - lats_1

    a = np.sin(d_lats / 2.0) ** 2 + np.cos(lats_1) * np.cos(lats_2) * np.sin(
        d_lons / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371e3
    return R * c


def calculate_tiles_and_cuts(n, tile_size, overlap):
    """
    Calculates slices to extract input batches and masks for
    output data.

    Args:
        n: The size of the dimension to tile
        tile_size: The size of each tile.
        overlap: The overlap between tiles.

    Returns:
        A tuple ``(tiles, cuts)`` containing lists ``tiles`` and ``cuts``
        containing slice objects to extract inputs and outputs.
    """
    if overlap > tile_size:
        raise ValueError(
            f"Tile size {tile_size} must be at least as large as "
            f"overlap {overlap}."
            )

    i_start = 0
    i_end = tile_size
    i_start_old = i_start
    i_start_end = i_end

    overlap_left = 0
    overlap_right = overlap

    tiles = []
    cuts = []

    while i_end < n:

        tiles.append(slice(i_start, i_end))

        i_start_old = i_start
        i_end_old = i_end

        i_start = min(i_start + tile_size - overlap, n - tile_size)
        i_end = i_start + tile_size

        overlap_right = i_end_old - i_start

        cuts.append(slice(overlap_left, tile_size - overlap_right // 2))
        overlap_left = overlap_right - overlap_right // 2

    tiles.append(slice(i_start, i_end))
    cuts.append(slice(overlap_left, None))

    return tiles, cuts


def expand_tbs(tbs):
    """
    Helper functions to expand GMI observations to the 15 channels.

    The GMI preprocessor as well as the simulator all produce observation
    data with 15 channels for GMI with two of them containing only missing
    values. Since the GPROF-NN networks expect 15 channel as input, data
    that comes directly from a L1C file must extended accordingly.

    Args:
        tbs: An array containing 13 brightness temperatures of GMI
            oriented along its last axis.

    Return:
        Array containing the same observations but with two empty
        chanels added at indices 5 and 12.
    """
    tbs_e = np.zeros(tbs.shape[:-1] + (15,), dtype=np.float32)
    tbs_e[..., :5] = tbs[..., :5]
    tbs_e[..., 5] = np.nan
    tbs_e[..., 6:12] = tbs[..., 5:11]
    tbs_e[..., 12] = np.nan
    tbs_e[..., 13:] = tbs[..., 11:]
    return tbs_e


def adapt_normalizer(gmi_normalizer, sensor):
    """
    Create input normalizer for a given sensor based on the normalizer used
    for GMI.

    Args:
        gmi_normalizer: The ``quantnn.normalizer.Normalizer`` object
            used for GMI.
        sensor: Sensor object representing the sensor for which to create
            the normalizer.

    """
    ch_inds = sensor.gmi_channels
    stats = {}
    for ind, ind_gmi in enumerate(ch_inds):
        stats[ind] = gmi_normalizer.stats[ind_gmi]
    for ind, ind_gmi in enumerate(range(15, 17)):
        stats[ind + sensor.n_chans] = gmi_normalizer.stats[ind_gmi]
    normalizer = copy(gmi_normalizer)
    normalizer.stats = stats
    return normalizer


def calculate_smoothing_kernel(
        fwhm_a,
        fwhm_x,
        res_a=1.0,
        res_x=1.0
):
    """
    Calculate Gaussian smoothing kernal with given FWHM.

    Args:
        fwhm_a: FWHM in along the first dimension
        fwhm_x: FWHM in the second dimension
        res_a: Grid resolution along first dimension.
        res_x: Grid resolution along second dimension.

    Return:
        A 2D array containing the Gaussian smoothing kernel.
    """
    fwhm_a = fwhm_a / res_a
    w_a = int(fwhm_a) + 1
    fwhm_x = fwhm_x / res_x
    w_x = int(fwhm_x) + 1
    d_a = 2 * np.arange(-w_a, w_a + 1e-6).reshape(-1, 1) / fwhm_a
    d_x = 2 * np.arange(-w_x, w_x + 1e-6).reshape(1, -1) / fwhm_x
    k = np.exp(np.log(0.5) * (d_a ** 2 + d_x ** 2))
    k = k / k.sum()
    return k
