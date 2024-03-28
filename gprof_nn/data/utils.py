"""
===================
gprof_nn.data.utils
===================

Functions that are shared across multiple sub modules of
the ``gprof_nn.data`` module.
"""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

from gprof_nn.augmentation import extract_domain
from gprof_nn.definitions import LIMITS
from gprof_nn.utils import apply_limits

CENTER = 110
N_PIXELS_CENTER = 31


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
        "ocean_fraction",
        "land_fraction",
        "ice_fraction",
        "snow_depth",
        "leaf_area_index",
        "orographic_wind",
        "moisture_convergence",
        "source",
    ] + targets
    variables = [var for var in variables if var in scene]

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
        "land_fraction",
        "ice_fraction",
        "snow_depth",
        "leaf_area_index",
        "orographic_wind",
        "moisture_convergence",
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


def save_scene(
        scene: xr.Dataset,
        path: Path
) -> None:
    """
    Saves a GPROF-NN training scene as NetCDF file applying appropriate
    compression.

    Args:
        scene: An xarray.Dataset containing the training scene to save
        path: Path pointing to the file to which to write the scene.
    """
    scene = scene.copy(deep=True)

    tbs = scene.brightness_temperatures.data
    tbs[tbs < 0] = np.nan
    tbs[tbs > 400] = np.nan

    if "simulated_brightness_temperatures" in scene:
        tbs = scene.simulated_brightness_temperatures.data
        tbs[tbs < 0] = np.nan
        tbs[tbs > 400] = np.nan
        dtbs = scene.brightness_temperature_biases.data
        dtbs[dtbs < -150] = np.nan
        dtbs[dtbs > 150] = np.nan

    sfc = scene.surface_type.data
    sfc[sfc < 0] = -99
    ali = scene.airlifting_index.data
    ali[ali < 0] = -99
    mtn = scene.mountain_type.data
    mtn[mtn < 0] = -99
    lfr = scene.land_fraction.data
    lfr[lfr < 0] = -99
    ifr = scene.ice_fraction.data
    ifr[ifr < 0] = -99

    encoding = {
        "brightness_temperatures": {
            "dtype": "uint16",
            "scale_factor": 0.01,
            "add_offset": 1,
            "_FillValue": 2 ** 16 - 1,
            "zlib": True
        },
        "simulated_brightness_temperatures": {
            "dtype": "uint16",
            "scale_factor": 0.01,
            "add_offset": 1,
            "_FillValue":  2 ** 16 - 1,
            "zlib": True
        },
        "brightness_temperature_biases": {
            "dtype": "int16",
            "scale_factor": 0.01,
            "add_offset": 0,
            "_FillValue": - (2 ** 15),
            "zlib": True
        },
        "surface_type": {
            "dtype": "int8",
            "zlib": True
        },
        "mountain_type": {
            "dtype": "int8",
            "zlib": True
        },
        "airlifting_index": {
            "dtype": "int8",
            "zlib": True
        },
        "mountain_type": {
            "dtype": "int8",
            "zlib": True
        },
        "land_fraction": {
            "dtype": "int8",
            "zlib": True
        },
        "ice_fraction": {
            "dtype": "int8",
            "zlib": True
        }

    }

    for var in [
            "ice_water_path",
            "cloud_water_path",
            "rain_water_path",
            "surface_precip",
            "surface_precip_combined",
            "convective_precip"
    ]:
        encoding[var] = {"dtype": "float32", "zlib": True}

    for var in scene.variables:
        if not var in encoding:
            encoding[var] = {"zlib": True}

    encoding = {key: val for key, val in encoding.items() if key in scene}
    scene.to_netcdf(path, encoding=encoding)


def extract_scenes(
        data: xr.Dataset,
        n_scans: int = 128,
        n_pixels: int = 64,
        min_valid: int = 20,
        overlapping: bool = False,
        reference_var: str = "surface_precip"
) -> List[xr.Dataset]:
    """
    Extract scenes from an xr.Dataset containing satellite
    observations.

    Args:
        data: xarray.Dataset containing GPM observations from which to
            extract training scenes.
        n_scans: The number of scans in a single scene.
        n_pixels: The number of pixels in a single scene.
        min_valid: The minimum number of pixels with valid surface
            precipitation.
        overlapping: Boolean flag indicating whether or not the
            extracted scenes should be overlapping or not.
        reference_var: The variable to use to determine valid
            pixels.

    Return:
        A list of scenes extracted from the provided observations.
    """
    n_scans_tot = data.scans.size
    n_pixels_tot = data.pixels.size

    def get_valid(dataset):
        valid = np.isfinite(dataset[reference_var])
        if valid.ndim > 2:
            valid = valid.any([dim for dim in valid.dims if not dim in ["scans", "pixels"]])
        return valid

    valid = np.stack(np.where(get_valid(data)), -1)
    valid_inds = list(np.random.permutation(valid.shape[0]))

    scenes = []

    while len(valid_inds) > 0:

        ind = np.random.choice(valid_inds)
        scan_cntr, pixel_cntr = valid[ind]

        scan_start = min(max(scan_cntr - n_scans // 2, 0), n_scans_tot - n_scans)
        scan_end = scan_start + n_scans
        pixel_start = min(max(pixel_cntr - n_pixels // 2, 0), n_pixels_tot - n_pixels)
        pixel_end = pixel_start + n_pixels

        subscene = data[
            {
                "scans": slice(scan_start, scan_end),
                "pixels": slice(pixel_start, pixel_end),
            }
        ]
        n_valid = get_valid(subscene).sum()
        if n_valid >= min_valid:
            scenes.append(subscene)
            if overlapping:
                covered = (
                    (valid[..., 0] >= scan_start) *
                    (valid[..., 0] < scan_end) *
                    (valid[..., 1] >= pixel_start) *
                    (valid[..., 1] < pixel_end)
                )
            else:
                covered = (
                    (valid[..., 0] >= scan_start - n_scans // 2) *
                    (valid[..., 0] < scan_end + n_scans // 2) *
                    (valid[..., 1] >= pixel_start - n_pixels // 2) *
                    (valid[..., 1] < pixel_end + n_pixels // 2)
                )

            covered = {ind for ind in valid_inds if covered[ind]}
            valid_inds = [ind for ind in valid_inds if not ind in covered]
        else:
            valid_inds.remove(ind)

    return scenes


def write_training_samples_1d(
        output_path: Path,
        prefix: str,
        dataset: xr.Dataset,
) -> None:
    """
    Write training data in GPROF-NN 1D format.

    Args:
        dataset: An 'xarray.Dataset' containing collocated input
            observations and reference data.
        output_path: Path to which the training data will be written.
    """
    dataset = dataset[{"pixels": slice(*compressed_pixel_range())}]
    mask = np.isfinite(dataset.surface_precip.data)
    if mask.ndim > 2:
        mask = mask.all(-1)

    valid = {}
    for var in dataset.variables:
        arr = dataset[var]
        if arr.data.ndim < 2:
            arr_data = np.broadcast_to(arr.data[..., None], mask.shape)
        else:
            arr_data = arr.data
        valid[var] = ((("samples",) + arr.dims[2:]), arr_data[mask])

    valid = xr.Dataset(valid, attrs=dataset.attrs)
    start_time = pd.to_datetime(dataset.scan_time.data[0].item())
    start_time = start_time.strftime("%Y%m%d%H%M%S")
    end_time = pd.to_datetime(dataset.scan_time.data[-1].item())
    end_time = end_time.strftime("%Y%m%d%H%M%S")
    filename = f"{prefix}_{start_time}_{end_time}.nc"
    save_scene(valid, output_path / filename)


def write_training_samples_3d(
        output_path: Path,
        prefix: str,
        data : xr.Dataset,
        n_scans: int = 128,
        n_pixels: int = 64,
        overlapping: bool = True,
        min_valid = 20,
        reference_var: str = "surface_precip"
):
    """
    Write training data in GPROF-NN 3D format.

    Args:
        output_path: Path pointing to the directory to which to
            write the training files.
        data: xarray.Dataset containing GPM observations from which to
            extract training scenes.
        n_scans: The number of scans in a single scene.
        n_pixels: The number of pixels in a single scene.
        min_valid: The minimum number of pixels with valid surface
            precipitation.
        overlapping: Boolean flag indicating whether or not the
            extracted scenes should be overlapping or not.
        reference_var: The variable to use to determine valid
            pixels.

    """
    scenes = extract_scenes(
        data,
        n_scans,
        n_pixels,
        min_valid=min_valid,
        overlapping=overlapping,
        reference_var=reference_var
    )

    for scene in scenes:
        start_time = pd.to_datetime(scene.scan_time.data[0].item())
        start_time = start_time.strftime("%Y%m%d%H%M%S")
        end_time = pd.to_datetime(scene.scan_time.data[-1].item())
        end_time = end_time.strftime("%Y%m%d%H%M%S")
        filename = f"{prefix}_{start_time}_{end_time}.nc"
        save_scene(scene, output_path / filename)
