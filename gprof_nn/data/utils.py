"""
===================
gprof_nn.data.utils
===================

Functions that are shared across multiple sub modules of
the ``gprof_nn.data`` module.
"""
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pansat import Granule, TimeRange
from pansat.utils import resample_data
from pyresample.geometry import SwathDefinition
import xarray as xr

from gprof_nn.augmentation import extract_domain
from gprof_nn.definitions import LIMITS, ANCILLARY_VARIABLES
from gprof_nn.utils import apply_limits
from gprof_nn.data import preprocessor
from pansat.products.satellite.gpm import (
    l1c_gpm_gmi,
    l1c_npp_atms,
    l1c_noaa20_atms,
    l1c_gcomw1_amsr2,
    merged_ir
)


PANSAT_PRODUCTS = {
    "gmi": (l1c_gpm_gmi,),
    "atms": (l1c_npp_atms, l1c_noaa20_atms),
    "amsr2": (l1c_gcomw1_amsr2,)
}


UPSAMPLING_FACTORS = {
    "gmi": (3, 1),
    "atms": (3, 3,),
    "amsr2": (1, 1)
}


RADIUS_OF_INFLUENCE = {
    "gmi": 20e3,
    "atms": 100e3,
    "amsr2": 10e3
}


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
    variables = (
        ANCILLARY_VARIABLES +
        targets +
        ["brightness_temperatures", "source"]
    )
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
    variables = (
        ["brightness_temperatures"] +
        ANCILLARY_VARIABLES +
        targets
    )

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
        reference_var: str = "surface_precip",
        offset: int = Optional[None]
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
        offset: An optional, random offset  applied to the center of each extracted
            scene.

    Return:
        A list of scenes extracted from the provided observations.
    """
    n_scans_tot = data.scans.size
    n_pixels_tot = data.pixels.size

    def get_valid(dataset):
        if isinstance(reference_var, (list, tuple)):
            valid = np.isfinite(dataset[reference_var[0]])
            for ref_var in reference_var[1:]:
                valid *= np.isfinite(dataset[ref_var])
        else:
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

        if offset is not None:
            lower_bound = max(scan_cntr - offset, n_scans // 2)
            upper_bound = min(scan_cntr + offset, n_scans_tot - n_scans // 2)
            if lower_bound < upper_bound:
                scan_cntr = np.random.randint(lower_bound, upper_bound)
            lower_bound = max(pixel_cntr - offset, n_pixels // 2)
            upper_bound = min(pixel_cntr + offset, n_pixels_tot - n_scans // 2)
            if lower_bound < upper_bound:
                pixel_cntr = np.random.randint(lower_bound, upper_bound)

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
        reference_var: str = "surface_precip"
) -> None:
    """
    Write training data in GPROF-NN 1D format.

    Args:
        dataset: An 'xarray.Dataset' containing collocated input
            observations and reference data.
        output_path: Path to which the training data will be written.
    """
    if "pixels_center" in dataset.dims:
        dataset = dataset[{"pixels": slice(*compressed_pixel_range())}]
    mask = np.isfinite(dataset[reference_var].data)
    if mask.ndim > 2:
        mask = mask.all(-1)

    valid = {}
    for var in dataset.variables:
        if var == "angles":
            continue
        arr = dataset[var]
        if arr.data.ndim < 2:
            arr_data = np.broadcast_to(arr.data[..., None], mask.shape)
        else:
            arr_data = arr.data
        valid[var] = ((("samples",) + arr.dims[2:]), arr_data[mask])

    valid = xr.Dataset(valid, attrs=dataset.attrs)
    if "angles" in dataset:
        valid["angles"] = (("angles",), dataset.angles.data)
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


def extract_scans(
        granule: Granule,
        dest: Path,
        min_scans: Optional[int] = None
) -> Path:
    """
    Extract and write scans from L1C file into a separate file.

    Args:
        granule: A pansat granule specifying a subset of an orbit.
        dest: A directory to which the extracted scans will be written.
        min_scans: A minimum number of scans to extract.

    Return:
        The path of the file containing the extracted scans.
    """
    from gprof_nn.data.l1c import L1CFile
    scan_start, scan_end = granule.primary_index_range
    n_scans = scan_end - scan_start
    if min_scans is not None and n_scans < min_scans:
        scan_c = (scan_end + scan_start) // 2
        scan_start = scan_c - min_scans // 2
        scan_end = scan_start + min_scans
    l1c_path = granule.file_record.local_path
    l1c_file = L1CFile(granule.file_record.local_path)
    output_filename = dest / l1c_path.name
    l1c_file.extract_scan_range(scan_start, scan_end, output_filename)
    return output_filename


def run_preprocessor(gpm_granule: Granule) -> xr.Dataset:
    """
    Run preprocessor on a GPM granule.

    Args:
        gpm_granule: A pansat granule identifying a subset of an orbit
            of GPM L1C files.

    Return:
        An xarray.Dataset containing the results from the preprocessor.
    """
    from gprof_nn.data.l1c import L1CFile
    old_dir = os.getcwd()

    try:
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            l1c_file = extract_scans(gpm_granule, tmp, min_scans=128)
            os.chdir(tmp)
            sensor = L1CFile(l1c_file).sensor
            preprocessor_data = preprocessor.run_preprocessor(
                l1c_file, sensor, robust=False
            )
    finally:
        os.chdir(old_dir)

    invalid = preprocessor_data.brightness_temperatures.data < 0
    preprocessor_data.brightness_temperatures.data[invalid] = np.nan

    return preprocessor_data


def upsample_data(
        data: xr.Dataset,
        upsampling_factors: Tuple[int, int]
) -> xr.Dataset:
    """
    Upsample preprocessor data along scans and pixels.

    Args:
        data: An xarray.Dataset containing preprocessor data.
        upsampling_factors: A tuple describing the upsampling factors alon scans and pixels.

    Return:
        The preprocessor data upsampled by the given factors along scans and pixels.
    """
    float_vars = [
        "latitude", "longitude", "brightness_temperatures", "total_column_water_vapor", "two_meter_temperature",
        "moisture_convergence", "leaf_area_index", "snow_depth", "land_fraction", "ice_fraction", "elevation",
    ]
    scan_time = data["scan_time"]
    data = data[float_vars]

    n_scans = data.scans.size
    n_scans_up = upsampling_factors[0] * n_scans
    new_scans = np.linspace(data.scans[0], data.scans[-1], n_scans_up)

    n_pixels = data.pixels.size
    n_pixels_up = upsampling_factors[1] * n_pixels
    new_pixels = np.linspace(data.pixels[0], data.pixels[-1], n_pixels_up)

    data = data.interp(scans=new_scans, pixels=new_pixels).drop_vars(["pixels", "scans"])
    scan_time_int = scan_time.astype(np.int64)
    scan_time_new = scan_time_int.interp(scans=new_scans, method="nearest")
    data["scan_time"] = (("scans",), scan_time_new.data.astype("datetime64[ns]"))

    return data


def mask_invalid_values(preprocessor_data: xr.Dataset):
    """
    Mask unphysical values in preprocessor data.
    """
    # Two meter temperature
    data = preprocessor_data.two_meter_temperature.data
    invalid = (data < 0) * (data > 1_000)
    data[invalid] = np.nan

    # Total column water vapor
    data = preprocessor_data.total_column_water_vapor.data
    invalid = (data < 0) * (data > 1_000)
    data[invalid] = np.nan

    # Leaf area index
    data = preprocessor_data.leaf_area_index.data
    invalid = (data < 0) * (data > 1_000)
    data[invalid] = np.nan

    # Elevation
    data = preprocessor_data.elevation.data.astype("float32")
    invalid = (data < 0) * (data > 1_000)
    data[invalid] = np.nan

    # Land fraction
    data = preprocessor_data.land_fraction.data
    invalid = (data < 0) * (data > 100)
    data[invalid] = -1

    return preprocessor_data


def add_cpcir_data(
        preprocessor_data: xr.Dataset,
) -> xr.Dataset:
    """
    Add CPCIR 11um IR observations to the preprocessor data.

    Args:
        preprocessor_data: An xarray.Dataset containing the data from the preprocessor

    Return:
        The preprocessor data with an additional variable 'ir_observations' containing CPCIR 11 um
        Tbs if available.
    """
    scan_time_start = preprocessor_data.scan_time.data[0]
    scan_time_end = preprocessor_data.scan_time.data[-1]
    time_c = scan_time_start + 0.5 * (scan_time_end - scan_time_start)
    time_range = TimeRange(time_c)
    recs = merged_ir.get(time_range)

    if len(recs) == 0:
        preprocessor_data["ir_observations"] = (("scans", "pixels"), np.nan * np.zeros_like(preprocessor_data.longitude.data))
        return preprocessor_data

    with xr.open_dataset(recs[0].local_path) as cpcir_data:
        lons = cpcir_data.lon.data
        lats = cpcir_data.lat.data

        lat_min = preprocessor_data.latitude.data.min()
        lat_max = preprocessor_data.latitude.data.max()
        inds = np.where((lat_min <= lats) * (lats <= lat_max))[0]
        if len(inds) < 2:
            lat_start, lat_end = 0, None
        else:
            lat_start, lat_end = inds[[0, -1]]

        lon_min = preprocessor_data.longitude.data.min()
        lon_max = preprocessor_data.longitude.data.max()
        inds = np.where((lon_min <= lons) * (lons <= lon_max))[0]
        if len(inds) < 2:
            lon_start, lon_end = 0, None
        else:
            lon_start, lon_end = inds[[0, -1]]

        cpcir_tbs = cpcir_data.Tb[{"lat": slice(lat_start, lat_end), "lon": slice(lon_start, lon_end)}]

        scan_time = preprocessor_data.scan_time
        scan_time, _ = xr.broadcast(scan_time, preprocessor_data.longitude)

        cpcir_tbs = cpcir_tbs.interp(
            lat = preprocessor_data.latitude,
            lon = preprocessor_data.longitude,
        ).rename(time="ir_obs")
        preprocessor_data["ir_observations"] = cpcir_tbs

    return preprocessor_data


def lla_to_ecef(coords_lla: np.ndarray):
    """
    Converts latitude-longitude-altitude (LLA) coordinates to
    earth-centric earth-fixed coordinates (ECEF)

    Params:
        coords_lla: A numpy.ndarray containing the three coordinates oriented along the last axis.

    Return:
        coords_ecef: An array of the same shape as 'coords_lla' but containing the x, y, and z
             coordinates along the last axis.
    """
    SEM_A = 6_378_137.0
    SEM_B = 6_356_752.0
    ECC2 = 1.0 - (SEM_B ** 2 / SEM_A ** 2)

    lon = np.radians(coords_lla[..., 0])
    lat = np.radians(coords_lla[..., 1])
    alt = coords_lla[..., 2]

    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)

    x = (roc + alt) * np.cos(lat) * np.cos(lon)
    y = (roc + alt) * np.cos(lat) * np.sin(lon)
    z = (roc * (1 - ECC2) + alt) * np.sin(lat)

    return np.stack((x, y, z), -1)


def calculate_angles(
        fp_lons: np.ndarray,
        fp_lats: np.ndarray,
        sensor_lons: np.ndarray,
        sensor_lats: np.ndarray,
        sensor_alts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate zenith and azimuth angles describing the observation geometry.

    Args:
        fp_lons: Array containing the longitude coordinates of the observation
            footprints.
        fp_lats: Array containing the latitude coordinates of the observation
            footprints.
        sensor_lons: The longitude coordinates of the sensor.
        sensor_lats: The latitude coordinates of the sensor.
        sensor_alts: The altitude coordinates of the sensor.

    Return:
        A tuple ``zenith, azimuth`` containing the zenith and azimuth coordinates
        of all lines of sights.
    """
    sensor_lla = np.stack((sensor_lons, sensor_lats, sensor_alts), -1)
    sensor_ecef = lla_to_ecef(sensor_lla)

    fp_lla = np.stack((fp_lons, fp_lats, np.zeros_like(fp_lons)), -1)
    fp_ecef = lla_to_ecef(fp_lla)
    local_up = fp_ecef / np.linalg.norm(fp_ecef, axis=-1, keepdims=True)
    fp_west = fp_lla.copy()
    fp_west[..., 0] -= 0.1
    fp_west = lla_to_ecef(fp_west) - fp_ecef
    fp_west /= np.linalg.norm(fp_west, axis=-1, keepdims=True)
    fp_north = fp_lla.copy()
    fp_north[..., 1] += 0.1
    fp_north = lla_to_ecef(fp_north) - fp_ecef
    fp_north /= np.linalg.norm(fp_north, axis=-1, keepdims=True)

    if sensor_ecef.ndim < fp_lla.ndim:
        sensor_ecef = np.broadcast_to(sensor_ecef[..., None, :], fp_lla.shape)
    los = sensor_ecef - fp_ecef
    zenith = np.arccos((local_up * los).sum(-1) / np.linalg.norm(los, axis=-1))

    azimuth = np.arctan2((los * fp_west).sum(-1), (los * fp_north).sum(-1))
    azimuth = np.nan_to_num(azimuth, nan=0.0)

    return np.rad2deg(zenith), np.rad2deg(azimuth)


CHANNEL_REGEXP = re.compile("([\d\.\s\+\/-]*)\s*GHz\s*(\w*)-Pol")

POLARIZATIONS = {
    "H": 0,
    "QH": 1,
    "V": 2,
    "QV": 3,
}

BEAM_WIDTHS = {
    "gmi": [1.75, 1.75, 1.0, 1.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    "atms": [5.2, 5.2, 2.2, 1.1, 1.1, 1.1, 1.1, 1.1],
    "amsr2": [1.2, 1.2, 0.65, 0.65, 0.75, 0.75, 0.35, 0.35, 0.15, 0.15, 0.15, 0.15],
}

def calculate_obs_properties(
        preprocessor_data: xr.Dataset,
        granule: Granule,
        radius_of_influence: float = 5e3,
) -> xr.Dataset:
    """
    Extract observations and corresponding meta data from granule.

    Args:
        preprocessor_data: The preprocessor data to which to resample all
            observaitons.
        granule: A pansat granule defining the section of a orbit containing the overpass.
        radius_of_influence: The radius of influence to use for resampling the L1C observations.
    """
    from gprof_nn.data.l1c import L1CFile
    lons = preprocessor_data.longitude.data
    lats = preprocessor_data.latitude.data
    swath = SwathDefinition(lats=lats, lons=lons)

    observations = []
    meta_data = []

    l1c_file = L1CFile(granule.file_record.local_path)
    sensor = l1c_file.sensor.name.lower()

    granule_data = granule.open()
    if "latitude" in granule_data:
        pass

    else:
        swath_ind = 1
        while f"latitude_s{swath_ind}" in granule_data:

            freqs = []
            offsets = []
            pols = []

            for match in CHANNEL_REGEXP.findall(granule_data[f"tbs_s{swath_ind}"].attrs["LongName"]):
                freq, pol = match
                freq = freq.replace("/", "")
                if freq.find("+-") > 0:
                    freq, offs = freq.split("+-")
                    freqs.append(float(freq))
                    offsets.append(float(offs))
                else:
                    freqs.append(float(freq))
                    offsets.append(0.0)
                pols.append(POLARIZATIONS[pol])

            swath_data = granule_data[[
                f"longitude_s{swath_ind}",
                f"latitude_s{swath_ind}",
                f"tbs_s{swath_ind}",
                f"channels_s{swath_ind}"
            ]]

            fp_lons = swath_data[f"longitude_s{swath_ind}"].data
            fp_lats = swath_data[f"latitude_s{swath_ind}"].data
            sensor_lons = granule_data["spacecraft_longitude"].data
            sensor_lats = granule_data["spacecraft_latitude"].data
            sensor_alt = granule_data["spacecraft_altitude"].data * 1e3
            zenith, azimuth = calculate_angles(fp_lons, fp_lats, sensor_lons, sensor_lats, sensor_alt)
            sensor_alt = np.broadcast_to(sensor_alt[..., None], zenith.shape) / 100e3

            swath_data = swath_data.rename({
                f"longitude_s{swath_ind}": "longitude",
                f"latitude_s{swath_ind}": "latitude"
            })
            swath_data["sensor_alt"] = (("scans", "pixels"), sensor_alt)
            swath_data["zenith"] = (("scans", "pixels"), zenith)
            swath_data["azimuth"] = (("scans", "pixels"), azimuth)

            swath_data_r = resample_data(
                swath_data,
                swath,
                radius_of_influence=radius_of_influence,
            )
            sensor_alt = swath_data_r.sensor_alt.data
            zenith = swath_data_r.zenith.data
            azimuth = swath_data_r.azimuth.data

            for chan_ind in range(swath_data_r[f"channels_s{swath_ind}"].size):
                observations.append(swath_data_r[f"tbs_s{swath_ind}"].data[..., chan_ind])
                meta = np.stack((
                    freqs[chan_ind] * np.ones_like(observations[-1]),
                    offsets[chan_ind] * np.ones_like(observations[-1]),
                    pols[chan_ind] * np.ones_like(observations[-1]),
                    BEAM_WIDTHS[sensor][chan_ind] * np.ones_like(observations[-1]),
                    sensor_alt,
                    zenith,
                    np.sin(np.deg2rad(azimuth)),
                    np.cos(np.deg2rad(azimuth))
                ))
                meta_data.append(meta)

            swath_ind += 1

        observations = np.stack(observations)
        meta_data = np.stack(meta_data)
        return xr.Dataset({
            "observations": (("channels", "scans", "pixels"), observations),
            "meta_data": (("channels", "meta", "scans", "pixels"), meta_data)
        })
