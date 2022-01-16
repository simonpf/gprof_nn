"""
========================
gprof_nn.data.validation
========================

This module provides functionality to download and process GPM ground
validation data.
"""
from datetime import datetime
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
import scipy
from scipy.interpolate import LinearNDInterpolator
import xarray as xr

from gprof_nn import augmentation

_BASE_URL = "https://pmm-gv.gsfc.nasa.gov/pub/NMQ/level2/"


PATHS = {
    "GMI": "GPM/"
}

LINK_REGEX = re.compile(
    r"<a href=\"([\w\.]*)\">"
)
PRECIPRATE_REGEX = re.compile(
    r"PRECIPRATE\.GC\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz"
)
MASK_REGEX = re.compile(
    r"MASK\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz"
)
RQI_REGEX = re.compile(
    r"RQI\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz"
)


def download_file(sensor, filename, destination):
    """
    Download validation file.

    Args:
        sensor: Sensor object representing the sensor for which to
            download the validation data.
        filename: The name of the file to download.
        destination: Path to the output file to which to write the downloaded
            data

    Return:

        The path to the downloaded file.
    """
    date = ValidationData.filename_to_date(filename)
    year = date.year
    month = date.month
    url = _BASE_URL + PATHS[sensor.name] + f"{year:04}/{month:02}/"
    url = url + filename
    with open(destination, "wb") as output:
        output.write(urlopen(url).read())
    return destination


def open_validation_data(files):
    """
    Open the validation data for a given granule number
    as xarray.Dataset.

    Args:
        granule_number: GPM granule number for which to open the validation
             data.
        base_directory: Path to root of the directory tree containing the
             validation data.

    Returns:
        xarray.Dataset containing the validation data.
    """
    # Load precip-rate data.
    precip_files = [f for f in files if PRECIPRATE_REGEX.match(f.name)]
    times = [ValidationData.filename_to_date(f) for f in precip_files]

    header = np.loadtxt(files[0], usecols=(1,), max_rows=6)
    n_cols = int(header[0])
    n_rows = int(header[1])
    lon_ll = float(header[2])
    lat_ll = float(header[3])
    dl = float(header[4])

    lons = lon_ll + np.arange(n_cols) * dl
    lats = (lat_ll + np.arange(n_rows) * dl)[::-1]

    get_date = ValidationData.filename_to_date
    precip_files = sorted(precip_files, key=get_date)
    precip_rate = np.zeros((len(times), n_rows, n_cols))
    for i, f in enumerate(precip_files):
        precip_rate[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.float32)
    precip_rate[precip_rate < 0.0] = np.nan

    rqi_files = [f for f in files if RQI_REGEX.match(f.name)]
    rqi_files = sorted(rqi_files, key=get_date)
    rqi = np.zeros((len(times), n_rows, n_cols), dtype=np.float32)
    for i, f in enumerate(rqi_files):
        rqi[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.float32)
    rqi[rqi < 0.0] = np.nan

    mask_files = [f for f in files if MASK_REGEX.match(f.name)]
    mask_files = sorted(mask_files, key=get_date)
    mask = np.zeros((len(times), n_rows, n_cols), dtype=np.int32)
    for i, f in enumerate(mask_files):
        mask[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.int32)

    dims = ("time", "latitude", "longitude")
    data = {
        "latitude": (("latitude",), lats),
        "longitude": (("longitude",), lons),
        "time": (("time",), times),
        "precip_rate": (dims, precip_rate),
        "mask": (dims, mask),
        "radar_quality_index": (dims, rqi)
    }

    return xr.Dataset(data).sortby(["time"])


def upsample_swath(swath_data, variables):

    if "latitude" not in variables:
        variables.append("latitude")

    lats = swath_data.latitude.data
    lons = swath_data.longitude.data

    swath = augmentation.Swath(lats, lons)

    x_5 = np.linspace(-500e3, 500e3, 201)
    y_5 = np.arange(swath.y.min(), lats.shape[0] * swath.y_a, 5e3)
    xy_5 = np.stack(np.meshgrid(y_5, x_5, indexing="ij"))

    print(xy_5[1])
    coords = swath.euclidean_to_pixel_coordinates(xy_5)

    results = {}
    for variable in variables:
        # Special treatment for longitude below.
        if variable == "longitude":
            continue
        data_v = swath_data[variable].data
        data_r = augmentation.extract_domain(data_v, coords)

        dims = ("x", "y") + swath_data[variable].dims[2:]
        results[variable] = (dims, data_r)

    # Special treatment of longitudes to avoid artifacts due to
    # interpolation over longitude jumps.
    lons = swath_data["longitude"].data.copy()
    d_l = lons[1:] - lons[:-1]
    if np.any(d_l > 180):
        for i in range(lons.shape[1]):
            indices = np.where(d_l[:, i] > 180)[0]
            if len(indices) > 0:
                lons[:indices[0] + 1, i] += 360
    if np.any(d_l < -180):
        for i in range(lons.shape[1]):
            indices = np.where(d_l[:, i] < -180)[0]
            if len(indices) > 0:
                print(indices[0])
                lons[indices[0] + 1:, i] += 360
    lons_r = augmentation.extract_domain(lons, coords)
    lons_r[lons_r < -180] += 360
    lons_r[lons_r > 180] -= 360
    dims = ("x", "y")
    results["longitude"] = (dims, lons_r)

    return xr.Dataset(results)


def unify_grid(latitude, longitude):

    N = latitude.shape[1]
    xyz = np.stack(
        augmentation.latlon_to_ecef().transform(
            longitude,
            latitude,
            np.zeros_like(latitude),
            radians=False),
        axis=-1
    )

    # Unit vector perpendicular to surface
    xyz_1 = np.stack(
        augmentation.latlon_to_ecef().transform(
            longitude,
            latitude,
            np.ones_like(latitude),
            radians=False
        ),
        axis=-1
    )

    d_a = np.zeros_like(latitude)

    a = xyz[1:] - xyz[:-1]
    a = a / np.sqrt((a ** 2).sum(axis=-1, keepdims=True))
    a_e = np.zeros((a.shape[0] + 1, a.shape[1], a.shape[2]))
    a_e[1:] += 0.5 * a
    a_e[:-1] += 0.5 * a
    a_e[0] *= 2
    a_e[-1] *= 2

    d_a[1:, :] = np.cumsum(np.sum((xyz[1:] - xyz[:-1]) * a, axis=-1), axis=0)
    d_a += np.sum((xyz - xyz[:, [N // 2]]) * a_e, axis=-1)

    z = xyz_1[:, N // 2] - xyz[:, N // 2]
    x = np.cross(a_e[:, N // 2], z)
    x = x / np.sqrt((x ** 2).sum(axis=-1, keepdims=True))

    d_x = np.sum((xyz - xyz[:, [N // 2]]) * x[:, np.newaxis, :], axis=-1)

    #return d_a, d_x

    points = np.stack([d_a.ravel(), d_x.ravel()], axis=-1)
    values = latitude

    a = np.arange(d_a.min(), d_a.max(), 5e3)
    x = np.linspace(-500e3, 500e3, 201)[::-1]
    x, a = np.meshgrid(x, a)
    print(x)

    f = LinearNDInterpolator(
        points,
        np.stack([latitude.ravel(), longitude.ravel()], axis=-1),
        fill_value=np.nan
    )

    points_i = np.stack([a.ravel(), x.ravel()], axis=-1)
    result = f(points_i)

    lats = result[..., 0].reshape(-1, 201)
    lons = result[..., 1].reshape(-1, 201)

    return lats, lons





class ValidationData:
    """
    Interface class to download and open validation data.
    """

    @staticmethod
    def filename_to_date(filename):
        """
        Parse date from filename.

        Args:
            Name of the validation data file.

        Return:
            The data as a Python datetime object.
        """
        date = Path(filename).name.split(".")[-5:-3]
        return datetime.strptime("".join(date), "%Y%m%d%H%M%S")

    @staticmethod
    def filename_to_granule(filename):
        """
        Parse granule number from filename.

        Args:
            Name of the validation data file.

        Return:
            The granule number as an integer.
        """
        granule = Path(filename).name.split(".")[-3]
        return int(granule)


    def __init__(self, sensor):
        self.sensor = sensor

    def get_granules(self, year, month):
        """
        List available files for a given year and month.

        Return:
            A dict mapping granule numbers to corresponding ground
            validation files.
        """
        url = _BASE_URL + PATHS[self.sensor.name] + f"{year:04}/{month:02}/"
        html = urlopen(url).read().decode()

        results = {}
        for match in LINK_REGEX.finditer(html):
            filename = match.group(1)
            granule = self.filename_to_granule(filename)
            results.setdefault(granule, []).append(filename)
        return results


    def open_granule(self, year, month, granule_number):
        """
        Download and open validation files from a given year and month.

        Args:
            year: The year.
            month: The month.
            granule_number: The requested granule number.
        """
        granules = self.get_granules(year, month)
        if not granule_number in granules:
            raise ValueError(
                f"The requested granule {granule_number} is not found in the"
                f" files from month {year}/{month}"
            )

        # Download files
        with TemporaryDirectory() as tmp:
            files = granules[granule_number]
            local_files = [
                download_file(self.sensor, f, Path(tmp) / f) for f in files
            ]
            return open_validation_data(local_files)


class ValidationFileProcessor:

    def __init__(self, sensor, month, year):
        self.granules = self.list_file


    def process_granule(self, granule):
        pass
    


