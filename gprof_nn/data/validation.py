"""
========================
gprof_nn.data.validation
========================

This module provides functionality to download and process GPM ground
validation data.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import logging
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
from pyresample import geometry, kd_tree
from rich.progress import Progress
import scipy
from scipy.interpolate import LinearNDInterpolator
from scipy.signal import convolve
import xarray as xr

from gprof_nn import augmentation
from gprof_nn.logging import get_console
from gprof_nn.utils import great_circle_distance
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.preprocessor import run_preprocessor


LOGGER = logging.getLogger(__name__)


_BASE_URL = "https://pmm-gv.gsfc.nasa.gov/pub/NMQ/level2/"


PATHS = {
    "GMI": "GPM/",
    "TMIPO": "TRMM/"
}

LINK_REGEX = re.compile(r"<a href=\"([\w\.]*)\">")
PRECIPRATE_REGEX = re.compile(r"PRECIPRATE(\.HSR)?\.GC\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz")
MASK_REGEX = re.compile(r"MASK\.(\d{8})\.(\d{6})\.(\d{5})[\w\.]*\.gz")
RQI_REGEX = re.compile(r"RQI\.(\d{8})\.(\d{6})\.(\d{5})[\w\.]*\.gz")


# lon_min, lat_min, lon_max, lat_max of CONUS
CONUS = [-130.0, 20, -60, 55]


def download_file(sensor, filename, year, month, destination):
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
    if len(precip_files) == 0:
        raise ValueError(
            "Didn't find any files matching the REGEX for precipitation data."
        )

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

    dims = ("time", "latitude", "longitude")
    data = {
        "latitude": (("latitude",), lats),
        "longitude": (("longitude",), lons),
        "time": (("time",), times),
        "surface_precip": (dims, precip_rate),
    }

    rqi_files = [f for f in files if RQI_REGEX.match(f.name)]
    rqi_files = sorted(rqi_files, key=get_date)
    if len(rqi_files) == 0:
        raise ValueError(
            "Didn't find any files matching the REGEX for RQI data."
        )
    rqi = np.zeros((len(times), n_rows, n_cols), dtype=np.float32)
    for i, f in enumerate(rqi_files):
        rqi[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.float32)
    print(rqi.max())
    rqi[rqi < 0.0] = np.nan
    data["radar_quality_index"] = (dims, rqi)

    mask_files = [f for f in files if MASK_REGEX.match(f.name)]
    mask_files = sorted(mask_files, key=get_date)
    if len(mask_files) > 0:
        mask = np.zeros((len(times), n_rows, n_cols), dtype=np.float32)
        for i, f in enumerate(mask_files):
            mask[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.int32)
        data["mask"] = (dims, mask)

    return xr.Dataset(data).sortby(["time"])


def shift(latitude, longitude, d):
    """
    Shifts a sequency of latitude and longitude coordinates a given distance
    into the direction opposite that of the sequence.

    The shifted latitudes are calculated by computing the local horizontal
    direction that is orthogonal to the sequence direction. This vector is
    then used to translate the sequence points. Doesn't take into account
    curvature of the globe so results are only approximative and will likely
    deteriorate with increasing d.

    Args:
        latitude: 1D array of latitude.
        longitude: 1D array of longitudes.

    Return:
        A tuple ``(lats, lons)`` of shifted latitudes and longitudes.
    """

    # Unit vector perpendicular to surface
    xyz = np.stack(
        augmentation.latlon_to_ecef().transform(
            longitude, latitude, np.zeros_like(latitude), radians=False
        ),
        axis=-1,
    )

    xyz_1 = np.stack(
        augmentation.latlon_to_ecef().transform(
            longitude, latitude, np.ones_like(latitude), radians=False
        ),
        axis=-1,
    )

    a = xyz[1:] - xyz[:-1]
    a_e = np.zeros((a.shape[0] + 1, 3))
    a_e[1:] += 0.5 * a
    a_e[:-1] += 0.5 * a
    a_e[0] *= 2
    a_e[-1] *= 2
    a = a_e

    x = np.cross(a_e, xyz_1 - xyz)
    x /= np.sqrt((x ** 2).sum(axis=-1, keepdims=True))

    k = np.exp(-((np.arange(-10, 10.1) / 5) ** 2))
    k = k / k.sum()
    k = k.reshape(-1, 1)

    x = convolve(x, k, mode="same")
    x[:10] = x[10]
    x[-10:] = x[-11]

    xyz = xyz + x * d

    lons, lats, _ = augmentation.ecef_to_latlon().transform(
        xyz[..., 0], xyz[..., 1], xyz[..., 2], radians=False
    )

    return lats, lons


def unify_grid(latitude, longitude):
    """
    Give an arbitrary grid of latitude and longitude coordinates this
    function computes a grid that is approximately rectangular and equidistant
    grid with a resolution of 5 km. Accuracy in across track direction is a
    few percent while in along-track direction it was found to be as large as
    10%. Anyways, this was the best solution I could find for this problem.

    Args:
        latitude: A 2D array of latitude coordinates.
        longitude. A 2D array of longitude coordinates.

    Return:
       A tuple ``(lats, lons)`` containing the latitude and longitude coordinates
       of the equidistant grid.
    """
    N = latitude.shape[1]
    xyz = np.stack(
        augmentation.latlon_to_ecef().transform(
            longitude, latitude, np.zeros_like(latitude), radians=False
        ),
        axis=-1,
    )

    # Unit vector perpendicular to surface
    xyz_1 = np.stack(
        augmentation.latlon_to_ecef().transform(
            longitude, latitude, np.ones_like(latitude), radians=False
        ),
        axis=-1,
    )

    slices_lat = []
    slices_lon = []

    # Start with central pixels
    lats_c = latitude[:, N // 2].copy()
    lons_c = longitude[:, N // 2].copy()

    d_lon = np.diff(lons_c)
    if (d_lon > 180).any():
        ind = np.where(d_lon > 180)[0][0]
        lons_c[: ind + 1] += 360
    if (d_lon < -180).any():
        ind = np.where(d_lon < -180)[0][0]
        lons_c[ind + 1 :] += 360

    d = great_circle_distance(lats_c[1:], lons_c[1:], lats_c[:-1], lons_c[:-1])
    d = np.concatenate([np.array([0]), np.cumsum(d)])
    lats_c_5 = np.interp(np.arange(0, d.max(), 5e3), d, lats_c)
    lons_c_5 = np.interp(np.arange(0, d.max(), 5e3), d, lons_c)

    lons_c_5[lons_c_5 < -180] += 360
    lons_c_5[lons_c_5 > 180] -= 360

    slices_lat.append(lats_c_5)
    slices_lon.append(lons_c_5)

    for i in range(100):
        lats_5, lons_5 = shift(slices_lat[0], slices_lon[0], -5e3)
        slices_lat.insert(0, lats_5)
        slices_lon.insert(0, lons_5)

    for i in range(100):
        lats_5, lons_5 = shift(slices_lat[-1], slices_lon[-1], 5e3)
        slices_lat.append(lats_5)
        slices_lon.append(lons_5)

    lats = np.stack(slices_lat, axis=-1)
    lons = np.stack(slices_lon, axis=-1)

    return lats, lons


def calculate_angles(l1c_data):
    """
    For L1C data from a conical scanner, calculate the angle that each
    footprint in the scan is rotated with respect to along-track
    direction.

    Args:
        l1c_data: xarray.Dataset containing the L1C data.

    Return:
        2D array containing the angles between local along-track direction
        and the projection of the viewing vector onto the horizontal local
        horizontal plane.
    """
    lats = l1c_data.latitude
    lats_sc = l1c_data.spacecraft_latitude
    lats, lats_sc = xr.broadcast(lats, lats_sc)
    lats = lats.data
    lats_sc = lats_sc.data

    lons = l1c_data.longitude
    lons_sc = l1c_data.spacecraft_longitude
    lons, lons_sc = xr.broadcast(lons, lons_sc)
    lons = lons.data
    lons_sc = lons_sc.data

    alt_sc = l1c_data.spacecraft_altitude
    alt_sc, _ = xr.broadcast(l1c_data.longitude, alt_sc)
    alt_sc = alt_sc.data

    t = augmentation.latlon_to_ecef()
    xyz = np.stack(t.transform(
        lons, lats, np.zeros_like(lons), radians=False
    ), axis=-1)
    xyz_1 = np.stack(t.transform(
        lons, lats, np.ones_like(lons), radians=False
    ), axis=-1)
    xyz_sc = np.stack(t.transform(
        lons_sc, lats_sc, alt_sc, radians=False
    ), axis=-1)

    # Calculate local along track vector
    a = xyz[1:] - xyz[:-1]
    a /= np.sqrt((a ** 2).sum(axis=-1, keepdims=True))
    a_e = np.zeros((a.shape[0] + 1, a.shape[1], 3))
    a_e[:-1] += 0.5 * a
    a_e[1:] += 0.5 * a
    a_e[0] *= 2
    a_e[-1] *= 2
    a = a_e

    # Calculate local across track vector
    z = xyz_1 - xyz
    z = z / np.sqrt((z ** 2).sum(axis=-1, keepdims=True))
    x = np.cross(z, a)

    v = xyz - xyz_sc
    v_p_a = (v * a).sum(axis=-1)
    v_p_x = (v * x).sum(axis=-1)
    return np.rad2deg(np.arctan2(v_p_x, v_p_a))


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
        parts = Path(filename).name.split(".")
        if parts[-3] in ["gz", "asc", "dat"]:
            end = -4
        else:
            end = -3
        date = parts[end - 2: end]

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
        parts = Path(filename).name.split(".")
        if parts[-3] in ["gz", "asc", "dat"]:
            end = -4
        else:
            end = -3
        granule = parts[end]
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
                download_file(self.sensor, f, year, month, Path(tmp) / f) for f in files
            ]
            return open_validation_data(local_files)


class ValidationFileProcessor:
    """
    Processor class to extract satellite radar co-locations to validate the GPROF
    retrieval.
    """

    def __init__(self, sensor, year, month):
        """
        Create processor to extract co-locations for a give sensor, month and
        year.

        Args:
            sensor: The sensor for which to extract the co-locations.
            month: The month for which to extract the co-locations.
            year: The year for which to extract the co-locations.
        """
        self.sensor = sensor
        self.month = month
        self.year = year
        self.validation_data = ValidationData(sensor)
        self.granules = self.validation_data.get_granules(year, month)

    def process_mrms_file(self, granule, l1c_file, mrms_file, scan_start, scan_end):
        """
        Helper function to process a single MRMS file.

        Downloads the MRMS data and interpolates it to a 5km x 5km grid
        derived from the L1C file.

        Args:
            granule: Index identifying the granules for which to extract
                the reference data.
            l1c_file: Path to the corresponding L1C file.
            mrms_file: Filename to which to write the results.
            scan_start: Index of the first scan covering CONUS
            scan_end: Index of the last scan covering CONUS
        """
        mrms_data = self.validation_data.open_granule(self.year, self.month, granule)
        l1c_data = L1CFile(l1c_file).to_xarray_dataset()
        lats = l1c_data.latitude.data
        lons = l1c_data.longitude.data
        angles = calculate_angles(l1c_data)

        # Calculate 5km x 5km grid.
        lats_5, lons_5 = unify_grid(lats, lons)
        lats_5 = xr.DataArray(data=lats_5, dims=["along_track", "across_track"])
        lons_5 = xr.DataArray(data=lons_5, dims=["along_track", "across_track"])

        swath = geometry.SwathDefinition(lats=lats, lons=lons)
        swath_5 = geometry.SwathDefinition(lats=lats_5, lons=lons_5)
        angles = kd_tree.resample_nearest(
            swath, angles, swath_5, radius_of_influence=20e3
        )

        scans = xr.DataArray(
            data=np.linspace(l1c_data.scans[0], l1c_data.scans[-1], lons_5.shape[0]),
            dims=["along_track"],)
        dtype = l1c_data.scan_time.dtype
        time = l1c_data.scan_time.astype(np.int64).interp({"scans": scans})
        time = time.astype(dtype)
        time, lons_5 = xr.broadcast(time, lons_5)

        # Smooth and interpolate surface precip

        surface_precip = mrms_data.surface_precip
        k = np.arange(-5, 5 + 1e-6, 1) / 2.5
        k2 = (k.reshape(-1, 1) ** 2) + (k.reshape(1, -1) ** 2)
        k = np.exp(np.log(0.5) * k2)
        k /= k.sum()
        k = k[np.newaxis]

        sp = np.nan_to_num(mrms_data.surface_precip.data)
        counts = np.isfinite(mrms_data.surface_precip.data).astype(np.float32)

        # Use direct method to avoid negative values in results.
        sp_mean = convolve(sp, k, mode="same", method="direct")
        sp_cts = convolve(counts, k, mode="same", method="direct")
        sp = sp_mean / sp_cts
        # Set pixel with too few valid neighboring pixels to nan.
        sp[sp_cts < 1e-1] = np.nan

        surface_precip.data[:] = sp
        surface_precip = surface_precip.interp(
            latitude=lats_5, longitude=lons_5, time=time
        )
        datasets = [surface_precip]
        rqi = mrms_data.radar_quality_index.interp(
            latitude=lats_5, longitude=lons_5, time=time, method="nearest"
        )
        datasets.append(rqi)
        if "mask" in mrms_data.variables:
            mask = mrms_data.mask.interp(
                latitude=lats_5, longitude=lons_5, time=time, method="nearest"
            )
            datasets.append(mask)
        mrms_data = xr.merge(datasets)
        mrms_data["angles"] = (("along_track", "across_track"), angles)

        mrms_data.attrs["sensor"] = self.sensor.sensor_name
        mrms_data.attrs["platform"] = self.sensor.platform.name
        mrms_data.attrs["granule"] = granule
        mrms_data.attrs["scan_start"] = scan_start
        mrms_data.attrs["scand_end"] = scan_end

        # Remove empty scan lines.
        has_data = np.any(np.isfinite(mrms_data.surface_precip.data), -1)
        indices = np.where(has_data)[0]

        if len(indices) == 0:
            raise ValueError(
                "No valid precipitation pixels in overpass."
                )
        along_track_start = indices.min()
        along_track_end = indices.max()
        mrms_data = mrms_data[{
            "along_track": slice(along_track_start, along_track_end)
        }]
        mrms_data.to_netcdf(mrms_file)

    def process_granule(self, granule, mrms_file, preprocessor_file):
        """
        Helper function to extract validation data for a single granule.
        This function does two things:
            - Find and read L1C file.
            - If file with MRMS results doesn't exist yet, download the
              data and interpolate to 5km x 5km grid.
            - Run preprocessor and save file.

        Args:
            granule: The number of the granule to process.
            mrms_file: The name of the file to which to write the MRMS
                reference data.
            preprocessor_file: The name of the file to which to write the
                results from the preprocessor run.
        """
        mrms_file = Path(mrms_file)
        preprocessor_file = Path(preprocessor_file)

        l1c_file = L1CFile.open_granule(granule, self.sensor.l1c_file_path, self.sensor)
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            l1c_sub_file = tmp / "l1c_file.HDF5"
            scan_start, scan_end = l1c_file.extract_scans(CONUS, l1c_sub_file)

            # Extract reference data from MRMS file.
            if not mrms_file.exists():
                self.process_mrms_file(
                    granule, l1c_sub_file, mrms_file, scan_start, scan_end
                )

            # Run preprocessor on L1C file.
            run_preprocessor(
                str(l1c_sub_file),
                self.sensor,
                configuration="ERA5",
                output_file=preprocessor_file,
            )

    def run(self, mrms_path, pp_path, n_workers=4):
        """
        Run validation file processor for all files in the month.

        Args:
            mrms_path: Root of the directory tree to which to write the MRMS
                reference files.
            pp_path: Root of the directory tree to which to write the
                preprocessor files.
            n_workers: How many worker processes to use.
        """
        mrms_file_pattern = (
            "mrms_{sensor}_{year}{month:02}{day:02}_{hour:02}{minute:02}_{granule}.nc"
        )
        pp_file_pattern = (
            "{sensor}_{year}{month:02}{day:02}_{hour:02}{minute:02}_{granule}.pp"
        )

        mrms_path = Path(mrms_path) / f"{self.year}" / f"{self.month:02}"
        mrms_path.mkdir(exist_ok=True, parents=True)
        pp_path = Path(pp_path) / f"{self.year}" / f"{self.month:02}"
        pp_path.mkdir(exist_ok=True, parents=True)

        tasks = []
        # Submit task for each granule in month
        pool = ProcessPoolExecutor(max_workers=n_workers)
        for granule in self.granules:

            granule_file = self.granules[granule][0]
            date = ValidationData.filename_to_date(granule_file)

            fname_kwargs = {
                "sensor": self.sensor.name.lower(),
                "year": date.year,
                "month": date.month,
                "day": date.day,
                "hour": date.hour,
                "minute": date.minute,
                "granule": granule
            }
            mrms_file = mrms_path / mrms_file_pattern.format(**fname_kwargs)
            pp_file = pp_path / pp_file_pattern.format(**fname_kwargs)
            tasks.append(pool.submit(self.process_granule, granule, mrms_file, pp_file))

        # Collect results and track progress.
        with Progress(console=get_console()) as progress:
            pbar = progress.add_task("Extracting validation data:", total=len(tasks))
            for task, granule in zip(tasks, self.granules):
                try:
                    task.result()
                except Exception as e:
                    LOGGER.error(
                        "The following error occurred when processing granule "
                        "%s: \n %s",
                        granule,
                        e,
                    )

                progress.advance(pbar)
