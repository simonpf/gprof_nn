"""
==================
gprof_nn.data.kwaj
==================

This module provides functionality to download and process Radar
observations from the Kwajalein Atoll.
"""
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
import re
import shutil
import subprocess
import tarfile

import numpy as np
from pyresample import geometry, kd_tree
import xarray as xr

from gprof_nn.data.training_data import decompress_and_load
from gprof_nn.augmentation import latlon_to_ecef, ecef_to_latlon
from gprof_nn.data.validation import unify_grid, calculate_angles
from gprof_nn.data.l1c import L1CFile


LOGGER = logging.getLogger(__name__)


_BASE_URL = "https://pmm-gv.gsfc.nasa.gov/pub/gpmarchive/Radar/KPOL/"

LINK_REGEX = re.compile(r"<a href=\"(KPOL_[\w\.]*)\">")


LATITUDE = 8.71
LONGITUDE = 167.732
ALTITUDE = 24.0

ROI = [166.5, 7.5, 169.0, 10.0]

XYZ = None
N = None
W = None
U = None

VALIDATION_VARIABLES = ["ZZ", "RR", "RP", "RC"]


def get_overpasses(sensor_name):
    """
    Load overpasses over Kwaj radar for a given sensor.

    Args:
        sensor_name: The sensor name as string.

    Return:
        A dictionary mapping days of overpasses to corresponding
        granules.

    """
    data_path = Path(__file__).parent / ".." / "files"
    path = data_path / f"kwaj_overpasses_{sensor_name.lower()}.txt"
    files = open(path, "r").readlines()

    overpasses = {}
    for url in files:
        filename = url.split("/")[-1]
        date, granule = filename.split(".")[-4:-2]
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        granule = int(granule)

        key = (year, month, day)
        overpasses.setdefault(key, []).append(granule)

    return overpasses


def calculate_directions():
    """
    Calculates local Norh, West and Up vector for Kwajalein.
    """
    global N, W, U, XYZ

    t = latlon_to_ecef()
    lats = [LATITUDE] * 4
    lats[1] += 0.01
    lons = [LONGITUDE] * 4
    lons[2] += 0.01
    z = [ALTITUDE] * 4
    lats[3] += 0.01

    xyz = np.stack(t.transform(lons, lats, z, radians=False), axis=-1)

    XYZ = xyz[0]
    n = xyz[1] - xyz[0]
    N = n / np.sqrt(np.sum(n ** 2))
    w = xyz[2] - xyz[0]
    W = w / np.sqrt(np.sum(w ** 2))
    z = xyz[3] - xyz[0]
    U = z / np.sqrt(np.sum(z ** 2))

    return N


calculate_directions()


def weighting_function(distance):
    return np.exp(np.log(0.5) * (2.0  * (distance / 5e3)) ** 2)


def spherical_to_ecef(R, phi, theta):
    """
    Convert spherical radar coordinates to ECEF coordinates relative to the
    location of the Kwajalein radar.

    Args:
        R: The radial distance from the radar.
        phi: The azimuth angle.
        theta: The elevation angle.

    Return:
        Array with one more dimension than the broadcasted shapes of R, phi
        and theta containing the ECEF coordinates corresponding to the given
        coordinates. The x, y, z coordinates are oriented along the last
        dimension of the array.
    """
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    h = R * np.cos(theta)
    v = R * np.sin(theta)
    r = (v[..., np.newaxis] * U +
         (h * np.cos(phi))[..., np.newaxis] * N +
         (h * np.sin(phi))[..., np.newaxis] * W)
    return r


def extract_first_sweep(data):
    """
    Extracts the first sweep from a radar dataset.

    Args:
        data: xarray.Dataset containing the radar data.

    Return:
        A new 2D dataset with dimensions corresponding to range gate
        and azimuth beam direction.
    """
    rays = []

    e_0 = data.elevation[0]
    i = 0

    variables = ["ZZ", "RR", "RP", "RC"]

    while data.elevation[i].data <= 0.5:
        ray_start = int(data.ray_start_index[i].data)
        ray_end = int(data.ray_start_index[i + 1].data)
        rays.append(data[VALIDATION_VARIABLES][{"n_points": slice(ray_start, ray_end)}])
        i += 1
    dataset = xr.concat(rays, "rays").rename(n_points="range")
    dataset["azimuth"] = (("rays",), data.azimuth.data[:i])
    dataset["elevation"] = (("rays",), data.elevation.data[:i])
    dataset["time"] = (("rays",), data.time.data[:i])
    return dataset.assign_coords({"range": data.range.data})


class RadarFile:
    """
    The radar file class provides an inteface to download and read daily
    archives of radar observations from the Kwajalein radar.

    Note: This class will only be able to handle files in CDRadial format,
       i.e. files that are older than roughly 2017.
    """
    @staticmethod
    def get_files(year, month):
        """
        Return a dictionary of radar files available for a given year
        and month.

        Args:
            year: The year given as integer.
            month: The month given as integer.

        Return:
            results: A dictionary mapping tuples of ``(year, month, day)``
            corresponding to days for which radar data is available
            to corresponding filenames.
        """
        url = _BASE_URL + f"{year:04}/{month:02}/"
        html = urlopen(url).read().decode()

        results = {}
        for match in LINK_REGEX.finditer(html):
            filename = match.group(1)
            _, year, monthday = filename.split("_")
            year = int(year)
            month = int(monthday[:2])
            day = int(monthday[2:4])
            results[(year, month, day)] = filename
        return results

    @staticmethod
    def download_file(year, month, day, destination=None):
        """
        Download file archive for given date.

        Args:
            year: The year given as int
            month: The month given as int
            day: The day given as int
            destination: Optional folder to which to write the downloaded file.
                If None the current working directory is used.

        Return:
            Path pointing to the downloaded file.
        """
        files = RadarFile.get_files(year, month)
        if (year, month, day) not in files:
            raise ValueError(f"No files for {year}-{month:02}-{day:02}.")

        filename = files[(year, month, day)]
        url = _BASE_URL + f"{year:04}/{month:02}/" + filename

        if destination is None:
            output_file = filename
        else:
            output_file = Path(destination) / filename
        with open(output_file, "wb") as output:
            shutil.copyfileobj(urlopen(url), output)

        return output_file

    def __init__(self, filename):
        """
        Open radar archive for a given day.

        Args:
            filename: Path pointing to the compressed archive of daily
               radar observations downloaded from the PMM website.
        """
        self.filename = filename
        self.archive = tarfile.open(filename)

        self.times = {}
        for name in self.archive.getnames():
            filename = Path(name)
            if filename.suffix == ".gz":
                _, year, monthday, time = filename.name.split("_")
                month = monthday[:2]
                day = monthday[2:4]
                hour = time[:2]
                min = time[2:4]
                sec = time[4:6]
                datestr = f"{year}-{month}-{day}T{hour}:{min}:{sec}"
                date = np.datetime64(datestr)
                self.times[date] = name
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)

    def open_file(self, name):
        """
        Open a file from the archive as xarray.Dataset.

        Args:
            name: String containing the name of the file in the archive.

        Return:
            xarray.Dataset containing the radar of the sweep with the lowest
            elevation.
        """
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            extracted = tmp / "output.nc.gz"
            with open(extracted, "wb") as output:
                shutil.copyfileobj(self.archive.extractfile(name), output)

            args = ["gunzip", "-f", str(extracted)]
            subprocess.run(args)

            decompressed = tmp / "output.nc"
            dataset = xr.load_dataset(decompressed)

        data = extract_first_sweep(dataset)
        r = data.range.data.reshape(1, -1)
        phi = data.azimuth.data.reshape(-1, 1)
        theta = data.elevation.data.reshape(-1, 1)
        xyz_r = spherical_to_ecef(r, phi, theta)
        xyz = XYZ + xyz_r
        lons, lats, _ = ecef_to_latlon().transform(
            xyz[..., 0], xyz[..., 1], xyz[..., 2], radians=False
        )

        data["latitude"] = (("rays", "range"), lats.astype(np.float32))
        data["longitude"] = (("rays", "range"), lons.astype(np.float32))

        return data


    def open_files(self, date):
        """
        Open the observations closest to a given time from the archive.

        Args:
            date: The data for which to extract the observations.

        Return:
            An xarray.Dataset containing the observations closest in time
            to the given date.
        """
        times = np.array(list(self.times.keys()))
        names = list(self.times.values())
        delta = (times - date)
        print(times, date)

        closest = []
        indices = delta < np.timedelta64(0)
        delta_n = delta[indices]
        if len(delta_n) > 0:
            index = np.where(indices)[0][np.argmax(delta_n)]
            closest.append(names[index])

        indices = delta >= np.timedelta64(0)
        delta_p = delta[indices]
        if len(delta_p) > 0:
            index = np.where(indices)[0][np.argmin(delta_p)]
            closest.append(names[index])

        data = []
        for name in closest:
            print(name)
            data_t = self.open_file(name)
            data_t["time"] = data_t.time.mean()
            data.append(data_t)

        return xr.concat(data, "time")


class FileExtractor:
    def __init__(self, sensor):
        self.sensor = sensor
        self.overpasses = get_overpasses(sensor.name)

    def process_radar_file(self, granule, l1c_file, radar_file, scan_start, scan_end):

        data_directory = l1c_file.parent
        l1c_data = L1CFile(l1c_file).to_xarray_dataset()
        l1c_time = l1c_data.scan_time.mean()
        day = l1c_time.dt.day.data.item()
        month = l1c_time.dt.month.data.item()
        year = l1c_time.dt.year.data.item()


        radar_archive = list(data_directory.glob("*.tar.gz"))
        if len(radar_archive) > 0:
            radar_archive = RadarFile(radar_archive[0])
            if (radar_archive.day != day or
                radar_archive.month != month or
                radar_archive.year != year):
                radar_archive = RadarFile.download_file(year, month, day)
                radar_archive = RadarFile(radar_archive)
        else:
            #radar_archive = RadarFile.download_file(year, month, day)
            radar_archive = RadarFile(radar_archive)

        lats = l1c_data.latitude.data
        lons = l1c_data.longitude.data
        angles = calculate_angles(l1c_data)

        # Calculate 5km x 5km grid.
        lats_5, lons_5 = unify_grid(lats, lons)
        lats_5 = xr.DataArray(data=lats_5, dims=["along_track", "across_track"])
        lons_5 = xr.DataArray(data=lons_5, dims=["along_track", "across_track"])

        scans = xr.DataArray(
            data=np.linspace(l1c_data.scans[0], l1c_data.scans[-1], lons_5.shape[0]),
            dims=["along_track"],
        )
        dtype = l1c_data.scan_time.dtype
        time = l1c_data.scan_time.astype(np.int64).interp({"scans": scans})
        time = time.astype(dtype)
        time, lons_5 = xr.broadcast(time, lons_5)

        l1c_swath = geometry.SwathDefinition(lats=lats, lons=lons)
        swath_5 = geometry.SwathDefinition(lats=lats_5, lons=lons_5)

        angles = kd_tree.resample_nearest(
            l1c_swath, angles, swath_5, radius_of_influence=20e3
        )

        kwaj_data = radar_archive.open_files(l1c_time.data)
        datasets = []
        for t in range(kwaj_data.time.size):
            data = kwaj_data[{"time": t}]

            lats = data.latitude.data
            lons = data.longitude.data
            kwaj_swath = geometry.SwathDefinition(lats=lats, lons=lons)
            swath_5 = geometry.SwathDefinition(lats=lats_5, lons=lons_5)

            resampling_info = kd_tree.get_neighbour_info(
                kwaj_swath,
                swath_5,
                20e3,
                neighbours=8
            )
            valid_inputs, valid_outputs, indices, distances = resampling_info
            print(distances.min(), distances.max())

            data_r = {}
            for variable in VALIDATION_VARIABLES:
                resampled = kd_tree.get_sample_from_neighbour_info(
                    'custom', swath_5.shape, data[variable].data,
                    valid_inputs, valid_outputs, indices, distance_array=distances,
                    weight_funcs=weighting_function
                )
                print(data[variable].data.min(), data[variable].data.max())
                print(resampled.min(), resampled.max())
                data_r[variable] = (("along_track", "across_track"), resampled)
            data_r = xr.Dataset(data_r)
            data_r["time"] = (("time",), [data.time.data])
            datasets.append(data_r)
        data_r = xr.concat(datasets, "time")
        data_r["angles"] = (("along_track", "across_track"), angles)
        data_r["latitude"] = (("along_track", "across_track"), lats_5)
        data_r["longitude"] = (("along_track", "across_track"), lons_5)
        print(data_r.time, time)
        data_r = data_r.interp(
            time=time,
            along_track=time.along_track,
            across_track=time.across_track
        )

        data_r.to_netcdf(radar_file)




    def process_day(self, year, month, day, radar_file, preprocessor_file):

        key = year, month, day
        radar_files = RadarFile.get_files(year, month)
        if key not in self.overpasses or key not in radar_files:
            return None
        granules = self.overpasses[key]

        radar_file = Path(radar_file)
        preprocessor_file = Path(preprocessor_file)

        for granule in granules:
            l1c_file = L1CFile.open_granule(granule, self.sensor.l1c_file_path, self.sensor)
            with TemporaryDirectory() as tmp:
                tmp = Path(".")

                l1c_sub_file = tmp / "l1c_file.HDF5"
                scan_start, scan_end = l1c_file.extract_scans(ROI, l1c_sub_file)

                # Extract reference data from Kwajalein radar archive.
                if not radar_file.exists():
                    self.process_radar_file(
                        granule, l1c_sub_file, radar_file, scan_start, scan_end
                    )

from gprof_nn import sensors
ex = FileExtractor(sensors.GMI)
