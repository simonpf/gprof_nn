"""
==================
gprof_nn.data.kwaj
==================

This module provides functionality to download and process Radar
observations from the Kwajalein Atoll.
"""
from concurrent.futures import ProcessPoolExecutor
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
from rich.progress import Progress
import xarray as xr

from gprof_nn.logging import get_console
from gprof_nn.data.training_data import decompress_and_load
from gprof_nn.augmentation import latlon_to_ecef, ecef_to_latlon
from gprof_nn.data.validation import unify_grid, calculate_angles
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.preprocessor import run_preprocessor


LOGGER = logging.getLogger(__name__)


_BASE_URL = "https://pmm-gv.gsfc.nasa.gov/pub/gpmarchive/Radar/KPOL/"

LINK_REGEX = re.compile(r"<a href=\"(KPOL_[\w\.]*)\">")


LATITUDE = 8.71
LONGITUDE = 167.732
ALTITUDE = 24.0

ROI = [163.732, 4.71, 171.731, 12.71]

XYZ = None
N = None
W = None
U = None

VALIDATION_VARIABLES = {
    "ZZ": "radar_reflectivity",
    "RR": "surface_precip_rr",
    "RP": "surface_precip_rp",
    "RC": "surface_precip_rc",
}


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

    variables = [
        var for var in VALIDATION_VARIABLES.keys()
        if var in data.variables
    ]

    n_gates = data.range.size
    ray_index = 0
    while data.elevation[ray_index].data <= 1.0:
        if "n_points" in data.dims:
            start_index = ray_index * n_gates
            end_index = start_index + n_gates
            rays.append(data[variables][{"n_points": slice(start_index, end_index)}])
        else:
            rays.append(data[variables][{"time": ray_index}])
        ray_index = ray_index + 1

    dataset = xr.concat(rays, "rays")
    if "n_points" in data.dims:
        dataset = dataset.rename(n_points="range")
    dataset["azimuth"] = (("rays",), data.azimuth.data[:ray_index])
    dataset["elevation"] = (("rays",), data.elevation.data[:ray_index])
    dataset["time"] = (("rays",), data.time.data[:ray_index])
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

        return data.transpose("rays", "range")

    def open_file_raw(self, name):
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

        r = dataset.range.data.reshape(1, -1)
        phi = dataset.azimuth.data.reshape(-1, 1)
        theta = dataset.elevation.data.reshape(-1, 1)
        xyz_r = spherical_to_ecef(r, phi, theta)
        xyz = XYZ + xyz_r
        lons, lats, _ = ecef_to_latlon().transform(
            xyz[..., 0], xyz[..., 1], xyz[..., 2], radians=False
        )

        dataset["latitude"] = (("rays", "range"), lats.astype(np.float32))
        dataset["longitude"] = (("rays", "range"), lons.astype(np.float32))

        return dataset


    def open_files(self, start, end):
        """
        Open the observations closest to a given time from the archive.

        Args:
            date: The data for which to extract the observations.

        Return:
            An xarray.Dataset containing the observations closest in time
            to the given date.
        """
        times = np.array(list(self.times.keys()))
        times.sort()
        names = list(self.times.values())
        print(start, end)

        delta = (times - start)
        indices = np.where(delta > np.timedelta64(0))[0]
        if len(indices) == 0:
            return xr.Dataset()
        start = indices[0]

        delta = (times - end)
        indices = np.where(delta > np.timedelta64(0))[0]
        if len(indices) == 0:
            end = None
        else:
            end = indices[0] + 1
        print(start, end)

        times = times[slice(start, end)]
        data = []
        for time in times:
            data_t = self.open_file(self.times[time])
            print(start, end, time)
            data_t["time"] = data_t.time.mean()
            data.append(data_t)

        return xr.concat(data, "time")


class FileExtractor:
    """
    Helper class to coordinate the extraction of validation data from the
    Kwajalein online archive.
    """
    def __init__(self, sensor, year, month):
        """
        Create validation data extractor for a given sensor.

        Args:
            sensor: Sensor object representing the sensor for which to extract
                validation data.
        """
        self.sensor = sensor
        self.year = year
        self.month = month
        overpasses = get_overpasses(sensor.name)
        self.overpasses = {}
        for (year, month, day), granules in overpasses.items():
            if year == self.year and month == self.month:
                self.overpasses[(year, month, day)] = granules

    def process_radar_file(
            self,
            granule,
            l1c_file,
            radar_file,
            scan_start,
            scan_end
    ):
        """
        This extracts validation data from the Kwajalein radar for a given granule.

        Args:
            granule: The number of the granule.
            l1c_file: Path to the L1C file containing the observations cropped to
                Kwajalein atoll.
            radar_file: Name of the output file to write the extracted results to.
            scan_start: The scan index at which the co-located L1C observations
                start.
            scan_start: The scan index at which the co-located observations end.
        """
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
                radar_archive = RadarFile.download_file(
                    year, month, day, destination=data_directory
                )
                radar_archive = RadarFile(radar_archive)
        else:
            radar_archive = RadarFile.download_file(
                year, month, day, destination=data_directory
            )
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


        start_time = l1c_data.scan_time.min()
        end_time = l1c_data.scan_time.max()
        kwaj_data = radar_archive.open_files(start_time.data, end_time.data)
        datasets = []

        # Iterate over the closest time frames and smooth each variable.
        for t in range(kwaj_data.time.size):
            data = kwaj_data[{"time": t}]

            lats = data.latitude.data
            lons = data.longitude.data
            kwaj_swath = geometry.SwathDefinition(lats=lats, lons=lons)
            swath_5 = geometry.SwathDefinition(lats=lats_5, lons=lons_5)

            resampling_info = kd_tree.get_neighbour_info(
                kwaj_swath,
                swath_5,
                10e3,
                neighbours=128
            )
            valid_inputs, valid_outputs, indices, distances = resampling_info

            data_r = {}

            # Smooth variable taking handling nans by replacing them with
            # valid values.
            for variable, name in VALIDATION_VARIABLES.items():

                if variable not in data.variables:
                    continue

                # We mask all values further than 150k from radar.
                range_mask = data.range > 150e3

                data_in = data[variable].data.copy()
                if variable == "ZZ":
                    data_in = 10 ** (np.nan_to_num(data_in, -50) / 10.0)
                else:
                    data_in = np.nan_to_num(data_in, 0)
                data_in[:, range_mask] = np.nan

                resampled = kd_tree.get_sample_from_neighbour_info(
                    'custom', swath_5.shape, data_in,
                    valid_inputs, valid_outputs, indices, distance_array=distances,
                    weight_funcs=weighting_function, fill_value=np.nan
                )
                if variable == "ZZ":
                    resampled = 10 * np.log10(resampled)
                    print(np.isfinite(resampled).sum())
                data_r[name] = (("along_track", "across_track"), resampled)

            data_r = xr.Dataset(data_r)
            data_r["time"] = (("time",), [data.time.data])

            # Include raining fraction
            # Include range in extracted data.
            rf = data["RC"].data > 0
            rf = kd_tree.get_sample_from_neighbour_info(
                'custom', swath_5.shape, rf,
                valid_inputs, valid_outputs, indices,
                distance_array=distances,
                weight_funcs=weighting_function,
                fill_value=np.nan
            )
            data_r["raining_fraction"] = (("along_track", "across_track"), rf)

            # Include range in extracted data.
            ranges, _ = xr.broadcast(data.range, data.ZZ)
            ranges = ranges.transpose("rays", "range")
            ranges = kd_tree.get_sample_from_neighbour_info(
                'custom', swath_5.shape, ranges.data,
                valid_inputs, valid_outputs, indices,
                distance_array=distances,
                weight_funcs=weighting_function,
                fill_value=np.nan
            )
            data_r["range"] = (("along_track", "across_track"), ranges)

            datasets.append(data_r)

        data_r = xr.concat(datasets, "time")
        data_r["angles"] = (("along_track", "across_track"), angles)
        data_r["latitude"] = (("along_track", "across_track"), lats_5)
        data_r["longitude"] = (("along_track", "across_track"), lons_5)

        # Finally interpolate all reference data to scan time.

        if data.time.size > 1:
            data_r = data_r.interp(
                time=time.mean(),
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )
        else:
            data_r = data_r[{"time": 0}]

        data_r.attrs["sensor"] = self.sensor.sensor_name
        data_r.attrs["platform"] = self.sensor.platform.name
        data_r.attrs["granule"] = granule
        data_r.attrs["scan_start"] = scan_start
        data_r.attrs["scand_end"] = scan_end

        # Remove empty scan lines.
        has_data = np.any(np.isfinite(data_r.radar_reflectivity.data), -1)
        indices = np.where(has_data)[0]
        if len(indices) == 0:
            raise ValueError(
                "No valid precipitation pixels in overpass."
                )
        along_track_start = indices.min()
        along_track_end = indices.max()
        data_r = data_r[{
            "along_track": slice(along_track_start, along_track_end)
        }]
        data_r.to_netcdf(radar_file)


    def process_day(self, year, month, day, kwaj_path, pp_path):
        """
        Extract validation data for a given day.

        Args:
            year: The year for which to extract the validation data.
            month: The month for which to extract the validation data.
            day: The day for which to extract the validation data.
            radar_file: The name of the output file.
            preprocessor_file: The name of the preprocessor file.
        """
        key = year, month, day
        radar_files = RadarFile.get_files(year, month)
        if key not in self.overpasses or key not in radar_files:
            return None
        granules = self.overpasses[key]

        kwaj_path = Path(kwaj_path)
        pp_path = Path(pp_path)

        radar_file_pattern = (
            "kwaj_{sensor}_{year}{month:02}{day:02}_{granule}.nc"
        )
        pp_file_pattern = (
            "{sensor}_kwaj_{year}{month:02}{day:02}_{granule}.pp"
        )

        for granule in granules:
            l1c_file = L1CFile.open_granule(granule, self.sensor.l1c_file_path, self.sensor)
            fname_kwargs = {
                "sensor": self.sensor.name.lower(),
                "year": year,
                "month": month,
                "day": day,
                "granule": granule
            }
            kwaj_file = kwaj_path / radar_file_pattern.format(**fname_kwargs)
            pp_file = pp_path / pp_file_pattern.format(**fname_kwargs)

            with TemporaryDirectory() as tmp:
                tmp = Path(tmp)

                l1c_sub_file = tmp / "l1c_file.HDF5"
                scan_start, scan_end = l1c_file.extract_scans(ROI, l1c_sub_file)

                # Extract reference data from Kwajalein radar archive.
                if not kwaj_file.exists():
                    self.process_radar_file(
                        granule, l1c_sub_file, kwaj_file, scan_start, scan_end
                    )

                if kwaj_file.exists():
                    # Run preprocessor on L1C file.
                    run_preprocessor(
                        str(l1c_sub_file),
                        self.sensor,
                        configuration="ERA5",
                        output_file=pp_file,
                    )

    def run(self, kwaj_path, pp_path, n_workers=4):
        kwaj_path = Path(kwaj_path) / f"{self.year}" / f"{self.month:02}"
        kwaj_path.mkdir(exist_ok=True, parents=True)
        pp_path = Path(pp_path) / f"{self.year}" / f"{self.month:02}"
        pp_path.mkdir(exist_ok=True, parents=True)

        tasks = []
        # Submit task for each granule in month
        pool = ProcessPoolExecutor(max_workers=n_workers)
        for key in self.overpasses:
            year, month, day = key
            tasks.append(
                pool.submit(
                    self.process_day, year, month, day, kwaj_path, pp_path
                )
            )

        # Collect results and track progress.
        with Progress(console=get_console()) as progress:
            pbar = progress.add_task("Extracting validation data:", total=len(tasks))
            for task, key in zip(tasks, self.overpasses):
                try:
                    task.result()
                except Exception as e:
                    LOGGER.error(
                        "The following error occurred when processing day "
                        "%s: \n %s",
                        key,
                        e,
                    )

                progress.advance(pbar)
