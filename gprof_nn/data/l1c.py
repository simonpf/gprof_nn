"""
=================
gprof_nn.data.l1c
=================

Functionality to read and manipulate GPROF L1C-R files.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import logging
from pathlib import Path
import re

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import pandas as pd
from rich.progress import track
import xarray as xr

from gprof_nn import sensors
from gprof_nn.definitions import DATABASE_MONTHS
from gprof_nn.logging import get_console

_RE_META_INFO = re.compile(r"NumberScansGranule=(\d*);")

LOGGER = logging.getLogger(__name__)


class L1CFile:
    """
    Interface class to GPROF L1C-R files in HDF5 format.
    """
    @classmethod
    def open_granule(cls, granule, path, sensor, date=None):
        """
        Find and open L1C file with a given granule number.

        Args:
            granule: The granule number as integer.
            path: The root of the directory tree containing the
                L1C files.
            sensor: Sensor object representing the sensor of which to open the
                corresponding L1C file.
            date: The date of the file used to determine sub-folders
                corresponding to month and day.

        Return:
            L1CFile object providing access to the requested file.
        """
        if date is not None:
            date = pd.Timestamp(date)
            year = date.year - 2000
            month = date.month
            day = date.day
            path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"
            files = path.glob(sensor.l1c_file_prefix + f"*.{granule:06}.*V07*.HDF5")
        else:
            path = Path(path)
            files = path.glob(
                "**/" + sensor.l1c_file_prefix + f"*.{granule:06}.*V07*.HDF5"
            )

        files = list(files)
        if len(files) > 1:
            raise Exception(
                f"Found more than 1 matching L1C file. This indicates something"
                f"went wrong. Found the following files: {files}"
            )
        try:
            f = next(iter(files))
            return L1CFile(f)
        except StopIteration:
            if date is not None:
                return cls.open_granule(granule, path, None)
            raise Exception(f"Could not find a L1C file with granule number {granule}.")

    @classmethod
    def find_file(cls, date, path, sensor=sensors.GMI):
        """
        Find L1C files for given time.

        Args:
            date: The date of the file used to determine sub-folders
                corresponding to month and day.
            path: The root of the directory tree containing the
                L1C files.

        Return:
            L1CFile object providing access to the requested file.
        """
        path = Path(path)

        date = pd.Timestamp(date)
        year = date.year - 2000
        month = date.month
        day = date.day
        data_path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"
        files = list(data_path.glob(sensor.l1c_file_prefix + f"*V07*.HDF5"))

        # Add files from following day.
        date_f = date + pd.DateOffset(1)
        year = date_f.year - 2000
        month = date_f.month
        day = date_f.day
        data_path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"
        files += list(data_path.glob(sensor.l1c_file_prefix + f"*V07*.HDF5"))

        # Add files from previous day.
        date_f = date - pd.DateOffset(1)
        year = date_f.year - 2000
        month = date_f.month
        day = date_f.day
        data_path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"

        files += list(data_path.glob(sensor.l1c_file_prefix + f"*V07*.HDF5"))
        files += list(path.glob(sensor.l1c_file_prefix + f"*V07*.HDF5"))
        files = sorted(files)

        start_times = []
        end_times = []

        for f in files:
            l1c = cls(f)
            start = l1c.start_time
            end = l1c.end_time
            if end < start:
                end += np.timedelta64(1, "D")
            start_times.append(start)
            end_times.append(end)
        start_times = np.array(start_times)
        end_times = np.array(end_times)
        date = date.to_datetime64()


        if len(start_times) == 0 or len(end_times) == 0:
            raise ValueError("No file found for the requested date.")
        inds = np.where((start_times <= date) * (end_times >= date))[0]
        if len(inds) == 0:
            raise ValueError("No file found for the requested date.")
        filename = files[inds[0]]

        return L1CFile(filename)

    @classmethod
    def find_files(cls, date, path, roi=None, sensor=sensors.GMI):
        """
        Find L1C files for a given day covering a rectangular region
        of interest (ROI).

        Args:
            date: A date specifying a day for which to find observations.
            path: The root of the directory tree containing the
                L1C files.
            roi: Tuple ``(lon_min, lat_min, lon_max, lat_max)`` describing a
                rectangular bounding box around the region of interest.
            sensor: Sensor object defining the sensor for which to find the
                L1C file.

        Return:
             Generator providing files with observations within the given ROI
             on the requested day.
        """
        path = Path(path)

        date = pd.Timestamp(date)
        year = date.year - 2000
        month = date.month
        day = date.day
        data_path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"
        print("DATA PATH :: ", data_path)
        print("PATTERN :: ", sensor.l1c_file_prefix + f"*{date.year:04}{month:02}{day:02}*V07*.HDF5")
        files = list(
            data_path.glob(
                sensor.l1c_file_prefix + f"*{date.year:04}{month:02}{day:02}*V07*.HDF5"
            )
        )
        files += list(
            path.glob(
                sensor.l1c_file_prefix + f"*{date.year:04}{month:02}{day:02}*V07*.HDF5"
            )
        )
        for l1c_file in files:
            try:
                l1c_file = L1CFile(l1c_file)
            except Exception:
                continue
            if roi is not None:
                if l1c_file.covers_roi(roi):
                    yield l1c_file
            else:
                yield l1c_file

    def __init__(self, path):
        """
        Open a GPROF GMI L1C file.

        Args:
            path: The path to the file.
        """
        self.filename = path
        self.path = Path(path)

        import h5py
        with h5py.File(self.path, "r") as data:
            header = data.attrs["FileHeader"].decode().splitlines()
            satellite = header[6].split("=")[1][:-1]
            sensor = header[7].split("=")[1][:-1]
            self.header = header
            self.granule = int(header[11].split("=")[1][:-1])
            date = self.start_time
            self.sensor = sensors.get_sensor(
                sensor,
                platform=satellite,
                date=self.start_time
            )

    @property
    def start_time(self):
        import h5py
        with h5py.File(self.path, "r") as input:
            year = input["S1/ScanTime/Year"][0]
            month = input["S1/ScanTime/Month"][0]
            day = input["S1/ScanTime/DayOfMonth"][0]
            hour = input["S1/ScanTime/Hour"][0]
            minute = input["S1/ScanTime/Minute"][0]
            second = input["S1/ScanTime/Second"][0]
        return np.datetime64(
            f"{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}"
        )

    @property
    def end_time(self):
        import h5py
        with h5py.File(self.path, "r") as input:
            year = input["S1/ScanTime/Year"][-1]
            month = input["S1/ScanTime/Month"][-1]
            day = input["S1/ScanTime/DayOfMonth"][-1]
            hour = input["S1/ScanTime/Hour"][-1]
            minute = input["S1/ScanTime/Minute"][-1]
            second = input["S1/ScanTime/Second"][-1]
        return np.datetime64(
            f"{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}"
        )

    def __repr__(self):
        """String representation for file."""
        return f"L1CFile(filename='{self.path.name}')"

    def extract_scans(self, roi, output_filename, min_scans=None):
        """
        Extract scans over a rectangular region of interest (ROI).

        Args:
            roi: The region of interest given as an length-4 iterable
                 containing the lower-left corner longitude and latitude
                 coordinates followed by the upper-right corner longitude
                 and latitude coordinates.
            output_filename: Name of the file to which to write the extracted
                 scans.
        """
        lon_min, lat_min, lon_max, lat_max = roi

        import h5py
        with h5py.File(self.path, "r") as input:
            lats = input["S1/Latitude"][:]
            lons = input["S1/Longitude"][:]

            indices = np.where(
                np.any(
                    (lats >= lat_min)
                    * (lats < lat_max)
                    * (lons >= lon_min)
                    * (lons < lon_max),
                    axis=-1,
                )
            )[0]
            d_inds = np.diff(indices)
            if any(d_inds > 1):
                ind = np.where(d_inds > 1)[0][0]
                indices = indices[:ind]

            if len(indices) > 0:
                i_start, i_end = indices[[0, -1]]
                n_scans = i_end - i_start
                if min_scans is not None and n_scans < min_scans:
                    diff_l = (min_scans - n_scans) // 2
                    i_start = max(0, i_start - diff_l)
                    diff_r = min_scans - (i_end - i_start)
                    i_end = i_end + diff_r
            else:
                i_start = 0
                i_end = 0
            scans = slice(i_start, i_end)

            with h5py.File(output_filename, "w") as output:

                group_index = 1
                while f"S{group_index}" in input:

                    group_name = f"S{group_index}"

                    g = output.create_group(group_name)
                    n_scans = i_end - i_start
                    for name, item in input[group_name].items():
                        if isinstance(item, h5py.Dataset):
                            shape = item.shape
                            g.create_dataset(
                                name, shape=(n_scans,) + shape[1:], data=item[scans]
                            )

                    for a in input[group_name].attrs:
                        s = input[group_name].attrs[a].decode()
                        s = _RE_META_INFO.sub(f"NumberScansGranule={n_scans};", s)
                        s = np.bytes_(s)
                        g.attrs[a] = s

                    g_st = g.create_group("ScanTime")
                    for name, item in input[f"{group_name}/ScanTime"].items():
                        if isinstance(item, h5py.Dataset):
                            shape = item.shape
                            g_st.create_dataset(
                                name, shape=(n_scans,) + shape[1:], data=item[scans]
                            )

                    g_sc = g.create_group("SCstatus")
                    for name, item in input[f"{group_name}/SCstatus"].items():
                        if isinstance(item, h5py.Dataset):
                            shape = item.shape
                            g_sc.create_dataset(
                                name, shape=(n_scans,) + shape[1:], data=item[scans]
                            )

                    group_index += 1

                for a in input.attrs:
                    output.attrs[a] = input.attrs[a]

        return i_start, i_end

    def extract_scan_range(self, scan_start, scan_end, output_filename):
        """
        Extract a range of scans from the L1C file and write to new
        L1C file.

        Args:
            scans: Indices of the scans to extract.
            output_filename: Name of the file to which to write the extracted
                 scans.
        """
        import h5py
        with h5py.File(self.path, "r") as input:

            with h5py.File(output_filename, "w") as output:

                n_scans_total = input["S1/Latitude"].shape[0]

                i = 1
                while f"S{i}" in input:
                    group_name = f"S{i}"

                    shape = input[f"{group_name}/Latitude"].shape
                    n_scans = shape[0]
                    scaling = n_scans // n_scans_total
                    n_scans = (scan_end - scan_start) * scaling
                    scan_range = slice(
                        scan_start * scaling,
                        scan_end * scaling
                    )
                    g = output.create_group(group_name)
                    for name, item in input[group_name].items():
                        if isinstance(item, h5py.Dataset):
                            shape = item.shape
                            g.create_dataset(
                                name,
                                shape=(n_scans,) + shape[1:],
                                data=item[scan_range],
                            )

                    for a in input[group_name].attrs:
                        s = input[group_name].attrs[a].decode()
                        s = _RE_META_INFO.sub(f"NumberScansGranule={n_scans};", s)
                        s = np.bytes_(s)
                        g.attrs[a] = s

                    g_st = g.create_group("ScanTime")
                    for name, item in input[f"{group_name}/ScanTime"].items():
                        if isinstance(item, h5py.Dataset):
                            shape = item.shape
                            g_st.create_dataset(
                                name,
                                shape=(n_scans,) + shape[1:],
                                data=item[scan_range],
                            )

                    g_sc = g.create_group("SCstatus")
                    for name, item in input[f"{group_name}/SCstatus"].items():
                        if isinstance(item, h5py.Dataset):
                            shape = item.shape
                            g_sc.create_dataset(
                                name,
                                shape=(n_scans,) + shape[1:],
                                data=item[scan_range],
                            )
                    i += 1

                for a in input.attrs:
                    output.attrs[a] = input.attrs[a]

    def covers_roi(self, roi):
        """
        Determine whether any observations in file cover given ROI.

        Args:
            roi: Tuple ``(lon_min, lat_min, lon_max, lat_max)`` defining a
                 a rectangular bounding box around the region of interest.

        Returns:
            True if the file contains any observations over the given ROI.
        """
        import h5py
        lon_min, lat_min, lon_max, lat_max = roi
        with h5py.File(self.path, "r") as input:
            lats = input["S1/Latitude"][:]
            lons = input["S1/Longitude"][:]
            return np.any(
                (lons >= lon_min)
                * (lons < lon_max)
                * (lats >= lat_min)
                * (lats < lat_max)
            )

    def to_xarray_dataset(self, roi=None):
        """
        Read data into xarray.Dataset.

        Args:
            roi: If provided should be a tuple
                 ``(lon_min, lat_min, lon_max, lat_max)`` defining a
                 rectangular bounding box around a region of interest. In this
                 case only swaths that at least partially cover the give ROI
                 will be loaded

        Returns:
            An xarray.Dataset containing the data from this L1C file.
        """
        import h5py
        with h5py.File(self.path, "r") as input:

            swath = "S1"
            n_pixels = input["S1/Latitude"].shape[1]
            if "S2" in input.keys():
                if input["S2/Latitude"].shape[1] > n_pixels:
                    swath = "S2"
            if "S3" in input.keys():
                if input["S3/Latitude"].shape[1] > n_pixels:
                    swath = "S3"

            lats = input[f"{swath}/Latitude"][:]
            lons = input[f"{swath}/Longitude"][:]

            if roi is not None:
                lon_min, lat_min, lon_max, lat_max = roi
                indices = np.any(
                    (lons >= lon_min)
                    * (lons < lon_max)
                    * (lats >= lat_min)
                    * (lats < lat_max),
                    axis=-1,
                )
            else:
                indices = slice(0, None)

            lats = lats[indices]
            lons = lons[indices]

            lats_sc = input[f"{swath}/SCstatus/SClatitude"][indices]
            lons_sc = input[f"{swath}/SCstatus/SClongitude"][indices]
            alt_sc = input[f"{swath}/SCstatus/SCaltitude"][indices]

            # Handle case that observations are split up.
            tbs = []
            eia = []
            tbs.append(input[f"{swath}/Tc"][:][indices])
            eia_s = input[f"{swath}/incidenceAngle"][:][indices]
            eia_s = np.broadcast_to(eia_s, tbs[-1].shape)
            eia.append(eia_s)
            if "S2" in input.keys():
                tbs.append(input["S2/Tc"][:][indices])
                eia_s = input[f"S2/incidenceAngle"][:][indices]
                eia_s = np.broadcast_to(eia_s, tbs[-1].shape)
                eia.append(eia_s)
            if "S3" in input.keys():
                tbs.append(input["S3/Tc"][:][indices])
                eia_s = input[f"S3/incidenceAngle"][:][indices]
                eia_s = np.broadcast_to(eia_s, tbs[-1].shape)
                eia.append(eia_s)
            if "S4" in input.keys():
                tbs.append(input["S4/Tc"][:][indices])
                eia_s = input[f"S4/incidenceAngle"][:][indices]
                eia_s = np.broadcast_to(eia_s, tbs[-1].shape)
                eia.append(eia_s)
            if "S5" in input.keys():
                tbs_s = input["S5/Tc"][:][indices]
                eia_s = input[f"S5/incidenceAngle"][:][indices]
                tbs.append(tbs_s)
                eia_s = np.broadcast_to(eia_s, tbs[-1].shape)
                eia.append(eia_s)
            if "S6" in input.keys():
                tbs_s = input["S6/Tc"][:][indices]
                eia_s = input[f"S6/incidenceAngle"][:][indices]
                tbs.append(tbs_s)
                eia_s = input[f"S6/incidenceAngle"][:][indices]
                eia_s = np.broadcast_to(eia_s, tbs[-1].shape)
                eia.append(eia_s)

            n_pixels = max([array.shape[1] for array in tbs])
            tbs_r = []
            eia_r = []
            for tbs_s, eia_s in zip(tbs, eia):
                if tbs_s.shape[1] < n_pixels:
                    f = interp1d(
                        np.linspace(0, n_pixels - 1, tbs_s.shape[1]),
                        tbs_s,
                        axis=1
                    )
                    x = np.arange(n_pixels)
                    tbs_s = f(x)
                    f = interp1d(
                        np.linspace(0, n_pixels - 1, eia_s.shape[1]),
                        eia_s,
                        axis=1
                    )
                    eia_s = f(x)
                tbs_r.append(tbs_s)
                eia_r.append(eia_s)
            tbs = np.concatenate(tbs_r, axis=-1)
            eia = np.concatenate(eia_r, axis=-1)

            if lats.shape != tbs.shape[:-1]:
                n_pixels = tbs.shape[1]
                x = np.arange(n_pixels)
                f = interp1d(
                    np.linspace(0, n_pixels - 1, lats.shape[1]),
                    lats,
                    axis=1
                )
                lats = f(x)
                f = interp1d(
                    np.linspace(0, n_pixels - 1, lons.shape[1]),
                    lons,
                    axis=1
                )
                lons = f(x)

            n_scans = lats.shape[0]

            year = input[f"{swath}/ScanTime/Year"][indices] - 1970
            month = input[f"{swath}/ScanTime/Month"][indices] - 1
            day = input[f"{swath}/ScanTime/DayOfMonth"][indices] - 1
            hour = input[f"{swath}/ScanTime/Hour"][indices]
            minute = input[f"{swath}/ScanTime/Minute"][indices]
            second = input[f"{swath}/ScanTime/Second"][indices]
            milli_second = input[f"{swath}/ScanTime/MilliSecond"][indices]

            time = year.astype("datetime64[Y]").astype("datetime64[M]")
            time += month.astype("timedelta64[M]")
            time = time.astype("datetime64[D]")
            time += day.astype("timedelta64[D]")
            time = time.astype("datetime64[h]")
            time += hour.astype("timedelta64[h]")
            time = time.astype("datetime64[m]")
            time += minute.astype("timedelta64[m]")
            time = time.astype("datetime64[s]")
            time += second.astype("timedelta64[s]")
            time = time.astype("datetime64[ms]")
            time += milli_second.astype("timedelta64[ms]")

            times = time.astype("datetime64[ns]")

            dims = ("scans", "pixels")
            data = {
                "latitude": (dims, lats),
                "longitude": (dims, lons),
                "spacecraft_latitude": (dims[:1], lats_sc),
                "spacecraft_longitude": (dims[:1], lons_sc),
                "spacecraft_altitude": (dims[:1], alt_sc),
                "brightness_temperatures": (dims + ("channels",), tbs),
                "incidence_angle": (dims + ("channels",), eia),
                "scan_time": (dims[:1], times),
            }

            if "SCorientation" in input[f"{swath}/SCstatus"]:
                data["sensor_orientation"] = (
                    ("scans",),
                    input[f"{swath}/SCstatus/SCorientation"][indices],
                )

        return xr.Dataset(data)


def extract_scenes(data):
    """
    Organizes the data in 'data' into quadratic scenes with a side
    length matching the number of pixels of the sensor.

    Args:
        data: 'xarray.Dataset' containing swath data.
        sensor: Sensor object representing the sensor from which the
            observations stem.

    Return:
        data: A new 'xarray.Dataset' containing as much as possible
            of the data in 'data' organised into scenes.
    """
    n = data.pixels.size

    i_start = 0
    i_end = data.scans.size

    scenes = []
    i_start
    while i_start + n < i_end:
        subscene = data[{"scans": slice(i_start, i_start + n)}]
        scenes.append(subscene)
        i_start += n

    if scenes:
        return xr.concat(scenes, "samples")
    return None


def process_l1c_file(l1c_filename, sensor):
    """
    Run preprocessor for L1C file and extract resulting data.

    Args:
        l1c_filename: Path to the L1C file to process.
        sensor: Sensor object representing the sensor from which
            the data originates.
    """
    import gprof_nn.logging
    from gprof_nn.data.preprocessor import run_preprocessor

    data_pp = run_preprocessor(l1c_filename, sensor=sensor)
    return extract_scenes(data_pp)


class ObservationProcessor:
    """
    Processor class to extract observations from L1C files.
    """

    def __init__(
        self,
        output_file,
        sensor,
        n_workers=4,
        day=None,
    ):
        """
        Create observation processor..

        Args:
            path: The folder containing the input files.
            pattern: glob pattern to use to subselect input files.
            output_path: The path to which to write the retrieval
                 results
            input_class: The class to use to read and process the input files.
            n_workers: The number of worker processes to use.
            days: The days of each month to process.
        """

        self.output_file = output_file
        self.sensor = sensor
        self.pool = ProcessPoolExecutor(max_workers=n_workers)

        if day is None:
            self.day = 1
        else:
            self.day = day

    def run(self):
        """
        Start the processing.

        This will start processing all suitable input files that have been found and
        stores the names of the produced result files in the ``processed`` attribute
        of the driver.
        """
        l1c_file_path = self.sensor.l1c_file_path
        l1c_files = []
        for year, month in DATABASE_MONTHS:
            try:
                date = datetime(year, month, self.day)
                l1c_files += L1CFile.find_files(date, l1c_file_path, sensor=self.sensor)
            except ValueError:
                pass
        l1c_files = [f.filename for f in l1c_files]
        l1c_files = np.random.permutation(l1c_files)

        n_l1c_files = len(l1c_files)
        i = 0

        # Submit tasks interleaving .sim and MRMS files.
        tasks = []
        for l1c_file in l1c_files:
            tasks.append(self.pool.submit(process_l1c_file, l1c_file, self.sensor))
            i += 1

        n_datasets = len(tasks)
        datasets = []
        output_path = Path(self.output_file).parent
        output_file = Path(self.output_file).stem

        # Retrieve extracted observations and concatenate into
        # single dataset.
        for t in track(tasks, description="Extracting data ..."):
            try:
                dataset = t.result()
            except Exception as e:
                LOGGER.warning(
                    "The follow error was encountered while collecting " " results: %s",
                    e,
                )
                get_console().print_exception()
                dataset = None

            if dataset is not None:
                datasets.append(dataset)
        dataset = xr.concat(datasets, "samples")

        # Store dataset with sensor name as attribute.
        filename = output_path / (output_file + ".nc")
        dataset.attrs["sensor"] = self.sensor.name
        dataset.to_netcdf(filename)
