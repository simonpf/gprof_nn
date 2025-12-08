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
from typing import Dict

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


def consolidate_swath_data_gmi(swath_data: Dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Combines data from multiple swaths into a single xarray.Dataset in the way
    it is done by the GPROF preprocessor.

    Args:
        swath_data: A dictionary containing the observations from the separate swaths.

    Return:
        A new xarray.Dataset containing the combined observations and incidence angles.
    """
    pixels = swath_data[1].pixels.data
    scans = swath_data[1].scans.data

    full_data = xr.Dataset({
        "pixels": (("pixels",), pixels),
        "scans": (("scans",), scans)
    })

    full_tbs = np.nan * np.zeros((scans.size, pixels.size, 15), dtype=np.float32)
    full_eia = np.nan * np.zeros((scans.size, pixels.size, 15), dtype=np.float32)

    chan_ind_in = 0
    for chan_ind_out in [0, 1, 2, 3, 4, 6, 7, 8, 9]:
        full_tbs[..., chan_ind_out] = swath_data[1].brightness_temperatures[..., chan_ind_in]
        full_eia[..., chan_ind_out] = swath_data[1].earth_incidence_angle[..., chan_ind_in]
        chan_ind_in += 1

    chan_ind_in = 0
    for chan_ind_out in [10, 11, 13, 14]:
        full_tbs[..., chan_ind_out] = swath_data[2].brightness_temperatures[..., chan_ind_in]
        full_eia[..., chan_ind_out] = swath_data[2].earth_incidence_angle[..., chan_ind_in]
        chan_ind_in += 1

    scan_time = swath_data[1].scan_time.data

    qflag1 = swath_data[1].quality_flag.data
    qflag2 = swath_data[2].quality_flag.data
    qflag = np.minimum(qflag1, qflag2)
    pos_qflag = (0 < qflag1) * (0 < qflag2)
    qflag[pos_qflag] = np.maximum(qflag1, qflag2)[pos_qflag]

    full_data["brightness_temperatures"] = (("scans", "pixels", "channels"), full_tbs)
    full_data["earth_incidence_angle"] = (("scans", "pixels", "channels"), full_eia)
    full_data["scan_time"] = (("scans",), scan_time)
    full_data["quality_flag"] = (("scans", "pixels"), qflag)

    full_data["longitude"] = (("scans", "pixels"), swath_data[1].longitude.data)
    full_data["latitude"] = (("scans", "pixels"), swath_data[1].latitude.data)

    return full_data


def consolidate_swath_data_atms(swath_data: Dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Combines data from multiple swaths into a single xarray.Dataset in the way
    it is done by the GPROF preprocessor.

    Args:
        swath_data: A dictionary containing the observations from the separate swaths.

    Return:
        A new xarray.Dataset containing the combined observations and incidence angles.
    """
    pixels = swath_data[1].pixels.data
    scans = swath_data[1].scans.data

    full_data = xr.Dataset({
        "pixels": (("pixels",), pixels),
        "scans": (("scans",), scans)
    })

    full_tbs = np.nan * np.zeros((scans.size, pixels.size, 5))
    full_tbs[..., 0] = swath_data[3].brightness_temperatures.data[..., 0]
    full_tbs[..., 1] = swath_data[4].brightness_temperatures.data[..., 0]
    full_tbs[..., 2] = swath_data[4].brightness_temperatures.data[..., 5]
    full_tbs[..., 3] = swath_data[4].brightness_temperatures.data[..., 3]
    full_tbs[..., 4] = swath_data[4].brightness_temperatures.data[..., 1]

    eia = swath_data[3].earth_incidence_angle.data[..., 0]
    scan_time = swath_data[1].scan_time.data
    qflag = swath_data[1].quality_flag.data

    full_data["brightness_temperatures"] = (("scans", "pixels", "channels"), full_tbs)
    full_data["earth_incidence_angle"] = (("scans", "pixels"), eia)
    full_data["scan_time"] = (("scans",), scan_time)
    full_data["quality_flag"] = (("scans", "pixels"), qflag)

    full_data["longitude"] = (("scans", "pixels"), swath_data[1].longitude.data)
    full_data["latitude"] = (("scans", "pixels"), swath_data[1].latitude.data)

    return full_data


def consolidate_swath_data_amsr2(swath_data: Dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Combines data from multiple swaths into a single xarray.Dataset in the way
    it is done by the GPROF preprocessor.

    Args:
        swath_data: A dictionary containing the observations from the separate swaths.

    Return:
        A new xarray.Dataset containing the combined observations and incidence angles.
    """
    pixels = swath_data[5].pixels.data
    scans = swath_data[5].scans.data

    full_data = xr.Dataset({
        "pixels": (("pixels",), pixels),
        "scans": (("scans",), scans)
    })

    full_tbs = np.nan * np.zeros((scans.size, pixels.size, 10))
    full_eia = np.nan * np.zeros((scans.size, pixels.size, 10))

    full_tbs[..., 8] = swath_data[5].brightness_temperatures.data[..., 0]
    full_tbs[..., 9] = swath_data[5].brightness_temperatures.data[..., 1]
    full_eia[..., 8:] = swath_data[5].earth_incidence_angle.data

    # Duplicate low-res channels.
    for chan, swath in zip([0, 2, 4, 6], [1, 2, 3, 4]):
        full_tbs[:, ::2, chan] = swath_data[swath].brightness_temperatures.data[..., 0]
        full_tbs[:, 1::2, chan] = swath_data[swath].brightness_temperatures.data[..., 0]
        full_tbs[:, ::2, chan + 1] = swath_data[swath].brightness_temperatures.data[..., 1]
        full_tbs[:, 1::2, chan + 1] = swath_data[swath].brightness_temperatures.data[..., 1]

        full_eia[:, ::2, chan:chan+2] = swath_data[swath].earth_incidence_angle.data
        full_eia[:, 1::2, chan:chan+2] = swath_data[swath].earth_incidence_angle.data

    scan_time = swath_data[5].scan_time.data
    qflag = swath_data[5].quality_flag.data

    full_data["brightness_temperatures"] = (("scans", "pixels", "channels"), full_tbs)
    full_data["earth_incidence_angle"] = (("scans", "pixels", "channels"), full_eia)
    full_data["scan_time"] = (("scans",), scan_time)
    full_data["quality_flag"] = (("scans", "pixels"), qflag)

    full_data["longitude"] = (("scans", "pixels"), swath_data[5].longitude.data)
    full_data["latitude"] = (("scans", "pixels"), swath_data[5].latitude.data)

    return full_data


def consolidate_swath_data_mhs(swath_data: Dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Combines data from MHS L1C files into a single xarray.Dataset in the way
    it is done by the GPROF preprocessor.

    Args:
        swath_data: A dictionary containing the observations from the separate swaths.

    Return:
        A new xarray.Dataset containing the combined observations and incidence angles.
    """
    pixels = swath_data[1].pixels.data
    scans = swath_data[1].scans.data

    full_data = xr.Dataset({
        "pixels": (("pixels",), pixels),
        "scans": (("scans",), scans)
    })

    full_tbs = swath_data[1].brightness_temperatures.data

    eia = swath_data[1].brightness_temperatures.data[..., 0]
    scan_time = swath_data[1].scan_time.data
    qflag = swath_data[1].quality_flag.data

    full_data["brightness_temperatures"] = (("scans", "pixels", "channels"), full_tbs)
    full_data["earth_incidence_angle"] = (("scans", "pixels"), eia)
    full_data["scan_time"] = (("scans",), scan_time)
    full_data["quality_flag"] = (("scans", "pixels"), qflag)

    full_data["longitude"] = (("scans", "pixels"), swath_data[1].longitude.data)
    full_data["latitude"] = (("scans", "pixels"), swath_data[1].latitude.data)

    return full_data


def consolidate_swath_data_ssmis(swath_data: Dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Combines data from SSMS L1C files into a single xarray.Dataset in the way
    it is done by the GPROF preprocessor.

    Args:
        swath_data: A dictionary containing the observations from the separate swaths.

    Return:
        A new xarray.Dataset containing the combined observations and incidence angles.
    """
    pixels = swath_data[3].pixels.data
    scans = swath_data[3].scans.data

    full_data = xr.Dataset({
        "pixels": (("pixels",), pixels),
        "scans": (("scans",), scans)
    })

    full_tbs = np.nan * np.zeros((scans.size, pixels.size, 11))
    full_eia = np.nan * np.zeros((scans.size, pixels.size, 11))

    full_tbs[:, ::2, :3] =  swath_data[1].brightness_temperatures.data
    full_tbs[:, 1::2, :3] =  swath_data[1].brightness_temperatures.data
    full_eia[:, ::2, :3] =  swath_data[1].earth_incidence_angle.data
    full_eia[:, 1::2, :3] =  swath_data[1].earth_incidence_angle.data

    full_tbs[:, ::2, 3:5] =  swath_data[2].brightness_temperatures.data
    full_tbs[:, 1::2, 3:5] =  swath_data[2].brightness_temperatures.data
    full_eia[:, ::2, 3:5] =  swath_data[2].earth_incidence_angle.data
    full_eia[:, 1::2, 3:5] =  swath_data[2].earth_incidence_angle.data

    full_tbs[:, :, 5:7] =  swath_data[4].brightness_temperatures.data
    full_eia[:, :, 5:7] =  swath_data[4].earth_incidence_angle.data

    full_tbs[:, :, 7:11] =  swath_data[3].brightness_temperatures.data
    full_eia[:, :, 7:11] =  swath_data[3].earth_incidence_angle.data

    scan_time = swath_data[3].scan_time.data
    qflag = swath_data[3].quality_flag.data

    full_data["brightness_temperatures"] = (("scans", "pixels", "channels"), full_tbs)
    full_data["earth_incidence_angle"] = (("scans", "pixels", "channels"), full_eia)
    full_data["scan_time"] = (("scans",), scan_time)
    full_data["quality_flag"] = (("scans", "pixels"), qflag)

    full_data["longitude"] = (("scans", "pixels"), swath_data[3].longitude.data)
    full_data["latitude"] = (("scans", "pixels"), swath_data[3].latitude.data)

    return full_data


def consolidate_swath_data_tmi(swath_data: Dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Combines data from TMI L1C files into a single xarray.Dataset in the way
    it is done by the GPROF preprocessor.

    Args:
        swath_data: A dictionary containing the observations from the separate swaths.

    Return:
        A new xarray.Dataset containing the combined observations and incidence angles.
    """
    pixels = swath_data[3].pixels.data
    scans = swath_data[3].scans.data

    full_data = xr.Dataset({
        "pixels": (("pixels",), pixels),
        "scans": (("scans",), scans)
    })

    full_tbs = np.nan * np.zeros((scans.size, pixels.size, 9))
    full_eia = np.nan * np.zeros((scans.size, pixels.size, 9))

    full_tbs[:, ::2, :2] =  swath_data[1].brightness_temperatures.data
    full_tbs[:, 1::2, :2] =  swath_data[1].brightness_temperatures.data
    full_eia[:, ::2, :2] =  swath_data[1].earth_incidence_angle.data
    full_eia[:, 1::2, :2] =  swath_data[1].earth_incidence_angle.data

    full_tbs[:, ::2, 2:7] =  swath_data[2].brightness_temperatures.data
    full_tbs[:, 1::2, 2:7] =  swath_data[2].brightness_temperatures.data
    full_eia[:, ::2, 2:7] =  swath_data[2].earth_incidence_angle.data
    full_eia[:, 1::2, 2:7] =  swath_data[2].earth_incidence_angle.data

    full_tbs[:, :, 7:9] =  swath_data[3].brightness_temperatures.data
    full_eia[:, :, 7:9] =  swath_data[3].earth_incidence_angle.data

    scan_time = swath_data[3].scan_time.data
    qflag = swath_data[3].quality_flag.data

    full_data["brightness_temperatures"] = (("scans", "pixels", "channels"), full_tbs)
    full_data["earth_incidence_angle"] = (("scans", "pixels", "channels"), full_eia)
    full_data["scan_time"] = (("scans",), scan_time)
    full_data["quality_flag"] = (("scans", "pixels"), qflag)

    full_data["longitude"] = (("scans", "pixels"), swath_data[3].longitude.data)
    full_data["latitude"] = (("scans", "pixels"), swath_data[3].latitude.data)

    return full_data


CONSOLIDATION_FUNCTIONS = {
    "gmi": consolidate_swath_data_gmi,
    "atms": consolidate_swath_data_atms,
    "amsr2": consolidate_swath_data_amsr2,
    "mhs": consolidate_swath_data_mhs,
    "ssmis": consolidate_swath_data_ssmis,
    "tmi": consolidate_swath_data_tmi,
}


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
            year = date.year % 100
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
        year = date.year % 100
        month = date.month
        day = date.day
        data_path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"
        files = list(data_path.glob(sensor.l1c_file_prefix + f"*V07*.HDF5"))

        # Add files from following day.
        date_f = date + pd.DateOffset(1)
        year = date_f.year % 100
        month = date_f.month
        day = date_f.day
        data_path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"
        files += list(data_path.glob(sensor.l1c_file_prefix + f"*V07*.HDF5"))

        # Add files from previous day.
        date_f = date - pd.DateOffset(1)
        year = date_f.year % 100
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
        year = date.year % 100
        month = date.month
        day = date.day
        data_path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"
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

                    scale_fac = input[group_name]["Tc"].shape[0] / input["S1"]["Tc"].shape[0]
                    scans_scaled = slice(int(i_start * scale_fac), int(i_end * scale_fac))

                    g = output.create_group(group_name)
                    n_scans = int(scale_fac * (i_end - i_start))
                    for name, item in input[group_name].items():
                        if isinstance(item, h5py.Dataset):
                            shape = item.shape
                            g.create_dataset(
                                name, shape=(n_scans,) + shape[1:], data=item[scans_scaled]
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
                                name, shape=(n_scans,) + shape[1:], data=item[scans_scaled]
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
                scan_start = max(scan_start, 0)

                i = 1
                while f"S{i}" in input:
                    group_name = f"S{i}"

                    shape = input[f"{group_name}/Latitude"].shape
                    n_scans = shape[0]
                    scaling = n_scans / n_scans_total
                    scan_start_s = int(min(n_scans_total, scan_start) * scaling)
                    scan_end_s = int(min(n_scans_total, scan_end) * scaling)
                    n_scans = scan_end_s - scan_start_s
                    scan_range = slice(scan_start_s, scan_end_s)
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

    def to_xarray_dataset(self):
        """
        Read data into xarray.Dataset.

        Returns:
            An xarray.Dataset containing the data from this L1C file.
        """
        import h5py
        with h5py.File(self.path, "r") as inpt:

            swath_data = {}

            swath_ind = 1
            while f"S{swath_ind}" in inpt.keys():
                swath = f"S{swath_ind}"
                lats = inpt[f"{swath}/Latitude"][:]
                lons = inpt[f"{swath}/Longitude"][:]
                tbs = inpt[f"{swath}/Tc"][:]
                eia = inpt[f"{swath}/incidenceAngle"][:]
                qual = inpt[f"{swath}/Quality"][:]

                tbs[tbs < 0] = np.nan
                eia[eia < 0] = np.nan

                lats_sc = inpt[f"{swath}/SCstatus/SClatitude"]
                lons_sc = inpt[f"{swath}/SCstatus/SClongitude"]
                alt_sc = inpt[f"{swath}/SCstatus/SCaltitude"]

                data_s = xr.Dataset({
                    "longitude": (("scans", "pixels"), lons),
                    "latitude": (("scans", "pixels"), lats),
                    "brightness_temperatures": (("scans", "pixels", "channels"), tbs),
                    "quality_flag": (("scans", "pixels"), qual)
                })
                n_chans = tbs.shape[-1]
                if eia.ndim == 3:
                    eia = np.broadcast_to(eia, lons.shape + (n_chans,))
                    data_s["earth_incidence_angle"] = (
                        ("scans", "pixels", "channels"),
                        eia
                    )
                else:
                    data_s["earth_incidence_angle"] = (
                        ("scans", "pixels"),
                        eia
                    )

                year = inpt[f"{swath}/ScanTime/Year"][:] - 1970
                month = inpt[f"{swath}/ScanTime/Month"][:] - 1
                day = inpt[f"{swath}/ScanTime/DayOfMonth"][:] - 1
                hour = inpt[f"{swath}/ScanTime/Hour"][:]
                minute = inpt[f"{swath}/ScanTime/Minute"][:]
                second = inpt[f"{swath}/ScanTime/Second"][:]
                milli_second = inpt[f"{swath}/ScanTime/MilliSecond"][:]
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
                data_s["scan_time"] = (("scans",), times)

                swath_data[swath_ind] = data_s

                swath_ind += 1

            consolidation_fn = CONSOLIDATION_FUNCTIONS[self.sensor.name.lower()]
            data =  consolidation_fn(swath_data)

            if "FileHeader" in inpt.keys():
                granule = inpt["FileHeader/GranuleNumber"][:]
                satellite = inpt["FileHeader/SatelliteName"][:]
                sensor = inpt["FileHeader/InstrumentName"][:]
                l1c_file = inpt["FileHeader/FileName"][:]
                data.attrs = {
                    "granule": granule,
                    "platform": satellite,
                    "sensor": sensor,
                    "l1c_file": l1c_file,
                }

        return data


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
