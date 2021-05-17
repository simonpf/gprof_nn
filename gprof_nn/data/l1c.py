"""
=================
gprof_nn.data.l1c
=================

Functionality to read and manipulate GPROF L1C-R files.
"""
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import h5py
import pandas as pd
import xarray as xr

_RE_META_INFO = re.compile("NumberScansGranule=(\d*);")
FILENAME_RE = re.compile(
    r"1C-R\.GPM\.GMI\.([\w-]*)\.(\d{8})-"
    r"S(\d{6})-E(\d{6})\.(\w*)\.(\w*)\.HDF5"
)

class L1CFile:
    """
    Interface class to GPROF L1C-R files in HDF5 format.
    """
    @classmethod
    def open_granule(cls, granule, path, date=None):
        """
        Find and open L1C file with a given granule number.

        Args:
            granule: The granule number as integer.
            path: The root of the directory tree containing the
                L1C files.
            date: The date of the file used to determine sub-folders
                corresponding to month and day.

        Return:
            L1CFile object providing access to the requested file.
        """
        if date is not None:
            date = pd.TimeStamp(date)
            year = date.year - 2000
            month = date.month
            day = date.day
            path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"
            files = path.glob(f"1C-R.GPM.GMI.*{granule:06}.V05A.HDF5")
        else:
            path = Path(path)
            files = path.glob(f"**/1C-R.GPM.GMI.*{granule:06}.V05A.HDF5")

        try:
            f = next(iter(files))
            return L1CFile(f)
        except StopIteration:
            if date is not None:
                return cls.open_granule(granule, path, None)
            raise Exception(
                f"Could not find a L1C file with granule number {granule}."
            )

    @classmethod
    def find_file(cls, date, path):
        """
        Find and open L1C file with a given granule number.

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
        files = list(data_path.glob(f"1C-R.GPM.GMI.*.V05A.HDF5"))
        files += list(path.glob(f"1C-R.GPM.GMI.*.V05A.HDF5"))

        start_times = []
        end_times = []
        for f in files:
            match = FILENAME_RE.match(f.name)
            year = match.group(2)
            start = match.group(3)
            end = match.group(4)

            start = np.datetime64(
                f"{year[0:4]}-{year[4:6]}-{year[6:8]}"
                f"T{start[0:2]}:{start[2:4]}:{start[4:6]}"
            )
            end = np.datetime64(
                f"{year[0:4]}-{year[4:6]}-{year[6:8]}"
                f"T{end[0:2]}:{end[2:4]}:{end[4:6]}"
            )
            if end < start:
                end += np.timedelta64(1, "D")
            start_times.append(start)
            end_times.append(end)
        start_times = np.array(start_times)
        end_times = np.array(end_times)
        date = date.to_datetime64()

        ind = np.where((start_times <= date) * (end_times >= date))[0][0]
        filename = files[ind]

        return L1CFile(filename)

    @classmethod
    def find_files(cls, date, roi, path):
        """
        Find and open L1C file with a given granule number.

        Args:
            date: The date of the file used to determine sub-folders
                corresponding to month and day.
            roi: Tuple ``(lon_min, lat_min, lon_max, lat_max)`` describing a
                rectangular bounding box around the region of interest.
            path: The root of the directory tree containing the
                L1C files.

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
        files = list(
            data_path.glob(
                f"1C-R.GPM.GMI.*{date.year:04}{month:02}{day:02}*.V05A.HDF5"
            )
        )
        files += list(
            path.glob(
                f"1C-R.GPM.GMI.*{date.year:04}{month:02}{day:02}*.V05A.HDF5"
            )
        )

        for f in files:
            f = L1CFile(f)
            if f.covers_roi(roi):
                yield f

    def __init__(self, path):
        """
        Open a GPROF GMI L1C file.

        Args:
            path: The path to the file.
        """
        self.filename = path
        self.path = Path(path)

    def __repr__(self):
        """String representation for file."""
        return f"L1CFile(filename='{self.path.name}')"

    def extract_scans(self, roi, output_filename):
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

        with h5py.File(self.path, "r") as input:
            lats = input["S1/Latitude"][:]
            lons = input["S1/Longitude"][:]

            indices = np.where(np.any(
                (lats > lat_min) * (lats < lat_max) *
                (lons > lon_min) * (lons < lon_max),
                axis=-1
            ))[0]

            with h5py.File(output_filename, "w") as output:

                g = output.create_group("S1")
                n_scans = indices.size
                for name, item in input["S1"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g.create_dataset(name,
                                         shape=(n_scans, ) + shape[1:],
                                         data=item[indices])

                for a in input["S1"].attrs:
                    s = input["S1"].attrs[a].decode()
                    s = _RE_META_INFO.sub(f"NumberScansGranule={n_scans};", s)
                    s = np.bytes_(s)
                    g.attrs[a] = s

                g_st = g.create_group("ScanTime")
                for name, item in input["S1/ScanTime"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g_st.create_dataset(name,
                                            shape=(n_scans, ) + shape[1:],
                                            data=item[indices])

                g_sc = g.create_group("SCstatus")
                for name, item in input["S1/SCstatus"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g_sc.create_dataset(name,
                                            shape=(n_scans, ) + shape[1:],
                                            data=item[indices])

                g = output.create_group("S2")
                for name, item in input["S2"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g.create_dataset(name,
                                         shape=(n_scans, ) + shape[1:],
                                         data=item[indices])
                for a in input["S2"].attrs:
                    s = input["S2"].attrs[a].decode()
                    s = _RE_META_INFO.sub(f"NumberScansGranule={n_scans};", s)
                    s = np.bytes_(s)
                    g.attrs[a] = s

                g_st = g.create_group("ScanTime")
                for name, item in input["S2/ScanTime"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g_st.create_dataset(name,
                                            shape=(n_scans, ) + shape[1:],
                                            data=item[indices])

                g_sc = g.create_group("SCstatus")
                for name, item in input["S2/SCstatus"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g_sc.create_dataset(name,
                                            shape=(n_scans, ) + shape[1:],
                                            data=item[indices])

                for a in input.attrs:
                    output.attrs[a] = input.attrs[a]

    def extract_scans_and_pixels(self,
                                 scans,
                                 output_filename):
        """
        Extract first pixel from each scan in file.

        The main purposed of this method is to simplify the generation
        of small files for testing purposes.

        Args:
            scans: Indices of the scans to extract.
            output_filename: Name of the file to which to write the extracted
                 scans.
        """
        with h5py.File(self.path, "r") as input:
            lats = input["S1/Latitude"][scans, 0]
            lons = input["S1/Longitude"][scans, 0]

            with h5py.File(output_filename, "w") as output:

                g = output.create_group("S1")
                n_scans = len(scans)
                n_pixels = 1
                for name, item in input["S1"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g.create_dataset(name,
                                         shape=(n_scans, n_pixels) + shape[2:],
                                         data=item[scans, 0])

                g_st = g.create_group("ScanTime")
                for name, item in input["S1/ScanTime"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        if len(shape) > 1:
                            g_st.create_dataset(name,
                                                shape=(n_scans, n_pixels) + shape[2:],
                                                data=item[scans, 0])
                        else:
                            g_st.create_dataset(name,
                                                shape=(n_scans,),
                                                data=item[scans])

                g_sc = g.create_group("SCstatus")
                for name, item in input["S1/SCstatus"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        if len(shape) > 1:
                            g_sc.create_dataset(name,
                                                shape=(n_scans, n_pixels) + shape[2:],
                                                data=item[scans, 0])
                        else:
                            g_sc.create_dataset(name,
                                                shape=(n_scans,),
                                                data=item[scans])

                g = output.create_group("S2")
                for name, item in input["S2"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g.create_dataset(name,
                                         shape=(n_scans, n_pixels) + shape[2:],
                                         data=item[scans, 0])

                g_st = g.create_group("ScanTime")
                for name, item in input["S2/ScanTime"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        if len(shape) > 1:
                            g_st.create_dataset(name,
                                                shape=(n_scans, n_pixels) + shape[2:],
                                                data=item[scans, 0])
                        else:
                            g_st.create_dataset(name,
                                                shape=(n_scans,),
                                                data=item[scans])

                g_sc = g.create_group("SCstatus")
                for name, item in input["S2/SCstatus"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        if len(shape) > 1:
                            g_sc.create_dataset(name,
                                                shape=(n_scans, n_pixels) + shape[2:],
                                                data=item[scans, 0])
                        else:
                            g_sc.create_dataset(name,
                                                shape=(n_scans,),
                                                data=item[scans])

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
        lon_min, lat_min, lon_max, lat_max = roi
        with h5py.File(self.path, "r") as input:
            lats = input["S1/Latitude"][:]
            lons = input["S1/Longitude"][:]
            return np.any(
                (lons >= lon_min) * (lons < lon_max) *
                (lats >= lat_min) * (lats < lat_min)
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
        with h5py.File(self.path, "r") as input:

            lats = input["S1/Latitude"][:]
            lons = input["S1/Longitude"][:]

            if roi is not None:
                lon_min, lat_min, lon_max, lat_max = roi
                indices = np.any(
                    (lons >= lon_min) * (lons < lon_max) *
                    (lats >= lat_min) * (lats < lon_min),
                    axis=-1
                )
            else:
                indices = slice(0, None)

            lats = lats[indices]
            lons = lons[indices]

            lats_sc = input["S1/SCstatus/SClatitude"][indices]
            lons_sc = input["S1/SCstatus/SClongitude"][indices]
            alt_sc = input["S1/SCstatus/SClongitude"][indices]
            tbs = np.concatenate([input["S1/Tc"][indices, :],
                                  input["S2/Tc"][indices, :]],
                                 axis=-1)

            n_scans = lats.shape[0]
            times = np.zeros(n_scans, dtype='datetime64[ms]')

            year = input["S1/ScanTime/Year"][indices]
            month = input["S1/ScanTime/Month"][indices]
            day_of_month = input["S1/ScanTime/DayOfMonth"][indices]
            hour = input["S1/ScanTime/Hour"][indices]
            minute = input["S1/ScanTime/Minute"][indices]
            second = input["S1/ScanTime/Second"][indices]
            milli_second = input["S1/ScanTime/MilliSecond"][indices]
            for i in range(n_scans):
                times[i] = datetime(year[i],
                                    month[i],
                                    day_of_month[i],
                                    hour[i],
                                    minute[i],
                                    second[i],
                                    milli_second[i] * 1000)

            dims = ("scans", "pixels")
            data = {
                "latitude": (dims, lats),
                "longitude": (dims, lons),
                "spacecraft_latitude": (dims[:1], lats_sc),
                "spacecraft_longitude": (dims[:1], lons_sc),
                "spacecraft_altitude": (dims[:1], alt_sc),
                "brightness_temperatures": (dims + ("channels",), tbs),
                "scan_time": (dims[:1], times)
            }

        return xr.Dataset(data)
