"""
==================
gprof_nn.data.mrms
==================

Interface class to read GMI-MRMS match ups used over snow surfaces.
"""
import gzip
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
from pykdtree.kdtree import KDTree

from gprof_nn.coordinates import latlon_to_ecef


DATA_RECORD_TYPES = np.dtype(
    [
        ("latitude", "f4"),
        ("longitude", "f4"),
        ("scan_time", f"5i4"),
        ("quality_flag", f"f4"),
        ("surface_precip", "f4"),
        ("surface_rain", "f4"),
        ("convective_rain", "f4"),
        ("stratiform_rain", "f4"),
        ("snow", "f4"),
        ("quality_index", "f4"),
        ("gauge_fraction", "f4"),
        ("standard_deviation", "f4"),
        ("n_stratiform", "i4"),
        ("n_convective", "i4"),
        ("n_rain", "i4"),
        ("n_snow", "i4"),
        ("fraction_missing", "f4"),
        ("brightness_temperatures", "15f4"),
    ]
)


class MRMSMatchFile:
    """
    Class to read GMI-MRMS match up files.

    Attributes:
        data: Numpy structured array holding raw MRMS match-up data.
        n_obs: The number of observations in the file.
        scan_time: Array holding the scan times for each match up.
        year: The year from which the observations stem.
        month: The month from which the observations stem.
    """

    file_pattern = "????_MRMS2GMI_gprof_db_??all.bin.gz"

    @classmethod
    def find_files(cls, path):
        """
        Generator providing access to all files that match the naming scheme
        for GMI-MRMS match file in a give folder.

        Args:
            path: Path to the directory containing the GMI-MRMS matchup files.

        Return:
            Generator object returning the paths of all GMI-MRMS match up
            files in the given directory.
        """
        path = Path(path)
        return list(path.glob(cls.file_pattern))

    def __init__(self, filename):
        """
        Reads gzipped matchup file.

        Args:
            filename: The name of the file to read.
        """
        with open(filename, "rb") as source:
            buffer = gzip.decompress(source.read())
        self.data = np.frombuffer(buffer, DATA_RECORD_TYPES)
        self.n_obs = self.data.size

        self.scan_time = np.zeros(self.n_obs, dtype="datetime64[ns]")
        for i in range(self.n_obs):
            dates = self.data["scan_time"]
            year = dates[i, 0]
            month = dates[i, 1]
            day = dates[i, 2]
            hour = dates[i, 3]
            minute = dates[i, 4]
            self.scan_time[i] = np.datetime64(
                f"{year:04}-{month:02}-{day:02}" f"T{hour:02}:{minute:02}:00"
            )

        name = Path(filename).name
        self.year = int(name[:2])
        self.month = int(name[2:4])

    def to_xarray_dataset(self, day=None):
        """
        Load data into xarray.Dataset.

        Args:
            day: If given only the data for a given day of the month will
                be included in the dataset.

        Return:
            xarray.Dataset containing the MRMS match-up data.
        """
        if day is not None:
            indices = self.data["scan_time"][:, 2] == day
            if not np.any(indices):
                return xr.Dataset()

        else:
            indices = np.arange(self.data.size)
        data = self.data[indices]
        dims = ("samples",)
        dataset = {}
        for k in DATA_RECORD_TYPES.names:
            if k == "brightness_temperatures":
                ds = dims + ("channel",)
            elif k == "scan_time":
                dataset[k] = (("samples",), self.scan_time[indices])
                continue
            else:
                ds = dims
            dataset[k] = (ds, data[k])
        return xr.Dataset(dataset)

    def match_targets(self, input_data):
        """
        Match available retrieval targets from MRMS data to points in
        xarray dataset.

        Args:
            input_data: xarray dataset containing the input data from
                the preprocessor.
            targets: List of retrieval target variables to extract from
                the sim file.
        Return:
            The input dataset but with the surface_precip field added.
        """
        start_time = input_data["scan_time"].data[0]
        end_time = input_data["scan_time"].data[-1]
        indices = (self.scan_time >= start_time) * (self.scan_time < end_time)

        data = self.data[indices]

        n_scans = input_data.scans.size
        n_pixels = 221

        lats_1c = input_data["latitude"].data.reshape(-1, 1)
        lons_1c = input_data["longitude"].data.reshape(-1, 1)
        coords_1c = latlon_to_ecef(lons_1c, lats_1c)
        coords_1c = np.concatenate(coords_1c, axis=1)

        lats = data["latitude"].reshape(-1, 1)
        lons = data["longitude"].reshape(-1, 1)
        z = np.zeros_like(lats)
        coords_sim = latlon_to_ecef(lons, lats)
        coords_sim = np.concatenate(coords_sim, 1)

        kdtree = KDTree(coords_1c)
        dists, indices = kdtree.query(coords_sim)

        matched = np.zeros(n_scans * n_pixels)
        matched[:] = np.nan
        matched[indices] = data["surface_precip"]
        matched[indices][dists > 5e3] = np.nan
        matched = matched.reshape((n_scans, n_pixels))

        input_data["surface_precip"] = (("scans", "pixels"), matched)
        return input_data
