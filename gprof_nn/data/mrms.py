"""
==================
gprof_nn.data.mrms
==================

Interface class to read GPROF MRMS match ups used over snow surfaces.
"""
import gzip
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
from pykdtree.kdtree import KDTree

from gprof_nn import sensors
from gprof_nn.coordinates import latlon_to_ecef


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

    file_pattern = "????_MRMS2{sensor}_*.bin.gz"

    @classmethod
    def find_files(cls, path, sensor=sensors.GMI):
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
        return list(path.glob(cls.file_pattern.format(sensor=sensor.name)))

    def __init__(self, filename, sensor=None):
        """
        Reads gzipped matchup file.

        Args:
            filename: The name of the file to read.
        """
        filename = Path(filename)
        if sensor is None:
            if "GMI" in filename.name:
                sensor = sensors.GMI
            elif "MHS" in filename.name:
                sensor = sensors.MHS
            else:
                raise ValueError(
                    "Could not infer sensor from filename. Consider passing "
                    "the sensor argument explicitly."
                )
        self.sensor = sensor

        with open(filename, "rb") as source:
            buffer = gzip.decompress(source.read())
        self.data = np.frombuffer(buffer, sensor.mrms_file_record)
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
        for k in self.sensor.mrms_file_record.names:
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
        n_pixels = input_data.pixels.size

        if indices.sum() <= 0:
            surface_precip = np.zeros((n_scans, n_pixels))
            surface_precip[:] = np.nan
            input_data["surface_precip"] = (("scans", "pixels"), surface_precip)

            input_data["convective_precip"] = (("scans", "pixels"), surface_precip)
            return input_data

        lats_1c = input_data["latitude"].data.reshape(-1, 1)
        lons_1c = input_data["longitude"].data.reshape(-1, 1)
        coords_1c = latlon_to_ecef(lons_1c, lats_1c)
        coords_1c = np.concatenate(coords_1c, axis=1)

        lats = data["latitude"].reshape(-1, 1)
        lons = data["longitude"].reshape(-1, 1)
        coords_sim = latlon_to_ecef(lons, lats)
        coords_sim = np.concatenate(coords_sim, 1)

        kdtree = KDTree(coords_1c)
        dists, indices = kdtree.query(coords_sim)

        matched = np.zeros(n_scans * n_pixels)
        matched[:] = np.nan
        matched[indices] = data["surface_precip"]

        mrms_ratios = get_mrms_ratios()
        ratios = mrms_ratios.interp(
            latitude=xr.DataArray(lats.ravel(), dims="samples"),
            longitude=xr.DataArray(lons.ravel(), dims="samples"),
        )
        corrected = data["surface_precip"] - data["snow"] + data["snow"] * ratios
        corrected[ratios == 1.0] = np.nan

        matched[indices] = corrected
        matched[indices][dists > 15e3] = np.nan
        matched = matched.reshape((n_scans, n_pixels))
        input_data["surface_precip"] = (("scans", "pixels"), matched)

        matched = np.zeros(n_scans * n_pixels)
        matched[:] = np.nan
        matched[indices] = data["convective_rain"]
        matched[indices][dists > 15e3] = np.nan
        matched = matched.reshape((n_scans, n_pixels))
        input_data["convective_precip"] = (("scans", "pixels"), matched)

        return input_data


################################################################################
# MRMS / snodas correction factors
################################################################################

_RATIO_FILE = (
    "/qdata1/pbrown/dbaseV7/mrms_snow_scale_factors/"
    "201710-201805_10km_snodas_mrms_ratio_scale.asc."
    "bin"
)

_MRMS_RATIOS = None


def has_snowdas_ratios():
    """
    Simple test function to determine whether snowdas ratio files are
    present on system.
    """
    return Path(_RATIO_FILE).exists()


def get_mrms_ratios():
    """
    Cached loading of the MRMS correction factors into an xarray
    data array.

    Return:
        xrray.Dataarray containing the MRMS/SNODAS ratios used to correct
        MRMS snow.
    """
    global _MRMS_RATIOS
    if _MRMS_RATIOS is None:
        with open(_RATIO_FILE, "rb") as file:
            buffer = file.read()
            lon_ll = -129.994995117188
            d_lon = 0.009998570017
            n_lon = 7000
            lons = lon_ll + d_lon * np.arange(n_lon)

            lat_ll = 20.005001068115
            d_lat = 0.009997142266
            n_lat = 3500
            lats = lat_ll + d_lat * np.arange(n_lat)

            offset = 2 * 2 + 4 * 8 + 4
            array = np.frombuffer(
                buffer, dtype="f4", offset=offset, count=n_lon * n_lat
            )
            ratios = array.reshape((n_lat, n_lon)).copy()
            ratios[ratios < 0] = np.nan

            _MRMS_RATIOS = xr.DataArray(
                data=ratios,
                dims=["latitude", "longitude"],
                coords={"latitude": lats, "longitude": lons},
            ).fillna(1.0)
    return _MRMS_RATIOS
