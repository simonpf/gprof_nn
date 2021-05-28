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

        if indices.sum() <= 0:
            surface_precip = np.zeros((n_scans, n_pixels))
            surface_precip[:] = np.nan
            input_data["surface_precip"] = (("scans", "pixels"),
                                            surface_precip)


            input_data["convective_precip"] = (("scans", "pixels"),
                                               surface_precip)
            return input_data

        lats_1c = input_data["latitude"].data.reshape(-1, 1)
        lons_1c = input_data["longitude"].data.reshape(-1, 1)
        coords_1c = latlon_to_ecef(lons_1c, lats_1c)
        coords_1c = np.concatenate(coords_1c, axis=1)

        lats = data["latitude"].reshape(-1, 1)
        lons = data["longitude"].reshape(-1, 1)
        coords_sim = latlon_to_ecef(lons, lats)
        coords_sim = np.concatenate(coords_sim, 1)

        surface_types = get_surface_type_map(start_time)
        surface_types = surface_types.interp(
            latitude=input_data["latitude"],
            longitude=input_data["longitude"],
            method="nearest"
        )
        input_data["surface_type"].data[:] = surface_types.data

        kdtree = KDTree(coords_1c)
        dists, indices = kdtree.query(coords_sim)

        matched = np.zeros(n_scans * n_pixels)
        matched[:] = np.nan
        matched[indices] = data["surface_precip"]

        mrms_ratios = get_mrms_ratios()
        ratios = mrms_ratios.interp(
            latitude=xr.DataArray(lats.ravel(), dims="samples"),
            longitude=xr.DataArray(lons.ravel(), dims="samples")
        )
        corrected = (data["surface_precip"] -
                     data["snow"] +
                     data["snow"] * ratios)
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

_RATIO_FILE = ("/qdata1/pbrown/dbaseV7/mrms_snow_scale_factors/"
              "201710-201805_10km_snodas_mrms_ratio_scale.asc."
              "bin")

_MRMS_RATIOS = None

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
            array = np.frombuffer(buffer,
                                  dtype="f4",
                                  offset=offset,
                                  count=n_lon * n_lat)
            ratios = array.reshape((n_lat, n_lon)).copy()
            ratios[ratios < 0] = np.nan

            _MRMS_RATIOS = xr.DataArray(
                data=ratios,
                dims=["latitude", "longitude"],
                coords={
                    "latitude": lats,
                    "longitude": lons
                }
            ).fillna(1.0)
    return _MRMS_RATIOS


def get_surface_type_map(time,
                         sensor="GMI"):
    """
    Return dataset contining global surface types for a given
    data.

    Args:
        time: datetime object specifying the date for which
            to load the surface type.
        sensor: Name of the sensor for which to load the surface type.

    Rerturn:
        xarray.DataArray containing the global surface types.
    """
    time = pd.to_datetime(time)
    year = time.year - 2000
    month = time.month
    day = time.day

    filename = (f"/xdata/drandel/gpm/surfdat/{sensor}_surfmap_{year:02}"
                f"{month:02}_V7.dat")

    N_LON = 32 * 360
    N_LAT = 32 * 180

    LATS = np.arange(-90, 90, 1.0 / 32)
    LONS = np.arange(-180, 180, 1.0 / 32)

    offset = (day - 1) * (20 + N_LON * N_LAT) + 20
    count = N_LON * N_LAT
    data = np.fromfile(filename, count=count, offset=offset, dtype="u1")
    data = data.reshape((N_LAT, N_LON))
    data = data[::-1]

    attrs = np.fromfile(filename, count=5, offset=offset - 20, dtype="i4")

    arr = xr.DataArray(
        data=data,
        dims=["latitude", "longitude"],
        coords={
            "latitude": LATS,
            "longitude": LONS
        }
    )
    arr.attrs["header"] = attrs
    return arr


def get_surface_type_map_legacy(time,
                                sensor="GMI"):
    """
    Return dataset contining pre GPROF V6 global surface types for given
    data.

    Args:
        time: datetime object specifying the date for which
            to load the surface type.
        sensor: Name of the sensor for which to load the surface type.

    Rerturn:
        xarray.DataArray containing the global surface types.
    """
    time = pd.to_datetime(time)
    year = time.year - 2000
    month = time.month
    day = time.day

    filename = (f"/xdata/drandel/gpm/surfdat/{sensor}_surfmap_{year:02}"
                f"{month:02}_V3.dat")

    N_LON = 16 * 360
    N_LAT = 16 * 180

    LATS = np.arange(-90, 90, 1.0 / 16)
    LONS = np.arange(-180, 180, 1.0 / 16)

    offset = (day - 1) * (20 + N_LON * N_LAT) + 20
    count = N_LON * N_LAT
    data = np.fromfile(filename, count=count, offset=offset, dtype="u1")
    data = data.reshape((N_LAT, N_LON))
    data = data[::-1]
    attrs = np.fromfile(filename, count=5, offset=offset - 20, dtype="i4")

    arr = xr.DataArray(
        data=data,
        dims=["latitude", "longitude"],
        coords={
            "latitude": LATS,
            "longitude": LONS
        }
    )
    arr.attrs["header"] = attrs
    return arr