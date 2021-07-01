"""
=====================
gprof_nn.data.surface
=====================

This module contains files to load surface type maps for different sensors.

"""
from pathlib import Path
import struct

import numpy as np
import pandas as pd
import xarray as xr


def has_surface_type_maps():
    """
    Helper function to detect whether surface type maps are available
    on the system.
    """
    path = Path("/qdata1/pbrown/gpm/surfdat/")
    return path.exists()


def get_surface_type_map(time,
                         sensor="GMI"):
    """
    Return dataset contining global surface types for a given
    time.

    Args:
        time: datetime object specifying the date for which
            to load the surface type.
        sensor: Name of the sensor for which to load the surface type.

    Rerturn:
        'xarray.DataArray' containing the global surface types.
    """
    time = pd.to_datetime(time)
    year = time.year - 2000
    month = time.month
    day = time.day

    filename = (f"/qdata1/pbrown/gpm/surfdat/{sensor}_surfmap_{year:02}"
                f"{month:02}_V7.dat")

    with open(filename, "rb") as file:
        n_lon, = struct.unpack("i", file.read(4))
        n_lat, = struct.unpack("i", file.read(4))

    N_LON = 32 * 360
    N_LAT = 32 * 180

    lats = np.linspace(-90, 90, n_lat + 1)
    lats = 0.5 * (lats[1:] + lats[:-1])
    lons = np.linspace(-180, 180, n_lon + 1)
    lons = 0.5 * (lons[1:] + lons[:-1])

    offset = (day - 1) * (20 + n_lon* n_lat) + 20
    count = n_lon * n_lat 
    data = np.fromfile(filename, count=count, offset=offset, dtype="u1")
    data = data.reshape((n_lat, n_lon))
    data = data[::-1]

    attrs = np.fromfile(filename, count=5, offset=offset - 20, dtype="i4")

    arr = xr.DataArray(
        data=data,
        dims=["latitude", "longitude"],
        coords={
            "latitude": lats,
            "longitude": lons
        }
    )
    arr.attrs["header"] = attrs

    mountain_mask = get_mountain_mask()
    lons = arr.longitude.data.copy()
    lons = np.where(lons < 0.0, lons + 360, lons)
    mountain = mountain_mask.interp(
        longitude=lons,
        latitude=arr.latitude,
        method="nearest"
    )
    mountain = mountain.data >= 1
    land = (arr.data >= 3) * (arr.data <= 7)
    arr.data[mountain * land] = 17
    snow = (arr.data >= 8) * (arr.data <= 11)
    arr.data[mountain * snow] = 18

    return arr


_MOUNTAIN_MASK = None
_MOUNTAIN_MASK_FILE = "/gdata/simon/mountain_mask.bin"


def get_mountain_mask():
    """
    Load surface altitude mask indicating the presence of mountains on the
    Earth surface.

    Return:
        A ``xarray.DataArray`` containing the mask and corresponding latitude
        and longitude coordinates.
    """
    global _MOUNTAIN_MASK
    if _MOUNTAIN_MASK is None:
        with open(_MOUNTAIN_MASK_FILE, "rb") as file:
            file.read(4)
            n_lon, = struct.unpack("i", file.read(4))
            n_lat, = struct.unpack("i", file.read(4))
            file.read(8)
            data = np.fromfile(file, dtype="i", count=n_lon * n_lat)
            data = data.reshape((n_lat, n_lon))

            lons = np.linspace(0, 360, n_lon + 1)
            lons = 0.5 * (lons[1:] + lons[:-1])
            lats = np.linspace(-90, 90, n_lat + 1)
            lats = 0.5 * (lats[1:] + lats[:-1])
            _MOUNTAIN_MASK = xr.DataArray(
                data=data,
                dims=["latitude", "longitude"],
                coords={
                    "latitude": lats,
                    "longitude": lons
                })
    return _MOUNTAIN_MASK


def get_surface_type_map_legacy(time,
                                sensor="GMI"):
    """
    Return dataset contining pre GPROF V6 global surface types for given
    data. 

    Intended for testing purposes.

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
