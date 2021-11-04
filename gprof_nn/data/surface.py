"""
=====================
gprof_nn.data.surface
=====================

This module contains files to load surface type maps for different sensors.

"""
import gzip
import io
from pathlib import Path
import struct
import subprocess
from tempfile import mkstemp

import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
from scipy.signal import convolve

from gprof_nn.data.preprocessor import PREPROCESSOR_SETTINGS


LANDMASKS = {
        "AMSRE": "landmask51_32.bin",
        "AMSR2": "landmask42_32.bin",
        "GMI": "landmask32_32.bin",
        "SSMI": "landmask69_32.bin",
        "SSMIS": "landmask74_32.bin",
        "TMI": "landmask60_32.bin",
        "ATMS": "landmask34_16.bin",
        "MHS": "landmask34_16.bin"
        }


def read_landmask(sensor):
    """
    Read the landmask for a given sensor.

    Args:
        sensor: The sensor for which to read the landmask.

    Return:
        An xarray dataset containing the land mask with corresponding
        latitude and longitude coordinates.
    """
    ancdir = Path(PREPROCESSOR_SETTINGS["ancdir"])
    path = ancdir / LANDMASKS[sensor]
    data = np.fromfile(ancdir / LANDMASKS[sensor], dtype="i1")

    # Filename contains the resolution per degree
    resolution = int(path.stem.split("_")[-1])

    shape = (360 * resolution, 180 * resolution)

    lats = np.linspace(-90, 90, shape[1] + 1)
    lats = 0.5 * (lats[:-1] + lats[1:])
    lons = np.linspace(-180, 180, shape[0] + 1)
    lons = 0.5 * (lons[:-1] + lons[1:])
    data = data.reshape(shape)

    dataset = xr.Dataset({
        "latitude": (("latitude",), lats),
        "longitude": (("longitude",), lons),
        "mask": (("longitude", "latitude"), data)
    })
    return dataset


def read_autosnow(date):
    """
    Read autosnow mask for a given date.

    Args:
        date: The date for which to read the autosnow file.

    Return:
        An xarray dataset containing the autosnow mask with the
        with the postprocessing similar to the CSU preprocessor.
    """
    ingest_dir = Path(PREPROCESSOR_SETTINGS["ingestdir"])

    date = pd.Timestamp(date)

    N_LAT = 4500
    N_LON = 9000
    MISSING = -99

    filename = None
    for i in range(31):
        delta = pd.Timedelta(i, 'D')
        search_date = date - delta
        year = search_date.year
        day = search_date.dayofyear
        year_s = f"{year:04}"
        day_s = f"{day:03}"
        path = ingest_dir / "autosnowV3" / year_s
        filename = path / f"gmasi_snowice_reproc_v003_{year_s}{day_s}.Z"

        if filename.exists():
            break

    if filename is None:
        raise Exception(f"Couldn't find autosnow file for date {date}")

    if filename.suffix == ".Z":
        _, tmp = mkstemp()
        tmp = Path(tmp)
        try:
            with open(tmp, "wb") as buffer:
                subprocess.run(["gunzip", "-c", filename],
                            stdout=buffer,
                            check=True)
            data = np.fromfile(tmp, "i1")
        finally:
            tmp.unlink()
    else:
        data = np.fromfile(filename, "rb")

    data = data.reshape(N_LAT, N_LON)

    # Mark invalid values
    mask = (data < 0) + (data > 3)
    data[mask] = MISSING

    for i in range(1, 10):

        # South-East direction
        valid = mask[:-i, :-i]
        missing_mask = valid == MISSING
        source = mask[i:, i:]
        replace = missing_mask * (source != MISSING)
        valid[replace] = source[replace]

        # North-East direction
        valid = mask[i:, :-i]
        missing_mask = valid == MISSING
        source = mask[:-i, i:]
        replace = missing_mask * (source != MISSING)
        valid[replace] = source[replace]

        # South-West direction
        valid = mask[:-i, i:]
        missing_mask = valid == MISSING
        source = mask[i:, :-i]
        replace = missing_mask * (source != MISSING)
        valid[replace] = source[replace]

        # North-East direction
        valid = mask[i:, i:]
        missing_mask = valid == MISSING
        source = mask[:-i, :-i]
        replace = missing_mask * (source != MISSING)
        valid[replace] = source[replace]

    # Replace 3 that have too few neighbors
    k = np.ones((3, 3))
    k[1, 1] = 0
    sums = convolve(mask, k, mode="valid")
    source = data[1:-1, 1:-1]
    source[sums <= 12] = 0

    return data





