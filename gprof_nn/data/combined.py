"""
======================
gprof_nn.data.combined
======================

Interface to read in L2b files of the GPM combined product.
"""
from datetime import datetime
from pathlib import Path

from h5py import File

import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import convolve
import xarray as xr

def calculate_smoothing_kernels(
        fwhm_a,
        fwhm_x):
    """
    Calculate smoothing kernels for GPM combined data.

    Args:
        fwhm_a: The full width at half maximum in along-track direction.
        fwhm_x: The fill width at half maximum in across-track direction.

    Return:
        'numpy.ndarray' containing the convolution kernel to apply the
        smoothing.
    """
    d_a = 4.9e3
    d_x = 5.09e3
    n_a = int(3 * fwhm_a / d_a)
    n_x = int(3 * fwhm_x / d_x)

    k = np.ones((2 * n_a + 1, 2 * n_x + 1), dtype=np.float32)
    x_a = np.arange(-n_a, n_a + 1).reshape(-1, 1) * d_a
    x_x = np.arange(-n_x, n_x + 1).reshape(1, -1) * d_x
    r = np.sqrt((x_a / fwhm_a) ** 2 + (x_x / fwhm_x) ** 2)
    k = np.exp(np.log(0.5) * r ** 2)
    k /= k.sum()
    return k


def smooth_field(field, kernel):
    """
    Smooth field using a given convolution kernel.

    Args:
        field: A 2D array containing a variable to smooth.
        kernel: The smoothing kernel to use to smooth the field.

    Return:
        The smoothed field.
    """
    field_s = convolve(field, kernel, mode="same")
    weights = convolve(np.ones_like(field), kernel, mode="same")
    field_s /= weights
    return field_s


class GPMCMBFile:
    """
    Class to read in GPM combined data.
    """
    def __init__(self, filename):
        """
        Create GPMCMB object to read a given file.

        Args:
            filename: Path pointing to the file to read.

        """
        self.filename = Path(filename)
        time = self.filename.stem.split(".")[4][:-8]
        self.start_time = datetime.strptime(
            time,
            "%Y%m%d-S%H%M%S"
        )

    def to_xarray_dataset(self,
                          smooth=False,
                          roi=None):
        """
        Load data in file into 'xarray.Dataset'.

        Args:
            smooth: If set to true the 'surface_precip' field will be smoothed
                to match the footprint of the GMI 23.8 GHz channels.
            roi: Optional bounding box given as list
                 ``[lon_0, lat_0, lon_1, lat_1]`` specifying the longitude
                 and latitude coordinates of the lower left
                 (``lon_0, lat_0``) and upper right (``lon_1, lat_1``)
                 corners. If given, only scans containing at least one pixel
                 within the given bounding box will be returned.
        """
        with File(str(self.filename), "r") as data:

            data = data['MS']
            latitude = data["Latitude"][:]
            longitude = data["Longitude"][:]

            if roi is not None:
                lon_0, lat_0, lon_1, lat_1 = roi
                inside = ((longitude >= lon_0) *
                          (latitude >= lat_0) *
                          (longitude < lon_1) *
                          (latitude < lat_1))
                inside = np.any(inside, axis=1)
                i_start, i_end = np.where(inside)[0][[0, -1]]
            else:
                i_start = 0
                i_end = latitude.shape[0]

            latitude = latitude[i_start:i_end]
            longitude = longitude[i_start:i_end]

            date = {
                "year": data["ScanTime"]["Year"][i_start:i_end],
                "month": data["ScanTime"]["Month"][i_start:i_end],
                "day": data["ScanTime"]["DayOfMonth"][i_start:i_end],
                "hour": data["ScanTime"]["Hour"][i_start:i_end],
                "minute": data["ScanTime"]["Minute"][i_start:i_end],
                "second": data["ScanTime"]["Second"][i_start:i_end]
            }
            date = pd.to_datetime(date)

            surface_precip = data["surfPrecipTotRate"][i_start:i_end]
            if smooth:
                k = calculate_smoothing_kernels(16e3, 10e3)
                surface_precip = smooth_field(surface_precip, k)

            dataset = xr.Dataset({
                "scan_time": (("scans",), date),
                "latitude": (("scans", "pixels"), latitude),
                "longitude": (("scans", "pixels"), longitude),
                "surface_precip": (("scans", "pixels"), surface_precip)
            })
            return dataset
