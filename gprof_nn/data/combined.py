"""
======================
gprof_nn.data.combined
======================

Interface to read in L2b files of the GPM combined product.
"""
from copy import copy
from datetime import datetime
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import pandas as pd
import scipy
from scipy.signal import convolve
import xarray as xr


# Known dimensions of variables of combined data in .bin
# format.
BIN_DIMENSIONS = {
    "pixels": 25,
    "levels": 88,
    "spectral_levels": 80,
    "nodes": 10,
    "env_nodes": 10,
    "psd_nodes": 9,
    "phase_nodes": 5,
    "channels": 13,
    "top_bottom": 2,
    "indices": 2,
    "date": 6
}

# Types of the variables of the combined data in .bin
# format.
BIN_TYPES = [
    ("latitude", "f4", ("scans", "pixels")),
    ("longitude", "f4", ("scans", "pixels")),
    ("elevation", "f4", ("scans", "pixels")),
    ("surface_type", "f4", ("scans", "pixels")),
    ("precip_flag", "f4", ("scans", "pixels")),
    ("precip_type", "f4", ("scans", "pixels")),
    ("quality", "f4", ("scans", "pixels")),
    ("surface_range_bin", "f4", ("scans", "pixels")),
    ("surface_pressure", "f4", ("scans", "pixels")),
    ("skin_temperature", "f4", ("scans", "pixels")),
    ("envnode", "f4", ("scans", "pixels", "nodes")),
    ("water_vapor", "f4", ("scans", "pixels", "nodes")),
    ("pressure", "f4", ("scans", "pixels", "nodes")),
    ("surface_air_temperature", "f4", ("scans", "pixels")),
    ("temperature", "f4", ("scans", "pixels", "nodes")),
    ("ten_meter_wind", "f4", ("scans", "pixels")),
    ("emissivity", "f4", ("scans", "pixels", "channels")),
    ("surface_precip", "f4", ("scans", "pixels")),
    ("cloud_water", "f4", ("scans", "pixels", "levels")),
    ("phase_bin_nodes", "f4", ("scans", "pixels", "phase_nodes")),
    ("total_water_content", "f4", ("scans", "pixels", "levels")),
    ("total_precip_rate", "f4", ("scans", "pixels", "levels")),
    ("liquid_water_content", "f4", ("scans", "pixels", "levels")),
    ("total_precip_d0", "f4", ("scans", "pixels", "levels")),
    ("total_precip_n0", "f4", ("scans", "pixels", "levels")),
    ("total_precip_mu", "f4", ("scans", "pixels", "levels")),
    ("simulated_brightness_temperatures", "f4", ("scans", "pixels", "channels")),
    ("pia", "f4", ("scans", "pixels", "top_bottom")),
    ("radar_reflectivity", "f4", ("scans", "pixels", "levels", "top_bottom")),
    ("scan_time", "f4", ("date", "scans")),
    ("theta_dpr", "f4", ("scans", "pixels")),
    ("latent_heat", "f4", ("scans", "pixels", "spectral_levels")),
]


def load_combined_data_bin(filename):
    """
    Load data from combined retrieval special run into a '.bin; file.

    Args:
        filename: Path pointing to the file to read.

    Return:
        An xarray.Dataset containing the data from the file.
    """
    filename = Path(filename)

    # Decompress file if compressed.
    if filename.suffix == ".gz":
        with TemporaryDirectory() as tmp:
            data = io.BytesIO()
            args = ["gunzip", "-c", str(filename)]
            with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
                data.write(proc.stdout.read())
            data.seek(0)
            data = data.read()
    else:
        data = open(filename, "rb").read()

    dataset = {}
    n_scans = np.frombuffer(data, "i", 1)[0]
    dimensions = copy(BIN_DIMENSIONS)
    dimensions["scans"] = n_scans
    offset = np.dtype("i").itemsize
    for name, dtype, dims in BIN_TYPES:
        shape = [dimensions[name] for name in dims]
        size = np.prod(shape)
        array = np.frombuffer(data, dtype=dtype, count=size, offset=offset)
        array = array.reshape(shape)
        dataset[name] = (dims, array)
        offset += size * np.dtype(dtype).itemsize

    # Ensure all data is read.
    assert offset == len(data)

    return xr.Dataset(dataset)


def calculate_smoothing_kernels(fwhm_a, fwhm_x):
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
    n_a = int(2.0 * fwhm_a / d_a)
    n_x = int(2.0 * fwhm_x / d_x)

    x_a = np.arange(-n_a, n_a + 1).reshape(-1, 1) * d_a
    x_x = np.arange(-n_x, n_x + 1).reshape(1, -1) * d_x
    radius = np.sqrt((2.0 * x_a / fwhm_a) ** 2 + (2.0 * x_x / fwhm_x) ** 2)
    kernel = np.exp(np.log(0.5) * radius ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_field(field, kernel):
    """
    Smooth field using a given convolution kernel.

    Args:
        field: A 2D or 3D array containing a variable to smooth along
            the first dimension.
        kernel: The smoothing kernel to use to smooth the field.

    Return:
        The smoothed field.
    """
    shape = kernel.shape + (1,) * (field.ndim - 2)
    kernel = kernel.reshape(shape)

    field_m = np.nan_to_num(field, nan=0.0)
    field_m[field < -1000] = 0.0
    field_s = convolve(field_m, kernel, mode="same")

    mask = (field > -1000).astype(np.float)
    weights = convolve(mask, kernel, mode="same")
    field_s /= weights
    field_s[weights < 1e-6] = np.nan
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
        if self.filename.suffix != ".HDF5":
            self.format = "BIN"
            time, granule = self.filename.stem.split(".")[2:4]
            self.start_time = datetime.strptime(time, "%Y%m%d")
            self.granule = int(granule)
        else:
            self.format = "HDF5"
            time, granule = self.filename.stem.split(".")[4:6]
            self.start_time = datetime.strptime(time[:-8], "%Y%m%d-S%H%M%S")
            self.granule = int(granule)


    def to_xarray_dataset(self, smooth=False, profiles=False, roi=None):
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
        if self.format == "BIN":
            data = load_combined_data_bin(self.filename)
            return data

        from h5py import File
        with File(str(self.filename), "r") as data:

            data = data["NS"]
            latitude = data["Latitude"][:]
            longitude = data["Longitude"][:]

            if roi is not None:
                lon_0, lat_0, lon_1, lat_1 = roi
                inside = (
                    (longitude >= lon_0)
                    * (latitude >= lat_0)
                    * (longitude < lon_1)
                    * (latitude < lat_1)
                )
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
                "second": data["ScanTime"]["Second"][i_start:i_end],
            }
            date = pd.to_datetime(date)

            surface_precip = data["surfPrecipTotRate"][i_start:i_end]
            if smooth:
                k = calculate_smoothing_kernels(18e3, 11e3)
                surface_precip = smooth_field(surface_precip, k)

            dataset = {
                "scan_time": (("scans",), date),
                "latitude": (("scans", "pixels"), latitude),
                "longitude": (("scans", "pixels"), longitude),
                "surface_precip": (("scans", "pixels"), surface_precip),
            }

            if profiles:
                twc = data["precipTotWaterCont"][i_start:i_end]
                l_frac = data["liqMassFracTrans"][i_start:i_end]
                l_frac[l_frac < 0] = 1.0
                levels = (np.arange(88) + 1).reshape(1, 1, -1)
                phases = data["phaseBinNodes"][i_start:i_end]
                top = np.expand_dims(phases[..., 1], 2)

                rwc = twc.copy()
                indices = levels < top
                rwc[indices] = 0.0
                indices = (levels >= top) * (levels < top + 10.0)

                lf_mask = (np.arange(10)).reshape(1, 1, -1) + top
                lf_mask = lf_mask <= 88
                rwc[indices] *= l_frac[lf_mask].ravel()
                rwc[twc < -1000] = np.nan

                swc = twc - rwc
                swc[twc < -1000] = np.nan

                if smooth:
                    swc = smooth_field(swc, k)
                    rwc = smooth_field(rwc, k)
                    twc = smooth_field(twc, k)

                dataset["layers"] = (("layers"), np.arange(0.125e3, 22e3, 0.25e3))
                dataset["rain_water_content"] = (
                    ("scans", "pixels", "layers"),
                    rwc[..., ::-1],
                )
                dataset["snow_water_content"] = (
                    ("scans", "pixels", "layers"),
                    swc[..., ::-1],
                )
                dataset["total_water_content"] = (
                    ("scans", "pixels", "layers"),
                    twc[..., ::-1],
                )
            return xr.Dataset(dataset)


class GPMLHFile:
    """
    Class to read in DPR spectral latent heating files.
    """
    def __init__(self, filename):
        """
        Create GPMCMB object to read a given file.

        Args:
            filename: Path pointing to the file to read.

        """
        self.filename = Path(filename)
        time = self.filename.stem.split(".")[4][:-8]
        self.start_time = datetime.strptime(time, "%Y%m%d-S%H%M%S")

    def to_xarray_dataset(self, smooth=False, roi=None):
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
        with xr.open_dataset(str(self.filename)) as data:
            latitude = data["Swath_Latitude"][:]
            longitude = data["Swath_Longitude"][:]

            if roi is not None:
                lon_0, lat_0, lon_1, lat_1 = roi
                inside = (
                    (longitude >= lon_0)
                    * (latitude >= lat_0)
                    * (longitude < lon_1)
                    * (latitude < lat_1)
                )
                inside = np.any(inside, axis=1)
                i_start, i_end = np.where(inside)[0][[0, -1]]
            else:
                i_start = 0
                i_end = latitude.shape[0]

            latitude = latitude[i_start:i_end]
            longitude = longitude[i_start:i_end]

            date = {
                "year": data["Swath_ScanTime_Year"].data[i_start:i_end],
                "month": data["Swath_ScanTime_Month"].data[i_start:i_end],
                "day": 1,
            }
            date = np.array(pd.to_datetime(date))
            date = date + data["Swath_ScanTime_DayOfMonth"].data[i_start:i_end]
            date = date + data["Swath_ScanTime_Hour"].data[i_start:i_end]
            date = date + data["Swath_ScanTime_Minute"].data[i_start:i_end]

            latent_heat = data["Swath_latentHeating"][i_start:i_end]
            if smooth:
                k = calculate_smoothing_kernels(18e3, 10e3)
                latent_heat = smooth_field(latent_heat, k)

            dataset = {
                "scan_time": (("scans",), date),
                "latitude": (("scans", "pixels"), latitude),
                "longitude": (("scans", "pixels"), longitude),
                "latent_heat": (("scans", "pixels", "levels"), latent_heat),
            }
            return xr.Dataset(dataset)
