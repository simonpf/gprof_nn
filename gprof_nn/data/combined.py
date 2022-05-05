"""
======================
gprof_nn.data.combined
======================

Interface to read in L2b files of the GPM combined product.
"""
from copy import copy
from datetime import datetime
import io
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from concurrent import futures

import numpy as np
import pandas as pd
from pyresample import geometry, kd_tree
from rich.progress import Progress
import scipy
from scipy.signal import convolve
import xarray as xr

from gprof_nn.definitions import ALL_TARGETS, PROFILE_NAMES, DATABASE_MONTHS
from gprof_nn.data.utils import compressed_pixel_range
from gprof_nn.data.l1c import L1CFile
from gprof_nn.logging import get_console
import gprof_nn.logging
from gprof_nn import sensors


LOGGER = logging.getLogger(__name__)


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


def load_combined_data_special(
        filename,
        smooth=False,
        profiles=True,
        slh_path=None,
        roi=None
):
    """
    Load data from specal combined run for generation of GPROF database.

    Args:
        filename: Path pointing to the file to read.
        profiles: Wether or not to include profiles in the output.
        slh_path: If provided the function will look into the given
            path to load latent heating data from the 2A_SLH product.
        roi: Optional bounding box given as list
                ``[lon_0, lat_0, lon_1, lat_1]`` specifying the longitude
                and latitude coordinates of the lower left
                (``lon_0, lat_0``) and upper right (``lon_1, lat_1``)
                corners. If given, only scans containing at least one pixel
                within the given bounding box will be returned.

    Return:
        An xarray.Dataset containing the data from the file.
    """
    from h5py import File
    filename = Path(filename)

    # Decompress file if compressed.
    if filename.suffix == ".gz":
        with TemporaryDirectory() as tmp:
            data = io.BytesIO()
            args = ["gunzip", "-c", str(filename)]
            with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
                data.write(proc.stdout.read())
            data.seek(0)
            data = File(data, "r")
    else:
        data = File(filename, "r")

    data = data["KuKaGMI"]

    surface_fields = {
        "latitude": "Latitude",
        "longitude": "Longitude",
        "surface_precip": "nearSurfPrecipTotRate",
    }

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
    scans = slice(i_start, i_end)

    dataset = {}
    for name, var in surface_fields.items():
        array = data[var][scans]
        if name == "surface_precip" and smooth:
            k = calculate_smoothing_kernels(18e3, 11e3)
            array = smooth_field(array, k)
        dataset[name] = (("scans", "pixels"), array)

    dataset = xr.Dataset(dataset)

    if profiles:
        total_water = data["precipTotWaterCont"][scans][..., ::-1]
        total_water[total_water < 0] = np.nan

        liquid_water = data["precipLiqWaterCont"][scans][..., ::-1]
        liquid_water[liquid_water < 0] = np.nan

        ice_water = total_water - liquid_water

        cloud_water = data["cloudLiqWaterCont"][scans][..., ::-1]
        cloud_water[cloud_water < 0] = np.nan

        cloud_water = cloud_water[..., :80]
        liquid_water = liquid_water[..., :80]
        ice_water = ice_water[..., :80]

        dims = ("scans", "pixels", "levels")
        dataset["rain_water_content"] = (dims, liquid_water)
        dataset["snow_water_content"] = (dims, ice_water)
        dataset["cloud_water_content"] = (dims, cloud_water)

        if slh_path is not None:
            year = data["ScanTime/Year"][0]
            month = data["ScanTime/Month"][0]
            day = data["ScanTime/DayOfMonth"][0]
            hour = data["ScanTime/Hour"][0]
            minute = data["ScanTime/Minute"][0]

            path = (
                Path(slh_path) /
                f"{year % 2000:02}{month:02}" /
                f"{year % 2000:02}{month:02}{day:02}"
            )
            files = list(path.glob(f"*S{hour:02}{minute:02}*.HDF5"))
            if len(files) == 0:
                LOGGER.warning(
                    "Did not find SLH file for start time %s-%s-%s %s:%s",
                    year, month, day, hour, minute
                )
                slh = np.zeros_like(liquid_water)
                slh[:] = np.nan
            else:
                slh_data = File(files[0], "r")
                slh = slh_data["Swath/latentHeating"][scans]
            slh[slh < -999] = np.nan

            dataset["latent_heat"] = (dims, slh)

        # Fill to surface
        dataset = dataset.bfill("levels")
        dataset = 0.5 * (
            dataset[{"levels": slice(0, 80, 2)}] +
            dataset[{"levels": slice(1, 80, 2)}]
        )

        if smooth:
            profile_names = [
                "rain_water_content",
                "snow_water_content",
                "cloud_water_content",
                "latent_heat"
            ]
            for var in profile_names:
                if var in dataset:
                    array = dataset[var].data
                    dataset[var].data = smooth_field(array, k)

    levels = np.linspace(0, 20e3, 41)
    levels = 0.5 * (levels[1:] + levels[:-1])
    dataset["levels"] = (("levels",), levels)

    return dataset


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
    @staticmethod
    def find_file(path, granule, date=None):
        """
        Find GPM combined file by granule number.

        Args:
            path: The root of the directory tree containing the observation
                files.
            granule: The granule number as integer,
            data: The data of the granule.

        Return:
            Path object pointing to the GPM CMB file.
        """
        pattern = f"2B.GPM.DPRGMI.*.{granule:06}.*.HDF5"
        if date:
            year = date.year
            month = date.month
            day = date.day
            pattern = (f"{(year - 2000):02}{month:02}" +
                       f"/{(year - 2000):02}{month:02}{day:02}/" +
                       pattern)
        else:
            pattern = "**/" + pattern

        files = list(Path(path).glob(pattern))
        if len(files) == 0:
            raise ValueError(
                f"Couldn't find a combind file for granule {granule}"
                f"at the given path."
            )
        return files[0]

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
            time, granule, format = self.filename.stem.split(".")[4:7]
            self.format = format
            self.start_time = datetime.strptime(time[:-8], "%Y%m%d-S%H%M%S")
            self.granule = int(granule)


    def to_xarray_dataset(
            self,
            smooth=False,
            profiles=False,
            roi=None,
            slh_path=None
    ):
        """
        Load data in file into 'xarray.Dataset'.

        Args:
            smooth: If set to true the 'surface_precip' field will be smoothed
                to match the footprint of the GMI 23.8 GHz channels.
            profiles: Whether or not to also load profile variables.
            roi: Optional bounding box given as list
                 ``[lon_0, lat_0, lon_1, lat_1]`` specifying the longitude
                 and latitude coordinates of the lower left
                 (``lon_0, lat_0``) and upper right (``lon_1, lat_1``)
                 corners. If given, only scans containing at least one pixel
                 within the given bounding box will be returned.
            slh_path: Path to load spectral latent heating data from. Only
                 used for ITE768 format.
        """
        if self.format == "BIN":
            data = load_combined_data_bin(self.filename)
            return data
        elif self.format == "ITE768":
            data = load_combined_data_special(
                self.filename,
                smooth=smooth,
                profiles=profiles,
                slh_path=slh_path,
                roi=roi
            )
            return data

        from h5py import File
        with File(str(self.filename), "r") as data:

            if self.format.startswith("ITE"):
                data = data["KuKaGMI"]
                sp_var = "estimSurfPrecipTotRate"
            else:
                data = data["NS"]
                sp_var = "surfPrecipTotRate"

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

            surface_precip = data[sp_var][i_start:i_end]
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

                if self.format.startswith("ITE"):
                    l_frac = data["precipLiqRate"][i_start:i_end]
                else:
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


    def match_targets(self, input_data, targets=None, slh_path=None):
        """
        Match retrieval targets from combined file to points in
        xarray dataset.

        Args:
            input_data: xarray dataset containing the input data from
                the preprocessor.
            targets: List of retrieval target variables to extract from
                the sim file.
        Return:
            The input dataset but with the requested retrieval targets added.
        """
        if targets is None:
            targets = ALL_TARGETS

        n_scans = input_data.scans.size
        n_pixels = 221
        w_c = 40
        i_c = 110
        ix_start = i_c - w_c // 2
        ix_end = i_c + 1 + w_c // 2
        i_left, i_right = compressed_pixel_range()

        lats_1c = input_data["latitude"][:, ix_start:ix_end]
        lons_1c = input_data["longitude"][:, ix_start:ix_end] + 360

        m, n = lats_1c.shape
        lats3 = np.zeros((3 * m - 2, n), dtype=np.float32)
        lons3 = np.zeros((3 * m - 2, n), dtype=np.float32)

        lats_l = lats_1c[:-1]
        lats_r = lats_1c[1:]
        lats3[::3, :] = lats_1c
        lats3[1::3, :] = 2 / 3 * lats_l + 1 / 3 * lats_r
        lats3[2::3, :] = 1 / 3 * lats_l + 2 / 3 * lats_r

        lons_l = lons_1c[:-1]
        lons_r = lons_1c[1:]
        lons3[::3, :] = lons_1c
        lons3[1::3, :] = 2 / 3 * lons_l + 1 / 3 * lons_r
        lons3[2::3, :] = 1 / 3 * lons_l + 2 / 3 * lons_r
        lons3 -= 360

        data = self.to_xarray_dataset(profiles=True, slh_path=slh_path)

        lats = data["latitude"].data
        lons = data["longitude"].data
        levels = data["levels"].data
        input_data["levels"] = (("levels"), levels)

        # Prepare nearest neighbor interpolation.
        input_grid = geometry.SwathDefinition(lats=lats, lons=lons)
        output_grid = geometry.SwathDefinition(lats=lats3, lons=lons3)
        resampling_info = kd_tree.get_neighbour_info(
            input_grid,
            output_grid,
            4.0e3,
            neighbours=1
        )
        valid_inputs, valid_outputs, indices, distances = resampling_info

        # Extract matching data
        for target in targets:

            if not target in data:
                continue

            resampled = kd_tree.get_sample_from_neighbour_info(
                'nn', output_grid.shape, data[target].data,
                valid_inputs, valid_outputs, indices,
                fill_value=np.nan
            )

            if target in PROFILE_NAMES:
                input_data[target] = (
                    ("scans_3", "pixels_3", "levels"),
                    resampled.astype(np.float32),
                )
                if "content" in target:
                    path = np.trapz(resampled, x=levels, axis=-1) * 1e-3
                    path_name = target.replace("content", "path").replace("snow", "ice")
                    input_data[path_name] = (("scans_3", "pixels_3"), path)
            else:
                if target in ["surface_precip", "convective_precip"]:
                    dims = ("scans_3", "pixels_3")
                    input_data[target] = (dims, resampled.astype(np.float32))
                else:
                    input_data[target] = (
                        ("scans_3", "pixels_3"),
                        resampled[:, i_left:i_right].astype(np.float32),
                    )
        return input_data


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


###############################################################################
# Extraction of training data from combined.
###############################################################################


def _extract_scenes(data):
    """
    Extract 221 x 221 pixel wide scenes from dataset where
    ground truth surface precipitation rain rates are
    available.

    Args:
        xarray.Dataset containing the data from the preprocessor together
        with the matches surface precipitation from the .sim file.

    Return:
        New xarray.Dataset which containing 128x128 patches of input data
        and corresponding surface precipitation.
    """
    n = 221
    surface_precip = data["surface_precip"].data

    if np.all(np.isnan(surface_precip)):
        return None

    i_start = 0
    i_end = data.scans.size

    scenes = []
    while i_start + n < i_end:
        subscene = data[{
            "scans": slice(i_start, i_start + n),
            "scans_3": slice(3 * i_start, 3 * (i_start + n))
        }]
        surface_precip = subscene["surface_precip"].data
        if np.isfinite(surface_precip).sum() > 50:
            scenes.append(subscene)
            i_start += n
        else:
            i_start += n // 2

    if scenes:
        return xr.concat(scenes, "samples")
    return None


def process_l1c_file(
        l1c_filename,
        cmb_path,
        slh_path,
        log_queue=None):
    """
    Match L1C files with ERA5 surface and convective precipitation for
    sea-ice and sea-ice-edge surfaces.

    Args:
        l1c_filename: Path to a L1C file which to match with ERA5 precip.
        sensor: Sensor class defining the sensor for which to process the
            L1C files.
        era5_path: Root of the directory tree containing the ERA5 data.
    """
    import gprof_nn.logging
    if log_queue is not None:
        gprof_nn.logging.configure_queue_logging(log_queue)
    LOGGER.info("Starting processing L1C file %s.", l1c_filename)

    l1c_file = L1CFile(l1c_filename)
    l1c_data = l1c_file.to_xarray_dataset()
    granule = l1c_file.granule

    cmb_filename = GPMCMBFile.find_file(cmb_path, granule)
    cmb_file = GPMCMBFile(cmb_filename)
    data = cmb_file.match_targets(l1c_data, slh_path=slh_path)

    scenes = _extract_scenes(data)
    return scenes


class CombinedFileProcessor:
    """
    Processor class that manages the extraction of GPROF training data. A
    single processor instance processes all *.sim, MRMRS matchup and L1C
    files for a given day from each month of the database period.
    """

    def __init__(
            self,
            output_file,
            combined_path,
            slh_path,
            n_workers=4,
            day=None,
    ):
        """
        Create retrieval driver.

        Args:
            output_file: The file in which to store the extracted data.
            sensor: Sensor object defining the sensor for which to extract
                training data.
            era_5_path: Path to the root of the directory tree containing
                ERA5 data.
            n_workers: The number of worker processes to use.
            day: Day of the month for which to extract the data.
        """
        self.output_file = output_file
        self.combined_path = combined_path
        self.slh_path = slh_path
        self.pool = futures.ProcessPoolExecutor(max_workers=n_workers)

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
        l1c_file_path = sensors.GMI.l1c_file_path
        l1c_files = []
        for year, month in DATABASE_MONTHS:
            try:
                date = datetime(year, month, self.day)
                l1c_files += L1CFile.find_files(
                    date,
                    l1c_file_path,
                    sensor=sensors.GMI
                )
            except ValueError:
                pass

        # If no L1C files are found use GMI co-locations.
        if len(l1c_files) < 1:
            for year, month in DATABASE_MONTHS:
                try:
                    date = datetime(year, month, self.day)
                    l1c_file_path = sensors.GMI.l1c_file_path
                    l1c_files += L1CFile.find_files(
                        date, l1c_file_path, sensor=sensors.GMI
                    )
                except ValueError:
                    pass
        l1c_files = [f.filename for f in l1c_files]
        l1c_files = np.random.permutation(l1c_files)

        n_l1c_files = len(l1c_files)
        LOGGER.debug("Found %s L1C files.", n_l1c_files)

        # Submit tasks interleaving .sim and MRMS files.
        log_queue = gprof_nn.logging.get_log_queue()
        tasks = []
        for l1c_file in l1c_files:
            tasks.append(
                self.pool.submit(
                    process_l1c_file,
                    l1c_file,
                    self.combined_path,
                    self.slh_path,
                    log_queue=log_queue,
                )
            )

        datasets = []
        output_path = Path(self.output_file).parent
        output_file = Path(self.output_file).stem

        # Retrieve extracted observations and concatenate into
        # single dataset.

        n_tasks = len(tasks)
        n_chunks = 32
        chunk = 1

        with Progress(console=get_console()) as progress:
            gprof_nn.logging.set_log_level("INFO")
            pbar = progress.add_task("Extracting data:", total=len(tasks))
            for task in tasks:
                # Log messages from processes.
                task_done = False
                dataset = None
                while not task_done:
                    try:
                        gprof_nn.logging.log_messages()
                        dataset = task.result(timeout=1)
                        task_done = True
                    except futures.TimeoutError:
                        pass
                    except Exception as exc:
                        LOGGER.warning(
                            "The following error was encountered while "
                            "collecting results: %s",
                            exc,
                        )
                        get_console().print_exception()
                        task_done = True
                progress.advance(pbar)

                if dataset is not None:
                    datasets.append(dataset)
                    if len(datasets) > n_tasks // n_chunks:
                        dataset = xr.concat(datasets, "samples")
                        filename = output_path / (output_file + f"_{chunk:02}.nc")
                        dataset.attrs["sensor"] = sensors.GMI.name
                        dataset.to_netcdf(filename)
                        subprocess.run(["lz4", "-f", "--rm", filename], check=True)
                        LOGGER.info("Finished writing file: %s", filename)
                        datasets = []
                        chunk += 1

        # Store dataset with sensor name as attribute.
        if len(datasets) > 0:
            dataset = xr.concat(datasets, "samples")
            filename = output_path / (output_file + f"_{chunk:02}.nc")
            dataset.attrs["sensor"] =  sensors.GMI.name
            LOGGER.info("Writing file: %s", filename)
            dataset.to_netcdf(filename)
            subprocess.run(["lz4", "-f", "--rm", filename], check=True)

        # Explicit clean up to avoid memory leak.
        del datasets
        del dataset
