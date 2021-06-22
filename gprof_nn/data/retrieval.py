"""
=======================
gprof_nn.data.retrieval
=======================

This module contains functions to read and write the binary retrieval output format
for GPROF 7.
"""
import logging
import gzip
from pathlib import Path

import numpy as np
from quantnn.normalizer import Normalizer
import torch
import xarray
import scipy as sp
import scipy.interpolate

from gprof_nn.definitions import MISSING

LOGGER = logging.getLogger(__name__)

N_SPECIES = 5
N_TEMPERATURES = 12
N_LAYERS = 28
N_PROFILES = 80

DATE6_TYPE = np.dtype(
    [
        ("year", "i2"),
        ("month", "i2"),
        ("day", "i2"),
        ("hour", "i2"),
        ("minute", "i2"),
        ("second", "i2"),
    ]
)

DATE7_TYPE = np.dtype(
    [
        ("year", "i2"),
        ("month", "i2"),
        ("day", "i2"),
        ("hour", "i2"),
        ("minute", "i2"),
        ("second", "i2"),
        ("millisecond", "i2"),
    ]
)

ORBIT_HEADER_TYPES = np.dtype(
    [
        ("satellite", "a12"),
        ("sensor", "a12"),
        ("preprocessor", "a12"),
        ("algorithm", "a12"),
        ("radiometer_file", "a128"),
        ("profile_database_file", "a128"),
        ("creation_date", DATE6_TYPE),
        ("granule_start_date", DATE6_TYPE),
        ("granule_end_date", DATE6_TYPE),
        ("granule_number", "i"),
        ("number_of_scans", "i2"),
        ("number_of_pixels", "i2"),
        ("profile_struct", "i1"),
        ("spares", "a51"),
    ]
)

PROFILE_INFO_TYPES = np.dtype(
    [
        ("n_species", "i1"),
        ("n_temps", "i1"),
        ("n_layers", "i1"),
        ("n_profiles", "i1"),
        ("species_description", f"{N_SPECIES}a20"),
        ("height_top_layers", f"{N_LAYERS}f4"),
        ("temperature", f"{N_TEMPERATURES}f4"),
        ("profiles", f"{N_SPECIES * N_TEMPERATURES * N_LAYERS * N_PROFILES}f4"),
    ]
)

SCAN_HEADER_TYPES = np.dtype(
    [
        ("scan_latitude", "f4"),
        ("scan_longitude", "f4"),
        ("scan_altitude", "f4"),
        ("scan_date", DATE7_TYPE),
        ("spare", "i2"),
    ]
)

DATA_RECORD_TYPES = np.dtype(
    [
        ("pixel_status", "i1"),
        ("quality_flag", "i1"),
        ("l1c_quality_flag", "i1"),
        ("surface_type", "i1"),
        ("total_column_water_vapor", "i1"),
        ("pop", "i1"),
        ("two_meter_temperature", "i2"),
        ("airmass_type", "i2"),
        ("sunglint_angle", "i1"),
        ("precip_flag", "i1"),
        ("latitude", "f4"),
        ("longitude", "f4"),
        ("surface_precip", "f4"),
        ("frozen_precip", "f4"),
        ("convective_precip", "f4"),
        ("rain_water_path", "f4"),
        ("cloud_water_path", "f4"),
        ("ice_water_path", "f4"),
        ("most_likely_precip", "f4"),
        ("precip_1st_tercile", "f4"),
        ("precip_3rd_tercile", "f4"),
        ("profile_t2m_index", "i2"),
        ("profile_index", f"{N_SPECIES}i2"),
        ("profile_scale", f"{N_SPECIES}f4"),
    ]
)

DATA_RECORD_TYPES_PROFILES = np.dtype(
    [
        ("pixel_status", "i1"),
        ("quality_flag", "i1"),
        ("l1c_quality_flag", "i1"),
        ("surface_type", "i1"),
        ("total_column_water_vapor", "i1"),
        ("pop", "i1"),
        ("two_meter_temperature", "i2"),
        ("airmass_type", "i2"),
        ("sunglint_angle", "i1"),
        ("precip_flag", "i1"),
        ("latitude", "f4"),
        ("longitude", "f4"),
        ("surface_precip", "f4"),
        ("frozen_precip", "f4"),
        ("convective_precip", "f4"),
        ("rain_water_path", "f4"),
        ("cloud_water_path", "f4"),
        ("ice_water_path", "f4"),
        ("most_likely_precip", "f4"),
        ("precip_1st_tercile", "f4"),
        ("precip_3rd_tercile", "f4"),
        ("profile_t2m_index", "i2"),
        ("profile_index", f"{N_SPECIES}i2"),
        ("profile_scale", f"{N_SPECIES}f4"),
        ("profiles", f"{4 * N_LAYERS}f4"),
    ]
)

DATA_RECORD_TYPES_SENSITIVITY = np.dtype(
    [
        ("pixel_status", "i1"),
        ("quality_flag", "i1"),
        ("l1c_quality_flag", "i1"),
        ("surface_type", "i1"),
        ("total_column_water_vapor", "i1"),
        ("pop", "i1"),
        ("two_meter_temperature", "i2"),
        ("airmass_type", "i2"),
        ("sun_glint_angle", "i1"),
        ("precip_flag", "i1"),
        ("latitude", "f4"),
        ("longitude", "f4"),
        ("surface_precip", "f4"),
        ("surface_precip_1", "f4"),
        ("surface_precip_2", "f4"),
        ("dsurface_precip_dy", "15f4"),
        ("frozen_precip", "f4"),
        ("convective_precip", "f4"),
        ("rain_water_path", "f4"),
        ("cloud_water_path", "f4"),
        ("ice_water_path", "f4"),
        ("most_likely_precip", "f4"),
        ("precip_1st_tercile", "f4"),
        ("precip_3rd_tercile", "f4"),
        ("profile_t2m_index", "i2"),
        ("profile_index", f"{N_SPECIES}i2"),
        ("profile_scale", f"{N_SPECIES}f4")
    ]
)


class RetrievalFile:
    """
    Class to read binary retrieval results from the GPROF 7 algorithm.
    """

    def __init__(self,
                 filename,
                 has_sensitivity=False,
                 has_profiles=False):
        """
        Read retrieval results.

        Args:
            filename: The path to the file to open.
            has_sensitivity: Flag indicating whether the file contains
                gradients of the surface precip retrieval. This special
                is only produced by a customized version of the GPROF
                retrieval.
        """
        if has_sensitivity and has_profiles:
            raise ValueError(
                "Retrieval file can only include either sensitivity data or "
                "profile data. Not both."
            )
        filename = Path(filename)
        self.filename = filename
        if filename.suffix == ".gz":
            with gzip.open(filename, "rb") as file:
                self.data = file.read()
        else:
            with open(filename, "rb") as file:
                self.data = file.read()
        self.orbit_header = np.frombuffer(self.data,
                                          ORBIT_HEADER_TYPES,
                                          count=1)
        self.profile_info = np.frombuffer(self.data,
                                          PROFILE_INFO_TYPES,
                                          count=1,
                                          offset=ORBIT_HEADER_TYPES.itemsize)
        self.n_scans = self.orbit_header["number_of_scans"][0]
        self.n_pixels = self.orbit_header["number_of_pixels"][0]

        np.random.seed(self.orbit_header["granule_number"])
        self.scan_indices = np.random.permutation(np.arange(self.n_scans))
        self.pixel_indices = np.random.permutation(np.arange(self.n_pixels))

        if has_sensitivity:
            self.data_record_types = DATA_RECORD_TYPES_SENSITIVITY
        elif has_profiles:
            self.data_record_types = DATA_RECORD_TYPES_PROFILES
        else:
            self.data_record_types = DATA_RECORD_TYPES

    @property
    def satellite(self):
        """The satellite platform from which the input observations stem. """
        return self.orbit_header["satellite"]

    @property
    def sensor(self):
        """The sensor from which the input observations stem. """
        return self.orbit_header["sensor"]

    @property
    def scans(self):
        """
        Iterates of the scans in the file. Each scan is returned as Numpy
        structured array of size n_pixels and dtype DATA_RECORD_TYPES.
        """
        for i in range(self.n_scans):
            yield self.get_scan(i)

    def get_scan(self, i):
        """
        Return scan as Numpy structured array of size n_pixels and dtype
        DATA_RECORD_TYPES.
        """
        offset = ORBIT_HEADER_TYPES.itemsize + PROFILE_INFO_TYPES.itemsize
        offset += i * (
            SCAN_HEADER_TYPES.itemsize + self.n_pixels * self.data_record_types.itemsize
        )
        offset += SCAN_HEADER_TYPES.itemsize
        return np.frombuffer(
            self.data, self.data_record_types, count=self.n_pixels, offset=offset
        )

    def get_scan_header(self, i):
        """
        Args:
            i: The index of the scan to return.

        Returns:
            The header of the ith scan in the file as numpy structured array
            of size n_pixels and dtype DATA_RECORD_TYPES.
        """
        if i < 0:
            i = self.n_scans + i

        offset = ORBIT_HEADER_TYPES.itemsize + PROFILE_INFO_TYPES.itemsize
        offset += i * (
            SCAN_HEADER_TYPES.itemsize + self.n_pixels * self.data_record_types.itemsize
        )
        return np.frombuffer(self.data, SCAN_HEADER_TYPES, count=1, offset=offset)

    def to_xarray_dataset(self):
        """
        Return retrieval results as xarray dataset.
        """
        data = {}
        for s in self.scans:
            for k in self.data_record_types.fields:
                data.setdefault(k, []).append(s[k])

        for k in data:
            data[k] = np.stack(data[k], axis=0)

        if "profile_scale" in data:
            shape = (N_SPECIES, N_TEMPERATURES, N_LAYERS, N_PROFILES)
            profiles = self.profile_info["profiles"].reshape(shape, order="f")

            profile_indices = data["profile_index"]
            temperature_indices = data["profile_t2m_index"]
            factors = data["profile_scale"]

            invalid = (profile_indices <= 0)
            profile_indices[invalid] = 1
            temperature_indices[np.all(invalid, axis=-1)] = 1

            rwc = (
                profiles[0, temperature_indices - 1, :, profile_indices[..., 0] - 1]
                * factors[..., np.newaxis, 0]
            )
            rwc[invalid[..., 0]] = MISSING
            cwc = (
                profiles[1, temperature_indices - 1, :, profile_indices[..., 1] - 1]
                * factors[..., np.newaxis, 1]
            )
            cwc[invalid[..., 1]] = MISSING
            swc = (
                profiles[2, temperature_indices - 1, :, profile_indices[..., 2] - 1]
                * factors[..., np.newaxis, 2]
            )
            swc[invalid[..., 2]] = MISSING
            lh = (
                profiles[4, temperature_indices - 1, :, profile_indices[..., 4] - 1]
                * factors[..., np.newaxis, 4]
            )
            lh[invalid[..., 4]] = MISSING
            dataset = {
                "rain_water_content": (("scans", "pixels", "levels"), rwc),
                "cloud_water_content": (("scans", "pixels", "levels"), cwc),
                "snow_water_content": (("scans", "pixels", "levels"), swc),
                "latent_heat": (("scans", "pixels", "levels"), lh)
            }
        else:
            dataset = {}

        dims = ["scans", "pixels", "channels", "levels"]
        for k, d in data.items():
            if "profile" not in k:
                dataset[k] = (dims[: len(d.shape)], d)

        return xarray.Dataset(dataset)


PREPROCESSOR = "preprocessor"
TRAINING_DATA = "training_data"

class RetrievalDriver:
    """
    Helper class that implements the logic to run the GPROF-NN retrieval.
    """
    N_CHANNELS = 15

    def __init__(self,
                 input_file,
                 normalizer,
                 model,
                 ancillary_data=None,
                 output_file=None):
        """
        Args:
            input_file: Path to the file containing the input data for the
                 retrieval.
            normalizer_file: Path to the file containing the normalizer.
            model: The neural network to use for the retrieval.



        """
        from gprof_nn.data.preprocessor import PreprocessorFile
        input_file = Path(input_file)
        self.input_file = input_file
        self.normalizer = normalizer
        self.model = model
        self.ancillary_data = ancillary_data

        suffix = input_file.suffix
        if suffix.endswith("pp"):
            self.format = PREPROCESSOR
        else:
            self.format = TRAINING_DATA

        if self.format == PREPROCESSOR:
            self.input_data = model.preprocessor_class(input_file,
                                                       self.normalizer)
        else:
            self.input_data = model.netcdf_class(
                input_file,
                normalizer=normalizer,
                shuffle=False,
                augment=False
            )

        output_suffix = ".BIN"
        if self.format == TRAINING_DATA:
            output_suffix = ".nc"

        if output_file is None:
            self.output_file = Path(
                self.input_file.name.replace(suffix, output_suffix)
            )
        elif Path(output_file).is_dir():
            self.output_file = (
                Path(output_file) /
                self.input_file.name.replace(suffix, output_suffix)
                )
        else:
            self.output_file = output_file

    def run_retrieval(self):
        """
        Run retrieval and store results to file.

        Return:
            Name of the output file that the results have been written
            to.
        """
        results = self.input_data.run_retrieval(self.model)
        if self.format == PREPROCESSOR:
            return self.input_data.write_retrieval_results(
                self.output_file.parent,
                results,
                ancillary_data=self.ancillary_data
            )
        else:
            results.to_netcdf(self.output_file)
            return self.output_file


def calculate_frozen_precip(
        wet_bulb_temperature,
        surface_type,
        surface_precip
):
    """
    Calculate amount of frozen precipitation based on wet-bulb
    temperature lookup table.

    Args:
        wet_bulb_temperature: The wet bulb temperature in K.
        surface_type: The surface type for each observation.
        surface_precip: The total amount of surface precipitation.

    Returns:
        Array of same shape as 'surface_precip' containing the corresponding,
        estimated amount of frozen precipitation.
    """
    t = np.clip(wet_bulb_temperature,
                TWB_TABLE[0, 0] + 273.15,
                TWB_TABLE[-1, 0] + 273.15)
    f_ocean = TWB_INTERP_OCEAN(t)
    f_land = TWB_INTERP_LAND(t)

    ocean_pixels = (surface_type == 1)
    f = 1.0 - np.where(ocean_pixels, f_ocean, f_land) / 100.0
    return f * surface_precip


TWB_TABLE = np.array([
    [-6.5,    0.00,    0.00],
    [-6.4,    0.10,    0.30],
    [-6.3,    0.20,    0.60],
    [-6.2,    0.30,    0.90],
    [-6.1,    0.40,    1.20],
    [-6.0,    0.50,    1.50],
    [-5.9,    0.60,    1.80],
    [-5.8,    0.70,    2.10],
    [-5.7,    0.80,    2.40],
    [-5.6,    0.90,    2.70],
    [-5.5,    1.00,    3.00],
    [-5.4,    1.05,    3.10],
    [-5.3,    1.10,    3.20],
    [-5.2,    1.15,    3.30],
    [-5.1,    1.20,    3.40],
    [-5.0,    1.25,    3.50],
    [-4.9,    1.30,    3.60],
    [-4.8,    1.35,    3.70],
    [-4.7,    1.40,    3.80],
    [-4.6,    1.45,    3.90],
    [-4.5,    1.50,    4.00],
    [-4.4,    1.60,    4.10],
    [-4.3,    1.70,    4.20],
    [-4.2,    1.80,    4.30],
    [-4.1,    1.90,    4.40],
    [-4.0,    2.00,    4.50],
    [-3.9,    2.10,    4.60],
    [-3.8,    2.20,    4.70],
    [-3.7,    2.30,    4.80],
    [-3.6,    2.40,    4.90],
    [-3.5,    2.50,    5.00],
    [-3.4,    2.55,    5.20],
    [-3.3,    2.60,    5.40],
    [-3.2,    2.65,    5.60],
    [-3.1,    2.70,    5.80],
    [-3.0,    2.75,    6.00],
    [-2.9,    2.80,    6.20],
    [-2.8,    2.85,    6.40],
    [-2.7,    2.90,    6.60],
    [-2.6,    2.95,    6.80],
    [-2.5,    3.00,    7.00],
    [-2.4,    3.10,    7.10],
    [-2.3,    3.20,    7.20],
    [-2.2,    3.30,    7.30],
    [-2.1,    3.40,    7.40],
    [-2.0,    3.50,    7.50],
    [-1.9,    3.60,    7.60],
    [-1.8,    3.70,    7.70],
    [-1.7,    3.80,    7.80],
    [-1.6,    3.90,    7.90],
    [-1.5,    4.00,    8.00],
    [-1.4,    4.10,    8.20],
    [-1.3,    4.20,    8.40],
    [-1.2,    4.30,    8.60],
    [-1.1,    4.40,    8.80],
    [-1.0,    4.50,    9.00],
    [-0.9,    4.60,    9.20],
    [-0.8,    4.70,    9.40],
    [-0.7,    4.80,    9.60],
    [-0.6,    4.90,    9.80],
    [-0.5,    5.00,   10.00],
    [-0.4,    6.60,   11.60],
    [-0.3,    8.20,   13.20],
    [-0.2,    9.80,   14.80],
    [-0.1,   11.40,   16.40],
    [0.0,   13.00,   18.00],
    [0.1,   14.60,   19.60],
    [0.2,   16.20,   21.20],
    [0.3,   17.80,   22.80],
    [0.4,   19.40,   24.40],
    [0.5,   21.00,   26.00],
    [0.6,   25.80,   29.00],
    [0.7,   30.60,   32.00],
    [0.8,   35.40,   35.00],
    [0.9,   40.20,   38.00],
    [1.0,   45.00,   41.00],
    [1.1,   49.80,   44.00],
    [1.2,   54.60,   47.00],
    [1.3,   59.40,   50.00],
    [1.4,   64.20,   53.00],
    [1.5,   69.00,   56.00],
    [1.6,   71.30,   57.90],
    [1.7,   73.60,   59.80],
    [1.8,   75.90,   61.70],
    [1.9,   78.20,   63.60],
    [2.0,   80.50,   65.50],
    [2.1,   82.80,   67.40],
    [2.2,   85.10,   69.30],
    [2.3,   87.40,   71.20],
    [2.4,   89.70,   73.10],
    [2.5,   92.00,   75.00],
    [2.6,   92.55,   76.30],
    [2.7,   93.10,   77.60],
    [2.8,   93.65,   78.90],
    [2.9,   94.20,   80.20],
    [3.0,   94.75,   81.50],
    [3.1,   95.30,   82.80],
    [3.2,   95.85,   84.10],
    [3.3,   96.40,   85.40],
    [3.4,   96.95,   86.70],
    [3.5,   97.50,   88.00],
    [3.6,   97.60,   88.70],
    [3.7,   97.70,   89.40],
    [3.8,   97.80,   90.10],
    [3.9,   97.90,   90.80],
    [4.0,   98.00,   91.50],
    [4.1,   98.10,   92.20],
    [4.2,   98.20,   92.90],
    [4.3,   98.30,   93.60],
    [4.4,   98.40,   94.30],
    [4.5,   98.50,   95.00],
    [4.6,   98.55,   95.25],
    [4.7,   98.60,   95.50],
    [4.8,   98.65,   95.75],
    [4.9,   98.70,   96.00],
    [5.0,   98.75,   96.25],
    [5.1,   98.80,   96.50],
    [5.2,   98.85,   96.75],
    [5.3,   98.90,   97.00],
    [5.4,   98.95,   97.25],
    [5.5,   99.00,   97.50],
    [5.6,   99.10,   97.75],
    [5.7,   99.20,   98.00],
    [5.8,   99.30,   98.25],
    [5.9,   99.40,   98.50],
    [6.0,   99.50,   98.75],
    [6.1,   99.60,   99.00],
    [6.2,   99.70,   99.25],
    [6.3,   99.80,   99.50],
    [6.4,   99.90,   99.75],
    [6.5,  100.00,  100.00]
])


TWB_INTERP_LAND = sp.interpolate.interp1d(
    TWB_TABLE[:, 0] + 273.15,
    TWB_TABLE[:, 1],
    assume_sorted=True,
    kind="linear"
)


TWB_INTERP_OCEAN = sp.interpolate.interp1d(
    TWB_TABLE[:, 0] + 273.15,
    TWB_TABLE[:, 2],
    assume_sorted=True,
    kind="linear"
)

