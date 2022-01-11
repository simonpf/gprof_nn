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
import xarray

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
        ("precip_2nd_tercile", "f4"),
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
        ("precip_2nd_tercile", "f4"),
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
        ("precip_2nd_tercile", "f4"),
        ("profile_t2m_index", "i2"),
        ("profile_index", f"{N_SPECIES}i2"),
        ("profile_scale", f"{N_SPECIES}f4"),
    ]
)


class RetrievalFile:
    """
    Class to read binary retrieval results from the GPROF 7 algorithm.
    """

    def __init__(self, filename, has_sensitivity=False, has_profiles=False):
        """
        Read retrieval results.

        Args:
            filename: The path to the file to open.
            has_sensitivity: Flag indicating whether the file contains
                gradients of the surface precip retrieval. This special
                is only produced by a customized version of the GPROF
                retrieval.
            has_profiles: Flag indicating whether the file contains full
                profile output.
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
        self.orbit_header = np.frombuffer(self.data, ORBIT_HEADER_TYPES, count=1)
        self.profile_info = np.frombuffer(
            self.data, PROFILE_INFO_TYPES, count=1, offset=ORBIT_HEADER_TYPES.itemsize
        )
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
        """The satellite platform from which the input observations stem."""
        return self.orbit_header["satellite"]

    @property
    def sensor(self):
        """The sensor from which the input observations stem."""
        return self.orbit_header["sensor"]

    @property
    def scans(self):
        """
        Iterates of the scans in the file. Each scan is returned as Numpy
        structured array of size n_pixels and dtype DATA_RECORD_TYPES.
        """
        for i in range(self.n_scans):
            yield self.get_scan(i)

    def __repr__(self):
        return (
            f"RetrievalFile(filename={self.filename}, "
            f"satellite={self.satellite}, sensor={self.sensor})"
        )

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

    def write(self, filename, n_scans=None):
        """
        Write the data in this retrieval file to another file.

        Args:
            filename: Name of the file to which write the content of this
                file.
            n_scans: Limit of the number of scans in the file to write.
        """
        if n_scans is None:
            n_scans = self.n_scans
        n_scans = min(self.n_scans, n_scans)
        with open(filename, "wb") as output:
            orbit_header = self.orbit_header.copy()
            orbit_header["number_of_scans"][:] = n_scans

            # Write orbit header.
            orbit_header.tofile(output)

            self.profile_info.tofile(output)

            for i in range(n_scans):
                self.get_scan_header(i).tofile(output)
                self.get_scan(i).tofile(output)

    def to_xarray_dataset(self, full_profiles=True):
        """
        Return retrieval results as xarray dataset.

        Args:
            full_profiles: If 'True' the full profile variables will be
                read in when available. If 'False' the compressed profiles
                will be loaded even if full profiles are available.

        Return:
            'xarray.Dataset' containing all variables in the retrieval file.
        """
        data = {}
        for scan in self.scans:
            for key in self.data_record_types.fields:
                data.setdefault(key, []).append(scan[key])

        for k in data:
            data[k] = np.stack(data[k], axis=0)

        if "profiles" in data and full_profiles:
            species = [
                "rain_water_content",
                "cloud_water_content",
                "snow_water_content",
                "latent_heat",
            ]
            dataset = {}
            dims = ("scans", "pixels", "layers")
            for i, spec in enumerate(species):
                i_start = i * 28
                i_end = i_start + 28
                dataset[spec] = (dims, data["profiles"][..., i_start:i_end])

        elif "profile_scale" in data:
            shape = (N_SPECIES, N_TEMPERATURES, N_LAYERS, N_PROFILES)
            profiles = self.profile_info["profiles"].reshape(shape, order="f")

            profile_indices = data["profile_index"]
            temperature_indices = data["profile_t2m_index"]
            factors = data["profile_scale"]

            invalid = profile_indices <= 0
            profile_indices[invalid] = 1
            profile_indices = np.clip(profile_indices, 1, 12)
            temperature_indices = np.clip(temperature_indices, 1, 12)

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
                "rain_water_content": (("scans", "pixels", "layers"), rwc),
                "cloud_water_content": (("scans", "pixels", "layers"), cwc),
                "snow_water_content": (("scans", "pixels", "layers"), swc),
                "latent_heat": (("scans", "pixels", "layers"), lh),
            }
        else:
            dataset = {}

        dims = ["scans", "pixels", "channels", "layers"]
        for k, d in data.items():
            if "profile" not in k:
                dataset[k] = (dims[: len(d.shape)], d)

        return xarray.Dataset(dataset)
