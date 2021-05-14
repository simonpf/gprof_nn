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

LOGGER = logging.getLogger(__name__)

N_SPECIES = 5
N_TEMPERATURES = 12
N_LAYERS = 28
N_PROFILES = 80

DATE6_TYPE = np.dtype(
    [("year", "i2"),
     ("month", "i2"),
     ("day", "i2"),
     ("hour", "i2"),
     ("minute", "i2"),
     ("second", "i2")]
)

DATE7_TYPE = np.dtype(
    [("year", "i2"),
     ("month", "i2"),
     ("day", "i2"),
     ("hour", "i2"),
     ("minute", "i2"),
     ("second", "i2"),
     ("millisecond", "i2")]
)

ORBIT_HEADER_TYPES = np.dtype(
    [("satellite", "a12"),
     ("sensor", "a12"),
     ("preprocessor", "a12"),
     ("algorithm", "a12"),
     ("profile_database_file", "a128"),
     ("radiometer_file", "a128"),
     ("creation_date", DATE6_TYPE),
     ("granule_start_date", DATE6_TYPE),
     ("granule_end_date", DATE6_TYPE),
     ("granule_number", "i"),
     ("number_of_scans", "i2"),
     ("number_of_pixels", "i2"),
     ("profile_struct", "i1"),
     ("spares", "a51")]
)

PROFILE_INFO_TYPES = np.dtype(
    [("n_species", "i1"),
     ("n_temps", "i1"),
     ("n_layers", "i1"),
     ("n_profiles", "i1"),
     ("species_description", f"{N_SPECIES}a20"),
     ("height_top_layers", f"{N_LAYERS}f4"),
     ("temperature", f"{N_TEMPERATURES}f4"),
     ("profiles", f"{N_SPECIES * N_TEMPERATURES * N_LAYERS * N_PROFILES}f4")]
)

SCAN_HEADER_TYPES = np.dtype(
    [("scan_latitude", "f4"),
     ("scan_longitude", "f4"),
     ("scan_altitude", "f4"),
     ("scan_date", DATE7_TYPE),
     ("spare", "i2")])

DATA_RECORD_TYPES = np.dtype(
    [("pixel_status", "i1"),
     ("quality_flag", "i1"),
     ("l1c_quality_flag", "i1"),
     ("surface_type_index", "i1"),
     ("tcwv_index", "i1"),
     ("pop_index", "i1"),
     ("t2m_index", "i2"),
     ("airmass_index", "i2"),
     ("sun_glint_angle", "i1"),
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
     ("precip_1st_tertial", "f4"),
     ("precip_3rd_tertial", "f4"),
     ("profile_t2m_index", "i2"),
     ("profile_number", f"{N_SPECIES}i2"),
     ("profile_scale", f"{N_SPECIES}f4")
     ]
)

DATA_RECORD_TYPES_SENSITIVITY = np.dtype(
    [("pixel_status", "i1"),
     ("quality_flag", "i1"),
     ("l1c_quality_flag", "i1"),
     ("surface_type_index", "i1"),
     ("tcwv_index", "i1"),
     ("pop_index", "i1"),
     ("t2m_index", "i2"),
     ("airmass_index", "i2"),
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
     ("precip_1st_tertial", "f4"),
     ("precip_3rd_tertial", "f4"),
     ("profile_t2m_index", "i2"),
     ("profile_number", f"{N_SPECIES}i2"),
     ("profile_scale", f"{N_SPECIES}f4")
     ]
)

class RetrievalFile:
    """
    Class to read binary retrieval results from the GPROF 7 algorithm.
    """
    def __init__(self,
                 filename,
                 has_sensitivity=False):
        """
        Read retrieval results.

        Args:
            filename: The path to the file to open.
            has_sensitivity: Flag indicating whether the file contains
                gradients of the surface precip retrieval. This special
                is only produced by a customized version of the GPROF
                retrieval.
        """
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
        self.n_scans = self.orbit_header["number_of_scans"][0]
        self.n_pixels = self.orbit_header["number_of_pixels"][0]

        np.random.seed(self.orbit_header["granule_number"])
        self.scan_indices = np.random.permutation(np.arange(self.n_scans))
        self.pixel_indices = np.random.permutation(np.arange(self.n_pixels))

        if has_sensitivity:
            self.data_record_types = DATA_RECORD_TYPES_SENSITIVITY
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
        offset += i * (SCAN_HEADER_TYPES.itemsize
                       + self.n_pixels * self.data_record_types.itemsize)
        offset += SCAN_HEADER_TYPES.itemsize
        return np.frombuffer(self.data,
                             self.data_record_types,
                             count=self.n_pixels,
                             offset=offset)

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
        offset += i * (SCAN_HEADER_TYPES.itemsize
                       + self.n_pixels * self.data_record_types.itemsize)
        return np.frombuffer(self.data,
                             SCAN_HEADER_TYPES,
                             count=1,
                             offset=offset)

    def to_xarray_dataset(self):
        """
        Return retrieval results as xarray dataset.
        """
        data = {k: np.zeros((self.n_scans, self.n_pixels) + d[0].shape)
                for k, d in self.data_record_types.fields.items()}
        for i, s in enumerate(self.scans):
            for k, d in data.items():
                d[i] = s[k]

        dims = ["scans", "pixels", "channels"]
        data = {k: (dims[:len(d.shape)], d) for k, d in data.items()
                if not k.startswith("profile")
                }
        return xarray.Dataset(data)

class Retrieval:
    N_CHANNELS = 15
    def __init__(self,
                 preprocessor_file,
                 normalizer_file,
                 model,
                 scans_per_batch=128):
        from gprof_nn.data.preprocessor import PreprocessorFile
        self.preprocessor_file = preprocessor_file
        self.normalizer_file = normalizer_file
        self.input_data = PreprocessorFile(preprocessor_file).to_xarray_dataset()
        self.normalizer = Normalizer.load(normalizer_file)
        self.model = model

        self.n_scans = self.input_data.scans.size
        self.n_pixels = self.input_data.pixels.size
        self.scans_per_batch = scans_per_batch
        self.n_batches = self.n_scans // scans_per_batch
        remainder = self.n_scans % scans_per_batch
        if remainder > 0:
            self.n_batches += 1

    def __len__(self):
        return self.n_batches

    def __getitem__(self, i):
        if i > self.n_batches:
            raise IndexError()
        return self._get_batch_0d(i)

    def _get_batch_0d(self, i):

        i_start = i * self.scans_per_batch
        i_end = (i + 1) * self.scans_per_batch

        bts = self.input_data["brightness_temperatures"][i_start:i_end, :, :].data
        bts = bts.reshape(-1, Retrieval.N_CHANNELS).copy()
        bts[bts < 0] = np.nan

        # 2m temperature
        t2m = self.input_data["two_meter_temperature"][i_start:i_end, :].data
        t2m = t2m.reshape(-1, 1)
        # Total precipitable water.
        tcwv = self.input_data["total_column_water_vapor"][i_start:i_end, :].data
        tcwv = tcwv.reshape(-1, 1)

        # Surface type
        n = bts.shape[0]
        st = self.input_data["surface_type"][i_start:i_end, :].data
        st = st.reshape(-1, 1).astype(int)
        n_types = 19
        st_1h = np.zeros((n, n_types), dtype=np.float32)
        st_1h[np.arange(n), st.ravel()] = 1.0

        # Airmass type
        am = self.input_data["airmass_type"][i_start:i_end, :].data
        am = np.maximum(am.reshape(-1, 1).astype(int), 0)
        n_types = 4
        am_1h = np.zeros((n, n_types), dtype=np.float32)
        am_1h[np.arange(n), am.ravel()] = 1.0

        x = np.concatenate([bts, t2m, tcwv, st_1h, am_1h], axis=1)
        return self.normalizer(x)


    def run_retrieval(self):

        means = {}
        precip_1st_tercile = []
        precip_3rd_tercile = []
        pop = []

        with torch.no_grad():
            for i in range(len(self)):
                x = self[i]

                y_pred = self.model.predict(x)
                if not isinstance(y_pred, dict):
                    y_pred = {"surface_precip": y_pred}

                y_mean = self.model.posterior_mean(y_pred=y_pred)
                for k, y in y_pred.items():
                    means.setdefault(k, []).append(y_mean[k].cpu())
                    if k == "surface_precip":
                        t = self.model.posterior_quantiles(
                            y_pred=y,
                            quantiles=[0.333, 0.667],
                            key=k
                        )
                        precip_1st_tercile.append(t[:, :1].cpu())
                        precip_3rd_tercile.append(t[:, 1:].cpu())
                        p = self.model.probability_larger_than(y_pred=y,
                                                               y=1e-4,
                                                               key=k)
                        pop.append(p.cpu())


        dims = ["scans", "pixels", "levels"]
        data = {}
        for k in means:
            y = np.concatenate(means[k].numpy())
            if y.ndim == 1:
                y = y.reshape(-1, 221)
            else:
                y = y.reshape(-1, 221, 28)
            data[k] = (dims[:y.ndim], y)

        data["precip_1st_tercile"] = (
            dims[:2],
            np.concatenate(precip_1st_tercile.numpy()).reshape(-1, 221)
        )
        data["precip_3rd_tercile"] = (
            dims[:2],
            np.concatenate(precip_3rd_tercile.numpy()).reshape(-1, 221)
        )
        data["precip_pip"] = (
            dims[:2],
            np.concatenate(pop.numpy()).reshape(-1, 221)
        )

        return xarray.Dataset(data)
