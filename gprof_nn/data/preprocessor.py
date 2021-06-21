"""
==========================
gprof_nn.data.preprocessor
==========================

This module contains the PreprocessorFile class that provides an interface
to read and write the binary preprocessor format that is used as direct input
for running the GPROF algorithm.
"""
from datetime import datetime
import logging
import os
import subprocess
import tempfile

import numpy as np
import torch
import xarray as xr

from gprof_nn.definitions import (MISSING,
                                  TCWV_MIN,
                                  TCWV_MAX,
                                  T2M_MIN,
                                  T2M_MAX)
from gprof_nn.data import retrieval
from gprof_nn.data.retrieval import calculate_frozen_precip
from gprof_nn.data.profiles import ProfileClusters
from pathlib import Path

LOGGER = logging.getLogger(__name__)

###############################################################################
# Struct types
###############################################################################

N_SPECIES = 5
N_TEMPERATURES = 12
N_LAYERS = 28
N_PROFILES = 80
N_CHANNELS = 15

TB_MIN = 40.0
TB_MAX = 325.0
LAT_MIN = -90.0
LAT_MAX = 90.0
LON_MIN = -180.0
LON_MAX = 180.0

DATE_TYPE = np.dtype(
    [
        ("year", "i2"),
        ("month", "i2"),
        ("day", "i2"),
        ("hour", "i2"),
        ("minute", "i2"),
        ("second", "i2"),
    ]
)

ORBIT_HEADER_TYPES = np.dtype(
    [
        ("satellite", "a12"),
        ("sensor", "a12"),
        ("preprocessor", "a12"),
        ("profile_database_file", "a128"),
        ("radiometer_file", "a128"),
        ("calibration_file", "a128"),
        ("granule_number", "i"),
        ("number_of_scans", "i"),
        ("number_of_pixels", "i"),
        ("n_channels", "i"),
        ("frequencies", f"{N_CHANNELS}f4"),
        ("comment", "a40"),
    ]
)

SCAN_HEADER_TYPES = np.dtype(
    [
        ("scan_date", DATE_TYPE),
        ("scan_latitude", "f4"),
        ("scan_longitude", "f4"),
        ("scan_altitude", "f4"),
    ]
)

DATA_RECORD_TYPES = np.dtype(
    [
        ("latitude", "f4"),
        ("longitude", "f4"),
        ("brightness_temperatures", f"{N_CHANNELS}f4"),
        ("earth_incidence_angle", f"{N_CHANNELS}f4"),
        ("wet_bulb_temperature", "f4"),
        ("lapse_rate", "f4"),
        ("total_column_water_vapor", "f4"),
        ("surface_temperature", "f4"),
        ("two_meter_temperature", "f4"),
        ("quality_flag", "i"),
        ("sunglint_angle", "i1"),
        ("surface_type", "i1"),
        ("airmass_type", "i2"),
    ]
)


def write_orbit_header(output, data, template=None):
    """
    Write header into preprocessor file.

    Args:
        output: File handle to write the file header to.
        data: xarray Dataset containing the data to write to
             the file handle.
    """
    new_header = np.recarray(1, dtype=ORBIT_HEADER_TYPES)

    if template is not None:
        for k in ORBIT_HEADER_TYPES.fields:
            new_header[k] = template.orbit_header[k]
    else:
        new_header = np.recarray(1, dtype=ORBIT_HEADER_TYPES)
        new_header["satellite"] = "GPM CO"
        new_header["sensor"] = "GMI"
        new_header["preprocessor"] = "NONE"
        new_header["profile_database_file"] = "NONE"
        new_header["radiometer_file"] = "NONE"
        new_header["calibration_file"] = "NONE"
        new_header["granule_number"] = 0
        new_header["n_channels"] = 15
        new_header["comment"] = "Custom preprocessor file for verification."
    new_header["number_of_scans"] = data.scans.size
    new_header["number_of_pixels"] = data.pixels.size
    new_header.tofile(output)


def write_scan_header(output, template=None):
    """
    Write scan header into a preprocessor file.

    Args:
        output: File handle to write the file header to.
        data: xarray Dataset containing the data of the
            given scan.
    """
    if template:
        header = template.get_scan_header(0)
    else:
        header = np.recarray(1, dtype=SCAN_HEADER_TYPES)
        header["scan_date"]["year"] = 6
        header["scan_date"]["month"] = 6
        header["scan_date"]["day"] = 6
        header["scan_date"]["hour"] = 6
        header["scan_date"]["minute"] = 6
        header["scan_date"]["second"] = 6
    header.tofile(output)


def write_scan(output, data):
    """
    Write single scan into a preprocessor file.

    Args:
        output: File handle to write the scan to.
        data: xarray Dataset containing the data of the
            given scan.
    """
    n_pixels = data.pixels.size
    scan = np.recarray(n_pixels, dtype=DATA_RECORD_TYPES)
    for k in DATA_RECORD_TYPES.fields:
        if k not in data:
            continue
        scan[k] = data[k]
    scan.tofile(output)


###############################################################################
# Preprocessor file class
###############################################################################


class PreprocessorFile:
    """
    Interface to read CSU preprocessor files.

    Attibutes:
        filename: The path of the source file.
        orbit_header: Numpy structured array containing the orbit header.
        n_scans: The number of scans in the file.
        n_pixels: The number of pixels in the file.
    """

    @classmethod
    def write(cls, filename, data, template=None):
        n_scans = data.scans.size
        with open(filename, "wb") as output:
            write_orbit_header(output, data, template=template)
            for i in range(n_scans):
                scan_data = data[{"scans": i}]
                write_scan_header(output, template=template)
                write_scan(output, scan_data)

    def __init__(self, filename):
        """
        Read preprocessor file.

        Args:
            filename: Path to the file to read.
        """
        self.filename = filename
        with open(self.filename, "rb") as file:
            self.data = file.read()
        self.orbit_header = np.frombuffer(self.data, ORBIT_HEADER_TYPES, count=1)
        self.n_scans = self.orbit_header["number_of_scans"][0]
        self.n_pixels = self.orbit_header["number_of_pixels"][0]

    def __repr__(self):
        """String representation for file."""
        return f"PreprocessorFile(filename='{self.filename}')"

    @property
    def satellite(self):
        """
        The satellite from which the data in this file originates.
        """
        return self.orbit_header["satellite"]

    @property
    def sensor(self):
        """
        The sensor from which the data in this file originates.
        """
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
        Args:
            i: The index of the scan to return.

        Returns:
            The ith scan in the file as numpy structured array of size n_pixels
            and dtype DATA_RECORD_TYPES.
        """
        if i < 0:
            i = self.n_scans + i

        offset = ORBIT_HEADER_TYPES.itemsize
        offset += i * (
            SCAN_HEADER_TYPES.itemsize + self.n_pixels * DATA_RECORD_TYPES.itemsize
        )
        offset += SCAN_HEADER_TYPES.itemsize
        return np.frombuffer(
            self.data, DATA_RECORD_TYPES, count=self.n_pixels, offset=offset
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

        offset = ORBIT_HEADER_TYPES.itemsize
        offset += i * (
            SCAN_HEADER_TYPES.itemsize + self.n_pixels * DATA_RECORD_TYPES.itemsize
        )
        return np.frombuffer(self.data, SCAN_HEADER_TYPES, count=1, offset=offset)

    def to_xarray_dataset(self):
        """
        Return data in file as xarray dataset.
        """
        data = {
            k: np.zeros((self.n_scans, self.n_pixels), dtype=d[0])
            for k, d in DATA_RECORD_TYPES.fields.items()
        }
        for i, s in enumerate(self.scans):
            for k, d in data.items():
                d[i] = s[k]

        dims = ["scans", "pixels", "channels"]
        data = {k: (dims[: len(d.shape)], d) for k, d in data.items()}

        scan_times = np.zeros(self.n_scans, dtype="datetime64[ns]")
        for i in range(self.n_scans):
            date = self.get_scan_header(i)["scan_date"]
            year = date["year"][0]
            month = date["month"][0]
            day = date["day"][0]
            hour = date["hour"][0]
            minute = date["minute"][0]
            second = date["second"][0]
            scan_times[i] = np.datetime64(
                f"{year:04}-{month:02}-{day:02}" f"T{hour:02}:{minute:02}:{second:02}"
            )
        data["scan_time"] = ("scans",), scan_times
        return xr.Dataset(data)

    def write_retrieval_results(self,
                                path,
                                results,
                                ancillary_data=None):
        """
        Write retrieval result to GPROF binary format.

        Args:
            path: The folder to which to write the result. The filename
                  itself follows the GPORF naming scheme.
            results: Dictionary containing the retrieval results.

        Returns:

            Path object to the created binary file.
        """
        path = Path(path)
        filename = path / self._get_retrieval_filename()

        if ancillary_data is not None:
            profiles_raining = ProfileClusters(ancillary_data, True)
            profiles_non_raining = ProfileClusters(ancillary_data, False)
        else:
            profiles_raining = None
            profiles_non_raining = None

        with open(filename, "wb") as file:
            self._write_retrieval_orbit_header(file)
            self._write_retrieval_profile_info(file,
                                               profiles_raining,
                                               profiles_non_raining)
            for i in range(self.n_scans):
                self._write_retrieval_scan_header(file, i)
                self._write_retrieval_scan(
                    file,
                    i,
                    results,
                    profiles_raining=profiles_raining,
                    profiles_non_raining=profiles_non_raining
                )
        return filename

    def _get_retrieval_filename(self):
        """
        Produces GPROF compliant filename from retrieval results dict.
        """
        start_date = self.get_scan_header(0)["scan_date"]
        end_date = self.get_scan_header(-1)["scan_date"]

        name = "2A.GCORE-NN.GMI.V7."

        year, month, day = [start_date[k][0] for k in ["year", "month", "day"]]
        name += f"{year:02}{month:02}{day:02}-"

        hour, minute, second = [start_date[k][0] for k in ["hour", "minute", "second"]]
        name += f"S{hour:02}{minute:02}{second:02}-"

        hour, minute, second = [end_date[k][0] for k in ["hour", "minute", "second"]]
        name += f"E{hour:02}{minute:02}{second:02}."

        granule_number = self.orbit_header["granule_number"][0]
        name += f"{granule_number:06}.BIN"

        return name

    def _write_retrieval_orbit_header(self, file):
        """
        Writes the retrieval orbit header to an opened binary file..

        Args:
            file: Handle to the binary file to write the data to.
        """
        new_header = np.recarray(1, dtype=retrieval.ORBIT_HEADER_TYPES)
        for k in retrieval.ORBIT_HEADER_TYPES.fields:
            if k not in self.orbit_header.dtype.fields:
                continue
            new_header[k] = self.orbit_header[k]

        new_header["algorithm"] = "GPROF-NN"
        date = datetime.now()
        creation_date = np.recarray(1, dtype=retrieval.DATE6_TYPE)
        creation_date["year"] = date.year
        creation_date["month"] = date.month
        creation_date["day"] = date.day
        creation_date["hour"] = date.hour
        creation_date["minute"] = date.minute
        creation_date["second"] = date.second
        new_header["creation_date"] = creation_date

        scan = self.get_scan_header(0)
        new_header["granule_start_date"] = scan["scan_date"]
        scan = self.get_scan_header(self.n_scans - 1)
        new_header["granule_end_date"] = scan["scan_date"]
        new_header["profile_struct"] = 1
        new_header["spares"] = "no calibration table used               "
        new_header.tofile(file)

    def _write_retrieval_profile_info(self,
                                      file,
                                      clusters_raining=None,
                                      clusters_non_raining=None):

        """
        Write the retrieval profile info to an opened binary file.

        Args:
            file: Handle to the binary file to write the data to.
        """
        profile_info = np.recarray(1, dtype=retrieval.PROFILE_INFO_TYPES)

        profile_info["n_species"] = N_SPECIES
        profile_info["n_temps"] = N_TEMPERATURES
        profile_info["n_layers"] = N_LAYERS
        profile_info["n_profiles"] = N_PROFILES
        profile_info["species_description"][0][0] = "Rain water content  ".encode()
        profile_info["species_description"][0][1] = "Cloud water content ".encode()
        profile_info["species_description"][0][2] = "Snow water content  ".encode()
        profile_info["species_description"][0][3] = "Graupel/Hail content".encode()
        profile_info["species_description"][0][4] = "Latent heating      ".encode()
        profile_info["height_top_layers"] = np.concatenate([
            np.linspace(0.5, 10, 20),
            np.linspace(11, 18, 8)
            ])
        profile_info["temperature"] = np.linspace(270.0, 303.0, 12)

        if ((clusters_raining is not None) and
            (clusters_non_raining is not None)):
            profiles_combined = []
            for i, s in enumerate([
                    "rain_water_content",
                    "cloud_water_content",
                    "snow_water_content",
                    "graupel_water_content",
                    "latent_heat"
            ]):
                profiles = [clusters_raining.get_profile_data(s),
                            clusters_non_raining.get_profile_data(s)]
                profiles = np.concatenate(profiles, axis=-1)
                profiles_combined.append(profiles)

            profiles_combined = np.stack(profiles_combined)
            profile_info["profiles"][0] = profiles_combined.ravel(order="f")
        else:
            profile_info["profiles"] = MISSING
        profile_info.tofile(file)


    def _write_retrieval_scan_header(self, file, scan_index):
        """
        Write the scan header corresponding to the ith header in the file
        to a given file stream.

        Args:
            file: Handle to the binary file to write the data to.
            scan_index: The index of the scan for which to write the header.
        """
        header = self.get_scan_header(scan_index)
        scan_header = np.recarray(1, dtype=retrieval.SCAN_HEADER_TYPES)
        scan_header["scan_latitude"] = header["scan_latitude"]
        scan_header["scan_longitude"] = header["scan_longitude"]
        scan_header["scan_altitude"] = header["scan_altitude"]
        scan_header["scan_date"]["year"] = header["scan_date"]["year"]
        scan_header["scan_date"]["month"] = header["scan_date"]["month"]
        scan_header["scan_date"]["day"] = header["scan_date"]["day"]
        scan_header["scan_date"]["hour"] = header["scan_date"]["hour"]
        scan_header["scan_date"]["minute"] = header["scan_date"]["minute"]
        scan_header["scan_date"]["second"] = header["scan_date"]["second"]
        scan_header["scan_date"]["millisecond"] = 0.0
        scan_header.tofile(file)

    def _write_retrieval_scan(
            self,
            file,
            scan_index,
            retrieval_data,
            profiles_raining=None,
            profiles_non_raining=None
    ):
        """
        Write retrieval data from a full scan to a binary stream.


        Args:
            file: Handle to the binary file to write the data to.
            precip_mean: 1D array containing the mean retrieved precipitation for
                 each pixel.
            precip_1st_tertial: 1D array containing the 1st tertial retrieved from the data.
            precip_3rd_tertial: 1D array containing the 3rd tertial retrieved from the data
            precip_pop: 1D array containing the probability of precipitation in the scan.
        """
        data = retrieval_data[{"scans": scan_index}]
        scan_data = self.get_scan(scan_index)

        out_data = np.recarray(self.n_pixels, dtype=retrieval.DATA_RECORD_TYPES)

        # Pixel status
        ps = out_data["pixel_status"]
        ps[:] = 0
        indices = ((scan_data["latitude"] < LAT_MIN) +
                   (scan_data["latitude"] > LAT_MAX) +
                   (scan_data["longitude"] < LON_MIN) +
                   (scan_data["longitude"] > LON_MAX))
        ps[indices] = 1
        indices = np.any(((scan_data["brightness_temperatures"] < TB_MIN) +
                          (scan_data["brightness_temperatures"] > TB_MAX)),
                         axis=-1)
        ps[indices] = 2
        indices = ((scan_data["two_meter_temperature"] < 0) +
                   (scan_data["total_column_water_vapor"] < 0) +
                   (scan_data["surface_type"] < 0) +
                   (scan_data["airmass_type"] < 0))
        ps[indices] = 4

        out_data["l1c_quality_flag"] = scan_data["quality_flag"]
        out_data["surface_type"] = scan_data["surface_type"]

        tcwv = np.round(scan_data["total_column_water_vapor"]).astype(int)
        tcwv = np.clip(tcwv, TCWV_MIN, TCWV_MAX)
        out_data["total_column_water_vapor"] = tcwv
        t2m = np.round(scan_data["two_meter_temperature"]).astype(int)
        t2m = np.clip(t2m, T2M_MIN, T2M_MAX)
        out_data["two_meter_temperature"] = t2m

        out_data["pop"] = data["pop"].astype(int)
        out_data["airmass_type"] = scan_data["airmass_type"]
        out_data["sunglint_angle"] = scan_data["sunglint_angle"]
        out_data["precip_flag"] = data["precip_flag"]
        out_data["latitude"] = scan_data["latitude"]
        out_data["longitude"] = scan_data["longitude"]

        out_data["surface_precip"] = data["surface_precip"]

        wet_bulb_temperature = scan_data["wet_bulb_temperature"]
        surface_type = scan_data["surface_type"]
        surface_precip = data["surface_precip"]
        frozen_precip = calculate_frozen_precip(
            wet_bulb_temperature,
            surface_type,
            surface_precip
        )
        frozen_precip[surface_precip < 0] = MISSING
        out_data["frozen_precip"] = frozen_precip
        out_data["convective_precip"] = data["convective_precip"]
        out_data["rain_water_path"] = data["rain_water_path"]
        out_data["cloud_water_path"] = data["cloud_water_path"]
        out_data["ice_water_path"] = data["ice_water_path"]
        out_data["most_likely_precip"] = data["most_likely_precip"]
        out_data["precip_1st_tercile"] = data["precip_1st_tercile"]
        out_data["precip_3rd_tercile"] = data["precip_3rd_tercile"]
        if "pixel_status" in data.variables:
            out_data["pixel_status"] = data["pixel_status"]
        if "quality_flag" in data.variables:
            out_data["quality_flag"] = data["quality_flag"]

        if profiles_raining is not None and profiles_non_raining is not None:
            t2m = scan_data["two_meter_temperature"]
            t2m_indices = profiles_raining.get_t2m_indices(t2m)
            out_data["profile_t2m_index"] = t2m_indices + 1

            profile_indices = np.zeros((self.n_pixels, N_SPECIES),
                                       dtype=np.float32)
            profile_scales = np.zeros((self.n_pixels, N_SPECIES),
                                      dtype=np.float32)
            precip_flag = data["precip_flag"].data
            for i, s in enumerate([
                    "rain_water_content",
                    "cloud_water_content",
                    "snow_water_content",
                    "latent_heat"
            ]):
                invalid = np.all(data[s].data < -500, axis=-1)
                scales_r, indices_r = profiles_raining.get_scales_and_indices(
                    s, t2m, data[s].data
                )
                scales_nr, indices_nr = profiles_non_raining.get_scales_and_indices(
                    s, t2m, data[s].data
                )
                scales = np.where(surface_precip > 0.01, scales_r, scales_nr)
                indices = np.where(surface_precip > 0.01,
                                   indices_r,
                                   indices_nr + 40)

                profile_indices[:, i] = indices + 1
                profile_indices[invalid, i] = 0
                profile_scales[:, i] = scales
                profile_scales[invalid, i] = 1.0
            out_data["profile_index"] = profile_indices
            out_data["profile_scale"] = profile_scales

        else:
            out_data["profile_t2m_index"] = 0
            out_data["profile_scale"] = 1.0
            out_data["profile_index"] = 0
        out_data.tofile(file)


class PreprocessorLoader0D:
    """
    Interface class to run the GPROF-NN retrieval on preprocessor files.
    """
    def __init__(self,
                 filename,
                 normalizer,
                 scans_per_batch=256):
        """
        Create preprocessor loader.

        Args:
            filename: Path to the preprocessor file from which to load the
                input data.
            normalizer: The normalizer object to use to normalize the input
                data.
            scans_per_batch: How scans should be combined into a single
                batch.
        """
        self.filename = filename
        preprocessor_file = PreprocessorFile(filename)
        self.data = preprocessor_file.to_xarray_dataset()
        self.normalizer = normalizer
        self.n_scans = self.data.scans.size
        self.n_pixels = self.data.pixels.size
        self.scans_per_batch = scans_per_batch

    def __len__(self):
        """
        The number of batches in the preprocessor file.
        """
        n = self.n_scans // self.scans_per_batch
        if (self.n_scans % self.scans_per_batch) > 0:
            n = n + 1
        return n

    def get_batch(self, i):
        """
        Return batch of retrieval inputs as PyTorch tensor.

        Args:
            i: The index of the batch.

        Return:
            PyTorch Tensor ``x`` containing the normalized inputs.
        """
        i_start = i * self.scans_per_batch
        i_end = min(i_start + self.scans_per_batch,
                    self.n_scans)

        n = (i_end - i_start) * self.data.pixels.size
        x = np.zeros((n, 39), dtype=np.float32)

        tbs = self.data["brightness_temperatures"].data[i_start:i_end]
        tbs = tbs.reshape(-1, 15)
        t2m = self.data["two_meter_temperature"].data[i_start:i_end]
        t2m = t2m.reshape(-1)
        tcwv = self.data["total_column_water_vapor"].data[i_start:i_end]
        tcwv = tcwv.reshape(-1)
        st = self.data["surface_type"].data[i_start:i_end]
        st = st.reshape(-1)
        at = np.maximum(self.data["airmass_type"].data[i_start:i_end], 0.0)
        at = at.reshape(-1)

        x[:, :15] = tbs
        x[:, :15][x[:, :15] < 0] = np.nan

        x[:, 15] = t2m
        x[:, 15][x[:, 15] < 0] = np.nan

        x[:, 16] = tcwv
        x[:, 16][x[:, 16] < 0] = np.nan

        for i in range(18):
            x[:, 17 + i][st == i + 1] = 1.0

        for i in range(4):
            x[:, 35 + i][at == i] = 1.0

        x = self.normalizer(x)
        return torch.tensor(x)

    def run_retrieval(self,
                      xrnn):
        """
        Run retrieval on input data.

        Args:
            xrnn: The network to run the retrieval with.

        Return:
            ``xarray.Dataset`` containing the retrieval results.
        """
        means = {}
        precip_1st_tercile = []
        precip_3rd_tercile = []
        pop = []

        with torch.no_grad():
            device = next(iter(xrnn.model.parameters())).device
            for i in range(len(self)):
                x = self.get_batch(i)
                x = x.float().to(device)
                y_pred = xrnn.predict(x)
                if not isinstance(y_pred, dict):
                    y_pred = {"surface_precip": y_pred}

                y_mean = xrnn.posterior_mean(y_pred=y_pred)
                for k, y in y_pred.items():
                    means.setdefault(k, []).append(y_mean[k].cpu())
                    if k == "surface_precip":
                        t = xrnn.posterior_quantiles(
                            y_pred=y, quantiles=[0.333, 0.667], key=k
                        )
                        precip_1st_tercile.append(t[:, :1].cpu())
                        precip_3rd_tercile.append(t[:, 1:].cpu())
                        p = xrnn.probability_larger_than(y_pred=y, y=1e-4, key=k)
                        pop.append(p.cpu())

        dims = ["scans", "pixels", "levels"]
        data = {}
        for k in means:
            y = np.concatenate([t.numpy() for t in means[k]])
            if y.ndim == 1:
                y = y.reshape(-1, 221)
            else:
                y = y.reshape(-1, 221, 28)
            data[k] = (dims[:y.ndim], y)

        data["precip_1st_tercile"] = (
            dims[:2],
            np.concatenate([t.numpy() for t in precip_1st_tercile]).reshape(-1, 221),
        )
        data["precip_3rd_tercile"] = (
            dims[:2],
            np.concatenate([t.numpy() for t in precip_3rd_tercile]).reshape(-1, 221),
        )
        pop = np.concatenate([t.numpy() for t in pop]).reshape(-1, 221)
        data["pop"] = (dims[:2], pop)
        data["precip_flag"] = (dims[:2], pop > 0.5)
        data = xr.Dataset(data)
        return data

    def write_retrieval_results(self,
                                output_path,
                                results,
                                ancillary_data=None):
        """
        Write retrieval results to file.

        Args:
            output_path: The folder to which to write the output.
            results: ``xarray.Dataset`` containing the retrieval results.
            ancillary_data: The folder containing the profile clusters.

        Return:
            The filename of the retrieval output file.
        """
        preprocessor_file = PreprocessorFile(self.filename)
        return preprocessor_file.write_retrieval_results(
            output_path,
            results,
            ancillary_data=ancillary_data
        )


###############################################################################
# Running the preprocessor
###############################################################################

PREPROCESSOR_SETTINGS = {
    "prodtype": "CLIMATOLOGY",
    "prepdir": "/qdata2/archive/ERA5/",
    "ancdir": "/qdata1/pbrown/gpm/ppancillary/",
    "ingestdir": "/qdata1/pbrown/gpm/ppingest/",
}


def get_preprocessor_settings():
    """
    Return preprocessor settings as list of command line arguments to invoke
    the preprocessor.
    """
    return [v for _, v in PREPROCESSOR_SETTINGS.items()]


def run_preprocessor(l1c_file, output_file=None):
    """
    Run preprocessor on L1C GMI file.

    Args:
        l1c_file: Path of the L1C file for which to extract the input data
             using the preprocessor.
        output_file: Optional name of an output file. Results will be written
            to a temporary file and the results returned as xarray.Dataset.

    Returns:
        xarray.Dataset containing the retrieval input data for the given L1C
        file or None when the 'output_file' argument is given.
    """
    file = None
    if output_file is None:
        file = tempfile.NamedTemporaryFile(dir="/gdata/simon/tmp")
        output_file = file.name
    try:
        jobid = str(os.getpid()) + "_pp"
        args = [jobid] + get_preprocessor_settings()
        args.insert(2, l1c_file)
        args.append(output_file)
        subprocess.run(["gprof2020pp_GMI_L1C"] + args,
                       check=True,
                       capture_output=True)
        if file is not None:
            data = PreprocessorFile(output_file).to_xarray_dataset()

    except subprocess.CalledProcessError as error:
        LOGGER.warning(
            "Running the preprocessor for file %s failed with the following"
            " error: %s",
            l1c_file,
            error.stdout + error.stderr,
        )
        return None
    finally:
        if file is not None:
            file.close()
    if file is not None:
        return data
