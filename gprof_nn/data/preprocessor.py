"""
==========================
gprof_nn.data.preprocessor
==========================

This module defines the 'PreprocessorFile' that provides an interface
to read and write preprocessor files.

Additionally, it defines functions to run the preprocessor on the CSU
 systems.
"""
from datetime import datetime
import logging
from math import ceil
import os
import pickle
import shutil
import subprocess
import tempfile

import numpy as np
import scipy as sp
import scipy.interpolate
import xarray as xr

from gprof_nn.definitions import (
    MISSING,
    TCWV_MIN,
    TCWV_MAX,
    T2M_MIN,
    T2M_MAX,
    ERA5,
    GANAL,
)
from gprof_nn import sensors
from gprof_nn.data import retrieval
from gprof_nn.data.profiles import ProfileClusters
from pathlib import Path


###############################################################################
# Struct types
###############################################################################

CHANNEL_INDICES = {
    "TMIPR": [0, 1, 2, 3, 4, 6, 7, 8, 9],
    "TMIPO": [0, 1, 2, 3, 4, 6, 7, 8, 9],
    "SSMI": [2, 3, 4, 6, 7, 8, 9],
    "SSMIS": [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14],
    "AMSR2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "AMSRE": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}

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

SCAN_HEADER_TYPE = np.dtype(
    [
        ("scan_date", DATE_TYPE),
        ("scan_latitude", "f4"),
        ("scan_longitude", "f4"),
        ("scan_altitude", "f4"),
    ]
)

# Generic orbit that reads the parts that are similar
# for all sensors.
ORBIT_HEADER = np.dtype(
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
        ("chan_freq", "f4", 15),
        ("comment", "a40"),
    ]
)


def is_available(sensor: "sensors.Sensor") -> bool:
    """
    Check whether preprocessor for given sensor is available.
    """
    key = sensor.sensor_id
    executable = PREPROCESSOR_EXECUTABLES.get(key, None)
    if executable is None:
        return False
    return Path(executable).exists()


def write_orbit_header(output, data, sensor, template=None):
    """
    Write header into preprocessor file.

    Args:
        output: File handle to write the file header to.
        data: xarray Dataset containing the data to write to
             the file handle.
    """
    new_header = np.recarray(1, dtype=sensor.preprocessor_orbit_header)

    if template is not None:
        for k in sensor.preprocessor_orbit_header.fields:
            new_header[k] = template.orbit_header[k]
    else:
        new_header = np.recarray(1, dtype=sensor.preprocessor_orbit_header)
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
        header = np.recarray(1, dtype=SCAN_HEADER_TYPE)
        header["scan_date"]["year"] = 6
        header["scan_date"]["month"] = 6
        header["scan_date"]["day"] = 6
        header["scan_date"]["hour"] = 6
        header["scan_date"]["minute"] = 6
        header["scan_date"]["second"] = 6
    header.tofile(output)


def write_scan(output, data, sensor):
    """
    Write single scan into a preprocessor file.

    Args:
        output: File handle to write the scan to.
        data: xarray Dataset containing the data of the
            given scan.
    """
    n_pixels = data.pixels.size
    scan = np.recarray(n_pixels, dtype=sensor.preprocessor_pixel_record)
    for k in sensor.preprocessor_pixel_record.fields:
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
    def write(cls, filename, data, sensor, template=None):
        n_scans = data.scans.size
        with open(filename, "wb") as output:
            write_orbit_header(output, data, sensor, template=template)
            for i in range(n_scans):
                scan_data = data[{"scans": i}]
                write_scan_header(output, template=template)
                write_scan(output, scan_data, sensor)

    def __init__(self, filename):
        """
        Read preprocessor file.

        Args:
            filename: Path to the file to read.
        """
        self.filename = filename
        with open(self.filename, "rb") as file:
            self.data = file.read()
        # Read generic part of header.
        self.orbit_header = np.frombuffer(self.data, ORBIT_HEADER, count=1)

        # Parse sensor.
        sensor = self.orbit_header["sensor"][0].decode().strip()
        platform = self.orbit_header["satellite"][0].decode().strip()
        try:
            # First get generic sensor which is required to determine
            # offset of first scan.
            self._sensor = sensors.get_sensor(sensor)
            # Now query sensor again with date information.
            self._sensor = sensors.get_sensor(
                sensor,
                platform,
                self.first_scan_time
            )
        except AttributeError as e:
            raise ValueError(f"The sensor '{sensor}' is not yet supported.")

        # Reread full header.
        self.orbit_header = np.frombuffer(
            self.data, self.sensor.preprocessor_orbit_header, count=1
        )
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
        return self._sensor

    @property
    def scans(self):
        """
        Iterates over the scans in the file. Each scan is returned as Numpy
        structured array of size 'n_pixels' and dtype corresponding to the
        'preprocessor_pixel_record' type of the sensor.
        """
        for i in range(self.n_scans):
            yield self.get_scan(i)

    def write_subset(self, filename, n_scans=None):
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

            for i in range(n_scans):
                self.get_scan_header(i).tofile(output)
                self.get_scan(i).tofile(output)

    def get_scan(self, i):
        """
        Args:
            i: The index of the scan to return.

        Returns:
            The ith scan in the file as numpy structured array of size n_pixels
            and and dtype corresponding to the 'preprocessor_pixel_record' type of
            the sensor.
        """
        if i < 0:
            i = self.n_scans + i

        offset = self.sensor.preprocessor_orbit_header.itemsize
        record_type = self.sensor.preprocessor_pixel_record
        record_size = record_type.itemsize
        offset += i * (SCAN_HEADER_TYPE.itemsize + self.n_pixels * record_size)
        offset += SCAN_HEADER_TYPE.itemsize
        return np.frombuffer(self.data, record_type, count=self.n_pixels, offset=offset)

    def get_scan_header(self, i):
        """
        Args:
            i: The index of the scan to return.

        Returns:
            The header of the ith scan in the file as numpy structured array
            of size n_pixels and dtype SCAN_HEADER_TYPE.
        """
        if i < 0:
            i = self.n_scans + i

        offset = self.sensor.preprocessor_orbit_header.itemsize
        record_type = self.sensor.preprocessor_pixel_record
        record_size = record_type.itemsize
        offset += i * (SCAN_HEADER_TYPE.itemsize + self.n_pixels * record_size)
        return np.frombuffer(self.data, SCAN_HEADER_TYPE, count=1, offset=offset)

    @property
    def first_scan_time(self):
        """
        Returns the first scant time as a numpy.datetime64 object.
        """
        offset = self.sensor.preprocessor_orbit_header.itemsize
        data = np.frombuffer(self.data, SCAN_HEADER_TYPE, count=1, offset=offset)
        date = data["scan_date"]
        date = ((date["year"] - 1971 + 1).astype("datetime64[Y]") +
                (date["month"] - 1).astype("timedelta64[M]") +
                (date["day"] - 1).astype("timedelta64[D]") +
                date["hour"].astype("timedelta64[h]") +
                date["minute"].astype("timedelta64[m]") +
                date["second"].astype("timedelta64[s]"))
        return date


    def to_xarray_dataset(self):
        """
        Return data in file as xarray dataset.
        """
        record_type = self.sensor.preprocessor_pixel_record
        data = {
            k: np.zeros((self.n_scans, self.n_pixels), dtype=d[0])
            for k, d in record_type.fields.items()
        }
        for i, s in enumerate(self.scans):
            for k, d in data.items():
                if k != "__padding__":
                    d[i] = s[k]

        if self.sensor.sensor_name in CHANNEL_INDICES:
            ch_inds = CHANNEL_INDICES[self.sensor.sensor_name]
            tbs = data["brightness_temperatures"]
            data["brightness_temperatures"] = tbs[..., ch_inds]
            eia = data["earth_incidence_angle"]
            data["earth_incidence_angle"] = eia[..., ch_inds]

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
        dataset = xr.Dataset(data)

        sensor = self.orbit_header["sensor"][0].decode().strip()
        satellite = self.orbit_header["satellite"][0].decode().strip()
        preprocessor = self.orbit_header["preprocessor"][0].decode().strip()
        dataset.attrs["satellite"] = satellite
        dataset.attrs["sensor"] = sensor
        dataset.attrs["preprocessor"] = preprocessor
        dataset.attrs["frequencies"] = self.orbit_header["frequencies"][0]
        return dataset

    def write_retrieval_results(self, path, results, suffix=""):
        """
        Write retrieval result to GPROF binary format.

        Args:
            path: The folder to which to write the result. The filename
                  itself follows the GPORF naming scheme.
            results: Dictionary containing the retrieval results.
            suffix: Suffix to append to algorithm name in filename.

        Returns:

            Path object to the created binary file.
        """
        path = Path(path)
        if path.is_dir():
            filename = path / self._get_retrieval_filename(suffix=suffix)
        else:
            filename = path

        LOGGER = logging.getLogger(__name__)
        LOGGER.info(
            "Writing retrieval results in GPROF binary format "
            "to file '%s'.", str(filename)
        )

        n_scans = results.scans.size
        with open(filename, "wb") as file:
            self._write_retrieval_orbit_header(file)
            self._write_retrieval_profile_info(file)
            for i in range(n_scans):
                self._write_retrieval_scan_header(file, i)
                self._write_retrieval_scan(file, i, results)
        return filename

    def _get_retrieval_filename(self, suffix=""):
        """
        Produces GPROF compliant filename from retrieval results dict.
        """
        start_date = self.get_scan_header(0)["scan_date"]
        end_date = self.get_scan_header(-1)["scan_date"]

        if suffix != "":
            suffix = "_" + suffix
        name = f"2A.GPROF-NN{suffix}.GMI.V0."

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

    def _write_retrieval_profile_info(
        self, file, clusters_raining=None, clusters_non_raining=None
    ):
        """
        Write the retrieval profile info to an opened binary file.

        Args:
            file: Handle to the binary file to write the data to.
        """
        profile_info = np.recarray(1, dtype=retrieval.PROFILE_INFO_TYPES)
        profile_info["height_top_layers"] = np.concatenate(
            [np.linspace(0.5, 10, 20), np.linspace(11, 18, 8)]
        )
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
    ):
        """
        Write retrieval data from a full scan to a binary stream.


        Args:
            file: Handle to the binary file to write the data to.
            scan_index: The index of the scan to write.
            retrieval_data: The ``xarray.Dataset`` containing the retrieval
                results.
        """
        data = retrieval_data[{"scans": scan_index}]
        scan_data = self.get_scan(scan_index)

        out_data = np.recarray(self.n_pixels, dtype=retrieval.DATA_RECORD_TYPES)

        # Pixel status
        out_data["pixel_status"] = data["pixel_status"]
        out_data["quality_flag"] = data["quality_flag"]
        out_data["l1c_quality_flag"] = scan_data["quality_flag"]

        carry_over = [
            "total_column_water_vapor", "two_meter_temperature", "convective_precipitation", "moisture_convergence",
            "snow_depth", "orographic_wind", "10m_wind", "mountain_type", "land_fraction", "ice_fraction", "elevation",
            "snow_mask", "sunglint_angle"
        ]
        lai = scan_data["leaf_area_index"]
        if np.any(np.isfinite(lai)):
            carry_over.insert(4, "leaf_area_index")
        else:
            carry_over.insert(4, "leaf_area_index_climatology")
        for name in carry_over:
            out_data[name] = scan_data[name]

        out_data["probability_of_precipitation"] = data.probability_of_precipitation.data
        out_data["precipitation_flag"] = 0.5 < data.probability_of_precipitation.data
        out_data["latitude"] = scan_data["latitude"]
        out_data["longitude"] = scan_data["longitude"]

        out_data["surface_precip"] = data["surface_precip"]

        wet_bulb_temperature = scan_data["wet_bulb_temperature"]
        land_fraction = scan_data["land_fraction"]
        surface_precip = data["surface_precip"]
        frozen_precip = calculate_frozen_precip(
            wet_bulb_temperature, land_fraction, surface_precip.data
        )
        frozen_precip[surface_precip < 0] = MISSING
        out_data["frozen_precip"] = frozen_precip
        out_data["convective_precip"] = data["convective_precip"]
        out_data["rain_water_path"] = data["rain_water_path"]
        out_data["cloud_water_path"] = data["cloud_water_path"]
        out_data["ice_water_path"] = data["ice_water_path"]
        out_data["most_likely_precip"] = data["most_likely_precip"]
        out_data["surface_precip_1st_tercile"] = data["surface_precip_1st_tercile"]
        out_data["surface_precip_2nd_tercile"] = data["surface_precip_2nd_tercile"]
        out_data["rain_water_content"] = data["rain_water_content"]
        out_data["cloud_water_content"] = data["cloud_water_content"]
        out_data["snow_water_content"] = data["snow_water_content"]
        out_data["latent_heating"] = data["latent_heating"]

        out_data.tofile(file)


###############################################################################
# Running the preprocessor
###############################################################################


def has_preprocessor():
    """
    Function to determine whether a GMI preprocessor is available on the
    system.
    """
    return shutil.which("gprof2020pp_GMI_L1C") is not None


# Dictionary mapping sensor IDs to preprocessor executables.
PREPROCESSOR_EXECUTABLES = {
    "GMI": "gprof2024pp_GMI_L1C",
    "MHS": "gprof2024pp_MHS_L1C",
    "TMIPR": "gprof2021pp_TMI_L1C",
    "TMIPO": "gprof2021pp_TMI_L1C",
    "SSMI": "gprof2021pp_SSMI_L1C",
    "SSMIS": "gprof2024pp_SSMIS_L1C",
    "AMSR2": "gprof2024pp_AMSR2_L1C",
    "AMSRE": "gprof2021pp_AMSRE_L1C",
    "ATMS": "gprof2024pp_ATMS_L1C",
    "TMS": "gprof2023pp_TMS_L1C",
}


# The default preprocessor settings for CSU computers.
PREPROCESSOR_SETTINGS = {
    "prodtype": "CLIMATOLOGY",
    "prepdir": "/qdata2/archive/ERA5/",
    "ancdir": "/qdata1/pbrown/gpm/ppancillary/",
    "ingestdir": "/qdata1/pbrown/gpm/ppingest/",
}



def run_preprocessor(
    l1c_file, sensor, output_file=None, robust=True
):
    """
    Run preprocessor on L1C GMI file.

    Args:
        l1c_file: Path of the L1C file for which to extract the input data
            using the preprocessor.
        sensor: Sensor object representing the sensor for which to run the
            preprocessor.
        output_file: Optional name of an output file. Results will be written
            to a temporary file and the results returned as xarray.Dataset.

    Returns:
        xarray.Dataset containing the retrieval input data for the given L1C
        file or None when the 'output_file' argument is given.
    """
    from gprof_nn.data.l1c import L1CFile

    file = None
    if output_file is None:
        file = tempfile.NamedTemporaryFile()
        output_file = file.name
    try:
        sensor_l1c = L1CFile(l1c_file).sensor
        key = sensor_l1c.sensor_id
        executable = PREPROCESSOR_EXECUTABLES.get(key, None)

        if executable is None:
            raise ValueError(
                f"Could not find preprocessor executable for the key '{key}'."
            )

        jobid = str(os.getpid()) + "_pp"
        args = [jobid] + list(PREPROCESSOR_SETTINGS.values())
        args.insert(2, str(l1c_file))
        args.append(str(output_file))

        LOGGER = logging.getLogger(__name__)
        LOGGER.info(
            "Invoking the preprocesor '%s' using the " "following command: %s",
            executable,
            " ".join([executable] + args)

        )

        subprocess.run([executable] + args, check=True, capture_output=True)
        if file is not None:
            data = PreprocessorFile(output_file).to_xarray_dataset()

    except subprocess.CalledProcessError as error:
        LOGGER.error(
            "Running the preprocessor for file %s failed with the following"
            " error: %s",
            l1c_file,
            error.stdout + error.stderr,
        )
        if robust:
            return None
        else:
            raise error
    finally:
        if file is not None:
            file.close()
    if file is not None:
        return data
    return None


###############################################################################
# Frozen precip
###############################################################################


def calculate_frozen_precip(wet_bulb_temperature, land_fraction, surface_precip):
    """
    Calculate amount of frozen precipitation based on wet-bulb
    temperature lookup table.

    Args:
        wet_bulb_temperature: The wet bulb temperature in K.
        land_fraction: The surface type for each observation.
        surface_precip: The total amount of surface precipitation.

    Returns:
        Array of same shape as 'surface_precip' containing the corresponding,
        estimated amount of frozen precipitation.
    """
    t_wb = np.clip(
        wet_bulb_temperature, TWB_TABLE[0, 0] + 273.15, TWB_TABLE[-1, 0] + 273.15
    )
    f_ocean = TWB_INTERP_OCEAN(t_wb)
    f_land = TWB_INTERP_LAND(t_wb)

    land_pixels = 100.0 <= land_fraction
    frac = 1.0 - np.where(land_pixels, f_land, f_ocean) / 100.0
    return frac * surface_precip


TWB_TABLE = np.array(
    [
        [-6.5, 0.00, 0.00],
        [-6.4, 0.10, 0.30],
        [-6.3, 0.20, 0.60],
        [-6.2, 0.30, 0.90],
        [-6.1, 0.40, 1.20],
        [-6.0, 0.50, 1.50],
        [-5.9, 0.60, 1.80],
        [-5.8, 0.70, 2.10],
        [-5.7, 0.80, 2.40],
        [-5.6, 0.90, 2.70],
        [-5.5, 1.00, 3.00],
        [-5.4, 1.05, 3.10],
        [-5.3, 1.10, 3.20],
        [-5.2, 1.15, 3.30],
        [-5.1, 1.20, 3.40],
        [-5.0, 1.25, 3.50],
        [-4.9, 1.30, 3.60],
        [-4.8, 1.35, 3.70],
        [-4.7, 1.40, 3.80],
        [-4.6, 1.45, 3.90],
        [-4.5, 1.50, 4.00],
        [-4.4, 1.60, 4.10],
        [-4.3, 1.70, 4.20],
        [-4.2, 1.80, 4.30],
        [-4.1, 1.90, 4.40],
        [-4.0, 2.00, 4.50],
        [-3.9, 2.10, 4.60],
        [-3.8, 2.20, 4.70],
        [-3.7, 2.30, 4.80],
        [-3.6, 2.40, 4.90],
        [-3.5, 2.50, 5.00],
        [-3.4, 2.55, 5.20],
        [-3.3, 2.60, 5.40],
        [-3.2, 2.65, 5.60],
        [-3.1, 2.70, 5.80],
        [-3.0, 2.75, 6.00],
        [-2.9, 2.80, 6.20],
        [-2.8, 2.85, 6.40],
        [-2.7, 2.90, 6.60],
        [-2.6, 2.95, 6.80],
        [-2.5, 3.00, 7.00],
        [-2.4, 3.10, 7.10],
        [-2.3, 3.20, 7.20],
        [-2.2, 3.30, 7.30],
        [-2.1, 3.40, 7.40],
        [-2.0, 3.50, 7.50],
        [-1.9, 3.60, 7.60],
        [-1.8, 3.70, 7.70],
        [-1.7, 3.80, 7.80],
        [-1.6, 3.90, 7.90],
        [-1.5, 4.00, 8.00],
        [-1.4, 4.10, 8.20],
        [-1.3, 4.20, 8.40],
        [-1.2, 4.30, 8.60],
        [-1.1, 4.40, 8.80],
        [-1.0, 4.50, 9.00],
        [-0.9, 4.60, 9.20],
        [-0.8, 4.70, 9.40],
        [-0.7, 4.80, 9.60],
        [-0.6, 4.90, 9.80],
        [-0.5, 5.00, 10.00],
        [-0.4, 6.60, 11.60],
        [-0.3, 8.20, 13.20],
        [-0.2, 9.80, 14.80],
        [-0.1, 11.40, 16.40],
        [0.0, 13.00, 18.00],
        [0.1, 14.60, 19.60],
        [0.2, 16.20, 21.20],
        [0.3, 17.80, 22.80],
        [0.4, 19.40, 24.40],
        [0.5, 21.00, 26.00],
        [0.6, 25.80, 29.00],
        [0.7, 30.60, 32.00],
        [0.8, 35.40, 35.00],
        [0.9, 40.20, 38.00],
        [1.0, 45.00, 41.00],
        [1.1, 49.80, 44.00],
        [1.2, 54.60, 47.00],
        [1.3, 59.40, 50.00],
        [1.4, 64.20, 53.00],
        [1.5, 69.00, 56.00],
        [1.6, 71.30, 57.90],
        [1.7, 73.60, 59.80],
        [1.8, 75.90, 61.70],
        [1.9, 78.20, 63.60],
        [2.0, 80.50, 65.50],
        [2.1, 82.80, 67.40],
        [2.2, 85.10, 69.30],
        [2.3, 87.40, 71.20],
        [2.4, 89.70, 73.10],
        [2.5, 92.00, 75.00],
        [2.6, 92.55, 76.30],
        [2.7, 93.10, 77.60],
        [2.8, 93.65, 78.90],
        [2.9, 94.20, 80.20],
        [3.0, 94.75, 81.50],
        [3.1, 95.30, 82.80],
        [3.2, 95.85, 84.10],
        [3.3, 96.40, 85.40],
        [3.4, 96.95, 86.70],
        [3.5, 97.50, 88.00],
        [3.6, 97.60, 88.70],
        [3.7, 97.70, 89.40],
        [3.8, 97.80, 90.10],
        [3.9, 97.90, 90.80],
        [4.0, 98.00, 91.50],
        [4.1, 98.10, 92.20],
        [4.2, 98.20, 92.90],
        [4.3, 98.30, 93.60],
        [4.4, 98.40, 94.30],
        [4.5, 98.50, 95.00],
        [4.6, 98.55, 95.25],
        [4.7, 98.60, 95.50],
        [4.8, 98.65, 95.75],
        [4.9, 98.70, 96.00],
        [5.0, 98.75, 96.25],
        [5.1, 98.80, 96.50],
        [5.2, 98.85, 96.75],
        [5.3, 98.90, 97.00],
        [5.4, 98.95, 97.25],
        [5.5, 99.00, 97.50],
        [5.6, 99.10, 97.75],
        [5.7, 99.20, 98.00],
        [5.8, 99.30, 98.25],
        [5.9, 99.40, 98.50],
        [6.0, 99.50, 98.75],
        [6.1, 99.60, 99.00],
        [6.2, 99.70, 99.25],
        [6.3, 99.80, 99.50],
        [6.4, 99.90, 99.75],
        [6.5, 100.00, 100.00],
    ]
)


TWB_INTERP_LAND = sp.interpolate.interp1d(
    TWB_TABLE[:, 0] + 273.15, TWB_TABLE[:, 1], assume_sorted=True, kind="linear"
)


TWB_INTERP_OCEAN = sp.interpolate.interp1d(
    TWB_TABLE[:, 0] + 273.15, TWB_TABLE[:, 2], assume_sorted=True, kind="linear"
)
