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

SCAN_HEADER_TYPE = np.dtype(
    [
        ("scan_date", DATE_TYPE),
        ("scan_latitude", "f4"),
        ("scan_longitude", "f4"),
        ("scan_altitude", "f4"),
    ]
)

# Generic orbit that reads the parts the is similar
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
    ]
)


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
        try:
            self._sensor = getattr(sensors, sensor.upper())
        except AttributeError:
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
        offset += i * (SCAN_HEADER_TYPE.itemsize + self.n_pixels * record_type.itemsize)
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
        offset += i * (SCAN_HEADER_TYPE.itemsize + self.n_pixels * record_type.itemsize)
        return np.frombuffer(self.data, SCAN_HEADER_TYPE, count=1, offset=offset)

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
                d[i] = s[k]

        if isinstance(self.sensor, sensors.ConstellationScanner):
            tbs = data["brightness_temperatures"]
            data["brightness_temperatures"] = tbs[..., self.sensor.gmi_channels]
            eia = data["earth_incidence_angle"]
            data["earth_incidence_angle"] = eia[..., self.sensor.gmi_channels]

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
        return dataset

    def write_retrieval_results(self, path, results, ancillary_data=None, suffix=""):
        """
        Write retrieval result to GPROF binary format.

        Args:
            path: The folder to which to write the result. The filename
                  itself follows the GPORF naming scheme.
            results: Dictionary containing the retrieval results.
            ancillary_data: The folder containing the profile clusters.
            suffix: Suffix to append to algorithm name in filename.

        Returns:

            Path object to the created binary file.
        """
        path = Path(path)
        if path.is_dir():
            filename = path / self._get_retrieval_filename(suffix=suffix)
        else:
            filename = path

        LOGGER.info("Writing retrieval results to file '%s'.", str(filename))

        if ancillary_data is not None:
            profiles_raining = ProfileClusters(ancillary_data, True)
            profiles_non_raining = ProfileClusters(ancillary_data, False)
        else:
            profiles_raining = None
            profiles_non_raining = None

        n_scans = results.scans.size
        with open(filename, "wb") as file:
            self._write_retrieval_orbit_header(file)
            self._write_retrieval_profile_info(
                file, profiles_raining, profiles_non_raining
            )
            for i in range(n_scans):
                self._write_retrieval_scan_header(file, i)
                self._write_retrieval_scan(
                    file,
                    i,
                    results,
                    profiles_raining=profiles_raining,
                    profiles_non_raining=profiles_non_raining,
                )
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

        profile_info["n_species"] = N_SPECIES
        profile_info["n_temps"] = N_TEMPERATURES
        profile_info["n_layers"] = N_LAYERS
        profile_info["n_profiles"] = N_PROFILES
        profile_info["species_description"][0][0] = "Rain water content  ".encode()
        profile_info["species_description"][0][1] = "Cloud water content ".encode()
        profile_info["species_description"][0][2] = "Snow water content  ".encode()
        profile_info["species_description"][0][3] = "Graupel/Hail content".encode()
        profile_info["species_description"][0][4] = "Latent heating      ".encode()
        profile_info["height_top_layers"] = np.concatenate(
            [np.linspace(0.5, 10, 20), np.linspace(11, 18, 8)]
        )
        profile_info["temperature"] = np.linspace(270.0, 303.0, 12)

        if (clusters_raining is not None) and (clusters_non_raining is not None):
            profiles_combined = []
            for i, s in enumerate(
                [
                    "rain_water_content",
                    "cloud_water_content",
                    "snow_water_content",
                    "graupel_water_content",
                    "latent_heat",
                ]
            ):
                profiles = [
                    clusters_raining.get_profile_data(s),
                    clusters_non_raining.get_profile_data(s),
                ]
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
        profiles_non_raining=None,
    ):
        """
        Write retrieval data from a full scan to a binary stream.


        Args:
            file: Handle to the binary file to write the data to.
            precip_mean: 1D array containing the mean retrieved precipitation for
                 each pixel.
            precip_1st_tertial: 1D array containing the 1st tertial retrieved from the data.
            precip_2nd_tertial: 1D array containing the 2nd tertial retrieved from the data
            precip_pop: 1D array containing the probability of precipitation in the scan.
        """
        data = retrieval_data[{"scans": scan_index}]
        scan_data = self.get_scan(scan_index)

        out_data = np.recarray(self.n_pixels, dtype=retrieval.DATA_RECORD_TYPES)

        # Pixel status
        ps = out_data["pixel_status"]
        ps[:] = 0
        indices = (
            (scan_data["latitude"] < LAT_MIN)
            + (scan_data["latitude"] > LAT_MAX)
            + (scan_data["longitude"] < LON_MIN)
            + (scan_data["longitude"] > LON_MAX)
        )
        ps[indices] = 1
        indices = np.any(
            (
                (scan_data["brightness_temperatures"] < TB_MIN)
                + (scan_data["brightness_temperatures"] > TB_MAX)
            ),
            axis=-1,
        )
        ps[indices] = 2
        indices = (
            (scan_data["two_meter_temperature"] < 0)
            + (scan_data["total_column_water_vapor"] < 0)
            + (scan_data["surface_type"] < 0)
            + (scan_data["airmass_type"] < 0)
        )
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
            wet_bulb_temperature, surface_type, surface_precip.data
        )
        frozen_precip[surface_precip < 0] = MISSING
        out_data["frozen_precip"] = frozen_precip
        out_data["convective_precip"] = data["convective_precip"]
        out_data["rain_water_path"] = data["rain_water_path"]
        out_data["cloud_water_path"] = data["cloud_water_path"]
        out_data["ice_water_path"] = data["ice_water_path"]
        out_data["most_likely_precip"] = data["most_likely_precip"]
        out_data["precip_1st_tercile"] = data["precip_1st_tercile"]
        out_data["precip_2nd_tercile"] = data["precip_2nd_tercile"]
        if "pixel_status" in data.variables:
            out_data["pixel_status"] = data["pixel_status"]
        if "quality_flag" in data.variables:
            out_data["quality_flag"] = data["quality_flag"]

        if profiles_raining is not None and profiles_non_raining is not None:
            t2m = scan_data["two_meter_temperature"]
            t2m_indices = profiles_raining.get_t2m_indices(t2m)
            out_data["profile_t2m_index"] = t2m_indices + 1

            profile_indices = np.zeros((self.n_pixels, N_SPECIES), dtype=np.float32)
            profile_scales = np.zeros((self.n_pixels, N_SPECIES), dtype=np.float32)
            for i, s in enumerate(
                [
                    "rain_water_content",
                    "cloud_water_content",
                    "snow_water_content",
                    "latent_heat",
                ]
            ):
                invalid = np.all(data[s].data < -500, axis=-1)
                scales_r, indices_r = profiles_raining.get_scales_and_indices(
                    s, t2m, data[s].data
                )
                scales_nr, indices_nr = profiles_non_raining.get_scales_and_indices(
                    s, t2m, data[s].data
                )
                scales = np.where(surface_precip > 0.01, scales_r, scales_nr)
                indices = np.where(surface_precip > 0.01, indices_r, indices_nr + 40)

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
    "GMI": "gprof2020pp_GMI_L1C",
    "MHS": "gprof2020pp_MHS_L1C",
    "TMI": "gprof2021pp_TMI_L1C",
    ("GMI", "MHS"): "gprof2020pp_GMI_MHS_L1C",
    ("GMI", "TMI"): "gprof2020pp_GMI_TMI_L1C"
}


# The default preprocessor settings for CSU computers.
PREPROCESSOR_SETTINGS = {
    "prodtype": "CLIMATOLOGY",
    "prepdir": "/qdata2/archive/ERA5/",
    "ancdir": "/qdata1/pbrown/gpm/ppancillary/",
    "ingestdir": "/qdata1/pbrown/gpm/ppingest/",
}


def get_preprocessor_settings(configuration):

    """
    Return preprocessor settings as list of command line arguments to invoke
    the preprocessor.
    """
    settings = PREPROCESSOR_SETTINGS.copy()
    if configuration != ERA5:
        settings["prodtype"] = "STANDARD"
        settings["prepdir"] = "/qdata1/pbrown/gpm/modelprep/GANALV7/"
    return [s for _, s in settings.items()]


def run_preprocessor(
    l1c_file, sensor, configuration=ERA5, output_file=None, robust=True
):
    """
    Run preprocessor on L1C GMI file.

    Args:
        l1c_file: Path of the L1C file for which to extract the input data
            using the preprocessor.
        sensor: Sensor object representing the sensor for which to run the
            preprocessor.
        configuration: The configuration(ERA5 of GANAL)
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
        if sensor_l1c.sensor_id == sensor.sensor_id:
            key = sensor.sensor_id
        else:
            key = (sensor_l1c.sensor_id, sensor.sensor_id)
        executable = PREPROCESSOR_EXECUTABLES.get(key, None)

        if executable is None:
            raise ValueError(
                f"Could not find preprocessor executable for the key '{key}'."
            )
        LOGGER.info("Using preprocesor '%s'.", executable)

        jobid = str(os.getpid()) + "_pp"
        args = [jobid] + get_preprocessor_settings(configuration)
        args.insert(2, str(l1c_file))
        args.append(output_file)

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


def calculate_frozen_precip(wet_bulb_temperature, surface_type, surface_precip):
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
    t_wb = np.clip(
        wet_bulb_temperature, TWB_TABLE[0, 0] + 273.15, TWB_TABLE[-1, 0] + 273.15
    )
    f_ocean = TWB_INTERP_OCEAN(t_wb)
    f_land = TWB_INTERP_LAND(t_wb)

    ocean_pixels = surface_type == 1
    frac = 1.0 - np.where(ocean_pixels, f_ocean, f_land) / 100.0
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
