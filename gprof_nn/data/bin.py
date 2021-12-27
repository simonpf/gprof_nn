"""
=================
gprof_nn.data.bin
=================

This module contains interfaces to read the GPROF V7 bin files.
"""
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import re

import numpy as np
import pandas as pd
from rich.progress import track
import xarray as xr

from gprof_nn import sensors

LOGGER = logging.getLogger(__name__)
N_LAYERS = 28
GENERIC_HEADER = np.dtype(
    [
        ("satellite_code", "a5"),
        ("sensor", "a5"),
    ]
)

###############################################################################
# Input file.
###############################################################################


class BinFile:
    """
    This class can be used to read a GPROF v7 .bin file.

    Attributes:
        temperature(``float``): The surface temperature corresponding to the
            bin.
        tcwv(``float``): The total precipitable water corresponding to the bin.
        surface_type(``int``): The surface type corresponding to the bin.
        airmass_type(``int``): The airmass type corresponding to the bin.
        header: Structured numpy array containing header data of the file.
        handle: Structured numpy array containing the profile data of the
            file.
    """

    def __init__(self, filename, include_profiles=False):
        """
        Open file.

        Args:
            filename(``str``): File to open
            include_profiles(``bool``): Whether or not to include profiles
                 in the extracted data.
        """
        self.filename = filename
        self.include_profiles = include_profiles

        parts = Path(filename).name[:-4].split("_")
        self.temperature = float(parts[1])
        self.tcwv = float(parts[2])
        self.surface_type = int(parts[-1])

        if len(parts) == 4:
            self.airmass_type = 0
        elif len(parts) == 5:
            self.airmass_type = int(parts[-2])
        else:
            raise Exception(f"Filename {filename} does not match expected format!")

        # Read the header
        header = np.fromfile(self.filename, GENERIC_HEADER, count=1)
        sensor = header["sensor"][0].decode().strip()
        try:
            self.sensor = getattr(sensors, sensor)
        except AttributeError:
            raise Exception(f"The sensor '{sensor}' is not yet supported.")
        self.header = np.fromfile(self.filename, self.sensor.bin_file_header, count=1)
        self.handle = np.fromfile(
            self.filename,
            self.sensor.get_bin_file_record(self.surface_type),
            offset=self.sensor.bin_file_header.itemsize,
        )
        self.n_profiles = self.handle.shape[0]

        np.random.seed(
            np.array(
                [(self.temperature), self.tcwv, self.surface_type, self.airmass_type]
            ).astype(np.int64)
        )
        self.indices = np.random.permutation(self.n_profiles)

    def get_attributes(self):
        """
        Return file header as dictionary of attributes.

        Returns:
            Dictionary containing frequencies and nominal earth incidence
            angles in this file.
        """
        attributes = {
            "frequencies": self.header["frequencies"][0],
            "nominal_eia": self.header["nominal_eia"][0],
            "sensor": self.sensor.name,
        }
        return attributes

    def to_xarray_dataset(self, start=None, end=None):
        """
        Load and return data as ``xarray.Dataset``. If a start and
        and fraction are given, a fixed fraction of radomly sampled elements
        corresponding to the difference of end and start will be returned.
        The order is pseudo random and will be the same for a file with
        given corresponding two-meter temperature, water vapor, surface
        type and airlifting type.

        Args:
            start: Fractional position from which to start reading the data.
            end: Fractional position up to which to read the data.

        Returns:
            ``xarray.Dataset`` containing the data from the bin file.
        """
        if start is None or end is None:
            indices = np.arange(self.n_profiles)
            n_samples = self.n_profiles
        else:
            n_start = int(start * self.n_profiles)
            n_end = int(end * self.n_profiles)
            n_samples = n_end - n_start
            indices = self.indices[n_start:n_end]

        # Parse variable from structured array and sort into
        # dictionary.
        results = {}
        dim_dict = {
            self.sensor.n_chans: "channels",
            N_LAYERS: "layers",
        }
        if self.sensor.n_angles > 1:
            dim_dict[self.sensor.n_angles] = "angles"

        record_type = self.sensor.get_bin_file_record(self.surface_type)
        for key, _, *shape in record_type.descr:
            if key == "scan_time":
                data = self.handle[key]
                date = pd.DataFrame(
                    {
                        "year": data[:, 0],
                        "month": data[:, 1],
                        "day": data[:, 2],
                        "hour": data[:, 3],
                        "minute": data[:, 4],
                        "second": data[:, 5],
                    }
                )
                results[key] = (("samples",), pd.to_datetime(date, errors="coerce"))
            else:
                data = self.handle[key]
                if key in ["brightness_temperatures",
                           "delta_tb"]:
                    if isinstance(self.sensor, sensors.ConstellationScanner):
                        data = data[..., self.sensor.gmi_channels]
                dims = ("samples",)
                if len(data.shape) > 1:
                    dims = dims + tuple([dim_dict[s] for s in data.shape[1:]])
                results[key] = dims, data[indices]

        results["surface_type"] = (
            ("samples",),
            self.surface_type * np.ones(n_samples, dtype=np.int32),
        )
        results["airmass_type"] = (
            ("samples",),
            self.airmass_type * np.ones(n_samples, dtype=np.int32),
        )
        source = 0
        if self.surface_type in [8, 9, 10, 11]:
            source = 1
        elif self.surface_type in [2, 16]:
            source = 2
        results["source"] = (
            ("samples",),
            source * np.ones(n_samples, dtype=np.int32),
        )

        results["tcwv_bin"] = (("samples"), self.tcwv * np.ones(n_samples, dtype=np.float32))
        results["temperature"] = (
            ("samples",),
            self.temperature * np.ones(n_samples, np.float32),
        )

        dataset = xr.Dataset(results)
        dataset.attrs.update(self.get_attributes())
        return dataset


def load_data(filename, start=0.0, end=1.0, include_profiles=False):
    """
    Wrapper function to load data from a file.

    Args:
        filename: The path of the file to load the data from.
        start: Fractional position from which to start reading the data.
        end: Fractional position up to which to read the data.

    Returns:
        Dictionary containing each database variables as numpy array.
    """
    input_file = BinFile(filename, include_profiles=include_profiles)
    return input_file.to_xarray_dataset(start, end)


###############################################################################
# File processor.
###############################################################################

GPM_FILE_REGEXP = re.compile(r"gpm_(\d\d\d)_(\d\d)(_(\d\d))?_(\d\d).bin")


def process_input(input_filename, start=1.0, end=1.0, include_profiles=False):
    """
    Helper function to process and an input '.bin' file from which to extract
    training data.
    """
    data = load_data(input_filename, start, end, include_profiles=include_profiles)
    return data


class FileProcessor:
    """
    Asynchronous file processor to extract profile from GPROF .bin files
    in given a folder.
    """

    def __init__(
        self,
        path,
        t2m_min=227.0,
        t2m_max=307.0,
        tcwv_min=0.0,
        tcwv_max=76.0,
        include_profiles=False,
    ):
        """
        Create file processor to process file in given path.

        Args:
            path: The path containing the files to process.
            t2m_min: The minimum bin surface temperature for which to consider
                bins.
            t2m_max: The maximum bin surface temperature for which to consider
                bins.
            tcwv_min: The minimum bin-tcwv value to consider.
            tcwv_max: The maximum bin-tcwv value to consider.

        """
        self.path = path
        self.include_profiles = include_profiles
        self.files = []

        for input_file in Path(path).iterdir():
            match = re.match(GPM_FILE_REGEXP, input_file.name)
            if match:
                groups = match.groups()
                t2m = float(groups[0])
                tcwv = float(groups[1])
                if (t2m < t2m_min or t2m > t2m_max):
                    continue
                if (tcwv < tcwv_min or tcwv > tcwv_max):
                    continue
                self.files.append(input_file)

    def run_async(self, output_file, start_fraction, end_fraction, n_processes=4):
        """
        Asynchronous processing of files in folder.

        Args:
            output_file(``str``): Filename of the output file.
            start_fraction(``float``): Fractional start value for the observations
                 to extract from each bin file.
            end_fraction(``float``): Fractional end value for the observations
                 to extract from each bin file.
            n_processes(``int``): How many processes to use for the parallel reading
                 of input files.
        """
        pool = ProcessPoolExecutor(max_workers=n_processes)

        tasks = [
            pool.submit(
                process_input,
                f,
                start=start_fraction,
                end=end_fraction,
                include_profiles=self.include_profiles,
            )
            for f in self.files
        ]

        datasets = []
        for task in track(tasks, "Processing files: "):
            datasets.append(task.result())

        dataset = xr.concat(datasets, "samples")
        dataset.to_netcdf(output_file)
