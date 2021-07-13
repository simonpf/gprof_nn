"""
=================
gprof_nn.data.bin
=================

This module contains interfaces to read the binned database database
format of GPROF V7.
"""
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import re

import numpy as np
import xarray as xr
import tqdm.asyncio

from gprof_nn import sensors
from gprof_nn.definitions import PROFILE_NAMES

LOGGER = logging.getLogger(__name__)
N_LAYERS = 28
N_FREQS = 15
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
        tpw(``float``): The total precipitable water corresponding to the bin.
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
        self.tpw = float(parts[2])
        self.surface_type = int(parts[-1])

        if len(parts) == 4:
            self.airmass_type = 0
        elif len(parts) == 5:
            self.airmass_type = int(parts[-2])
        else:
            raise Exception(
                f"Filename {filename} does not match expected format!"
            )

        # Read the header
        header = np.fromfile(self.filename, GENERIC_HEADER, count=1)
        sensor = header["sensor"][0].decode().strip()
        try:
            self.sensor = getattr(sensors, sensor)
        except AttributeError:
            raise Exception(f"The sensor '{sensor}' is not yet supported.")
        self.header = np.fromfile(self.filename,
                                  self.sensor.bin_file_header,
                                  count=1)
        self.handle = np.fromfile(
            self.filename,
            self.sensor.bin_file_record,
            offset=self.sensor.bin_file_header.itemsize
        )
        self.n_profiles = self.handle.shape[0]

        np.random.seed(
            np.array(
                [(self.temperature), self.tpw, self.surface_type, self.airmass_type]
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
            "frequencies": self.header["frequencies"],
            "nominal_eia": self.header["nominal_eia"],
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
            self.sensor.n_freqs: "channels",
            N_LAYERS: "layers"
        }
        if hasattr(self.sensor, "n_angles"):
            dim_dict[self.sensor.n_angles] = "angles"

        for k, t, *shape in self.sensor.bin_file_record.descr:
            dims = ("samples",)
            if shape:
                dims = dims + tuple([dim_dict[s] for s in shape[0]])
            results[k] = dims, self.handle[k][indices]

        results["surface_type"] = (
            ("samples",),
            self.surface_type * np.ones(n_samples, dtype=np.int),
        )
        results["airmass_type"] = (
            ("samples",),
            self.airmass_type * np.ones(n_samples, dtype=np.int),
        )
        results["tpw"] = (("samples"), self.tpw * np.ones(n_samples, dtype=np.float))
        results["temperature"] = (("samples",), self.temperature * np.ones(n_samples))

        return xr.Dataset(results)


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
    data = load_data(input_filename,
                     start,
                     end,
                     include_profiles=include_profiles)
    return data


class FileProcessor:
    """
    Asynchronous file processor to extract profile from GPROF .bin files
    in given a folder.
    """

    def __init__(
        self,
        path,
        st_min=227.0,
        st_max=307.0,
        tpw_min=0.0,
        tpw_max=76.0,
        include_profiles=False,
    ):
        """
        Create file processor to process file in given path.

        Args:
            path: The path containing the files to process.
            st_min: The minimum bin surface temperature for which to consider
                bins.
            st_max: The maximum bin surface temperature for which to consider
                bins.
            tpw_min: The minimum bin-tpw value to consider.
            tpw_max: The maximum bin-tpw value to consider.

        """
        self.path = path
        self.include_profiles = include_profiles
        self.files = []

        for f in Path(path).iterdir():
            match = re.match(GPM_FILE_REGEXP, f.name)
            if match:
                groups = match.groups()
                t = float(groups[0])
                tpw = float(groups[1])
                if t < st_min or t > st_max or tpw < tpw_min or tpw > tpw_max:
                    continue
                self.files.append(f)

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
        loop = asyncio.new_event_loop()

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
        for t in tqdm.tqdm(tasks):
            datasets.append(t.result())

        dataset = xr.concat(datasets, "samples")
        dataset.to_netcdf(output_file)
