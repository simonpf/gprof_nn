"""
=================
gprof_nn.data.bin
=================

This module contains functions to read and convert the .bin files containing
the non-clustered database observations for GPROF V7.
"""
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import re

from netCDF4 import Dataset
import numpy as np
import xarray as xr
import tqdm.asyncio

LOGGER = logging.getLogger(__name__)

N_LAYERS = 28
N_FREQS = 15
GMI_BIN_HEADER_TYPES = np.dtype(
    [
        ("satellite_code", "a5"),
        ("sensor", "a5"),
        ("frequencies", [(f"f_{i:02}", np.float32) for i in range(N_FREQS)]),
        ("nominal_eia", [(f"f_{i:02}", np.float32) for i in range(N_FREQS)]),
    ]
)


GMI_BIN_RECORD_TYPES = np.dtype(
    [
        ("dataset_number", "i4"),
        ("surface_precip", np.float32),
        ("convective_precip", np.float32),
        (
            "brightness_temperatures",
            [(f"tb_{i:02}", np.float32) for i in range(N_FREQS)],
        ),
        ("delta_tb", [(f"tb_{i:02}", np.float32) for i in range(N_FREQS)]),
        ("rain_water_path", np.float32),
        ("cloud_water_path", np.float32),
        ("ice_water_path", np.float32),
        ("total_column_water_vapor", np.float32),
        ("two_meter_temperature", np.float32),
        ("rain_water_content", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
        ("cloud_water_content", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
        ("snow_water_content", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
        ("latent_heat", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
    ]
)

PROFILE_NAMES = [
    "rain_water_content",
    "cloud_water_content",
    "snow_water_content",
    "latent_heat",
]

###############################################################################
# Input file.
###############################################################################


class GPROFGMIBinFile:
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
            raise Exception(f"Filename {filename} does not match expected format!")

        # Read the header
        self.header = np.fromfile(self.filename, GMI_BIN_HEADER_TYPES, count=1)

        self.handle = np.fromfile(
            self.filename, GMI_BIN_RECORD_TYPES, offset=GMI_BIN_HEADER_TYPES.itemsize
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
            "frequencies": self.header["frequencies"].view("15f4"),
            "nominal_eia": self.header["nominal_eia"].view("15f4"),
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
        else:
            n_start = int(start * self.n_profiles)
            n_end = int(end * self.n_profiles)
            n_samples = n_end - n_start
            indices = self.indices[n_start:n_end]

        results = {}
        for k, t in GMI_BIN_RECORD_TYPES.descr:

            if (not self.include_profiles) and k in PROFILE_NAMES:
                continue
            if type(t) is str:
                view = self.handle[k].view(t)
                results[k] = ("samples",), view[indices]
            else:
                view = self.handle[k].view(f"{len(t)}{t[0][1]}")
                if len(t) == 15:
                    results[k] = (("samples", "channel"), view[indices])
                else:
                    results[k] = (("samples", "layer"), view[indices])

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
    input_file = GPROFGMIBinFile(filename, include_profiles=include_profiles)
    return input_file.to_xarray_dataset(start, end)


###############################################################################
# File processor.
###############################################################################

GPM_FILE_REGEXP = re.compile(r"gpm_(\d\d\d)_(\d\d)(_(\d\d))?_(\d\d).bin")


def process_input(input_filename, start=1.0, end=1.0, include_profiles=False):
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
