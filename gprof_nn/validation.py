"""
===================
gprof_nn.validation
===================

This module defines functions to collect validation data from MRMS
co-locations and GPROF retrievals.
"""
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path

from h5py import File
import numpy as np
from pyresample import geometry, kd_tree
import xarray as xr
from rich.progress import track
from scipy.ndimage import rotate
from scipy.signal import convolve

from gprof_nn.data.training_data import decompress_and_load
from gprof_nn.data.retrieval import RetrievalFile
from gprof_nn.data.sim import SimFile
from gprof_nn.utils import calculate_interpolation_weights, interpolate


VALIDATION_VARIABLES = [
    "surface_precip",
    "frozen_precip",
    "pop",
    "precip_1st_tercile",
    "precip_2nd_tercile"
]


LOGGER = logging.getLogger(__name__)


def smooth_mrms_field(mrms_surface_precip, angles, steps=11):
    """
    Smooth the MRMS precip field using rotating smoothing kernels.

    Rotating smoothing kernels are employed to account for the conical
    scanning of the sensor.

    Args:
        mrms_surface_precip: The MRMS precip field interpolated to a
            5km x 5km grid following the satellite swath.
        angles: The corresponding orientation angles of the footprints.
        steps: The number of kernels to use to approximate the changing
            viewing angles across the scan.
    """
    angles[angles > 90] -= 180
    angles[angles < -90] += 180

    #
    # Calculate smoothing kernels for different angles.
    #

    fwhm_a = 18.0 / 5.0
    w_a = int(fwhm_a) + 1
    fwhm_x = 11.0 / 5.0
    w_x = int(fwhm_x) + 1
    d_a = np.arange(-w_a, w_a + 1e-6).reshape(-1, 1) / fwhm_a
    d_x = np.arange(-w_x, w_x + 1e-6).reshape(1, -1) / fwhm_x
    k = np.exp(np.log(0.5) * 2.0 * (d_a ** 2 + d_x **2))
    k = k / k.sum()
    ks = []
    kernel_angles = np.linspace(-70, 70, steps)
    for angle in kernel_angles:
        # Need to inverse angle because y-axis points in
        # opposite direction.
        ks.append(rotate(k, -angle))

    cts = (mrms_surface_precip >= 0).astype(np.float32)
    sp = np.nan_to_num(mrms_surface_precip, 0.0)

    fields = []
    for k in ks:
        counts = convolve(cts, k, mode="same", method="direct")
        smoothed = convolve(sp, k, mode="same", method="direct")
        smoothed = smoothed / counts
        smoothed[counts < 1e-3] = 0.0
        fields.append(smoothed)
    fields = np.stack(fields, axis=-1)

    weights = calculate_interpolation_weights(angles, kernel_angles)
    return interpolate(fields, weights)


class GPROFNN1DResults:
    """
    Data interface class to collect results from GPROF-NN retrieval.
    """
    def __init__(self, path):
        """
        Args:
            path: Path pointing to the root of the directory tree containing
                the retrieval results.
        """
        self.path = Path(path)
        files = self.path.glob("**/*.bin")
        self.granules = {}
        for filename in files:
            try:
                granule = int(str(filename).split("_")[-1].split(".")[0])
                self.granules[granule] = filename
            except ValueError:
                pass
            except IndexError:
                pass

    def __len__(self):
        return len(self.granules)

    @property
    def group_name(self):
        return "gprof_nn_1d"

    def open_granule(self, granule):
        data = RetrievalFile(self.granules[granule]).to_xarray_dataset()
        print(self.granules[granule])
        return data[VALIDATION_VARIABLES + ["latitude", "longitude"]]


class GPROFNN3DResults(GPROFNN1DResults):
    """
    Data interface class to collect results from GPROF-NN retrieval.
    """
    def __init__(self, path, name=None):
        """
        Args:
            path: Path pointing to the root of the directory tree containing
                the retrieval results.
        """
        super().__init__(path)
        self.name = name

    @property
    def group_name(self):
        if self.name is None:
            return "gprof_nn_3d"
        else:
            return self.name


class GPROFResults:
    """
    Data interface class to collect results from GPROF V7 result files.
    """
    def __init__(self, path):
        """
        Args:
            path: Path pointing to the root of the directory tree containing
                the retrieval results.
        """
        self.path = Path(path)
        files = self.path.glob("**/*.nc")
        self.granules = {}
        for filename in files:
            try:
                granule = int(str(filename.name).split("_")[-1].split(".")[0])
                self.granules[granule] = filename
            except ValueError:
                pass
            except IndexError:
                pass

    def __len__(self):
        return len(self.granules)

    @property
    def group_name(self):
        return "gprof_v7"

    def open_granule(self, granule):
        dataset = xr.load_dataset(self.granules[granule])
        return dataset


class GPROFLegacyResults:
    """
    Data interface class to collect results from GPROF V6 result files.
    """
    def __init__(self, path):
        """
        Args:
            path: Path pointing to the root of the directory tree containing
                the retrieval results.
        """
        self.path = Path(path)
        files = self.path.glob("**/*.HDF5")
        self.granules = {}
        for filename in files:
            try:
                granule = int(str(filename.name).split(".")[-3])
                self.granules[granule] = filename
            except ValueError:
                pass
            except IndexError:
                pass

    def __len__(self):
        return len(self.granules)

    @property
    def group_name(self):
        return "gprof_v5"

    def open_granule(self, granule):
        with File(str(self.granules[granule]), "r") as data:
            data = data["S1"]
            latitude = data["Latitude"][:]
            longitude = data["Longitude"][:]
            surface_precip = data["surfacePrecipitation"][:]
            pop = data["probabilityOfPrecip"][:]
            frozen_precip = data["frozenPrecipitation"][:]
            precip_1st_tercile = data["precip1stTertial"][:]
            precip_2nd_tercile = data["precip2ndTertial"][:]

        dims = ("scans", "pixels")
        dataset = xr.Dataset({
            "latitude": (dims, latitude),
            "longitude": (dims, longitude),
            "surface_precip": (dims, surface_precip),
            "pop": (dims, pop),
            "frozen_precip": (dims, frozen_precip),
            "precip_1st_tercile": (dims, precip_1st_tercile),
            "precip_2nd_tercile": (dims, precip_2nd_tercile)
        })
        return dataset


class GPROFCMBResults(GPROFLegacyResults):
    """
    Data interface class to collect results from GPROF V6 result files.
    """
    def __init__(self, path):
        """
        Args:
            path: Path pointing to the root of the directory tree containing
                the retrieval results.
        """
        self.path = Path(path)
        files = self.path.glob("**/*.HDF5")
        self.granules = {}
        for filename in files:
            try:
                granule = int(str(filename.name).split(".")[-3])
                self.granules[granule] = filename
            except ValueError:
                pass
            except IndexError:
                pass

    def __len__(self):
        return len(self.granules)

    @property
    def group_name(self):
        return "cmb"

    def open_granule(self, granule):
        with File(str(self.granules[granule]), "r") as data:
            data = data["NS"]
            latitude = data["Latitude"][:]
            longitude = data["Longitude"][:]
            surface_precip = data["surfPrecipTotRate"][:]

        dims = ("scans", "pixels")
        dataset = xr.Dataset({
            "latitude": (dims, latitude),
            "longitude": (dims, longitude),
            "surface_precip": (dims, surface_precip),
        })
        return dataset


class SimulatorFiles():
    """
    Interface class to read data from simulator files.
    """
    def __init__(self, path):
        """
        Args:
            path: Base path containing the simulator files.
        """
        self.path = Path(path)
        files = self.path.glob("**/*.sim")
        self.granules = {}
        for filename in files:
            try:
                granule = int(str(filename.name).split(".")[-2])
                self.granules[granule] = filename
            except ValueError:
                pass
            except IndexError:
                pass

    def __len__(self):
        return len(self.granules)

    @property
    def group_name(self):
        return "simulator"

    def open_granule(self, granule):
        return SimFile(self.granules[granule]).to_xarray_dataset()


class ResultCollector:
    """
    This class collects validation results from MRMS files and combines
    them with retrieval results from a selection of datasets.
    """
    def __init__(
            self,
            mrms_path,
            datasets
    ):
        self.mrms_path = Path(mrms_path)
        self.datasets = datasets

        self.mrms_files = list(self.mrms_path.glob("**/*.nc"))


    def process_mrms_file(self, filename, output_file):
        """
        Extract data from MRMS file and extract corresponding data from
        retrieval datasets.

        Results are written in NetCDF4 format to the given output file with
        each dataset stored in its own group.

        Args:
            filename: Path to the MRMS file containing the extract overpass
                data.
            output_file: Path to the output file to which to write the combined
                results.
        """
        granule = int(str(filename).split("_")[-1].split(".")[0])
        mrms_data = xr.load_dataset(filename)

        surface_precip = mrms_data.surface_precip.data
        angles = mrms_data.angles.data
        surface_precip_smoothed = smooth_mrms_field(
            surface_precip,
            angles
        )
        mrms_data["surface_precip_avg"] = (
            ("along_track", "across_track"),
            surface_precip_smoothed
        )
        mrms_data.to_netcdf(output_file, group="mrms")

        lats = mrms_data.latitude.data
        lons = mrms_data.longitude.data
        result_grid = geometry.SwathDefinition(lats=lats, lons=lons)

        for dataset in self.datasets:
            try:
                data = dataset.open_granule(granule)
                data_grid = geometry.SwathDefinition(
                    lats=data.latitude.data,
                    lons=data.longitude.data
                )

                resampling_info = kd_tree.get_neighbour_info(
                    data_grid,
                    result_grid,
                    10e3,
                    neighbours=1
                )
                valid_inputs, valid_outputs, indices, distances = resampling_info

                data_r = xr.Dataset({
                    "along_track": (("along_track",), mrms_data.along_track.data),
                    "across_track": (("across_track",), mrms_data.across_track.data),
                })

                for variable in VALIDATION_VARIABLES:
                    if variable in data.variables:
                        missing = np.nan
                        if data[variable].dtype not in [np.float32, np.float64]:
                            missing = -999
                        resampled = kd_tree.get_sample_from_neighbour_info(
                            'nn', result_grid.shape, data[variable].data,
                            valid_inputs, valid_outputs, indices,
                            fill_value=missing
                        )
                        data_r[variable] = (("along_track", "across_track"), resampled)

                data_r.to_netcdf(output_file, group=dataset.group_name, mode="a")
            except KeyError:
                pass


    def run(self, output_path, start=None, end=None):
        """
        Run collector.

        Args:
            output_path: Directory to which to write the result files.
            start: If given, only files after this data will be considered.
            end: If given, only files before this date will be considered.
        """
        output_path = Path(output_path)
        pool = ProcessPoolExecutor(max_workers=4)

        tasks = []
        files = []

        for mrms_file in self.mrms_files:

            # Process only files in given range
            yearmonthday, hourminute = mrms_file.name.split("_")[2:4]
            year = yearmonthday[:4]
            month = yearmonthday[4:6]
            day = yearmonthday[6:]
            hour = hourminute[:2]
            minute = hourminute[2:]
            date = np.datetime64(f"{year}-{month}-{day}T{hour}:{minute}:00")
            if start is not None and date < start:
                continue
            if end is not None and date >= end:
                continue

            output_filename = output_path / mrms_file.name
            tasks.append(pool.submit(
                self.process_mrms_file, mrms_file, output_filename
            ))
            files.append(mrms_file)

        for filename, task in track(list(zip(files, tasks))):
            try:
                task.result()
            except Exception as e:
                LOGGER.error(
                    "The following error was encountered when processing "
                    "file %s: \n %s", filename, e
                )
