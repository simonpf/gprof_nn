"""
===================
gprof_nn.validation
===================

This module defines functions to collect validation data from MRMS
and Kwajalein co-locations and GPROF retrievals.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import copy
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
    "precip_2nd_tercile",
    "surface_type",
    "airmass_type"
]


LOGGER = logging.getLogger(__name__)


PRECIP_TYPES = {
    "Stratiform, warm": [1.0, 2.0],
    "Stratiform, cool": [10.0],
    "Snow": [3.0, 4.0],
    "Convective": [6.0],
    "Hail": [7.0],
    "Tropical/stratiform mix": [91.0],
    "Tropical/convective rain mix": [96.0]
}


def smooth_reference_field(surface_precip, angles, steps=11):
    """
    Smooth the reference precip field using rotating smoothing kernels.

    Rotating smoothing kernels are employed to account for the conical
    scanning of the sensor.

    Args:
        surface_precip: The refence precip field interpolated to a
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
    d_a = 2 * np.arange(-w_a, w_a + 1e-6).reshape(-1, 1) / fwhm_a
    d_x = 2 * np.arange(-w_x, w_x + 1e-6).reshape(1, -1) / fwhm_x
    k = np.exp(np.log(0.5) *  (d_a ** 2 + d_x **2))
    k = k / k.sum()
    ks = []
    kernel_angles = np.linspace(-70, 70, steps)
    for angle in kernel_angles:
        # Need to inverse angle because y-axis points in
        # opposite direction.
        ks.append(rotate(k, -angle))

    cts = (surface_precip >= 0).astype(np.float32)
    sp = np.nan_to_num(surface_precip.copy(), 0.0)

    fields = []
    for k in ks:
        counts = convolve(cts, k, mode="same", method="direct")
        smoothed = convolve(sp, k, mode="same", method="direct")
        smoothed = smoothed / counts
        smoothed[counts < 0.5] = np.nan
        fields.append(smoothed)
    fields = np.stack(fields, axis=-1)

    weights = calculate_interpolation_weights(angles, kernel_angles)
    smoothed = interpolate(fields, weights)
    smoothed[np.isnan(sp)] = np.nan
    return smoothed


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
            surface_type = data["surfaceTypeIndex"][:]

        dims = ("scans", "pixels")
        dataset = xr.Dataset({
            "latitude": (dims, latitude),
            "longitude": (dims, longitude),
            "surface_precip": (dims, surface_precip),
            "pop": (dims, pop),
            "frozen_precip": (dims, frozen_precip),
            "precip_1st_tercile": (dims, precip_1st_tercile),
            "precip_2nd_tercile": (dims, precip_2nd_tercile),
            "surface_type": (dims, surface_type)
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
            reference_path,
            datasets
    ):
        self.reference_path = Path(reference_path)
        self.datasets = datasets

        self.reference_files = list(self.reference_path.glob("**/*.nc"))


    def process_reference_file(self, filename, output_file):
        """
        Extract data from reference file and extract corresponding data from
        retrieval datasets.

        Results are written in NetCDF4 format to the given output file with
        each dataset stored in its own group.

        Args:
            filename: Path to the reference file containing the extract overpass
                data.
            output_file: Path to the output file to which to write the combined
                results.
        """
        granule = int(str(filename).split("_")[-1].split(".")[0])
        reference_data = xr.load_dataset(filename)

        precip_names = ["surface_precip",
                        "surface_precip_rr",
                        "surface_precip_rp",
                        "surface_precip_rc"]
        for variable in precip_names:
            if variable in reference_data.variables:
                surface_precip = reference_data[variable].data
                angles = reference_data.angles.data
                surface_precip_smoothed = smooth_reference_field(
                    surface_precip,
                    angles
                )
                reference_data[variable + "_avg"] = (
                    ("along_track", "across_track"),
                    surface_precip_smoothed
                )

        if "mask" in reference_data.variables:
            fields = []

            for values in PRECIP_TYPES.values():
                mask = reference_data["mask"]
                mask = np.stack([np.isclose(mask, v) for v in values]).any(axis=0)
                fields.append(smooth_reference_field(
                    mask,
                    angles
                ))
            fields = np.stack(fields)
            classes_smoothed = np.argmax(fields, axis=0)

            no_rain = (fields.max(axis=0) < 0.5)
            classes_smoothed[no_rain] = 0
            mask = reference_data["mask"].data
            classes_smoothed[mask < 0] = -1

            reference_data["mask_avg"] = (
                ("along_track", "across_track"),
                classes_smoothed
            )

        if "raining_fraction" in reference_data.variables:
            rf = reference_data["raining_fraction"].data
            angles = reference_data.angles.data
            rf_smoothed = smooth_reference_field(
                rf,
                angles
            )
            reference_data["raining_fraction_avg"] = (
                ("along_track", "across_track"),
                rf_smoothed
            )

        if "surface_precip_rp" in reference_data.variables:
            reference_data = reference_data.rename({
                "surface_precip_rc": "surface_precip",
                "surface_precip_rc_avg": "surface_precip_avg"
                })

        reference_data.to_netcdf(output_file, group="reference")

        lats = reference_data.latitude.data
        lons = reference_data.longitude.data
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
                    "along_track": (("along_track",), reference_data.along_track.data),
                    "across_track": (("across_track",), reference_data.across_track.data),
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
            except KeyError as error:
                LOGGER.error(
                    "The following error was encountered while processing granule "
                    "'%s' of dataset '%s':\n %s",
                    granule,
                    dataset.group_name,
                    error)


    def run(self, output_path, start=None, end=None, n_processes=4):
        """
        Run collector.

        Args:
            output_path: Directory to which to write the result files.
            start: If given, only files after this data will be considered.
            end: If given, only files before this date will be considered.
            n_processes: The number of processes to use to collect the
                validation data.
        """
        output_path = Path(output_path)
        pool = ProcessPoolExecutor(max_workers=n_processes)

        tasks = []
        files = []

        for reference_file in self.reference_files:

            # Process only files in given range
            parts = reference_file.name.split("_")
            if len(parts) > 4:
                yearmonthday, hourminute = reference_file.name.split("_")[2:4]
                year = yearmonthday[:4]
                month = yearmonthday[4:6]
                day = yearmonthday[6:]
                hour = hourminute[:2]
                minute = hourminute[2:]
            else:
                yearmonthday = reference_file.name.split("_")[2]
                year = yearmonthday[:4]
                month = yearmonthday[4:6]
                day = yearmonthday[6:]
                hour = "00"
                minute = "00"
            date = np.datetime64(f"{year}-{month}-{day}T{hour}:{minute}:00")

            if start is not None and date < start:
                continue
            if end is not None and date >= end:
                continue

            output_filename = output_path / reference_file.name
            tasks.append(pool.submit(
                self.process_reference_file, reference_file, output_filename
            ))
            files.append(reference_file)

        for filename, task in track(list(zip(files, tasks))):
            try:
                task.result()
            except Exception as e:
                LOGGER.error(
                    "The following error was encountered when processing "
                    "file %s: \n %s", filename, e
                )

###############################################################################
# Utilities
###############################################################################

def open_reference_file(reference_path, granule):
    """
    Open the reference data file corresponding to a given granule
    number.

    Args:
        reference_path: Root of the directory tree containing the
            reference data.
        granule: The granule number.

    Return:
        An ``xarray.Dataset`` containing the loaded reference data
    """
    files = Path(reference_path).glob(f"**/*{granule}.nc")
    return xr.load_dataset(next(iter(files)))


def plot_granule(reference_path, granule, datasets, n_cols=3, height=4, width=4):
    """
    Plot overpass over reference data and corresponding retrieval results
    for a given granule.

    Args:
        reference_path: Path to the root of the directory tree containing
            the reference data.
        granule: The granule number as an iteger.
        dataset: List of dataset object providing access to the retrieval
            results.

    Return:
        The matplotlib Figure object containing the plotted data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LogNorm
    from matplotlib import cm
    import cartopy.crs as ccrs

    n = n_cols
    m = ((1 + len(datasets)) // n)
    if ((1 + len(datasets)) % n):
        m = m + 1
    f = plt.figure(figsize=(width * n + 1, height * m))

    gs = GridSpec(m, n + 1, width_ratios=[1.0] * n + [0.05])
    axs = np.array(
        [[f.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
          for j in range(n)]
         for i in range(m)]
    )
    axs = axs.ravel()

    precip_norm = LogNorm(1e-1, 1e2)
    cmap = copy(cm.get_cmap("plasma"))
    cmap.set_under("#FFFFFF10")

    ref_data = open_reference_file(reference_path, granule)
    ax = axs[0]

    lats = ref_data.latitude.data
    lons = ref_data.longitude.data
    xlims = [lons.min(), lons.max()]
    ylims = [lats.min(), lats.max()]

    surface_precip = ref_data["surface_precip"].data
    angles = ref_data.angles.data
    surface_precip_smoothed = smooth_reference_field(
        surface_precip,
        angles
    )
    sp = np.maximum(surface_precip_smoothed, 1e-4)
    ax.stock_img()
    m = ax.pcolormesh(lons, lats, sp, cmap=cmap, norm=precip_norm)
    ax.plot(lons[:, 0], lats[:, 0], ls="--", c="k")
    ax.plot(lons[:, -1], lats[:, -1], ls="--", c="k")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.coastlines()
    ax.set_title("(a) Reference", loc="left")

    for i, dataset in enumerate(datasets):
        ax = axs[i + 1]

        data = dataset.open_granule(granule)
        sp = np.maximum(data.surface_precip.data, 1e-4)
        lats = data.latitude.data
        lons = data.longitude.data

        ax.stock_img()
        ax.pcolormesh(lons, lats, sp, cmap=cmap, norm=precip_norm)
        ax.plot(lons[:, 0], lats[:, 0], ls="--", c="k")
        ax.plot(lons[:, -1], lats[:, -1], ls="--", c="k")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.coastlines()

        ax.set_title(f"({chr(ord('a') + i + 1)}) {dataset.group_name}", loc="left")

    for ax in axs[i + 2:]:
        ax.set_axis_off()

    ax = f.add_subplot(gs[:, -1])
    plt.colorbar(m, label="Surface precip. [mm/h]", cax=ax)
    return f
