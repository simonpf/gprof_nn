"""
===================
gprof_nn.statistics
===================

This module provides a framework to calculate statistics over large
datasets split across multiple files.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing
from pathlib import Path
import re

import numpy as np
import xarray as xr
from rich.progress import track

import gprof_nn.logging
from gprof_nn.definitions import ALL_TARGETS
from gprof_nn.data.retrieval import RetrievalFile
from gprof_nn.data.bin import BinFile
from gprof_nn.data.preprocessor import PreprocessorFile
from gprof_nn.data.training_data import (GPROF_NN_0D_Dataset,
                                         GPROF_NN_2D_Dataset)


LOGGER = logging.getLogger(__name__)

###############################################################################
# Statistics
###############################################################################


def calculate_angle_bins(angles):
    """
    Helper function to create sequence of angle bins from a sensors
    simulated viewing angles.

    Args:
        angles: 1D array containing the viewing angles in ascending
            order.
    """
    angle_bins = np.zeros(angles.size + 1)
    angle_bins[1:-1] = 0.5 * (angles[1:] + angles[:-1])
    angle_bins[0] = 2.0 * angle_bins[1] - angle_bins[2]
    angle_bins[-1] = 2.0 * angle_bins[-2] - angle_bins[-3]
    return angle_bins


def open_file(filename):
    """
    Generic function to open a file and return data as ``xarray.Dataset``.

    Args:
        filename: The path to the file to open.

    Return:
        ``xarray.Dataset`` containing the data in the file.
    """
    filename = Path(filename)
    suffix = filename.suffix
    if suffix == ".nc":
        return xr.open_dataset(filename)
    elif re.match(r"gpm.*\.bin", filename.name):
        file = BinFile(filename, include_profiles=True)
        return file.to_xarray_dataset()
    elif re.match(r".*\.bin(\.gz)?", filename.name.lower()):
        file = RetrievalFile(filename)
        return file.to_xarray_dataset()
    elif suffix == ".pp":
        file = PreprocessorFile(filename)
        return file.to_xarray_dataset()
    raise ValueError(
        f"Could not figure out how handle the file f{filename.name}."
    )


class Statistic(ABC):
    """
    Basic interface for statistics calculated across multiple retrieval files.
    """
    @abstractmethod
    def process_file(self, sensor, filename):
        """
        Process data from a single file.

        Args:
            filename: Path to the file to process.
        """

    @abstractmethod
    def merge(self, statistic):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """

    @abstractmethod
    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """


class ScanPositionMean(Statistic):
    """
    Calculates the mean of a retrieval variable across scan positions.
    """
    def __init__(self,
                 variable="surface_precip"):
        """
        Instantiate scan position mean statistic for given variable.

        Args:
            Name of the retrieval variable for which to compute the
            scan position mean.
        """
        self.variable = variable
        self.sum = None
        self.counts = None

    def process_file(self, sensor, filename):
        """
        Process data from a single file.

        Args:
            filename: Path to the file to process.
        """
        data = open_file(filename)

        v = data[self.variable]
        s = v.fillna(0).sum(dim="scans")
        c = v.scans.size

        if self.sum is None:
            self.sum = s
            self.counts = c
        else:
            self.sum += s
            self.counts += c

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        if self.sum is None:
            self.sum = other.sum
            self.counts = other.counts
        else:
            if other.sum is not None:
                self.sum += other.sum
                self.counts += other.counts

    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        destination = Path(destination)
        output_file = destination / "scan_position_mean.nc"
        if self.sum is not None:
            mean = self.sum / self.counts
            mean.attrs["counts"] = self.counts
            mean.to_netcdf(output_file)


class ZonalDistribution(Statistic):
    """
    Calculates zonal distributions of retrieval targets on a 1-degree
    latitude grid.
    """
    def __init__(self):
        """
        Args:
            Name of the retrieval variable for which to compute the
            scan position mean.
        """
        self.latitude_bins = np.linspace(-90, 90, 181)
        self.has_time = None
        self.sums = None
        self.counts = None
        self.sensor = None

    def _initialize(self, data):
        self.sums = {}
        self.counts = {}
        if "scan_time" in data.variables:
            self.has_time = True
            for k in ALL_TARGETS:
                if k in data.variables:
                    self.sums[k] = np.zeros(
                        (12, self.latitude_bins.size - 1)
                    )
                    self.counts[k] = np.zeros(
                        (12, self.latitude_bins.size - 1)
                    )
        else:
            self.has_time = False
            for k in ALL_TARGETS:
                if k in data.variables:
                    self.sums[k] = np.zeros(self.latitude_bins.size - 1)
                    self.counts[k] = np.zeros(self.latitude_bins.size - 1)


    def process_file(self, sensor, filename):
        """
        Process data from a single file.

        Args:
            filename: Path to the file to process.
        """
        self.sensor = sensor
        data = open_file(filename)
        if self.counts is None:
            self._initialize(data)

        if self.has_time:
            for month in range(12):
                indices = data["scan_time"].dt.month == (month + 1)
                if indices.ndim > 1:
                    indices = indices.all(axis=tuple(np.arange(indices.ndim)[1:]))
                data.latitude.load()

                for k in ALL_TARGETS:
                    if k in self.counts:
                        lats = data.latitude[indices].data
                        data[k].load()
                        v = data[k][indices].data

                        selection = []
                        for i in range(lats.ndim):
                            n_lats = lats.shape[i]
                            n_v = v.shape[i]
                            d_n = (n_lats - n_v) // 2
                            if d_n > 0:
                                selection.append(slice(d_n, -d_n))
                            else:
                                selection.append(slice(0, None))
                        lats = lats[tuple(selection)]

                        if v.ndim > lats.ndim:
                            shape = (lats.shape  +
                                     tuple([1] * (v.ndim - lats.ndim)))
                            lats_v = lats.reshape(shape)
                            lats_v = np.broadcast_to(lats_v, v.shape)
                        else:
                            lats_v = lats
                        weights = np.isfinite(v).astype(np.float32)
                        weights[v < -500] = 0.0
                        cs, _ = np.histogram(lats_v.ravel(),
                                             bins=self.latitude_bins,
                                             weights=weights.ravel())
                        self.counts[k][month] += cs
                        weights = np.nan_to_num(v, nan=0.0)
                        weights[v < -500] = 0.0
                        cs, _ = np.histogram(lats_v.ravel(),
                                             bins=self.latitude_bins,
                                             weights=weights.ravel())
                        self.sums[k][month] += cs
        else:
            data.latitude.load()
            for k in ALL_TARGETS:
                if k in self.counts:
                    v = data[k].data
                    lats = data.latitude.data
                    data[k].load()
                    v = data[k].data

                    selection = []
                    for i in range(lats.ndim):
                        n_lats = lats.shape[i]
                        n_v = v.shape[i]
                        d_n = (n_lats - n_v) // 2
                        if d_n > 0:
                            selection.append(slice(d_n, -d_n))
                        else:
                            selection.append(slice(0, None))
                    lats = lats[tuple(selection)]

                    if v.ndim > lats.ndim:
                        shape = (lats.shape  +
                                 tuple([1] * (v.ndim - lats.ndim)))
                        lats_v = lats.reshape(shape)
                        lats_v = np.broadcast_to(lats_v, v.shape)
                    else:
                        lats_v = lats
                    weights = np.isfinite(v).astype(np.float32)
                    weights[v < -500] = 0.0
                    cs, _ = np.histogram(lats_v.ravel(),
                                         bins=self.latitude_bins,
                                         weights=weights.ravel())
                    self.counts[k] += cs
                    weights = np.nan_to_num(v, nan=0.0)
                    weights[v < -500] = 0.0
                    cs, _ = np.histogram(lats_v.ravel(),
                                         bins=self.latitude_bins,
                                         weights=weights.ravel())
                    self.sums[k] += cs

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        if self.counts is None:
            self.sums = other.sums
            self.counts = other.counts
        elif other.counts is not None:
            for k in self.counts:
                self.sums[k] += other.sums[k]
                self.counts[k] += other.counts[k]

    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        lats = 0.5 * (self.latitude_bins[1:] + self.latitude_bins[:-1])
        data = xr.Dataset({
            "latitude": (("latitude",), lats)
        })
        if self.has_time:
            data["months"] = (("months"), np.arange(1, 13))
            for k in self.counts:
                data[k] = (("months", "latitude"),
                           self.sums[k] / self.counts[k])
                data[k + "_counts"] = (("months", "latitude"),
                                       self.counts[k])
        else:
            for k in self.counts:
                data[k] = (("latitude",),
                           self.sums[k] / self.counts[k])
                data[k + "_counts"] = (("latitude",),
                                       self.counts[k])

        destination = Path(destination)
        output_file = (destination /
                       f"zonal_distribution_{self.sensor.name.lower()}.nc")
        data.to_netcdf(output_file)


class GlobalDistribution(Statistic):
    """
    Calculates global distributions of retrieval targets on a 1-degree
    latitude and longitude grid.
    """
    def __init__(self):
        """
        Args:
            Name of the retrieval variable for which to compute the
            scan position mean.
        """
        self.latitude_bins = np.linspace(-90, 90, 181)
        self.longitude_bins = np.linspace(-180, 180, 361)
        self.has_time = None
        self.counts = None
        self.sums = None
        self.sensor = None

    def _initialize(self, data):
        self.counts = {}
        self.sums = {}
        if "scan_time" in data.variables:
            self.has_time = True
            for k in ALL_TARGETS:
                if k in data.variables:
                    self.sums[k] = np.zeros(
                        (12,
                         self.latitude_bins.size - 1,
                         self.longitude_bins.size - 1)
                    )
                    self.counts[k] = np.zeros(
                        (12,
                         self.latitude_bins.size - 1,
                         self.longitude_bins.size - 1)
                    )
        else:
            self.has_time = False
            for k in ALL_TARGETS:
                if k in data.variables:
                    self.counts[k] = np.zeros((self.latitude_bins.size - 1,
                                               self.longitude_bins.size - 1))
                    self.sums[k] = np.zeros((self.latitude_bins.size - 1,
                                             self.longitude_bins.size - 1))


    def process_file(self, sensor, filename):
        """
        Process data from a single file.

        Args:
            filename: Path to the file to process.
        """
        self.sensor = sensor
        data = open_file(filename)
        if self.counts is None:
            self._initialize(data)

        if self.has_time:
            for month in range(12):
                indices = data["scan_time"].dt.month == (month + 1)
                if indices.ndim > 1:
                    indices = indices.all(axis=tuple(np.arange(indices.ndim)[1:]))
                data.latitude.load()
                data.longitude.load()

                for k in ALL_TARGETS:
                    if k in self.counts:
                        lats = data.latitude[indices].data
                        lons = data.longitude[indices].data
                        data[k].load()
                        v = data[k][indices].data

                        selection = []
                        for i in range(lats.ndim):
                            n_lats = lats.shape[i]
                            n_v = v.shape[i]
                            d_n = (n_lats - n_v) // 2
                            if d_n > 0:
                                selection.append(slice(d_n, -d_n))
                            else:
                                selection.append(slice(0, None))
                        lats = lats[tuple(selection)]
                        lons = lons[tuple(selection)]

                        if v.ndim > lats.ndim:
                            shape = (lats.shape  +
                                     tuple([1] * (v.ndim - lats.ndim)))
                            lats_v = lats.reshape(shape)
                            lats_v = np.broadcast_to(lats_v, v.shape)
                            lons_v = lons.reshape(shape)
                            lons_v = np.broadcast_to(lons_v, v.shape)
                        else:
                            lats_v = lats
                            lons_v = lons
                        weights = np.isfinite(v).astype(np.float32)
                        weights[v < -500] = 0.0
                        cs, _, _ = np.histogram2d(lats_v.ravel(),
                                                  lons_v.ravel(),
                                                  bins=(self.latitude_bins,
                                                        self.longitude_bins),
                                                  weights=weights.ravel())
                        self.counts[k][month] += cs
                        weights = np.nan_to_num(v, nan=0.0)
                        weights[v < -500] = 0.0
                        cs, _, _ = np.histogram2d(lats_v.ravel(),
                                                  lons_v.ravel(),
                                                  bins=(self.latitude_bins,
                                                        self.longitude_bins),
                                                  weights=weights.ravel())
                        self.sums[k][month] += cs
        else:
            data.latitude.load()
            data.longitude.load()
            for k in ALL_TARGETS:
                if k in self.counts:
                    lats = data.latitude.data
                    lons = data.longitude.data
                    data[k].load()
                    v = data[k].data

                    selection = []
                    for i in range(lats.ndim):
                        n_lats = lats.shape[i]
                        n_v = v.shape[i]
                        d_n = (n_lats - n_v) // 2
                        if d_n > 0:
                            selection.append(slice(d_n, -dn))
                        else:
                            selection.append(slice(0, None))
                    lats = lats[tuple(selection)]
                    lons = lons[tuple(selection)]

                    if v.ndim > lats.ndim:
                        shape = (lats.shape  +
                                 tuple([1] * (v.ndim - lats.ndim)))
                        lats_v = lats.reshape(shape)
                        lats_v = np.broadcast_to(lats_v, v.shape)
                        lons_v = lons.reshape(shape)
                        lons_v = np.broadcast_to(lons_v, v.shape)
                    else:
                        lats_v = lats
                        lons_v = lons
                    weights = np.isfinite(v).astype(np.float32)
                    weights[v < -500] = 0.0
                    cs, _, _ = np.histogram2d(lats_v.ravel(),
                                              lons_v.ravel(),
                                              bins=(self.latitude_bins,
                                                    self.longitude_bins),
                                              weights=weights.ravel())
                    self.counts[k] += cs
                    weights = np.nan_to_num(v, nan=0.0)
                    weights[v < -500] = 0.0
                    cs, _, _ = np.histogram2d(lats_v.ravel(),
                                              lons_v.ravel(),
                                              bins=(self.latitude_bins,
                                                    self.longitude_bins),
                                              weights=weights.ravel())
                    self.sums[k] += cs

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        if self.counts is None:
            self.sums = other.sums
            self.counts = other.counts
        elif other.counts is not None:
            for k in self.counts:
                self.sums[k] += other.sums[k]
                self.counts[k] += other.counts[k]

    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        lats = 0.5 * (self.latitude_bins[1:] + self.latitude_bins[:-1])
        lons = 0.5 * (self.longitude_bins[1:] + self.longitude_bins[:-1])
        data = xr.Dataset({
            "latitude": (("latitude",), lats),
            "longitude": (("longitude",), lons)
        })
        if self.has_time:
            data["months"] = (("months"), np.arange(1, 13))
            for k in self.counts:
                data[k] = (("months", "latitude", "longitude"),
                           self.sums[k] / self.counts[k])
                data[k + "_counts"] = (("months", "latitude", "longitude"),
                                       self.counts[k])
        else:
            for k in self.counts:
                data[k] = (("latitude", "longitude"),
                           self.sums[k] / self.counts[k])
                data[k + "_counts"] = (("latitude", "longitude"),
                                       self.counts[k])

        destination = Path(destination)
        output_file = (destination /
                       f"global_distribution_{self.sensor.name.lower()}.nc")
        data.to_netcdf(output_file)


class PositionalZonalMean(Statistic):
    """
    Calculates zonal mean on a 1-degree latitude grid for all scan
    positions.
    """
    def __init__(self,
                 variable="surface_precip"):
        """
        Instantiate scan position mean statistic for given variable.

        Args:
            Name of the retrieval variable for which to compute the
            scan position mean.
        """
        self.variable = variable
        self.bins = np.linspace(-90, 90, 181)
        self.sum = None
        self.counts = None

    def process_file(self, sensor, filename):
        """
        Process data from a single file.

        Args:
            data: ``xarray.Dataset`` contaning the data from the given file.
        """
        data = xr.open_dataset(filename)
        n_pixels = data.pixels.size
        if self.counts is None:
            self.counts = np.zeros((n_pixels, 180))
            self.sum = np.zeros((n_pixels, 180))

        for i in range(n_pixels):
            d_s = data[{"pixels": i}]
            v = d_s[self.variable]
            lats = d_s.latitude.data.ravel()

            indices = np.digitize(lats, self.bins) - 1
            self.sum[i, indices] += v
            self.counts[i, indices] += 1.0

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        if self.sum is None:
            self.sum = other.sum
            self.counts = other.counts
        else:
            if other.sum is not None:
                self.sum = self.sum + other.sum
                self.sum = self.counts + other.counts

    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        lats = 0.5 * (self.bins[1:] + self.bins[:-1])
        data = xr.Dataset({
            "latitude": (("latitude",), lats),
            "mean": (("pixels", "latitude",), self.sum / self.counts),
            "counts": (("pixels", "latitude",),  self.counts)
            })

        destination = Path(destination)
        output_file = destination / "positional_zonal_mean.nc"
        data.to_netcdf(output_file)


LIMITS = {
    "surface_precip": (1e-3, 2e2),
    "convective_precip": (1e-3, 2e2),
    "cloud_water_content": (1e-3, 2e2),
    "rain_water_content": (1e-4, 2e1),
    "snow_water_content": (1e-4, 2e1),
    "latent_heat": (-200, 200),
    "ice_water_path": (1e-3, 2e2),
    "rain_water_path": (1e-3, 2e2),
    "rain_water_path": (1e-3, 2e2),
    "cloud_water_path": (1e-3, 2e2),
}


class TrainingDataStatistics(Statistic):
    """
    Class to calculate relevant statistics from training data files.
    Calculates statistics of brightness temperatures, retrieval targets
    as well as ancillary data.
    """
    def __init__(self,
                 kind="0D"):
        """
        Args:
            kind: The kind of dataset to use to load the data: '0D' or
                '2D'.
        """
        self.kind = kind.upper()

        self.tbs = None
        self.tbs_sim = None
        self.angle_bins = None
        self.tb_bins = np.linspace(0, 400, 401)

        self.targets = {}
        self.bins = {}
        self.t2m = np.zeros((18, 100), dtype=np.float32)
        self.t2m_bins = np.linspace(239.5, 339.5, 101)
        self.tcwv = np.zeros((18, 100), dtype=np.float32)
        self.tcwv_bins = np.linspace(-0.5, 99.5, 101)
        self.st = np.zeros(18, dtype=np.float32)
        self.at = np.zeros(4, dtype=np.float32)

        self.sums_tcwv = {}
        self.counts_tcwv = {}

    def _initialize_data(self,
                         sensor,
                         data):
        """
        Initialize internal storage that depends on data.

        Args:
            sensor: Sensor object identifying the sensor from which the data
                stems.
            data: ``xarray.Dataset`` containing the data to process.
        """
        n_chans = sensor.n_chans
        self.has_angles = sensor.n_angles > 1
        if self.has_angles:
            n_angles = sensor.n_angles
            self.angle_bins = calculate_angle_bins(sensor.angles[::-1])
            self.tbs = np.zeros((18, n_chans, n_angles, self.tb_bins.size - 1),
                                dtype=np.float32)
            self.tbs_tcwv = np.zeros(
                (18, n_chans, n_angles, 100, self.tb_bins.size - 1,),
                dtype=np.float32
            )
        else:
            self.tbs = np.zeros((18, n_chans, self.tb_bins.size - 1),
                                dtype=np.float32)
            self.tbs_tcwv = np.zeros(
                (18, n_chans, 100, self.tb_bins.size - 1,),
                dtype=np.float32
            )

        for k in ALL_TARGETS:
            if k in data.variables:
                # Surface and convective precip have angle depen
                self.targets[k] = np.zeros((18, 200), dtype=np.float32)
                l, h = LIMITS[k]
                if l > 0:
                    self.bins[k] = np.logspace(np.log10(l), np.log10(h), 201)
                else:
                    self.bins[k] = np.linspace(l, h, 201)

                self.sums_tcwv[k] = np.zeros((18, 100))
                self.counts_tcwv[k] = np.zeros((18, 100))

    def process_file(self,
                     sensor,
                     filename):
        """
        Process data from a single file.

        Args:
            filename: The path of the data to process.
        """
        if self.kind == "0D":
            dataset = GPROF_NN_0D_Dataset(filename,
                                          targets=ALL_TARGETS,
                                          normalize=False,
                                          shuffle=False,
                                          augment=False)
            dataset = dataset.to_xarray_dataset()
        else:
            dataset = GPROF_NN_2D_Dataset(filename,
                                          targets=ALL_TARGETS,
                                          normalize=False,
                                          shuffle=False,
                                          augment=False)
            dataset = dataset.to_xarray_dataset()

        self.sensor = sensor

        if self.tbs is None:
            self._initialize_data(sensor, dataset)

        st = dataset["surface_type"]
        sp = dataset["surface_precip"]

        for i in range(18):
            # Select only TBs that are actually used for training.
            i_st = ((st == i + 1) * (sp >= 0)).data
            tcwv = dataset["total_column_water_vapor"].data[i_st]
            tbs = dataset["brightness_temperatures"].data[i_st]

            for i_c in range(sensor.n_chans):
                if self.has_angles:
                    eia = np.abs(dataset["earth_incidence_angle"].data[i_st])
                    for j in range(sensor.n_angles):
                        lower = self.angle_bins[j + 1]
                        upper = self.angle_bins[j]
                        i_a = (eia >= lower) * (eia < upper)


                        cs, _ = np.histogram(
                            tbs[i_a, i_c],
                            bins=self.tb_bins
                        )
                        self.tbs[i, i_c, j] += cs

                        cs, _, _ = np.histogram2d(
                            tcwv[i_a], tbs[i_a, i_c],
                            bins=(self.tcwv_bins, self.tb_bins)
                        )
                        self.tbs_tcwv[i, i_c, j] += cs
                else:
                    cs, _ = np.histogram(tbs[..., i_c], bins=self.tb_bins)
                    self.tbs[i, i_c] += cs
                    cs, _, _ = np.histogram2d(
                        tcwv, tbs[..., i_c],
                        bins=(self.tcwv_bins, self.tb_bins)
                    )
                    self.tbs_tcwv[i, i_c] += cs

            # Retrieval targets
            for k in self.bins:
                v = dataset[k].data[i_st]
                # Surface precip and convective precip must be treated
                # separately.
                cs, _ = np.histogram(v, bins=self.bins[k])
                self.targets[k][i] += cs


                # Conditional mean
                tcwv = dataset["total_column_water_vapor"][i_st].data
                if v.ndim > tcwv.ndim:
                    tcwv = np.repeat(tcwv.reshape(-1, 1), v.shape[-1], axis=-1)

                mask = v > -999
                v = v.copy()
                v[~mask] = 0.0

                self.sums_tcwv[k][i] += np.histogram(
                    tcwv, bins=self.tcwv_bins, weights=v
                )[0]
                self.counts_tcwv[k][i] += np.histogram(
                    tcwv, bins=self.tcwv_bins, weights=mask.astype(np.float64)
                )[0]


            # Ancillary data
            v = dataset["two_meter_temperature"].data[i_st]
            cs, _ = np.histogram(v, bins=self.t2m_bins)
            self.t2m[i] += cs
            v = dataset["total_column_water_vapor"].data[i_st]
            cs, _ = np.histogram(v, bins=self.tcwv_bins)
            self.tcwv[i] += cs


        bins = np.arange(0, 19) + 0.5
        cs, _ = np.histogram(st, bins=bins)
        self.st += cs
        at = dataset["airmass_type"]
        bins = np.arange(-1, 4) + 0.5
        cs, _ = np.histogram(at, bins=bins)
        self.at += cs

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        if self.tbs is None:
            self.tbs = other.tbs
            self.tbs_tcwv = other.tbs_tcwv
            self.targets = other.targets
            self.t2m = other.t2m
            self.tcwv = other.tcwv
            self.st = other.st
            self.at = other.at
            self.sums_tcwv = other.sums_tcwv
            self.counts_tcwv = other.counts_tcwv

        elif other.tbs is not None:
            self.tbs += other.tbs
            self.tbs_tcwv += other.tbs_tcwv
            for k in self.targets:
                self.targets[k] += other.targets[k]
                self.sums_tcwv[k] += other.sums_tcwv[k]
                self.counts_tcwv[k] += other.counts_tcwv[k]
            self.t2m += other.t2m
            self.tcwv += other.tcwv
            self.st += other.st
            self.at += other.at

    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        data = {}
        tb_bins = 0.5 * (self.tb_bins[1:] + self.tb_bins[:-1])
        data["brightness_temperature_bins"] = (
            ("brightness_temperature_bins",),
            tb_bins
        )

        if self.has_angles:
            data["brightness_temperatures"] = (
                ("surface_type_bins",
                 "channels",
                 "angles",
                 "brightness_temperature_bins"),
                self.tbs
            )
            data["brightness_temperatures_tcwv"] = (
                ("surface_type_bins",
                 "channels",
                 "angles",
                 "total_column_water_vapor_bins",
                 "brightness_temperature_bins"),
                self.tbs_tcwv
                )
        else:
            data["brightness_temperatures"] = (
                ("surface_type_bins",
                 "channels",
                 "brightness_temperature_bins"),
                self.tbs
            )
            data["brightness_temperatures_tcwv"] = (
                ("surface_type_bins",
                 "channels",
                 "total_column_water_vapor_bins",
                 "brightness_temperature_bins"),
                self.tbs_tcwv
                )

        for k in self.targets:
            bins = 0.5 * (self.bins[k][1:] + self.bins[k][:-1])
            bin_dim = k + "_bins"
            data[bin_dim] = (bin_dim,), bins
            data[k] = ("surface_type", bin_dim), self.targets[k]

            data[k + "_mean_tcwv"] = (
                ("surface_type", "tcwv_bins"),
                self.sums_tcwv[k] / self.counts_tcwv[k]
            )
            data[k + "_sums_tcwv"] = (
                ("surface_type", "tcwv_bins"),
                self.sums_tcwv[k]
            )
            data[k + "_counts_tcwv"] = (
                ("surface_type", "tcwv_bins"),
                self.counts_tcwv[k]
            )

        bins = 0.5 * (self.t2m_bins[1:] + self.t2m_bins[:-1])
        bin_dim = "two_meter_temperature_bins"
        data[bin_dim] = (bin_dim,), bins
        data["two_meter_temperature"] = ("surface_type", bin_dim), self.t2m

        bins = 0.5 * (self.tcwv_bins[1:] + self.tcwv_bins[:-1])
        bin_dim = "total_column_water_vapor_bins"
        data[bin_dim] = (bin_dim,), bins
        data["total_column_water_vapor"] = ("surface_type", bin_dim), self.tcwv

        data["surface_type"] = ("surface_type_bins",), self.st
        data["airmass_type"] = ("airmass_type_bins"), self.st

        data = xr.Dataset(data)

        destination = Path(destination)
        output_file = (destination /
                       (f"training_data_statistics_{self.sensor.name.lower()}"
                        ".nc"))
        data.to_netcdf(output_file)


class CorrectedObservations(Statistic):
    """
    Class to calculate relevant statistics from training data files.
    Calculates statistics of brightness temperatures, retrieval targets
    as well as ancillary data.
    """
    def __init__(self,
                 equalizer):
        """
        Args:
            conditional: If provided should identify a channel for which
                conditional of all other channels will be calculated.
        """
        self.tbs = None
        self.angle_bins = None
        self.tb_bins = np.linspace(0, 400, 401)
        self.equalizer = equalizer

    def _initialize_data(self,
                         sensor,
                         data):
        """
        Initialize internal storage that depends on data.

        Args:
            sensor: Sensor object identifying the sensor from which the data
                stems.
            data: ``xarray.Dataset`` containing the data to process.
        """
        n_chans = sensor.n_chans
        self.has_angles = sensor.n_angles > 1
        if self.has_angles:
            n_angles = sensor.n_angles
            self.angle_bins = np.zeros(sensor.angles.size + 1)
            self.angle_bins[1:-1] = 0.5 * (sensor.angles[1:] +
                                           sensor.angles[:-1])
            self.angle_bins[0] = 2.0 * self.angle_bins[1] - self.angle_bins[2]
            self.angle_bins[-1] = (2.0 * self.angle_bins[-2] -
                                   self.angle_bins[-3])
            self.tbs = np.zeros((18, n_chans, n_angles, self.tb_bins.size - 1),
                                dtype=np.float32)
        else:
            self.tbs = np.zeros((18, n_chans, self.tb_bins.size - 1),
                                dtype=np.float32)

    def process_file(self,
                     sensor,
                     filename):
        """
        Process data from a single file.

        Args:
            filename: The path of the data to process.
        """
        self.sensor = sensor
        dataset = xr.open_dataset(filename)
        if self.tbs is None:
            self._initialize_data(sensor, dataset)

        x, y = sensor.load_training_data_0d(filename,
                                            [],
                                            False,
                                            np.random.default_rng(),
                                            equalizer=self.equalizer)

        st = np.copy(x[:, -22:-4])
        st[np.all(st == 0, axis=-1), 0] = 1
        st = np.where(st)[1]
        tbs = x[:, :sensor.n_chans]

        for i in range(18):
            # Select only TBs that are actually used for training.
            i_st = (st == i)

            # Sensor with varying EIA (cross track).
            if self.has_angles:
                eia = np.abs(x[:, sensor.n_chans])
                for j in range(sensor.n_angles):
                    lower = self.angle_bins[j + 1]
                    upper = self.angle_bins[j]
                    inds = i_st * (eia >= lower) * (eia < upper)
                    for k in range(sensor.n_chans):
                        cs, _ = np.histogram(tbs[inds, k],
                                             bins=self.tb_bins)
                        self.tbs[i, k, j] += cs
            # Sensor with constant EIA
            else:
                for j in range(sensor.n_chans):
                    cs, _ = np.histogram(tbs[i_st, j], bins=self.tb_bins)
                    self.tbs[i, j] += cs

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        if self.tbs is None:
            self.tbs = other.tbs
        elif other.tbs is not None:
            self.tbs += other.tbs

    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        data = {}
        tb_bins = 0.5 * (self.tb_bins[1:] + self.tb_bins[:-1])
        data["brightness_temperature_bins"] = (
            ("brightness_temperature_bins",),
            tb_bins
        )

        if self.has_angles:
            data["brightness_temperatures"] = (
                ("surface_type_bins",
                 "channels",
                 "angles",
                 "brightness_temperature_bins"),
                self.tbs
            )
        else:
            data["brightness_temperatures"] = (
                ("surface_type_bins",
                 "channels",
                 "brightness_temperature_bins"),
                self.tbs
            )

        data = xr.Dataset(data)
        destination = Path(destination)
        output_file = (destination /
                       (f"corrected_observation_statistics_{self.sensor.name.lower()}"
                        ".nc"))
        data.to_netcdf(output_file)


class BinFileStatistics(Statistic):
    """
    Class to calculate relevant statistics from training data files.
    Calculates statistics of brightness temperatures, retrieval targets
    as well as ancillary data.
    """
    def __init__(self):
        self.targets = {}
        self.sums_t2m = {}
        self.counts_t2m = {}
        self.sums_tcwv = {}
        self.counts_tcwv = {}

    def _initialize_data(self,
                         sensor,
                         data):
        self.tb_bins = np.linspace(0, 400, 401)
        n_chans = sensor.n_chans
        self.has_angles = sensor.n_angles > 1
        if self.has_angles:
            n_angles = sensor.n_angles
            self.angle_bins = np.zeros(sensor.angles.size + 1)
            self.angle_bins[1:-1] = 0.5 * (sensor.angles[1:] +
                                           sensor.angles[:-1])
            self.angle_bins[0] = 2.0 * self.angle_bins[1] - self.angle_bins[2]
            self.angle_bins[-1] = (2.0 * self.angle_bins[-2] -
                                   self.angle_bins[-3])
            self.tbs = np.zeros((18, n_chans, n_angles, self.tb_bins.size - 1),
                                dtype=np.float32)
        else:
            self.tbs = np.zeros((18, n_chans, self.tb_bins.size - 1),
                                dtype=np.float32)

        self.targets = {}
        self.bins = {}
        for k in ALL_TARGETS:
            if k in data.variables:
                # Surface and convective precip have angle depen
                if (k in ["surface_precip", "convective_precip"] and
                    self.has_angles):
                    self.targets[k] = np.zeros((18, n_angles, 200),
                                               dtype=np.float32)
                else:
                    self.targets[k] = np.zeros((18, 200),
                                               dtype=np.float32)
                l, h = LIMITS[k]
                if l > 0:
                    self.bins[k] = np.logspace(np.log10(l), np.log10(h), 201)
                else:
                    self.bins[k] = np.linspace(l, h, 201)

                self.sums_t2m[k] = np.zeros((18, 100))
                self.counts_t2m[k] = np.zeros((18, 100))
                self.sums_tcwv[k] = np.zeros((18, 100))
                self.counts_tcwv[k] = np.zeros((18, 100))

        self.t2m = np.zeros((18, 100), dtype=np.float32)
        self.t2m_bins = np.linspace(239.5, 339.5, 101)
        self.tcwv = np.zeros((18, 100), dtype=np.float32)
        self.tcwv_bins = np.linspace(-0.5, 99.5, 101)
        self.st = np.zeros(18, dtype=np.float32)
        self.at = np.zeros(4, dtype=np.float32)

    def open_file(self,
                  filename):
        """
        Open input file with given name and returns data as 'xarray.Dataset'.

        Args:
            filename: The name of the input file.

        Return:
            'xarray.Dataset' containing the data in the file.
        """
        return BinFile(filename).to_xarray_dataset()

    def process_file(self,
                     sensor,
                     filename):
        """
        Process data from a single file.

        Args:
            filename: The path of the data to process.
        """
        self.sensor = sensor
        dataset = self.open_file(filename)
        dataset = BinFile(filename).to_xarray_dataset()
        if not hasattr(self, "tb_bins"):
            self._initialize_data(sensor, dataset)

        if not dataset.surface_type.size:
            return

        st = dataset["surface_type"][0]

        # Sensor with varying EIA need to be treated separately.
        if self.has_angles:
            if st in [2, 8, 9, 10, 11, 16]:
                for a in range(sensor.n_angles):
                    i_a = dataset["pixel_position"].data == (a + 1)
                    tbs = dataset["brightness_temperatures"].data[i_a]
                    for k in range(sensor.n_chans):
                        cs, _ = np.histogram(tbs[:, k],
                                             bins=self.tb_bins)
                        self.tbs[st - 1, k, a] += cs
            else:
                for a in range(sensor.n_angles):
                    tbs = dataset["brightness_temperatures"].data
                    for k in range(sensor.n_chans):
                        cs, _ = np.histogram(tbs[:, a, k],
                                             bins=self.tb_bins)
                        self.tbs[st - 1, k, a] += cs
        # Sensor with constant EIA.
        else:
            tbs = dataset["brightness_temperatures"]
            for k in range(sensor.n_chans):
                cs, _ = np.histogram(tbs[:, k],
                                     bins=self.tb_bins)
                self.tbs[st - 1, k] += cs

        # Retrieval targets
        for k in self.bins:
            v = dataset[k].data
            # Surface precip and convective precip must be treated
            # separately.
            if ((k in ["surface_precip", "convective_precip"]) and
                self.has_angles):
                if st in [2, 8, 9, 10, 11, 16]:
                    for a in range(sensor.n_angles):
                        i_a = dataset["pixel_position"].data == (a + 1)
                        cs, _ = np.histogram(v[i_a], bins=self.bins[k])
                        self.targets[k][st - 1, a] += cs
                else:
                    for a in range(sensor.n_angles):
                        cs, _ = np.histogram(v[:, a], bins=self.bins[k])
                        self.targets[k][st - 1, a] += cs


            else:
                cs, _ = np.histogram(v, bins=self.bins[k])
                self.targets[k][st - 1] += cs

            t2m = dataset["two_meter_temperature"].data
            tcwv = dataset["total_column_water_vapor"].data
            if v.ndim > t2m.ndim:
                t2m = np.repeat(t2m.reshape(-1, 1), v.shape[1], axis=-1)
                tcwv = np.repeat(tcwv.reshape(-1, 1), v.shape[1], axis=-1)

            inds = v > -999

            # Conditional mean
            self.sums_t2m[k][st - 1] += np.histogram(
                t2m[inds], bins=self.t2m_bins, weights=v[inds]
            )[0]
            self.counts_t2m[k][st - 1] += np.histogram(
                t2m[inds], bins=self.t2m_bins
            )[0]
            self.sums_tcwv[k][st - 1] += np.histogram(
                tcwv[inds], bins=self.tcwv_bins, weights=v[inds]
            )[0]
            self.counts_tcwv[k][st - 1] += np.histogram(
                tcwv[inds], bins=self.tcwv_bins
            )[0]

        # Ancillary data
        v = dataset["two_meter_temperature"].data
        cs, _ = np.histogram(v, bins=self.t2m_bins)
        self.t2m[st - 1] += cs
        v = dataset["total_column_water_vapor"].data
        cs, _ = np.histogram(v, bins=self.tcwv_bins)
        self.tcwv[st - 1] += cs

        bins = np.arange(0, 19) + 0.5
        cs, _ = np.histogram(dataset["surface_type"].data, bins=bins)
        self.st += cs
        at = dataset["airmass_type"].data
        bins = np.arange(-1, 4) + 0.5
        cs, _ = np.histogram(at, bins=bins)
        self.at += cs

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        if not hasattr(other, "tbs"):
            return None
        self.tbs += other.tbs
        for k in self.targets:
            self.targets[k] += other.targets[k]
            self.sums_t2m[k] += other.sums_t2m[k]
            self.counts_t2m[k] += other.counts_t2m[k]
            self.sums_tcwv[k] += other.sums_tcwv[k]
            self.counts_tcwv[k] += other.counts_tcwv[k]
        self.t2m += other.t2m
        self.tcwv += other.tcwv
        self.st += other.st
        self.at += other.at

    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        data = {}
        tb_bins = 0.5 * (self.tb_bins[1:] + self.tb_bins[:-1])
        data["brightness_temperature_bins"] = (
            ("brightness_temperature_bins",),
            tb_bins
        )

        if self.has_angles:
            data["brightness_temperatures"] = (
                ("surface_type_bins",
                 "channels",
                 "angles",
                 "brightness_temperature_bins"),
                self.tbs
            )
        else:
            data["brightness_temperatures"] = (
                ("surface_type_bins",
                 "channels",
                 "brightness_temperature_bins"),
                self.tbs
            )

        for k in self.targets:
            bins = 0.5 * (self.bins[k][1:] + self.bins[k][:-1])
            bin_dim = k + "_bins"
            data[bin_dim] = (bin_dim,), bins
            if (self.has_angles
                and k in ["surface_precip", "convective_precip"]):
                data[k] = ("surface_type", "angles", bin_dim), self.targets[k]
            else:
                data[k] = ("surface_type", bin_dim), self.targets[k]
            data[k + "_mean_tcwv"] = (
                ("surface_type", "tcwv_bins"),
                self.sums_tcwv[k] / self.counts_tcwv[k]
            )
            data[k + "_sums_tcwv"] = (
                ("surface_type", "tcwv_bins"),
                self.sums_tcwv[k]
            )
            data[k + "_counts_tcwv"] = (
                ("surface_type", "tcwv_bins"),
                self.counts_tcwv[k]
            )
            data[k + "_mean_t2m"] = (
                ("surface_type", "t2m_bins"),
                self.sums_t2m[k] / self.counts_t2m[k]
            )

        bins = 0.5 * (self.t2m_bins[1:] + self.t2m_bins[:-1])
        bin_dim = "two_meter_temperature_bins"
        data[bin_dim] = (bin_dim,), bins
        data["two_meter_temperature"] = ("surface_type", bin_dim), self.t2m

        bins = 0.5 * (self.tcwv_bins[1:] + self.tcwv_bins[:-1])
        bin_dim = "total_column_water_vapor_bins"
        data[bin_dim] = (bin_dim,), bins
        data["total_column_water_vapor"] = ("surface_type", bin_dim), self.tcwv

        data["surface_type"] = ("surface_type_bins",), self.st
        data["airmass_type"] = ("airmass_type_bins"), self.st

        data = xr.Dataset(data)

        destination = Path(destination)
        output_file = (destination /
                       (f"bin_file_statistics_{self.sensor.name.lower()}"
                        ".nc"))
        data.to_netcdf(output_file)


class SinFileStatistics(BinFileStatistics):
    """
    Class to calculate relevant statistics from training data files.
    Calculates statistics of brightness temperatures, retrieval targets
    as well as ancillary data.
    """
    def __init__(self):
        super().__init__()
    def open_file(self,
                  filename):
        """
        Open input file with given name and returns data as 'xarray.Dataset'.

        Args:
            filename: The name of the input file.

        Return:
            'xarray.Dataset' containing the data in the file.
        """
        return SimFile(filename).to_xarray_dataset()


class ObservationStatistics(Statistic):
    """
    Class to extract relevant statistics from observation datasets.
    """
    def __init__(self,
                 conditional=None):
        """
        Args:
            conditional: If provided should identify a channel for which
                conditional of all other channels will be calculated.
        """
        self.angle_bins = None
        self.has_angles = None
        self.tbs = None
        self.tb_bins = np.linspace(0, 400, 401)

        self.t2m = np.zeros((18, 200), dtype=np.float32)
        self.t2m_bins = np.linspace(240, 330, 201)
        self.tcwv = np.zeros((18, 100), dtype=np.float32)
        self.tcwv_bins = np.linspace(-0.5, 99.5, 101)
        self.st = np.zeros(18, dtype=np.float32)
        self.at = np.zeros(4, dtype=np.float32)

    def _initialize_data(self,
                         sensor,
                         data):
        n_chans = sensor.n_chans
        self.has_angles = sensor.n_angles > 1
        if self.has_angles:
            n_angles = sensor.n_angles
            self.angle_bins = np.zeros(sensor.angles.size + 1)
            self.angle_bins[1:-1] = 0.5 * (sensor.angles[1:] +
                                           sensor.angles[:-1])
            self.angle_bins[0] = 2.0 * self.angle_bins[1] - self.angle_bins[2]
            self.angle_bins[-1] = (2.0 * self.angle_bins[-2] -
                                   self.angle_bins[-3])
            self.tbs = np.zeros((18, n_chans, n_angles, self.tb_bins.size - 1),
                                dtype=np.float32)
            self.tbs_tcwv = np.zeros(
                (18, n_chans, n_angles, 100, self.tb_bins.size - 1,),
                dtype=np.float32
            )

        else:
            self.tbs = np.zeros((18, n_chans, self.tb_bins.size - 1),
                                dtype=np.float32)
            self.tbs_tcwv = np.zeros(
                (18, n_chans, 100, self.tb_bins.size - 1,),
                dtype=np.float32
            )

    def process_file(self,
                     sensor,
                     filename):
        """
        Process data from a single file.

        Args:
            filename: The path of the data to process.
        """
        self.sensor = sensor
        dataset = open_file(filename)
        if self.tbs is None:
            self._initialize_data(sensor, dataset)

        st = dataset["surface_type"]
        for i in range(18):
            i_st = (st == i + 1).data
            tcwv = dataset["total_column_water_vapor"].data[i_st]

            # Sensor with varying EIA (cross track).
            tbs = (dataset["brightness_temperatures"] .data[i_st.data])
            if self.has_angles:
                eia = np.abs(dataset["earth_incidence_angle"].data[i_st])
                for j in range(sensor.n_angles):
                    lower = self.angle_bins[j + 1]
                    upper = self.angle_bins[j]
                    i_a = (eia >= lower) * (eia < upper)
                    for k in range(sensor.n_chans):
                        cs, _ = np.histogram(tbs[i_a, k],
                                             bins=self.tb_bins)
                        self.tbs[i, k, j] += cs

                        cs, _, _ = np.histogram2d(
                            tcwv[i_a], tbs[i_a, k],
                            bins=(self.tcwv_bins, self.tb_bins)
                        )
                        self.tbs_tcwv[i, k, j] += cs

            # Sensor with constant EIA
            else:
                for j in range(sensor.n_chans):
                    cs, _ = np.histogram(tbs[:, j], bins=self.tb_bins)
                    self.tbs[i, j] += cs
                    cs, _, _ = np.histogram2d(
                        tcwv, tbs[:, j],
                        bins=(self.tcwv_bins, self.tb_bins)
                    )
                    self.tbs_tcwv[i, j] += cs

            # Ancillary data
            v = dataset["two_meter_temperature"].data[i_st]
            cs, _ = np.histogram(v, bins=self.t2m_bins)
            self.t2m[i] += cs
            v = dataset["total_column_water_vapor"].data[i_st]
            cs, _ = np.histogram(v, bins=self.tcwv_bins)
            self.tcwv[i] += cs

        bins = np.arange(0, 19) + 0.5
        cs, _ = np.histogram(st, bins=bins)
        self.st += cs
        at = dataset["airmass_type"]
        bins = np.arange(-1, 4) + 0.5
        cs, _ = np.histogram(at, bins=bins)
        self.at += cs

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        if self.tbs is None:
            self.tbs = other.tbs
            self.tbs_tcwv = other.tbs_tcwv
            self.t2m = other.t2m
            self.tcwv = other.tcwv
            self.st = other.st
            self.at = other.at
        elif other.tbs is not None:
            self.tbs += other.tbs
            self.tbs_tcwv += other.tbs_tcwv
            self.t2m += other.t2m
            self.tcwv += other.tcwv
            self.st += other.st
            self.at += other.at


    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        data = {}
        tb_bins = 0.5 * (self.tb_bins[1:] + self.tb_bins[:-1])
        data["brightness_temperature_bins"] = (
            ("brightness_temperature_bins",),
            tb_bins
        )

        if self.has_angles:
            data["brightness_temperatures"] = (
                ("surface_type_bins",
                 "channels",
                 "angles",
                 "brightness_temperature_bins"),
                self.tbs
            )
            data["brightness_temperatures_tcwv"] = (
                ("surface_type_bins",
                 "channels",
                 "angles",
                 "total_column_water_vapor_bins",
                 "brightness_temperature_bins"),
                self.tbs_tcwv
                )
        else:
            data["brightness_temperatures"] = (
                ("surface_type_bins",
                 "channels",
                 "brightness_temperature_bins"),
                self.tbs
            )
            data["brightness_temperatures_tcwv"] = (
                ("surface_type_bins",
                 "channels",
                 "total_column_water_vapor_bins",
                 "brightness_temperature_bins"),
                self.tbs_tcwv
                )

        bins = 0.5 * (self.t2m_bins[1:] + self.t2m_bins[:-1])
        bin_dim = "two_meter_temperature_bins"
        data[bin_dim] = (bin_dim,), bins
        data["two_meter_temperature"] = ("surface_type", bin_dim), self.t2m

        bins = 0.5 * (self.tcwv_bins[1:] + self.tcwv_bins[:-1])
        bin_dim = "total_column_water_vapor_bins"
        data[bin_dim] = (bin_dim,), bins
        data["total_column_water_vapor"] = ("surface_type", bin_dim), self.tcwv

        data["surface_type"] = ("surface_type_bins",), self.st
        data["airmass_type"] = ("airmass_type_bins"), self.st

        data = xr.Dataset(data)

        destination = Path(destination)
        output_file = (destination /
                       (f"observation_statistics_{self.sensor.name.lower()}"
                        ".nc"))
        data.to_netcdf(output_file)


class RetrievalStatistics(Statistic):
    """
    Class to calculate statistics of retrieval results
    """
    def __init__(self):
        """
        Args:
            conditional: If provided should identify a channel for which
                conditional of all other channels will be calculated.
        """
        self.targets = {}
        self.sums_t2m = {}
        self.counts_t2m = {}
        self.sums_tcwv = {}
        self.counts_tcwv = {}
        self.sums_tcwv_center = {}
        self.counts_tcwv_center = {}

        self.bins = {}
        self.t2m_bins = np.linspace(239.5, 339.5, 101)
        self.tcwv_bins = np.linspace(-0.5, 99.5, 101)

    def _initialize_data(self,
                         sensor,
                         data):
        """
        Initialize internal storage that depends on data.

        Args:
            sensor: Sensor object identifying the sensor from which the data
                stems.
            data: ``xarray.Dataset`` containing the data to process.
        """
        self.has_angles = sensor.n_angles > 1
        n_angles = sensor.n_angles

        for k in ALL_TARGETS:
            if k in data.variables:
                self.targets[k] = np.zeros((18, 200),
                                           dtype=np.float32)
                l, h = LIMITS[k]
                if l > 0:
                    self.bins[k] = np.logspace(np.log10(l), np.log10(h), 201)
                else:
                    self.bins[k] = np.linspace(l, h, 201)

                self.sums_t2m[k] = np.zeros((18, 100))
                self.counts_t2m[k] = np.zeros((18, 100))
                self.sums_tcwv[k] = np.zeros((18, 100))
                self.counts_tcwv[k] = np.zeros((18, 100))
                self.sums_tcwv_center[k] = np.zeros((18, 100))
                self.counts_tcwv_center[k] = np.zeros((18, 100))

    def process_file(self,
                     sensor,
                     filename):
        """
        Process data from a single file.

        Args:
            filename: The path of the data to process.
        """
        self.sensor = sensor
        dataset = xr.open_dataset(filename)
        if not len(self.targets):
            self._initialize_data(sensor, dataset)

        t_s = dataset["surface_type"].data

        for i in range(18):

            # Select obs for given surface type.
            i_s = (t_s == i + 1)

            # Retrieval targets
            for k in self.bins:

                v = dataset[k].data
                if v.ndim <= 2:
                    inds = i_s * (v > -999)
                else:
                    inds = i_s * np.all(v > -999, axis=-1)

                v = v[inds]
                t2m = dataset["two_meter_temperature"].data[inds]
                tcwv = dataset["total_column_water_vapor"].data[inds]

                # Histogram
                cs, _ = np.histogram(v, bins=self.bins[k])
                self.targets[k][i] += cs

                if v.ndim > t2m.ndim:
                    t2m = np.repeat(t2m.reshape(-1, 1), 28, axis=-1)
                    tcwv = np.repeat(tcwv.reshape(-1, 1), 28, axis=-1)

                # Conditional mean
                mask = v > -999
                v = v.copy()
                v[~mask] = 0.0

                self.sums_t2m[k][i] += np.histogram(
                    t2m, bins=self.t2m_bins, weights=v
                )[0]
                self.counts_t2m[k][i] += np.histogram(
                    t2m, bins=self.t2m_bins, weights=mask.astype(np.float64),
                )[0]
                self.sums_tcwv[k][i] += np.histogram(
                    tcwv, bins=self.tcwv_bins, weights=v
                )[0]
                self.counts_tcwv[k][i] += np.histogram(
                    tcwv, bins=self.tcwv_bins, weights=mask.astype(np.float64)
                )[0]


                #
                # Swath center
                #

                i_start = 110 - 12
                i_end = 110 + 13
                v = dataset[k].data[:, i_start:i_end]
                if v.ndim <= 2:
                    inds = i_s[:, i_start:i_end] * (v > -999)
                else:
                    inds = i_s[:, i_start:i_end] * np.all(v > -999, axis=-1)
                v = v[inds]
                mask = v > -999
                v = v.copy()
                v[~mask] = 0.0
                t2m = dataset["two_meter_temperature"]
                t2m = t2m.data[:, i_start:i_end][inds]
                tcwv = dataset["total_column_water_vapor"]
                tcwv = tcwv.data[:, i_start:i_end][inds]
                if v.ndim > t2m.ndim:
                    tcwv = np.repeat(tcwv.reshape(-1, 1), 28, axis=-1)
                self.sums_tcwv_center[k][i] += np.histogram(
                    tcwv, bins=self.tcwv_bins, weights=v
                )[0]
                self.counts_tcwv_center[k][i] += np.histogram(
                    tcwv, bins=self.tcwv_bins, weights=mask.astype(np.float64)
                )[0]

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        if not len(self.targets):
            self.targets = other.targets
            self.bins = other.bins
            self.sums_t2m = other.sums_t2m
            self.counts_t2m = other.counts_t2m
            self.sums_tcwv = other.sums_tcwv
            self.counts_tcwv = other.counts_tcwv

        elif len(self.targets):
            for k in self.targets:
                self.targets[k] += other.targets[k]
                self.sums_t2m[k] += other.sums_t2m[k]
                self.counts_t2m[k] += other.counts_t2m[k]
                self.sums_tcwv[k] += other.sums_tcwv[k]
                self.counts_tcwv[k] += other.counts_tcwv[k]
                self.sums_tcwv_center[k] += other.sums_tcwv_center[k]
                self.counts_tcwv_center[k] += other.counts_tcwv_center[k]

    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        data = {}

        t2m_bins = 0.5 * (self.t2m_bins[1:] + self.t2m_bins[:-1])
        data["two_meter_temperature_bins"] = (
            ("two_meter_temperature_bins",),
            t2m_bins
        )
        tcwv_bins = 0.5 * (self.tcwv_bins[1:] + self.tcwv_bins[:-1])
        data["total_column_water_vapor_bins"] = (
            ("total_column_water_vapor_bins",),
            tcwv_bins
        )

        for k in self.targets:
            bins = 0.5 * (self.bins[k][1:] + self.bins[k][:-1])
            bin_dim = k + "_bins"
            data[bin_dim] = (bin_dim,), bins
            data[k] = ("surface_type", bin_dim), self.targets[k]

            data[k + "_mean_tcwv"] = (
                ("surface_type", "tcwv_bins"),
                self.sums_tcwv[k] / self.counts_tcwv[k]
            )
            data[k + "_sums_tcwv"] = (
                ("surface_type", "tcwv_bins"),
                self.sums_tcwv[k]
            )
            data[k + "_counts_tcwv"] = (
                ("surface_type", "tcwv_bins"),
                self.counts_tcwv[k]
            )
            data[k + "_mean_tcwv_center"] = (
                ("surface_type", "tcwv_bins"),
                self.sums_tcwv_center[k] / self.counts_tcwv_center[k]
            )
            data[k + "_sums_tcwv_center"] = (
                ("surface_type", "tcwv_bins"),
                self.sums_tcwv_center[k]
            )
            data[k + "_counts_tcwv_center"] = (
                ("surface_type", "tcwv_bins"),
                self.counts_tcwv_center[k]
            )
            data[k + "_mean_t2m"] = (
                ("surface_type", "t2m_bins"),
                self.sums_t2m[k] / self.counts_t2m[k]
            )
            data[k + "_sums_t2m"] = (
                ("surface_type", "t2m_bins"),
                self.sums_t2m[k]
            )
            data[k + "_counts_t2m"] = (
                ("surface_type", "t2m_bins"),
                self.counts_t2m[k]
            )

        data = xr.Dataset(data)
        destination = Path(destination)
        output_file = (destination /
                       (f"retrieval_statistics_{self.sensor.name.lower()}"
                        ".nc"))
        data.to_netcdf(output_file)


###############################################################################
# Statistics
###############################################################################


def process_files(sensor,
                  files,
                  statistics,
                  log_queue):
    """
    Helper function to process a list of files in a separate
    process.

    Args:
        files: The list of files to process.
        statistics: List of the statistics to calculate.
        log_queue: The queue to use to log messages to.

    Return:
        List of the calculated statistics.
    """
    gprof_nn.logging.configure_queue_logging(log_queue)
    for f in files:
        for s in statistics:
            try:
                s.process_file(sensor, f)
            except Exception as e:
                LOGGER.error( "Error during processing of %s: %s", f, e)

    return statistics


def _split_files(files, n):
    """
    Split list of 'files' into 'n' parts.
    """
    start = 0
    n_files = len(files)
    for i in range(n):
        n_batch = n_files // n
        if i < n_files % n:
            n_batch += 1
        yield files[start: start + n_batch]
        start = start + n_batch


class StatisticsProcessor:
    """
    Class to manage the distributed calculation of statistics over a
    range of files.

    Attributes:
        files: List of the files to process.
        statistics: Statistics objects defining the statistics to compute.
    """
    def __init__(self,
                 sensor,
                 files,
                 statistics):
        """
        Args:
            sensor: Sensor object defining the sensor to which the files to
                process correspond.
            files: List of files to process.
            statistics: List of ``Statistic`` object to calculate.
        """
        self.sensor = sensor
        self.files = files
        self.statistics = statistics


    def run(self,
            n_workers,
            output_path):
        """
        Start calculation of statistics over given files.

        Args:
            n_workers: The number of process to use to calculate the
                statistics.
            output_path: The path in which to store the results.
        """
        LOGGER.info("Starting processing of %s files.", len(self.files))

        pool = ProcessPoolExecutor(n_workers)
        batches = [[f] for f in np.random.permutation(self.files)]

        log_queue = gprof_nn.logging.get_log_queue()
        tasks = []
        for b in batches:
            tasks.append(pool.submit(process_files,
                                     self.sensor,
                                     b,
                                     self.statistics,
                                     log_queue))
        stats = tasks.pop(0).result()
        for t in track(tasks,
                       description="Processing files:",
                       console=gprof_nn.logging.get_console()):
            gprof_nn.logging.log_messages()
            for s_old, s_new in zip(stats, t.result()):
                s_old.merge(s_new)

        for s in stats:
            s.save(output_path)

        pool.shutdown()
