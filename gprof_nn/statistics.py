"""
===================
gprof_nn.evaluation
===================

This module provides functionality to evaluate retrieval results.
"""
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn.data.retrieval import RetrievalFile

class Statistic(ABC):
    """
    Basic interface for statistics calculated across multiple retrieval files.
    """
    @abstractmethod
    def process_data(self, data):
        """
        Process data from a single file.

        Args:
            data: ``xarray.Dataset`` contaning the data from the given file.
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

    def process_data(self, data):
        """
        Process data from a single file.

        Args:
            data: ``xarray.Dataset`` contaning the data from the given file.
        """
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
            if not other.sum is None:
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


class ZonalMean(Statistic):
    """
    Calculates zonal mean on a 1-degree latitude grid.
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
        self.sum = np.zeros(180)
        self.counts = np.zeros(180)

    def process_data(self, data):
        """
        Process data from a single file.

        Args:
            data: ``xarray.Dataset`` contaning the data from the given file.
        """
        v = data[self.variable].data
        valid = np.isfinite(v) * (v > -400.0)
        if v.ndim > data.latitude.data.ndim:
            valid = np.all(valid, axis=-1)

        #v = v[valid, -1]
        v = v[valid]

        lats = data.latitude.data[valid]

        for i in range(self.bins.size - 1):
            l = self.bins[i]
            r = self.bins[i + 1]
            mask = (l <= lats) * (lats < r)
            self.counts[i] += mask.sum()
            self.sum[i] += v[mask].sum()

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        self.sum += other.sum
        self.counts += other.counts

    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        lats = 0.5 * (self.bins[1:] + self.bins[:-1])
        data = xr.Dataset({
            "latitude": (("latitude",), lats),
            "mean": (("latitude",), self.sum / self.counts),
            "counts": (("latitude",),  self.counts)
            })

        destination = Path(destination)
        output_file = destination / "zonal_mean.nc"
        data.to_netcdf(output_file)

class GlobalMean(Statistic):
    """
    Calculates gloab precipitation map.
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
        self.n_lats = 90
        self.lat_bins = np.linspace(-90, 90, self.n_lats + 1)
        self.n_lons = 180
        self.lon_bins = np.linspace(-180, 181, self.n_lons + 1)
        self.sum = np.zeros((self.n_lats, self.n_lons))
        self.counts = np.zeros((self.n_lats, self.n_lons))

    def process_data(self, data):
        """
        Process data from a single file.

        Args:
            data: ``xarray.Dataset`` contaning the data from the given file.
        """
        v = data[self.variable].data
        valid = np.isfinite(v) * (v > -400.0)
        if v.ndim > data.latitude.data.ndim:
            valid = np.all(valid, axis=-1)

        #v = v[valid, -1]
        v = v[valid]

        lats = data.latitude.data[valid]
        lons = data.longitude.data[valid]

        bins = (self.lat_bins, self.lon_bins)
        sum, _, _ = np.histogram2d(lats, lons, bins=bins, weights=v)
        counts, _, _ = np.histogram2d(lats, lons, bins=bins)
        self.sum += sum
        self.counts += counts

    def merge(self, other):
        """
        Merge the data of this statistic with that calculated in a different
        process.
        """
        self.sum += other.sum
        self.counts += other.counts

    def save(self, destination):
        """
        Save results to file in NetCDF format.
        """
        lats = 0.5 * (self.lat_bins[1:] + self.lat_bins[:-1])
        lons = 0.5 * (self.lon_bins[1:] + self.lon_bins[:-1])
        data = xr.Dataset({
            "latitude": (("latitude",), lats),
            "longitude": (("longitude",), lons),
            "mean": (("latitude", "longitude"), self.sum / self.counts),
            "counts": (("latitude", "longitude"),  self.counts)
            })

        destination = Path(destination)
        output_file = destination / "global_mean.nc"
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

    def process_data(self, data):
        """
        Process data from a single file.

        Args:
            data: ``xarray.Dataset`` contaning the data from the given file.
        """
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


def process_files(files, statistics):
    for f in files:
        f = Path(f)
        if ".BIN" in f.suffixes:
            data = RetrievalFile(f).to_xarray_dataset()
        else:
            data = xr.open_dataset(f)

        for s in statistics:
            s.process_data(data)
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
    def __init__(self,
            files,
            statistics):
       self.files = files
       self.statistics = statistics


    def run(self,
            n_workers,
            output_path):
        pool = ProcessPoolExecutor(n_workers)
        batches = list(_split_files(self.files, n_workers))

        tasks = []
        for b in batches:
            tasks.append(pool.submit(process_files, b, self.statistics))

        stats = tasks.pop(0).result()
        for t in tasks:
            for s_old, s_new in zip(stats, t.result()):
                s_old.merge(s_new)

        for s in stats:
            s.save(output_path)






      








