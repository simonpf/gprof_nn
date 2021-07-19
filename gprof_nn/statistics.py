"""
===================
gprof_nn.statistics
===================

This module provides a framework to calculate statistics over large
datasets split across multiple files.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

import numpy as np
import xarray as xr

from gprof_nn.definitions import ALL_TARGETS
from gprof_nn.data.retrieval import RetrievalFile
from gprof_nn.data.bin import BinFile

###############################################################################
# Statistics
###############################################################################


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
        return xr.open_datset(filename)
    elif re.match("gpm.*\.bin", filename.name):
        file = BinFile(filename, include_profiles=True)
        return file.to_xarray_datset()
    elif re.match(".*\.bin{\.gz}+", filename.name.lower()):
        file = RetrievalFile(filename, include_profiles=True)
        return file.to_xarray_datset()
    raise ValueError(
        "Could not figure out how handle the file f{filename.name}."
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

    def process_file(self, filename):
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
            if ther.sum is not None:
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

    def process_file(self, filename):
        """
        Process data from a single file.

        Args:
            filename: Path to the file to process.
        """
        data = open_file(filename)

        v = data[self.variable].data
        valid = np.isfinite(v) * (v > -400.0)
        if v.ndim > data.latitude.data.ndim:
            valid = np.all(valid, axis=-1)

        v = v[valid]
        lats = data.latitude.data[valid]
        lons = data.longitude.data[valid]
        shape = lats.shape + tuple([1] * (v.ndim - lats.ndim))
        lats = np.broadcast_to(lats, shape)
        lons = np.broadcast_to(lons, shape)

        lats = data.latitude.data[valid]

        for i in range(self.bins.size - 1):
            lower = self.bins[i]
            upper = self.bins[i + 1]
            mask = (lower <= lats) * (lats < upper)
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

    def process_file(self, filename):
        """
        Process data from a single file.

        Args:
            data: ``xarray.Dataset`` contaning the data from the given file.
        """
        data = xr.open_dataset(filename)

        v = data[self.variable].data
        valid = np.isfinite(v) * (v > -400.0)
        if v.ndim > data.latitude.data.ndim:
            valid = np.all(valid, axis=-1)

        v = v[valid]
        lats = data.latitude.data[valid]
        lons = data.longitude.data[valid]
        shape = lats.shape + tuple([1] * (v.ndim - lats.ndim))
        lats = np.broadcast_to(lats, shape)
        lons = np.broadcast_to(lons, shape)

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

    def process_file(self, filename):
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
    def __init__(self):
        pass

    def _initialize_data(self,
                         sensor,
                         data):
        n_freqs = sensor.n_freqs
        self.has_angles = hasattr(sensor, "angles")
        if self.has_angles:
            n_angles = sensor.n_angles
            self.angle_bins = np.zeros(sensor.angles.size + 1)
            self.angle_bins[1:-1] = 0.5 * (sensor.angles[1:] +
                                           sensor.angles[:-1])
            self.angle_bins[0] = 2.0 * self.angle_bins[1] - self.angle_bins[2]
            self.angle_bins[-1] = (2.0 * self.angle_bins[-2] -
                                   self.angle_bins[-3])
            self.tbs = np.zeros((18, n_freqs, n_angles, 300),
                                dtype=np.float32)
        else:
            self.tbs = np.zeros((18, n_freqs, 300),
                                dtype=np.float32)
        self.tb_bins = np.linspace(100, 400, 301)

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

        self.t2m = np.zeros((18, 200), dtype=np.float32)
        self.t2m_bins = np.linspace(240, 330, 201)
        self.tcwv = np.zeros((18, 200), dtype=np.float32)
        self.tcwv_bins = np.linspace(0, 100, 201)
        self.st = np.zeros(18, dtype=np.float32)
        self.at = np.zeros(4, dtype=np.float32)

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
        if not hasattr(self, "tb_bins"):
            self._initialize_data(sensor, dataset)

        st = dataset["surface_type"]
        for i in range(18):
            i_st = (st == i + 1).data

            # Sensor with varying EIA (cross track).
            tbs = (dataset["brightness_temperatures"] .data[i_st.data])
            if self.has_angles:
                eia = dataset["earth_incidence_angle"].data[i_st, 0]

                # For samples with real observations (snow + sea ice)
                # observations must be selected based on earth incidence
                # angle variable.
                if (i + 1) in [2, 8, 9, 10, 11, 16]:
                    for j in range(sensor.n_angles):
                        lower = self.angle_bins[j + 1]
                        upper = self.angle_bins[j]
                        i_a = (eia >= lower) * (eia < upper)
                        for k in range(sensor.n_freqs):
                            cs, _ = np.histogram(tbs[i_a, k],
                                                 bins=self.tb_bins)
                            self.tbs[i, k, j] += cs
                # For samples with simulated observations, values are already
                # binned but bias correction must be applied.
                else:
                    tbs = (dataset["simulated_brightness_temperatures"]
                           .data[i_st[:, :, 90:-90].data])
                    b = (dataset["brightness_temperature_biases"]
                         .data[i_st[:, :, 90:-90]])
                    for j in range(sensor.n_angles):
                        for k in range(sensor.n_freqs):
                            x = tbs[:, j, k] - b[:, k]
                            cs, _ = np.histogram(x, bins=self.tb_bins)
                            self.tbs[i, k, j] += cs
            # Sensor with constant EIA
            else:
                for j in range(sensor.n_freqs):
                    cs, _ = np.histogram(tbs[:, j], bins=self.tb_bins)
                    self.tbs[i, j] += cs

            # Retrieval targets
            for k in self.bins:
                v = dataset[k].data
                if v.shape[2] < i_st.shape[2]:
                    inds = i_st[:, :, 90:-90]
                else:
                    inds = i_st
                v = v[inds]
                # Surface precip and convective precip must be treated
                # separately.
                if ((k in ["surface_precip", "convective_precip"]) and
                    self.has_angles):
                    if (i + 1) in [2, 8, 9, 10, 11, 16]:
                        for j in range(sensor.n_angles):
                            lower = self.angle_bins[j + 1]
                            upper = self.angle_bins[j]
                            i_a = (eia >= lower) * (eia < upper)
                            cs, _ = np.histogram(v[i_a, 0],
                                                 bins=self.bins[k])
                            self.targets[k][i, j] += cs
                    else:
                        for j in range(sensor.n_angles):
                            cs, _ = np.histogram(v[:, j], bins=self.bins[k])
                            self.targets[k][i, j] += cs
                else:
                    cs, _ = np.histogram(v, bins=self.bins[k])
                    self.targets[k][i] += cs

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
        if not hasattr(other, "tbs"):
            return None
        self.tbs += other.tbs
        for k in self.targets:
            self.targets[k] += other.targets[k]
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


class BinFileStatistics(Statistic):
    """
    Class to calculate relevant statistics from training data files.
    Calculates statistics of brightness temperatures, retrieval targets
    as well as ancillary data.
    """
    def __init__(self):
        pass

    def _initialize_data(self,
                         sensor,
                         data):
        n_freqs = sensor.n_freqs
        self.has_angles = hasattr(sensor, "angles")
        if self.has_angles:
            n_angles = sensor.n_angles
            self.angle_bins = np.zeros(sensor.angles.size + 1)
            self.angle_bins[1:-1] = 0.5 * (sensor.angles[1:] +
                                           sensor.angles[:-1])
            self.angle_bins[0] = 2.0 * self.angle_bins[1] - self.angle_bins[2]
            self.angle_bins[-1] = (2.0 * self.angle_bins[-2] -
                                   self.angle_bins[-3])
            self.tbs = np.zeros((18, n_freqs, n_angles, 300),
                                dtype=np.float32)
        else:
            self.tbs = np.zeros((18, n_freqs, 300),
                                dtype=np.float32)
        self.tb_bins = np.linspace(100, 400, 301)

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

        self.t2m = np.zeros((18, 200), dtype=np.float32)
        self.t2m_bins = np.linspace(240, 330, 201)
        self.tcwv = np.zeros((18, 200), dtype=np.float32)
        self.tcwv_bins = np.linspace(0, 100, 201)
        self.st = np.zeros(18, dtype=np.float32)
        self.at = np.zeros(4, dtype=np.float32)

    def process_file(self,
                     sensor,
                     filename):
        """
        Process data from a single file.

        Args:
            filename: The path of the data to process.
        """
        self.sensor = sensor
        dataset = BinFile(filename).to_xarray_dataset()
        if not hasattr(self, "tb_bins"):
            self._initialize_data(sensor, dataset)

        st = dataset["surface_type"][0]

        # Sensor with varying EIA need to be treated separately.
        if self.has_angles:
            if st in [2, 8, 9, 10, 11, 16]:
                for a in range(sensor.n_angles):
                    i_a = dataset["pixel_position"].data == (a + 1)
                    tbs = dataset["brightness_temperatures"].data[i_a]
                    for k in range(sensor.n_freqs):
                        cs, _ = np.histogram(tbs[:, k],
                                             bins=self.tb_bins)
                        self.tbs[st - 1, k, a] += cs
            else:
                for a in range(sensor.n_angles):
                    tbs = dataset["brightness_temperatures"].data
                    for k in range(sensor.n_freqs):
                        cs, _ = np.histogram(tbs[:, k, a],
                                             bins=self.tb_bins)
                        self.tbs[st - 1, k, a] += cs
        # Sensor with constant EIA.
        else:
            tbs = dataset["brightness_temperatures"]
            for k in range(sensor.n_freqs):
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
                        i_a = dataset["pixel_position"] == (a + 1)
                        cs, _ = np.histogram(v[i_a], bins=self.bins[k])
                        self.targets[k][st - 1, a] += cs
                else:
                    for a in range(sensor.n_angles):
                        cs, _ = np.histogram(v[:, j], bins=self.bins[k])
                        self.targets[k][st - 1, a] += cs
            else:
                cs, _ = np.histogram(v, bins=self.bins[k])
                self.targets[k][st - 1] += cs

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

###############################################################################
# Statistics
###############################################################################


def process_files(sensor, files, statistics):
    for f in files:
        for s in statistics:
            s.process_file(sensor, f)
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
        pool = ProcessPoolExecutor(n_workers)
        batches = list(_split_files(self.files, n_workers))

        tasks = []
        for b in batches:
            tasks.append(pool.submit(process_files,
                                     self.sensor,
                                     b,
                                     self.statistics))

        stats = tasks.pop(0).result()
        for t in tasks:
            for s_old, s_new in zip(stats, t.result()):
                s_old.merge(s_new)

        for s in stats:
            s.save(output_path)
