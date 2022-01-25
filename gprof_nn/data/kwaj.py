"""
==================
gprof_nn.data.kwaj
==================

This module provides functionality to download and process Radar
observations from the Kwajalein Atoll.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
import re
import shutil
import subprocess
import tarfile

import numpy as np
import xarray as xr

from gprof_nn.data.training_data import decompress_and_load


_BASE_URL = "https://pmm-gv.gsfc.nasa.gov/pub/gpmarchive/Radar/KPOL/"

LINK_REGEX = re.compile(r"<a href=\"(KPOL_[\w\.]*)\">")



class RadarFile:
    """
    The radar file class provides an inteface to download and read daily
    archives of radar observations from the Kwajalein radar.

    Note: This class will only be able to handle files in CDRadial format,
       i.e. files that are older than roughly 2017.
    """
    @staticmethod
    def get_files(year, month):
        """
        Return a dictionary of radar files available for a given year
        and month.

        Args:
            year: The year given as integer.
            month: The month given as integer.

        Return:
            results: A dictionary mapping tuples of ``(year, month, day)``
            corresponding to days for which radar data is available
            to corresponding filenames.
        """
        url = _BASE_URL + f"{year:04}/{month:02}/"
        html = urlopen(url).read().decode()

        results = {}
        for match in LINK_REGEX.finditer(html):
            filename = match.group(1)
            _, year, monthday = filename.split("_")
            year = int(year)
            month = int(monthday[:2])
            day = int(monthday[2:4])
            results[(year, month, day)] = filename
        return results

    @staticmethod
    def download_file(year, month, day, destination=None):
        """
        Download file archive for given date.

        Args:
            year: The year given as int
            month: The month given as int
            day: The day given as int
            destination: Optional folder to which to write the downloaded file.
                If None the current working directory is used.

        Return:
            Path pointing to the downloaded file.
        """
        files = RadarFile.get_files(year, month)
        if (year, month, day) not in files:
            raise ValueError(f"No files for {year}-{month:02}-{day:02}.")

        filename = files[(year, month, day)]
        url = _BASE_URL + f"{year:04}/{month:02}/" + filename

        if destination is None:
            output_file = filename
        else:
            output_file = Path(destination) / filename
        with open(output_file, "wb") as output:
            shutil.copyfileobj(urlopen(url), output)

        return output

    def __init__(self, filename):
        """
        Open radar archive for a given day.

        Args:
            filename: Path pointing to the compressed archive of daily
               radar observations downloaded from the PMM website.
        """
        self.filename = filename
        self.archive = tarfile.open(filename)

        self.times = {}
        for name in self.archive.getnames():
            filename = Path(name)
            if filename.suffix == ".gz":
                _, year, monthday, time = filename.name.split("_")
                month = monthday[:2]
                day = monthday[2:4]
                hour = time[:2]
                min = time[2:4]
                sec = time[4:6]
                datestr = f"{year}-{month}-{day}T{hour}:{min}:{sec}"
                date = np.datetime64(datestr)
                self.times[date] = name

    def open_file(self, name):
        """
        Open a file from the archive as xarray.Dataset.

        Args:
            name: String containing the name of the file in the archive.

        Return:
            xarray.Dataset containing the radar data for the given file.
        """
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            extracted = tmp / "output.nc.gz"
            with open(extracted, "wb") as output:
                shutil.copyfileobj(self.archive.extractfile(name), output)

            args = ["gunzip", "-f", str(extracted)]
            subprocess.run(args)

            decompressed = tmp / "output.nc"
            dataset = xr.load_dataset(decompressed)
        return dataset


    def open_observations(self, date):
        """
        Open the observations closest to a given time from the archive.

        Args:
            date: The data for which to extract the observations.

        Return:
            An xarray.Dataset containing the observations closest in time
            to the given date.
        """
        times = np.array(list(self.times.keys()))
        names = list(self.times.values())
        delta = times - date
        delta_n = delta[delta < 0]

        indices = []
        if len(delta_n) > 0:
            indices.append(np.argmax(delta_n))
        delta_p = delta[delta >= 0]
        if len(delta_p) > 0:
            indices.append(np.argmin(delta_p))

        data = []
        for index in indices:
            name = names[index]

        closest = np.argmin(np.abs(delta))
