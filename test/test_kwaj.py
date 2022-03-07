"""
Tests for gprof_nn.data.kwaj module.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn.data.kwaj import RadarFile, get_overpasses


DATA_PATH = Path(__file__).parent / "data"


def test_radar_file_open_file():
    """
    Test opening of a file from daily archive as xarray.Datset.
    """
    filename = DATA_PATH / "kwaj_data.tar.gz"
    radar_file = RadarFile(filename)

    filename = "180620/KWAJ_2018_0620_000650.cf.gz"
    data = radar_file.open_file(filename)

    assert "RR" in data.variables
    assert "RC" in data.variables

    assert data.RR.min() >= 0.0
    assert data.RR.max() >= 0.0

    assert data.latitude.max() >= 9.0
    assert data.latitude.min() <= 8.0
    assert data.longitude.max() >= 168
    assert data.longitude.min() <= 167


def test_radar_file_open_files():
    """
    Test opening of a file from daily archive as xarray.Datset.
    """
    filename = DATA_PATH / "kwaj_data.tar.gz"
    radar_file = RadarFile(filename)

    start = np.datetime64("2018-06-20T00:15:00")
    end = np.datetime64("2018-06-20T00:20:00")
    data = radar_file.open_files(start, end)

    assert "RR" in data.variables
    assert "RC" in data.variables

    assert data.RR.min() >= 0.0
    assert data.RR.max() >= 0.0

    assert data.latitude.max() >= 9.0
    assert data.latitude.min() <= 8.0
    assert data.longitude.max() >= 168
    assert data.longitude.min() <= 167


def test_get_overpasses_gmi():
    overpasses = get_overpasses("gmi")
    assert len(overpasses) > 0
