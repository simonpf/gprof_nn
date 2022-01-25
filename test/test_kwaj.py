"""
Tests for gprof_nn.data.kwaj module.
"""
from pathlib import Path

from gprof_nn.data.kwaj import RadarFile


DATA_PATH = Path(__file__).parent / "data"


def test_radar_file_open_file():
    """
    Test opening of a file from daily archive as xarray.Datset.
    """
    filename = DATA_PATH / "kwaj_data.tar.gz"
    radar_file = RadarFile(filename)
    print(radar_file.times)

    filename = "180620/KWAJ_2018_0620_000650.cf.gz"
    data = radar_file.open_file(filename)

    assert "RR" in data.variables
    assert "RC" in data.variables

    assert data.RR.min() >= 0.0
    assert data.RR.max() >= 0.0

