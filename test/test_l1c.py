"""
Test reading and manipulation of L1C files.
"""
from pathlib import Path

import numpy as np

from gprof_nn.data.l1c import L1CFile

def test_open_granule():
    """
    Test finding of specific L1C file and reading data into
    xarray.Dataset.
    """
    l1c_path = Path(__file__).parent / "data"

    l1c_file = L1CFile.open_granule(27510, l1c_path)
    l1c_data = l1c_file.to_xarray_dataset()
    print(l1c_data)

    assert l1c_data.pixels.size == 221
    assert l1c_data.scans.size == 2962

def test_find_file():
    """
    Tests finding a L1C file for a given date.
    """
    l1c_path = Path(__file__).parent / "data"
    date = np.datetime64("2019-01-01T00:30:00")
    l1c_file = L1CFile.find_file(date, l1c_path).to_xarray_dataset()
    assert date > l1c_file.scan_time[0]
    assert date < l1c_file.scan_time[-1]

