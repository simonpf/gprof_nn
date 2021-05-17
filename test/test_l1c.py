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

def test_find_files():
    """
    Ensure that finding file covering a give ROI works as expected
    as well the loading of observations covering only a certain
    ROI.
    """
    l1c_path = Path(__file__).parent / "data"
    date = np.datetime64("2019-01-01T00:30:00")

    roi = (-35, -68, -10, -62)
    files = list(L1CFile.find_files(date, roi, l1c_path))
    assert len(files) == 1

    data = files[0].to_xarray_dataset(roi)
    n_scans = data.scans.size
    lats = data["latitude"].data
    lons = data["longitude"].data
    assert n_scans < 200

    # Ensure each scan has at least one obs at a longitude larger than the
    # minimum requested.
    assert np.all(np.sum(lons >= -35, -1) > 1)
    assert np.all(np.sum(lons < -10, -1) > 1)
    assert np.all(np.sum(lats >= -68, -1) > 1)
    assert np.all(np.sum(lats < -62, -1) > 1)

    roi = (-35, 60, -10, 62)
    files = list(L1CFile.find_files(date, roi, l1c_path))
    assert len(files) == 0
