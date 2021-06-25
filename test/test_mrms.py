"""
Test reading of MRMS-GMI match ups used for the surface precip
prediction over snow surfaces.
"""
from pathlib import Path

import numpy as np
import pytest

from gprof_nn import sensors
from gprof_nn.data.mrms import MRMSMatchFile
from gprof_nn.data.l1c import L1CFile
from gprof_nn.utils import CONUS

TEST_FILE_GMI = "1801_MRMS2GMI_gprof_db_08all.bin.gz"
TEST_FILE_MHS = "1801_MRMS2GMI_gprof_db_08all.bin.gz"


def test_read_file_gmi():
    """
    Read GMI match file and ensure that all latitudes roughly match
    CONUS coordinates.
    """
    path = Path(__file__).parent / "data" / TEST_FILE_GMI
    ms = MRMSMatchFile(path)

    assert np.all(ms.data["latitude"] > 20.0)
    assert np.all(ms.data["latitude"] < 70.0)
    assert np.all(ms.data["longitude"] > -130.0)
    assert np.all(ms.data["longitude"] < -50.0)

    data = ms.to_xarray_dataset(day=23)


def test_read_file_mhs():
    """
    Read MHS match file and ensure that all latitudes roughly match
    CONUS coordinates.
    """
    path = Path(__file__).parent / "data" / TEST_FILE_MHS
    ms = MRMSMatchFile(path)

    assert np.all(ms.data["latitude"] > 20.0)
    assert np.all(ms.data["latitude"] < 70.0)
    assert np.all(ms.data["longitude"] > -130.0)
    assert np.all(ms.data["longitude"] < -50.0)

    data = ms.to_xarray_dataset(day=23)

@pytest.mark.xfail(reason="SNOWDAS files not available.")
def test_match_precip():
    """
    Match surface precip from MRMS file to observations in L1C file.
    """

    date = np.datetime64("2018-01-24T00:00:00")
    roi = CONUS
    path = Path(__file__).parent / "data"

    mrms_file = MRMSMatchFile(path / TEST_FILE_GMI)
    l1c_files = L1CFile.find_files(date, path, roi=roi)
    for f in l1c_files:
        print(f)
        data = mrms_file.match_targets(f.to_xarray_dataset(roi=CONUS))
        data.to_netcdf("test.nc")


def test_find_files_gmi():
    """
    Ensure that exactly one GMI MRMS file is found in test data.
    """
    path = Path(__file__).parent / "data"
    files = MRMSMatchFile.find_files(path, sensor=sensors.GMI)
    assert len(files) == 1


def test_find_files_mhs():
    """
    Ensure that exactly one GMI MRMS file is found in test data.
    """
    path = Path(__file__).parent / "data"
    files = MRMSMatchFile.find_files(path, sensor=sensors.MHS)
    assert len(files) == 1
