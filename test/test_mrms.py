"""
Test reading of MRMS-GMI match ups used for the surface precip
prediction over snow surfaces.
"""
from pathlib import Path

import numpy as np
import pytest

from gprof_nn import config
from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.mrms import MRMSMatchFile, has_snowdas_ratios
from gprof_nn.data.l1c import L1CFile
from gprof_nn.utils import CONUS


MRMS_PATH = Path(config.CONFIG.data.mrms_path)
NEEDS_MRMS_DATA = pytest.mark.skipif(
    not MRMS_PATH.exists(),
    reason="MRMS collocations are not available."
)


TEST_FILE_GMI = "1801_MRMS2GMI_gprof_db_08all.bin.gz"
TEST_FILE_MHS = "1801_MRMS2MHS_DB1_01.bin.gz"
TEST_FILE_SSMIS = "1810_MRMS2SSMIS_01.bin.gz"
TEST_FILE_AMSR2 = "1810_MRMS2AMSR2_01.bin.gz"

HAS_SNOWDAS_RATIOS = has_snowdas_ratios()

###############################################################################
# GMI
###############################################################################

@NEEDS_MRMS_DATA
def test_read_file_gmi():
    """
    Read GMI match file and ensure that all latitudes roughly match
    CONUS coordinates.
    """
    path = MRMS_PATH / "GMI2MRMS_match2019" / "db_mrms4GMI" / TEST_FILE_GMI
    ms = MRMSMatchFile(path)
    assert np.all(ms.data["latitude"] > 20.0)
    assert np.all(ms.data["latitude"] < 70.0)
    assert np.all(ms.data["longitude"] > -130.0)
    assert np.all(ms.data["longitude"] < -50.0)
    data = ms.to_xarray_dataset(day=23)

@NEEDS_MRMS_DATA
def test_match_precip_gmi():
    """
    Match surface precip from MRMS file to observations in L1C file.
    """
    path = MRMS_PATH / "GMI2MRMS_match2019" / "db_mrms4GMI" / TEST_FILE_GMI
    date = np.datetime64("2018-01-24T00:00:00")
    roi = CONUS

    mrms_file = MRMSMatchFile(path)
    l1c_files = L1CFile.find_files(date, path, roi=roi)
    for f in l1c_files:
        data = mrms_file.match_targets(f.to_xarray_dataset(roi=CONUS))
        data.to_netcdf("test.nc")

@NEEDS_MRMS_DATA
def test_find_files_gmi():
    """
    Ensure that exactly one GMI MRMS file is found in test data.
    """
    path = MRMS_PATH / "GMI2MRMS_match2019" / "db_mrms4GMI"
    files = MRMSMatchFile.find_files(path, sensor=sensors.GMI)
    assert len(files) > 0

###############################################################################
# MHS
###############################################################################

@NEEDS_MRMS_DATA
def test_read_file_mhs():
    """
    Read MHS match file and ensure that all latitudes roughly match
    CONUS coordinates.
    """
    path = MRMS_PATH / "MHS2MRMS_match2019" / "monthly_2021" / TEST_FILE_MHS
    ms = MRMSMatchFile(path)

    assert np.all(ms.data["latitude"] > 20.0)
    assert np.all(ms.data["latitude"] < 70.0)
    assert np.all(ms.data["longitude"] > -130.0)
    assert np.all(ms.data["longitude"] < -50.0)


@NEEDS_MRMS_DATA
def test_match_precip_mhs():
    """
    Match surface precip from MRMS file to observations in L1C file.
    """
    path = MRMS_PATH / "MHS2MRMS_match2019" / "monthly_2021" / TEST_FILE_MHS
    date = np.datetime64("2018-01-01T01:00:00")
    roi = CONUS

    mrms_file = MRMSMatchFile(path, sensor=sensors.MHS)
    l1c_files = L1CFile.find_files(date, path, roi=roi, sensor=sensors.MHS)
    for f in l1c_files:
        data = mrms_file.match_targets(f.to_xarray_dataset(roi=CONUS))
        data.to_netcdf("test.nc")


@NEEDS_MRMS_DATA
def test_find_files_mhs():
    """
    Ensure that exactly one GMI MRMS file is found in test data.
    """
    path = MRMS_PATH / "MHS2MRMS_match2019" / "monthly_2021"
    files = MRMSMatchFile.find_files(path, sensor=sensors.MHS)
    assert len(files) == 58

###############################################################################
# SSMIS
###############################################################################

@NEEDS_MRMS_DATA
def test_read_file_ssmis():
    """
    Read SSMIS match file and ensure that all latitudes roughly match
    CONUS coordinates.
    """
    path = MRMS_PATH / "SSMIS2MRMS_match2019" / "monthly_2021" / TEST_FILE_SSMIS
    ms = MRMSMatchFile(path)
    ms = MRMSMatchFile(path)

    assert np.all(ms.data["latitude"] > 20.0)
    assert np.all(ms.data["latitude"] < 70.0)
    assert np.all(ms.data["longitude"] > -130.0)
    assert np.all(ms.data["longitude"] < -50.0)

    data = ms.to_xarray_dataset(day=23)
    tbs = data.brightness_temperatures.data
    valid = tbs >= 0
    tbs = tbs[valid]
    assert np.all((tbs >= 0) * (tbs <= 400))


###############################################################################
# AMSR2
###############################################################################

@NEEDS_MRMS_DATA
def test_read_file_amsr2():
    """
    Read AMSR2 match file and ensure that all latitudes roughly match
    CONUS coordinates.
    """
    path = MRMS_PATH / "AMSR22MRMS_match2019" / "monthly_2021" / TEST_FILE_AMSR2
    ms = MRMSMatchFile(path)

    assert np.all(ms.data["latitude"] > 20.0)
    assert np.all(ms.data["latitude"] < 70.0)
    assert np.all(ms.data["longitude"] > -130.0)
    assert np.all(ms.data["longitude"] < -50.0)

    data = ms.to_xarray_dataset(day=23)
    tbs = data.brightness_temperatures.data
    valid = tbs >= 0
    tbs = tbs[valid]
    assert np.all((tbs >= 0) * (tbs <= 400))

    surface_precip = data.surface_precip.data
    assert np.any(np.isfinite(surface_precip))
