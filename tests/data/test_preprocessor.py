"""
Tests for reading the preprocessor format.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from quantnn.normalizer import Normalizer

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.preprocessor import (
    PreprocessorFile,
    has_preprocessor,
    run_preprocessor,
    calculate_frozen_precip,
    ERA5,
    GANAL,
)
from gprof_nn.data.l1c import L1CFile


GPM_DATA = Path("/pdata4/archive/GPM")

NEEDS_GPM_DATA = pytest.mark.skipif(
    not GPM_DATA.exists(), reason="Needs GPM L1C data."
)

@NEEDS_GPM_DATA
def test_preprocessor_gmi(tmp_path):
    """
    Test running the preprocessor for GMI and loading the
    results.
    """
    l1c_file = L1CFile(
        GPM_DATA /
        "1CR_GMI_V7/1801/180101/1C-R.GPM.GMI.XCAL2016-C"
        ".20180101-S010928-E024202.021833.V07A.HDF5"
    )
    l1c_file.extract_scan_range(1000, 1005, tmp_path / "gmi_l1c.HDF5")
    pp_data = run_preprocessor(
        tmp_path / "gmi_l1c.HDF5",
        sensors.GMI
    )

    assert np.all(
        pp_data.scan_time > np.datetime64("2018-01-01T00:00:00")
    )
    tbs = pp_data.brightness_temperatures.data
    valid = tbs >= 0.0
    tbs = tbs[valid]

    assert tbs.size > 0
    assert tbs.max() < 320


@NEEDS_GPM_DATA
def test_preprocessor_amsr2(tmp_path):
    """
    Test running the preprocessor for AMSR2 and loading the
    results.
    """
    l1c_file = L1CFile(
        GPM_DATA /
        "1C_AMSR2_V7/1501/150101/1C.GCOMW1.AMSR2.XCAL2016-"
        "V.20150101-S000954-E014846.013958.V07A.HDF5"
    )
    l1c_file.extract_scan_range(1000, 1005, tmp_path / "amsr2_l1c.HDF5")
    pp_data = run_preprocessor(
        tmp_path / "amsr2_l1c.HDF5",
        sensors.AMSR2
    )

    assert np.all(
        pp_data.scan_time > np.datetime64("2015-01-01T00:00:00")
    )
    tbs = pp_data.brightness_temperatures.data
    valid = tbs >= 0.0
    tbs = tbs[valid]

    assert tbs.size > 0
    assert tbs.max() < 320
