"""
Tests for the 'gprof_nn.data.combined' module.
"""
from pathlib import Path

import numpy as np

from gprof_nn.data import get_test_data_path
from gprof_nn.data.combined import (
    GPMCMBFile,
    calculate_smoothing_kernels,
    load_combined_data_bin
)


DATA_PATH = get_test_data_path()


def test_read_gpm_cmb_file():
    """
    Test reading of GPM combined file.
    """
    path = DATA_PATH / "cmb"
    filename = (
        path /
        "2B.GPM.DPRGMI.CORRA2018.20210829-S205206-E222439.042628.V06A.HDF5"
    )
    data = GPMCMBFile(filename).to_xarray_dataset()
    assert "latitude" in data.variables
    assert "longitude" in data.variables
    assert "surface_precip" in data.variables


def test_read_gpm_cmb_file_smoothed():
    """
    Test reading of GPM combined file with smoothing of surface precip data.
    """
    path = Path(__file__).parent
    filename = (
        path / "data" / "cmb" /
        "2B.GPM.DPRGMI.CORRA2018.20210829-S205206-E222439.042628.V06A.HDF5"
    )
    data = GPMCMBFile(filename).to_xarray_dataset(smooth=True)
    assert "latitude" in data.variables
    assert "longitude" in data.variables
    assert "surface_precip" in data.variables

def test_read_gpm_cmb_file_profiles_smoothed():
    """
    Test reading of GPM combined file with profiles and smoothing.
    """
    path = DATA_PATH / "cmb"
    filename = (
        path /
        "2B.GPM.DPRGMI.CORRA2018.20210829-S205206-E222439.042628.V06A.HDF5"
    )
    data = GPMCMBFile(filename).to_xarray_dataset(profiles=True, smooth=True)
    assert "latitude" in data.variables
    assert "longitude" in data.variables
    assert "surface_precip" in data.variables


def test_smoothing_kernels():
    """
    Test that calculation of smoothing kernels works correctly.
    """
    k = calculate_smoothing_kernels(2.0 * 4.9e3, 2.0 * 5.09e3)

    # Assert kernel has expected shape and is normalized.
    assert k.shape == (9, 9)
    assert np.isclose(k.sum(), 1.0)

    # Assert that full-width at half maximum is at the correct location.
    k_max = k[4, 4]
    assert np.isclose(k[4, 5] / k_max, 0.5)
    assert np.isclose(k[4, 3] / k_max, 0.5)
    assert np.isclose(k[5, 4] / k_max, 0.5)
    assert np.isclose(k[3, 4] / k_max, 0.5)


def test_read_combined_bin():
    """
    Ensure that values read from combined data in .bin format
    are consistent.
    """
    data_path = Path(__file__).parent / "data"
    filename = data_path / "CMB.20181001.026079.ITE768.bin.gz"
    data = load_combined_data_bin(filename)

    lats = data.latitude.data
    assert np.all((lats >= -90) * (lats <= 90))

    lons = data.longitude.data
    assert np.all((lons >= -180) * (lons <= 180))

    sp = data.surface_precip.data
    sp[sp >= 0.0]
    assert np.all((sp >= 0.0) * (sp <= 500))

    t = data.temperature.data
    t = t[t >= 0]
    assert np.all((t > 150) * (t <= 400))

    tbs = data.simulated_brightness_temperatures.data
    tbs = tbs[tbs >= 0.0]
    assert np.all((tbs >= 20.0) * (tbs < 400))

    scan_time = data.scan_time.data
    assert np.all(np.isclose(scan_time[0], 2018))
    assert np.all(np.isclose(scan_time[1], 10))
