"""
Tests for the gprof_nn.data.era5 module.
"""
from datetime import datetime

from conftest import NEEDS_ARCHIVES
import numpy as np

from gprof_nn.data.era5 import (
    load_era5_data,
    add_era5_precip
)


@NEEDS_ARCHIVES
def test_load_era5_data():
    """
    Tests adding ERA5 precip data to preprocessor data.
    """

    start_time = np.datetime64("2020-01-01T23:00:00")
    end_time = np.datetime64("2020-01-02T01:00:00")

    era5_data  = load_era5_data(start_time, end_time)
    assert era5_data.time[0] < start_time
    assert era5_data.time[-1] > end_time


@NEEDS_ARCHIVES
def test_add_era5_precip(preprocessor_data_gmi):
    """
    Tests adding ERA5 precip data to preprocessor data.
    """
    data_pp = preprocessor_data_gmi
    start_time = preprocessor_data_gmi.scan_time.data[0]
    end_time = preprocessor_data_gmi.scan_time.data[-1]
    era5_data  = load_era5_data(start_time, end_time)

    add_era5_precip(data_pp, era5_data)

    surface_type = data_pp.surface_type.data
    surface_precip = data_pp.surface_precip.data

    sea_ice = (surface_type == 2) + (surface_type == 16)
    assert np.all(np.isfinite(surface_precip[sea_ice]))
    assert np.all(np.isnan(surface_precip[~sea_ice]))
