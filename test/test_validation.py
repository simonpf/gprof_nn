"""
Tests for the gprof_nn.data.validation module.
"""
import numpy as np
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.validation import (ValidationData,
                                      unify_grid,
                                      ValidationFileProcessor)
from gprof_nn.utils import great_circle_distance


DATA_PATH = get_test_data_path()


def test_get_granules():
    """
    Test listing of granules.
    """
    validation_data = ValidationData(sensors.GMI)
    granules = validation_data.get_granules(2016, 10)
    assert 15199 in granules


def test_open_granule():
    """
    Test listing of granules.
    """
    validation_data = ValidationData(sensors.GMI)
    data = validation_data.open_granule(2016, 10, 15199)

    sp = data.surface_precip.data
    sp = sp[sp >= 0]
    assert np.all(sp <= 500)

    lats = data.latitude.data
    assert np.all((lats >= 0) * (lats <= 60))

    lons = data.longitude.data
    assert np.all((lons >= -180) * (lons <= 0))


def test_unify_grid():
    """
    Test that the unify grid function yields grids with resolutions
    close to 5km.
    """
    l1c_file = DATA_PATH / "gmi" / "l1c" / (
        "1C-R.GPM.GMI.XCAL2016-C.20190101-S001447-E014719.027510.V05A.HDF5"
    )
    l1c_data = L1CFile(l1c_file).to_xarray_dataset()
    lats = l1c_data.latitude.data
    lons = l1c_data.longitude.data

    lats_5, lons_5 = unify_grid(lats, lons)

    # Along track distance.
    d = great_circle_distance(
        lats_5[:-1], lons_5[:-1],
        lats_5[1:], lons_5[1:]
    )
    assert d.min() > 4.4e3
    assert d.max() < 5.5e3


    # Across track distance
    d = great_circle_distance(
        lats_5[:, :-1], lons_5[:, :-1],
        lats_5[:, 1:], lons_5[:, 1:]
    )
    assert d.min() > 4.8e3
    assert d.max() < 5.2e3


def test_validation_file_processor(tmp_path):
    """
    Ensure that mrmrs data is interpolated to 5km x 5km grid.
    """
    mrms_file = tmp_path / "mrms.nc"
    pp_file = tmp_path / "preprocessor.pp"

    processor = ValidationFileProcessor(sensors.GMI, 2016, 10)
    processor.process_granule(15199, mrms_file, pp_file)

    mrms_data = xr.load_dataset(mrms_file)

    assert mrms_data.attrs["sensor"] == "GMI"

    lats = mrms_data.latitude.data
    lons = mrms_data.longitude.data

    # Along track distance.
    d = great_circle_distance(
        lats[:-1], lons[:-1],
        lats[1:], lons[1:]
    )
    assert d.min() > 4.4e3
    assert d.max() < 5.5e3

    # Across track distance
    d = great_circle_distance(
        lats[:, :-1], lons[:, :-1],
        lats[:, 1:], lons[:, 1:]
    )
    assert d.min() > 4.8e3
    assert d.max() < 5.2e3

    sp = mrms_data.surface_precip.data
    sp = sp[np.isfinite(sp)]
    assert np.all((sp >= 0.0) * (sp <= 500.0))
