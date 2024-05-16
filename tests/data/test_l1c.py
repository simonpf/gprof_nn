"""
Test reading and manipulation of L1C files.
"""
from pathlib import Path

import numpy as np

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.l1c import L1CFile

from conftest import NEEDS_ARCHIVES


@NEEDS_ARCHIVES
def test_open_granule_gmi():
    """
    Test finding of specific GMI L1C file and reading data into
    xarray.Dataset.
    """
    l1c_path = sensors.GMI.l1c_file_path
    l1c_file = L1CFile.open_granule(27510, l1c_path, sensors.GMI)
    l1c_data = l1c_file.to_xarray_dataset()

    assert l1c_data.pixels.size == 221
    assert l1c_file.sensor == sensors.GMI

    tbs = l1c_data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))


@NEEDS_ARCHIVES
def test_open_granule_mhs():
    """
    Test finding of specific MHS L1C file and reading data into
    xarray.Dataset.
    """
    l1c_path = sensors.MHS.l1c_file_path
    l1c_file = L1CFile.open_granule(51010, l1c_path, sensors.MHS)
    l1c_data = l1c_file.to_xarray_dataset()

    assert l1c_data.pixels.size == 90
    assert l1c_data.scans.size == 2295
    assert l1c_file.sensor == sensors.MHS

    tbs = l1c_data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))


@NEEDS_ARCHIVES
def test_open_granule_tmi():
    """
    Test finding of specific TMI L1C file and reading data into
    xarray.Dataset.
    """
    l1c_path = Path(sensors.TMI.l1c_file_path)

    l1c_file = L1CFile.open_granule(91866, l1c_path, sensors.TMI)
    l1c_data = l1c_file.to_xarray_dataset()

    assert l1c_data.pixels.size == 208
    assert l1c_file.sensor == sensors.TMIPO

    tbs = l1c_data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))


@NEEDS_ARCHIVES
def test_open_granule_ssmis():
    """
    Test finding of specific SSMIS L1C file and reading data into
    xarray.Dataset.
    """
    l1c_path = Path(sensors.SSMIS.l1c_file_path)

    l1c_file = L1CFile.open_granule(61436, l1c_path, sensors.SSMIS)
    l1c_data = l1c_file.to_xarray_dataset()

    assert l1c_data.pixels.size == 180
    assert l1c_file.sensor == sensors.SSMIS

    tbs = l1c_data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))


@NEEDS_ARCHIVES
def test_open_granule_atms():
    """
    Test finding of specific ATMS L1C file and reading data into
    xarray.Dataset.
    """
    l1c_path = Path(sensors.ATMS.l1c_file_path)

    l1c_file = L1CFile.open_granule(59355, l1c_path, sensors.ATMS)
    l1c_data = l1c_file.to_xarray_dataset()

    assert l1c_data.pixels.size == 96
    assert l1c_file.sensor == sensors.ATMS

    tbs = l1c_data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))


@NEEDS_ARCHIVES
def test_find_file_gmi():
    """
    Tests finding a GMI L1C file for a given date.
    """
    l1c_path = sensors.GMI.l1c_file_path
    date = np.datetime64("2019-01-01T00:30:00")
    l1c_file = L1CFile.find_file(date, l1c_path)
    l1c_data = l1c_file.to_xarray_dataset()
    assert date > l1c_data.scan_time[0]
    assert date < l1c_data.scan_time[-1]
    assert l1c_file.sensor == sensors.GMI


@NEEDS_ARCHIVES
def test_find_file_mhs():
    """
    Tests finding an MHS file for a given date.
    """
    l1c_path = Path(sensors.MHS.l1c_file_path)
    date = np.datetime64("2019-01-01T01:33:00")
    l1c_file = L1CFile.find_file(date, l1c_path, sensor=sensors.MHS)
    data = l1c_file.to_xarray_dataset()

    assert date > data.scan_time[0]
    assert date < data.scan_time[-1]
    assert l1c_file.sensor == sensors.MHS
    assert "incidence_angle" in data.variables


@NEEDS_ARCHIVES
def test_find_file_tmi():
    """
    Tests finding an TMI file for a given date.
    """
    l1c_path = Path(sensors.TMIPO.l1c_file_path)
    date = np.datetime64("2014-01-01T01:00:00")
    l1c_file = L1CFile.find_file(date, l1c_path, sensor=sensors.TMI)
    data = l1c_file.to_xarray_dataset()

    assert date > data.scan_time[0]
    assert date < data.scan_time[-1]
    assert l1c_file.sensor == sensors.TMIPO


@NEEDS_ARCHIVES
def test_find_file_ssmis():
    """
    Tests finding a GMI L1C file for a given date.
    """
    l1c_path = Path(sensors.SSMIS.l1c_file_path)
    date = np.datetime64("2018-10-01T00:30:00")
    l1c_file = L1CFile.find_file(date, l1c_path, sensor=sensors.SSMIS)
    l1c_data = l1c_file.to_xarray_dataset()
    assert date > l1c_data.scan_time[0]
    assert date < l1c_data.scan_time[-1]
    assert l1c_file.sensor == sensors.SSMIS



@NEEDS_ARCHIVES
def test_find_files():
    """
    Ensure that finding a file covering a given ROI works as expected
    as well the loading of observations covering only a certain
    ROI.
    """
    l1c_path = sensors.GMI.l1c_file_path
    date = np.datetime64("2019-01-01T00:30:00")

    roi = (-37, -65, -35, -63)
    files = list(L1CFile.find_files(date, l1c_path, roi=roi))
    assert len(files) == 5

    data = files[0].to_xarray_dataset(roi=roi)
    n_scans = data.scans.size
    lats = data["latitude"].data
    lons = data["longitude"].data
    assert n_scans < 200

    # Ensure each scan has at least one obs at a longitude larger than the
    # minimum requested.
    assert np.all(np.sum(lons >= roi[0], -1) > 1)
    assert np.all(np.sum(lons < roi[2], -1) > 1)
    assert np.all(np.sum(lats >= roi[1], -1) > 1)
    assert np.all(np.sum(lats < roi[3], -1) > 1)


def test_extract_scans(tmpdir):
    """
    Test finding of specific GMI L1C file and reading data into
    xarray.Dataset.
    """
    l1c_path = sensors.GMI.l1c_file_path

    l1c_file = L1CFile.open_granule(27510, l1c_path, sensors.GMI)
    l1c_data = l1c_file.to_xarray_dataset()

    lat = l1c_data.latitude.data[250, :].mean()
    lon = l1c_data.longitude.data[250, :].mean()
    lon_0 = lon - 0.5
    lon_1 = lon + 0.5
    lat_0 = lat - 0.5
    lat_1 = lat + 0.5
    roi = [lon_0, lat_0, lon_1, lat_1]

    roi_path = Path(tmpdir) / "roi.HDF5"
    l1c_file.extract_scans(roi, roi_path, min_scans=256)
    roi_file = L1CFile(roi_path)
    roi_data = roi_file.to_xarray_dataset()

    lats = roi_data.latitude.data
    lons = roi_data.longitude.data
    print(lats.mean())
    print(lons.mean())
    print(roi)

    assert roi_data.scans.size >= 256
    inside = ((lons >= lon_0) *
              (lons < lon_1) *
              (lats >= lat_0) *
              (lats < lat_1))
    assert np.any(inside)


def test_extract_scan_range(tmpdir):
    """
    Test finding of specific GMI L1C file and reading data into
    xarray.Dataset.
    """
    l1c_path = sensors.GMI.l1c_file_path
    l1c_file = L1CFile.open_granule(27510, l1c_path, sensors.GMI)
    l1c_data = l1c_file.to_xarray_dataset()

    roi_path = tmpdir / "l1c_file.HDF5"
    l1c_file.extract_scan_range(0, 256, roi_path)
    roi_file = L1CFile(roi_path)
    roi_data = roi_file.to_xarray_dataset()

    assert roi_data.scans.size == 256
