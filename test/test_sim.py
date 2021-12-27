"""
This file contains tests for the reading and processing of .sim files
defined in 'gprof_nn.data.sim.py'.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.sim import (SimFile,
                               _load_era5_data,
                               _add_era5_precip,
                               apply_orographic_enhancement,
                               extend_pixels,
                               process_mrms_file,
                               process_sim_file,
                               process_l1c_file)
from gprof_nn.data.preprocessor import PreprocessorFile


DATA_PATH = get_test_data_path()
HAS_ARCHIVES = Path(sensors.GMI.l1c_file_path).exists()


def test_open_sim_file_gmi():
    """
    Tests reading simulator output file for GMI.
    """
    input_file = DATA_PATH / "gmi" / "sim" / "GMI.dbsatTb.20190101.027510.sim"
    sim_file = SimFile(input_file)
    data = sim_file.to_xarray_dataset()

    assert "surface_precip" in data.variables.keys()
    assert "latent_heat" in data.variables.keys()
    assert "snow_water_content" in data.variables.keys()
    assert "rain_water_content" in data.variables.keys()

    valid = data.surface_precip.data > -9999
    assert np.all(data.surface_precip[valid] >= 0.0)
    assert np.all(data.surface_precip[valid] <= 1000.0)
    assert np.all(data.latitude >= -90.0)
    assert np.all(data.latitude <= 90.0)
    assert np.all(data.longitude >= -180.0)
    assert np.all(data.longitude <= 180.0)

def test_open_sim_file_mhs():
    """
    Tests reading simulator output file for MHS.
    """
    input_file = DATA_PATH / "mhs" / "sim" / "MHS.dbsatTb.20190101.027510.sim"
    sim_file = SimFile(input_file)
    data = sim_file.to_xarray_dataset()

    assert "surface_precip" in data.variables.keys()
    assert "latent_heat" in data.variables.keys()
    assert "snow_water_content" in data.variables.keys()
    assert "rain_water_content" in data.variables.keys()

    valid = data.surface_precip.data > -9999
    assert np.all(data.surface_precip.data[valid] >= 0.0)
    assert np.all(data.surface_precip.data[valid] <= 1000.0)
    assert np.all(data.latitude >= -90.0)
    assert np.all(data.latitude <= 90.0)
    assert np.all(data.longitude >= -180.0)
    assert np.all(data.longitude <= 180.0)


def test_open_sim_file_tmi():
    """
    Tests reading simulator output file for MHS.
    """
    DATA_PATH = Path(__file__).parent / "data"
    input_file = DATA_PATH / "tmi" / "sim" / "TMIPR.dbsatTb.20190101.027510.sim"
    sim_file = SimFile(input_file)
    data = sim_file.to_xarray_dataset()

    assert "surface_precip" in data.variables.keys()
    assert "latent_heat" in data.variables.keys()
    assert "snow_water_content" in data.variables.keys()
    assert "rain_water_content" in data.variables.keys()

    valid = data.surface_precip.data > -9999
    assert np.all(data.surface_precip[valid] >= 0.0)
    assert np.all(data.surface_precip[valid] <= 1000.0)
    assert np.all(data.latitude >= -90.0)
    assert np.all(data.latitude <= 90.0)
    assert np.all(data.longitude >= -180.0)
    assert np.all(data.longitude <= 180.0)


def test_match_l1c_gmi():
    """
    Tests reading a GMI L1C file and matching it to data in
    a GMI .sim file.
    """
    l1c_path = DATA_PATH / "gmi" / "l1c"
    l1c_file = L1CFile.open_granule(27510, l1c_path, sensors.GMI)
    l1c_data = l1c_file.to_xarray_dataset()
    tbs = l1c_data.brightness_temperatures.data
    tbs_new = np.zeros(tbs.shape[:2] + (15,))
    tbs_new[:, :, :5] = tbs[..., :5]
    tbs_new[:, :, 6:12] = tbs[..., 5:11]
    tbs_new[:, :, 13:] = tbs[..., 11:]
    l1c_data["brightness_temperatures"] = (("scans", "pixels", "channels"),
                                           tbs_new)

    sim_path = DATA_PATH / "gmi" / "sim"
    sim_file = SimFile(sim_path / "GMI.dbsatTb.20190101.027510.sim")

    targets = ["surface_precip",
               "latent_heat",
               "rain_water_content",
               "ice_water_path"]

    sim_file.match_targets(l1c_data, targets=targets)

    assert "latent_heat" in l1c_data.variables.keys()
    assert "ice_water_path" in l1c_data.variables.keys()
    assert "snow_water_content" in l1c_data.variables.keys()
    assert "rain_water_content" in l1c_data.variables.keys()

    tbs = l1c_data.brightness_temperatures.data
    valid = tbs > 0
    assert np.all((tbs[valid] > 20) * (tbs[valid] < 400))

    lats = l1c_data.latitude.data
    assert np.all((lats >= -90) * (lats <= 90))
    lons = l1c_data.longitude.data
    assert np.all((lons >= -180) * (lons <= 180))

    sp = l1c_data.surface_precip.data
    valid = np.isfinite(sp)
    assert np.all((sp[valid] >= 0.0) * (sp[valid] < 300))

    lh = l1c_data.latent_heat.data
    valid = np.isfinite(lh)
    assert np.all(lh[valid] < 1000)


def test_match_l1c_mhs():
    """
    Tests reading a GMI L1C file and matching it to data in
    a MHS .sim file.
    """
    l1c_path = DATA_PATH / "gmi" / "l1c"
    l1c_file = L1CFile.open_granule(27510, l1c_path, sensors.GMI)
    l1c_data = l1c_file.to_xarray_dataset()
    l1c_data = l1c_data.rename({
        "channels": "channels_gmi",
        "brightness_temperatures": "brightness_temperatures_gmi"}
    )

    sim_path = DATA_PATH / "mhs" / "sim"
    sim_file = SimFile(sim_path / "MHS.dbsatTb.20190101.027510.sim")

    targets = ["surface_precip",
               "latent_heat",
               "rain_water_content",
               "ice_water_path"]

    sim_file.match_targets(l1c_data, targets=targets)

    assert "latent_heat" in l1c_data.variables.keys()
    assert "ice_water_path" in l1c_data.variables.keys()
    assert "snow_water_content" in l1c_data.variables.keys()
    assert "rain_water_content" in l1c_data.variables.keys()

    assert "brightness_temperature_biases" in l1c_data.variables.keys()
    assert "simulated_brightness_temperatures" in l1c_data.variables.keys()

    tbs = l1c_data.brightness_temperatures_gmi.data
    valid = tbs > 0
    assert np.all((tbs[valid] > 20) * (tbs[valid] < 400))

    lats = l1c_data.latitude.data
    assert np.all((lats >= -90) * (lats <= 90))
    lons = l1c_data.longitude.data
    assert np.all((lons >= -180) * (lons <= 180))

    sp = l1c_data.surface_precip.data
    valid = np.isfinite(sp)
    assert np.all((sp[valid] >= 0.0) * (sp[valid] < 300))

    lh = l1c_data.latent_heat.data
    valid = np.isfinite(lh)
    assert np.all(lh[valid] < 1000)


def test_match_l1c_tmi():
    """
    Tests reading a GMI L1C file and matching it to data in
    a TMI .sim file.
    """
    l1c_path = DATA_PATH / "gmi" / "l1c"
    l1c_file = L1CFile.open_granule(27510, l1c_path, sensors.GMI)
    l1c_data = l1c_file.to_xarray_dataset()
    l1c_data = l1c_data.rename({
        "channels": "channels_gmi",
        "brightness_temperatures": "brightness_temperatures_gmi"}
    )

    sim_path = Path(".").parent / "data" / "tmi" / "sim"
    sim_file = SimFile(sim_path / "TMIPR.dbsatTb.20190101.027510.sim")

    targets = ["surface_precip",
               "latent_heat",
               "rain_water_content",
               "ice_water_path"]

    sim_file.match_targets(l1c_data, targets=targets)

    assert "latent_heat" in l1c_data.variables.keys()
    assert "ice_water_path" in l1c_data.variables.keys()
    assert "snow_water_content" in l1c_data.variables.keys()
    assert "rain_water_content" in l1c_data.variables.keys()

    assert "brightness_temperature_biases" in l1c_data.variables.keys()
    assert "simulated_brightness_temperatures" in l1c_data.variables.keys()

    tbs = l1c_data.brightness_temperatures_gmi.data
    valid = tbs > 0
    assert np.all((tbs[valid] > 20) * (tbs[valid] < 400))

    lats = l1c_data.latitude.data
    assert np.all((lats >= -90) * (lats <= 90))
    lons = l1c_data.longitude.data
    assert np.all((lons >= -180) * (lons <= 180))

    sp = l1c_data.surface_precip.data
    valid = np.isfinite(sp)
    assert np.all((sp[valid] >= 0.0) * (sp[valid] < 300))

    lh = l1c_data.latent_heat.data
    valid = np.isfinite(lh)
    assert np.all(lh[valid] < 1000)


def test_find_files():
    """
    Assert that find_file functions successfully finds file in test data folder
    except when search is restricted to a different day.
    """
    path = DATA_PATH / "gmi" / "sim"
    sim_files = SimFile.find_files(path)
    assert len(sim_files) == 1

    sim_files = SimFile.find_files(path, day=1)
    assert len(sim_files) == 1

    sim_files = SimFile.find_files(path, day=2)
    assert len(sim_files) == 0


@pytest.mark.skipif(not HAS_ARCHIVES, reason="Data archives not available.")
def test_match_era5_precip():
    """
    Test loading and matching of data from ERA5.
    """
    l1c_path = DATA_PATH / "gmi" / "l1c"
    l1c_file = L1CFile.open_granule(27510, l1c_path, sensors.GMI)
    l1c_data = l1c_file.to_xarray_dataset()[{"scans": slice(0, 2)}]

    pp_path = DATA_PATH / "gmi" / "pp"
    preprocessor_file = PreprocessorFile(pp_path  / "GMIERA5_190101_027510.pp")
    input_data = preprocessor_file.to_xarray_dataset()

    start_time = l1c_data["scan_time"][0].data
    end_time = l1c_data["scan_time"][-1].data

    era5_path = "/qdata2/archive/ERA5"
    era5_data = _load_era5_data(start_time,
                                end_time,
                                era5_path)

    sim_path = DATA_PATH / "gmi" / "sim"
    sim_file = SimFile(sim_path / "GMI.dbsatTb.20190101.027510.sim")
    sim_file.match_targets(input_data)
    _add_era5_precip(input_data,
                     l1c_data,
                     era5_data)


def test_orographic_enhancement():
    """
    Writes dataset to preprocessor file and ensures that the
    data from the preprocessor file matches that in the original
    dataset.
    """
    preprocessor_file = PreprocessorFile(
        DATA_PATH / "gmi" / "pp" / "GMIERA5_190101_027510.pp"
    )
    input_data = preprocessor_file.to_xarray_dataset()
    sim_file = SimFile(DATA_PATH / "gmi" / "sim" / "GMI.dbsatTb.20190101.027510.sim")
    sim_file.match_targets(input_data)

    st = input_data["surface_type"].data
    at = input_data["airmass_type"].data
    sp = input_data["surface_precip"].data

    sp_ref = input_data["surface_precip"].data.copy()
    cp_ref = input_data["convective_precip"].data.copy()
    apply_orographic_enhancement(input_data, "ERA5")
    sp = input_data["surface_precip"].data
    cp = input_data["convective_precip"].data

    indices = (st == 17) * (at == 1) * np.isfinite(sp)
    assert np.all(np.isclose(sp[indices], sp_ref[indices] * 2.05213))
    assert np.all(np.isclose(cp[indices], cp_ref[indices] * 2.05213))

    indices = (st != 17) * (st != 18) * np.isfinite(sp)
    assert np.all(np.isclose(sp[indices], sp_ref[indices], rtol=1e-3))
    assert np.all(np.isclose(cp[indices], cp_ref[indices], rtol=1e-3))


def test_extend_dataset():
    """
    Ensure that extending a dataset along pixel dimensions works as expected.
    """
    data = xr.Dataset({
        "data": (("scans", "pixels"), np.zeros((221, 1)))
    })
    extended = extend_pixels(data)
    assert np.all(np.isclose(extended.data[:, 110], 0.0))


@pytest.mark.skipif(not HAS_ARCHIVES, reason="Data archives not available.")
def test_process_sim_file_gmi_era5():
    sim_file = DATA_PATH / "gmi" / "sim" / "GMI.dbsatTb.20190101.027510.sim"
    era5_path = "/qdata2/archive/ERA5/"
    data = process_sim_file(sim_file,
                            sensors.GMI,
                            "ERA5",
                            era5_path)

    assert np.all(data["source"].data == 0)
    sp = data["surface_precip"].data
    st = data["surface_type"].data
    snow = (st >= 8) * (st <= 11)
    assert np.all(np.isnan(sp[snow]))


@pytest.mark.skipif(not HAS_ARCHIVES, reason="Data archives not available.")
def test_process_mrms_file_gmi():
    """
    Test processing of MRMS-GMI matches for a given day.
    """
    mrms_file = DATA_PATH / "gmi" / "mrms" / "1801_MRMS2GMI_gprof_db_08all.bin.gz"
    era5_path = "/qdata2/archive/ERA5/"
    data = process_mrms_file(sensors.GMI, mrms_file, "ERA5", 24)
    assert data is not None
    assert data.pixels.size == 221
    assert np.all(data["source"].data == 1)


@pytest.mark.skipif(not HAS_ARCHIVES, reason="Data archives not available.")
def test_process_l1c_file_gmi():
    l1c_file = (Path(__file__).parent /
                "data" /
                "gmi" /
                "l1c" /
                "1C-R.GPM.GMI.XCAL2016-C.20190101-S001447-E014719.027510.V05A.HDF5")
    era5_path = "/qdata2/archive/ERA5/"
    data = process_l1c_file(l1c_file, sensors.GMI, "ERA5", era5_path)

    assert np.all(data["source"].data == 2)

    sp = data["surface_precip"].data
    st = data["surface_type"].data
    si = (st == 2) + (st == 16)
    assert np.all(np.isfinite(sp[si]))
    assert np.all(np.isnan(sp[~si]))

    # Ensure surface precip has valid values.
    sp = data.surface_precip.data
    valid = sp >= 0
    assert np.all(sp[valid] < 500)

    # Check that TBs are valid
    tbs = data["brightness_temperatures"].data
    valid = tbs >= 0
    assert np.all(tbs[valid] > 50)
    assert np.all(tbs[valid] < 500)

@pytest.mark.skipif(not HAS_ARCHIVES, reason="Data archives not available.")
def test_process_sim_file_mhs():
    sim_file = DATA_PATH / "mhs" / "sim" / "MHS.dbsatTb.20190101.027510.sim"
    era5_path = "/qdata2/archive/ERA5/"
    data = process_sim_file(sim_file, sensors.MHS, "ERA5", era5_path)

    assert np.all(data["source"].data == 0)
    sp = data["surface_precip"].data
    st = data["surface_type"].data
    snow = (st >= 8) * (st <= 11)
    sea_ice = (st == 2) + (st == 16)

    assert np.all(np.isnan(sp[snow + sea_ice]))


@pytest.mark.skipif(not HAS_ARCHIVES, reason="Data archives not available.")
def test_process_mrms_file_mhs():
    """
    Test processing of MRMS-MHS (NOAA-19) matches for a given day.
    """
    mrms_file = DATA_PATH / "mhs" / "mrms" / "1801_MRMS2MHS_DB1_01.bin.gz"
    era5_path = "/qdata2/archive/ERA5/"
    data = process_mrms_file(sensors.MHS, mrms_file, "ERA5", 23)
    assert data is not None
    assert data.pixels.size == 221
    assert np.all(data["source"].data == 1)

    # Ensure surface precip has valid values.
    sp = data.surface_precip.data
    valid = sp >= 0
    assert np.all(sp[valid] < 500)

    # Check that TBs are valid
    tbs = data["brightness_temperatures"].data
    valid = tbs >= 0
    assert np.all(tbs[valid] > 50)
    assert np.all(tbs[valid] < 500)

@pytest.mark.skipif(not HAS_ARCHIVES, reason="Data archives not available.")
def test_process_l1c_file_mhs():
    l1c_file = (
        DATA_PATH /
        "mhs" /
        "l1c" /
        "1C.NOAA19.MHS.XCAL2016-V.20190101-S013203-E031403.051010.V05A.HDF5"
    )
    era5_path = "/qdata2/archive/ERA5/"
    data = process_l1c_file(l1c_file, sensors.MHS, "ERA5", era5_path)

    assert np.all(data["source"].data == 2)

    sp = data["surface_precip"].data
    st = data["surface_type"].data
    si = (st == 2) + (st == 16)
    assert np.all(np.isfinite(sp[si]))
    assert np.all(np.isnan(sp[~si]))

    # Ensure surface precip has valid values.
    sp = data.surface_precip.data
    valid = sp >= 0
    assert np.all(sp[valid] < 500)

    # Check that TBs are valid
    tbs = data["brightness_temperatures"].data
    valid = tbs >= 0
    assert np.all(tbs[valid] > 50)
    assert np.all(tbs[valid] < 500)


@pytest.mark.skipif(not HAS_ARCHIVES, reason="Data archives not available.")
def test_process_sim_file_tmi():
    #sim_file = DATA_PATH / "tmi" / "sim" / "MHS.dbsatTb.20190101.027510.sim"
    sim_file = Path(__file__).parent / "data" / "tmi" / "sim" / "TMIPR.dbsatTb.20190101.027510.sim"
    era5_path = "/qdata2/archive/ERA5/"
    data = process_sim_file(sim_file, sensors.TMIPR, "ERA5", era5_path)

    assert np.all(data["source"].data == 0)
    sp = data["surface_precip"].data
    st = data["surface_type"].data
    snow = (st >= 8) * (st <= 11)
    sea_ice = (st == 2) + (st == 16)

    assert np.all(np.isnan(sp[snow + sea_ice]))

    # Check that GMI TBs are valid
    tbs = data["brightness_temperatures_gmi"].data
    valid = tbs >= 0
    assert np.all(tbs[valid] > 50)
    assert np.all(tbs[valid] < 500)

    # Check that simulated TBs are valid
    tbs = data["simulated_brightness_temperatures"].data
    valid = tbs >= 0
    assert np.all(tbs[valid] > 50)
    assert np.all(tbs[valid] < 500)

    # Check that TB biases are valid
    tbs = data["brightness_temperature_biases"].data
    valid = tbs >= 0
    assert np.all(tbs[valid] > -200)
    assert np.all(tbs[valid] < 200)

    # Check that surface precip values are valid
    sp = data["surface_precip"].data
    valid = sp >= 0
    assert np.all(sp[valid] < 400)


@pytest.mark.skipif(not HAS_ARCHIVES, reason="Data archives not available.")
def test_process_mrms_file_tmi():
    """
    Test processing of MRMS-GMI matches for a given day for the TMI sensor.
    """
    mrms_file = DATA_PATH / "gmi" / "mrms" / "1801_MRMS2GMI_gprof_db_08all.bin.gz"
    era5_path = "/qdata2/archive/ERA5/"
    data = process_mrms_file(sensors.TMIPR, mrms_file, "ERA5", 24)

    assert data is not None
    assert data.channels.size == 9
    assert data.pixels.size == 221
    assert np.all(data["source"].data == 1)

    # Ensure surface precip has valid values.
    sp = data.surface_precip.data
    valid = sp >= 0
    assert np.all(sp[valid] < 500)

    # Check that TBs are valid
    tbs = data["brightness_temperatures"].data
    valid = tbs >= 0
    assert np.all(tbs[valid] > 50)
    assert np.all(tbs[valid] < 500)


@pytest.mark.skipif(not HAS_ARCHIVES, reason="Data archives not available.")
def test_process_l1c_file_tmi():
    l1c_file = (
        DATA_PATH /
        "mhs" /
        "l1c" /
        "1C.NOAA19.MHS.XCAL2016-V.20190101-S013203-E031403.051010.V05A.HDF5"
    )
    era5_path = "/qdata2/archive/ERA5/"
    data = process_l1c_file(l1c_file, sensors.MHS, "ERA5", era5_path)

    assert np.all(data["source"].data == 2)

    sp = data["surface_precip"].data
    st = data["surface_type"].data
    si = (st == 2) + (st == 16)
    assert np.all(np.isfinite(sp[si]))
    assert np.all(np.isnan(sp[~si]))

    # Ensure surface precip has valid values.
    sp = data.surface_precip.data
    valid = sp >= 0
    assert np.all(sp[valid] < 500)

    # Check that TBs are valid
    tbs = data["brightness_temperatures"].data
    valid = tbs >= 0
    assert np.all(tbs[valid] > 50)
    assert np.all(tbs[valid] < 500)
