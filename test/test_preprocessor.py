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
from gprof_nn.data.training_data import GPROF_NN_1D_Dataset, write_preprocessor_file
from gprof_nn.data.l1c import L1CFile


DATA_PATH = get_test_data_path()


HAS_PREPROCESSOR = has_preprocessor()


def test_read_preprocessor_gmi():
    """
    Tests reading a GMI preprocessor file.
    """
    data_path = Path(__file__).parent / "data"
    input_file = PreprocessorFile(
        data_path / "gmi" / "pp" / "GMIERA5_190901_031303.pp"
    )
    input_data = input_file.to_xarray_dataset()

    assert input_file.n_pixels == 221
    assert input_data.pixels.size == 221
    assert input_data.scans.size == input_file.n_scans

    tbs = input_data.brightness_temperatures.data
    tbs = tbs[tbs > 0]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = input_data.surface_type.data
    assert np.all((st >= 0) * (st <= 18))

    lat = input_data.latitude
    assert np.all((lat >= -90) * (lat <= 90))
    lon = input_data.longitude
    assert np.all((lat >= -180) * (lat <= 180))

    t2m = input_data.two_meter_temperature
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = input_data.total_column_water_vapor
    assert np.all((tcwv >= 0) * (tcwv < 200))

    date = input_file.first_scan_time
    assert date == input_data.scan_time[0].data

    ocean_frac = input_data.ocean_fraction.data
    assert np.all((ocean_frac >= 0) * (ocean_frac <= 100))
    land_frac = input_data.land_fraction.data
    assert np.all((land_frac >= 0) * (land_frac <= 100))
    ice_frac = input_data.ice_fraction.data
    assert np.all((ice_frac >= 0) * (ice_frac <= 100))


def test_read_preprocessor_mhs():
    """
    Tests reading a GMI preprocessor file.
    """
    input_file = PreprocessorFile(DATA_PATH / "mhs" / "pp" / "MHS.pp")
    input_data = input_file.to_xarray_dataset()

    assert input_file.n_pixels == 90
    assert input_data.pixels.size == 90
    assert input_data.scans.size == input_file.n_scans

    tbs = input_data.brightness_temperatures.data
    tbs = tbs[tbs > 0]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = input_data.surface_type.data
    assert np.all((st >= 0) * (st <= 18))

    am = input_data.airmass_type.data
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = input_data.latitude
    assert np.all((lat >= -90) * (lat <= 90))
    lon = input_data.longitude
    assert np.all((lat >= -180) * (lat <= 180))

    t2m = input_data.two_meter_temperature
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = input_data.total_column_water_vapor
    assert np.all((tcwv >= 0) * (tcwv < 200))

    date = input_file.first_scan_time
    assert date == input_data.scan_time[0].data


def test_read_preprocessor_tmi():
    """
    Tests reading a TMI preprocessor file.
    """
    DATA_PATH = Path(__file__).parent / "data"
    input_file = PreprocessorFile(DATA_PATH / "tmi" / "pp" / "GPM_TMI_100101.pp")
    input_data = input_file.to_xarray_dataset()

    assert input_file.n_pixels == 208
    assert input_data.pixels.size == 208
    assert input_data.scans.size == input_file.n_scans

    tbs = input_data.brightness_temperatures.data
    tbs = tbs[tbs > 0]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = input_data.surface_type.data
    assert np.all((st >= 0) * (st <= 18))

    am = input_data.airmass_type.data
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = input_data.latitude
    assert np.all((lat >= -90) * (lat <= 90))
    lon = input_data.longitude
    assert np.all((lat >= -180) * (lat <= 180))

    t2m = input_data.two_meter_temperature
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = input_data.total_column_water_vapor
    assert np.all((tcwv >= 0) * (tcwv < 200))

    date = input_file.first_scan_time
    assert date == input_data.scan_time[0].data


def test_read_preprocessor_ssmi():
    """
    Tests reading a SSMI preprocessor file.
    """
    DATA_PATH = Path(__file__).parent / "data"
    input_file = PreprocessorFile(DATA_PATH / "ssmi" / "pp" / "GPM_SSMI_030101.pp")
    input_data = input_file.to_xarray_dataset()

    assert input_file.n_pixels == 128
    assert input_data.pixels.size == 128
    assert input_data.scans.size == input_file.n_scans

    tbs = input_data.brightness_temperatures.data
    tbs = tbs[tbs > 0]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = input_data.surface_type.data
    assert np.all((st >= 0) * (st <= 18))

    am = input_data.airmass_type.data
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = input_data.latitude
    assert np.all((lat >= -90) * (lat <= 90))
    lon = input_data.longitude
    assert np.all((lat >= -180) * (lat <= 180))

    t2m = input_data.two_meter_temperature
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = input_data.total_column_water_vapor
    assert np.all((tcwv >= 0) * (tcwv < 200))

    date = input_file.first_scan_time
    assert date == input_data.scan_time[0].data


def test_read_preprocessor_ssmis():
    """
    Tests reading a SSMIS preprocessor file.
    """
    DATA_PATH = Path(__file__).parent / "data"
    input_file = PreprocessorFile(DATA_PATH / "ssmis" / "pp" / "F17_190101_062736_ITE.pp")
    input_data = input_file.to_xarray_dataset()

    assert input_file.n_pixels == 180
    assert input_data.pixels.size == 180
    assert input_data.scans.size == input_file.n_scans

    tbs = input_data.brightness_temperatures.data
    tbs = tbs[tbs > 0]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = input_data.surface_type.data
    assert np.all((st >= 0) * (st <= 18))

    am = input_data.airmass_type.data
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = input_data.latitude
    assert np.all((lat >= -90) * (lat <= 90))
    lon = input_data.longitude
    assert np.all((lat >= -180) * (lat <= 180))

    t2m = input_data.two_meter_temperature
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = input_data.total_column_water_vapor
    assert np.all((tcwv >= 0) * (tcwv < 200))

    date = input_file.first_scan_time
    assert date == input_data.scan_time[0].data


def test_read_preprocessor_amsr2():
    """
    Tests reading a AMSR2 preprocessor file.
    """
    DATA_PATH = Path(__file__).parent / "data"
    input_file = PreprocessorFile(
        DATA_PATH / "amsr2" / "pp" / "AMSR2_190101_035234.pp"
    )
    input_data = input_file.to_xarray_dataset()

    assert input_file.n_pixels == 486
    assert input_data.pixels.size == 486
    assert input_data.scans.size == input_file.n_scans

    tbs = input_data.brightness_temperatures.data
    tbs = tbs[tbs > 0]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = input_data.surface_type.data
    assert np.all((st >= 0) * (st <= 18))

    am = input_data.airmass_type.data
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = input_data.latitude
    assert np.all((lat >= -90) * (lat <= 90))
    lon = input_data.longitude
    assert np.all((lat >= -180) * (lat <= 180))

    t2m = input_data.two_meter_temperature
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = input_data.total_column_water_vapor
    assert np.all((tcwv >= 0) * (tcwv < 200))

    date = input_file.first_scan_time
    assert date == input_data.scan_time[0].data


def test_read_preprocessor_atms():
    """
    Tests reading a ATMS preprocessor file.
    """
    DATA_PATH = Path(__file__).parent / "data"
    input_file = PreprocessorFile(
        DATA_PATH / "atms" / "pp" /
        "1C.NPP.ATMS.XCAL2019-V.20181001-S004905-E023034.035889.ITE753.pp"
    )
    input_data = input_file.to_xarray_dataset()

    assert input_file.n_pixels == 90
    assert input_data.pixels.size == 90
    assert input_data.scans.size == input_file.n_scans

    tbs = input_data.brightness_temperatures.data
    tbs = tbs[tbs > 0]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = input_data.surface_type.data
    assert np.all((st >= -99) * (st <= 18))

    am = input_data.airmass_type.data
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = input_data.latitude
    assert np.all((lat >= -9999) * (lat <= 90))
    lon = input_data.longitude
    assert np.all((lat >= -9999) * (lat <= 180))

    t2m = input_data.two_meter_temperature
    assert np.all((t2m > -9999) * (t2m < 400))
    tcwv = input_data.total_column_water_vapor
    assert np.all((tcwv >= -9999) * (tcwv < 200))

    date = input_file.first_scan_time
    assert date == input_data.scan_time[0].data


def test_write_preprocessor_file(tmp_path):
    """
    Writes dataset to preprocessor file and ensures that the
    data from the preprocessor file matches that in the original
    dataset.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc"

    targets = [
        "surface_precip",
        "ice_water_path",
        "rain_water_path",
        "cloud_water_path",
        "latent_heat",
        "rain_water_content",
        "snow_water_content",
        "cloud_water_content",
    ]

    dataset = GPROF_NN_1D_Dataset(
        input_file,
        batch_size=1,
        normalize=False,
        transform_zeros=False,
        targets=targets,
    )
    write_preprocessor_file(dataset.to_xarray_dataset(),
                            tmp_path / "preprocessor_file.nc")

    preprocessor_file = PreprocessorFile(tmp_path / "preprocessor_file.nc")
    preprocessor_data = preprocessor_file.to_xarray_dataset()

    bts_pp = preprocessor_data["brightness_temperatures"].data
    n = preprocessor_data.scans.size * preprocessor_data.pixels.size
    assert np.all(np.isclose(dataset.x[:n, :5], bts_pp[:, :, :5].reshape(-1, 5)))

    t2m_pp = preprocessor_data["two_meter_temperature"].data
    assert np.all(np.isclose(dataset.x[:n, 15], t2m_pp[:, :].ravel()))

    tcwv_pp = preprocessor_data["total_column_water_vapor"].data
    assert np.all(np.isclose(dataset.x[:n, 16], tcwv_pp[:, :].ravel()))

    st_pp = preprocessor_data["surface_type"].data
    assert np.all(
        np.isclose(np.where(dataset.x[:n, 17 : 17 + 18])[1] + 1, st_pp[:n, :].ravel())
    )

    at_pp = preprocessor_data["airmass_type"].data
    assert np.all(
        np.isclose(np.where(dataset.x[:n, 17 + 18 : 17 + 22])[1], at_pp[:n, :].ravel())
    )


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_run_preprocessor_gmi_era5():
    """
    Test running the GMI preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/1CR_GMI_ITE")
    l1c_file = L1CFile.open_granule(27510, l1c_path, sensors.GMI)
    data = run_preprocessor(l1c_file.filename, sensor=sensors.GMI)

    tbs = data.brightness_temperatures.data
    tbs = tbs[tbs > 0]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = data.surface_type.data
    assert np.all((st >= 0) * (st <= 18))

    lat = data.latitude
    assert np.all((lat >= -90) * (lat <= 90))
    lon = data.longitude
    assert np.all((lat >= -180) * (lat <= 180))

    t2m = data.two_meter_temperature
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = data.total_column_water_vapor
    assert np.all((tcwv >= 0) * (tcwv < 200))


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_run_preprocessor_gmi_ganal():
    """
    Test running the GMI preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/1CR_GMI_ITE")
    l1c_file = L1CFile.open_granule(27510, l1c_path, sensors.GMI)
    data = run_preprocessor(l1c_file.filename, sensor=sensors.GMI, configuration=GANAL)

    tbs = data.brightness_temperatures.data
    tbs = tbs[tbs > 0]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = data.surface_type.data
    assert np.all((st >= 0) * (st <= 18))

    am = data.airmass_type.data
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = data.latitude
    assert np.all((lat >= -90) * (lat <= 90))
    lon = data.longitude
    assert np.all((lat >= -180) * (lat <= 180))

    t2m = data.two_meter_temperature
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = data.total_column_water_vapor
    assert np.all((tcwv >= 0) * (tcwv < 200))


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_run_preprocessor_mhs_era5():
    """
    Test running the MHS preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/1C_NOAA19")
    date = datetime(2019, 1, 1, 0, 30)
    l1c_file = L1CFile.find_file(
        date,
        l1c_path,
        sensor=sensors.MHS,
    )
    data = run_preprocessor(l1c_file.filename, sensor=sensors.MHS)

    tbs = data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = data.surface_type.data
    valid = st >= 0
    assert np.all((st[valid] >= 0) * (st[valid] <= 18))

    am = data.airmass_type.data[valid]
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = data.latitude.data[valid]
    assert np.all((lat >= -90) * (lat <= 90))
    lon = data.longitude.data[valid]
    assert np.all((lon >= -180) * (lon <= 180))

    t2m = data.two_meter_temperature.data[valid]
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = data.total_column_water_vapor.data[valid]
    tcwv = tcwv[tcwv >= 0]
    assert np.all((tcwv >= 0) * (tcwv < 200))


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_run_preprocessor_mhs_ganal():
    """
    Test running the MHS preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/1C_NOAA19")
    date = datetime(2019, 1, 1, 0, 30)
    l1c_file = L1CFile.find_file(
        date,
        l1c_path,
        sensor=sensors.MHS,
    )
    data = run_preprocessor(l1c_file.filename, configuration=GANAL, sensor=sensors.MHS)

    tbs = data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = data.surface_type.data
    valid = np.all(st >= 0, axis=-1)
    st = st[valid]
    assert np.all((st >= 0) * (st <= 18))

    am = data.airmass_type.data[valid]
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = data.latitude[valid]
    assert np.all((lat >= -90) * (lat <= 90))
    lon = data.longitude[valid]
    assert np.all((lat >= -180) * (lat <= 180))

    t2m = data.two_meter_temperature[valid]
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = data.total_column_water_vapor[valid]
    assert np.all((tcwv >= 0) * (tcwv < 200))


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_run_preprocessor_tmi_era5():
    """
    Test running the TMI preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/1C_TMI")
    date = datetime(2010, 1, 1, 0, 30)
    l1c_file = L1CFile.find_file(
        date,
        l1c_path,
        sensor=sensors.TMI,
    )
    data = run_preprocessor(l1c_file.filename, sensor=sensors.TMIPO)

    tbs = data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = data.surface_type.data
    valid = st >= 0
    assert np.all((st[valid] >= 0) * (st[valid] <= 18))

    am = data.airmass_type.data[valid]
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = data.latitude.data[valid]
    assert np.all((lat >= -90) * (lat <= 90))
    lon = data.longitude.data[valid]
    assert np.all((lon >= -180) * (lon <= 180))

    t2m = data.two_meter_temperature.data[valid]
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = data.total_column_water_vapor.data[valid]
    tcwv = tcwv[tcwv >= 0]
    assert np.all((tcwv >= 0) * (tcwv < 200))


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_run_preprocessor_ssmis_era5():
    """
    Test running the SSMIS preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/1C_F17_ITE")
    date = datetime(2018, 10, 1, 0, 30)
    l1c_file = L1CFile.find_file(
        date,
        l1c_path,
        sensor=sensors.SSMIS,
    )
    data = run_preprocessor(l1c_file.filename, sensor=sensors.SSMIS)

    tbs = data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = data.surface_type.data
    valid = st >= 0
    assert np.all((st[valid] >= 0) * (st[valid] <= 18))

    am = data.airmass_type.data[valid]
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = data.latitude.data[valid]
    assert np.all((lat >= -90) * (lat <= 90))
    lon = data.longitude.data[valid]
    assert np.all((lon >= -180) * (lon <= 180))

    t2m = data.two_meter_temperature.data[valid]
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = data.total_column_water_vapor.data[valid]
    tcwv = tcwv[tcwv >= 0]
    assert np.all((tcwv >= 0) * (tcwv < 200))


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_run_preprocessor_amsr2_era5():
    """
    Test running the AMSR2 preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/AMSR2_ITE")
    date = datetime(2018, 10, 1, 0, 30)
    l1c_file = L1CFile.find_file(
        date,
        l1c_path,
        sensor=sensors.AMSR2,
    )
    data = run_preprocessor(l1c_file.filename, sensor=sensors.AMSR2)

    tbs = data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = data.surface_type.data
    valid = st >= 0
    assert np.all((st[valid] >= 0) * (st[valid] <= 18))

    am = data.airmass_type.data[valid]
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = data.latitude.data[valid]
    assert np.all((lat >= -90) * (lat <= 90))
    lon = data.longitude.data[valid]
    assert np.all((lon >= -180) * (lon <= 180))

    t2m = data.two_meter_temperature.data[valid]
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = data.total_column_water_vapor.data[valid]
    tcwv = tcwv[tcwv >= 0]
    assert np.all((tcwv >= 0) * (tcwv < 200))


@pytest.mark.skipif(not HAS_PREPROCESSOR, reason="Preprocessor missing.")
def test_run_preprocessor_atms_era5():
    """
    Test running the ATMS preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/1C_ATMS_ITE")
    date = datetime(2018, 10, 1, 1, 30)
    l1c_file = L1CFile.find_file(
        date,
        l1c_path,
        sensor=sensors.ATMS,
    )
    data = run_preprocessor(l1c_file.filename, sensor=sensors.ATMS)

    tbs = data.brightness_temperatures.data
    valid = np.all(tbs > 0, axis=-1)
    tbs = tbs[valid]
    assert np.all((tbs > 20) * (tbs <= 350))

    st = data.surface_type.data
    valid = st >= 0
    assert np.all((st[valid] >= 0) * (st[valid] <= 18))

    am = data.airmass_type.data[valid]
    am = am[am >= 0]
    assert np.all((am >= 0) * (am <= 18))

    lat = data.latitude.data[valid]
    assert np.all((lat >= -90) * (lat <= 90))
    lon = data.longitude.data[valid]
    assert np.all((lon >= -180) * (lon <= 180))

    t2m = data.two_meter_temperature.data[valid]
    assert np.all((t2m > 200) * (t2m < 400))
    tcwv = data.total_column_water_vapor.data[valid]
    tcwv = tcwv[tcwv >= 0]
    assert np.all((tcwv >= 0) * (tcwv < 200))


def test_calculate_frozen_precip():
    """
    That below -6.5 all precip is frozen and above all precip is liquid.
    """
    fp = calculate_frozen_precip(263.15, 0, 10.0)
    assert np.isclose(fp, 10.0)
    fp = calculate_frozen_precip(263.15, 1, 10.0)
    assert np.isclose(fp, 10.0)
    fp = calculate_frozen_precip(283.15, 0, 10.0)
    assert np.isclose(fp, 0.0)
    fp = calculate_frozen_precip(283.15, 1, 10.0)
    assert np.isclose(fp, 0.0)
