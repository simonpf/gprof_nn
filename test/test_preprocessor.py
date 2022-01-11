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
    input_file = PreprocessorFile(DATA_PATH / "gmi" / "GMIERA5_190101_027510.pp")
    input_data = input_file.to_xarray_dataset()

    assert input_file.n_pixels == 221
    assert input_data.pixels.size == 221
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


def test_read_preprocessor_tmi():
    """
    Tests reading a GMI preprocessor file.
    """
    DATA_PATH = Path(__file__).parent / "data"
    input_file = PreprocessorFile(DATA_PATH / "tmi" / "pp" / "GPM_TMI_100101.pp")
    input_data = input_file.to_xarray_dataset()

    assert input_file.n_pixels == 208
    assert input_data.pixels.size == 208
    assert input_data.scans.size == input_file.n_scans

    tbs = input_data.brightness_temperatures.data
    tbs = tbs[tbs > 0]
    print(tbs.min(), tbs.max())
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
    l1c_path = Path("/pdata4/archive/GPM/1CR_GMI")
    l1c_file = L1CFile.open_granule(27510, l1c_path, sensors.GMI)
    data = run_preprocessor(l1c_file.filename, sensor=sensors.GMI)

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
def test_run_preprocessor_gmi_ganal():
    """
    Test running the GMI preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/1CR_GMI")
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
    data = run_preprocessor(l1c_file.filename, sensor=sensors.TMI)

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
