"""
Tests for reading the preprocessor format.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from quantnn.normalizer import Normalizer

from gprof_nn.data.preprocessor import (PreprocessorFile,
                                        run_preprocessor,
                                        calculate_frozen_precip)
from gprof_nn.data.training_data import (GPROF0DDataset,
                                         write_preprocessor_file)
from gprof_nn.data.l1c import L1CFile


def test_read_preprocessor_gmi():
    """
    Tests reading a GMI preprocessor file.
    """
    path = Path(__file__).parent
    input_file = PreprocessorFile(path / "data" / "gmi" / "GMIERA5_190101_027510.pp")
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
    path = Path(__file__).parent
    input_file = PreprocessorFile(path / "data" / "mhs" / "MHS.pp")
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


def test_write_preprocessor_file(tmp_path):
    """
    Writes dataset to preprocessor file and ensures that the
    data from the preprocessor file matches that in the original
    dataset.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"

    targets = ["surface_precip",
               "ice_water_path",
               "rain_water_path",
               "cloud_water_path",
               "latent_heat",
               "rain_water_content",
               "snow_water_content",
               "cloud_water_content"]

    dataset = GPROF0DDataset(input_file,
                             batch_size=1,
                             normalize=False,
                             transform_zeros=False,
                             targets=targets)
    dataset.save(tmp_path / "training_data.nc")

    write_preprocessor_file(tmp_path / "training_data.nc",
                            tmp_path / "preprocessor_file.nc")

    preprocessor_file = PreprocessorFile(tmp_path / "preprocessor_file.nc")
    preprocessor_data = preprocessor_file.to_xarray_dataset()

    bts_pp = preprocessor_data["brightness_temperatures"].data
    n = dataset.x.shape[0]
    assert np.all(np.isclose(dataset.x[:, :5],
                             bts_pp[:, :, :5].reshape(-1, 5)[:n]))

    t2m_pp = preprocessor_data["two_meter_temperature"].data
    assert np.all(np.isclose(dataset.x[:, 15],
                             t2m_pp[:, :].ravel()[:n]))

    tcwv_pp = preprocessor_data["total_column_water_vapor"].data
    assert np.all(np.isclose(dataset.x[:, 16],
                             tcwv_pp[:, :].ravel()[:n]))

    st_pp = preprocessor_data["surface_type"].data
    assert np.all(np.isclose(np.where(dataset.x[:, 17:17 + 18])[1],
                             st_pp[:, :].ravel()[:n]))

    at_pp = preprocessor_data["airmass_type"].data
    assert np.all(np.isclose(np.where(dataset.x[:, 17 + 18:17 + 22])[1],
                             at_pp[:, :].ravel()[:n]))


@pytest.mark.xfail
def test_run_preprocessor_gmi():
    """
    Test running the GMI preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/1CR_GMI")
    l1c_file = L1CFile.open_granule(27510,
                                    l1c_path,
                                    sensor=sensors.GMI)
    data = run_preprocessor(l1c_file.filename,
                            sensor=sensors.GMI)
    assert "two_meter_temperature" in data.variables


@pytest.mark.xfail
def test_run_preprocessor_mhs():
    """
    Test running the MHS preprocessor on a specific L1C file.
    """
    l1c_path = Path("/pdata4/archive/GPM/1C_METOPB")
    date = datetime(2019, 1, 1, 0, 30)
    l1c_file = L1CFile.find_file(date,
                                 l1c_path,
                                 sensor=sensors.MHS,)
    data = run_preprocessor(l1c_file.filename,
                            sensor=sensors.MHS)
    assert "two_meter_temperature" in data.variables


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
