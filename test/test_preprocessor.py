"""
Tests for reading the preprocessor format.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from quantnn.normalizer import Normalizer

from gprof_nn import sensors
from gprof_nn.data.preprocessor import PreprocessorFile, run_preprocessor
from gprof_nn.data.training_data import (GPROF0DDataset,
                                         write_preprocessor_file)
from gprof_nn.data.l1c import L1CFile


def test_preprocessor_file(tmp_path):
    """
    Writes dataset to preprocessor file and ensures that the
    data from the preprocessor file matches that in the original
    dataset.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "training_data.nc"

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
                             target=targets)
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

