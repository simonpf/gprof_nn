"""
Tests for reading the preprocessor format.
"""
from pathlib import Path

import numpy as np
from quantnn.normalizer import Normalizer

from gprof_nn.data.preprocessor import (PreprocessorFile,
                                        PreprocessorLoader0D)
from gprof_nn.data.training_data import (GPROF0DDataset,
                                         write_preprocessor_file,
                                         run_retrieval_0d)

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


def test_preprocessor_0d():
    """
    Ensure that preprocessor loader correctly loads batch from preprocessor
    file.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "GMIERA5_190101_027510.pp"
    normalizer = Normalizer.load(path / "data" / "normalizer.pckl")
    loader = PreprocessorLoader0D(input_file, normalizer)
    data = loader.data

    x = loader.get_batch(0).detach().numpy()
    x = normalizer.invert(x)
    tbs = np.nan_to_num(x[:, :15], nan=-9999.0)
    t2m = x[:, 15]
    tcwv = x[:, 16]
    st = np.where(x[:, 17:35])[1] + 1
    at = np.where(x[:, 35:])[1]
    n = st.shape[0]

    assert np.all(np.isclose(
        tbs,
        data["brightness_temperatures"].data.reshape(-1, 15)[:n],
        rtol=1e-3
    ))
    assert np.all(np.isclose(
        t2m,
        data["two_meter_temperature"].data.ravel()[:n],
        rtol=1e-3
    ))
    assert np.all(np.isclose(
        tcwv,
        data["total_column_water_vapor"].data.ravel()[:n],
        rtol=1e-3
    ))
    assert np.all(np.isclose(
        st,
        data["surface_type"].data.ravel()[:n],
        rtol=1e-3
    ))
    assert np.all(np.isclose(
        at,
        np.maximum(data["airmass_type"].data.ravel(), 0.0)[:n],
        rtol=1e-3
    ))




