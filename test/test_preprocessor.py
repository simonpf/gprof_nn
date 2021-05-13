"""
Tests for reading the preprocessor format.
"""
from pathlib import Path

import numpy as np

from gprof_nn.data.preprocessor import PreprocessorFile
from gprof_nn.data.training_data import (GPROF0DDataset,
                                         write_preprocessor_file)

def test_preprocessor_file(tmp_path):
    """
    Writes dataset to preprocessor file and ensures that the
    data from the preprocessor file matches that in the original
    dataset.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "dataset_0d.nc"

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

    bts_pp = preprocessor_data["brightness_temperatures"]
    assert np.all(np.isclose(dataset.x[:, :5],
                             bts_pp[0, :, :5]))
    t2m_pp = preprocessor_data["two_meter_temperature"]
    assert np.all(np.isclose(dataset.x[:, 15],
                             t2m_pp[0, :]))
    tcwv_pp = preprocessor_data["total_column_water_vapor"]
    assert np.all(np.isclose(dataset.x[:, 16],
                             tcwv_pp[0, :]))
    st_pp = preprocessor_data["surface_type"]
    assert np.all(np.isclose(np.where(dataset.x[:, 17:17 + 18])[1],
                             st_pp[0, :]))
    at_pp = preprocessor_data["airmass_type"]
    assert np.all(np.isclose(np.maximum(np.where(dataset.x[:, 17 + 18:17 + 21])[1], 1),
                             st_pp[0, :]))
