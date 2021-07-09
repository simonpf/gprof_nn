"""
Tests for the extraction of training samples for tehe GPROF-NN 0D
algorithm from .bin files.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data.bin import (FileProcessor,
                               BinFile)
from gprof_nn.data.training_data import GPROF0DDataset


def test_bin_file_gmi():
    """
    Test reading of GMI bin files and ensure all values are physical.
    """
    path = Path(__file__).parent
    input_file = BinFile(path / "data" / "gpm_300_40_00_18.bin").to_xarray_dataset()

    assert np.all(input_file["surface_precip"] >= 0)
    assert np.all(input_file["convective_precip"] >= 0)
    assert np.all(input_file["rain_water_path"] >= 0)
    assert np.all(input_file["two_meter_temperature"] > 300 - 0.5)
    assert np.all(input_file["two_meter_temperature"] < 300 + 0.5)
    assert np.all(input_file["total_column_water_vapor"] > 40 - 0.5)
    assert np.all(input_file["total_column_water_vapor"] < 40 + 0.5)
    assert np.all(input_file["surface_type"] == 18)
    assert np.all(input_file["airmass_type"] == 0)


def test_bin_file_mhs():
    """
    Test reading of MHS bin files and ensure all values are physical.
    """
    path = Path(__file__).parent
    input_file = BinFile(path / "data" / "mhs" / "gpm_290_60_01.bin").to_xarray_dataset()

    assert np.all(input_file["surface_precip"] >= 0)
    assert np.all(input_file["convective_precip"] >= 0)
    assert np.all(input_file["rain_water_path"] >= 0)
    assert np.all(input_file["two_meter_temperature"] > 290 - 0.5)
    assert np.all(input_file["two_meter_temperature"] < 290 + 0.5)
    assert np.all(input_file["total_column_water_vapor"] > 60 - 0.5)
    assert np.all(input_file["total_column_water_vapor"] < 60 + 0.5)
    assert np.all(input_file["surface_type"] == 1)
    assert np.all(input_file["airmass_type"] == 0)

def test_file_processor_gmi(tmp_path):
    """
    This tests the extraction of data from a bin file and ensures that
    the extracted dataset matches the original data.
    """
    path = Path(__file__).parent
    processor = FileProcessor(path / "data", include_profiles=True)
    output_file = tmp_path / "test_file.nc"
    processor.run_async(output_file, 0.0, 1.0, 1)

    input_file = BinFile(path / "data" / "gpm_300_40_00_18.bin")

    dataset = GPROF0DDataset(output_file,
                             normalize=False,
                             shuffle=False,
                             augment=False)
    normalizer = dataset.normalizer

    bts_input = input_file.handle["brightness_temperatures"]
    bts = dataset.x[:, :input_file.sensor.n_freqs]
    bts = np.nan_to_num(bts, nan=-9999.9)
    valid = bts[:, -1] > 0

    assert np.all(np.isclose(bts_input[valid], bts[valid]))
    assert np.all(np.isclose(input_file.handle["two_meter_temperature"].mean(),
                             dataset.x[:, 15].mean()))
    assert np.all(np.isclose(input_file.handle["total_column_water_vapor"].mean(),
                             dataset.x[:, 16].mean()))

    surface_types = np.where(dataset.x[:, 17:35])[1]
    assert np.all(np.isclose(input_file.surface_type,
                             surface_types + 1))
    airmass_types = np.where(dataset.x[:, 35:])[1]
    assert np.all(np.isclose(np.maximum(input_file.airmass_type, 1),
                             airmass_types + 1))

def test_file_processor_mhs(tmp_path):
    """
    This tests the extraction of data from a bin file and ensures that
    the extracted dataset matches the original data.
    """
    path = Path(__file__).parent
    processor = FileProcessor(path / "data" / "mhs", include_profiles=True)
    output_file = tmp_path / "test_file.nc"
    processor.run_async(output_file, 0.0, 1.0, 1)
    dataset = xr.load_dataset(output_file)

    assert np.all(dataset.surface_precip >= 0.0)
