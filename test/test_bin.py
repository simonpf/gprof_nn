"""
This file tests the 'gprof_nn.data.bin' module which provides functionality
to read and extract training data from the 'bin' files used by GPROF.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.bin import FileProcessor, BinFile
from gprof_nn.data.training_data import GPROF_NN_1D_Dataset


DATA_PATH = get_test_data_path()


def test_bin_file_gmi():
    """
    Test reading the different types of bin files for standard surface types,
    for sea ice and snow.
    """
    #
    # Simulator-derived bin files.
    #

    input_file = DATA_PATH / "gmi" / "bin" / "gpm_275_14_03_17.bin"
    input_data = BinFile(input_file).to_xarray_dataset()

    assert np.all(input_data["surface_precip"] >= 0)
    assert np.all(input_data["convective_precip"] >= 0)
    assert np.all(input_data["rain_water_path"] >= 0)
    assert np.all(input_data["two_meter_temperature"] > 275 - 0.5)
    assert np.all(input_data["two_meter_temperature"] < 275 + 0.5)
    assert np.all(input_data["total_column_water_vapor"] > 14 - 0.5)
    assert np.all(input_data["total_column_water_vapor"] < 14 + 0.5)
    assert np.all(input_data["surface_type"] == 17)
    assert np.all(input_data["airmass_type"] == 3)
    tbs = input_data.brightness_temperatures.data
    valid = tbs > 0
    tbs = tbs[valid]
    assert np.all(tbs > 20)
    assert np.all(tbs < 400)

    #
    # Seaice bin files.
    #

    input_file = DATA_PATH / "gmi" / "bin" / "gpm_269_00_16.bin"
    input_data = BinFile(input_file).to_xarray_dataset()

    assert np.all(input_data["surface_precip"] >= 0)
    assert np.all(input_data["convective_precip"] >= 0)
    assert np.all(input_data["rain_water_path"] < 0)
    assert np.all(input_data["two_meter_temperature"] > 269 - 0.5)
    assert np.all(input_data["two_meter_temperature"] < 269 + 0.5)
    assert np.all(input_data["total_column_water_vapor"] > 0 - 0.5)
    assert np.all(input_data["total_column_water_vapor"] < 0 + 0.5)
    assert np.all(input_data["surface_type"] == 16)
    assert np.all(input_data["airmass_type"] == 0)
    tbs = input_data.brightness_temperatures.data
    valid = tbs > 0
    tbs = tbs[valid]
    assert np.all(tbs > 20)
    assert np.all(tbs < 400)

    #
    # MRMS bin files.
    #

    input_file = DATA_PATH / "gmi" / "bin" / "gpm_298_28_11.bin"
    input_data = BinFile(input_file).to_xarray_dataset()

    assert np.all(input_data["surface_precip"] >= 0)
    assert np.all(input_data["convective_precip"] >= 0)
    assert np.all(input_data["rain_water_path"] < 0)
    assert np.all(input_data["two_meter_temperature"] > 298 - 0.5)
    assert np.all(input_data["two_meter_temperature"] < 298 + 0.5)
    assert np.all(input_data["total_column_water_vapor"] > 28 - 0.5)
    assert np.all(input_data["total_column_water_vapor"] < 28 + 0.5)
    assert np.all(input_data["surface_type"] == 11)
    assert np.all(input_data["airmass_type"] == 0)
    tbs = input_data.brightness_temperatures.data
    valid = tbs > 0
    tbs = tbs[valid]
    assert np.all(tbs > 20)
    assert np.all(tbs < 400)


def test_bin_file_mhs():
    """
    Test reading of MHS bin files and ensure all values are physical and match
    given bin.
    """
    #
    # Simulator-derived bin files.
    #

    input_file = DATA_PATH / "mhs" / "bin" / "gpm_289_52_04.bin"
    input_data = BinFile(input_file).to_xarray_dataset()

    assert np.all(input_data["surface_precip"] >= 0)
    assert np.all(input_data["convective_precip"] >= 0)
    assert np.all(input_data["rain_water_path"] >= 0)
    assert np.all(input_data["two_meter_temperature"] > 289 - 0.5)
    assert np.all(input_data["two_meter_temperature"] < 289 + 0.5)
    assert np.all(input_data["total_column_water_vapor"] > 52 - 0.5)
    assert np.all(input_data["total_column_water_vapor"] < 52 + 0.5)
    assert np.all(input_data["surface_type"] == 4)
    assert np.all(input_data["airmass_type"] == 0)
    tbs = input_data.brightness_temperatures.data
    valid = tbs > 0
    tbs = tbs[valid]
    assert np.all(tbs > 20)
    assert np.all(tbs < 400)

    #
    # Seaice bin files.
    #

    input_file = DATA_PATH / "mhs" / "bin" / "gpm_271_20_16.bin"
    input_data = BinFile(input_file).to_xarray_dataset()

    assert np.all(input_data["surface_precip"] >= 0)
    assert np.all(input_data["convective_precip"] >= 0)
    assert np.all(input_data["rain_water_path"] < 0)
    assert np.all(input_data["two_meter_temperature"] > 271 - 0.5)
    assert np.all(input_data["two_meter_temperature"] < 271 + 0.5)
    assert np.all(input_data["total_column_water_vapor"] > 20 - 0.5)
    assert np.all(input_data["total_column_water_vapor"] < 20 + 0.5)
    assert np.all(input_data["surface_type"] == 16)
    assert np.all(input_data["airmass_type"] == 0)
    tbs = input_data.brightness_temperatures.data
    valid = tbs > 0
    tbs = tbs[valid]
    assert np.all(tbs > 20)
    assert np.all(tbs < 400)

    #
    # MRMS bin files.
    #

    input_file = DATA_PATH / "mhs" / "bin" / "gpm_292_25_11.bin"
    input_data = BinFile(input_file).to_xarray_dataset()

    assert np.all(input_data["surface_precip"] >= 0)
    assert np.all(input_data["convective_precip"] >= 0)
    assert np.all(input_data["rain_water_path"] < 0)
    assert np.all(input_data["two_meter_temperature"] > 292 - 0.5)
    assert np.all(input_data["two_meter_temperature"] < 292 + 0.5)
    assert np.all(input_data["total_column_water_vapor"] > 25 - 0.5)
    assert np.all(input_data["total_column_water_vapor"] < 25 + 0.5)
    assert np.all(input_data["surface_type"] == 11)
    assert np.all(input_data["airmass_type"] == 0)
    tbs = input_data.brightness_temperatures.data
    valid = tbs > 0
    tbs = tbs[valid]
    assert np.all(tbs > 20)
    assert np.all(tbs < 400)


def test_bin_file_tmi():
    """
    Test reading of TMI bin files and ensure all values are physical and match
    given bin.
    """
    #
    # Simulator-derived bin files.
    #

    DATA_PATH = Path(__file__).parent/ "data"
    input_file = DATA_PATH / "tmi" / "bin" / "gpm_309_08_04.bin"

    input_data = BinFile(input_file).to_xarray_dataset()

    assert input_data.channels.size == 9

    assert np.all(input_data["surface_precip"] >= 0)
    assert np.all(input_data["surface_precip"] <= 500)
    assert np.all(input_data["convective_precip"] >= 0)
    assert np.all(input_data["convective_precip"] <= 500)
    assert np.all(input_data["rain_water_path"] >= 0)
    assert np.all(input_data["two_meter_temperature"] > 309 - 0.5)
    assert np.all(input_data["two_meter_temperature"] < 309 + 0.5)
    assert np.all(input_data["total_column_water_vapor"] > 8 - 0.5)
    assert np.all(input_data["total_column_water_vapor"] < 8 + 0.5)
    assert np.all(input_data["surface_type"] == 4)
    assert np.all(input_data["airmass_type"] == 0)
    tbs = input_data.brightness_temperatures.data
    valid = tbs > 0
    tbs = tbs[valid]
    assert np.all(tbs > 20)
    assert np.all(tbs < 400)

    #
    # Seaice bin files.
    #

    input_file = DATA_PATH / "tmi" / "bin" / "gpm_273_15_16.bin"
    input_data = BinFile(input_file).to_xarray_dataset()

    assert np.all(input_data["surface_precip"] >= 0)
    assert np.all(input_data["surface_precip"] <= 500)
    assert np.all(input_data["convective_precip"] >= 0)
    assert np.all(input_data["convective_precip"] <= 500)
    assert np.all(input_data["rain_water_path"] < 0)
    assert np.all(input_data["two_meter_temperature"] > 273 - 0.5)
    assert np.all(input_data["two_meter_temperature"] < 273 + 0.5)
    assert np.all(input_data["total_column_water_vapor"] > 15 - 0.5)
    assert np.all(input_data["total_column_water_vapor"] < 15 + 0.5)
    assert np.all(input_data["surface_type"] == 16)
    assert np.all(input_data["airmass_type"] == 0)
    tbs = input_data.brightness_temperatures.data
    valid = tbs > 0
    tbs = tbs[valid]
    assert np.all(tbs > 20)
    assert np.all(tbs < 400)

    #
    # MRMS bin files.
    #

    input_file = DATA_PATH / "tmi" / "bin" / "gpm_295_16_11.bin"
    input_data = BinFile(input_file).to_xarray_dataset()

    assert np.all(input_data["surface_precip"] >= 0)
    assert np.all(input_data["surface_precip"] <= 500)
    assert np.all(input_data["convective_precip"] >= 0)
    assert np.all(input_data["convective_precip"] <= 500)
    assert np.all(input_data["rain_water_path"] < 0)
    assert np.all(input_data["two_meter_temperature"] > 295 - 0.5)
    assert np.all(input_data["two_meter_temperature"] < 295 + 0.5)
    assert np.all(input_data["total_column_water_vapor"] > 16 - 0.5)
    assert np.all(input_data["total_column_water_vapor"] < 16 + 0.5)
    assert np.all(input_data["surface_type"] == 11)
    assert np.all(input_data["airmass_type"] == 0)
    tbs = input_data.brightness_temperatures.data
    valid = tbs > 0
    tbs = tbs[valid]
    assert np.all(tbs > 20)
    assert np.all(tbs < 400)


def test_file_processor_gmi(tmp_path):
    """
    This tests the extraction of data from a bin file and ensures that
    the extracted dataset matches the original data.
    """
    path = Path(__file__).parent
    processor = FileProcessor(
        path / "data" / "gmi" / "bin" / "processor", include_profiles=True
    )
    output_file = tmp_path / "test_file.nc"
    processor.run_async(output_file, 0.0, 1.0, 1)

    input_file = BinFile(path / "data" / "gmi" / "bin" / "gpm_275_14_03_17.bin")
    input_data = input_file.to_xarray_dataset()

    dataset = GPROF_NN_1D_Dataset(
        output_file, normalize=False, shuffle=False, augment=False
    )
    normalizer = dataset.normalizer

    tbs_input = input_data["brightness_temperatures"].data
    tbs = dataset.x[:, : input_file.sensor.n_chans]
    tbs = np.nan_to_num(tbs, nan=-9999.9)
    valid = tbs[:, -1] > 0

    assert np.all(np.isclose(tbs_input[valid].mean(), tbs[valid].mean()))
    assert np.all(
        np.isclose(input_data["two_meter_temperature"].mean(), dataset.x[:, 15].mean())
    )
    assert np.all(
        np.isclose(
            input_data["total_column_water_vapor"].mean(), dataset.x[:, 16].mean()
        )
    )

    surface_types = np.where(dataset.x[:, 17:35])[1]
    assert np.all(np.isclose(input_data.surface_type, surface_types + 1))
    airmass_types = np.where(dataset.x[:, 35:])[1]
    assert np.all(np.isclose(np.maximum(input_data.airmass_type, 1), airmass_types))
