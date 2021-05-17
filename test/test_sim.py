from pathlib import Path

from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.sim import (GMISimFile,
                               _load_era5_data,
                               _add_era5_precip)

from gprof_nn.data.preprocessor import PreprocessorFile

def test_find_granule():
    """
    Test finding of specific L1C file and reading data into
    xarray.Dataset.
    """
    data_path = Path(__file__).parent / "data"
    l1c_file = L1CFile.open_granule(27510, data_path)
    l1c_data = l1c_file.to_xarray_dataset()
    sim_file = GMISimFile(data_path / "GMI.dbsatTb.20190101.027510.sim")

    targets = ["latent_heat",
               "rain_water_content",
               "ice_water_path"]

    sim_file.match_targets(l1c_data, targets=targets)

    assert "latent_heat" in l1c_data.variables.keys()
    assert "ice_water_path" in l1c_data.variables.keys()
    assert "snow_water_content" in l1c_data.variables.keys()
    assert "rain_water_content" in l1c_data.variables.keys()

def test_find_files():
    """
    Assert that find_file functions successfully finds file in test data folder
    except when search is restricted to a different day.
    """
    path = Path(__file__).parent / "data"
    sim_files = GMISimFile.find_files(path)
    assert len(sim_files) == 1

    sim_files = GMISimFile.find_files(path, day=1)
    assert len(sim_files) == 1

    sim_files = GMISimFile.find_files(path, day=2)
    assert len(sim_files) == 0




def test_match_era5_precip():
    """
    Test loading and matching of data from ERA5.
    """
    data_path = Path(__file__).parent / "data"
    l1c_file = L1CFile.open_granule(27510, data_path)
    l1c_data = l1c_file.to_xarray_dataset()

    preprocessor_file = PreprocessorFile(
        data_path / "GMIERA5_190101_027510.pp"
    )
    input_data = preprocessor_file.to_xarray_dataset()

    start_time = l1c_data["scan_time"][0].data
    end_time = l1c_data["scan_time"][-1].data
    era5_data = _load_era5_data(start_time,
                                end_time,
                                data_path)

    sim_file = GMISimFile(data_path / "GMI.dbsatTb.20190101.027510.sim")
    sim_file.match_targets(input_data)
    _add_era5_precip(input_data,
                     l1c_data,
                     era5_data)
