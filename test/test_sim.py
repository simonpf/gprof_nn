from pathlib import Path

import numpy as np

from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.sim import (GMISimFile,
                               _load_era5_data,
                               _add_era5_precip,
                               apply_orographic_enhancement)

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


def test_orographic_enhancement(tmp_path):
    """
    Writes dataset to preprocessor file and ensures that the
    data from the preprocessor file matches that in the original
    dataset.
    """
    path = Path(__file__).parent / "data"

    preprocessor_file = PreprocessorFile(path / "GMIERA5_190101_027510.pp")
    input_data = preprocessor_file.to_xarray_dataset()
    sim_file = GMISimFile(path / "GMI.dbsatTb.20190101.027510.sim")
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
    assert np.all(np.isclose(sp[indices] * 2.05213, sp_ref[indices]))
    assert np.all(np.isclose(cp[indices] * 2.05213, cp_ref[indices]))

    indices = (st != 17) * (st != 18) * np.isfinite(sp)
    assert np.all(np.isclose(sp[indices], sp_ref[indices]))
    assert np.all(np.isclose(cp[indices], cp_ref[indices]))

