"""
This file contains tests for the reading and processing of .sim files
defined in 'gprof_nn.data.sim.py'.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.sim import (
    SimFile,
    apply_orographic_enhancement,
    collocate_targets,
    write_training_samples_1d,
    write_training_samples_3d,
    process_sim_file
)
from gprof_nn.data.preprocessor import PreprocessorFile


DATA_PATH = get_test_data_path()
HAS_ARCHIVES = Path(sensors.GMI.l1c_file_path).exists()

SIM_DATA = Path("/qdata1/pbrown/dbaseV8")
NEEDS_SIM_DATA = pytest.mark.skipif(
    not SIM_DATA.exists(), reason="Needs sim files."
)

@NEEDS_SIM_DATA
def test_open_sim_file_gmi():
    """
    Tests reading simulator output file for GMI.
    """
    input_file = SIM_DATA / "simV8/1810/GMI.dbsatTb.20181031.026559.sim"

    sim_file = SimFile(input_file)
    data = sim_file.to_xarray_dataset()

    assert "surface_precip" in data.variables.keys()
    assert "latent_heat" in data.variables.keys()
    assert "snow_water_content" in data.variables.keys()
    assert "rain_water_content" in data.variables.keys()

    valid = data.surface_precip.data > -9999
    assert valid.sum() > 0
    assert np.all(data.surface_precip[valid] >= 0.0)
    assert np.all(data.surface_precip[valid] <= 1000.0)
    assert np.all(data.latitude >= -90.0)
    assert np.all(data.latitude <= 90.0)
    assert np.all(data.longitude >= -180.0)
    assert np.all(data.longitude <= 180.0)

@NEEDS_SIM_DATA
def test_open_sim_file_mhs():
    """
    Tests reading simulator output file for MHS.
    """
    input_file = SIM_DATA / "simV8x_mhs/1810/MHS.dbsatTb.20181031.026559.sim"

    sim_file = SimFile(input_file)
    data = sim_file.to_xarray_dataset()

    assert "surface_precip" in data.variables.keys()
    assert "latent_heat" in data.variables.keys()
    assert "snow_water_content" in data.variables.keys()
    assert "rain_water_content" in data.variables.keys()

    valid = data.surface_precip.data > -9999
    assert valid.sum() > 0
    assert np.all(data.surface_precip.data[valid] >= 0.0)
    assert np.all(data.surface_precip.data[valid] <= 1000.0)
    assert np.all(data.latitude >= -90.0)
    assert np.all(data.latitude <= 90.0)
    assert np.all(data.longitude >= -180.0)
    assert np.all(data.longitude <= 180.0)
    assert data.angles.data.max() > 50


@NEEDS_SIM_DATA
def test_collocate_targets(tmp_path):
    """
    Test collocating L1C file with simulator data.
    """
    input_file = SIM_DATA / "simV8/1810/GMI.dbsatTb.20181031.026559.sim"
    data = collocate_targets(
        input_file,
        sensors.GMI,
        None,
    )
    sp = data.surface_precip.data
    assert (sp >= 0.0).sum() > 0

    output_path = tmp_path / "1d"
    output_path.mkdir()
    write_training_samples_1d(output_path, "sim", data)
    training_files = list(output_path.glob("*.nc"))

    assert len(training_files) == 1
    with xr.open_dataset(training_files[0]) as training_data:
        assert training_data.attrs["sensor"] == "GMI"

    output_path = tmp_path / "3d"
    output_path.mkdir()
    write_training_samples_3d(output_path, "sim", data)
    training_files = list(output_path.glob("*.nc"))
    assert len(training_files) > 1
    with xr.open_dataset(training_files[0]) as data:
        assert data.attrs["sensor"] == "GMI"


def test_extract_scenes_bounded(tmp_path) -> xr.Dataset:
    """
    Ensure that extracting training data with given lon/lat bounds only contains valid
    data within this domain.
    """
    input_file = SIM_DATA / "simV8/1810/GMI.dbsatTb.20181001.026082.sim"

    path_1d = tmp_path / "1d"
    path_1d.mkdir()
    path_3d = tmp_path / "3d"
    path_3d.mkdir()

    process_sim_file(
        sensors.GMI,
        input_file,
        None,
        path_1d,
        path_3d,
        lonlat_bounds=(120, -25, -110, 25)
    )

    extracted = sorted(list(path_1d.glob("*.nc")))
    assert len(extracted) == 1

    td_1d = xr.load_dataset(extracted[0])

    lons = td_1d.longitude.data
    lats = td_1d.latitude.data

    assert lons.size > 0
    assert (lats >= -25).all()
    assert (lats <= 25).all()
    assert not ((lons > -110) * (lons < 120)).any()
