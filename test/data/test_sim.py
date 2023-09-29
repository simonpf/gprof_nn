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
    SubsetConfig,
    apply_orographic_enhancement,
    collocate_targets,
    write_training_samples_1d,
    write_training_samples_3d,
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
def test_collocate_targets(tmp_path):

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
    write_training_samples_1d(data, output_path)
    training_files = list(output_path.glob("*.nc"))
    assert len(training_files) == 1

    output_path = tmp_path / "3d"
    output_path.mkdir()
    write_training_samples_3d(data, output_path)
    training_files = list(output_path.glob("*.nc"))
    assert len(training_files) > 1
