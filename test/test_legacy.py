"""
Tests for the gprof_nn.legacy module.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.data.training_data import (GPROF_NN_1D_Dataset,
                                         write_preprocessor_file)
from gprof_nn.legacy import (has_gprof,
                             write_sensitivity_file,
                             DEFAULT_SENSITIVITIES,
                             run_gprof_training_data,
                             run_gprof_standard)
from gprof_nn.data.preprocessor import PreprocessorFile


DATA_PATH = get_test_data_path()


HAS_GPROF = has_gprof()


def test_write_sensitivity_file(tmp_path):
    """
    Write sensitivity file containing default sensitivities and ensure
    that sensitivities in files match the original ones.
    """
    nedts_ref = DEFAULT_SENSITIVITIES
    sensitivity_file = tmp_path / "sensitivities.txt"
    write_sensitivity_file(sensors.GMI, sensitivity_file, nedts=nedts_ref)
    nedts = np.loadtxt(sensitivity_file)
    assert np.all(np.isclose(nedts_ref, nedts))


@pytest.mark.skipif(not HAS_GPROF, reason="GPROF executable missing.")
def test_run_gprof_training_data():
    """
    Test running the legacy GPROF algorithm on training data.
    """
    path = Path(__file__).parent
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc"

    results = run_gprof_training_data(sensors.GMI,
                                      "ERA5",
                                      input_file,
                                      "STANDARD",
                                      False)
    assert "surface_precip" in results.variables
    assert "surface_precip_true" in results.variables


@pytest.mark.skipif(not HAS_GPROF, reason="GPROF executable missing.")
def test_run_gprof_training_data_preserve_structure():
    """
    Test running the legacy GPROF algorithm on training data while
    preserving the spatial structure.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc"

    results = run_gprof_training_data(sensors.GMI,
                                      "ERA5",
                                      input_file,
                                      "STANDARD",
                                      False,
                                      preserve_structure=True)
    assert "surface_precip" in results.variables
    assert "surface_precip_true" in results.variables


@pytest.mark.skipif(not HAS_GPROF, reason="GPROF executable missing.")
def test_run_gprof_standard():
    """
    Test running legacy GPROF on a preprocessor input file.
    """
    input_file = DATA_PATH / "gmi" / "pp" / "GMIERA5_190101_027510.pp"
    results = run_gprof_standard(sensors.GMI,
                                 "ERA5",
                                 input_file,
                                 "STANDARD",
                                 False)
    assert "surface_precip" in results.variables
