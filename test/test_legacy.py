"""
Tests for the gprof_nn.legacy module.
"""
from pathlib import Path

import numpy as np
import pytest

from gprof_nn.data.training_data import (GPROF_NN_0D_Dataset,
                                         write_preprocessor_file)
from gprof_nn.legacy import (has_gprof,
                             write_sensitivity_file,
                             DEFAULT_SENSITIVITIES,
                             run_gprof_training_data,
                             run_gprof_standard)
from gprof_nn.data.preprocessor import PreprocessorFile


HAS_GPROF = has_gprof()


def test_write_sensitivity_file(tmp_path):
    """
    Write sensitivity file containing default sensitivities and ensure
    that sensitivities in files match the original ones.
    """
    nedts_ref = DEFAULT_SENSITIVITIES
    sensitivity_file = tmp_path / "sensitivities.txt"
    write_sensitivity_file(sensitivity_file, nedts=nedts_ref)
    nedts = np.loadtxt(sensitivity_file)
    assert np.all(np.isclose(nedts_ref, nedts))


@pytest.mark.skipif(not HAS_GPROF, reason="GPROF executable missing.")
def test_run_gprof_training_data(tmp_path):
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"

    results = run_gprof_training_data(input_file,
                                      "STANDARD",
                                      False)
    assert "surface_precip" in results.variables
    assert "surface_precip_true" in results.variables


@pytest.mark.skipif(not HAS_GPROF, reason="GPROF executable missing.")
def test_run_gprof_standard(tmp_path):
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "GMIERA5_190101_027510.pp"
    results = run_gprof_standard(input_file,
                                 "STANDARD",
                                 False)
    assert "surface_precip" in results.variables
