"""
Tests for the gprof_nn.legacy module.
"""
from pathlib import Path

import numpy as np
import pytest

from gprof_nn.data.training_data import (GPROF_NN_0D_Dataset,
                                         write_preprocessor_file)
from gprof_nn.legacy  import (write_sensitivity_file,
                              DEFAULT_SENSITIVITIES,
                              execute_gprof,
                              has_gprof)
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
    with open(sensitivity_file, "r") as f:
        print("")
        print(f.read())


@pytest.mark.skipif(not HAS_GPROF, reason="GPROF executable missing.")
def test_run_gprof(tmp_path):
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"
    dataset = GPROF_NN_0D_Dataset(input_file).to_xarray_dataset()

    write_preprocessor_file(dataset, tmp_path / "input.pp")

    execute_gprof(tmp_path,
                  tmp_path / "input.pp",
                  "STANDARD",
                  False)
