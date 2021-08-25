"""
Tests for the gprof_nn.legacy module.
"""
import numpy as np

from gprof_nn.legacy  import (write_sensitivity_file,
                              DEFAULT_SENSITIVITIES)


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


