"""
This file tests the 'gprof_nn.data' module.
"""
from pathlib import Path

import gprof_nn.data
from gprof_nn import sensors

def test_get_model_path(tmpdir, monkeypatch):
    """
    Test that retrieving a model file from the server downloads
    the requested file if not present.
    """
    path = tmpdir
    monkeypatch.setattr(gprof_nn.data, "_DATA_DIR", tmpdir)
    tmpdir.mkdir("models")

    path = gprof_nn.data.get_model_path("1D", sensors.GMI, "ERA5")

    assert path.exists()
    assert (tmpdir / "models" / "gprof_nn_1d_gmi_era5.pckl").exists()
