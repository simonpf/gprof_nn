"""
Tests for the gprof_nn.data.pretraining module.
"""
from pathlib import Path

from conftest import NEEDS_ARCHIVES, l1c_file_gmi

import pytest
import xarray as xr
import numpy as np

from gprof_nn import sensors
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data.pretraining import process_l1c_file
from gprof_nn.data.training_data import PretrainingDataset
from gprof_nn.models import GPROFNet3D


@pytest.fixture(scope="module")
def gmi_pretraining_data(tmp_path_factory, l1c_file_gmi):
    input_path = tmp_path_factory.mktemp("l1c")
    input_file = input_path / l1c_file_gmi.name
    L1CFile(l1c_file_gmi).extract_scan_range(0, 512, input_file)

    path = tmp_path_factory.mktemp("data")
    scenes = process_l1c_file(sensors.GMI, input_file, path)
    return path


@NEEDS_ARCHIVES
def test_process_l1c_file_gmi(gmi_pretraining_data):
    """
    Assert that processing a l1c file produces sample files suitable for
    pretraining.
    """
    files = sorted(list(gmi_pretraining_data.glob("*.nc")))
    assert len(files) > 0

    data = xr.load_dataset(files[0])
    data.brightness_temperatures[..., 0].min() > 0.0


@NEEDS_ARCHIVES
def test_pretraining_dataset(gmi_pretraining_data):
    """
    Assert that data loaded using the PretrainingDataset is valid.
    """
    dataset = PretrainingDataset(gmi_pretraining_data)
    for ind in range(len(dataset)):
        x, y = dataset[ind]
        x = x.numpy()
        assert x.min() >= -1.5
        assert x.max() <= 1.0
        for key in y:
            assert np.all(np.isfinite(y[key].numpy()))


@NEEDS_ARCHIVES
def test_pretraining_model(gmi_pretraining_data):

    dataset = PretrainingDataset(gmi_pretraining_data)

    x, y = dataset[0]
    x = x[None]

    targets = {f"channel_{ind}": (32,) for ind in range(15)}

    model = GPROFNet3D(
        [32, 64, 128, 256, 512],
        [2, 2, 2, 2, 2],
        targets=targets,
        ancillary_data=True
    )
    y = model(x)

    assert "channel_1" in y
