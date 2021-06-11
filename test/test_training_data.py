"""
Tests for the Pytorch dataset classes used to load the training
data.
"""
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from quantnn.qrnn import QRNN
from quantnn.normalizer import Normalizer
from quantnn.models.pytorch.xception import XceptionFpn

from gprof_nn.data.training_data import (GPROF0DDataset,
                                         run_retrieval_0d,
                                         GPROF2DDataset)


def test_gprof_0d_dataset():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "training_data.nc"
    dataset = GPROF0DDataset(input_file, batch_size=1, augment=False)

    xs = []
    ys = []

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = dataset.y.sum(axis=0)

    for x, y in dataset:
        xs.append(x)
        ys.append(y)

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    x_mean = xs.sum(dim=0).detach().numpy()
    y_mean = ys.sum(dim=0).detach().numpy()

    assert np.all(np.isclose(x_mean, x_mean_ref, rtol=1e-3))
    assert np.all(np.isclose(y_mean, y_mean_ref, rtol=1e-3))


def test_gprof_0d_dataset_multi_target():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "training_data.nc"
    dataset = GPROF0DDataset(
        input_file, target=["surface_precip", "rain_water_content"], batch_size=1
    )

    xs = []
    ys = {}

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = {k: dataset.y[k].sum(axis=0) for k in dataset.y}

    for x, y in dataset:
        xs.append(x)
        for k in y:
            ys.setdefault(k, []).append(y[k])

    xs = torch.cat(xs, dim=0)
    ys = {k: torch.cat(ys[k], dim=0) for k in ys}
    ys = {k: torch.where(torch.isnan(ys[k]), torch.zeros_like(ys[k]), ys[k])
          for k in ys}

    x_mean = xs.sum(dim=0).detach().numpy()
    y_mean = {k: ys[k].sum(dim=0).detach().numpy() for k in ys}

    assert np.all(np.isclose(x_mean, x_mean_ref, atol=1e-3))
    for k in y_mean_ref:
        assert np.all(np.isclose(y_mean[k], y_mean_ref[k], rtol=1e-3))


def test_profile_variables():
    """
    Ensure profile variables are available everywhere except over sea ice
    or snow.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "training_data.nc"

    PROFILE_TARGETS = [
        "rain_water_content",
        "snow_water_content",
        "cloud_water_content",
        "latent_heat"
    ]
    dataset = GPROF0DDataset(
        input_file, target=PROFILE_TARGETS, batch_size=1
    )

    for t in PROFILE_TARGETS:
        x = dataset.x
        y = dataset.y[t]

        st = np.where(x[:, 17:35])[1]
        indices = (st >= 8) * (st <= 11)


def test_run_retrieval_0d(tmp_path):
    """
    Test running 0D version of retrieval on training data file.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "training_data.nc"
    qrnn = QRNN.load(path / "data" / "gprof_nn_0d.pckl")
    normalizer = Normalizer.load(path / "data" / "normalizer.pckl")
    run_retrieval_0d(input_file,
                     qrnn,
                     normalizer,
                     tmp_path / "results.nc")

    results = xr.load_dataset(tmp_path / "results.nc")
    assert "surface_precip" in results.variables


def test_run_retrieval_0d_sim(tmp_path):
    """
    Test running 0D version of retrieval on training data file extracted
    from bin data.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "training_data_bin.nc"
    qrnn = QRNN.load(path / "data" / "gprof_nn_0d.pckl")
    normalizer = Normalizer.load(path / "data" / "normalizer.pckl")
    run_retrieval_0d(input_file,
                     qrnn,
                     normalizer,
                     tmp_path / "results.nc")

    results = xr.load_dataset(tmp_path / "results.nc")
    assert "surface_precip" in results.variables


def test_gprof_2d_dataset():
    """
    Ensure that iterating over 2D dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "training_data.nc"
    dataset = GPROF2DDataset(input_file,
                             batch_size=1,
                             augment=False,
                             transform_zeros=True)

    xs = []
    ys = []

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = dataset.y.sum(axis=0)

    for x, y in dataset:
        xs.append(x)
        ys.append(y)

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    x_mean = xs.sum(dim=0).detach().numpy()
    y_mean = ys.sum(dim=0).detach().numpy()

    y_mean = y_mean[np.isfinite(y_mean)]
    y_mean_ref = y_mean_ref[np.isfinite(y_mean_ref)]

    assert np.all(np.isclose(x_mean, x_mean_ref, atol=1e-3))
    assert np.all(np.isclose(y_mean, y_mean_ref, atol=1e-3))


def test_gprof_2d_dataset_profiles():
    """
    Ensure that loading of profile variables works.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "training_data.nc"
    dataset = GPROF2DDataset(input_file,
                             batch_size=1,
                             augment=False,
                             transform_zeros=True,
                             target=[
                                 "rain_water_content",
                                 "snow_water_content",
                                 "cloud_water_content"
                             ])

    xs = []
    ys = {}

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = {}
    for k in dataset.target:
        y_mean_ref[k] = dataset.y[k].sum(axis=0)

    for x, y in dataset:
        xs.append(x)
        for k in y:
            ys.setdefault(k, []).append(y[k])

    xs = torch.cat(xs, dim=0)
    for k in dataset.target:
        ys[k] = torch.cat(ys[k], dim=0)

    x_mean = xs.sum(dim=0).detach().numpy()
    y_mean = {}
    for k in dataset.target:
        y_mean[k] = ys[k].sum(dim=0).detach().numpy()

    for k in dataset.target:
        y_mean[k] = y_mean[k][np.isfinite(y_mean[k])]
        y_mean_ref[k] = y_mean_ref[k][np.isfinite(y_mean_ref[k])]

    assert np.all(np.isclose(x_mean, x_mean_ref, atol=1e-3))
    for k in dataset.target:
        assert np.all(np.isclose(y_mean[k], y_mean_ref[k], atol=1e-3))
