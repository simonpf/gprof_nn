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

from gprof_nn import sensors
from gprof_nn.data.training_data import (GPROF0DDataset,
                                         TrainingObsDataset0D,
                                         GPROF2DDataset,
                                         SimulatorDataset)


def test_permutation_gmi():
    """
    Ensure that permutation permutes the right input features.
    """
    # Permute continuous input
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"
    dataset_1 = GPROF0DDataset(input_file,
                               batch_size=16,
                               shuffle=False,
                               augment=False,
                               transform_zeros=False,
                               targets=["surface_precip"])
    dataset_2 = GPROF0DDataset(input_file,
                               batch_size=16,
                               shuffle=False,
                               augment=False,
                               transform_zeros=False,
                               targets=["surface_precip"],
                               permute=0)
    x_1, y_1 = dataset_1[0]
    y_1 = y_1["surface_precip"]
    x_2, y_2 = dataset_2[0]
    y_2 = y_2["surface_precip"]

    assert np.all(np.isclose(y_1, y_2))
    assert ~np.all(np.isclose(x_1[:, :1], x_2[:, :1]))
    assert np.all(np.isclose(x_1[:, 1:], x_2[:, 1:]))

    # Permute surface type
    dataset_2 = GPROF0DDataset(input_file,
                               batch_size=16,
                               shuffle=False,
                               augment=False,
                               transform_zeros=False,
                               targets=["surface_precip"],
                               permute=17)
    x_2, y_2 = dataset_2[0]
    y_2 = y_2["surface_precip"]

    assert np.all(np.isclose(y_1, y_2))
    assert np.all(np.isclose(x_1[:, :-24], x_2[:, :-24]))
    assert ~np.all(np.isclose(x_1[:, -24:-4], x_2[:, -24:-4]))
    assert np.all(np.isclose(x_1[:, -4:], x_2[:, -4:]))

    # Permute airmass type
    dataset_2 = GPROF0DDataset(input_file,
                               batch_size=16,
                               shuffle=False,
                               augment=False,
                               transform_zeros=False,
                               targets=["surface_precip"],
                               permute=18)
    x_2, y_2 = dataset_2[0]
    y_2 = y_2["surface_precip"]

    assert np.all(np.isclose(y_1, y_2))
    assert np.all(np.isclose(x_1[:, :-4], x_2[:, :-4]))


def test_gprof_0d_dataset_gmi():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"
    dataset = GPROF0DDataset(input_file,
                             batch_size=1,
                             augment=False,
                             targets=["surface_precip"])

    xs = []
    ys = []

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = dataset.y["surface_precip"].sum(axis=0)

    for x, y in dataset:
        xs.append(x)
        ys.append(y["surface_precip"])

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    x_mean = xs.sum(dim=0).detach().numpy()
    y_mean = ys.sum(dim=0).detach().numpy()

    assert np.all(np.isclose(x_mean, x_mean_ref, rtol=1e-3))
    assert np.all(np.isclose(y_mean, y_mean_ref, rtol=1e-3))


def test_gprof_0d_dataset_multi_target_gmi():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"
    dataset = GPROF0DDataset(
        input_file,
        targets=["surface_precip",
                 "latent_heat",
                 "rain_water_content"],
        batch_size=1,
        transform_zeros=False

    )

    xs = []
    ys = {}

    x_mean_ref = np.sum(dataset.x, axis=0)
    y_mean_ref = {k: np.sum(dataset.y[k], axis=0) for k in dataset.y}

    for x, y in dataset:
        xs.append(x)
        for k in y:
            ys.setdefault(k, []).append(y[k])

    xs = torch.cat(xs, dim=0)
    ys = {k: torch.cat(ys[k], dim=0) for k in ys}

    x_mean = np.sum(xs.detach().numpy(), axis=0)
    y_mean = {k: np.sum(ys[k].detach().numpy(), axis=0) for k in ys}

    assert np.all(np.isclose(x_mean, x_mean_ref, atol=1e-3))
    for k in y_mean_ref:
        assert np.all(np.isclose(y_mean[k], y_mean_ref[k], rtol=1e-3))


def test_gprof_0d_dataset_mhs():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gprof_nn_mhs_era5.nc"
    dataset = GPROF0DDataset(input_file,
                             batch_size=1,
                             augment=False,
                             targets=["surface_precip"],
                             sensor=sensors.MHS)

    xs = []
    ys = []

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = dataset.y["surface_precip"].sum(axis=0)

    for x, y in dataset:
        xs.append(x)
        ys.append(y["surface_precip"])

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    x_mean = xs.sum(dim=0).detach().numpy()
    y_mean = ys.sum(dim=0).detach().numpy()

    assert np.all(np.isclose(x_mean, x_mean_ref, rtol=1e-3))
    assert np.all(np.isclose(y_mean, y_mean_ref, rtol=1e-3))

    assert(np.all(np.isclose(x[:, 8:26].sum(-1),
                             1.0)))


def test_gprof_0d_dataset_multi_target_mhs():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gprof_nn_mhs_era5.nc"
    dataset = GPROF0DDataset(
        input_file,
        targets=["surface_precip",
                 "latent_heat",
                 "rain_water_content"],
        batch_size=1,
        transform_zeros=False,
        sensor=sensors.MHS
    )

    xs = []
    ys = {}

    x_mean_ref = np.sum(dataset.x, axis=0)
    y_mean_ref = {k: np.sum(dataset.y[k], axis=0) for k in dataset.y}

    for x, y in dataset:
        xs.append(x)
        for k in y:
            ys.setdefault(k, []).append(y[k])

    xs = torch.cat(xs, dim=0)
    ys = {k: torch.cat(ys[k], dim=0) for k in ys}

    x_mean = np.sum(xs.detach().numpy(), axis=0)
    y_mean = {k: np.sum(ys[k].detach().numpy(), axis=0) for k in ys}

    assert np.all(np.isclose(x_mean, x_mean_ref, atol=1e-3))
    for k in y_mean_ref:
        assert np.all(np.isclose(y_mean[k], y_mean_ref[k], rtol=1e-3))


def test_observation_dataset_0d():
    """
    Test loading of observations data from MHS training data.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gprof_nn_mhs_era5.nc"
    input_data = xr.load_dataset(input_file)
    dataset = TrainingObsDataset0D(
        input_file,
        batch_size=1,
        sensor=sensors.MHS,
        normalize=False,
        shuffle=False
    )

    x, y = dataset[0]
    x = x.detach().numpy()
    y = y.detach().numpy()

    assert x.shape[1] == 19
    assert y.shape[1] == 5

    sp = input_data["surface_precip"].data
    valid = np.all(sp >= 0, axis=-1)
    st = input_data["surface_type"].data[valid]
    st_x = np.where(x[0, 1:])[0][0] + 1
    assert st[0] == st_x


def test_profile_variables():
    """
    Ensure profile variables are available everywhere except over sea ice
    or snow.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"

    PROFILE_TARGETS = [
        "rain_water_content",
        "snow_water_content",
        "cloud_water_content",
        "latent_heat"
    ]
    dataset = GPROF0DDataset(
        input_file, targets=PROFILE_TARGETS, batch_size=1
    )

    for t in PROFILE_TARGETS:
        x = dataset.x
        y = dataset.y[t]

        st = np.where(x[:, 17:35])[1]
        indices = (st >= 8) * (st <= 11)


def test_gprof_2d_dataset_gmi():
    """
    Ensure that iterating over 2D dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"
    dataset = GPROF2DDataset(input_file,
                             batch_size=1,
                             augment=False,
                             transform_zeros=True)

    xs = []
    ys = []

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = dataset.y["surface_precip"].sum(axis=0)

    for x, y in dataset:
        xs.append(x)
        ys.append(y["surface_precip"])

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
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"
    dataset = GPROF2DDataset(input_file,
                             batch_size=1,
                             augment=False,
                             transform_zeros=True,
                             targets=[
                                 "rain_water_content",
                                 "snow_water_content",
                                 "cloud_water_content"
                             ])

    xs = []
    ys = {}

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = {}
    for k in dataset.targets:
        y_mean_ref[k] = dataset.y[k].sum(axis=0)

    for x, y in dataset:
        xs.append(x)
        for k in y:
            ys.setdefault(k, []).append(y[k])

    xs = torch.cat(xs, dim=0)
    for k in dataset.targets:
        ys[k] = torch.cat(ys[k], dim=0)

    x_mean = xs.sum(dim=0).detach().numpy()
    y_mean = {}
    for k in dataset.targets:
        y_mean[k] = ys[k].sum(dim=0).detach().numpy()

    for k in dataset.targets:
        y_mean[k] = y_mean[k][np.isfinite(y_mean[k])]
        y_mean_ref[k] = y_mean_ref[k][np.isfinite(y_mean_ref[k])]

    assert np.all(np.isclose(x_mean, x_mean_ref, atol=1e-3))
    for k in dataset.targets:
        assert np.all(np.isclose(y_mean[k], y_mean_ref[k], atol=1e-3))


def test_gprof_2d_dataset_mhs():
    """
    Test loading of 2D training data for MHS sensor.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gprof_nn_mhs_era5_sim.nc"
    dataset = GPROF2DDataset(input_file,
                             batch_size=1,
                             augment=False,
                             transform_zeros=True)

    xs = []
    ys = []

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = dataset.y["surface_precip"].sum(axis=0)

    for x, y in dataset:
        xs.append(x)
        ys.append(y["surface_precip"])

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    x_mean = xs.sum(dim=0).detach().numpy()
    y_mean = ys.sum(dim=0).detach().numpy()

    y_mean = y_mean[np.isfinite(y_mean)]
    y_mean_ref = y_mean_ref[np.isfinite(y_mean_ref)]

    assert np.all(np.isclose(x_mean, x_mean_ref, atol=1e-3))
    assert np.all(np.isclose(y_mean, y_mean_ref, atol=1e-3))


def test_simulator_dataset_gmi():
    """
    Test loading of simulator training data.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"
    dataset = SimulatorDataset(input_file,
                               normalize=False,
                               batch_size=1024)
    x, y = dataset[0]
    x = x.numpy()
    y = {k: y[k].numpy() for k in y}

    # Input Tbs must match simulated plus biases.
    for i in range(x.shape[0]):
        tbs_in = x[i, :15, :, :]
        tbs_sim = y["simulated_brightness_temperatures"][i, :, :, :]
        tbs_bias = y["brightness_temperature_biases"][i, :, :, :]
        tbs_sim[tbs_sim <= -9000] = np.nan
        tbs_bias[tbs_bias <= -9000] = np.nan
        tbs_out = tbs_sim - tbs_bias

        tbs_in = tbs_in[np.isfinite(tbs_out)]
        tbs_out = tbs_out[np.isfinite(tbs_out)]

        if tbs_in.size == 0:
            continue
        assert np.all(np.isclose(tbs_in, tbs_out, atol=1e-3))


def test_simulator_dataset_mhs():
    """
    Test loading of simulator training data.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gprof_nn_mhs_era5_5.nc"
    dataset = SimulatorDataset(input_file,
                               batch_size=1024,
                               augment=True)
    x, y = dataset[0]

    assert np.all(np.isfinite(x.numpy()))
    assert np.all(np.isfinite(y["brightness_temperature_biases"].numpy()))
    assert np.all(np.isfinite(y["simulated_brightness_temperatures"].numpy()))
    assert "brightness_temperature_biases" in y
    assert len(y["brightness_temperature_biases"].shape) == 4
    assert "simulated_brightness_temperatures" in y
    assert len(y["simulated_brightness_temperatures"].shape) == 5
