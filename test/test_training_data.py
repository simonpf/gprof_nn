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
from gprof_nn.data import get_test_data_path
from gprof_nn.data.training_data import (
    load_variable,
    decompress_scene,
    decompress_and_load,
    remap_scene,
    GPROF_NN_1D_Dataset,
    GPROF_NN_3D_Dataset,
    SimulatorDataset,
)


DATA_PATH = get_test_data_path()


def test_to_xarray_dataset_1d_gmi():
    """
    Ensure that converting training data to 'xarray.Dataset' yield same
    Tbs as the ones found in the first batch of the training data when
    data is not shuffled.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_1D_Dataset(
        input_file,
        batch_size=64,
        normalize=False,
        shuffle=False,
        targets=["surface_precip", "rain_water_content"]
    )

    #
    # Conversion using datasets 'x' attribute.
    #

    data = dataset.to_xarray_dataset()
    x, y = dataset[0]
    x = x.numpy()

    tbs = data.brightness_temperatures.data[:x.shape[0]]
    tbs_ref = x[:, :15]
    valid = np.isfinite(tbs_ref)
    assert np.all(np.isclose(tbs[valid], tbs_ref[valid]))

    t2m = data.two_meter_temperature.data[:x.shape[0]]
    t2m_ref = x[:, 15]
    assert np.all(np.isclose(t2m, t2m_ref))

    tcwv = data.total_column_water_vapor.data[:x.shape[0]]
    tcwv_ref = x[:, 16]
    assert np.all(np.isclose(tcwv, tcwv_ref))

    st = data.surface_type.data[:x.shape[0]]
    inds, st_ref = np.where(x[:, -22:-4])
    assert np.all(np.isclose(st[inds], st_ref + 1))

    at = data.airmass_type.data[:x.shape[0]]
    inds, at_ref = np.where(x[:, -4:])
    assert np.all(np.isclose(at[inds], at_ref))

    #
    # Conversion using only first batch
    #

    x, y = dataset[0]
    data = dataset.to_xarray_dataset(batch=(x, y))
    x = x.numpy()

    tbs = data.brightness_temperatures.data
    tbs_ref = x[:, :15]
    valid = np.isfinite(tbs_ref)
    assert np.all(np.isclose(tbs[valid], tbs_ref[valid]))

    t2m = data.two_meter_temperature.data
    t2m_ref = x[:, 15]
    assert np.all(np.isclose(t2m, t2m_ref))

    tcwv = data.total_column_water_vapor.data
    tcwv_ref = x[:, 16]
    assert np.all(np.isclose(tcwv, tcwv_ref))

    st = data.surface_type.data
    inds, st_ref = np.where(x[:, -22:-4])
    assert np.all(np.isclose(st[inds], st_ref + 1))

    at = data.airmass_type.data
    inds, at_ref = np.where(x[:, -4:])
    assert np.all(np.isclose(at[inds], at_ref))


def test_to_xarray_dataset_1d_mhs():
    """
    Ensure that converting training data to 'xarray.Dataset' yield same
    Tbs as the ones found in the first batch of the training data when
    data is not shuffled.
    """
    input_file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc.gz"
    dataset = GPROF_NN_1D_Dataset(
        input_file,
        batch_size=64,
        normalize=False,
        shuffle=False,
        targets=["surface_precip", "rain_water_content"]
    )

    #
    # Conversion using datasets 'x' attribute.
    #

    data = dataset.to_xarray_dataset()
    x, y = dataset[0]
    x = x.numpy()

    t2m = data.two_meter_temperature.data[:x.shape[0]]
    t2m_ref = x[:, 6]
    assert np.all(np.isclose(t2m, t2m_ref))

    tcwv = data.total_column_water_vapor.data[:x.shape[0]]
    tcwv_ref = x[:, 7]
    assert np.all(np.isclose(tcwv, tcwv_ref))

    st = data.surface_type.data[:x.shape[0]]
    inds, st_ref = np.where(x[:, -22:-4])
    assert np.all(np.isclose(st[inds], st_ref + 1))

    at = data.airmass_type.data[:x.shape[0]]
    inds, at_ref = np.where(x[:, -4:])
    assert np.all(np.isclose(at[inds], at_ref))


def test_to_xarray_dataset_3d():
    """
    Ensure that converting training data to 'xarray.Dataset' yield same
    Tbs as the ones found in the first batch of the training data when
    data is not shuffled.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_3D_Dataset(
        input_file,
        batch_size=32,
        normalize=False,
        shuffle=False,
        targets=["surface_precip", "rain_water_content"]
    )
    data = dataset.to_xarray_dataset()
    x, y = dataset[0]
    x = x.numpy()

    tbs = data.brightness_temperatures.data
    tbs_ref = x[:, :15]
    tbs_ref = np.transpose(tbs_ref, (0, 2, 3, 1))
    valid = np.isfinite(tbs_ref)
    assert np.all(np.isclose(tbs[valid], tbs_ref[valid]))

    t2m = data.two_meter_temperature.data
    t2m_ref = x[:, 15]
    assert np.all(np.isclose(t2m, t2m_ref))

    tcwv = data.total_column_water_vapor.data
    tcwv_ref = x[:, 16]
    assert np.all(np.isclose(tcwv, tcwv_ref))

    st = data.surface_type.data
    st_ref = np.zeros(t2m.shape, dtype=np.int32)
    for i in range(18):
        mask = x[:, -22 + i] == 1
        st_ref[mask] = i + 1
    assert np.all(np.isclose(st, st_ref))

    at = data.airmass_type.data
    at_ref = np.zeros(t2m.shape, dtype=np.int32)
    for i in range(4):
        mask = x[:, -4 + i] == 1
        at_ref[mask] = i
    assert np.all(np.isclose(at, at_ref))


def test_permutation_gmi():
    """
    Ensure that permutation permutes the right input features.
    """
    # Permute continuous input
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset_1 = GPROF_NN_1D_Dataset(
        input_file,
        batch_size=16,
        shuffle=False,
        augment=False,
        transform_zeros=False,
        targets=["surface_precip"],
    )
    dataset_2 = GPROF_NN_1D_Dataset(
        input_file,
        batch_size=16,
        shuffle=False,
        augment=False,
        transform_zeros=False,
        targets=["surface_precip"],
        permute=0,
    )
    x_1, y_1 = dataset_1[0]
    y_1 = y_1["surface_precip"]
    x_2, y_2 = dataset_2[0]
    y_2 = y_2["surface_precip"]

    assert np.all(np.isclose(y_1, y_2))
    assert ~np.all(np.isclose(x_1[:, :1], x_2[:, :1]))
    assert np.all(np.isclose(x_1[:, 1:], x_2[:, 1:]))

    # Permute surface type
    dataset_2 = GPROF_NN_1D_Dataset(
        input_file,
        batch_size=16,
        shuffle=False,
        augment=False,
        transform_zeros=False,
        targets=["surface_precip"],
        permute=17,
    )
    x_2, y_2 = dataset_2[0]
    y_2 = y_2["surface_precip"]

    assert np.all(np.isclose(y_1, y_2))
    assert np.all(np.isclose(x_1[:, :-24], x_2[:, :-24]))
    assert ~np.all(np.isclose(x_1[:, -24:-4], x_2[:, -24:-4]))
    assert np.all(np.isclose(x_1[:, -4:], x_2[:, -4:]))

    # Permute airmass type
    dataset_2 = GPROF_NN_1D_Dataset(
        input_file,
        batch_size=16,
        shuffle=False,
        augment=False,
        transform_zeros=False,
        targets=["surface_precip"],
        permute=18,
    )
    x_2, y_2 = dataset_2[0]
    y_2 = y_2["surface_precip"]

    assert np.all(np.isclose(y_1, y_2))
    assert np.all(np.isclose(x_1[:, :-4], x_2[:, :-4]))


def test_gprof_1d_dataset_input_gmi():
    """
    Ensure that input variables have realistic values.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_1D_Dataset(
        input_file, batch_size=1, normalize=False, targets=["surface_precip"]
    )
    x, _ = dataset[0]
    x = x.numpy()

    tbs = x[:, :15]
    tbs = tbs[np.isfinite(tbs)]
    assert np.all((tbs > 30) * (tbs < 400))

    t2m = x[:, 15]
    assert np.all((t2m > 180) * (t2m < 350))

    tcwv = x[:, 16]
    assert np.all((tcwv > 0) * (tcwv < 100))


def test_gprof_1d_dataset_gmi():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_1D_Dataset(
        input_file, batch_size=1, augment=False, targets=["surface_precip"]
    )

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


def test_gprof_1d_dataset_multi_target_gmi():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_1D_Dataset(
        input_file,
        targets=["surface_precip", "latent_heat", "rain_water_content"],
        batch_size=1,
        transform_zeros=False,
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


def test_gprof_1d_dataset_mhs():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    input_file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc.gz"
    dataset = GPROF_NN_1D_Dataset(
        input_file,
        batch_size=1,
        augment=False,
        targets=["surface_precip"],
        sensor=sensors.MHS,
    )

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

    assert np.all(np.isclose(x[:, 8:26].sum(-1), 1.0))


def test_gprof_1d_dataset_multi_target_mhs():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    input_file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc.gz"
    dataset = GPROF_NN_1D_Dataset(
        input_file,
        targets=["surface_precip", "latent_heat", "rain_water_content"],
        batch_size=1,
        transform_zeros=False,
        sensor=sensors.MHS,
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


def test_gprof_1d_dataset_input_mhs():
    """
    Ensure that input variables have realistic values.
    """
    input_file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc.gz"
    dataset = GPROF_NN_1D_Dataset(
        input_file, batch_size=1, normalize=False, targets=["surface_precip"]
    )
    x, _ = dataset[0]
    x = x.numpy()

    tbs = x[:, :5]
    tbs = tbs[np.isfinite(tbs)]
    assert np.all((tbs > 30) * (tbs < 400))

    eia = x[:, 5]
    eia = eia[np.isfinite(eia)]
    assert np.all((eia >= -60) * (eia <= 60))

    t2m = x[:, 6]
    t2m = t2m[np.isfinite(t2m)]
    assert np.all((t2m > 180) * (t2m < 350))

    tcwv = x[:, 7]
    tcwv = tcwv[np.isfinite(tcwv)]
    assert np.all((tcwv > 0) * (tcwv < 100))


def test_gprof_1d_dataset_pretraining_mhs():
    """
    Test that the correct inputs are loaded when loading a pre-training dataset
    for MHS.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_1D_Dataset(
        input_file, batch_size=1, normalize=False, targets=["surface_precip"],
        sensor=sensors.MHS
    )

    x, _ = dataset[0]
    x = x.numpy()

    assert x.shape[1] == 5 + 3 + 18 + 4


def test_gprof_1d_dataset_pretraining_tmi():
    """
    Test that the correct inputs are loaded when loading a pre-training dataset
    for TMI.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_1D_Dataset(
        input_file, batch_size=1, normalize=False, targets=["surface_precip"],
        sensor=sensors.TMIPR
    )

    x, _ = dataset[0]
    x = x.numpy()

    assert x.shape[1] == 9 + 2 + 18 + 4


def test_profile_variables():
    """
    Test loading of profile variables.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "gmi" / "gprof_nn_gmi_era5.nc"

    PROFILE_TARGETS = [
        "rain_water_content",
        "snow_water_content",
        "cloud_water_content",
        "latent_heat",
    ]
    dataset = GPROF_NN_1D_Dataset(input_file, targets=PROFILE_TARGETS, batch_size=1)
    x, y = dataset[0]


def test_gprof_3d_dataset_input_gmi():
    """
    Ensure that input variables have realistic values.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_3D_Dataset(
        input_file, batch_size=1, normalize=False, targets=["surface_precip"]
    )
    x, _ = dataset[0]
    x = x.numpy()

    tbs = x[:, :15]
    tbs = tbs[np.isfinite(tbs)]
    assert np.all((tbs > 30) * (tbs < 400))

    t2m = x[:, 15]
    assert np.all((t2m > 180) * (t2m < 350))

    tcwv = x[:, 16]
    assert np.all((tcwv > 0) * (tcwv < 100))


def test_gprof_3d_dataset_gmi():
    """
    Ensure that iterating over 3D dataset conserves
    statistics.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_3D_Dataset(
        input_file, batch_size=1, augment=False, transform_zeros=True
    )

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


def test_gprof_3d_dataset_profiles():
    """
    Ensure that loading of profile variables works.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_3D_Dataset(
        input_file,
        batch_size=1,
        augment=False,
        transform_zeros=True,
        targets=["rain_water_content", "snow_water_content", "cloud_water_content"],
    )

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


def test_gprof_3d_dataset_input_mhs():
    """
    Ensure that input variables have realistic values.
    """
    input_file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc.gz"
    dataset = GPROF_NN_3D_Dataset(
        input_file, batch_size=1, normalize=False, targets=["surface_precip"]
    )
    x, _ = dataset[0]
    x = x.numpy()

    tbs = x[:, :5]
    tbs = tbs[np.isfinite(tbs)]
    assert np.all((tbs > 30) * (tbs < 400))

    eia = x[:, 5]
    eia = eia[np.isfinite(eia)]
    assert np.all((eia >= -60) * (eia <= 60))

    t2m = x[:, 6]
    t2m = t2m[np.isfinite(t2m)]
    assert np.all((t2m > 200) * (t2m < 350))

    tcwv = x[:, 7]
    tcwv = tcwv[np.isfinite(tcwv)]
    assert np.all((tcwv > 0) * (tcwv < 100))


def test_gprof_3d_dataset_mhs():
    """
    Test loading of 3D training data for MHS sensor.
    """
    input_file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc"
    dataset = GPROF_NN_3D_Dataset(
        input_file, batch_size=1, augment=False, transform_zeros=True
    )

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
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = SimulatorDataset(input_file, normalize=False, batch_size=1024)
    x, y = dataset[0]
    x = x.numpy()
    y = {k: y[k].numpy() for k in y}

    tbs = x[:, :15]
    tbs = tbs[np.isfinite(tbs)]
    assert np.all((tbs > 30) * (tbs < 400))
    t2m = x[:, 15]
    assert np.all((t2m > 180) * (t2m < 350))
    tcwv = x[:, 16]
    assert np.all((tcwv > 0) * (tcwv < 100))

    # Input Tbs must match simulated plus biases.
    for i in range(x.shape[0]):
        tbs_in = x[i, [0], :, :]
        tbs_sim = y[f"simulated_brightness_temperatures_0"][i, :, :, :]
        tbs_bias = y[f"brightness_temperature_biases_0"][i, :, :, :]
        tbs_sim[tbs_sim <= -900] = np.nan
        tbs_bias[tbs_bias <= -900] = np.nan
        tbs_out = tbs_sim - tbs_bias

        valid = np.isfinite(tbs_out) * np.isfinite(tbs_in)
        tbs_in = tbs_in[valid]
        tbs_out = tbs_out[valid]

        if tbs_in.size == 0:
            continue
        ind = np.argmax(np.abs(tbs_in - tbs_out))
        assert np.all(np.isclose(tbs_in, tbs_out, atol=1e-3))


def test_simulator_dataset_mhs():
    """
    Test loading of simulator training data.
    """
    input_file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc.gz"
    dataset = SimulatorDataset(
        input_file, batch_size=1024, augment=True, normalize=False
    )
    x, y = dataset[0]

    x = x.numpy()
    tbs = x[:, :15]
    tbs = tbs[np.isfinite(tbs)]
    assert np.all((tbs > 30) * (tbs < 400))
    t2m = x[:, 15]
    t2m = t2m[np.isfinite(t2m)]
    assert np.all((t2m > 180) * (t2m < 350))
    tcwv = x[:, 16]
    tcwv = tcwv[np.isfinite(tcwv)]
    assert np.all((tcwv > 0) * (tcwv < 100))

    assert np.all(np.isfinite(y["brightness_temperature_biases_0"].numpy()))
    assert np.all(np.isfinite(y["simulated_brightness_temperatures_0"].numpy()))
    assert "brightness_temperature_biases_0" in y
    assert len(y["brightness_temperature_biases_0"].shape) == 4
    assert "simulated_brightness_temperatures_0" in y
    assert len(y["simulated_brightness_temperatures_0"].shape) == 5


def test_gprof_3d_dataset_pretraining_mhs():
    """
    Test that the correct inputs are loaded when loading a pre-training dataset
    for MHS.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_3D_Dataset(
        input_file, batch_size=1, normalize=False, targets=["surface_precip"],
        sensor=sensors.MHS
    )

    x, _ = dataset[0]
    x = x.numpy()

    assert x.shape[1] == 5 + 3 + 18 + 4


def test_gprof_3d_dataset_pretraining_tmi():
    """
    Test that the correct inputs are loaded when loading a pre-training dataset
    for TMI.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = GPROF_NN_3D_Dataset(
        input_file, batch_size=1, normalize=False, targets=["surface_precip"],
        sensor=sensors.TMIPR
    )

    x, _ = dataset[0]
    x = x.numpy()

    assert x.shape[1] == 9 + 2 + 18 + 4
