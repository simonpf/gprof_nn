"""
Tests for the Pytorch dataset classes used to load the training
data.
"""

from gprof_nn import sensors
from gprof_nn.definitions import ALL_TARGETS
from gprof_nn.data.training_data import (
    load_tbs_1d_gmi,
    load_tbs_1d_xtrack_sim,
    load_tbs_1d_conical_sim,
    load_tbs_1d_xtrack_other,
    load_tbs_1d_conical_other,
    load_training_data_3d_gmi,
    load_training_data_3d_xtrack_sim,
    load_training_data_3d_conical_sim,
    load_training_data_3d_other,
    load_targets_1d,
    load_targets_1d_xtrack,
    load_ancillary_data,
    GPROFNN1DDataset,
    GPROFNN3DDataset,
    SimulatorDataset,
    interpolate_angles
)

import pytest

import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr

from conftest import (
    training_files_1d_gmi_sim,
    training_files_1d_gmi_mrms,
    training_files_1d_atms_sim,
    training_files_1d_atms_mrms,
    training_files_1d_atms_era5,
    training_files_3d_atms_sim
)


###############################################################################
# GMI
###############################################################################


def test_load_tbs_1d_gmi_sim(training_files_1d_gmi_sim):
    """
    Ensure that loading of brightness temperature data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_gmi_sim[0])
    tbs = load_tbs_1d_gmi(dataset)
    assert isinstance(tbs, torch.Tensor)
    assert tbs.shape[-1] == 15
    assert (tbs[torch.isfinite(tbs)] > 0).all()


def test_load_tbs_1d_gmi_mrms(training_files_1d_gmi_mrms):
    """
    Ensure that loading of brightness temperature data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_gmi_mrms[0])
    tbs = load_tbs_1d_gmi(dataset)
    assert isinstance(tbs, torch.Tensor)
    assert tbs.shape[-1] == 15
    assert (tbs[torch.isfinite(tbs)] > 0).all()


def test_load_ancillary_data_gmi_1d(training_files_1d_gmi_sim):
    """
    Ensure that loading of ancillary data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_gmi_sim[0])
    ancillary_data = load_ancillary_data(dataset, "CLI", 1)
    assert isinstance(ancillary_data, torch.Tensor)
    assert ancillary_data.shape[-1] == 13


def test_load_targets_gmi_1d(training_files_1d_gmi_sim):
    """
    Ensure that loading of ancillary data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_gmi_sim[0])
    targets = load_targets_1d(dataset, ALL_TARGETS)
    assert isinstance(targets, dict)
    surface_precip = targets["surface_precip"]
    assert isinstance(surface_precip, torch.Tensor)


@pytest.mark.parametrize(
    "training_files",
    [
        "training_files_1d_gmi_sim",
        "training_files_1d_gmi_mrms",
    ])
def test_load_training_data_1d_gmi(training_files, request):

    training_files = request.getfixturevalue(training_files)

    path = training_files[0].parent
    training_data = GPROFNN1DDataset(path)
    training_loader = DataLoader(training_data, batch_size=32)

    for x, y in training_loader:
        assert "brightness_temperatures" in x
        assert "earth_incidence_angles" in x
        assert "ancillary_data" in x

        tbs = x["brightness_temperatures"]
        tbs = tbs[torch.isfinite(tbs)]
        assert tbs.min() > 0
        assert tbs.max() < 350

        angs = x["earth_incidence_angles"]
        angs = angs[torch.isfinite(angs)]
        assert angs.min() > -60
        assert angs.max() < 60

        anc = x["ancillary_data"]
        anc = anc[torch.isfinite(anc)]
        assert anc.min() > -999
        assert anc.max() < 350

        for target in ALL_TARGETS:
            assert target in y


@pytest.mark.parametrize(
    "training_files_1d",
    [
        "training_files_1d_gmi_sim",
        "training_files_1d_gmi_mrms",
    ])
def test_gprof_nn_1d_dataset_gmi_loading(training_files_1d, request):
    """
    Ensure that batches generated using a torch.utils.data.DataLoader contain
    data from difference files and that data from all files is loaded in one
    epoch.
    """
    training_files_1d = request.getfixturevalue(training_files_1d)
    training_files = training_files_1d[0].parent
    training_data = GPROFNN1DDataset(training_files)
    training_loader = DataLoader(
        training_data,
        num_workers=4,
        worker_init_fn=training_data.worker_init_fn,
        batch_size=128
    )

    surface_precip = []
    for x, y in training_loader:
        surface_precip.append(y["surface_precip"].numpy().ravel())

    all_precip = np.concatenate(surface_precip)
    all_precip_ref = np.concatenate(
        [xr.open_dataset(file).surface_precip.data for file in training_files_1d]
    )

    assert np.isclose(
        np.sort(np.unique(all_precip)),
        np.sort(np.unique(all_precip_ref))
    ).all()


@pytest.mark.parametrize(
    "training_files_1d",
    [
        "training_files_1d_gmi_sim",
        "training_files_1d_gmi_mrms",
    ])
def test_gprof_nn_1d_dataset_gmi_validation(training_files_1d, request):
    """
    Ensure that batches from multiple instantiations of the GPROF-NN 1D
    dataset are different if not in validation mode. Ensure that they
    are identical if in validation mode.
    """
    training_files_1d = request.getfixturevalue(training_files_1d)
    training_files = training_files_1d[0].parent
    training_data_1 = GPROFNN1DDataset(training_files)
    training_data_2 = GPROFNN1DDataset(training_files)
    validation_data_1 = GPROFNN1DDataset(training_files, validation=True)
    validation_data_2 = GPROFNN1DDataset(training_files, validation=True)

    training_loader_1 = DataLoader(
        training_data_1,
        num_workers=4,
        worker_init_fn=training_data_1.worker_init_fn,
        batch_size=128
    )
    x1_train, y1_train = next(iter(training_loader_1))

    training_loader_2 = DataLoader(
        training_data_2,
        num_workers=4,
        worker_init_fn=training_data_2.worker_init_fn,
        batch_size=128
    )
    x2_train, y2_train = next(iter(training_loader_1))

    validation_loader_1 = DataLoader(
        validation_data_1,
        num_workers=4,
        worker_init_fn=validation_data_1.worker_init_fn,
        batch_size=128
    )
    validation_loader_2 = DataLoader(
        validation_data_2,
        num_workers=4,
        worker_init_fn=validation_data_2.worker_init_fn,
        batch_size=128
    )

    x1_val, y1_val = next(iter(validation_loader_1))
    x2_val, y2_val = next(iter(validation_loader_2))

    assert not (y1_train["surface_precip"] == y2_train["surface_precip"]).all()
    assert (y1_val["surface_precip"] == y2_val["surface_precip"]).all()


@pytest.mark.parametrize(
    "training_files_3d",
    [
        "training_files_3d_gmi_sim",
        "training_files_3d_gmi_mrms",
    ])
def test_load_training_data_3d_gmi(training_files_3d, request):
    """
    Test loading of 3D GMI training data for training data extracted from
    both .sim files and MRMS match-ups.
    """
    training_files_3d = request.getfixturevalue(training_files_3d)
    scene = xr.load_dataset(training_files_3d[0])
    x, y = load_training_data_3d_gmi(
        scene,
        ALL_TARGETS,
        augment=True,
        rng=np.random.default_rng()
    )
    assert "brightness_temperatures" in x
    assert x["brightness_temperatures"].shape == ((15, 128, 64))
    assert "earth_incidence_angles" in x
    assert x["earth_incidence_angles"].shape == ((15, 128, 64))
    assert "ancillary_data" in x
    assert x["ancillary_data"].shape == ((13, 128, 64))


@pytest.mark.parametrize(
    "training_files_3d",
    [
        "training_files_3d_gmi_sim",
        "training_files_3d_gmi_mrms",
    ])
def test_gprof_nn_3d_dataset_gmi(training_files_3d, request):
    """
    Ensure that the GPROFNN3DDataset correctly loads the GPROF-NN 3D training data.
    """
    training_files = request.getfixturevalue(training_files_3d)
    training_data = GPROFNN3DDataset(training_files[0].parent)

    x, y = training_data[0]
    assert "brightness_temperatures" in x
    tbs = x["brightness_temperatures"]
    assert (tbs[torch.isfinite(tbs)] > 0).all()
    assert "earth_incidence_angles" in x
    assert "ancillary_data" in x

    assert "surface_precip" in y
    sp = y["surface_precip"]
    assert sp.ndim == 2
    assert (sp[torch.isfinite(sp)] >= 0.0).all()


###############################################################################
# ATMS
###############################################################################

def test_load_tbs_1d_atms_sim(training_files_1d_atms_sim):
    """
    Ensure that loading of brightness temperature data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_atms_sim[0])
    angles = (2.0 * np.random.rand(dataset.samples.size) - 1.0) * 60.0
    rng = np.random.default_rng(seed=42)
    tbs, _ = load_tbs_1d_xtrack_sim(
        dataset,
        angles,
        sensors.ATMS,
        targets=ALL_TARGETS,
        rng=rng
    )
    assert isinstance(tbs, torch.Tensor)
    assert tbs.dtype == torch.float32
    assert tbs.shape[-1] == 15
    assert (tbs[torch.isfinite(tbs)] > 0).all()


@pytest.mark.skip(reason="Preprocessor not available.")
def test_load_tbs_1d_atms_mrms(training_files_1d_atms_mrms):
    """
    Ensure that loading of brightness temperature data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_atms_mrms[0])
    tbs, angles = load_tbs_1d_xtrack_other(dataset, sensors.ATMS)
    assert isinstance(tbs, torch.Tensor)
    assert tbs.dtype == torch.float32
    assert tbs.shape[-1] == 15
    assert angles.shape[-1] == 15
    assert (tbs[torch.isfinite(tbs)] > 0).all()


def test_load_tbs_1d_atms_era5(training_files_1d_atms_era5):
    """
    Ensure that loading of brightness temperature data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_atms_era5[0])
    tbs, angles = load_tbs_1d_xtrack_other(dataset, sensors.ATMS)
    assert tbs.dtype == torch.float32
    assert isinstance(tbs, torch.Tensor)
    assert tbs.shape[-1] == 15
    assert angles.shape[-1] == 15
    assert (tbs[torch.isfinite(tbs)] > 0).all()


def test_load_targets_1d_xtrack(training_files_1d_atms_sim):
    """
    Ensure that loading of brightness temperature data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_atms_sim[0])
    angles = (2.0 * np.random.rand(dataset.samples.size) - 1.0) * 60.0
    targs = load_targets_1d_xtrack(dataset, angles, ALL_TARGETS)
    assert isinstance(targs, dict)
    assert isinstance(targs["surface_precip"], torch.Tensor)


def test_load_ancillary_data_atms(training_files_1d_atms_sim):
    """
    Ensure that loading of ancillary data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_atms_sim[0])
    ancillary_data = load_ancillary_data(dataset, "CLI", 1)
    assert isinstance(ancillary_data, torch.Tensor)
    assert ancillary_data.shape[-1] == 13


@pytest.mark.parametrize(
    "training_files",
    [
        "training_files_1d_atms_sim",
        #"training_files_1d_atms_mrms",
        "training_files_1d_atms_era5"
    ])


def test_gprof_nn_1d_dataset_atms(training_files, request):

    training_files = request.getfixturevalue(training_files)

    path = training_files[0].parent
    training_data = GPROFNN1DDataset(path)
    training_loader = DataLoader(training_data, batch_size=32)

    for x, y in training_loader:
        assert "brightness_temperatures" in x
        assert "earth_incidence_angles" in x
        assert "ancillary_data" in x

        tbs = x["brightness_temperatures"]
        mask = torch.isfinite(tbs)
        assert tbs[mask].min() > 0
        assert tbs[mask].max() < 400

        angs = x["earth_incidence_angles"]
        assert angs.shape == tbs.shape
        assert angs[mask].min() > -65
        assert angs[mask].max() < 65

        anc = x["ancillary_data"]
        assert anc[mask.any(-1)].min() > -999
        assert anc[mask.any(-1)].max() < 350

        for target in ALL_TARGETS:
            assert target in y

        assert y["surface_precip"].shape[1] == 2048


@pytest.mark.skip(reason="Satformer not yet available.")
def test_load_training_data_3d_xtrack_sim(training_files_3d_atms_sim):
    """
    Test loading of 3D ATMS training data for training data extracted from
    both .sim files and MRMS match-ups.
    """
    scene = xr.load_dataset(training_files_3d_atms_sim[0])
    x, y = load_training_data_3d_xtrack_sim(
        sensors.ATMS,
        scene,
        ALL_TARGETS,
        augment=True,
        rng=np.random.default_rng()
    )
    assert "brightness_temperatures" in x
    assert x["brightness_temperatures"].shape == ((15, 128, 64))
    assert "earth_incidence_angles" in x
    assert x["earth_incidence_angles"].shape == ((15, 128, 64))
    assert "ancillary_data" in x
    assert x["ancillary_data"].shape == ((13, 128, 64))


@pytest.mark.parametrize(
    "training_files",
    [
        #"training_files_3d_atms_mrms",
        "training_files_3d_atms_era5"
    ])
def test_load_training_data_3d_xtrack_other(training_files, request):
    """
    Test loading of 3D ATMS training data for training data extracted from
    both MRMS or ERA5 files.
    """
    training_files = request.getfixturevalue(training_files)

    scene = xr.load_dataset(training_files[0])
    x, y = load_training_data_3d_other(
        sensors.ATMS,
        scene,
        ALL_TARGETS,
        augment=True,
        rng=np.random.default_rng()
    )
    assert "brightness_temperatures" in x
    assert x["brightness_temperatures"].shape == ((15, 128, 64))
    assert "earth_incidence_angles" in x
    assert x["earth_incidence_angles"].shape == ((15, 128, 64))
    assert "ancillary_data" in x
    assert x["ancillary_data"].shape == ((13, 128, 64))


@pytest.mark.parametrize(
    "training_files_3d",
    [
        #"training_files_3d_atms_sim",
        #"training_files_3d_atms_mrms",
        "training_files_3d_atms_era5",
    ])
def test_gprof_nn_3d_dataset_atms(training_files_3d, request):
    """
    Ensure that the GPROFNN3DDataset correctly loads the GPROF-NN 3D training data.
    """
    training_files = request.getfixturevalue(training_files_3d)
    training_data = GPROFNN3DDataset(training_files[0].parent)

    x, y = training_data[0]
    assert "brightness_temperatures" in x
    tbs = x["brightness_temperatures"]
    assert tbs.shape == (15, 128, 64)
    assert tbs.ndim == 3
    assert (tbs[torch.isfinite(tbs)] > 0).all()
    assert "earth_incidence_angles" in x
    assert x["earth_incidence_angles"].ndim == 3
    assert x["brightness_temperatures"].shape == (15, 128, 64)
    assert "ancillary_data" in x
    assert x["ancillary_data"].ndim == 3
    assert x["ancillary_data"].shape == (13, 128, 64)

    assert "surface_precip" in y
    sp = y["surface_precip"]
    assert sp.ndim == 2
    assert (sp[torch.isfinite(sp)] >= 0.0).all()


def test_simulator_dataset_atms(training_files_3d_atms_sim):
    """
    Ensure that the Simulator dataset loads the correct data.
    """
    training_data = SimulatorDataset(training_files_3d_atms_sim[0].parent)

    x, y = training_data[0]
    assert "brightness_temperatures" in x
    tbs = x["brightness_temperatures"]
    assert tbs.ndim == 3
    assert (tbs[torch.isfinite(tbs)] > 0).all()
    assert "earth_incidence_angles" in x
    assert x["earth_incidence_angles"].ndim == 3
    assert "ancillary_data" in x
    assert x["ancillary_data"].ndim == 3

    assert "simulated_brightness_temperatures" in y
    tbs = y["simulated_brightness_temperatures"]
    assert tbs.ndim == 4
    assert (tbs[np.isfinite(tbs)] >= 0.0).all()

    assert "brightness_temperature_biases" in y
    biases = y["brightness_temperature_biases"]
    assert biases.ndim == 3
    assert (biases[np.isfinite(biases)] >= -100).all()


###############################################################################
# AMSR2
###############################################################################


def test_load_tbs_1d_amsr2_sim(training_files_1d_amsr2_sim):
    """
    Ensure that loading of brightness temperature data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_amsr2_sim[0])
    tbs = load_tbs_1d_conical_sim(dataset, sensors.AMSR2)
    assert isinstance(tbs, torch.Tensor)
    assert tbs.shape[-1] == 15
    assert (tbs[torch.isfinite(tbs)] > 0).all()


def test_load_tbs_1d_amsr2_mrms(training_files_1d_amsr2_mrms):
    """
    Ensure that loading of brightness temperature data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_amsr2_mrms[0])
    tbs, angles = load_tbs_1d_conical_other(dataset, sensors.AMSR2)
    assert isinstance(tbs, torch.Tensor)
    assert tbs.shape[-1] == 15
    assert angles.shape[-1] == 15
    assert (tbs[torch.isfinite(tbs)] > 0).all()


def test_load_tbs_1d_amsr2_era5(training_files_1d_amsr2_era5):
    """
    Ensure that loading of brightness temperature data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_amsr2_era5[0])
    tbs, angles = load_tbs_1d_conical_other(dataset, sensors.AMSR2)
    assert isinstance(tbs, torch.Tensor)
    assert tbs.shape[-1] == 15
    assert angles.shape[-1] == 15
    assert (tbs[torch.isfinite(tbs)] > 0).all()


def test_load_targets_amsr2_sim(training_files_1d_amsr2_sim):
    """
    Ensure that loading of brightness temperature data from 1D training data works.
    """
    dataset = xr.load_dataset(training_files_1d_amsr2_sim[0])
    targets = load_targets_1d(dataset, ALL_TARGETS)
    assert isinstance(targets, dict)
    assert "surface_precip" in targets
    assert np.isfinite(targets["surface_precip"].numpy()).all()


@pytest.mark.parametrize(
    "training_files_1d",
    [
        "training_files_1d_amsr2_sim",
        "training_files_1d_amsr2_mrms",
        "training_files_1d_amsr2_era5",
    ])
def test_gprof_nn_1d_dataset_amsr2(training_files_1d, request):
    """
    Ensure that the GPROFNN3DDataset correctly loads the GPROF-NN 3D training data.
    """
    training_files = request.getfixturevalue(training_files_1d)
    training_data = GPROFNN1DDataset(training_files[0].parent)

    x, y = next(iter(training_data))
    assert "brightness_temperatures" in x
    tbs = x["brightness_temperatures"]
    assert tbs.shape[1:] == (15,)
    assert (tbs[torch.isfinite(tbs)] > 0).all()
    assert "earth_incidence_angles" in x
    angs = x["earth_incidence_angles"]
    gprof_chans = list(sensors.AMSR2.gprof_channels.keys())
    for ind in np.arange(angs.shape[1]):
        if ind in gprof_chans:
            assert torch.isfinite(angs[..., ind]).all()
        else:
            assert torch.isnan(angs[..., ind]).all()

    assert "ancillary_data" in x
    assert x["ancillary_data"].shape[1:] == (13,)

    assert "surface_precip" in y
    sp = y["surface_precip"]
    assert sp.shape[1:] == (1,)
    assert (sp[torch.isfinite(sp)] >= 0.0).all()


def test_load_training_data_3d_conical_sim(training_files_3d_amsr2_sim):
    """
    Test loading of 3D AMSR2 training data for training data extracted from
    both .sim files and MRMS match-ups.
    """
    scene = xr.load_dataset(training_files_3d_amsr2_sim[0])
    x, y = load_training_data_3d_conical_sim(
        sensors.AMSR2,
        scene,
        ALL_TARGETS,
        augment=True,
        rng=np.random.default_rng()
    )
    assert "brightness_temperatures" in x
    assert x["brightness_temperatures"].shape == ((15, 128, 64))

    assert "earth_incidence_angles" in x
    angs = x["earth_incidence_angles"]
    assert angs.shape == ((15, 128, 64))
    gprof_chans = list(sensors.AMSR2.gprof_channels.keys())
    for ind in np.arange(angs.shape[0]):
        if ind in gprof_chans:
            assert torch.isfinite(angs[ind]).all()
        else:
            assert torch.isnan(angs[ind]).all()

    assert x["ancillary_data"].shape == ((13, 128, 64))


@pytest.mark.parametrize(
    "training_files",
    [
        "training_files_3d_amsr2_mrms",
        "training_files_3d_amsr2_era5"
    ])
def test_load_training_data_3d_conical_other(training_files, request):
    """
    Test loading of 3D AMSR2 training data for training data extracted from
    both MRMS or ERA5 files.
    """
    training_files = request.getfixturevalue(training_files)

    scene = xr.load_dataset(training_files[0])
    x, y = load_training_data_3d_other(
        sensors.AMSR2,
        scene,
        ALL_TARGETS,
        augment=True,
        rng=np.random.default_rng()
    )
    assert "brightness_temperatures" in x
    assert x["brightness_temperatures"].shape == ((15, 128, 64))
    assert "earth_incidence_angles" in x

    angs = x["earth_incidence_angles"]
    assert angs.shape == ((15, 128, 64))
    gprof_chans = list(sensors.AMSR2.gprof_channels.keys())
    for ind in np.arange(angs.shape[0]):
        if ind in gprof_chans:
            assert torch.isfinite(angs[ind]).all()
        else:
            assert torch.isnan(angs[ind]).all()

    assert "ancillary_data" in x
    assert x["ancillary_data"].shape == ((13, 128, 64))


@pytest.mark.parametrize(
    "training_files_3d",
    [
        "training_files_3d_amsr2_sim",
        "training_files_3d_amsr2_mrms",
        "training_files_3d_amsr2_era5",
    ])
def test_gprof_nn_3d_dataset_amsr2(training_files_3d, request):
    """
    Ensure that the GPROFNN3DDataset correctly loads the GPROF-NN 3D training data.
    """
    training_files = request.getfixturevalue(training_files_3d)
    training_data = GPROFNN3DDataset(training_files[0].parent)

    x, y = training_data[0]
    assert "brightness_temperatures" in x
    tbs = x["brightness_temperatures"]
    assert tbs.ndim == 3
    assert (tbs[torch.isfinite(tbs)] > 0).all()

    assert "earth_incidence_angles" in x
    angs = x["earth_incidence_angles"]
    assert angs.shape == ((15, 128, 64))
    gprof_chans = list(sensors.AMSR2.gprof_channels.keys())
    for ind in np.arange(angs.shape[0]):
        if ind in gprof_chans:
            assert torch.isfinite(angs[ind]).all()
        else:
            assert torch.isnan(angs[ind]).all()

    assert "ancillary_data" in x
    assert x["ancillary_data"].ndim == 3

    assert "surface_precip" in y
    sp = y["surface_precip"]
    assert sp.ndim == 2
    assert (sp[torch.isfinite(sp)] >= 0.0).all()


def test_simulator_dataset_amsr2(training_files_3d_amsr2_sim):
    """
    Ensure that the Simulator dataset loads the correct data.
    """
    training_data = SimulatorDataset(training_files_3d_amsr2_sim[0].parent)

    x, y = training_data[0]
    assert "brightness_temperatures" in x
    tbs = x["brightness_temperatures"]
    assert tbs.ndim == 3
    assert (tbs[torch.isfinite(tbs)] > 0).all()
    assert "earth_incidence_angles" in x
    assert x["earth_incidence_angles"].ndim == 3
    assert "ancillary_data" in x
    assert x["ancillary_data"].ndim == 3

    assert "simulated_brightness_temperatures" in y
    tbs = y["simulated_brightness_temperatures"]
    assert tbs.ndim == 3
    assert (tbs[np.isfinite(tbs)] >= 0.0).all()

    assert "brightness_temperature_biases" in y
    biases = y["brightness_temperature_biases"]
    assert biases.ndim == 3
    assert (biases[np.isfinite(biases)] >= -100).all()


def test_angle_interpolation():
    """
    Test interpolation along angles.
    """
    angles = np.linspace(0, 100, 11)
    tbs = np.tile(np.arange(11)[None, :, None], (100, 1, 5))
    angs = np.random.uniform(0, 100, size=(100))

    tbs_i = interpolate_angles(tbs, angles, angs)

    assert tbs_i.shape == (100, 5)
    assert np.isclose(tbs_i[..., 0], angs / 10).all()
