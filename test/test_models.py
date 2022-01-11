"""
This file tests the neural network models defined in the
'gprof_nn.models' module.
"""
from pathlib import Path

import numpy as np
import torch
from quantnn.transformations import LogLinear

from gprof_nn import sensors
from gprof_nn.data import get_test_data_path
from gprof_nn.definitions import ALL_TARGETS
from gprof_nn.data.training_data import (SimulatorDataset,
                                         GPROF_NN_3D_Dataset)
from gprof_nn.models import (
    MLP,
    ResidualMLP,
    HyperResidualMLP,
    MultiHeadMLP,
    GPROF_NN_1D_QRNN,
    GPROF_NN_1D_DRNN,
    GPROF_NN_3D_QRNN,
    SimulatorNet
)


DATA_PATH = get_test_data_path()


def test_mlp():
    """
    Tests for MLP module.
    """
    # Make sure 0-layer configuration does nothing.
    network = MLP(39, 128, 128, 0)
    x = torch.ones(1, 39)
    y, acc = network(x, None)
    assert np.all(np.isclose(y.detach().numpy(), x.detach().numpy()))


def test_residual_mlp():
    """
    Tests for MLP module with residual connections.
    """
    # Make sure 0-layer configuration does nothing.
    network = ResidualMLP(39, 128, 128, 0)
    x = torch.ones(1, 39)
    y, acc = network(x, None)
    assert np.all(np.isclose(y.detach().numpy(), x.detach().numpy()))

    # Make sure residuals are forwarded through network in internal
    # configuration.
    network = ResidualMLP(39, 39, 39, 3, internal=True)
    x = torch.ones(1, 39)
    for p in network.parameters():
        p[:] = 0.0
    y, acc = network(x, None)
    assert np.all(np.isclose(y.detach().numpy(), x.detach().numpy()))


def test_hyper_residual_mlp():
    """
    Tests for MLP module with hyper-residual connections.
    """
    network = HyperResidualMLP(39, 128, 128, 0)
    x = torch.ones(1, 39)
    y, acc = network(x, None)
    assert np.all(np.isclose(y.detach().numpy(), x.detach().numpy()))

    # Make sure residuals and hyperresiduals are forwarded through network
    # in internal configuration.
    network = HyperResidualMLP(39, 39, 39, 3, internal=True)
    x = torch.ones(1, 39)
    for p in network.parameters():
        p[:] = 0.0
    y, acc = network(x, None)
    assert np.all(np.isclose(y.detach().numpy(), 3.0 * x.detach().numpy()))
    assert np.all(np.isclose(acc.detach().numpy(), 4.0 * x.detach().numpy()))


def test_gprof_nn_0d():
    """
    Tests for GPROFNN1D classes module with hyper-residual connections.
    """
    network = GPROF_NN_1D_QRNN(sensors.GMI,
                               3, 128, 2, 64,
                               activation="GELU",
                               transformation=LogLinear)
    x = torch.ones(1, 39)
    y = network.predict(x)
    assert all([t in y for t in ALL_TARGETS])
    network = GPROF_NN_1D_QRNN(sensors.GMI,
                               3, 128, 2, 64,
                               activation="GELU",
                               residuals="hyper",
                               transformation=LogLinear)
    x = torch.ones(1, 39)
    y = network.predict(x)
    assert all([t in y for t in ALL_TARGETS])

    network = GPROF_NN_1D_DRNN(sensors.GMI,
                               3, 128, 2, 64,
                               residuals="hyper",
                               activation="GELU")
    x = torch.ones(1, 39)
    y = network.predict(x)
    assert all([t in y for t in ALL_TARGETS])


def test_gprof_nn_2d_gmi():
    """
    Ensure that GPROF_NN_3D model is consistent with training data
    for GMI.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc"
    dataset = GPROF_NN_3D_Dataset(input_file)
    network = GPROF_NN_3D_QRNN(sensors.GMI, 2, 128, 2, 64)
    x, y = dataset[0]
    y_pred = network.predict(x)
    assert all([t in y_pred for t in y])


def test_gprof_nn_2d_mhs():
    """
    Ensure that GPROF_NN_3D model is consistent with training data
    for MHS.
    """
    path = Path(__file__).parent
    input_file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc"
    dataset = GPROF_NN_3D_Dataset(input_file)
    network = GPROF_NN_3D_QRNN(sensors.MHS, 2, 128, 2, 64)
    x, y = dataset[0]
    y_pred = network.predict(x)
    assert all([t in y_pred for t in y])


def test_simulator():
    """
    Test simulator network.
    """
    path = Path(__file__).parent
    file = DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc"
    data = SimulatorDataset(file, batch_size=1)

    simulator = SimulatorNet(sensors.MHS, 64, 2, 32)
    x, y = data[0]
    print(x.shape)
    y_pred = simulator(x)
    for k in y_pred:
        assert k in y
        assert y_pred[k].shape[1] == y[k].shape[1]
        assert y_pred[k].shape[2] == y[k].shape[2]
