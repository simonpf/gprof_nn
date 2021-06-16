import numpy as np
import torch
from quantnn.transformations import LogLinear
from gprof_nn import ALL_TARGETS
from gprof_nn.models import (
    MLP,
    ResidualMLP,
    HyperResidualMLP,
    MultiHeadMLP,
    GPROF_NN_0D_QRNN,
    GPROF_NN_0D_DRNN
)


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
    Tests for GPROFNN0D classes module with hyper-residual connections.
    """
    network = GPROF_NN_0D_QRNN(0, 128, 1, 64, activation="GELU", transformation=LogLinear)
    x = torch.ones(1, 39)
    y = network.predict(x)
    assert all([t in y for t in ALL_TARGETS])

    network = GPROF_NN_0D_DRNN(0, 128, 1, 64, activation="GELU")
    x = torch.ones(1, 39)
    y = network.predict(x)
    assert all([t in y for t in ALL_TARGETS])
