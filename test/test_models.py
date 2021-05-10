import numpy as np
import torch
from gprof_nn.models import GPROFNN0D, HyperResNetFC


def test_hyperres_fc():
    # Make sure 0-layer configuration does nothing.
    network = HyperResNetFC(38, 128, 128, 0)
    x = torch.ones(1, 38)
    y, acc = network(x, None)
    assert np.all(np.isclose(y.detach().numpy(), x.detach().numpy()))

    # Make sure 1-layer interal configuration returns accumulator
    # containing input.
    network = HyperResNetFC(38, 128, 128, 1, internal=True)
    x = torch.ones(1, 38)
    y, acc = network(x, None)
    assert np.all(np.isclose(acc.detach().numpy()[:, :38], x.detach().numpy()))

    # Make sure 2-layer interal configuration returns accumulator
    # containing input.
    network = HyperResNetFC(38, 128, 128, 2, internal=False)
    x = torch.ones(1, 38)
    y, acc = network(x, None)
    print(y)
    assert np.all(np.isclose(acc.detach().numpy()[:, :38], x.detach().numpy()))


def test_gprof_nn_0d():
    targets = "surface_precip"
    network = GPROFNN0D(0, 1, 128, 128, target=targets)

    x = torch.ones(1, 38)
    y = network(x)

    assert type(y) == torch.Tensor

    targets = ["surface_precip", "rain_water_content"]
    network = GPROFNN0D(0, 1, 128, 128, target=targets)

    x = torch.ones(1, 38)
    y = network(x)

    assert type(y) == dict
    assert type(y["surface_precip"]) == torch.Tensor
    assert type(y["rain_water_content"]) == torch.Tensor
    assert y["rain_water_content"].shape == (1, 128, 28)
