from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn.noise_estimation import ObservationDataset0D

def test_observation_dataset_0d():
    """
    Test loading of observation dataset.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "observations_mhs.nc"
    input_data = xr.load_dataset(input_file)
    dataset = ObservationDataset0D(
        input_file,
        batch_size=1
    )

    x, y = dataset[0]
    x = x.detach().numpy()
    y = y.detach().numpy()

    assert np.all(x > -1.5)
    assert np.all(x < 1.5)
    assert np.all(np.sum(x[:, 1:], -1) == 1.0)

    assert np.all(y > -1.5)
    assert np.all(y < 1.5)
