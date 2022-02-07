from pathlib import Path

import numpy as np

from gprof_nn.data import get_profile_clusters
from gprof_nn.data.profiles import ProfileClusters


def test_load_clusters():
    """
    Ensures that profiles are loaded correctly.
    """
    path = get_profile_clusters()
    profiles = ProfileClusters(path, True)
    rwc = profiles.get_profiles("rain_water_content", 280.0)
    assert np.all(np.isclose(rwc[:, -1], 0.0))

    profiles = ProfileClusters(path, False)
    rwc = profiles.get_profiles("cloud_water_content", 280)
    assert rwc.shape == (40, 28)


def test_get_profile_data():
    """
    Ensure that profile data has expected shape
    """
    path = get_profile_clusters()
    profiles = ProfileClusters(path, True)
    data = profiles.get_profile_data("rain_water_content")
    assert data.shape == (12, 28, 40)

    profiles = ProfileClusters(path, False)
    data = profiles.get_profile_data("rain_water_content")
    assert data.shape == (12, 28, 40)

def test_get_scales_and_indices():
    """
    Ensures that cluster centers are matched to their respective
    indices.
    """
    # Raining
    path = get_profile_clusters()
    profiles = ProfileClusters(path, True)
    cwc = profiles.get_profile_data("cloud_water_content")

    scales, indices = profiles.get_scales_and_indices(
        "cloud_water_content",
        269.0,
        cwc[0].transpose()
    )
    assert np.all(np.isclose(indices,
                             np.arange(40)))
    assert np.all(np.isclose(scales, 1.0, rtol=1e-3))


    # Non-raining
    profiles = ProfileClusters(path, False)
    cwc = profiles.get_profile_data("cloud_water_content")

    scales, indices = profiles.get_scales_and_indices(
        "cloud_water_content",
        269.0,
        cwc[0].transpose()
    )
    assert np.all(np.isclose(indices,
                             np.arange(40)))
    assert np.all(np.isclose(scales, 1.0, atol=0.1))
