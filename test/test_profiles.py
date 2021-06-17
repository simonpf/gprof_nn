from pathlib import Path

import numpy as np

from gprof_nn.data.profiles import ProfileClusters


def test_load_clusters():
    """
    Ensures that profiles are loaded correctly.
    """
    path = Path(__file__).parent / "data"
    profiles = ProfileClusters(path, True)
    rwc = profiles.get_profiles("rain_water_content", 280.0)

    profiles = ProfileClusters(path, False)
    cwc = profiles.get_profiles("cloud_water_content", 280.0)

    profiles = profiles.get_profile_data("cloud_water_content")
    assert profiles.shape == (12, 28, 40)


def test_get_scales_and_indices():
    """
    Ensures that profiles are loaded correctly.
    """
    path = Path(__file__).parent / "data"
    profiles = ProfileClusters(path, True)
    cwc = profiles.get_profile_data("cloud_water_content")

    scales, indices = profiles.get_scales_and_indices(
        "cloud_water_content",
        269.0,
        cwc[0].transpose()
    )
    assert np.all(np.isclose(indices,
                             np.arange(40)))
    assert np.all(np.isclose(scales, 1.0))

