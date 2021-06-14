from pathlib import Path

import numpy as np

from gprof_nn.data.profiles import ProfileClusters

def test_load_clusters():
    path = Path(__file__).parent / "data"
    profiles = ProfileClusters(path, True)
    rwc = profiles.get_profiles("rain_water_content", 280.0)

    profiles = ProfileClusters(path, False)
    cwc = profiles.get_profiles("cloud_water_content", 280.0)


