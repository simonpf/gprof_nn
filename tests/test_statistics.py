"""
Tests for the gprof_nn.statistics module.
"""
import numpy as np
import xarray as xr

from gprof_nn.statistics import TrainingDataStats


def test_training_data_statistics(tmp_path):
    """
    Ensure that scene statistics are correctly tracked.
    """
    training_data_stats = TrainingDataStats(tmp_path)

    lons = np.arange(-180, 181, 5)
    lats = np.arange(-90, 91, 5)
    lons = 0.5 * (lons[1:] + lons[:-1])
    lats = 0.5 * (lats[1:] + lats[:-1])
    lons, lats = np.meshgrid(lons, lats)
    surface_precip = np.ones_like(lons)

    scene_1 = xr.Dataset({
        "longitude": (("scans", "pixels"), lons),
        "latitude": (("scans", "pixels"), lats),
        "surface_precip": (("scans", "pixels"), surface_precip)
    })

    training_data_stats.track(scene_1)
    stats = xr.load_dataset(training_data_stats.stats_file_path)
    assert np.all(np.isclose(stats.counts.data, 1.0))
    assert np.all(np.isclose(stats.scene_counts.data, 1.0))

    training_data_stats.track(scene_1)
    stats = xr.load_dataset(training_data_stats.stats_file_path)
    assert np.all(np.isclose(stats.counts.data, 2.0))
    assert np.all(np.isclose(stats.scene_counts.data, 2.0))

    scene_1.surface_precip.data[:] = np.nan
    training_data_stats.track(scene_1)
    stats = xr.load_dataset(training_data_stats.stats_file_path)
    assert np.all(np.isclose(stats.counts.data, 2.0))
    assert np.all(np.isclose(stats.scene_counts.data, 2.0))
