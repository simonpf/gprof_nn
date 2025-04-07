"""
gprof_nn.data.statistics
========================

Module provide functionality to track training data statistics and calculate
according loss weights.
"""
from pathlib import Path
from typing import Tuple

from filelock import FileLock
import numpy as np
import xarray as xr



class TrainingDataStats:
    """
    The training data stats class keeps track of the training data extracted
    and stored in a specific training data folder. It tracks the global
    geospatial distributions as well as the distribution within the training-scene
    window.
    """
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lon_bins = np.arange(-180, 181, 5)
        self.lat_bins = np.arange(-90, 91, 5)
        self.bins = (self.lat_bins, self.lon_bins)

    @property
    def stats_file_path(self) -> Path:
        """
        Path object pointing to the stats file managed by this object.
        """
        return self.path / ".stats.nc"

    @property
    def lock(self) -> Path:
        """
        Return lock object for the stats file managed by the object.
        """
        lock_file = self.stats_file_path.with_suffix(".lock")
        return FileLock(lock_file)

    def init_stats_file(self, scene_size: Tuple[int, int] = (96, 96)) -> None:
        """
        Initializes stats file in training data folder.

        Args:
            path: The path pointing to the folder in wich to store the stats file.
            scene_size: The expected size of the training scenes.
        """
        stats_file = self.stats_file_path
        if not stats_file.exists():
            lons = 0.5 * (self.lon_bins[1:] + self.lon_bins[:-1])
            lats = 0.5 * (self.lat_bins[1:] + self.lat_bins[:-1])
            dims = (("latitude", "longitude"))
            dataset = xr.Dataset({
                "longitude": lons,
                "latitude": lats,
                "counts": (dims, np.zeros((lats.size, lons.size), dtype=np.float32)),
                "scene_counts": (("scans", "pixels"), np.zeros(scene_size, dtype=np.float32))
            })
            with self.lock:
                dataset.to_netcdf(stats_file)

    def track(
            self,
            training_scene: xr.Dataset,
            valid_var: str = "surface_precip"
    ) -> None:
        """
        Track stats for extract training scene.

            Args:
                training_scene:
                valid_var: The variable to use to determine sample validity.
        """
        lons = training_scene.longitude.data
        lats = training_scene.latitude.data
        scene_size = (
            training_scene.scans.size,
            training_scene.pixels.size,
        )
        valid = np.isfinite(training_scene[valid_var].data)
        lons = lons[valid]
        lats = lats[valid]

        self.init_stats_file(scene_size)
        counts = np.histogram2d(lats, lons, bins=self.bins)[0]
        scene_counts = valid.astype(np.float32)
        with self.lock:
            stats = xr.load_dataset(self.stats_file_path)
            stats["counts"].data += counts
            stats["scene_counts"].data += scene_counts
            stats.to_netcdf(self.stats_file_path)
