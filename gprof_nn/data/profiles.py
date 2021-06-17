"""
======================
gprof_nn.data.profiles
======================

Functionality to read GPROF profile cluster files.

The profile clusters are used to compress the retrieved hydrometeor classes.
Instead of a full profile for each pixel, the retrieval just stores a scaling
factor together with a profile class that determines the profile shape.

Raining and non-raining profiles are treated separately. For non-raining
 profiles all species are set to 0 except cloud water content.
"""
from pathlib import Path

import numpy as np
import xarray as xr

N_SPECIES = 5
N_TEMPS = 12
N_LAYERS = 28
N_CLUSTERS = 40

FILENAMES = {
    True: "GPM_profile_clustersV7.dat",
    False: "GPM_profile_clustersNRV7.dat",
}


class ProfileClusters:
    """
    Interface to read profile cluster databases.
    """
    def __init__(self,
                 ancillary_data_path,
                 raining):
        """
        Args:
            ancillary_data_path: Path pointing towards the folder containing
                the profile cluster files.
            raining: Flag indicating whether to load raining or non-raining
                profiles.
        """
        self.path = Path(ancillary_data_path)
        self.raining = raining

        filename = self.path / FILENAMES[self.raining]
        shape = (N_SPECIES, N_TEMPS, N_CLUSTERS, N_LAYERS)
        if not self.raining:
            shape = (1, N_TEMPS, N_CLUSTERS, N_LAYERS)
        self.data = np.fromfile(filename, dtype=np.float32).reshape(shape)

    def get_profiles(self,
                     species,
                     t2m):
        """
        Return array of profiles for given species and two-meter temperature.

        Args:
             species: The name of the species.
             t2m: The two meter temperature.

        Return:
           2D array containing the 40 profiles for the given species and
           two-meter temperature.
        """
        if self.raining:
            if species == "rain_water_content":
                species_index = 0
            elif species == "cloud_water_content":
                species_index = 1
            elif species == "snow_water_content":
                species_index = 2
            elif species == "latent_heat":
                species_index = 4
            else:
                raise ValueError(
                    f"{species} is not a valid species name. For raining "
                    "profiles it should be one of ['rain_water_content', "
                    "'cloud_water_content', 'snow_water_content', "
                    "'latent_heat']."
                )
        else:
            if species == "cloud_water_content":
                species_index = 0
            elif species == "latent_heat":
                species_index = 1
            else:
                raise ValueError(
                    f"{species} is not a valid species name. For non-raining "
                    "profiles it should be one of ['cloud_water_content', "
                    "'latent_heat']."
                )
        t2m_indices = self.get_t2m_indices(t2m)
        return self.data[species_index, t2m_indices]

    def get_profile_data(self, species):
        """
        Return all profiles for given species.

        Args:
            species: The name of the species.

        Return:
            Numpy array of shape ``(28, 40)`` containing the 40 profiles
            for the requested species.
        """
        if self.raining:
            if species == "rain_water_content":
                species_index = 0
            elif species == "cloud_water_content":
                species_index = 1
            elif species == "snow_water_content":
                species_index = 2
            elif species == "latent_heat":
                species_index = 4
            else:
                return -9999.9 * np.ones((N_TEMPS, N_LAYERS, N_CLUSTERS))
        else:
            if species == "cloud_water_content":
                species_index = 0
            elif species == "latent_heat":
                species_index = 1
            else:
                return -9999.9 * np.ones((N_TEMPS, N_LAYERS, N_CLUSTERS))
        return np.transpose(self.data[species_index], axes=(0, 2, 1))


    def get_t2m_indices(self, t2m):
        if isinstance(t2m, np.ndarray):
            t2m_indices = np.clip(((t2m - 268.0) / 3).astype(np.int), 0, 11)
        else:
            t2m_indices = np.clip(int((t2m - 268.0) / 3), 0, 11)
        return t2m_indices

    def get_scales_and_indices(self,
                               species,
                               t2m,
                               profiles):
        """
        Calculate scaling factors and profiles indices for given species.


        """
        output_shape = profiles.shape[:-1]
        if not self.raining and species != "cloud_water_content":
            return (np.zeros(output_shape, dtype=np.float32),
                    np.zeros(output_shape, dtype=np.int32))

        centers = self.get_profiles(species, t2m)
        scales = profiles.sum(axis=-1)
        profiles = profiles / scales[..., np.newaxis]

        shape = [1] * (len(profiles.shape) - 2) + [40, 28]

        mse = np.mean(
            (profiles[..., np.newaxis, :] - centers.reshape(shape)) ** 2,
            axis=-1
        )
        indices = np.argmin(mse, axis=-1)
        return scales, indices



