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
        t2m_index = np.clip(int((t2m - 268.0) / 3), 0, 11)
        return self.data[species_index, t2m_index]
