"""
====================
gprof_nn.definitions
====================

This module provides basic definitions that are used throughout the packages.
"""
import numpy as np

MISSING = -9999.9

ALL_TARGETS = [
    "surface_precip",
    "convective_precip",
    "cloud_water_content",
    "rain_water_content",
    "snow_water_content",
    "latent_heat",
    "ice_water_path",
    "rain_water_path",
    "cloud_water_path"
]

PROFILE_NAMES = [
    "rain_water_content",
    "cloud_water_content",
    "snow_water_content",
    "latent_heat",
]

# Minimum and maximum two-meter temperature. Defined in GPM_arraydef.f.
T2M_MIN = 220
T2M_MAX = 320
# Minimum and maximum total column water vapor. Defined in GPM_arraydef.f.
TCWV_MIN = 0
TCWV_MAX = 78

# Profile variables.
N_LAYERS = 28
LEVELS = np.concatenate([
    np.linspace(500.0, 1e4, 20),
    np.linspace(11e3, 18e3, 8)
])
