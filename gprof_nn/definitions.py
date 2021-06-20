"""
====================
gprof_nn.definitions
====================

This module provides basic definitions that are used throughout the packages.
"""
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
