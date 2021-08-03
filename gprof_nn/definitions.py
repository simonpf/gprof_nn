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
    "cloud_water_path",
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
LEVELS = np.concatenate([np.linspace(500.0, 1e4, 20), np.linspace(11e3, 18e3, 8)])

DATABASE_MONTHS = [
    (2018, 10),
    (2018, 11),
    (2018, 12),
    (2019, 1),
    (2019, 2),
    (2019, 3),
    (2019, 4),
    (2019, 5),
    (2019, 6),
    (2019, 7),
    (2019, 8),
    (2019, 9),
]


LIMITS = {
    "brightness_temperatures": (0, 400),
    "simulated_brightness_temperatures": (0, 400),
    "brightness_temperature_biases": (-50, 50),
    "total_column_water_vapor": (0, None),
    "two_meter_temperature": (150, 400),
    "surface_precip": (0, 500),
    "viewing_angle": (-180, 180),
    "surface_precip": (0, 500),
    "convective_precip": (0, 500),
    "cloud_water_content": (0, 500),
    "rain_water_content": (0, 500),
    "snow_water_content": (0, 500),
    "latent_heat": (-500, 500),
    "ice_water_path": (0, 500),
    "rain_water_path": (0, 500),
    "cloud_water_path": (0, 500),
    "surface_type": (1, 18),
    "airmass_type": (0, 4),
    "earth_incidence_angle": (-90, 90),
}
