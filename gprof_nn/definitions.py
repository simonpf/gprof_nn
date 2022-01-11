"""
====================
gprof_nn.definitions
====================

This module provides basic definitions that are used throughout the packages.
"""
from pathlib import Path
import numpy as np

MASKED_OUTPUT = -9999

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

SURFACE_TYPE_NAMES = [
    "Ocean",
    "Sea-Ice",
    "Vegetation 1",
    "Vegetation 2",
    "Vegetation 3",
    "Vegetation 4",
    "Vegetation 5",
    "Snow 1",
    "Snow 2",
    "Snow 3",
    "Snow 4",
    "Standing Water",
    "Land Coast",
    "Mixed land/ocean o. water",
    "Ocean or water Coast",
    "Sea-ice edge",
    "Mountain Rain",
    "Mountain Snow",
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

TEST_DAYS = [1, 2, 3]
VALIDATION_DAYS = [4, 5]
TRAINING_DAYS = list(range(6, 32))

LIMITS = {
    "brightness_temperatures": (0, 400),
    "brightness_temperatures_gmi": (0, 400),
    "simulated_brightness_temperatures": (0, 400),
    "brightness_temperature_biases": (-100, 100),
    "total_column_water_vapor": (0, None),
    "two_meter_temperature": (150, 400),
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
    "latitude": (-90, 90),
    "longitude": (-180, 180)
}

GPROF_NN_DATA_PATH = Path(__file__).parent.parent / "data"


# Constants to identify the two retrieval configurations.
ERA5 = "ERA5"
GANAL = "GANAL"
CONFIGURATIONS = [ERA5, GANAL]

TARGET_NAMES = {
    "surface_precip": "Surface precipitation",
    "convective_precip": "Convective precipitation",
    "rain_water_path": "Rain water path",
    "ice_water_path": "Ice water path",
    "cloud_water_content": "Cloud water content",
    "rain_water_content": "Rain water content",
    "snow_water_content": "Snow water content",
    "cloud_water_content": "Cloud water content",
    "latent_heat": "Latent heat"
}

UNITS = {
    "surface_precip": r"$\si{\milli \meter \per \hour}$",
    "convective_precip": r"$\si{\milli \meter \per \hour}$",
    "rain_water_path": r"$\si{\kilogram \per \meter \squared}$",
    "ice_water_path": "$\si{\kilo \gram \per \meter \squared}$",
    "cloud_water_path": "$\si{\kilo \gram \per \meter \squared}$",
    "rain_water_content": "$\si{\gram \per \meter \cubed}$",
    "snow_water_content": "$\si{\gram \per \meter \cubed}$",
    "cloud_water_content": "$\si{\gram \per \meter \cubed}$",
    "latent_heat": "$\si{\kelvin \per \hour}$"
}
