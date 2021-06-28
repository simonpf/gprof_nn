"""
================
gprof_nn.sensors
================

This module defines the different sensor class with their respective
properties.
"""
import numpy as np
from gprof_nn.definitions import N_LAYERS


DATE_TYPE = np.dtype(
    [
        ("year", "i4"),
        ("month", "i4"),
        ("day", "i4"),
        ("hour", "i4"),
        ("minute", "i4"),
        ("second", "i4"),
    ]
)


class GMI:
    """
    The GPROF Microwave Imager (GMI) sensor.
    """
    SIM_FILE_PATTERN = "GMI.dbsatTb.??????{day}.??????.sim"
    L1C_FILE_PREFIX = "1C-R.GPM.GMI."
    N_FREQS = 15
    L1C_PATH="/pdata4/archive/GPM/1CR_GMI"
    SIM_FILE_PATH="/qdata1/pbrown/dbaseV7/simV7"
    SIM_FILE_RECORD = np.dtype([
        ("pixel_index", "i4"),
        ("scan_index", "i4"),
        ("data_source", "f4"),
        ("latitude", "f4"),
        ("longitude", "f4"),
        ("elevation", "f4"),
        ("scan_time", DATE_TYPE),
        ("surface_type", "i4"),
        ("surface_precip", "f4"),
        ("convective_precip", "f4"),
        ("emissivity", f"{N_FREQS}f4"),
        ("rain_water_content", f"{N_LAYERS}f4"),
        ("snow_water_content", f"{N_LAYERS}f4"),
        ("cloud_water_content", f"{N_LAYERS}f4"),
        ("latent_heat", f"{N_LAYERS}f4"),
        ("tbs_observed", f"{N_FREQS}f4"),
        ("tbs_simulated", f"{N_FREQS}f4"),
        ("d_tbs", f"{N_FREQS}f4"),
        ("tbs_bias", f"{N_FREQS}f4"),
    ])
    MRMS_FILE_RECORD = np.dtype(
        [
            ("latitude", "f4"),
            ("longitude", "f4"),
            ("scan_time", f"5i4"),
            ("quality_flag", f"f4"),
            ("surface_precip", "f4"),
            ("surface_rain", "f4"),
            ("convective_rain", "f4"),
            ("stratiform_rain", "f4"),
            ("snow", "f4"),
            ("quality_index", "f4"),
            ("gauge_fraction", "f4"),
            ("standard_deviation", "f4"),
            ("n_stratiform", "i4"),
            ("n_convective", "i4"),
            ("n_rain", "i4"),
            ("n_snow", "i4"),
            ("fraction_missing", "f4"),
            ("brightness_temperatures", "15f4"),
        ]
    )
    PREPROCESSOR = "gprof2020pp_GMI_L1C"
    PREPROCESSOR_ORBIT_HEADER = np.dtype([
        ("satellite", "a12"),
        ("sensor", "a12"),
        ("preprocessor", "a12"),
        ("profile_database_file", "a128"),
        ("radiometer_file", "a128"),
        ("calibration_file", "a128"),
        ("granule_number", "i"),
        ("number_of_scans", "i"),
        ("number_of_pixels", "i"),
        ("n_channels", "i"),
        ("frequencies", f"{N_FREQS}f4"),
        ("comment", "a40"),
    ])
    PREPROCESSOR_RECORD = np.dtype([
        ("latitude", "f4"),
        ("longitude", "f4"),
        ("brightness_temperatures", f"{N_FREQS}f4"),
        ("earth_incidence_angle", f"{N_FREQS}f4"),
        ("wet_bulb_temperature", "f4"),
        ("lapse_rate", "f4"),
        ("total_column_water_vapor", "f4"),
        ("surface_temperature", "f4"),
        ("two_meter_temperature", "f4"),
        ("quality_flag", "i"),
        ("sunglint_angle", "i1"),
        ("surface_type", "i1"),
        ("airmass_type", "i2"),
    ])


class MHS:
    """
    The GPROF Microwave Imager (GMI) sensor.
    """
    SIM_FILE_PATTERN = "MHS.dbsatTb.??????{day}.??????.sim"
    L1C_FILE_PREFIX = "1C.METOP?.MHS."
    L1C_PATH="/pdata4/archive/GPM/1C_METOPB"
    SIM_FILE_PATH="/qdata1/pbrown/dbaseV7/simV7"
    N_FREQS = 5
    N_ANGLES = 10
    SIM_FILE_RECORD = np.dtype([
        ("pixel_index", "i4"),
        ("scan_index", "i4"),
        ("latitude", "f4"),
        ("longitude", "f4"),
        ("elevation", "f4"),
        ("scan_time", DATE_TYPE),
        ("surface_type", "i4"),
        ("surface_precip", f"{N_ANGLES}f4"),
        ("convective_precip", f"{N_ANGLES}f4"),
        ("emissivity", f"{N_FREQS * N_ANGLES}f4"),
        ("rain_water_content", f"{N_LAYERS}f4"),
        ("snow_water_content", f"{N_LAYERS}f4"),
        ("cloud_water_content", f"{N_LAYERS}f4"),
        ("latent_heat", f"{N_LAYERS}f4"),
        ("tbs_simulated", f"{N_FREQS * N_ANGLES}f4"),
        ("tbs_bias", f"{N_FREQS}f4")
    ])
    MRMS_FILE_RECORD = np.dtype(
        [
            ("latitude", "f4"),
            ("longitude", "f4"),
            ("scan_time", f"5i4"),
            ("quality_flag", f"f4"),
            ("surface_precip", "f4"),
            ("surface_rain", "f4"),
            ("convective_rain", "f4"),
            ("stratiform_rain", "f4"),
            ("snow", "f4"),
            ("quality_index", "f4"),
            ("gauge_fraction", "f4"),
            ("standard_deviation", "f4"),
            ("n_stratiform", "i4"),
            ("n_convective", "i4"),
            ("n_rain", "i4"),
            ("n_snow", "i4"),
            ("fraction_missing", "f4"),
            ("brightness_temperatures", "5f4"),
        ]
    )
    PREPROCESSOR = "gprof2020pp_MHS_L1C"
    PREPROCESSOR_ORBIT_HEADER = np.dtype([
        ("satellite", "a12"),
        ("sensor", "a12"),
        ("preprocessor", "a12"),
        ("profile_database_file", "a128"),
        ("radiometer_file", "a128"),
        ("calibration_file", "a128"),
        ("granule_number", "i"),
        ("number_of_scans", "i"),
        ("number_of_pixels", "i"),
        ("n_channels", "i"),
        ("frequencies", f"{N_FREQS}f4"),
        ("comment", "a40"),
    ])
    PREPROCESSOR_RECORD = np.dtype([
        ("latitude", "f4"),
        ("longitude", "f4"),
        ("brightness_temperatures", f"{N_FREQS}f4"),
        ("earth_incidence_angle", f"f4"),
        ("wet_bulb_temperature", "f4"),
        ("lapse_rate", "f4"),
        ("total_column_water_vapor", "f4"),
        ("surface_temperature", "f4"),
        ("two_meter_temperature", "f4"),
        ("quality_flag", "i"),
        ("sunglint_angle", "i1"),
        ("surface_type", "i1"),
        ("airmass_type", "i2"),
    ])
