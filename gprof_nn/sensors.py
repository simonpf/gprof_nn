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
    FILE_PATTERN = "GMI.dbsatTb.??????{day}.??????.sim"
    N_FREQS = 15 # The number of GMI channels.
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


class MHS:
    """
    The GPROF Microwave Imager (GMI) sensor.
    """
    FILE_PATTERN = "MHS.dbsatTb.??????{day}.??????.sim"
    N_FREQS = 5 # The number of GMI channels.
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
