"""
===================
gprof_nn.data.types
===================

Defines numpy structured array types used to read the different binary formats
used for the GPROF processing at CSU.
"""
import numpy as np

CONICAL = "CONICAL"
XTRACK = "XTRACK"


def get_preprocessor_orbit_header(n_freqs):
    """
    Args:
        n_freqs: The number of frequencies of the sensor.

    Return:
        The numpy datatype corresponding to the orbit header of the
        binary preprocessor file format.
    """
    dtype = np.dtype(
        [
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
            ("frequencies", f"{n_freqs}f4"),
            ("comment", "a40"),
        ]
    )
    dtype = np.dtype(
        [
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
            ("frequencies", f"{n_freqs}f4"),
            ("comment", "a40"),
        ]
    )
    return dtype


def get_preprocessor_pixel_record(n_freqs, kind):
    """
    Args:
        n_freqs: The number of frequencies of the sensor.
        kind: The sensor kind ('CONICAL' or 'XTRACK')
    Return:
        The numpy datatype corresponding to a pixel record of the
        binary preprocessor file format.
    """
    if kind == CONICAL:
        dtype = np.dtype(
            [
                ("latitude", "f4"),
                ("longitude", "f4"),
                ("brightness_temperatures", f"{n_freqs}f4"),
                ("earth_incidence_angle", f"{n_freqs}f4"),
                ("wet_bulb_temperature", "f4"),
                ("lapse_rate", "f4"),
                ("total_column_water_vapor", "f4"),
                ("surface_temperature", "f4"),
                ("two_meter_temperature", "f4"),
                ("quality_flag", "i"),
                ("sunglint_angle", "i1"),
                ("surface_type", "i1"),
                ("airmass_type", "i2"),
            ]
        )
    else:
        dtype = np.dtype(
            [
                ("latitude", "f4"),
                ("longitude", "f4"),
                ("brightness_temperatures", f"{n_freqs}f4"),
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
            ]
        )
    return dtype
