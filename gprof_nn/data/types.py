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


def get_preprocessor_orbit_header(n_chans):
    """
    Args:
        n_chans: The number of frequencies of the sensor.

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
            ("frequencies", f"{n_chans}f4"),
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
            ("frequencies", f"{n_chans}f4"),
            ("comment", "a40"),
        ]
    )
    return dtype

def get_preprocessor_pixel_record(n_chans, kind):
    """
    Args:
        n_chans: The number of frequencies of the sensor.
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
                ("brightness_temperatures", f"{n_chans}f4"),
                ("earth_incidence_angle", f"{n_chans}f4"),
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
                ("brightness_temperatures", f"{n_chans}f4"),
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

def get_bin_file_header(n_chans, n_angles, kind):
    if kind == CONICAL:
        dtype =  np.dtype([
                ("satellite_code", "a5"),
                ("sensor", "a5"),
                ("frequencies", f"{n_chans}f4"),
                ("nominal_eia", f"{n_chans}f4"),
            ])
    else:
        dtype = np.dtype([
                ("satellite_code", "a5"),
                ("sensor", "a5"),
                ("frequencies", "f4", (n_chans,)),
                ("nominal_eia", "f4", (n_angles,)),
            ])
    return dtype


def get_bin_file_record(n_chans,
                        n_angles,
                        n_layers,
                        surface_type,
                        kind):
    """
    Create 'numpy.dtype' describing the binary format used to represent an
    observation in a CSU GPROF bin file.

    Args:
        n_chans: The number of frequencies of the sensor.
        n_layers: The number of layers used to represent profiles.
        n_angles: The number of viewing angles of the sensor.
        surface_type: The surface type of the bin file.
        kind: The type of sensor ('CONICAL' or 'XTRACK')

    Return:
        Numpy dtype that can be used to read entries in a bin file into
        a numpy structured array.
    """
    if kind == CONICAL:
        dtype = np.dtype([
                ("dataset_number", "i4"),
                ("latitude", "f4"),
                ("longitude", "f4"),
                ("scan_time", "i4", (6,)),
                ("surface_precip", np.float32),
                ("convective_precip", np.float32),
                ("brightness_temperatures", "f4", (n_chans,)),
                ("delta_tb", "f4", (n_chans,)),
                ("rain_water_path", np.float32),
                ("cloud_water_path", np.float32),
                ("ice_water_path", np.float32),
                ("total_column_water_vapor", np.float32),
                ("two_meter_temperature", np.float32),
                ("rain_water_content", "f4", (n_layers,)),
                ("cloud_water_content", "f4", (n_layers,)),
                ("snow_water_content", "f4", (n_layers,)),
                ("latent_heat", "f4", (n_layers,)),
            ]
        )
    else:
        if surface_type in [2, 8, 9, 10, 11, 16]:
            dtype = np.dtype(
                [
                    ("dataset_number", "i4"),
                    ("latitude", "f4"),
                    ("longitude", "f4"),
                    ("surface_precip", "f4"),
                    ("convective_precip", "f4"),
                    ("pixel_position", "i4"),
                    ("brightness_temperatures", "f4", (n_chans,)),
                    ("rain_water_path", np.float32),
                    ("cloud_water_path", np.float32),
                    ("ice_water_path", np.float32),
                    ("total_column_water_vapor", np.float32),
                    ("two_meter_temperature", np.float32),
                    ("rain_water_content", "f4", (n_layers,)),
                    ("cloud_water_content", "f4", (n_layers,)),
                    ("snow_water_content", "f4", (n_layers,)),
                    ("latent_heat", "f4", (n_layers,)),
                ]
            )
        else:
            dtype = np.dtype(
                [
                    ("dataset_number", "i4"),
                    ("latitude", "f4"),
                    ("longitude", "f4"),
                    ("surface_precip", "f4", (n_angles,)),
                    ("convective_precip", "f4", (n_angles,)),
                    ("brightness_temperatures", "f4", (n_angles, n_chans)),
                    ("rain_water_path", np.float32),
                    ("cloud_water_path", np.float32),
                    ("ice_water_path", np.float32),
                    ("total_column_water_vapor", np.float32),
                    ("two_meter_temperature", np.float32),
                    ("rain_water_content", "f4", (n_layers,)),
                    ("cloud_water_content", "f4", (n_layers,)),
                    ("snow_water_content", "f4", (n_layers,)),
                    ("latent_heat", "f4", (n_layers,)),
                ]
            )
    return dtype
