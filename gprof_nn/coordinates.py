"""
====================
gprof_nn.coordinates
====================

Module providing helper function to convert between coordinate
systems.
"""
import numpy as np


LATLON_TO_ECEF_TRANSFORMER = None


def get_latlon_to_ecef_transformer():
    """
    Function to delay import of pyproj.
    """
    global LATLON_TO_ECEF_TRANSFORMER

    if LATLON_TO_ECEF_TRANSFORMER is None:
        try:
            from pyproj import Transformer

            LATLON_TO_ECEF_TRANSFORMER = Transformer.from_crs("epsg:4326", "epsg:4978")
        except:
            raise ModuleNotFoundError(
                """
                This functionality of the 'gprof_nn' package requires the 'pyproj'
                package to be installed on your system.
                """
            )
            LATLON_TO_ECEF_TRANSFORMER = None

    return LATLON_TO_ECEF_TRANSFORMER


def latlon_to_ecef(longitudes, latitudes, altitudes=None):
    """
    Convert latitudes and longitudes to Earth-centered earth-fixed (ECEF)
    euclidean coordinates.

    Args:
        longitudes: Array containing the longitudes which to transform
            to ECEF coordinates
        longitudes: Array containing the latitudes which to transform
            to ECEF coordinates
        altitudes: Optional array of altitudes corresponding to the given
            longitudes and latitudes. If not provided an altitude of
            of 0 is assumed for all points.

    Returns:
        Tuple ``(x, y, z)`` consisting of the arrays ``x``, ``y``, ``z``
        with each holding the corresponding component of the converted
        3D coordinates.
    """
    longitudes = np.array(longitudes)
    latitudes = np.array(latitudes)


    if altitudes is None:
        altitudes = np.zeros(latitudes.shape)
    else:
        altitudes = np.array(altitudes)

    transformer = get_latlon_to_ecef_transformer()
    return transformer.transform(latitudes, longitudes, altitudes)
