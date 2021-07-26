"""
=====================
gprof_nn.augmentation
=====================

This module defines functions for augmenting GPROF-NN training
data.
"""
import numpy as np
import scipy as sp
from scipy.ndimage import map_coordinates

M = 128
N = 96

SCANS_PER_SAMPLE = 221
R_EARTH = 6_371_000

class ViewingGeometry:
    pass


class Conical(ViewingGeometry):
    """
    Coordination transforms for a conical viewing geometry.

    Args:
        altitude: The altitude of the sensor in m.
        earth_incidence_angle: The approximate earth incidence angle of the
            sensor.
        scan_range: The active scan range of the sensor.
        pixels_per_scan: The number of pixels contained in each scan.
        scan_offset: The distance between consecutive scans.
    """
    def __init__(self,
                 altitude,
                 earth_incidence_angle,
                 scan_range,
                 pixels_per_scan,
                 scan_offset
    ):
        eia_rad = np.deg2rad(earth_incidence_angle)
        beta = np.arcsin(
            np.sin(np.pi - eia_rad) / (R_EARTH + altitude) * R_EARTH
        )
        self.zenith_angle = beta
        self.hypotenuse = (R_EARTH / np.sin(self.zenith_angle) *
                           np.sin(eia_rad - self.zenith_angle))
        self.scan_radius = self.hypotenuse * np.sin(self.zenith_angle)
        self.scan_range = scan_range
        self.pixels_per_scan = pixels_per_scan
        self.scan_offset = scan_offset

    def pixel_coordinates_to_euclidean(self, c_p):
        """
        Transform pixel coordinates to euclidean distances.

        Args:
            c_p: Array with shape ``(2, ...)`` containing the pixel coordinates with
                row indices contained in ``c_p[0]`` and column indices in ``c_p[1]``.

        Returns:
            Array of same shape as ``c_p`` with the along track coordinates in
            ``c_p[0]`` and the across track coordinates in ``c_p[1]``.
        """
        R = self.scan_radius
        a_0 = np.floor(self.pixels_per_scan / 2)
        a = (c_p[1] - a_0) / self.pixels_per_scan * self.scan_range
        x = R * np.sin(np.deg2rad(a))

        y = c_p[0] * self.scan_offset
        dy = R * (1.0 - np.cos(np.deg2rad(a)))
        y = y - dy

        return np.stack([y, x])


    def euclidean_to_pixel_coordinates(self, c_x):
        """
        Transform euclidean distances to pixel coordinates.

        Args:
            c_p: Array with shape ``(2, ...)`` containing the along-track
                coordinates in ``c_p[0]`` and across-track coordinates in
                in ``c_p[1]``.

        Returns:
            Array of same shape as ``c_x`` with the row pixel coordinates in
            ``c_p[0]`` and the column pixel coordinates in ``c_p[1]``.
        """

        R = self.scan_radius
        a = np.rad2deg(np.arcsin(c_x[1] / R))
        j = a / self.scan_range * self.pixels_per_scan
        a_0 = np.floor(self.pixels_per_scan / 2)
        j = j + a_0

        y_offset = R * (1.0 - np.cos(np.deg2rad(a)))
        i = (c_x[0] + y_offset) / self.scan_offset

        return np.stack([i, j])

    def get_window_center(self,
                          p_x,
                          width,
                          height):
        """
        Calculate pixel positions of the center of a window with given width
        and height.
        """
        i = SCANS_PER_SAMPLE // 2
        j = np.round((self.pixels_per_scan - width) * p_x + width / 2)
        return np.array([i, j]).reshape(2, 1, 1)


class CrossTrack(ViewingGeometry):
    """
    Coordination transforms for a conical viewing geometry.

    Args:
        altitude: The altitude of the sensor in m.
        earth_incidence_angle: The approximate earth incidence angle of the
            sensor.
        scan_range: The active scan range of the sensor.
        pixels_per_scan: The number of pixels contained in each scan.
        scan_offset: The distance between consecutive scans.
    """
    def __init__(self,
                 altitude,
                 scan_range,
                 pixels_per_scan,
                 scan_offset
    ):
        self.altitude = altitude
        self.scan_range = scan_range
        self.pixels_per_scan = pixels_per_scan
        self.scan_offset = scan_offset

    def pixel_coordinates_to_euclidean(self, c_p):
        """
        Transform pixel coordinates to euclidean distances.

        Args:
            c_p: Array with shape ``(2, ...)`` containing the pixel coordinates with
                row indices contained in ``c_p[0]`` and column indices in ``c_p[1]``.

        Returns:
            Array of same shape as ``c_p`` with the along track coordinates in
            ``c_p[0]`` and the across track coordinates in ``c_p[1]``.
        """
        a_0 = np.floor(self.pixels_per_scan / 2)
        beta = (c_p[1] - a_0) / self.pixels_per_scan * self.scan_range

        a = np.sin(np.deg2rad(beta)) / R_EARTH * (R_EARTH + self.altitude)
        gamma = -np.arcsin(a) + np.pi
        alpha = np.pi - gamma - np.deg2rad(beta)

        x = R_EARTH * alpha
        y = (SCANS_PER_SAMPLE - c_p[0]) * self.scan_offset

        return np.stack([y, x])

    def euclidean_to_pixel_coordinates(self, c_x):
        """
        Transform euclidean distances to pixel coordinates.

        Args:
            c_p: Array with shape ``(2, ...)`` containing the along-track
                coordinates in ``c_p[0]`` and across-track coordinates in
                in ``c_p[1]``.

        Returns:
            Array of same shape as ``c_x`` with the row pixel coordinates in
            ``c_p[0]`` and the column pixel coordinates in ``c_p[1]``.
        """
        alpha = c_x[1] / R_EARTH
        a = np.sqrt((self.altitude + (1 - np.cos(alpha)) * R_EARTH) ** 2 +
                    (R_EARTH * np.sin(alpha)) ** 2)

        beta = np.arcsin(np.sin(alpha) / a * R_EARTH)
        j = np.rad2deg(beta) / self.scan_range * self.pixels_per_scan
        j += np.floor(self.pixels_per_scan / 2)

        i = SCANS_PER_SAMPLE - (c_x[0] / self.scan_offset)
        return np.stack([i, j])

    def get_window_center(p_x,
                          width,
                          height):
        """
        Calculate pixel positions of the center of a window with given width
        and height.
        """
        i = SCANS_PER_SAMPLE // 2
        l = - self.pixels_per_scan // 2
        j = l + (self.pixels_per_scan - width) * p_x
        return np.array([i, j]).reshape(2, 1, 1)


GMI_GEOMETRY = Conical(
    altitude=407e3,
    earth_incidence_angle=52.8,
    scan_range=140.0,
    pixels_per_scan=221,
    scan_offset=13e3
)

MHS_GEOMETRY =  CrossTrack(
    altitude=855e3,
    scan_range=2.0 * 49.5,
    pixels_per_scan=90,
    scan_offset=17e3
)

def pixel_to_x(j):
    """
    Calculate across-track coordinates of pixel indices.

    Input:
        j: The pixel indices.

    Returns:
        Across-track coordinates in m
    """
    return R * np.sin(np.deg2rad((j - 110.0) / 220 * 150))

def x_to_pixel(x):
     return np.rad2deg(np.arcsin(x / R)) * 220 / 150 + 110


def pixel_to_y(i, j):
    """
    Calculate along track coordinates of pixel indices.
    """
    y = (i * D_A).reshape(-1, 1)
    dy = R * np.cos(np.deg2rad((j.reshape(1, -1) - 110.0) / 220 * 150)) - 1.0
    return y + dy


def get_center_pixels(p_o, p_i):
    """
    Calculate across-track pixel coordinates of input and output domains
    of size M x N.

    Args:
        p_o: Across-track, fractional position of the output window given as
            a fraction within [-1, 1].
        p_i: Along-track, fractional position of the input window give as
            a fraction within [-1, 1].

    Returns:
        Tuple ``(c_o, c_i)`` containing the across track pixel index of the
        output window ``c_o`` and the input window ``c_i``.
    """
    c_o = int(110 + p_o * (110 - N // 2))
    i_min = 130 - N // 2
    i_max = 90 + N // 2
    c_i = int(0.5 * (i_max + i_min + p_i * (i_max - i_min)))
    return c_o, c_i


def offset_x(p_o, p_i):
    """
    Calculate the pixel coordinates of the N pixels within a given
    output window relative to the center of a given input window.

    Args:
        p_o: Across-track, fractional position of the output window given as
            a fraction within [-1, 1].
        p_o: Across-track, fractional position of the input window give as
            a fraction within [-1, 1].

    Return:
        Array of size N containing the across-track pixel coordinates of
        the pixels in the output window transformed to the coordinates system
        of the input window.
    """
    c_o, c_i = get_center_pixels(p_o, p_i)
    x_o = np.arange(c_o - N // 2, c_o + N // 2)
    d_o = pixel_to_x(x_o)
    d_o = d_o - d_o[N // 2] + pixel_to_x(c_i)
    coords = x_to_pixel(d_o)
    coords -= coords[N // 2]
    return coords


def offset_y(p_o, p_i):
    """
    Calculate the vertical pixel coordinates of the a M x N output window
    output window relative to the center of a given input window.

    Args:
        p_o: Across-track, fractional position of the output window given as
            a fraction within [-1, 1].
        p_o: Across-track, fractional position of the input window given as
            a fraction within [-1, 1].

    Return:
        Image of size M x N containing the y-coordinates of the pixels in
        the output window transformed to the coordinate system of the input
        window.
    """
    c_o, c_i = get_center_pixels(p_o, p_i)
    x_o = np.arange(c_o - N // 2, c_o + N // 2)
    y = np.arange(1)

    d_o = pixel_to_x(x_o)
    d_o = d_o - d_o[N // 2] + pixel_to_x(c_i)
    x_coords = x_to_pixel(d_o)

    dy_i = pixel_to_y(y, x_coords)
    dy_o = pixel_to_y(y, x_o)

    return (dy_o - dy_i) / D_A


def get_center_pixel_input(x, width):
    """
    Returns pixel coordinate of the center of a window of given with
    that covers the swath center of GMI observations.
    """
    l = 110 - width // 2
    r = 110 + width // 2
    return int(np.round(l + (r - l) * x))


def get_transformation_coordinates(viewing_geometry,
                                   width,
                                   height,
                                   x_i,
                                   x_o,
                                   y):
    """
    Calculate transformation coordinates that reprojects data from GMI
    swath to a window of size 'width' and 'height' in a given viewing
    geometry and location in swath.

    The position of the observations that are reprojected is defined by
    the 'x_i' and 'y' parameters that define the center of the projection
    input data in terms of fractional position within a sample scene.

    The position of the output window is defined by the 'x_o' argument.

    Args:
        viewing_geometry: Viewing geometry object defining the viewing geometry
            to which the observations should be reprojected.
        width: The width of the output window in pixels.
        height: The height of the output window in pixels.
        x_i: Fractional horizontal position of the input window in the GMI
            swath.
        x_o: Fractional horizontal position of the output window in the given
            viewing geometry.
        y: Fractional vertical position of the input window in the sample
            scene.

    Return:

        Array of shape ``(2, height, width)`` containing the coordinates that
        define the reprojection of the input data with respect to the scene
        coordinates.
    """
    c_i = get_center_pixel_input(x_i, width)
    c_o = viewing_geometry.get_window_center(x_o, width, height)

    d_i = (np.arange(height) - np.floor(height / 2.0))
    d_j = (np.arange(width) - np.floor(width / 2.0))

    c_o = c_o + np.meshgrid(d_i, d_j, indexing="ij")

    c_e = viewing_geometry.pixel_coordinates_to_euclidean(c_o)
    c_i = GMI_GEOMETRY.euclidean_to_pixel_coordinates(c_e)

    y_min = c_i[0].min()
    y_max = SCANS_PER_SAMPLE - c_i[0].max()
    y = -y_min + 0.5 * (y + 1.0) * (y_max + y_min)
    c_i[0] += y

    return c_i


def extract_domain(data,
                   coordinates,
                   order=1):
    """
    Extract and reproject region from input data.

    Args:
        data: Tensor of rank 2 or 3 that containing the data to remap.
        coords: 3d array containing pre-computed coordinates.
        order: The interpolation order to use for the reprojection.

    Return:
        Reprojected subset of 'data'.
    """
    if data.ndim > 2:
        old_shape = data.shape
        data_c = data.reshape(old_shape[:2] + (-1,))
        results = np.zeros((M, N, data_c.shape[2]))
        for i in range(data_c.shape[2]):
            results[:, :, i] = map_coordinates(
                data_c[:, :, i],
                coordinates,
                order=order
            )
        results = results.reshape(coordinates.shape[1:] + old_shape[2:])
    else:
        results = map_coordinates(data, coordinates, order=order)
    return results
