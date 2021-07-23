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

class ViewingGeometry:
    pass

class Conical(ViewingGeometry):
    """
    Coordination transforms for a conical viewing geometry.

    Args:
        altitude: The altitude of the sensor in m.
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
        R = self.altitude
        a_0 = - self.scan_range / 2
        a = c_p[1] / self.pixels_per_scan * self.scan_range
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

        R = self.altitude
        a = np.rad2deg(np.arcsin(c_x[1] / R))
        j = a / self.scan_range * self.pixels_per_scan

        y_offset = R * (1.0 - np.cos(np.deg2rad(a)))
        i = (c_x[0] + y_offset) / self.scan_offset

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
    450543.0,
    150.0,
    221,
    13e3
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

def get_transformation_coordinates(x_i,
                                   x_o,
                                   y):
    """
    Calculate transformation coordinates that extract a window of
    size M x N. The data is extracted from a region horizontally
    centered around the across-track pixel defined by the fractional
    coordinates 'p_x_i' but transformed to emulate the effect
    of being located a horizontal location 'p_x_o'.

    Args:
        x_i: Across-track, fractional position of the input window given as
            a fraction within [-1, 1].
        x_o: Across-track, fractional position of the output window given as
            a fraction within [-1, 1].
        y: Along track fractional position of the input window.
    """
    c_o, c_i = get_center_pixels(x_o, x_i)
    d_x = offset_x(x_o, x_i)
    d_y = offset_y(x_o, x_i)
    d_y = d_y + np.arange(M).reshape(-1, 1)
    d_x = np.broadcast_to(d_x.reshape(1, -1), d_y.shape)
    coords = np.stack([d_y, d_x])

    y_min = coords[0].min()
    y_max = 220 - coords[0].max()
    y = -y_min + 0.5 * (y + 1.0) * (y_max + y_min)
    coords += np.array([y, c_i]).reshape(-1, 1, 1)
    return coords

def extract_domain(data, x_i, x_o, y,
                   coords=None,
                   order=1):
    """
    Extract and reproject region from input data.

    Args:
        data: Tensor of rank 2 or 3 that containing the data to remap.
        x_i: Fractional horizontal position of input window.
        x_o: Fractional horizontal postition of output window.
        y: Fractional vertical position of output window.
        coords: 2d array containing pre-computed coordinates.

    Return:
        Reprojected subset of 'data'.
    """
    if coords is None:
        coords = get_transformation_coordinates(x_i, x_o, y)
    if data.ndim > 2:
        old_shape = data.shape
        data_c = data.reshape(old_shape[:2] + (-1,))
        results = np.zeros((M, N, data_c.shape[2]))
        for i in range(data_c.shape[2]):
            results[:, :, i] = map_coordinates(
                data_c[:, :, i],
                coords,
                order=order
            )
        results = results.reshape(coords.shape[1:] + old_shape[2:])
    else:
        results = map_coordinates(data, coords, order=order)
    return results


D_A = 13000.0
R = 450543.0
X_S = pixel_to_x(np.arange(221))
