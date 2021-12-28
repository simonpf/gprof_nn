"""
=====================
gprof_nn.augmentation
=====================

This module defines functions for to apply perspective transformations
to GPM GMI observation and to transform them to other viewing
 geometries.
"""
from abc import ABC, abstractmethod
import numpy as np
import scipy
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d

M = 128
N = 96

SCANS_PER_SAMPLE = 221
R_EARTH = 6_371_000
_LATLON_TO_ECEF = None


def latlon_to_ecef():
    """
    Return pyproj transformer object to transform lat/lon to cartesian
    coordinates.
    """
    global _LATLON_TO_ECEF
    if _LATLON_TO_ECEF is None:
        import pyproj
        _LATLON_TO_ECEF =  pyproj.Transformer.from_crs(
            {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        )
    return _LATLON_TO_ECEF


###############################################################################
# Viewing geometries
###############################################################################


class ViewingGeometry(ABC):
    """
    The interface for viewing geometry objects.

    Each class of viewing geometries must provides function to transform
    pixel coordinates to euclidean coordinates and back

    """

    @abstractmethod
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

    @abstractmethod
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


class Swath(ViewingGeometry):
    """
    Viewing geometriy given by 2D fields of latitude and longitude
    coordinates.

    Attributes:
        n_pixels: The number of pixels in the swath
        f_x: Interpolator object to interpolate pixel indices to across-track
            distances.
        f_x: Interpolator object to interpolate across-track distances to pixel
            indices.
        f_y: Interpolator object to interpolate pixel indies to along-track
            offsets.
    """

    def __init__(self, lats, lons):
        """
        Args:
            lats: 2D array containing the latitude coordinates of the swath.
            lons: 2D array containing the longitude coordinates of the
                swath.
        """
        self._calculate_xy(lats, lons)
        self.n_pixels = lats.shape[1]

        j = np.arange(self.n_pixels)
        self.f_x = interp1d(j, self.x, bounds_error=False)
        self.f_x_i = interp1d(self.x, j, bounds_error=False)
        self.f_y = interp1d(j, self.y, bounds_error=False)

    def _calculate_xy(self, lats, lons):
        """
        Map latlon coordinates of swath to 2D euclidean coordinates.
        """
        m, n = lats.shape
        lat_c = lats[m // 2, n // 2]
        lon_c = lons[m // 2, n // 2]

        xyz = latlon_to_ecef().transform(
            lons[m // 2: m // 2 + 2],
            lats[m // 2: m // 2 + 2],
            np.zeros((2, lats.shape[1]), dtype=np.float32),
            radians=False,
        )
        xyz = np.stack(xyz, axis=-1)
        c = xyz[0, n // 2]
        xyz = xyz - c

        # Unit vector in along-track direction.
        d_a = xyz[1, n // 2]
        d_a = d_a / np.sqrt(np.sum(d_a ** 2))

        # Unit vector perpendicular to surface
        d_z = np.stack(latlon_to_ecef().transform(lon_c, lat_c, 1.0, radians=False))
        d_z -= c

        # Unit vector in across-track direction
        d_x = np.cross(d_a, d_z)

        x = np.dot(xyz[0], d_x)
        y = np.dot(xyz[0], d_a)

        self.x = x
        self.y = y
        self.y_a = np.sqrt(np.sum(xyz[1, n // 2] ** 2))

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
        x = self.f_x(c_p[1])
        d_y = self.f_y(c_p[1])

        y = self.y_a * c_p[0] + d_y
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
        j = self.f_x_i(c_x[1])
        d_y = self.f_y(j)

        y = c_x[0] - d_y
        i = y / self.y_a
        return np.stack([i, j])

    def get_window_center(self, p_x, width, height):
        """
        Calculate pixel positions of the center of a window with given width
        and height.
        """
        i = self.y.shape[0] // 2
        j = np.round((self.y.shape[1] - width) * p_x + width / 2)
        return np.array([i, j]).reshape(2, 1, 1)


class Conical(ViewingGeometry):
    """
    Viewing geometry of a conically-scanning sensor.
    """

    def __init__(
        self, altitude, earth_incidence_angle, scan_range, pixels_per_scan, scan_offset
    ):
        """
        Args:
            altitude: The altitude of the sensor in m.
            earth_incidence_angle: The approximate earth incidence angle of the
                sensor.
            scan_range: The active scan range of the sensor.
            pixels_per_scan: The number of pixels contained in each scan.
            scan_offset: The distance between consecutive scans.
        """
        self.earth_incidence_angle = earth_incidence_angle
        eia_rad = np.deg2rad(self.earth_incidence_angle)
        beta = np.arcsin(np.sin(np.pi - eia_rad) / (R_EARTH + altitude) * R_EARTH)
        self._altitude = altitude
        self.zenith_angle = beta
        self.hypotenuse = (
            R_EARTH / np.sin(self.zenith_angle) * np.sin(eia_rad - self.zenith_angle)
        )
        self.scan_radius = self.hypotenuse * np.sin(self.zenith_angle)
        self.scan_range = scan_range
        self.pixels_per_scan = pixels_per_scan
        self.scan_offset = scan_offset

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, altitude):
        eia_rad = np.deg2rad(self.earth_incidence_angle)
        beta = np.arcsin(np.sin(np.pi - eia_rad) / (R_EARTH + altitude) * R_EARTH)
        self._altitude = altitude
        self.zenith_angle = beta
        self.hypotenuse = (
            R_EARTH / np.sin(self.zenith_angle) * np.sin(eia_rad - self.zenith_angle)
        )
        self.scan_radius = self.hypotenuse * np.sin(self.zenith_angle)

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

    def get_window_center(self, p_x, width, height):
        """
        Calculate pixel positions of the center of a window with given width
        and height.
        """
        i = SCANS_PER_SAMPLE // 2
        j = np.round((self.pixels_per_scan - width) * p_x + width / 2)
        return np.array([i, j]).reshape(2, 1, 1)


class CrossTrack(ViewingGeometry):
    """
    Viewing geometry of an across-track-scanning sensor.

    Args:
        altitude: The altitude of the sensor in m.
        scan_range: The active scan range of the sensor.
        pixels_per_scan: The number of pixels contained in each scan.
        scan_offset: The distance between consecutive scans.
    """

    def __init__(self, altitude, scan_range, pixels_per_scan, scan_offset):
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
        y = c_p[0] * self.scan_offset

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
        a = np.sqrt(
            (self.altitude + (1 - np.cos(alpha)) * R_EARTH) ** 2
            + (R_EARTH * np.sin(alpha)) ** 2
        )

        beta = np.arcsin(np.sin(alpha) / a * R_EARTH)
        j = np.rad2deg(beta) / self.scan_range * self.pixels_per_scan
        j += np.floor(self.pixels_per_scan / 2)

        i = c_x[0] / self.scan_offset
        return np.stack([i, j])

    def get_window_center(self, p_x, width, height):
        """
        Calculate pixel positions of the center of a window with given width
        and height.
        """
        i = SCANS_PER_SAMPLE // 2
        j = np.round((self.pixels_per_scan - width) * p_x + width / 2)
        return np.array([i, j]).reshape(2, 1, 1)

    def get_interpolation_weights(self, earth_incidence_angles):
        # Reverse angle so they are in ascending order.
        sin_ang = (
            R_EARTH
            / (R_EARTH + self.altitude)
            * np.sin(np.pi - np.deg2rad(earth_incidence_angles))
        )
        angles = np.rad2deg(np.arcsin(sin_ang))
        angles = angles[::-1]

        weights = np.zeros((self.pixels_per_scan, angles.size), np.float32)
        scan_angles = np.abs(
            np.linspace(-self.scan_range / 2, self.scan_range / 2, self.pixels_per_scan)
        )
        indices = np.digitize(scan_angles, angles)

        for i in range(angles.size - 1):
            mask = (indices - 1) == i
            weights[mask, i] = (angles[i + 1] - scan_angles[mask]) / (
                angles[i + 1] - angles[i]
            )
            weights[mask, i + 1] = (scan_angles[mask] - angles[i]) / (
                angles[i + 1] - angles[i]
            )

        weights[indices == 0] = 0.0
        weights[indices == 0, 0] = 1.0
        weights[indices == angles.size] = 0.0
        weights[indices == angles.size, 0] = -1.0

        # Undo reversal
        return weights[:, ::-1]

    def get_earth_incidence_angles(self):
        """
        Return:
            A numpy array containing the earth incidence angles across a scan
            of the sensor in degrees.
        """
        beta = np.linspace(
            -self.scan_range / 2, self.scan_range / 2, self.pixels_per_scan
        )
        a = np.sin(np.deg2rad(beta)) / R_EARTH * (R_EARTH + self.altitude)
        gamma = -np.arcsin(a) + np.pi
        return np.rad2deg(np.pi - gamma)

    def get_resolution_x(self, earth_incidence_angles):
        """
        Calculate across track resolution for given earth incidence angles.

        Args:
            earth_incidence_angles: Array of earth incidence angles for which
                to compute the across track resolution.

        Return:
            Array containing the across-track resolution in meters.
        """
        # Convert earth incidence angle to viewing angle
        sin_ang = (
            R_EARTH
            / (R_EARTH + self.altitude)
            * np.sin(np.pi - np.deg2rad(earth_incidence_angles))
        )
        beta = np.arcsin(sin_ang)

        c = (R_EARTH + self.altitude) / R_EARTH
        dg_db = -c * np.cos(beta) / np.sqrt(1 - np.sin(beta * c) ** 2)
        da_db = -(dg_db + 1)

        r = da_db * R_EARTH * np.deg2rad(1.1)
        return r

    def get_resolution_a(self):
        """
        The along-track resolution.
        """
        return self.scan_offset


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


def get_center_pixel_input(x, width):
    """
    Returns pixel coordinate of the center of a window of given with
    that covers the swath center of GMI observations.
    """
    l = 110 - width // 2
    r = 110 + width // 2
    return int(np.round(l + (r - l) * x))


def get_transformation_coordinates(
    lats, lons, viewing_geometry, width, height, x_i, x_o, y
):
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

    center_out = viewing_geometry.get_window_center(x_o, width, height)
    d_i = np.arange(height) - np.floor(height / 2.0)
    d_j = np.arange(width) - np.floor(width / 2.0)
    coords_pixel_out = center_out + np.meshgrid(d_i, d_j, indexing="ij")
    coords_eucl_out = viewing_geometry.pixel_coordinates_to_euclidean(coords_pixel_out)
    offset = coords_eucl_out[:, height // 2, width // 2]
    coords_rel_out = coords_eucl_out - offset[:, np.newaxis, np.newaxis]

    center_in = center_out.copy()
    center_in[1, 0, 0] = get_center_pixel_input(x_i, 56)

    input_geometry = Swath(lats, lons)
    coords_center_in = input_geometry.pixel_coordinates_to_euclidean(center_in)
    coords_pixel_in = input_geometry.euclidean_to_pixel_coordinates(
        coords_rel_out + coords_center_in
    )
    coords_pixel_in = np.nan_to_num(coords_pixel_in, nan=-1)
    c = coords_rel_out + coords_center_in

    y_min = coords_pixel_in[0].min()
    y_max = SCANS_PER_SAMPLE - coords_pixel_in[0].max() - 1
    y = -y_min + y * (y_max + y_min)
    coords_pixel_in[0] += y

    return coords_pixel_in


def extract_domain(data, coordinates, order=1):
    """
    Extract and reproject region from input data.

    Args:
        data: Tensor of rank 2 or 3 containing the data to remap.
        coords: 3d array containing pre-computed coordinates.
        order: The interpolation order to use for the reprojection.

    Return:
        Reprojected subset of 'data'.
    """
    if data.ndim > 2:
        old_shape = data.shape
        data_c = data.reshape(old_shape[:2] + (-1,))
        results = []
        for i in range(data_c.shape[2]):
            results.append(
                map_coordinates(data_c[:, :, i], coordinates, order=order, cval=np.nan)
            )
        results = np.stack(results, axis=-1)
        results = results.reshape(coordinates.shape[1:] + old_shape[2:])
    else:
        results = map_coordinates(data, coordinates, order=order, cval=np.nan)
    return results
