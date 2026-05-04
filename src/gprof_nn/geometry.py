"""
gprof_nn.geometry
=================

This module defines geometric utility functions to manipulate sensor and footprint
positions.
"""
from typing import Optional, Tuple
import warnings

import numpy as np
import xarray as xr


def lla_to_ecef(coords_lla: np.ndarray):
    """
    Converts latitude-longitude-altitude (LLA) coordinates to
    earth-centric earth-fixed coordinates (ECEF)

    Params:
        coords_lla: A numpy.ndarray containing the three coordinates oriented along the last axis.

    Return:
        coords_ecef: An array of the same shape as 'coords_lla' but containing the x, y, and z
             coordinates along the last axis.
    """
    SEM_A = 6_378_137.0
    SEM_B = 6_356_752.0
    ECC2 = 1.0 - (SEM_B ** 2 / SEM_A ** 2)

    lon = np.radians(coords_lla[..., 0])
    lat = np.radians(coords_lla[..., 1])
    alt = coords_lla[..., 2]

    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)

    x = (roc + alt) * np.cos(lat) * np.cos(lon)
    y = (roc + alt) * np.cos(lat) * np.sin(lon)
    z = (roc * (1 - ECC2) + alt) * np.sin(lat)

    return np.stack((x, y, z), -1)


def ecef_to_lla(coords_ecef):
    """
    Converts ECEF coordinates back to LLA coordinates.

    Params:
        coords_ecef: A numpy.ndarray containing the coordinates along the last axis.

    Return:
        coords_lla: A numpy.ndarray of the same shape as 'coords_ecef' containing
            the longitude, latitude, and altitude along tis last axis.
    """
    SEM_A = 6_378_137.0
    SEM_B = 6_356_752.0
    ECC2 = 1.0 - (SEM_B ** 2 / SEM_A ** 2)

    lon = np.arctan2(coords_ecef[..., 1], coords_ecef[..., 0])
    lon = np.nan_to_num(lon, nan=0.0)
    lon = np.degrees(lon)

    p = np.sqrt(coords_ecef[..., 0]**2 + coords_ecef[..., 1]**2)

    lat = np.arctan2(coords_ecef[..., 2], p * (1 - ECC2))
    lat_prev = lat
    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - roc
    lat = np.arctan2(coords_ecef[..., -1], p * (1 - ECC2 * (roc / (roc + alt))))


    while np.max(np.abs(lat - lat_prev)) > 1e-6:
        lat_prev = lat
        roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - roc
        lat = np.arctan2(coords_ecef[..., 2], p * (1 - ECC2 * (roc / (roc + alt))))

    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - roc
    lat = np.degrees(lat)

    if not isinstance(lat, np.ndarray):
        if np.isclose(p, 0.0):
            alt = coords_ecef[..., -1]
            lat = np.sign(alt) * 90
            alt = np.abs(alt) - SEM_B
    else:
        mask = np.isclose(p, 0.0)
        alt[mask] = coords_ecef[mask, -1]
        lat[mask] = np.sign(alt[mask]) * 90
        alt[mask] = np.abs(alt[mask]) - SEM_B

    return np.stack([lon, lat, alt], -1)


def rotate_around(x: np.ndarray, axis: np.ndarray, theta: float):
    """
    Rotate vector x around axis by thate degrees.

    Args:
        x: The vector to rotate with coordinates oriented along the last dimension.
        axis: The axis around with to rotate the vector.
        theta: The number of degrees by which to rotate the vector.

    Return:
        The rotated vector x'

    """
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(theta)

    x_cos = x * np.cos(theta)
    x_cross = np.cross(axis, x, axis=-1) * np.sin(theta)
    x_dot = np.tensordot(x, axis, axes=(-1, -1)) * (1 - np.cos(theta))
    x_axis = x_dot[..., None] * np.broadcast_to(axis, x.shape)

    x_rot = x_cos + x_cross + x_axis
    return x_rot


def calculate_surface_intersection(
        sensor_pos_ecef: np.ndarray,
        sensor_los_ecef: np.ndarray,
):
    """
    Calculate intersection of a line-of-sigh (LOS) with the surface of the Earth.

    Args:
        sensor_pos_ecef: The sensor position in ECEF coordinates
        sensor_los_ecef: The line-of-sight direction in ECEF coordinates

    Return:
        The position of the surface intersection in ECEF coordinates.
    """
    SEM_A = 6_378_137.0
    SEM_B = 6_356_752.0
    ECC2 = 1.0 - (SEM_B ** 2 / SEM_A ** 2)

    coeff_a = (
        sensor_los_ecef[..., 0] ** 2 / SEM_A ** 2 +
        sensor_los_ecef[..., -2] ** 2 / SEM_A ** 2 +
        sensor_los_ecef[..., -1] ** 2 / SEM_B ** 2
    )
    coeff_b = 2.0 * (
        sensor_pos_ecef[..., 0] * sensor_los_ecef[..., 0] / SEM_A ** 2 +
        sensor_pos_ecef[..., 1] * sensor_los_ecef[..., 1] / SEM_A ** 2 +
        sensor_pos_ecef[..., 2] * sensor_los_ecef[..., 2] / SEM_B ** 2
    )
    coeff_c = (
        sensor_pos_ecef[..., 0] ** 2 / SEM_A ** 2 +
        sensor_pos_ecef[..., 1] ** 2 / SEM_A ** 2 +
        sensor_pos_ecef[..., 2] ** 2 / SEM_B ** 2
    ) - 1.0

    discr = coeff_b ** 2 - 4.0 * coeff_a * coeff_c
    root_1 = (np.sqrt(discr) - coeff_b) / (2.0 * coeff_a)
    root_2 = (-np.sqrt(discr) - coeff_b) / (2.0 * coeff_a)

    fac = np.minimum(root_1, root_2)
    pos = sensor_pos_ecef + fac * sensor_los_ecef
    return pos


def great_circle_distance(lon_start, lat_start, lon_end, lat_end):
    """
    Calculate the great-circle distance between two points on a sphere using the Haversine formula.

    Parameters:
        lat1, lon1: Latitude and longitude of the first point in decimal degrees.
        lat2, lon2: Latitude and longitude of the second point in decimal degrees.
        radius: Radius of the sphere (default is Earth's radius in kilometers: 6371 km).

    Returns:
        Distance between the two points on the sphere in the same units as the radius.
    """
    # Approximate radius accurate to upto 0.5%
    radius = 6371e3
    lon_start, lat_start, lon_end, lat_end = map(np.deg2rad, [lon_start, lat_start, lon_end, lat_end])
    dlat = lat_end - lat_start
    dlon = lon_end - lon_start
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_start) * np.cos(lat_end) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    distance = radius * c
    return distance


def calculate_footprints_xtrack(
        lons_fp_gmi: np.ndarray,
        lats_fp_gmi: np.ndarray,
        altitude: float,
        eia_range: Tuple[float, float],
        vai: float,
        n_pixels: int,
        n_scans: int,
        scan_dist: float,
        subsample: int = 1,
        rng: Optional[np.random.Generator] = None
):
    """
    Calculate footprints for a cross-track scanner centered on the GMI swath.

    Args:
        lons_fp_gmi: 2D-array containing the longitude coordinates of the GMI swath.
        lats_fp_gmi: 2D-array containing the latitude coordinates of the GMI swath.
        altitude: The altitude of the sensor.
        eia_range: The range of EIA values to sample.
        vai: The viewing-angle increment per scan position.
        n_pixels: The number of pixel positions to generate.
        n_scans: The number of scans to generate.
        scan_dist: The distance between consecutive scans.
        subsample: Subsampling factor applied to reduce the scans for which new footprints are calculated.
        rng: np.random.Generator = None

    Return:
        An xarray.Dataset containing the 'longitude' and 'latitude' coordinates of the footprints
        of the hypothetic cross-track sensor,  wl
    """
    n_scans_gmi, n_pixels_gmi = lons_fp_gmi.shape

    scan_lons = []
    scan_lats = []
    sat_track = []
    eias = []

    if rng is None:
        rng = np.random.default_rng()
    
    va_range = (
        incidence_angle_to_viewing_angle(eia_range[0], altitude),
        incidence_angle_to_viewing_angle(eia_range[1], altitude),
    )
    eia_max = viewing_to_incidence(va_range[-1] - vai * n_pixels, altitude)
    eia = rng.uniform(eia_range[0], eia_max)

    va = incidence_angle_to_viewing_angle(eia, altitude)
    print(eia, va)

    beta = eia - va
    R = 6.371e6

    if np.isclose(eia, 0.0, atol=1e-3):
        l_los = altitude
    else:
        l_los = np.sin(np.deg2rad(beta)) * (R + altitude) / np.sin(np.pi - np.deg2rad(eia))

    central_pixel = n_pixels_gmi  // 2
    central_scan = n_scans_gmi  // 2

    for scan_ind in range(0, n_scans_gmi, subsample):

        center_lon = lons_fp_gmi[scan_ind, central_pixel]
        center_lat = lats_fp_gmi[scan_ind, central_pixel]
        curr_pos = lla_to_ecef(np.array([center_lon, center_lat, 0.0]))

        if scan_ind < n_scans_gmi - 1:
            next_lon = lons_fp_gmi[scan_ind + 1, central_pixel]
            next_lat = lats_fp_gmi[scan_ind + 1, central_pixel]
            next_pos = np.array([next_lon, next_lat, 0.0])
            flight_dir = lla_to_ecef(next_pos) - curr_pos
        else:
            prev_lon = lons_fp_gmi[scan_ind - 1, central_pixel]
            prev_lat = lats_fp_gmi[scan_ind - 1, central_pixel]
            prev_pos = np.array([prev_lon, prev_lat, 0.0])
            flight_dir = curr_pos - lla_to_ecef(prev_pos)

        sat_pos = np.array([
            center_lon,
            center_lat,
            altitude
        ])
        sat_pos = lla_to_ecef(sat_pos)
        sat_pos = curr_pos + l_los / altitude * rotate_around(
            sat_pos - curr_pos,
            flight_dir,
            eia
        )
        sat_track.append(ecef_to_lla(sat_pos))

        los_center = curr_pos - sat_pos
        los_center /= np.linalg.norm(los_center, axis=-1)

        lim_right = va_range[1] - vai * n_pixels
        va_left = min(va, lim_right)
        print(va, lim_right)
        degs = va_left + vai * np.arange(n_pixels) - va
        all_los = [rotate_around(los_center, flight_dir, deg) for deg in degs]

        old_settings = np.seterr(all='ignore')
        try:
            vis = [
                np.rad2deg(np.arccos(np.sum(los * los_center) / np.linalg.norm(los, axis=-1)  / np.linalg.norm(los_center)))
                for los in all_los
            ]
        finally:
            np.seterr(**old_settings)

        eias.append(viewing_to_incidence(va + degs, altitude))

        footprints = [calculate_surface_intersection(sat_pos, los) for los in all_los]
        footprints = ecef_to_lla(np.array(footprints))

        scan_lons.append(footprints[..., 0])
        scan_lats.append(footprints[..., 1])

    along_track_dist = great_circle_distance(
        lons_fp_gmi[0, central_pixel],
        lats_fp_gmi[0, central_pixel],
        lons_fp_gmi[::subsample, central_pixel],
        lats_fp_gmi[::subsample, central_pixel],
    )
    center_dist = along_track_dist[along_track_dist.size // 2]
    target_dists = np.arange(-n_scans // 2, n_scans // 2) * scan_dist + center_dist

    sat_pos = np.stack(sat_track)
    scan_lons = np.stack(scan_lons)
    scan_lats = np.stack(scan_lats)
    eias = np.stack(eias)

    lon_c_in_1 = lons_fp_gmi[central_scan, 0]
    lat_c_in_1 = lats_fp_gmi[central_scan, 0]
    lon_c_out_1 = scan_lons[scan_lons.shape[0] // 2, -1]
    lat_c_out_1 = scan_lats[scan_lats.shape[0] // 2, -1]
    lon_offset_1 = lon_c_in_1 - lon_c_out_1
    lat_offset_1 = lat_c_in_1 - lat_c_out_1
    dist_cm_1 = np.sqrt(
        (lons_fp_gmi.mean() - scan_lons.mean() - lon_offset_1) ** 2 + (lats_fp_gmi.mean() - scan_lats.mean() - lat_offset_1) ** 2
    )

    lon_c_in_2 = lons_fp_gmi[central_scan, 0]
    lat_c_in_2 = lats_fp_gmi[central_scan, 0]
    lon_c_out_2 = scan_lons[scan_lons.shape[0] // 2, 0]
    lat_c_out_2 = scan_lats[scan_lats.shape[0] // 2, 0]
    lon_offset_2 = lon_c_in_2 - lon_c_out_2
    lat_offset_2 = lat_c_in_2 - lat_c_out_2
    dist_cm_2 = np.sqrt(
        (lons_fp_gmi.mean() - scan_lons.mean() - lon_offset_2) ** 2 + (lats_fp_gmi.mean() - scan_lats.mean() - lat_offset_2) ** 2
    )

    if dist_cm_1 < dist_cm_2:
        lon_offset = lon_c_in_1 - lon_c_out_1
        lat_offset = lat_c_in_1 - lat_c_out_1
    else:
        lon_offset = lon_c_in_2 - lon_c_out_2
        lat_offset = lat_c_in_2 - lat_c_out_2

    scan_lons += lon_offset
    scan_lats += lat_offset

    coords = xr.Dataset({
        "scans": (("scans",), along_track_dist),
        "longitude": (("scans", "pixels"), scan_lons),
        "latitude": (("scans", "pixels"), scan_lats),
        "sensor_longitude": (("scans",), sat_pos[..., 0]),
        "sensor_latitude": (("scans",), sat_pos[..., 1]),
        "sensor_altitude": (("scans",), sat_pos[..., 2]),
        "earth_incidence_angle": (("scans", "pixels"), eias),
    })
    return coords.interp(scans=target_dists)


def calculate_footprints_conical(
    lons_fp_gmi: np.ndarray,
    lats_fp_gmi: np.ndarray,
    altitude: float,
    eia: float,
    scan_angle_range: Tuple[float, float],
    sai: float,
    n_pixels: int,
    n_scans: int,
    scan_dist: float,
    subsample: int = 1,
    rng: Optional[np.random.Generator] = None
):
    """
    Calculate footprints for a conical scanner centered on the GMI swath.

    Args:
        lons_fp_gmi: 2D-array containing the longitude coordinates of the GMI swath.
        lats_fp_gmi: 2D-array containing the latitude coordinates of the GMI swath.
        altitude: The altitude of the sensor.
        eia: The range of EIA values to sample.
        scan_range: The range of scan-angles to sample
        sai: The viewing-angle increment per scan position.
        n_pixels: The number of pixel positions to generate.
        n_scans: The number of scans to generate.
        scan_dist: The distance between consecutive scans.
        subsample: Scalar factor used to subsample the scans for which the viewing coordinates
            are calculated.

    Return:
        An xarray.Dataset containing the 'longitude' and 'latitude' coordinates of the footprints
        of the hypothetic cross-track sensor,  wl
    """
    n_scans_gmi, n_pixels_gmi = lons_fp_gmi.shape

    scan_lons = []
    scan_lats = []
    sat_track = []

    va = incidence_angle_to_viewing_angle(eia, altitude)
    beta = eia - va
    R = 6.371e6
    l_los = R * np.sin(np.deg2rad(beta)) / np.sin(np.deg2rad(va))
        #
        #
    if rng is None:
        rng = np.random.default_rng()

    sa_min = scan_angle_range[0]
    sa_max = scan_angle_range[0] - n_pixels * sai
    scan_angle = rng.uniform(*scan_angle_range)

    #central_pixel = int(
    #    (n_pixels_gmi - 1) * (scan_angle - scan_angle_range[0]) / (scan_angle_range[1] - scan_angle_range[0])
    #)
    central_pixel = n_scans_gmi  // 2

    for scan_ind in range(0, n_scans_gmi, subsample):

        center_lon = lons_fp_gmi[scan_ind, central_pixel]
        center_lat = lats_fp_gmi[scan_ind, central_pixel]
        curr_pos = lla_to_ecef(np.array([center_lon, center_lat, 0.0]))

        if scan_ind < n_scans_gmi - 1:
            next_lon = lons_fp_gmi[scan_ind + 1, central_pixel]
            next_lat = lats_fp_gmi[scan_ind + 1, central_pixel]
            next_pos = np.array([next_lon, next_lat, 0.0])
            flight_dir = lla_to_ecef(next_pos) - curr_pos
        else:
            prev_lon = lons_fp_gmi[scan_ind - 1, central_pixel]
            prev_lat = lats_fp_gmi[scan_ind - 1, central_pixel]
            prev_pos = np.array([prev_lon, prev_lat, 0.0])
            flight_dir = curr_pos - lla_to_ecef(prev_pos)

        up = lla_to_ecef(np.array([center_lon, center_lat, 1.0])) - curr_pos
        x_track = np.cross(flight_dir, up, axis=-1)

        sat_pos = np.array([
            center_lon,
            center_lat,
            altitude
        ])

        sat_track.append(sat_pos)
        sat_pos = lla_to_ecef(sat_pos)

        radius = 6371e3
        sat_pos = curr_pos + l_los / altitude * rotate_around(sat_pos - curr_pos, x_track, -eia)
        sat_pos = curr_pos + rotate_around(sat_pos - curr_pos, up, scan_angle)

        los_center = curr_pos - sat_pos
        los_center /= np.linalg.norm(los_center, axis=-1)

        degs = scan_angle + sai * np.arange(n_pixels)

        sub_sensor = ecef_to_lla(sat_pos)
        sub_sensor[..., 2] = 0.0
        sub_sensor_up = sub_sensor.copy()
        sub_sensor_up[..., 2] = 1.0
        sensor_up = lla_to_ecef(sub_sensor_up) - lla_to_ecef(sub_sensor)
        all_los = [rotate_around(los_center, sensor_up, deg) for deg in degs]

        footprints = [calculate_surface_intersection(sat_pos, los) for los in all_los]
        footprints = ecef_to_lla(np.array(footprints))

        scan_lons.append(footprints[..., 0])
        scan_lats.append(footprints[..., 1])

    along_track_dist = great_circle_distance(
        lons_fp_gmi[0, central_pixel],
        lats_fp_gmi[0, central_pixel],
        lons_fp_gmi[::subsample, central_pixel],
        lats_fp_gmi[::subsample, central_pixel],
    )
    center_dist = along_track_dist[along_track_dist.size // 2]
    target_dists = np.arange(-n_scans // 2, n_scans // 2) * scan_dist + center_dist

    sat_pos = np.stack(sat_track)
    scan_lons = np.stack(scan_lons)
    scan_lats = np.stack(scan_lats)

    lon_c_0 = lons_fp_gmi.mean()
    lat_c_0 = lats_fp_gmi.mean()
    lon_c_1 = scan_lons.mean()
    lat_c_1 = scan_lats.mean()

    coords = xr.Dataset({
        "scans": (("scans",), along_track_dist),
        "longitude": (("scans", "pixels"), scan_lons - lon_c_1 + lon_c_0),
        "latitude": (("scans", "pixels"), scan_lats - lat_c_1 + lat_c_0),
        "sensor_longitude": (("scans",), sat_pos[..., 0]),
        "sensor_latitude": (("scans",), sat_pos[..., 1]),
        "sensor_altitude": (("scans",), sat_pos[..., 2]),
    })

    return coords.interp(scans=target_dists)


def incidence_angle_to_viewing_angle(
        incidence_angle: float,
        altitude: float
) -> float:
    """
    Calculates viewing angle from incidence angles.

    Args:
        incidence_angle: The earth incidence angles in degree for which to
            calculate the viewing angles.
        altitude: The altitude of the sensor in meters.

    Return:
        The calculated viewing angles in degree.
    """
    R = 6.371e6
    sin_alpha = R * np.sin(np.pi - np.deg2rad(incidence_angle)) / (R + altitude)
    alpha = np.arcsin(sin_alpha)
    return np.rad2deg(alpha)


def viewing_to_incidence(
        viewing_angle: float,
        altitude: float
) -> float:
    """
    Convert satellite viewing angle to Earth incidence angle.

    Args:
        viewing_angle: The viewing angle in degree
        altitude: The satellite altitude

    Return:
        The earth incidence angle.
    """
    R = 6371.0e3

    theta_v = np.radians(viewing_angle)
    theta_i = np.arcsin((R + altitude) / R * np.sin(theta_v))
    return np.rad2deg(theta_i)
