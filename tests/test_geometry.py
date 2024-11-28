"""
Tests for the gprof_nn.geometry module.
"""
import numpy as np

from gprof_nn.geometry import (
    ecef_to_lla,
    lla_to_ecef,
    rotate_around,
    calculate_surface_intersection,
    great_circle_distance
)


def test_lla_to_ecef():
    """
    Test conversion from lat-lon to ECEF for some simple geometry cases.
    """
    coords = np.array([0.0, 0.0, 0.0])
    ecef = lla_to_ecef(coords)

    assert np.isclose(ecef[0], 6_378_137.0)
    assert np.isclose(ecef[1], 0.0)
    assert np.isclose(ecef[2], 0.0)

    coords = np.array([90, 0.0, 0.0])
    ecef = lla_to_ecef(coords)
    assert np.isclose(ecef[0], 0.0)
    assert np.isclose(ecef[1], 6_378_137.0)
    assert np.isclose(ecef[2], 0.0)


    coords = np.array([0.0, 90.0, 0.0])
    ecef = lla_to_ecef(coords)
    assert np.isclose(ecef[0], 0.0)
    assert np.isclose(ecef[1], 0.0)
    assert np.isclose(ecef[2], 6_356_752.0)


def test_rotation():
    """
    Test rotation around z axis.
    """
    x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    z = np.array([0.0, 0.0, 1.0])
    x_r = rotate_around(x, z, 90.0)

    assert np.isclose(x_r[0], np.array([0.0, 1.0, 0.0])).all()
    assert np.isclose(x_r[1], np.array([-1.0, 0.0, 0.0])).all()
    assert np.isclose(x_r[2], np.array([0.0, 0.0, 1.0])).all()


def test_ecef_to_lla():
    """
    Test that conversion from ECEF to LLA  works.
    """
    SEM_A = 6_378_137.0
    SEM_B = 6_356_752.0

    coords_ecef = np.array([SEM_A, 0, 0])
    coords_lla = ecef_to_lla(coords_ecef)
    assert np.isclose(coords_lla[0], 0.0)
    assert np.isclose(coords_lla[1], 0.0)
    assert np.isclose(coords_lla[2], 0.0)

    coords_ecef = np.array([0, SEM_A, 0])
    coords_lla = ecef_to_lla(coords_ecef)
    assert np.isclose(coords_lla[0], 90.0)
    assert np.isclose(coords_lla[1], 0.0)
    assert np.isclose(coords_lla[2], 0.0)

    coords_ecef = np.array([0, 0, SEM_B])
    coords_lla = ecef_to_lla(coords_ecef)
    assert np.isclose(coords_lla[0], 0.0)
    assert np.isclose(coords_lla[1], 90.0)
    assert np.isclose(coords_lla[2], 0.0)


def test_calculate_surface_intersection():
    """
    Test calculate of intersection with Earth surface using sensor
    position, line-of-sight, and footprint position of actual AMSR2
    observations.
    """
    sensor_pos_ecef = np.array([-6595187.5 , -2491061.  ,   673912.94])
    sensor_los_ecef = np.array([1004408.  , -433672.75,  255562.06])
    fp_pos_ecef = calculate_surface_intersection(sensor_pos_ecef, sensor_los_ecef)
    fp_pos_lla = ecef_to_lla(fp_pos_ecef)

    assert np.isclose(fp_pos_lla[0], -152.38435)
    assert np.isclose(fp_pos_lla[1], 8.435728)
    assert np.isclose(fp_pos_lla[2], 0.0)


def test_great_circle_distance():
    """
    Test calculation of great circle distance.
    """
    radius = 6371e3
    dist_1deg = 2.0 * np.pi * radius / 360.0
    dist = great_circle_distance(0.0, 0.0, 1.0, 0.0)

    assert np.isclose(dist_1deg, dist)
