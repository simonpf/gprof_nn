"""
Tests for the loading of surface maps for the GPROF-NN data processing.
"""
from datetime import datetime

import pytest
from gprof_nn.data.surface import (has_surface_type_maps,
                                   get_surface_type_map,
                                   get_mountain_mask)

def test_mountain_mask():
    """
    Ensure that Mount Everest is classified as mountain.
    """
    date = datetime(2018, 1, 1)
    mountain_mask = get_mountain_mask()
    m = mountain_mask.interp(latitude=27.59,
                             longitude=86.55,
                             method="nearest")
    assert m >= 1

@pytest.mark.xfail(condition=not has_surface_type_maps(),
                   reason="Surface maps not available.")
def test_surface_type_map_gmi():
    """
    Ensure that Ngozumpa glacier is classified as mountain snow and
    that a random place in the Ocean is classfied as Ocean.
    """
    date = datetime(2018, 1, 1)
    surface_type_map = get_surface_type_map(date, "GMI")
    st = surface_type_map.interp(latitude=28.01909826766839,
                                 longitude=86.68545082090951,
                                 method="nearest")
    assert st == 18

    st = surface_type_map.interp(latitude=0,
                                 longitude=-24,
                                 method="nearest")
    assert st == 1

@pytest.mark.xfail(condition=not has_surface_type_maps(),
                   reason="Surface maps not available.")
def test_surface_type_map_mhs():
    """
    Ensure that Ngozumpa glacier is classified as mountain snow and
    that a random place in the Ocean is classfied as Ocean.
    """
    date = datetime(2018, 1, 1)
    surface_type_map = get_surface_type_map(date, "GMI")
    st = surface_type_map.interp(latitude=28.01909826766839,
                                 longitude=86.68545082090951,
                                 method="nearest")
    assert st == 18

    st = surface_type_map.interp(latitude=0,
                                 longitude=-24,
                                 method="nearest")
    assert st == 1
