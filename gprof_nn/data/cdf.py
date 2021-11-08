"""
=================
gprof_nn.data.cdf
=================

Functions to read Paula Browns binary files containing the CDFs
and histograms of the simulated and observed brightness
temperatures.
"""
from pathlib import Path

import numpy as np
import xarray as xr

def read_cdfs(filename):
    """
    Read CDF and histogram ofebrightness temperatures from file.

    Args:
        filename: Path pointing to the file.

    Return:
        An 'xarray.Dataset' containing the CDFs and histograms
        contained in the file.
    """
    with open(filename, "rb") as file:
        meta = np.fromfile(file, "7i", count=1)[0]
        n_surfaces = meta[0]
        n_channels = meta[1]
        n_angles = meta[2]
        tcwv_min = meta[3]
        tcwv_max = meta[4]
        tb_min = meta[5]
        tb_max = meta[6]

        n_tcwv = tcwv_max - tcwv_min + 1
        n_tbs = tb_max - tb_min + 1

        print(n_surfaces, n_channels, n_angles, n_tcwv, n_tbs)

        shape = (n_tbs, n_tcwv, n_angles, n_channels, n_surfaces)[::-1]
        n_elem = np.prod(shape)
        shape_all = (n_channels, n_angles, n_tbs)
        n_elem_all = np.prod(shape_all)

        cdf = np.fromfile(file, "f4", count=n_elem).reshape(shape, order="f")
        hist = np.fromfile(file, "i8", count=n_elem).reshape(shape, order="f")
        cdf_all = np.fromfile(file, "f4", count=n_elem_all).reshape(
            shape_all, order="f"
        )
        hist_all = np.fromfile(
            file,
            "i8",
            count=n_elem_all
        ).reshape(shape_all, order="f")

        tbs = np.arange(tb_min, tb_max + 1)
        tcwv = np.arange(tcwv_min, tcwv_max + 1)


        dims = ("brightness_temperatures",
                "total_column_water_vapor",
                "angles",
                "channels",
                "surfaces")[::-1]
        dims_all = ("channels", "angles", "brightness_temperatures")

        dataset = xr.Dataset({
            "brightness_temperatures": (("brightness_temperatures",), tbs),
            "total_column_water_vapor": (("total_column_water_vapor"), tcwv),
            "cdf": (dims, cdf),
            "histogram": (dims, hist),
            "cdf_all_surfaces": (dims_all, cdf_all),
            "hist_all_surfaces": (dims_all, hist_all)
            })

    return dataset
