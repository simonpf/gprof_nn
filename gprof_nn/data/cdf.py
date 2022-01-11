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
        #cdf_all = np.fromfile(file, "f4", count=n_elem_all).reshape(
        #    shape_all, order="f"
        #)
        #hist_all = np.fromfile(
        #    file,
        #    "i8",
        #    count=n_elem_all
        #).reshape(shape_all, order="f")

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
            #"cdf_all_surfaces": (dims_all, cdf_all),
            #"hist_all_surfaces": (dims_all, hist_all)
            })

    return dataset


class CdfCorrection:
    """
    This class implements an correction operator for simulated brightness
    temperatures based on the histogram matching of simulated and observed
    brightness temperatures.

    Attributes:
        corrections: ``xarray.Dataset`` containing the prepared correction
            data.
    """
    def __init__(self, correction_file):
        """
        Args:
            correction_file: Path to the NetCDF file containing the prepared
                correction data.
        """
        self.correction_file = correction_file
        self._corrections = None
        tbs = self.corrections.brightness_temperatures.data
        self.tbs_min = tbs.min()
        self.tbs_max = tbs.max()
        tcwv = self.corrections.total_column_water_vapor.data
        self.tcwv_min = tcwv.min()
        self.tcwv_max = tcwv.max()
        self.surface_types = self.corrections.surface_type.size

    @property
    def corrections(self):
        """
        Lazily loaded correctionas as 'xarray.Dataset'.
        """
        if self._corrections is None:
            self._corrections = xr.load_dataset(self.correction_file)
        return self._corrections

    def _apply_correction_cross_track(self,
                                      sensor,
                                      surface_type,
                                      earth_incidence_angle,
                                      total_column_water_vapor,
                                      brightness_temperatures,
                                      augment=False):
        st = surface_type
        eia = earth_incidence_angle
        tcwv = total_column_water_vapor
        tbs = brightness_temperatures.copy()

        if self.surface_types == 3:
            # CDF uses 3 surface type:
            # - 1: Ocean
            # - 2: Land
            # - 3: Coast
            st_inds = st.astype(np.int32)
            st_inds[st_inds > 1] = 2
            st_inds[st == 13] = 1
        else:
            st_inds = st.astype(np.int32)

        n_bins = sensor.n_angles
        eia_inds = np.digitize(np.abs(eia), sensor.angle_bins)
        eia_inds = np.clip(eia_inds, 1, sensor.n_angles) - 1

        tcwv_inds = np.trunc(tcwv).astype(np.int32)
        tcwv_inds = np.clip(tcwv_inds, self.tcwv_min, self.tcwv_max)

        n_chans = sensor.n_chans
        for i in range(n_chans):
            tbs_inds = np.trunc(tbs[..., i], ).astype(np.int32)
            tbs_inds = np.clip(tbs_inds, self.tbs_min, self.tbs_max)
            corrections = self.corrections.correction.data[
                st_inds - 1, i, eia_inds, tcwv_inds, tbs_inds
            ]

            if augment:
                quantiles = self.corrections.cdf.data[
                    st_inds - 1, i, eia_inds, tcwv_inds, tbs_inds
                ]
                err_lo, err_hi = np.random.normal(size=2)
                err = (1.0 - quantiles) * err_lo + quantiles * err_hi
                err = 0.1 * corrections * err
                shape = corrections.shape
                corrections += err

            #corrections[st_inds > 1] = 0.0

            tbs[..., i] += corrections

        return tbs

    def __call__(
            self,
            sensor,
            surface_type,
            earth_incidence_angle,
            total_column_water_vapor,
            brightness_temperatures,
            augment=False
    ):
        """
        Apply correction to scene.

        Args:
            scene: A training scene to which to apply the TB correction.
            augment: Whether or not to apply augment the correction.
        """
        return self._apply_correction_cross_track(
            sensor, surface_type, earth_incidence_angle,
            total_column_water_vapor, brightness_temperatures, augment=augment
        )


