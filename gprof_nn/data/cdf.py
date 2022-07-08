"""
=================
gprof_nn.data.cdf
=================

Functions to read GPM binary files containing the CDFs
and histograms of the simulated and observed brightness
temperatures.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve
import xarray as xr


def read_cdfs(filename):
    """
    Read CDF and histogram of brightness temperatures from file.

    Args:
        filename: Path pointing to the file.

    Return:
        An 'xarray.Dataset' containing the CDFs and histograms.
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

        shape = (n_tbs, n_tcwv, n_angles, n_channels, n_surfaces)[::-1]
        n_elem = np.prod(shape)

        cdf = np.fromfile(file, "f4", count=n_elem).reshape(shape, order="f")
        hist = np.fromfile(file, "i8", count=n_elem).reshape(shape, order="f")

        tbs = np.arange(tb_min, tb_max + 1)
        tcwv = np.arange(tcwv_min, tcwv_max + 1)

        dims = (
            "brightness_temperatures",
            "total_column_water_vapor",
            "angles",
            "channels",
            "surfaces",
        )[::-1]

        dataset = xr.Dataset(
            {
                "brightness_temperatures": (("brightness_temperatures",), tbs),
                "total_column_water_vapor": (("total_column_water_vapor"), tcwv),
                "cdf": (dims, cdf),
                "histogram": (dims, hist),
            }
        )

    return dataset


def calculate_cdfs(stats_obs, stats_sim):
    """
    Calculate TB CDFs from observation and training data statistics.

    Args:
        stats_obs: ``xarray.Dataset`` containing the statistics of the
            L1C observations.
        stats_sim: ``xarray.Dataset`` containing the statistics of the
            training data.

    Return:
        A tuple ``(cdf_obs, cdf_sim)`` of ``np.ndarray``s containing the CDFs
        of observations and simulations.
    """
    N = 11
    M = 5

    tbs_obs = stats_obs.brightness_temperatures_tcwv.copy()

    k = np.exp(-2 * np.arange(-5, 5.1, 1) ** 2 / 8)
    k = np.stack([k] * M)
    k = k / k.sum()
    shape = [1] * tbs_obs.ndim
    shape[-2] = M
    shape[-1] = N
    k = k.reshape(shape)

    tbs_smoothed = convolve(tbs_obs.data, k, mode="same", method="direct")
    tbs_obs.data[:] = tbs_smoothed
    cdf_obs = tbs_obs.cumsum("brightness_temperature_bins")
    cdf_obs.data /= cdf_obs.data[..., [-1]]
    cdf_obs = cdf_obs.interpolate_na(
        dim="total_column_water_vapor_bins",
        method="nearest"
    ).data

    # Preprocessor removes TBs at outermost angles.
    # Set dists to those of next angles
    if "angles" in stats_obs.dims:
        cdf_obs[:, :, 0] = cdf_obs[:, :, 1]

    tbs_sim = stats_sim.brightness_temperatures_tcwv.copy()
    tbs_smoothed = convolve(tbs_sim.data, k, mode="same", method="direct")
    tbs_sim.data[:] = tbs_smoothed
    cdf_sim = tbs_sim.cumsum("brightness_temperature_bins")
    cdf_sim.data /= cdf_sim.data[..., [-1]]
    cdf_sim = cdf_sim.interpolate_na(
        dim="total_column_water_vapor_bins",
        method="nearest"
    ).data

    # Replace outer angles also for obs derived surface
    # types
    if "angles" in stats_obs.dims:
        cdf_sim[1, :, 0] = cdf_sim[1, :, 1]
        cdf_sim[15, :, 0] = cdf_sim[15, :, 1]
        cdf_sim[7:11, :, 0] = cdf_sim[7:11, :, 1]
        cdf_obs[:, :, 0] = cdf_obs[:, :, 1]

    return cdf_obs, cdf_sim


def calculate_correction(cdf_obs, cdf_sim):
    """
    Calculate TB corrections from observations and training
    data TB CDfs.

    Args:
        cdf_obs: ``numpy.ndarray`` containing the CDFs of the L1C
             brightness temperatures.
        cdf_sim: ``numpy.ndarray`` containing the CDFs of the simulated
             brightness temperatures.

    Return:
        An 'xarray.Dataset' containing the correction statistics.
    """
    corrections = cdf_sim.copy()
    with np.nditer(cdf_sim[..., 0], flags=["multi_index"]) as index:
        while not index.finished:

            cdf_in = cdf_sim[index.multi_index]
            cdf_out = cdf_obs[index.multi_index]

            tbs = np.arange(cdf_in.size) + 0.5
            if not np.any(cdf_out > 0) or not np.any(cdf_in > 0):
                corrections[index.multi_index] = 0.0
                index.iternext()
                continue

            i_start = np.where(cdf_out > 0)[0][0]
            i_end = np.where(cdf_out < 1)[0][-1]
            if i_end - i_start < 2:
                tbs_c = tbs
            else:
                interp = interp1d(
                    cdf_out[i_start:i_end], tbs[i_start:i_end], bounds_error=False
                )
                tbs_c = interp(cdf_in)
                if np.all(np.isnan(tbs_c)):
                    tbs_c = tbs
            d_tb = tbs_c - tbs

            corrections[index.multi_index] = d_tb
            index.iternext()

    if corrections.ndim == 5:
        dims = (
            "surface_type",
            "channels",
            "angles",
            "total_column_water_vapor",
            "brightness_temperatures",
        )
    else:
        dims = (
            "surface_type",
            "channels",
            "total_column_water_vapor",
            "brightness_temperatures",
        )

    dataset = xr.Dataset({"correction": (dims, corrections), "cdf": (dims, cdf_obs)})
    dataset["correction"] = dataset.correction.fillna(0.0)
    return dataset


class CdfCorrection:
    """
    This class implements an correction operator for simulated brightness
    temperatures. This class loads an ``xarray.Dataset`` containing the
    correction statistics and can be used to apply the corresponding
    correction to GPROF-NN training data.

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

    def _apply_correction_cross_track(
        self,
        rng,
        sensor,
        brightness_temperatures,
        surface_type,
        earth_incidence_angle,
        total_column_water_vapor,
        augment=False,
    ):
        """
        Implementation of the TB correction for cross-track scanners.

        Args:
            rng: Random generator to use to generate random numbers.
            sensor: The sensor from which the brightness temperatures originate.
            brightness_temperatures: Array containing the brightness
                temperatures to correct.
            surface_type: Array containing the corresponding surface types.
            earth_incidence_angles: Array containing the corresponding earth
                incidence angles. May be ``None`` for conical scanners.
            total_column_water_vapor: Array containing the corresponding total
                column water vapor.
            augment: Whether or not to apply augment the correction.

        """
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

        # No correction for snow
        st_mask = (st >= 8) * (st <= 11)
        # No correction for sea ice
        st_mask += st == 2
        st_mask += st == 16

        eia_inds = np.digitize(np.abs(eia), sensor.angle_bins)
        eia_inds = np.clip(eia_inds, 1, sensor.n_angles) - 1

        tcwv_inds = np.trunc(tcwv).astype(np.int32)
        tcwv_inds = np.clip(tcwv_inds, self.tcwv_min, self.tcwv_max)

        n_chans = sensor.n_chans
        for i in range(n_chans):
            tbs_inds = np.trunc(
                tbs[..., i],
            ).astype(np.int32)
            tbs_inds = np.clip(tbs_inds, self.tbs_min, self.tbs_max)
            corrections = self.corrections.correction.data[
                st_inds - 1, i, eia_inds, tcwv_inds, tbs_inds
            ]

            if augment:
                quantiles = self.corrections.cdf.data[
                    st_inds - 1, i, eia_inds, tcwv_inds, tbs_inds
                ]
                if st.ndim > 1:
                    err_lo, err_hi = rng.uniform(-1.0, 1.0, size=2)
                else:
                    err_lo = rng.uniform(-1.0, 1.0, size=st.size)
                    err_hi = rng.uniform(-1.0, 1.0, size=st.size)
                err = (1.0 - quantiles) * err_lo + quantiles * err_hi
                err = 0.1 * corrections * err
                corrections += err

            corrections[st_mask] = 0.0
            tbs[..., i] += corrections

        return tbs

    def _apply_correction_conical(
        self,
        rng,
        brightness_temperatures,
        sensor,
        surface_type,
        total_column_water_vapor,
        augment=False,
    ):
        """
        Implementation of the TB correction for conical scanners.

        Args:
            rng: Random generator to use to generate random numbers.
            sensor: The sensor from which the brightness temperatures originate.
            brightness_temperatures: Array containing the brightness
                temperatures to correct.
            surface_type: Array containing the corresponding surface types.
            total_column_water_vapor: Array containing the corresponding total
                column water vapor.
            augment: Whether or not to apply augment the correction.

        """
        st = surface_type
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

        tcwv_inds = np.trunc(tcwv).astype(np.int32)
        tcwv_inds = np.clip(tcwv_inds, self.tcwv_min, self.tcwv_max)

        # No correction for snow
        st_mask = (st >= 8) * (st <= 11)
        # No correction for sea ice
        st_mask += st == 2
        st_mask += st == 16

        n_chans = sensor.n_chans
        for i in range(n_chans):
            tbs_inds = np.trunc(
                tbs[..., i],
            ).astype(np.int32)
            tbs_inds = np.clip(tbs_inds, self.tbs_min, self.tbs_max)
            corrections = self.corrections.correction.data[
                st_inds - 1, i, tcwv_inds, tbs_inds
            ]

            if augment:
                quantiles = self.corrections.cdf.data[
                    st_inds - 1, i, tcwv_inds, tbs_inds
                ]
                err_lo, err_hi = rng.uniform(-1.0, 1.0, size=2)
                err = (1.0 - quantiles) * err_lo + quantiles * err_hi
                err = 0.1 * corrections * err
                corrections += err

            corrections[st_mask] = 0.0
            tbs[..., i] += corrections

        return tbs

    def __call__(
        self,
        rng,
        sensor,
        brightness_temperatures,
        surface_type,
        earth_incidence_angle,
        total_column_water_vapor,
        augment=False,
    ):
        """
        Apply brightness temperature correction to scene.

        Args:
            rng: Random generator to use to generate random numbers.
            sensor: The sensor from which the brightness temperatures originate.
            brightness_temperatures: Array containing the brightness
                temperatures to correct.
            surface_type: Array containing the corresponding surface types.
            earth_incidence_angles: Array containing the corresponding earth
                incidence angles. May be ``None`` for conical scanners.
            total_column_water_vapor: Array containing the corresponding total
                column water vapor.
            augment: Whether or not to apply augment the correction.
        """
        if sensor.n_angles > 1:
            return self._apply_correction_cross_track(
                rng,
                sensor,
                brightness_temperatures,
                surface_type,
                earth_incidence_angle,
                total_column_water_vapor,
                augment=augment,
            )
        return self._apply_correction_conical(
            rng,
            sensor,
            brightness_temperatures,
            surface_type,
            total_column_water_vapor,
            augment=augment,
        )
