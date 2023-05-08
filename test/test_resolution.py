import numpy as np
import scipy as sp
from scipy.fft import fft2, ifft2
import xarray as xr

from gprof_nn.resolution import FourierAnalysis


def make_test_window_fourier():
    """
    Makes a test structure to test fourier analysis.
    """

    x = np.arange(64).reshape(1, -1)
    y = np.arange(64).reshape(-1, 1)
    # Signal at n = 2 in x and n = 8 in y direction.
    z = np.cos(2 * np.pi * x / 64 * 2) + np.cos(2 * np.pi * y / 64 * 16)

    results = xr.Dataset({
        "surface_precip": (("along_track", "across_track"), z)
    })

    # Only low frequ signel in GPROF
    z_gprof = np.cos(2 * np.pi * x / 64 * 2) + 0.0 * y
    results_gprof = xr.Dataset({
        "surface_precip": (("along_track", "across_track"), z_gprof)
    })

    window = {
        "reference": results,
        "gprof": results_gprof
    }
    return window


def test_fourier_analysis():
    """
    Test calculation of energy spectra using Fourier analysis.
    """
    window = make_test_window_fourier()
    fa = FourierAnalysis(["gprof"])
    fa.process(window)
    r = fa.get_statistics()["gprof"]

    print(r.energy_ret.data)
    # WN 2 present in both datasets
    assert not np.all(np.isclose(r.energy_ret[2 * 2], 0.0))
    assert not np.all(np.isclose(r.energy_ref[2 * 2], 0.0))

    # WN 16 present in only one of them
    assert np.all(np.isclose(r.energy_ret[2 * 16], 0.0))
    assert not np.all(np.isclose(r.energy_ref[2 * 16], 0.0))


