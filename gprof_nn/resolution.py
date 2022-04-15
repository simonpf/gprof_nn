"""
gprof_nn.resolution
===================

This module implements functionality to calculate the effective resolution
of the retrieval using wavelet decomposition.
"""
import numpy as np
import pywt
import xarray as xr


class ResolutionCalculator:
    """
    The ResolutionCalculator class accumulates statistics of reference
    and retrieved wavelet coefficients. It can be used to calculate
    total energy, correlation and NS coefficients of the retrieved fields.
    """
    def __init__(self,
                 reference_group,
                 retrieval_groups,
                 window_size=96):
        """
        Args:
            reference_group: Name of the NetCDF4 group containing the
                reference precipitation fields.
            retrieval_groups: The name of the groups containing the retrieval
                results.
            window_size: The size of the windows for which to calculate the
                wavelet coefficients.
        """
        self.reference_group = reference_group
        self.retrieval_groups = retrieval_groups
        self.window_size = window_size
        self.valid_fraction = 1.0
        self.results = {
            group: xr.Dataset() for group in retrieval_groups
        }


    def get_windows(self, filename):
        """
        Extract windows without missing data in reference field
        and retrieval fields from NetCDF file.

        Args:
            filename: Path to the NetCDF file containing co-located retrieval
                results.

        Return:
            An iterator the provides dicts of xarray.Datsets each containing
            a quadratic window of size ``self.window`` in which no surface
            precipitation values are missing.
        """
        reference_group = self.reference_group
        groups = self.retrieval_groups

        reference_scene = xr.load_dataset(filename, group=reference_group)
        other_scenes = [xr.load_dataset(filename, group=group)
                        for group in groups]

        sp_ref = reference_scene.surface_precip
        sp_fields = [scene["surface_precip"] for scene in other_scenes]
        n_scans = sp_ref.along_track.size
        n_pixels = sp_ref.across_track.size

        scan_start = 0
        scan_end = scan_start + self.window_size
        while scan_end < n_scans:

            pixel_start = 0
            pixel_end = pixel_start + self.window_size
            found_box = False

            while pixel_end < n_pixels:


                indices = {
                    "along_track": slice(scan_start, scan_end),
                    "across_track": slice(pixel_start, pixel_end)
                }
                window = reference_scene[indices]
                valid_frac = (window.surface_precip.data >= 0).mean()
                windows = [
                    scene[indices].interpolate_na(
                        "across_track",
                        "linear",
                    ) for scene in other_scenes
                ]
                valid_fracs = [
                    (scene.surface_precip.data >= 0).mean() for scene in windows
                ]

                valid = valid_frac >= self.valid_fraction
                valid &= min(valid_fracs) >= self.valid_fraction
                if "radar_quality_index" in window.variables:
                    rqi = window.radar_quality_index.data
                    rqi_min = rqi.min()
                    #valid &= rqi_min > 0.5


                if valid:
                    results = {
                        group: window for group, window in zip(groups, windows)
                    }
                    results["reference"] = window
                    yield results
                    found_box = True
                    pixel_start += self.window_size
                    pixel_end += self.window_size
                else:
                    pixel_start += self.window_size // 4
                    pixel_end += self.window_size // 4

            if found_box:
                scan_start += self.window_size
                scan_end += self.window_size
            else:
                scan_start += self.window_size // 4
                scan_end += self.window_size // 4


    def process_window(self, window):
        """
        Process a single window and accumulat statistics in ``results``
        attribute.

        Args:
            window: A dict containing the xarray.Datsets with the
                retrieved precipitation windows for the reference data
                and all retrieval datasets.
        """


        reference = window["reference"]

        w_ref = pywt.WaveletPacket2D(
            data=reference.surface_precip.data,
            wavelet="haar",
            mode="constant",
            maxlevel=6
        )

        for group in window:
            if group == "reference":
                continue

            w_ret = pywt.WaveletPacket2D(
                data=window[group].surface_precip.data,
                wavelet="haar",
                mode="constant",
                maxlevel=6
            )

            levels = w_ret.maxlevel

            coeffs_ref = w_ref
            coeffs_ret = w_ret

            for i in range(levels):

                dims = (f"h_{i}", f"v_{i}")

                for axis in ["h", "v", "d"]:

                    r = self.results[group]
                    c_ref = coeffs_ref[axis].data
                    c_ret = coeffs_ret[axis].data
                    varname = f"coeffs_ref_{axis}_{i}_sum"
                    if varname not in r.variables:
                        r[f"coeffs_ref_{axis}_{i}_sum"] = (dims, c_ref)
                        r[f"coeffs_ref_{axis}_{i}_sum2"] = (dims, c_ref ** 2)
                        r[f"coeffs_ret_{axis}_{i}_sum"] = (dims, c_ret)
                        r[f"coeffs_ret_{axis}_{i}_sum2"] = (dims, c_ret ** 2)
                        r[f"coeffs_refret_{axis}_{i}_sum"] = (dims, c_ret * c_ref)
                        r[f"coeffs_diff_{axis}_{i}_sum"] = (dims, c_ret - c_ref)
                        r[f"coeffs_diff_{axis}_{i}_sum2"] = (dims, (c_ret - c_ref) ** 2)
                        r[f"counts_{axis}_{i}"] = (
                            dims,
                            np.isfinite(c_ret - c_ref).astype(np.float64)
                        )
                    else:
                        r[f"coeffs_ref_{axis}_{i}_sum"] += c_ref
                        r[f"coeffs_ref_{axis}_{i}_sum2"] += c_ref ** 2
                        r[f"coeffs_ret_{axis}_{i}_sum"] += c_ret
                        r[f"coeffs_ret_{axis}_{i}_sum2"] += c_ret ** 2
                        r[f"coeffs_refret_{axis}_{i}_sum"] += c_ret * c_ref
                        r[f"coeffs_diff_{axis}_{i}_sum"] += c_ret - c_ref
                        r[f"coeffs_diff_{axis}_{i}_sum2"] += (c_ret - c_ref) ** 2
                        r[f"counts_{axis}_{i}"] += np.isfinite(c_ret).astype(
                            np.float64
                        )

                coeffs_ref = coeffs_ref["a"]
                coeffs_ret = coeffs_ret["a"]

            varname = f"a_ref_{i + 1}_sum"
            c_ref = coeffs_ref.data
            c_ret = coeffs_ret.data
            if varname not in r.variables:
                r[f"a_ref_{i + 1}_sum2"] = (dims, c_ref ** 2)
                r[f"a_ret_{i + 1}_sum2"] = (dims, c_ret ** 2)
                r[f"counts_a"] = (dims, np.isfinite(c_ret).astype(np.float64))
            else:
                r[f"a_ref_{i + 1}_sum2"] += c_ref ** 2
                r[f"a_ret_{i + 1}_sum2"] += c_ret ** 2
                r[f"counts_a"] += np.isfinite(c_ret).astype(np.float64)


    def process_file(self, filename):
        """
        Extract wavelet for all windows in a file.
        """
        for window in self.get_windows(filename):
            self.process_window(window)

    def get_statistics(self, scale=5e3):
        """
        Return error statistics for correlation coefficients
        by scale.

        Args:
            scale: Grid size of the underlying grid, which is required
                to calculate the scale corresponding to the wavelet
                coefficients.
        """
        results = {}

        for group in self.retrieval_groups:

            corr_coeffs = []
            coherence = []
            energy_ret = []
            energy_ref = []
            mse = []

            retrieved = self.results[group]
            variables = list(retrieved.variables.keys())
            a_coeffs = [var for var in variables if var.startswith("a_ref_")]
            n_levels = int(a_coeffs[0][6:7]) - 1
            scales = scale * np.array([2 ** i for i in range(n_levels)])

            for i in range(n_levels):

                cc_s = []
                co_s = []
                e_ref_s = []
                e_ret_s = []
                d2_s = []

                for axis in ["h", "v", "d"]:

                    c_ref_s = retrieved[f"coeffs_ref_{axis}_{i}_sum"].data
                    c_ref_s2 = retrieved[f"coeffs_ref_{axis}_{i}_sum2"].data
                    c_ret_s = retrieved[f"coeffs_ret_{axis}_{i}_sum"].data
                    c_ret_s2 = retrieved[f"coeffs_ret_{axis}_{i}_sum2"].data
                    c_refret_s = retrieved[f"coeffs_refret_{axis}_{i}_sum"].data
                    c_d_s2 = retrieved[f"coeffs_diff_{axis}_{i}_sum2"].data
                    counts = retrieved[f"counts_{axis}_{i}"].data

                    sigma_ref = c_ref_s2 / counts - (c_ref_s / counts) ** 2
                    sigma_ret = c_ret_s2 / counts - (c_ret_s / counts) ** 2
                    print(counts)
                    ref_mean = c_ref_s / counts
                    ret_mean = c_ret_s / counts
                    refret_mean = c_refret_s / counts
                    cc = (
                        (refret_mean - ref_mean * ret_mean) /
                        (np.sqrt(sigma_ref) * np.sqrt(sigma_ret))
                    )
                    co = np.abs(c_refret_s) / (np.sqrt(c_ref_s2) * np.sqrt(c_ret_s2))

                    cc = cc[np.isfinite(cc)]
                    cc_s.append(cc.ravel())
                    co_s.append(co.ravel())
                    e_ref_s.append(c_ref_s2.ravel())
                    e_ret_s.append(c_ret_s2.ravel())
                    d2_s.append(c_d_s2.ravel())

                cc_s = np.concatenate(cc_s)
                co_s = np.concatenate(co_s)
                e_ref_s = np.concatenate(e_ref_s)
                e_ret_s = np.concatenate(e_ret_s)
                d2_s = np.concatenate(d2_s)

                corr_coeffs.append(cc_s.mean())
                coherence.append(co_s.mean())
                energy_ret.append(e_ret_s.sum())
                energy_ref.append(e_ref_s.sum())
                mse.append(d2_s.sum())


            ns = 1 - (np.array(mse) / np.array(energy_ref))
            results[group] = xr.Dataset({
                "scales": (("scales"), scales),
                "correlation_coefficient": (("scales"), corr_coeffs),
                "energy_ref": (("scales"), energy_ref),
                "energy_ret": (("scales"), energy_ret),
                "mse": (("scales"), mse),
                "ns": (("scales"), ns),
                "coherence": (("scales"), coherence)
            })

        return results
