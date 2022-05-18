"""
gprof_nn.resolution
===================

This module implements functionality to calculate the effective resolution
of the retrieval using wavelet decomposition.
"""
from concurrent.futures import ProcessPoolExecutor
import logging

import numpy as np
import pywt
from rich.progress import track
import scipy as sp
from scipy.fft import dctn
import xarray as xr


LOGGER = logging.getLogger(__name__)


###############################################################################
# Analysis classes
###############################################################################

class AnalysisBase:
    """
    Base class for spectral analysis classes.

    """
    def merge(self, other):
        """
        Merge results from spectral analysis object with that from another
        object.

        Args:
            other: The object whose results to merge with.
        """
        results_ref = self.results["reference"]
        if len(results_ref.variables) == 0:
            self.results = {
                group: value.copy() for group, value in other.results.items()
            }
        else:
            if len(other.variables) > 0:
                for group, values in other.results.items():
                    self.results[group] += values


class WaveletAnalysis(AnalysisBase):
    """
    Implements spectral analysis of spatial resolution using Haar wavelets.
    """
    def __init__(self, retrieval_groups):
        self.retrieval_groups = retrieval_groups
        self.results = {group: xr.Dataset() for group in self.retrieval_groups}
        self.results["reference"] = xr.Dataset()

    def process(self, window):
        """
        Process a single, square window of observations.

        Results are accumulated in the objects ``results`` attribute.

        Args:
            window: A dict mapping group names to xarray.Datasets, which
                contain the retrieval results.
        """
        reference = window["reference"]

        w_ref = pywt.WaveletPacket2D(
            data=reference.surface_precip_avg.data,
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


class FourierAnalysis(AnalysisBase):
    """
    Implements spectral analysis of spatial resolution using 2D DCT transform.
    """
    def __init__(self, retrieval_groups):
        self.retrieval_groups = retrieval_groups
        self.results = {group: xr.Dataset() for group in self.retrieval_groups}
        self.results["reference"] = xr.Dataset()

    def process(self, window):
        """
        Process a single, square window of observations.

        Results are accumulated in the objects ``results`` attribute.

        Args:
            window: A dict mapping group names to xarray.Datasets, which
                contain the retrieval results.
        """
        reference = window["reference"]

        n = reference.surface_precip_avg.data.shape[0]
        w_ref = dctn(reference.surface_precip_avg.data, norm="ortho")

        for group in window:
            if group == "reference":
                continue

            data = window[group]
            w_ret = dctn(data.surface_precip.data, norm="ortho")

            dims = (f"n_y", f"n_x")

            r = self.results[group]
            varname = f"coeffs_ref_sum"
            if varname not in r.variables:
                r["coeffs_ref_sum"] = (dims, w_ref)
                r["coeffs_ref_sum2"] = (dims, w_ref * w_ref.conjugate())
                r["coeffs_ret_sum"] = (dims, w_ret)
                r["coeffs_ret_sum2"] = (dims, w_ret * w_ret.conjugate())
                r["coeffs_refret_sum"] = (dims, w_ref * w_ret.conjugate())
                r["coeffs_diff_sum"] = (dims, w_ret - w_ref)
                r["coeffs_diff_sum2"] = (
                    dims,
                    (w_ret - w_ref) * (w_ret - w_ref).conjugate()
                )
                r["counts"] = (dims, np.isfinite(w_ret).astype(np.float64))
            else:
                r["coeffs_ref_sum"] += w_ref
                r["coeffs_ref_sum2"] += w_ref * w_ref.conjugate()
                r["coeffs_ret_sum"] += w_ret
                r["coeffs_ret_sum2"] += w_ret * w_ret.conjugate()
                r["coeffs_refret_sum"] += w_ref * w_ret.conjugate()
                r["coeffs_diff_sum"] += w_ret - w_ref
                r["coeffs_diff_sum2"] += (w_ret - w_ref) * (w_ret - w_ref).conjugate()
                r["counts"] += np.isfinite(w_ret).astype(np.float64)


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

            w_ref_s = retrieved[f"coeffs_ref_sum"].data
            w_ref_s2 = retrieved["coeffs_ref_sum2"].data
            w_ret_s = retrieved["coeffs_ret_sum"].data
            w_ret_s2 = retrieved["coeffs_ret_sum2"].data
            w_refret_s = retrieved["coeffs_refret_sum"].data
            w_d_s2 = retrieved["coeffs_diff_sum2"].data
            counts = retrieved["counts"].data

            sigma_ref = w_ref_s2 / counts - (w_ref_s / counts) ** 2
            sigma_ret = w_ret_s2 / counts - (w_ret_s / counts) ** 2
            ref_mean = w_ref_s / counts
            ret_mean = w_ret_s / counts
            refret_mean = w_refret_s / counts
            cc = (
                (refret_mean - ref_mean * ret_mean) /
                (np.sqrt(sigma_ref) * np.sqrt(sigma_ret))
            )
            co = np.abs(w_refret_s) / (np.sqrt(w_ref_s2) * np.sqrt(w_ret_s2))

            n_y = np.arange(sigma_ref.shape[0])
            n_x = np.arange(sigma_ref.shape[1])
            n = np.sqrt(
                n_x.reshape(1, -1) ** 2 +
                n_y.reshape(-1, 1) ** 2
            )
            ext = min(n_y.max(), n_x.max())
            bins = np.arange(min(n_y.max(), n_x.max()) + 1) - 0.5
            counts, _ = np.histogram(n, bins)

            corr_coeffs, _ = np.histogram(n, bins=bins, weights=cc)
            corr_coeffs /= counts
            coherence, _ = np.histogram(n, bins=bins, weights=co)
            coherence /= counts
            energy_ret, _ = np.histogram(n, weights=w_ret_s2, bins=bins)
            energy_ref, _ = np.histogram(n, weights=w_ref_s2, bins=bins)
            se, _ = np.histogram(n, weights=w_d_s2, bins=bins)

            ns = 1 - (se / energy_ref)
            mse = se / counts
            n = 0.5 * (bins[1:] + bins[:-1])
            scales = ext * scale / n

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

###############################################################################
# Resolution calculator
###############################################################################


def process_files(calculator, files):
    """
    Process list of files with a given spectral analysis.

    Args:
        analysis: A spectral analysis object to accumulated the results.
        files: The list of files to process.

    Return:
        The analysis object with the accumulated results.
    """
    calculator.analysis = calculator.analysis_class(calculator.retrieval_groups)
    for filename in files:
        calculator.process_file(filename)
    return calculator


class ResolutionCalculator:
    """
    The ResolutionCalculator class accumulates statistics of reference
    and retrieved wavelet coefficients. It can be used to calculate
    total energy, correlation and NS coefficients of the retrieved fields.
    """
    def __init__(self,
                 reference_group,
                 retrieval_groups,
                 window_size=96,
                 analysis_class=FourierAnalysis,
                 minimum_radar_quality=0.5):
        """
        Args:
            reference_group: Name of the NetCDF4 group containing the
                reference precipitation fields.
            retrieval_groups: The name of the groups containing the retrieval
                results.
            window_size: The size of the windows for which to calculate the
                wavelet coefficients.
            analysis_class: A Spectral transform class defining the type of
                transformation to use to determine the spatial resolution.
            minimum_radar_quality: Lower bound for the radar_quality_index
                of the selected windows.
        """
        self.reference_group = reference_group
        self.retrieval_groups = retrieval_groups
        self.analysis_class = analysis_class
        self.analysis = analysis_class(retrieval_groups)
        self.window_size = window_size
        self.valid_fraction = 1.0
        self.minimum_radar_quality = minimum_radar_quality
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

        sp_ref = reference_scene.surface_precip_avg
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
                valid_frac = (window.surface_precip_avg.data >= 0).mean()
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
                    valid &= rqi_min > self.minimum_radar_quality

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
        self.analysis.process(window)

    def get_statistics(self, scale=5e3):
        """
        Return error statistics for correlation coefficients
        by scale.

        Args:
            scale: Grid size of the underlying grid, which is required
                to calculate the scale corresponding to the wavelet
                coefficients.
        """
        return self.analysis.get_statistics()

    def process_file(self, filename):
        """
        Extract wavelet for all windows in a file.
        """
        for window in self.get_windows(filename):
            self.process_window(window)

    def merge(self, other):
        self.analysis.merge(other.analysis)

    def process_files(self, files, n_processes=4):
        """
        Extract wavelet for all windows in a file.
        """
        from gprof_nn.statistics import _split_files

        pool = ProcessPoolExecutor(max_workers=n_processes)
        file_lists = _split_files(files, 5 * n_processes)

        self.analysis = self.analysis_class(self.retrieval_groups)
        tasks = []
        for files in file_lists:
            tasks.append(pool.submit(process_files, self, files))

        for task in track(
            tasks,
            description="Processing files:"
        ):
            try:
                results = task.result()
                self.merge(results)
            except Exception as exc:
                LOGGER.error("Error during processing of filess: %s", exc)
