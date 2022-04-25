"""
Results are stored sparately for different years following the DB year.
"""
import logging 

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pickle
from pathlib import Path

from gprof_nn.validation import GPROFResults, GPROFNN1DResults, GPROFNN3DResults
from gprof_nn.data.sim import apply_orographic_enhancement
from concurrent.futures import ProcessPoolExecutor


LOGGER = logging.Logger(__file__)

np.seterr(all="ignore")


DATA_PATH = Path("/home/simonpf/data/gprof_nn/validation/gmi")


mrms_files = list((DATA_PATH / "mrms").glob("**/*.nc"))
granules = [int(f.name.split("_")[-1].split(".")[0]) for f in mrms_files]
print(f"Found {len(granules)} granules.")

gprof_nn_1d = GPROFNN1DResults(DATA_PATH / "gprof_nn_1d")
gprof_nn_3d = GPROFNN3DResults(DATA_PATH / "gprof_nn_3d_tiled")
gprof_v7 = GPROFResults(DATA_PATH / "gprof")


def extract_stats(filename, groups, rqi_threshold, mask_group="gprof_v7", reference_group="reference"):
    """
    Helper function to extract dictionary of retrieval stats from a single
    file.
    """
    ref = xr.load_dataset(filename, group="reference")
    time_ref, _ = xr.broadcast(ref.time, ref.surface_precip)
    time_ref = time_ref.data

    try:
        ref = xr.load_dataset(filename, group=reference_group)
    except OSError:
        LOGGER.error("File '%s' has no '%s' group.", filename, reference_group)
        return None
    sp_ref = ref.surface_precip.data
    if "surface_precip_avg" in ref:
        sp_ref_avg = ref.surface_precip_avg.data
    else:
        sp_ref_avg = ref.surface_precip.data
    if "radar_quality_index" in ref.variables:
        rqi = ref.radar_quality_index.data
    else:
        rqi = None

    if "mask" in ref.variables:
        mask = ref.mask.data
    else:
        mask = None

    ref = xr.load_dataset(filename, group="reference")

    lats = ref.latitude.data
    lons = ref.longitude.data

    results = {}

    try:
        gprof = xr.load_dataset(filename, group=mask_group)
        gprof_mask = gprof.surface_precip.data >= 0
        if rqi is None and "radar_quality_index" in gprof.variables:
            rqi = gprof.radar_quality_index.data

    except OSError:
        LOGGER.error("File '%s' has no '%s' group.", filename, mask_group)
        return None

    try:
        surface_types = xr.load_dataset(filename, group="gprof_nn_3d").surface_type.data
    except OSError:
        LOGGER.error("File '%s' has no '%s' group.", filename, "gprof_nn_3d")
        return None

    for group in groups:
        try:
            data = xr.load_dataset(filename, group=group)
            #if group == "simulator":
                #data["airmass_type"] = (("along_track", "across_track"), at)
                #apply_orographic_enhancement(data)
            sp = data.surface_precip.data

            valid = (sp_ref >= 0.0) * (sp >= 0.0) * (gprof_mask)
            if rqi is not None:
                valid = valid * (rqi >= rqi_threshold)
            samples = xr.Dataset(
                {
                    "surface_precip": (("samples",), sp[valid]),
                    "surface_precip_avg": (("samples",), sp_ref_avg[valid]),
                    "surface_precip_ref": (("samples",), sp_ref[valid]),
                    "latitude": (("samples",), lats[valid]),
                    "longitude": (("samples",), lons[valid]),
                    "time": (("samples",), time_ref[valid]),
                }
            )
            if "pop" in data.variables:
                samples["pop"] = (("samples",), data.pop.data[valid])
            if rqi is not None:
                samples["rqi"] = (("samples",), rqi[valid])
            if mask is not None:
                samples["mask"] = (("samples",), mask[valid])
            samples["surface_type"] = (("samples"), surface_types[valid])
            #if "airmass_type" in gprof.variables:
            #    samples["airmass_type"] = (("samples"), gprof.airmass_type.data[valid])
            if "range" in ref.variables:
                ranges, _ = xr.broadcast(ref.range, ref.surface_precip)
                samples["range"] = (("samples"), ranges.data[valid])
            results[group] = samples
        except OSError as e:
            LOGGER.error(
                    "The following error occurred during processing of "
                    " file '%s': \n %s",  filename, e
            )
            return None
    return results


def extract_precip_stats(path, groups, rqi_threshold, mask_group="gprof_v7", reference_group="reference"):
    """
    Collects surface precip statistics for the products to validate

    Args:
        path: Directory containing the collected validation resullts.
        groups: List of the names of the groups containing the retrieval results.
        rqi_threshold: A threshold for the minimum Radar Quality Index (RQI) of the radar
            measurements to be included in the results.

    Return:
        A dict mapping the group names of the retrieval products to datasets containing the
        retrieved precipitation 'surface_precip' and the ref precipitation as
        'surface_precip_ref'.
    """
    files = list(Path(path).glob("*.nc"))
    results = {}
    pool = ProcessPoolExecutor(max_workers=8)
    tasks = []
    for filename in files:
        tasks.append(pool.submit(extract_stats, filename, groups, rqi_threshold, mask_group=mask_group, reference_group=reference_group))

    for filename, task in zip(files, tasks):
        result = task.result()
        if result is not None:
            for k, stats in result.items():
                results.setdefault(k, []).append(stats)

    for k in results:
        results[k] = xr.concat(results[k], "samples")
    pool.shutdown()
    return results

#DATA_PATH = Path("/home/simonpf/data/gprof_nn/validation/gmi")
#groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v5", "gprof_v7", "combined", "simulator"]
#results_db = extract_precip_stats(DATA_PATH / "../gmi/results_db", groups, 0.8, reference_group="simulator", mask_group="reference")
#pickle.dump(results_db, open(DATA_PATH / "../gmi/results_db_ref.pckl", "wb"))

#
#groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v5", "gprof_v7", "combined"]
#results_db = extract_precip_stats(DATA_PATH / "../gmi/results_db_1", groups, 0.8, mask_group="combined")
#pickle.dump(results_db, open(DATA_PATH / "../gmi/results_db_1.pckl", "wb"))
##
#groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v5", "gprof_v7", "combined"]
#results_db = extract_precip_stats(DATA_PATH / "../gmi/results_db_2", groups, 0.8, mask_group="combined")
#pickle.dump(results_db, open(DATA_PATH / "../gmi/results_db_2.pckl", "wb"))
#groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v5", "gprof_v7", "simulator", "combined"]
#results_db = extract_precip_stats(DATA_PATH / "results_db", groups, 0.9)
#pickle.dump(results_db, open(DATA_PATH / "results_db.pckl", "wb"))
#results_db = extract_precip_stats(DATA_PATH / "results_db_1", groups, 0.9)
#pickle.dump(results_db, open(DATA_PATH / "results_db_1.pckl", "wb"))
#groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v5", "gprof_v7"]
#results_db = extract_precip_stats(DATA_PATH / "results_db_2", groups, 0.9)
#pickle.dump(results_db, open(DATA_PATH / "results_db_2.pckl", "wb"))
##results_db = extract_precip_stats(DATA_PATH / "../gmi/results_db_1", groups, 0.9)
#pickle.dump(results_db, open("results_db_1.pckl", "wb"))

#DATA_PATH = Path("/home/simonpf/data/gprof_nn/validation/kwaj")
#
#
#groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v5", "gprof_v7", "combined", "simulator"]
#results_db = extract_precip_stats(DATA_PATH / "results_db", groups, 0.8, reference_group="simulator", mask_group="reference")
#pickle.dump(results_db, open(DATA_PATH / "results_db_ref.pckl", "wb"))

#groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v5", "gprof_v7", "simulator", "combined"]
#results_db = extract_precip_stats(DATA_PATH / "results_db", groups, 0.8, mask_group="combined")
#pickle.dump(results_db, open(DATA_PATH / "../kwaj/results_db.pckl", "wb"))
#
#groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v5", "gprof_v7", "combined"]
#results_db = extract_precip_stats(DATA_PATH / "results_db_1", groups, 0.8, mask_group="combined")
#pickle.dump(results_db, open(DATA_PATH / "../kwaj/results_db_1.pckl", "wb"))
##
#results_db = extract_precip_stats(DATA_PATH / "results_db_2", groups, 0.8, mask_group="combined")
#pickle.dump(results_db, open(DATA_PATH / "../kwaj/results_db_2.pckl", "wb"))
#
#DATA_PATH = Path("/home/simonpf/data/gprof_nn/validation/tmipo")
##
#groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v7"]
#results_db = extract_precip_stats(DATA_PATH / "results", groups, 0.8)
#pickle.dump(results_db, open(DATA_PATH / "results.pckl", "wb"))
#
#groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v5", "gprof_v7", "combined"]
#results_db = extract_precip_stats(DATA_PATH / "results_db_1", groups, 0.8, mask_group="combined")
#pickle.dump(results_db, open(DATA_PATH / "../kwaj/results_db_1.pckl", "wb"))
##
#results_db = extract_precip_stats(DATA_PATH / "results_db_2", groups, 0.8, mask_group="combined")
#pickle.dump(results_db, open(DATA_PATH / "../kwaj/results_db_2.pckl", "wb"))
#
DATA_PATH = Path("/home/simonpf/data/gprof_nn/validation/ssmis")
#
groups = ["gprof_nn_1d", "gprof_nn_3d", "gprof_v7"]
results_db = extract_precip_stats(DATA_PATH / "results", groups, 0.8)
pickle.dump(results_db, open(DATA_PATH / "results.pckl", "wb"))
