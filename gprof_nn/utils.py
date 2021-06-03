"""
==============
gprof_nn.utils
==============

Collection of utility attributes and functions.
"""
import xarray as xr

# Rectangular bounding box around continental united states (CONUS).
CONUS = (-130, 20, -60.0, 55)

SURFACE_TYPE_NAMES = [
    "Ocean",
    "Sea-Ice",
    "Vegetation 1",
    "Vegetation 2",
    "Vegetation 3",
    "Vegetation 4",
    "Vegetation 5",
    "Snow 1",
    "Snow 2",
    "Snow 3",
    "Snow 4",
    "Standing Water",
    "Land Coast",
    "Mixed land/ocean o. water",
    "Ocean or water Coast",
    "Sea-ice edge",
    "Mountain Rain",
    "Mountain Snow",
]


def surface_type_to_name(surface_index):
    """
    Transform surface type index to name.

    Args:
        surface_index: The integer surface type code.
    Return:
        String representation of the surface type.
    """
    return SURFACE_TYPE_NAMES[int(surface_index) - 1]


def load_retrieval_results(files_gprof,
                           files_gprof_nn_0d,
                           files_reference):
    """
    Load and combine retrieval results from different algorithm.

    Args:
        files_gprof: List of files containing the results from the legacy GPROF
             algorithm.
        files_gprof_nn_0d: List of files containing the results from the
             GPROF-NN 0D algorithm.
        files_reference: List of files containing the reference results.

    Returns:
        'xarray.Dataset' containing the reference data combined with the
        retrieval results.
    """
    results_gprof = []
    results_gprof_nn_0d = []
    reference = []

    for f in files_gprof:
        results_gprof.append(xr.load_dataset(f))
    for f in files_gprof_nn_0d:
        results_gprof_nn_0d.append(xr.load_dataset(f))
    for f in files_reference:
        reference.append(xr.load_dataset(f))

    results_gprof = xr.concat(results_gprof, "samples")
    results_gprof_nn_0d = xr.concat(results_gprof_nn_0d, "samples")
    reference = xr.concat(reference, "samples")

    for v in reference.variables:
        if not (v in results_gprof.variables
                and v in results_gprof_nn_0d.variables):
            continue
        data = reference[v].data
        reference[v + "_gprof"] = results_gprof[v]
        reference[v + "_gprof_nn_0d"] = results_gprof_nn_0d[v]

    return reference
