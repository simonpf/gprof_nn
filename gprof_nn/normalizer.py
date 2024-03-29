"""
gprof_nn.normalizer
===================

This module provides access to normalizer objects, whose task is
to normalize the retrieval inputs.
"""
from copy import deepcopy
from pathlib import Path
from quantnn.normalizer import Normalizer

from gprof_nn import sensors


NORMALIZER = Normalizer.load(Path(__file__).parent / "files" / "normalizer_gmi.pckl")


def get_normalizer(sensor, drop_inputs=None, nedts=None):
    """
    Return default normalizer for a given sensor.

    This function derives a normalizer for a given sensor from the
    normalizer used for GMI by extracting the statistics from the
    corresponding GMI channels.

    Args:
        sensor: Sensor object representing the sensor for which
            return a normalizer.
        drop_inputs

    Return:
        A new normalizer object for the given sensor.
    """
    if sensor == sensors.GMI:
        if drop_inputs is None:
            drop_inputs = []
        normalizer = deepcopy(NORMALIZER)
        new_stats = {}
        ch_ind = 0
        for index in NORMALIZER.stats:
            if index not in drop_inputs:
                stats = normalizer.stats[index]
                if nedts is not None and ch_ind < len(nedts):
                    sigma = nedts[ch_ind]
                    stats = (stats[0] - 3 * sigma, stats[1] + 3 * sigma)
                new_stats[ch_ind] = stats
                ch_ind += 1
        # Must ensure that normalizer indices are continuous.
        assert index == len(normalizer.stats) - 1
        normalizer.stats = new_stats
        return normalizer

    new_stats = {}
    for index, gmi_index in enumerate(sensor.gmi_channels):
        new_stats[index] = NORMALIZER.stats[gmi_index]

    index = sensor.n_chans
    if isinstance(sensor, sensors.CrossTrackScanner):
        eias = sensor.viewing_geometry.get_earth_incidence_angles()
        eia_min = eias.min()
        eia_max = eias.max()
        new_stats[index] = (eia_min, eia_max)
        index += 1

    for i in range(len(NORMALIZER.stats) - 15):
        new_stats[index + i] = NORMALIZER.stats[15 + i]

    if drop_inputs is None:
        drop_inputs = []
    stats_dropped = {}
    ch_ind = 0
    for index in new_stats:
        if index not in drop_inputs:
            stats = new_stats[index]
            if nedts is not None and ch_ind < len(nedts):
                sigma = nedts[ch_ind]
                stats = (stats[0] - 3 * sigma, stats[1] + 3 * sigma)
            stats_dropped[ch_ind] = stats
            ch_ind += 1

    normalizer = deepcopy(NORMALIZER)
    normalizer.stats = stats_dropped

    return normalizer
