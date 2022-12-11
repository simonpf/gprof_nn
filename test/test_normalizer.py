"""
Tests for the gprof_nn.normalizer module.
"""
from gprof_nn import sensors
from gprof_nn.normalizer import get_normalizer


def test_get_normalizer():

    normalizer_gmi = get_normalizer(sensors.GMI)
    assert len(normalizer_gmi.stats) == 15 + 9

    normalizer_gmi_2 = get_normalizer(sensors.GMI, [0, 3])
    assert len(normalizer_gmi_2.stats) == 15 + 7
    assert normalizer_gmi.stats[1] == normalizer_gmi_2.stats[0]

    # For cross-track scanners stats should an entry for each
    # channel, earth incidence angle, tcwv and t2m.
    normalizer_mhs = get_normalizer(sensors.MHS)
    assert len(normalizer_mhs.stats) == sensors.MHS.n_chans + 3

    for index, gmi_index in enumerate(sensors.MHS.gmi_channels):
        normalizer_mhs.stats[index] == normalizer_gmi.stats[gmi_index]

    # For conical scanners stats should an entry for each
    # channel, tcwv and t2m.
    normalizer_mhs = get_normalizer(sensors.SSMI)
    assert len(normalizer_mhs.stats) == sensors.SSMI.n_chans + 2
