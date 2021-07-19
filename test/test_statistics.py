"""
Tests for the gprof_nn.statistics module.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data.bin import BinFile
from gprof_nn.statistics import (StatisticsProcessor,
                                 TrainingDataStatistics,
                                 BinFileStatistics)


def test_training_statistics_gmi(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    test training data file.
    """
    data_path = Path(__file__).parent / "data"
    files = [data_path / "training_data.nc"] * 2


    stats = [TrainingDataStatistics()]
    processor = StatisticsProcessor(sensors.GMI,
                                    files,
                                    stats)
    processor.run(2, tmpdir)
    input_data = xr.open_dataset(files[0])

    results = xr.open_dataset(str(
        tmpdir /
        "training_data_statistics_gmi.nc"
    ))

    # Ensure TB dists match.
    for st in range(1, 19):
        bins = np.linspace(100, 400, 301)
        i_st = (input_data.surface_type == st).data
        tbs = input_data["brightness_temperatures"].data[i_st]
        counts_ref, _ = np.histogram(tbs[:, 0], bins=bins)
        counts = results["brightness_temperatures"][st - 1, 0].data
        assert np.all(np.isclose(counts, 2.0 * counts_ref))

        # Ensure surface_precip dists match.
        bins = np.logspace(-3, np.log10(2e2), 201)
        i_st = (input_data.surface_type == st).data
        x = input_data["surface_precip"].data[i_st]
        counts_ref, _ = np.histogram(x, bins=bins)
        counts = results["surface_precip"][st - 1].data
        assert np.all(np.isclose(counts, 2.0 * counts_ref))

        # Ensure RWC distributions match.
        bins = np.logspace(-4, np.log10(2e1), 201)
        i_st = (input_data.surface_type == st).data[:, :, 90:-90]
        x = input_data["rain_water_content"].data[i_st]
        counts_ref, _ = np.histogram(x, bins=bins)
        counts = results["rain_water_content"][st - 1].data
        assert np.all(np.isclose(counts, 2.0 * counts_ref))

        # Ensure two-meter-temperature distributions match.
        bins = np.linspace(240, 330, 201)
        i_st = (input_data.surface_type == st).data
        x = input_data["two_meter_temperature"].data[i_st]
        counts_ref, _ = np.histogram(x, bins=bins)
        counts = results["two_meter_temperature"][st - 1].data
        assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure surface type distributions match
    bins = np.arange(19) + 0.5
    x = input_data["surface_type"].data
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["surface_type"].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))


def test_training_statistics_mhs(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    test training data file.
    """
    data_path = Path(__file__).parent / "data"
    files = [data_path / "gprof_nn_mhs_era5_5.nc"] * 2


    stats = [TrainingDataStatistics()]
    processor = StatisticsProcessor(sensors.MHS,
                                    files,
                                    stats)
    processor.run(2, tmpdir)
    input_data = xr.open_dataset(files[0])

    results = xr.open_dataset(str(
        tmpdir /
        "training_data_statistics_mhs.nc"
    ))

    # Ensure TB dists match.
    st = 1
    bins = np.linspace(100, 400, 301)

    i_st_0 = ((input_data.source == 0) * (input_data.surface_type == 1)).data
    i_st_0 = i_st_0[:, :, 90:-90]
    tbs = input_data["simulated_brightness_temperatures"].data[i_st_0, 0, 0]
    b = input_data["brightness_temperature_biases"].data[i_st_0, 0]
    counts_ref, _ = np.histogram(tbs - b, bins=bins)
    counts = results["brightness_temperatures"][st - 1, 0, 0].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    st = 2
    sensor = sensors.MHS
    angle_bins = np.zeros(sensor.angles.size + 1)
    angle_bins[1:-1] = 0.5 * (sensor.angles[1:] + sensor.angles[:-1])
    angle_bins[0] = 2.0 * angle_bins[1] - angle_bins[2]
    angle_bins[-1] = 2.0 * angle_bins[-2] - angle_bins[-3]
    l = angle_bins[1]
    u = angle_bins[0]
    i_st_1 = ((input_data.source != 0) *
              (input_data.surface_type == 1) *
              (input_data.earth_incidence_angle[..., 0] >= l) *
              (input_data.earth_incidence_angle[..., 0] < u)).data
    tbs = input_data["brightness_temperatures"].data[i_st_1, 0]
    counts_ref, _ = np.histogram(tbs, bins=bins)
    counts = results["brightness_temperatures"][st - 1, 0, 0].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure surface_precip dists match.
    st = 1
    bins = np.logspace(-3, np.log10(2e2), 201)
    i_st = (input_data.surface_type == 1).data
    x = input_data["surface_precip"].data[i_st, 0]
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["surface_precip"][st - 1, 0].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))


def test_bin_statistics_gmi(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    test training data file.
    """
    data_path = Path(__file__).parent / "data"
    files = [data_path / "gmi" / "gpm_291_55_04.bin"] * 2


    stats = [BinFileStatistics()]
    processor = StatisticsProcessor(sensors.GMI,
                                    files,
                                    stats)
    processor.run(2, tmpdir)
    input_data = BinFile(files[0]).to_xarray_dataset()

    results = xr.open_dataset(str(
        tmpdir /
        "bin_file_statistics_gmi.nc"
    ))

    # Ensure TB dists match.
    st = 4
    bins = np.linspace(100, 400, 301)
    i_st = (input_data.surface_type == st).data
    tbs = input_data["brightness_temperatures"].data[i_st]
    counts_ref, _ = np.histogram(tbs[:, 0], bins=bins)
    counts = results["brightness_temperatures"][st - 1, 0].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure surface_precip dists match.
    bins = np.logspace(-3, np.log10(2e2), 201)
    i_st = (input_data.surface_type == st).data
    x = input_data["surface_precip"].data[i_st]
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["surface_precip"][st - 1].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure RWC distributions match.
    bins = np.logspace(-4, np.log10(2e1), 201)
    i_st = (input_data.surface_type == st).data
    x = input_data["rain_water_content"].data[i_st]
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["rain_water_content"][st - 1].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure two-meter-temperature distributions match.
    bins = np.linspace(240, 330, 201)
    i_st = (input_data.surface_type == st).data
    x = input_data["two_meter_temperature"].data[i_st]
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["two_meter_temperature"][st - 1].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure surface type distributions match
    bins = np.arange(19) + 0.5
    x = input_data["surface_type"].data
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["surface_type"].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))


def test_bin_statistics_mhs(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    test training data file.
    """
    data_path = Path(__file__).parent / "data"
    files = [data_path / "mhs" / "gpm_266_21_02.bin"] * 2


    stats = [BinFileStatistics()]
    processor = StatisticsProcessor(sensors.MHS,
                                    files,
                                    stats)
    processor.run(2, tmpdir)
    input_data = BinFile(files[0]).to_xarray_dataset()

    results = xr.open_dataset(str(
        tmpdir /
        "bin_file_statistics_mhs.nc"
    ))

    # Ensure TB dists match.
    st = 2
    bins = np.linspace(100, 400, 301)
    inds = (input_data.surface_type == st).data
    inds = inds * (input_data.pixel_position == 4).data
    tbs = input_data["brightness_temperatures"].data[inds]
    counts_ref, _ = np.histogram(tbs[:, 0], bins=bins)
    counts = results["brightness_temperatures"][st - 1, 0, 3].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure surface_precip dists match.
    bins = np.logspace(-3, np.log10(2e2), 201)
    x = input_data["surface_precip"].data[inds]
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["surface_precip"][st - 1, 3].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure RWC distributions match.
    bins = np.logspace(-4, np.log10(2e1), 201)
    i_st = (input_data.surface_type == st).data
    x = input_data["rain_water_content"].data[i_st]
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["rain_water_content"][st - 1].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure two-meter-temperature distributions match.
    bins = np.linspace(240, 330, 201)
    i_st = (input_data.surface_type == st).data
    x = input_data["two_meter_temperature"].data[i_st]
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["two_meter_temperature"][st - 1].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure surface type distributions match
    bins = np.arange(19) + 0.5
    x = input_data["surface_type"].data
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["surface_type"].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))
