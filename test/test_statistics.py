"""
Tests for the gprof_nn.statistics module.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn import sensors
from gprof_nn.data.bin import BinFile
from gprof_nn.data import get_test_data_path
from gprof_nn.definitions import ALL_TARGETS
from gprof_nn.data.preprocessor import PreprocessorFile
from gprof_nn.data.retrieval import RetrievalFile
from gprof_nn.data.training_data import (GPROF_NN_1D_Dataset,
                                         decompress_and_load)
from gprof_nn.data.combined import GPMCMBFile
from gprof_nn.statistics import (StatisticsProcessor,
                                 LatitudeDistribution,
                                 TrainingDataStatistics,
                                 BinFileStatistics,
                                 ObservationStatistics,
                                 GlobalDistribution,
                                 ZonalDistribution,
                                 RetrievalStatistics,
                                 GPMCMBStatistics)


DATA_PATH = get_test_data_path()


def test_training_statistics_gmi(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    GMI training data file.
    """
    files = [DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"] * 2


    stats = [TrainingDataStatistics(kind="1d"),
             ZonalDistribution(monthly=False),
             GlobalDistribution()]
    processor = StatisticsProcessor(sensors.GMI,
                                    files,
                                    stats)
    processor.run(2, tmpdir)
    input_data = GPROF_NN_1D_Dataset(files[0],
                                     normalize=False,
                                     shuffle=False,
                                     targets=ALL_TARGETS)
    input_data = input_data.to_xarray_dataset()

    results = xr.open_dataset(str(
        tmpdir /
        "training_data_statistics_gmi.nc"
    ))

    # Ensure TB dists match.
    for st in range(1, 19):
        bins = np.linspace(0, 400, 401)
        i_st = ((input_data.surface_type == st) *
                (input_data.surface_precip >= 0)).data

        tbs = input_data["brightness_temperatures"].data[i_st]
        counts_ref, _ = np.histogram(tbs[:, 0], bins=bins)
        counts = results["brightness_temperatures"][st - 1, 0].data

        assert np.all(np.isclose(counts, 2.0 * counts_ref))

        tcwv = input_data["total_column_water_vapor"].data[i_st]
        bins_tcwv = np.linspace(-0.5, 99.5, 101)
        counts_ref, _, _ = np.histogram2d(
            tcwv, tbs[:, 0],
            bins=(bins_tcwv, bins)
        )
        counts = results["brightness_temperatures_tcwv"][st - 1, 0].data
        assert np.all(np.isclose(counts, 2.0 * counts_ref))

        # Ensure surface_precip dists match.
        bins = np.logspace(-3, np.log10(2e2), 201)
        x = input_data["surface_precip"].data[i_st]
        counts_ref, _ = np.histogram(x, bins=bins)
        counts = results["surface_precip"][st - 1].data
        assert np.all(np.isclose(counts, 2.0 * counts_ref))

        # Ensure RWC distributions match.
        bins = np.logspace(-4, np.log10(2e1), 201)
        x = input_data["rain_water_content"].data[i_st]
        counts_ref, _ = np.histogram(x, bins=bins)
        counts = results["rain_water_content"][st - 1].data
        assert np.all(np.isclose(counts, 2.0 * counts_ref))

        # Ensure two-meter-temperature distributions match.
        bins = np.linspace(239.5, 339.5, 101)
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

    #
    # Zonal distributions
    #

    input_data = decompress_and_load(files[0])
    results = xr.open_dataset(str(tmpdir / "zonal_distribution_gmi.nc"))
    lat_bins = np.linspace(-90, 90, 181)
    sp_bins = np.logspace(-2, 2.5, 201)
    bins = (lat_bins, sp_bins)
    sp = input_data["surface_precip"].data
    lats = input_data["latitude"].data
    valid = sp >= 0.0
    sp = sp[valid]
    lats = lats[valid]
    cs_ref, _, _ = np.histogram2d(lats, sp, bins=bins)
    cs = results["surface_precip_mean"].data
    assert np.all(np.isclose(2.0 * cs_ref, cs))

    #
    # Global distributions
    #

    input_data = decompress_and_load(files[0])
    results = xr.open_dataset(str(tmpdir / "global_distribution_gmi.nc"))
    lat_bins = np.arange(-90, 90 + 1e-3, 5)
    lon_bins = np.arange(-180, 180 + 1e-3, 5)
    sp_bins = np.logspace(-2, 2.5, 201)
    sp = input_data["surface_precip"].data
    lons = input_data["longitude"].data
    lats = input_data["latitude"].data
    valid = sp >= 0.0
    sp = sp[valid]
    lats = lats[valid]
    lons = lons[valid]
    bins = (lat_bins, lon_bins, sp_bins)
    vals = np.stack([lats, lons, sp], axis=-1)
    cs_ref, _ = np.histogramdd(vals, bins=bins)
    cs = results["surface_precip_mean"].data
    assert np.all(np.isclose(2.0 * cs_ref, cs))


def test_training_statistics_mhs(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    MHS training data file.
    """
    files = [DATA_PATH / "mhs" / "gprof_nn_mhs_era5.nc"] * 2


    stats = [TrainingDataStatistics(kind="1D"),
             GlobalDistribution(),
             ZonalDistribution()]
    processor = StatisticsProcessor(sensors.MHS,
                                    files,
                                    stats)
    processor.run(2, tmpdir)
    input_data = GPROF_NN_1D_Dataset(files[0],
                                     normalize=False,
                                     shuffle=False,
                                     targets=ALL_TARGETS)
    input_data = input_data.to_xarray_dataset()

    results = xr.open_dataset(str(
        tmpdir /
        "training_data_statistics_mhs.nc"
    ))

    # Ensure total column water vapor distributions match.
    st = 1
    bins = np.linspace(-0.5, 99.5, 101)
    i_st = (input_data.surface_type == 1).data

    x = input_data["total_column_water_vapor"].data[i_st]
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["total_column_water_vapor"][st - 1].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))


def test_bin_statistics_gmi(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    GMI bin files.
    """
    files = [DATA_PATH / "gmi" / "bin" / "gpm_269_00_16.bin"] * 2


    stats = [BinFileStatistics(),
             ZonalDistribution(),
             GlobalDistribution()]
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
    bins = np.linspace(0, 400, 401)
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
    bins = np.linspace(239.5, 339.5, 101)
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

    # Ensure conditional means match
    st = input_data.surface_type.data[0]
    i_t2m = int(np.round(input_data.two_meter_temperature.data[0] - 240))
    mean_sp = results["surface_precip_mean_t2m"][st - 1, i_t2m]
    mean_sp_ref = input_data.surface_precip.data.mean()
    assert np.isclose(mean_sp_ref, mean_sp)

    # Ensure conditional means match
    st = input_data.surface_type.data[0]
    i_tcwv = int(np.round(input_data.total_column_water_vapor.data[0]))
    mean_sp = results["surface_precip_mean_tcwv"][st - 1, i_tcwv]
    mean_sp_ref = input_data.surface_precip.data.mean()
    assert np.isclose(mean_sp_ref, mean_sp)


def test_bin_statistics_mhs_sea_ice(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    MHS bin file for a sea ice surface.
    """
    files = [DATA_PATH / "mhs" / "bin" / "gpm_271_20_16.bin"] * 2


    stats = [BinFileStatistics(),
             ZonalDistribution(),
             GlobalDistribution()]
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
    bins = np.linspace(0, 400, 401)
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
    bins = np.linspace(239.5, 339.5, 101)
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


def test_bin_statistics_mhs_ocean(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    MHS bin file for a land surface.
    """
    files = [DATA_PATH / "mhs" / "gpm_289_52_04.bin"] * 2


    stats = [BinFileStatistics(),
             ZonalDistribution(),
             GlobalDistribution()]
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
    st = 1
    bins = np.linspace(0, 400, 401)
    inds = (input_data.surface_type == st).data
    tbs = input_data["brightness_temperatures"].data[inds]
    counts_ref, _ = np.histogram(tbs[:, 0, 0], bins=bins)
    counts = results["brightness_temperatures"][st - 1, 0, 0].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure surface_precip dists match.
    bins = np.logspace(-3, np.log10(2e2), 201)
    x = input_data["surface_precip"].data[inds, 0]
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["surface_precip"][st - 1, 0].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure RWC distributions match.
    bins = np.logspace(-4, np.log10(2e1), 201)
    i_st = (input_data.surface_type == st).data
    x = input_data["rain_water_content"].data[i_st]
    counts_ref, _ = np.histogram(x, bins=bins)
    counts = results["rain_water_content"][st - 1].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    # Ensure two-meter-temperature distributions match.
    bins = np.linspace(239.5, 339.5, 101)
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


def test_observation_statistics_gmi(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    MHS bin file for an ocean surface.
    """
    files = [DATA_PATH / "gmi" / "pp" / "GMIERA5_190101_027510.pp"] * 2

    stats = [ObservationStatistics(conditional=1),
             ZonalDistribution(),
             GlobalDistribution()]
    processor = StatisticsProcessor(sensors.GMI,
                                    files,
                                    stats)
    processor.run(2, tmpdir)
    input_data = PreprocessorFile(files[0]).to_xarray_dataset()

    results = xr.open_dataset(str(
        tmpdir /
        "observation_statistics_gmi.nc"
    ))

    # Ensure TB dists match.
    st = 1
    bins = np.linspace(0, 400, 401)
    inds = (input_data.surface_type == st).data
    tbs = input_data["brightness_temperatures"].data[inds]
    counts_ref, _ = np.histogram(tbs[:, 0], bins=bins)
    counts = results["brightness_temperatures"][st - 1, 0].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    bins_tcwv = np.linspace(-0.5, 99.5, 101)
    inds = (input_data.surface_type == st).data
    tcwv = input_data["total_column_water_vapor"].data[inds]
    counts_ref, _, _ = np.histogram2d(
        tcwv, tbs[:, 0], bins=(bins_tcwv, bins)
    )
    counts = results["brightness_temperatures_tcwv"][st - 1, 0].data
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


def test_observation_statistics_mhs(tmpdir):
    """
    Ensure that TrainingDataStatistics class reproduces statistic of
    MHS bin file for an ocean surface.
    """
    files = [DATA_PATH / "mhs" / "pp" / "MHS.pp"] * 2

    stats = [ObservationStatistics(),
             ZonalDistribution(),
             GlobalDistribution()]
    processor = StatisticsProcessor(sensors.MHS,
                                    files,
                                    stats)
    processor.run(2, tmpdir)
    input_data = PreprocessorFile(files[0]).to_xarray_dataset()

    results = xr.open_dataset(str(
        tmpdir /
        "observation_statistics_mhs.nc"
    ))

    # Ensure TB dists match.
    bins = np.linspace(0, 400, 401)
    st = 1
    sensor = sensors.MHS
    angle_bins = np.zeros(sensor.angles.size + 1)
    angle_bins[1:-1] = 0.5 * (sensor.angles[1:] + sensor.angles[:-1])
    angle_bins[0] = 2.0 * angle_bins[1] - angle_bins[2]
    angle_bins[-1] = 2.0 * angle_bins[-2] - angle_bins[-3]
    lower = angle_bins[3]
    upper = angle_bins[2]
    eia = np.abs(input_data.earth_incidence_angle.data)
    inds = ((input_data.surface_type.data == st) *
            (eia >= lower) *
            (eia < upper))
    tbs = input_data["brightness_temperatures"].data[inds]
    counts_ref, _ = np.histogram(tbs[:, 0], bins=bins)
    counts = results["brightness_temperatures"][st - 1, 0, 2].data
    assert np.all(np.isclose(counts, 2.0 * counts_ref))

    bins_tcwv = np.linspace(-0.5, 99.5, 101)
    tcwv = input_data["total_column_water_vapor"].data[inds]
    counts_ref, _, _ = np.histogram2d(
        tcwv, tbs[:, 0], bins=(bins_tcwv, bins)
    )
    counts = results["brightness_temperatures_tcwv"][st - 1, 0, 2].data
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


def test_retrieval_statistics_gmi(tmpdir):
    """
    Ensure that calculated means of retrieval results statistics match
    directly calculated ones.
    """
    source_file = DATA_PATH / "gmi" / "retrieval" / "GMIERA5_190101_027510.bin"
    # This retrieval file contains profiles so it has to be converted
    # to a netcdf file first.
    data = RetrievalFile(source_file, has_profiles=True).to_xarray_dataset()
    data.to_netcdf(tmpdir / "input.nc")

    files = [tmpdir / "input.nc"] * 2
    stats = [RetrievalStatistics()]
    processor = StatisticsProcessor(sensors.GMI,
                                    files,
                                    stats)
    processor.run(2, str(tmpdir))

    results = xr.load_dataset(str(tmpdir / "retrieval_statistics_gmi.nc"))

    mean_sp = results["surface_precip_mean_t2m"][0, 33]
    st = data.surface_type.data
    l_t2m, r_t2m = np.linspace(239.5, 339.5, 101)[33:35]
    indices = ((data.surface_type.data == 1) *
               (data.two_meter_temperature.data >= l_t2m) *
               (data.two_meter_temperature.data < r_t2m) *
               (data.surface_precip.data > -999))
    mean_sp_ref = data.surface_precip.data[indices].mean()

    assert np.isclose(mean_sp_ref, mean_sp)

    mean_sp = results["surface_precip_mean_tcwv"][0, 10]
    st = data.surface_type.data
    l_tcwv, r_tcwv = np.linspace(-0.5, 99.5, 101)[10:12]
    indices = ((data.surface_type.data == 1) *
               (data.total_column_water_vapor.data >= l_tcwv) *
               (data.total_column_water_vapor.data < r_tcwv) *
               (data.surface_precip.data > -999))
    mean_sp_ref = data.surface_precip.data[indices].mean()

    assert np.isclose(mean_sp_ref, mean_sp)


def test_gpm_cmb_statistics(tmpdir):
    input_file = (
        DATA_PATH / "cmb" /
        "2B.GPM.DPRGMI.CORRA2018.20210829-S205206-E222439.042628.V06A.HDF5"
    )
    files = [input_file] * 2
    stats = [GPMCMBStatistics()]
    processor = StatisticsProcessor(sensors.GMI,
                                    files,
                                    stats)
    processor.run(2, str(tmpdir))
    results = xr.load_dataset(str(tmpdir / "gpm_combined_statistics.nc"))

    input_data = GPMCMBFile(input_file).to_xarray_dataset(smooth=True)
    surface_precip = input_data.surface_precip.data
    lats = input_data.latitude.data
    latitude_bins = np.arange(-90, 90 + 1e-3, 5)
    sp_bins = np.logspace(-2, 2.5, 201)
    bins = (latitude_bins, sp_bins)

    cs_ref, _, _ = np.histogram2d(
        lats.ravel(),
        surface_precip.ravel(),
        bins=bins
    )
    cs = results["surface_precip"].data.sum(axis=1)

    assert np.all(np.isclose(cs, 2.0 * cs_ref))
