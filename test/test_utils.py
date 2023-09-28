"""
Tests for the ``gprof_nn.utils`` module.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from gprof_nn.augmentation import get_transformation_coordinates
from gprof_nn.data import get_test_data_path
from gprof_nn.sensors import GMI_VIEWING_GEOMETRY
from gprof_nn.utils import (
    apply_limits,
    get_mask,
    calculate_interpolation_weights,
    interpolate,
    calculate_tiles_and_cuts,
)
from gprof_nn.data.utils import (
    load_variable,
    decompress_scene,
    remap_scene,
    upsample_scans,
    save_scene,
    extract_scenes,
    write_training_samples_1d,
    write_training_samples_3d,
)
from gprof_nn.data.training_data import decompress_and_load

from conftest import (
    training_files_3d_gmi,
    sim_collocations_gmi
)


DATA_PATH = get_test_data_path()


def test_apply_limits():
    """
    Ensure that upper and lower bounds are applied correctly.
    """
    x = np.random.normal(size=(10, 10))

    x_l = apply_limits(x, 0.0, None)
    x_l = x_l[np.isfinite(x_l)]
    assert np.all(x_l >= 0.0)

    x_r = apply_limits(x, None, 0.0)
    x_r = x_r[np.isfinite(x_r)]
    assert np.all(x_r <= 0.0)

    x = apply_limits(x, 0.0, 0.0)
    x = x[np.isfinite(x)]
    assert x.size == 0


def test_get_mask():
    """
    Ensure that values extracted with mask are within given limits.
    """
    x = np.random.normal(size=(10, 10))

    mask = get_mask(x, 0.0, None)
    x_l = x[mask]
    assert np.all(x_l >= 0.0)

    mask = get_mask(x, None, 0.0)
    x_r = x[mask]
    assert np.all(x_r <= 0.0)

    mask = get_mask(x, 0.0, 0.0)
    x = x[mask]
    assert x.size == 0


def test_calculate_interpolation_weights():
    """
    Ensure that calculating interpolation weights for the grid values
    itself produces a diagonal matrix of weights.

    Also ensure that weights always sum to one across last dimension.
    """
    grid = np.arange(0, 11)
    weights = calculate_interpolation_weights(grid, grid)

    assert np.all(np.isclose(weights.diagonal(), 1.0))

    values = np.random.uniform(0, 10, size=(10, 10))
    weights = calculate_interpolation_weights(values, grid)
    assert np.all(np.isclose(np.sum(weights, 2), 1.0))


def test_interpolation():
    """
    Ensure that calculating interpolation weights for the grid values
    itself produces a diagonal matrix of weights.
    """
    grid = np.arange(0, 11)
    weights = calculate_interpolation_weights(grid, grid)
    y = interpolate(np.repeat(grid.reshape(1, -1), 11, 0), weights)
    assert np.all(np.isclose(grid, y))

    values = np.random.uniform(0, 10, size=(10))
    weights = calculate_interpolation_weights(values, grid)
    y = interpolate(np.repeat(grid.reshape(1, -1), 10, 0), weights)
    assert np.all(np.isclose(y, values))


def test_load_variable():
    """
    Ensure that loading a variable correctly replaces invalid value and
    conserves shape when used without mask.

    Also ensure that masking works.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc.gz"
    dataset = decompress_and_load(input_file)
    sp = load_variable(dataset, "surface_precip")

    expected_shape = (dataset.samples.size,
                      dataset.scans.size,
                      dataset.pixels.size)
    assert sp.shape == expected_shape

    sp = sp[np.isfinite(sp)]
    assert np.all((sp >= 0.0) * (sp < 500))

    sp = load_variable(dataset, "surface_precip")
    mask = sp > 10
    sp = load_variable(dataset, "surface_precip", mask)
    sp = sp[np.isfinite(sp)]
    assert np.all((sp > 10.0) * (sp < 500))


def test_decompress_scene(training_files_3d_gmi):
    """
    Ensure that loading a variable correctly replaces invalid value and
    conserves shape when used without mask.

    Also ensure that masking works.
    """
    input_file = training_files_3d_gmi[0]
    scene = xr.load_dataset(input_file) #decompress_and_load(input_file)[{"samples": 1}]

    scene_d = decompress_scene(
        scene,
        [
            "surface_precip",
            "rain_water_content",
            "rain_water_path",
            "surface_type"
        ])

    assert "pixels" in scene_d.rain_water_content.dims

    # Over ocean all pixels where IWP is defines should also
    # have a valid surface precip value.
    rwp = scene_d.rain_water_path.data
    sp = scene_d.surface_precip.data
    st = scene_d.surface_type
    inds = (st == 1) * (rwp >= 0.0)
    assert np.all(sp[inds] >= 0.0)


def test_calculate_tiles_and_cuts():
    """
    Test calculation of tiles and cuts for slicing of inputs.
    """
    array = np.random.rand(1234, 128)
    tiles, cuts = calculate_tiles_and_cuts(array.shape[0], 256, 8)
    arrays_raw = [array[tile] for tile in tiles]
    assert arrays_raw[-1].shape[0] == 256
    arrays = [arr[cut] for arr, cut in zip(arrays_raw, cuts)]
    array_rec = np.concatenate(arrays, 0)
    assert array_rec.shape == array.shape
    assert np.all(np.isclose(array, array_rec))

    array = np.random.rand(111, 128)
    tiles, cuts = calculate_tiles_and_cuts(array.shape[0], 256, 8)
    arrays_raw = [array[tile] for tile in tiles]
    assert arrays_raw[-1].shape[0] == 111
    arrays = [arr[cut] for arr, cut in zip(arrays_raw, cuts)]
    array_rec = np.concatenate(arrays, 0)
    assert array_rec.shape == array.shape
    assert np.all(np.isclose(array, array_rec))

def test_upsample_scans():

    array = np.arange(10).astype(np.float32)
    array_3 = upsample_scans(array)

    assert array_3.size == 28
    assert np.all(np.isclose(array_3, np.linspace(0, 9, 28)))


def test_save_scene(
        tmp_path,
        sim_collocations_gmi
):
    data = sim_collocations_gmi

    save_scene(sim_collocations_gmi, tmp_path / "scene.nc")
    data_loaded = xr.load_dataset(tmp_path / "scene.nc")

    # TB differences should be small and invalid values should
    # be the same
    for var in [
            "brightness_temperatures",
            "simulated_brightness_temperatures",
            "brightness_temperature_biases"
    ]:
        tbs = data[var].data
        tbs_l = data_loaded[var].data
        mask = np.isnan(tbs) + (tbs < -150)
        mask_l = np.isnan(tbs_l)
        assert np.all(mask == mask_l)
        delta = tbs[~mask] - tbs_l[~mask]
        assert np.abs(delta).max() <= 0.01
        assert np.abs(delta).max() > 0.0

    tbs = data.brightness_temperatures.data
    tbs_l = data.brightness_temperatures.data
    mask = np.isnan(tbs)
    mask_l = np.isnan(tbs_l)
    assert np.all(mask == mask_l)
    delta = tbs[~mask] - tbs_l[~mask]
    assert np.abs(delta).max() <= 0.01

    # Ensure that compression of ancillary data is
    # lossless.
    for var in [
            "surface_type",
            "mountain_type",
            "airlifting_index",
            "mountain_type",
            "land_fraction",
            "ice_fraction",
    ]:
        print("TARGET :: ", var)
        trgt = data[var].data
        trgt_l = data_loaded[var].data

        valid = trgt >= 0
        valid_l = trgt_l >= 0
        assert np.all(valid == valid_l)

        err = trgt[valid] - trgt_l[valid]
        assert np.all(err == 0.0)

    # Ensure that compression of ancillary data and targets is
    # lossless.
    for var in [
            "total_column_water_vapor",
            "two_meter_temperature",
            "surface_precip",
            "ice_water_path",
            "rain_water_path",
            "cloud_water_path",
            "rain_water_content",
            "cloud_water_content",
            "snow_water_content",
            "latent_heat"
    ]:
        trgt = data[var].data
        trgt_l = data_loaded[var].data

        valid = np.isfinite(trgt)
        valid_l = np.isfinite(trgt_l)
        assert np.all(valid == valid_l)

        err = trgt[valid] - trgt_l[valid]
        assert np.all(np.abs(err) < 1e-6)


def test_extract_scenes():
    """
    Ensure that extracting scenes produces the expected amount
    of valid pixels in the output.
    """
    brightness_temperatures = np.random.rand(100, 100, 12)
    surface_precip = np.random.rand(100, 100)
    surface_precip[surface_precip < 0.5] = np.nan
    data = xr.Dataset({
        "brightness_temperatures": (
            ("scans", "pixels", "channels"), brightness_temperatures
        ),
        "surface_precip": (
            ("scans", "pixels"), surface_precip
        )
    })

    scenes_sp_o = extract_scenes(
        data,
        10,
        10,
        overlapping=True,
        min_valid=50,
        reference_var="surface_precip"
    )
    scenes_tbs_o = extract_scenes(
        data,
        10,
        10,
        overlapping=True,
        min_valid=50,
        reference_var="brightness_temperatures"
    )
    scenes_sp = extract_scenes(
        data,
        10,
        10,
        overlapping=False,
        min_valid=50,
        reference_var="surface_precip"
    )
    scenes_tbs = extract_scenes(
        data,
        10,
        10,
        overlapping=False,
        min_valid=50,
        reference_var="brightness_temperatures"
    )

    for scene in scenes_sp_o:
        assert np.isfinite(scene.surface_precip.data).sum() >= 50

    assert len(scenes_sp_o) <= len(scenes_tbs_o)
    assert len(scenes_sp) <= len(scenes_sp_o)
    assert len(scenes_tbs) <= len(scenes_tbs_o)


def test_write_training_samples_1d(
        tmp_path,
        sim_collocations_gmi
):
    """
    Ensure that extracting and writing training samples produces
    scenes of the expected size and containing the expected amount
    of valid pixels.
    """
    data = sim_collocations_gmi

    write_training_samples_1d(
        tmp_path,
        "sim_",
        sim_collocations_gmi,
    )

    samples = sorted(list(tmp_path.glob("*.nc")))
    assert len(samples) == 1


def test_write_training_samples_3d(
        tmp_path,
        sim_collocations_gmi
):
    """
    Ensure that extracting and writing training samples produces
    scenes of the expected size and containing the expected amount
    of valid pixels.
    """
    data = sim_collocations_gmi

    write_training_samples_3d(
        tmp_path,
        "sim_",
        sim_collocations_gmi,
        min_valid=512,
        n_scans=128,
        n_pixels=64
    )

    samples = sorted(list(tmp_path.glob("*.nc")))
    for sample in samples:
        data = xr.load_dataset(sample)
        valid = np.isfinite(data.surface_precip.data)

        assert valid.shape == (128, 64)
        assert valid.sum() >= 512
