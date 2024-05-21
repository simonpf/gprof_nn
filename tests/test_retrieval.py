"""
Tests for code running, writing and reading retrieval data.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr


from conftest import NEEDS_ARCHIVES
from pytorch_retrieve.inference import run_inference

from gprof_nn.retrieval import (
    load_input_data_preprocessor,
    load_input_data_l1c,
    load_input_data_training_1d,
    load_input_data_training_3d,
    GPROFNNInputLoader
)


@NEEDS_ARCHIVES
@pytest.mark.parametrize("preprocessor_fixture", [
    "preprocessor_file_gmi",
    "preprocessor_file_mhs",
    "preprocessor_file_amsr2",
])
def test_load_input_data_preprocessor(preprocessor_fixture, request):

    preprocessor_file = request.getfixturevalue(preprocessor_fixture)

    input_data, _ = load_input_data_preprocessor(preprocessor_file)

    assert "brightness_temperatures" in input_data
    assert isinstance(input_data["brightness_temperatures"], np.ndarray)
    assert input_data["brightness_temperatures"].shape[0] == 15
    assert "viewing_angles" in input_data
    assert isinstance(input_data["viewing_angles"], np.ndarray)
    assert input_data["viewing_angles"].shape[0] == 15
    assert "ancillary_data" in input_data
    assert isinstance(input_data["ancillary_data"], np.ndarray)
    assert input_data["ancillary_data"].shape[0] == 8


@NEEDS_ARCHIVES
@pytest.mark.parametrize("l1c_fixture", [
    "l1c_file_gmi",
    "l1c_file_mhs",
    "l1c_file_amsr2",
])
def test_load_input_data_l1c(l1c_fixture, request):

    l1c_file = request.getfixturevalue(l1c_fixture)

    input_data, _ = load_input_data_l1c(l1c_file, needs_ancillary=True)
    assert "brightness_temperatures" in input_data
    assert isinstance(input_data["brightness_temperatures"], np.ndarray)
    assert input_data["brightness_temperatures"].shape[0] == 15
    assert "viewing_angles" in input_data
    assert isinstance(input_data["viewing_angles"], np.ndarray)
    assert input_data["viewing_angles"].shape[0] == 15
    assert "ancillary_data" in input_data
    assert isinstance(input_data["ancillary_data"], np.ndarray)
    assert input_data["ancillary_data"].shape[0] == 8

    input_data, _ = load_input_data_l1c(l1c_file, needs_ancillary=False)
    assert "brightness_temperatures" in input_data


@NEEDS_ARCHIVES
@pytest.mark.parametrize("training_data_fixture", [
    "training_files_1d_gmi_sim",
    "training_files_1d_gmi_mrms",
    "training_files_1d_mhs_sim",
    "training_files_1d_mhs_mrms",
    "training_files_1d_mhs_era5",
    "training_files_1d_amsr2_sim",
    "training_files_1d_amsr2_mrms",
    "training_files_1d_amsr2_era5",
])
def test_load_input_data_training_1d(training_data_fixture, request):

    training_file = request.getfixturevalue(training_data_fixture)[0]
    input_data, _ = load_input_data_training_1d(training_file)

    assert "brightness_temperatures" in input_data
    assert input_data["brightness_temperatures"].shape[-1] == 15
    assert "viewing_angles" in input_data
    assert input_data["viewing_angles"].shape[-1] == 15
    assert "ancillary_data" in input_data
    assert input_data["ancillary_data"].shape[-1] == 8


@NEEDS_ARCHIVES
@pytest.mark.parametrize("training_data_fixture", [
    "preprocessor_file_gmi",
    "preprocessor_file_mhs",
    "preprocessor_file_amsr2",
    "training_files_1d_gmi_sim",
    "training_files_1d_gmi_mrms",
    "training_files_1d_mhs_sim",
    "training_files_1d_mhs_mrms",
    "training_files_1d_mhs_era5",
    "training_files_1d_amsr2_sim",
    "training_files_1d_amsr2_mrms",
    "training_files_1d_amsr2_era5",
])
def test_input_loader_1d(training_data_fixture, request):

    training_files = request.getfixturevalue(training_data_fixture)

    input_loader = GPROFNNInputLoader(training_files, config="1d", needs_ancillary=True)
    for input_data, aux, filename in input_loader:
        assert "brightness_temperatures" in input_data
        assert "viewing_angles" in input_data
        assert "ancillary_data" in input_data
        assert "latitude" in aux
        assert "longitude" in aux

    input_loader = GPROFNNInputLoader(training_files, config="1d", needs_ancillary=False)
    for input_data, aux, filename in input_loader:
        assert "brightness_temperatures" in input_data
        assert "latitude" in aux
        assert "longitude" in aux


@NEEDS_ARCHIVES
@pytest.mark.parametrize("training_data_fixture", [
    "preprocessor_file_gmi",
    "preprocessor_file_mhs",
    "preprocessor_file_amsr2",
    "training_files_3d_gmi_sim",
    "training_files_3d_gmi_mrms",
    "training_files_3d_mhs_sim",
    "training_files_3d_mhs_mrms",
    "training_files_3d_mhs_era5",
    "training_files_3d_amsr2_sim",
    "training_files_3d_amsr2_mrms",
    "training_files_3d_amsr2_era5",
])
def test_input_loader_3d(training_data_fixture, request):

    training_files = request.getfixturevalue(training_data_fixture)

    input_loader = GPROFNNInputLoader(training_files, config="3d", needs_ancillary=True)
    for input_data, aux, filename in input_loader:
        assert "brightness_temperatures" in input_data
        assert input_data["brightness_temperatures"].shape[0] == 15
        assert "viewing_angles" in input_data
        assert input_data["viewing_angles"].shape[0] == 15
        assert "ancillary_data" in input_data
        assert input_data["ancillary_data"].shape[0] == 8
        assert "latitude" in aux
        assert "longitude" in aux

    input_loader = GPROFNNInputLoader(training_files, config="3d", needs_ancillary=False)
    for input_data, aux, filename in input_loader:
        assert "brightness_temperatures" in input_data
        assert input_data["brightness_temperatures"].shape[0] == 15
        assert input_data["viewing_angles"].shape[0] == 15
        assert input_data["ancillary_data"].shape[0] == 8
        assert "latitude" in aux
        assert "longitude" in aux


@pytest.mark.parametrize("input_data_fixture", [
    "preprocessor_file_gmi",
    "preprocessor_file_mhs",
    "preprocessor_file_amsr2",
    "training_files_1d_gmi_sim",
    "training_files_1d_gmi_mrms",
    "training_files_1d_mhs_sim",
    "training_files_1d_mhs_mrms",
    "training_files_1d_mhs_era5",
    "training_files_1d_amsr2_sim",
    "training_files_1d_amsr2_mrms",
    "training_files_1d_amsr2_era5",
    "training_files_3d_gmi_sim",
    "training_files_3d_gmi_mrms",
    "training_files_3d_mhs_sim",
    "training_files_3d_mhs_mrms",
    "training_files_3d_mhs_era5",
    "training_files_3d_amsr2_sim",
    "training_files_3d_amsr2_mrms",
    "training_files_3d_amsr2_era5",
])
def test_inference_gprof_nn_1d(
        input_data_fixture,
        request,
        tmp_path,
        gprof_nn_1d
):
    input_data = request.getfixturevalue(input_data_fixture)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    input_loader = GPROFNNInputLoader(input_data, config="1d")
    run_inference(
        gprof_nn_1d,
        input_loader,
        gprof_nn_1d.inference_config,
        output_path=output_dir
    )

    output_files = sorted(list(output_dir.glob("*.nc")))
    assert len(output_files) > 0
    with xr.open_dataset(output_files[0]) as results:
        assert "latitude" in results
        assert "longitude" in results
        assert "surface_precip" in results
        assert "probability_of_precipitation" in results


@pytest.mark.parametrize("input_data_fixture", [
    "preprocessor_file_gmi",
    "preprocessor_file_mhs",
    "preprocessor_file_amsr2",
    "training_files_3d_gmi_sim",
    "training_files_3d_gmi_mrms",
    "training_files_3d_mhs_sim",
    "training_files_3d_mhs_mrms",
    "training_files_3d_mhs_era5",
    "training_files_3d_amsr2_sim",
    "training_files_3d_amsr2_mrms",
    "training_files_3d_amsr2_era5",
])
def test_inference_gprof_nn_3d(
        input_data_fixture,
        request,
        tmp_path,
        gprof_nn_1d
):
    input_data = request.getfixturevalue(input_data_fixture)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    input_loader = GPROFNNInputLoader(input_data, config="1d")
    run_inference(
        gprof_nn_1d,
        input_loader,
        gprof_nn_1d.inference_config,
        output_path=output_dir
    )

    output_files = sorted(list(output_dir.glob("*.nc")))
    assert len(output_files) > 0
    with xr.open_dataset(output_files[0]) as results:
        assert "scans" in results.dims
        assert "pixels" in results.dims
        assert "latitude" in results
        assert "longitude" in results
        assert "surface_precip" in results
        assert "probability_of_precipitation" in results
