"""
Tests for code running, writing and reading retrieval data.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from quantnn import QRNN
from quantnn.normalizer import Normalizer

from gprof_nn import sensors
from gprof_nn.data import get_model_path, get_test_data_path
from gprof_nn.data import get_profile_clusters
from gprof_nn.data.training_data import GPROF_NN_1D_Dataset
from gprof_nn.data.preprocessor import PreprocessorFile
from gprof_nn.data.retrieval import (RetrievalFile,
                                     ORBIT_HEADER_TYPES,
                                     PROFILE_INFO_TYPES)
from gprof_nn.retrieval import (calculate_padding_dimensions,
                                RetrievalDriver,
                                RetrievalGradientDriver,
                                PreprocessorLoader1D,
                                PreprocessorLoader3D,
                                NetcdfLoader1D,
                                NetcdfLoader3D,
                                SimulatorLoader)


DATA_PATH = get_test_data_path()


def test_calculate_padding_dimensions():
    """
    Ensure that padding values match expected values and that
    the order is inverse to that of tensor axes.
    """
    x = torch.ones(32, 32)
    padding = calculate_padding_dimensions(x)
    assert padding == (0, 0, 0, 0)

    x = torch.ones(16, 32)
    padding = calculate_padding_dimensions(x)
    assert padding == (0, 0, 8, 8)

    x = torch.ones(32, 16)
    padding = calculate_padding_dimensions(x)
    assert padding == (8, 8, 0, 0)


def test_retrieval_read_and_write(tmp_path):
    """
    Ensure that reading data from a retrieval file and writing that
    data into a retrieval conserves data.

    This checks both the writing of the GPROF binary retrieval file
    format including all headers as well as the parsing of the format.
    """
    retrieval_file = (DATA_PATH / "gmi" /
                      "retrieval" / "GMIERA5_190101_027510.bin")
    retrieval_file = RetrievalFile(retrieval_file, has_profiles=True)
    retrieval_data = retrieval_file.to_xarray_dataset(full_profiles=False)
    preprocessor_file = PreprocessorFile(
        DATA_PATH / "gmi" / "pp" / "GMIERA5_190101_027510.pp"
    )
    ancillary_data = get_profile_clusters()
    output_file = preprocessor_file.write_retrieval_results(
        tmp_path,
        retrieval_data,
        ancillary_data=ancillary_data)
    output_file = RetrievalFile(output_file)

    #
    # Orbit header.
    #

    for k in ORBIT_HEADER_TYPES.fields:
        if k not in ["algorithm", "creation_date", "granule_end_date"]:
            assert retrieval_file.orbit_header[k] == output_file.orbit_header[k]

    #
    # Check profile info.
    #

    #for k in PROFILE_INFO_TYPES.fields:
    #    if not k == "species_description":
    #        assert(np.all(np.isclose(retrieval_file.profile_info[k],
    #                                 output_file.profile_info[k])))
    output_data = output_file.to_xarray_dataset()

    #
    # Check retrieval data.
    #

    for v in retrieval_data.variables:
        if v in ["two_meter_temperature", "frozen_precip"]:
            continue
        if v not in ["surface_precip", "convective_precip"]:
            continue
        assert np.all(np.isclose(retrieval_data[v].data,
                                 output_data[v].data,
                                 rtol=1e-2))


def test_retrieval_preprocessor_1d_gmi(tmp_path):
    """
    Ensure that GPROF-NN 1D retrieval works with preprocessor input.
    """
    input_file = DATA_PATH / "gmi" / "pp" / "GMIERA5_190101_027510.pp"

    model_path = get_model_path("1D", sensors.GMI, "ERA5")
    qrnn = QRNN.load(model_path)
    driver = RetrievalDriver(input_file,
                             qrnn,
                             output_file=tmp_path)
    output_file = driver.run()
    data = RetrievalFile(output_file).to_xarray_dataset()
    assert "rain_water_content" in data.variables

@pytest.mark.xfail
def test_retrieval_l1c_1d_gmi(tmp_path):
    """
    Ensure that GPROF-NN 1D retrieval works with preprocessor input.
    """
    input_file = (
        DATA_PATH / "gmi" /
        "1C-R.GPM.GMI.XCAL2016-C.20180124-S000358-E013632.022190.V05A.HDF5"
    )
    qrnn = QRNN.load(DATA_PATH / "gmi" / "gprof_nn_1d_gmi_era5_na.pckl")
    driver = RetrievalDriver(input_file,
                             qrnn,
                             output_file=tmp_path)
    output_file = driver.run()
    data = RetrievalFile(output_file).to_xarray_dataset()
    assert "rain_water_content" in data.variables


def test_retrieval_preprocessor_1d_mhs(tmp_path):
    """
    Ensure that GPROF-NN 1D retrieval works with preprocessor input.
    """
    input_file = DATA_PATH / "mhs" / "pp" / "MHS.pp"

    model_path = get_model_path("1D", sensors.MHS_NOAA19, "ERA5")
    qrnn = QRNN.load(model_path)
    driver = RetrievalDriver(input_file,
                             qrnn,
                             output_file=tmp_path)
    output_file = driver.run()
    data = RetrievalFile(output_file).to_xarray_dataset()
    assert "rain_water_content" in data.variables


def test_retrieval_preprocessor_3d(tmp_path):
    """
    Ensure that GPROF-NN 3D retrieval works with preprocessor input.
    """
    input_file = DATA_PATH / "gmi" / "pp" / "GMIERA5_190101_027510.pp"

    model_path = get_model_path("3D", sensors.GMI, "ERA5")
    qrnn = QRNN.load(model_path)
    qrnn.model.sensor = sensors.GMI
    driver = RetrievalDriver(input_file,
                             qrnn,
                             output_file=tmp_path)
    output_file = driver.run()
    data = RetrievalFile(output_file).to_xarray_dataset()
    assert "rain_water_content" in data.variables


@pytest.mark.xfail
def test_retrieval_l1c_3d(tmp_path):
    """
    Ensure that GPROF-NN 1D retrieval works with preprocessor input.
    """
    input_file = (
        DATA_PATH / "gmi" / "l1c"
        "1C-R.GPM.GMI.XCAL2016-C.20180124-S000358-E013632.022190.V05A.HDF5"
    )

    qrnn = QRNN.load(DATA_PATH / "gmi" / "gprof_nn_3d_gmi_era5_na.pckl")
    qrnn.model.sensor = sensors.GMI
    driver = RetrievalDriver(input_file,
                             qrnn,
                             output_file=tmp_path,
                             compress=False)
    output_file = driver.run()
    data = RetrievalFile(output_file).to_xarray_dataset()
    assert "rain_water_content" in data.variables


def test_retrieval_netcdf_1d(tmp_path):
    """
    Ensure that GPROF-NN 1D retrieval works with NetCDF input.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc"

    model_path = get_model_path("1D", sensors.GMI, "ERA5")
    qrnn = QRNN.load(model_path)
    qrnn.training_data_class = GPROF_NN_1D_Dataset
    qrnn.preprocessor_class = PreprocessorLoader1D
    driver = RetrievalDriver(input_file,
                             qrnn,
                             output_file=tmp_path,
                             compress=False)
    output_file = driver.run()
    data = xr.load_dataset(output_file)
    assert "rain_water_content" in data.variables
    assert "rain_water_content_true" in data.variables


def test_retrieval_netcdf_1d_full(tmp_path):
    """
    Test running the 1D retrieval with the spatial structure retained.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc"

    model_path = get_model_path("1D", sensors.GMI, "ERA5")
    qrnn = QRNN.load(model_path)
    qrnn.training_data_class = GPROF_NN_1D_Dataset
    qrnn.preprocessor_class = PreprocessorLoader1D
    driver = RetrievalDriver(input_file,
                             qrnn,
                             output_file=tmp_path,
                             compress=False,
                             preserve_structure=True)
    output_file = driver.run()
    data = xr.load_dataset(output_file)
    assert "rain_water_content" in data.variables
    assert "rain_water_content_true" in data.variables

def test_retrieval_netcdf_1d_gradients(tmp_path):
    """
    Ensure that GPROF-NN 1D retrieval with NetCDF input and gradients
    works.
    """
    data_path = Path(__file__).parent / "data"
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc"

    model_path = get_model_path("1D", sensors.GMI, "ERA5")
    qrnn = QRNN.load(model_path)
    qrnn.training_data_class = GPROF_NN_1D_Dataset
    qrnn.preprocessor_class = PreprocessorLoader1D
    driver = RetrievalGradientDriver(input_file,
                                     qrnn,
                                     output_file=tmp_path,
                                     compress=False)
    output_file = driver.run()
    data = xr.load_dataset(output_file)
    assert "surface_precip_grad" in data.variables


def test_retrieval_netcdf_3d(tmp_path):
    """
    Ensure that GPROF-NN 3D retrieval works with NetCDF input.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc"

    model_path = get_model_path("3D", sensors.GMI, "ERA5")
    qrnn = QRNN.load(model_path)
    qrnn.model.sensor = sensors.GMI
    driver = RetrievalDriver(input_file,
                             qrnn,
                             output_file=tmp_path,
                             compress=False)
    output_file = driver.run()
    data = xr.load_dataset(output_file)
    assert "rain_water_content" in data.variables
    assert "pixels" in data.dims.keys()
    assert "scans" in data.dims.keys()


@pytest.mark.xfail
def test_simulator_gmi(tmp_path):
    """
    Ensure that GPROF-NN 3D retrieval works with NetCDF input.
    """
    input_file = DATA_PATH / "gmi" / "gprof_nn_gmi_era5.nc"

    qrnn = QRNN.load(DATA_PATH / "gmi" / "simulator_gmi.pckl")
    qrnn.netcdf_class = SimulatorLoader
    driver = RetrievalDriver(input_file,
                             qrnn,
                             output_file=tmp_path,
                             compress=False)
    output_file = driver.run()
    data = xr.load_dataset(output_file)

    assert "simulated_brightness_temperatures" in data.variables
    assert "brightness_temperature_biases" in data.variables

    data_0 = data[{"samples": data.source == 0}]
    tbs_sim = data_0["simulated_brightness_temperatures"].data
    assert np.all(np.isfinite(tbs_sim))


@pytest.mark.xfail
def test_simulator_mhs(tmp_path):
    """
    Ensure that GPROF-NN 3D retrieval works with NetCDF input.
    """
    input_file = DATA_PATH / "gprof_nn_mhs_era5_5.nc"

    qrnn = QRNN.load(DATA_PATH / "mhs" / "simulator_mhs.pckl")
    driver = RetrievalDriver(input_file,
                             qrnn,
                             output_file=tmp_path,
                             compress=False)
    output_file = driver.run()
    data = xr.load_dataset(output_file)

    assert "simulated_brightness_temperatures" in data.variables
    assert "brightness_temperature_biases" in data.variables

    data_0 = data[{"samples": data.source == 0}]
    tbs_sim = data_0["simulated_brightness_temperatures"].data
    assert np.all(np.isfinite(tbs_sim))

