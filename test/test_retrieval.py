"""
Tests for code running, writing and reading retrieval data.
"""
from pathlib import Path

import numpy as np
import xarray as xr
import torch

from quantnn import QRNN
from quantnn.normalizer import Normalizer

from gprof_nn import sensors
from gprof_nn.data.training_data import GPROF_NN_0D_Dataset
from gprof_nn.data.preprocessor import PreprocessorFile
from gprof_nn.data.retrieval import (RetrievalFile,
                                     ORBIT_HEADER_TYPES,
                                     PROFILE_INFO_TYPES)
from gprof_nn.retrieval import (calculate_padding_dimensions,
                                RetrievalDriver,
                                PreprocessorLoader0D,
                                PreprocessorLoader2D,
                                NetcdfLoader0D,
                                NetcdfLoader2D,
                                SimulatorLoader)

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
    data_path = Path(__file__).parent / "data"
    retrieval_file = data_path / "GMIERA5_190101_027510_np.bin"
    retrieval_file = RetrievalFile(retrieval_file, has_profiles=True)
    retrieval_data = retrieval_file.to_xarray_dataset(full_profiles=False)
    preprocessor_file = PreprocessorFile(data_path / "GMIERA5_190101_027510.pp")
    output_file = preprocessor_file.write_retrieval_results(tmp_path,
                                                            retrieval_data,
                                                            data_path)
    output_file = RetrievalFile(output_file)

    #
    # Orbit header.
    #

    for k in ORBIT_HEADER_TYPES.fields:
        if not k in ["algorithm", "creation_date"]:
            assert retrieval_file.orbit_header[k] == output_file.orbit_header[k]

    #
    # Check profile info.
    #

    for k in PROFILE_INFO_TYPES.fields:
        if not k == "species_description":
            assert(np.all(np.isclose(retrieval_file.profile_info[k],
                                     output_file.profile_info[k])))
    output_data = output_file.to_xarray_dataset()

    #
    # Check retrieval data.
    #

    for v in retrieval_data.variables:
        if v in ["two_meter_temperature", "frozen_precip"]:
            continue
        assert np.all(np.isclose(retrieval_data[v].data,
                                 output_data[v].data,
                                 rtol=1e-2))


def test_retrieval_preprocessor_0d_gmi(tmp_path):
    """
    Ensure that GPROF-NN 0D retrieval works with preprocessor input.
    """
    data_path = Path(__file__).parent / "data"
    input_file = data_path / "GMIERA5_190101_027510.pp"

    qrnn = QRNN.load(data_path / "gprof_nn_0d.pckl")
    normalizer = Normalizer.load(data_path / "normalizer.pckl")
    driver = RetrievalDriver(input_file,
                             normalizer,
                             qrnn,
                             ancillary_data=data_path,
                             output_file=tmp_path)
    output_file = driver.run()
    data = RetrievalFile(output_file).to_xarray_dataset()
    assert "rain_water_content" in data.variables


def test_retrieval_preprocessor_0d_mhs(tmp_path):
    """
    Ensure that GPROF-NN 0D retrieval works with preprocessor input.
    """
    data_path = Path(__file__).parent / "data"
    input_file = data_path / "MHS.pp"

    qrnn = QRNN.load(data_path / "gprof_nn_0d_mhs.pckl")
    normalizer = Normalizer.load(data_path / "normalizer_mhs.pckl")
    driver = RetrievalDriver(input_file,
                             normalizer,
                             qrnn,
                             ancillary_data=data_path,
                             output_file=tmp_path)
    output_file = driver.run()
    data = RetrievalFile(output_file).to_xarray_dataset()
    assert "rain_water_content" in data.variables


def test_retrieval_preprocessor_2d(tmp_path):
    """
    Ensure that GPROF-NN 0D retrieval works with preprocessor input.
    """
    data_path = Path(__file__).parent / "data"
    input_file = data_path / "GMIERA5_190101_027510.pp"

    qrnn = QRNN.load(data_path / "gprof_nn_2d.pckl")
    qrnn.model.sensor = sensors.GMI
    normalizer = Normalizer.load(data_path / "normalizer.pckl")
    driver = RetrievalDriver(input_file,
                             normalizer,
                             qrnn,
                             ancillary_data=data_path,
                             output_file=tmp_path)
    output_file = driver.run()
    data = RetrievalFile(output_file).to_xarray_dataset()
    assert "rain_water_content" in data.variables


def test_retrieval_netcdf_0d(tmp_path):
    """
    Ensure that GPROF-NN 0D retrieval works with NetCDF input.
    """
    data_path = Path(__file__).parent / "data"
    input_file = data_path / "gmi" / "gprof_nn_gmi_era5.nc"

    qrnn = QRNN.load(data_path / "gprof_nn_0d.pckl")
    qrnn.training_data_class = GPROF_NN_0D_Dataset
    qrnn.preprocessor_class = PreprocessorLoader0D
    normalizer = Normalizer.load(data_path / "normalizer.pckl")
    driver = RetrievalDriver(input_file,
                             normalizer,
                             qrnn,
                             ancillary_data=data_path,
                             output_file=tmp_path)
    output_file = driver.run()
    data = xr.load_dataset(output_file)
    assert "rain_water_content" in data.variables
    assert "pixels" in data.dims.keys()
    assert "scans" in data.dims.keys()


def test_retrieval_netcdf_2d(tmp_path):
    """
    Ensure that GPROF-NN 2D retrieval works with NetCDF input.
    """
    data_path = Path(__file__).parent / "data"
    input_file = data_path / "gmi" / "gprof_nn_gmi_era5.nc"

    qrnn = QRNN.load(data_path / "gprof_nn_2d.pckl")
    qrnn.model.sensor = sensors.GMI
    normalizer = Normalizer.load(data_path / "normalizer.pckl")
    driver = RetrievalDriver(input_file,
                             normalizer,
                             qrnn,
                             ancillary_data=data_path,
                             output_file=tmp_path)
    output_file = driver.run()
    data = xr.load_dataset(output_file)
    assert "rain_water_content" in data.variables
    assert "pixels" in data.dims.keys()
    assert "scans" in data.dims.keys()


def test_simulator_gmi(tmp_path):
    """
    Ensure that GPROF-NN 2D retrieval works with NetCDF input.
    """
    data_path = Path(__file__).parent / "data"
    input_file = data_path / "gmi" / "gprof_nn_gmi_era5.nc"

    qrnn = QRNN.load(data_path / "gmi" / "simulator_gmi.pckl")
    qrnn.netcdf_class = SimulatorLoader
    normalizer = Normalizer.load(data_path / "normalizer_gmi.pckl")
    driver = RetrievalDriver(input_file,
                             normalizer,
                             qrnn,
                             ancillary_data=data_path,
                             output_file=tmp_path)
    output_file = driver.run()
    data = xr.load_dataset(output_file)

    assert "simulated_brightness_temperatures" in data.variables
    assert "brightness_temperature_biases" in data.variables

    data_0 = data[{"samples": data.source == 0}]
    tbs_sim = data_0["simulated_brightness_temperatures"].data
    assert np.all(np.isfinite(tbs_sim))


def test_simulator_mhs(tmp_path):
    """
    Ensure that GPROF-NN 2D retrieval works with NetCDF input.
    """
    data_path = Path(__file__).parent / "data"
    input_file = data_path / "gprof_nn_mhs_era5_5.nc"

    qrnn = QRNN.load(data_path / "mhs" / "simulator_mhs.pckl")
    normalizer = Normalizer.load(data_path / "normalizer_gmi.pckl")
    driver = RetrievalDriver(input_file,
                             normalizer,
                             qrnn,
                             ancillary_data=data_path,
                             output_file=tmp_path)
    output_file = driver.run()
    data = xr.load_dataset(output_file)

    assert "simulated_brightness_temperatures" in data.variables
    assert "brightness_temperature_biases" in data.variables

    data_0 = data[{"samples": data.source == 0}]
    tbs_sim = data_0["simulated_brightness_temperatures"].data
    assert np.all(np.isfinite(tbs_sim))

