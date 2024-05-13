import os
from pathlib import Path
from typing import List

import pytest

import xarray as xr

from gprof_nn import sensors
from gprof_nn.data.preprocessor import run_preprocessor
from gprof_nn.data import sim, mrms, era5, l1c

HAS_ARCHIVES = Path(sensors.GMI.l1c_file_path).exists()
NEEDS_ARCHIVES = pytest.mark.skipif(
    not HAS_ARCHIVES, reason="L1C data not available."
)

SIM_DATA = Path("/qdata1/pbrown/dbaseV8")
NEEDS_SIM_DATA = pytest.mark.skipif(
    not SIM_DATA.exists(), reason="Needs sim files."
)

MRMS_DATA = Path("/pdata4/veljko/")
NEEDS_MRMS_DATA = pytest.mark.skipif(
    not MRMS_DATA.exists(), reason="Needs MRMS files."
)

HAS_TEST_DATA = "GPROF_NN_TEST_DATA" in os.environ
NEEDS_TEST_DATA = pytest.mark.skipif(
    not HAS_TEST_DATA, reason="Test data not available."
)

TEST_DATA_PATH = Path(__file__).parent / "test_data"
TEST_DATA_PATH.mkdir(exist_ok=True)


@pytest.fixture()
def test_data():
    """
    The test data path as set in the 'GPROF_NN_TEST_DATA' environment variable.
    """
    return Path(os.environ["GPROF_NN_TEST_DATA"])


@pytest.fixture(scope="session")
def sim_collocations_gmi() -> xr.Dataset:
    """
    Provides an xarray.Dataset of a .sim file collocated with input
    data from the preprocessor.
    """
    input_file = SIM_DATA / "simV8/1810/GMI.dbsatTb.20181031.026559.sim"

    data_path = TEST_DATA_PATH / "gmi" / "sim"
    if not (data_path / "collocations.nc").exists():
        data_path.mkdir(parents=True, exist_ok=True)
        data = sim.collocate_targets(
            input_file,
            sensors.GMI,
            None,
        )
        data.to_netcdf(data_path / "collocations.nc")
    return xr.load_dataset(data_path / "collocations.nc")

@pytest.fixture(scope="session")
def l1c_file_gmi(tmpdir_factory) -> Path:
    l1c_path = Path(sensors.GMI.l1c_file_path) / "1801" / "180101"
    l1c_files = sorted(list(
        l1c_path.glob(f"**/{sensors.GMI.l1c_file_prefix}*.HDF5")
    ))
    l1c_path_gmi = tmpdir_factory.mktemp("l1c_gmi")
    l1c_file = l1c.L1CFile(l1c_files[0])
    new_file = l1c_path_gmi / l1c_files[0].name
    l1c_file.extract_scan_range(400, 700, new_file)
    return Path(new_file)

@pytest.fixture(scope="session")
def preprocessor_file_gmi(tmpdir_factory, l1c_file_gmi) -> Path:
    pp_path_gmi = tmpdir_factory.mktemp("pp_gmi")
    pp_file = pp_path_gmi / Path(l1c_file_gmi).with_suffix(".pp").name
    run_preprocessor(l1c_file_gmi, sensors.GMI, output_file=pp_file)
    return pp_file

@pytest.fixture(scope="session")
def preprocessor_data_gmi(l1c_file_gmi) -> xr.Dataset:
    data_pp = run_preprocessor(l1c_file_gmi, sensors.GMI)
    return data_pp

@pytest.fixture(scope="session")
def training_files_1d_gmi_sim(
        tmp_path_factory,
        sim_collocations_gmi: xr.Dataset
) -> List[Path]:
    """
    Provides GPROF-NN 3D training data for gmi.
    """
    output_path = tmp_path_factory.mktemp("1d")
    sim.write_training_samples_1d(output_path, "sim", sim_collocations_gmi)
    return sorted(list(output_path.glob("*.nc")))


@pytest.fixture(scope="session")
def training_files_3d_gmi_sim(
        tmp_path_factory,
        sim_collocations_gmi: xr.Dataset
) -> List[Path]:
    """
    Provides GPROF-NN 3D training data for gmi.
    """
    output_path = tmp_path_factory.mktemp("3d")
    sim.write_training_samples_3d(
        output_path,
        "sim",
        sim_collocations_gmi,
        n_scans=221,
        n_pixels=221
    )

    return sorted(list(output_path.glob("*.nc")))

@pytest.fixture(scope="session")
def mrms_match_file_gmi() -> List[Path]:
    """
    Provides a list containing trainig data files for the GPROF-NN 1D retrieval
    for GMI derived from collocations of MRMS and GMI.
    """
    match_file = MRMS_DATA / "GMI2MRMS_match2019" / "db_mrms4GMI" / "1901_MRMS2GMI_gprof_db_08all.bin.gz"
    return match_file

@pytest.fixture(scope="session")
def training_files_1d_gmi_mrms() -> List[Path]:
    """
    Provides a list containing trainig data files for the GPROF-NN 1D retrieval
    for GMI derived from collocations of MRMS and GMI.
    """
    match_file = MRMS_DATA / "GMI2MRMS_match2019" / "db_mrms4GMI" / "1901_MRMS2GMI_gprof_db_08all.bin.gz"
    l1c_file = (
        Path(sensors.GMI.l1c_file_path) /
        "1901/190101/1C-R.GPM.GMI.XCAL2016-C.20190101-S110239-E123512.027517.V07A.HDF5"
    )

    data_path_1d = TEST_DATA_PATH / "gmi" / "mrms" / "1d"
    data_path_3d = TEST_DATA_PATH / "gmi" / "mrms" / "3d"
    files = list(data_path_1d.glob("*.nc"))
    if len(files) == 0:
        data_path_1d.mkdir(parents=True, exist_ok=True)
        data_path_3d.mkdir(parents=True, exist_ok=True)
        data = mrms.extract_collocations(
            sensors.GMI,
            match_file,
            l1c_file,
            data_path_1d,
            data_path_3d
        )
    return sorted(list(data_path_1d.glob("*.nc")))

@pytest.fixture(scope="session")
def training_files_3d_gmi_mrms(training_files_1d_gmi_mrms) -> Path:
    """
    Provides a path containing trainig data for the GPROF-NN 3D retrieval for GMI
    derived from collocations of MRMS and GMI.
    """
    return sorted(list((training_files_1d_gmi_mrms[0].parent.parent / "3d").glob("*.nc")))

@pytest.fixture(scope="session")
def sim_collocations_mhs() -> xr.Dataset:
    """
    Provides an xarray.Dataset of a .sim file collocated with input
    data from the preprocessor.
    """
    input_file = SIM_DATA / "simV8x_mhs/1810/MHS.dbsatTb.20181031.026559.sim"

    data_path = TEST_DATA_PATH / "mhs" / "sim"
    if not (data_path / "collocations.nc").exists():
        data_path.mkdir(parents=True, exist_ok=True)
        data = sim.collocate_targets(
            input_file,
            sensors.MHS,
            None,
        )
        data.to_netcdf(data_path / "collocations.nc")
    return xr.load_dataset(data_path / "collocations.nc")


@pytest.fixture(scope="session")
def training_files_1d_mhs_sim(
        tmp_path_factory,
        sim_collocations_mhs: xr.Dataset
) -> List[Path]:
    """
    Provides GPROF-NN 3D training data for MHS.
    """
    output_path = tmp_path_factory.mktemp("1d")
    sim.write_training_samples_1d(output_path, "sim", sim_collocations_mhs)
    return sorted(list(output_path.glob("*.nc")))


@pytest.fixture(scope="session")
def training_files_3d_mhs_sim(
        tmp_path_factory,
        sim_collocations_mhs: xr.Dataset
) -> List[Path]:
    """
    Provides GPROF-NN 3D training data for gmi.
    """
    output_path = tmp_path_factory.mktemp("3d")
    sim.write_training_samples_3d(
        output_path,
        "sim",
        sim_collocations_mhs,
        n_scans=221,
        n_pixels=221
    )
    return sorted(list(output_path.glob("*.nc")))


@pytest.fixture(scope="session")
def training_files_1d_mhs_mrms() -> Path:
    """
    Provides a path containing trainig data for the GPROF-NN 1D retrieval for MHS
    derived from collocations of MRMS and MHS.
    """
    match_file = MRMS_DATA / "MHS2MRMS_match2019" / "monthly_2021" / "1901_MRMS2MHS_DB1_01.bin.gz"
    l1c_file = (
        Path(sensors.MHS.l1c_file_path) /
        "1901/190101/1C.NOAA19.MHS.XCAL2021-V.20190101-S082007-E100207.051014.V07A.HDF5"
    )

    data_path_1d = TEST_DATA_PATH / "mhs" / "mrms" / "1d"
    data_path_3d = TEST_DATA_PATH / "mhs" / "mrms" / "3d"
    files = list(data_path_1d.glob("*.nc"))
    if len(files) == 0:
        data_path_1d.mkdir(parents=True, exist_ok=True)
        data_path_3d.mkdir(parents=True, exist_ok=True)
        data = mrms.extract_collocations(
            sensors.MHS,
            match_file,
            l1c_file,
            data_path_1d,
            data_path_3d
        )
    return sorted(list(data_path_1d.glob("*.nc")))


@pytest.fixture(scope="session")
def training_files_3d_mhs_mrms(training_files_1d_mhs_mrms) -> Path:
    """
    Provides a path containing trainig data for the GPROF-NN 3D retrieval for MHS
    derived from collocations of MRMS and MHS.
    """
    return sorted(list((training_files_1d_mhs_mrms[0].parent.parent / "3d").glob("*.nc")))

@pytest.fixture(scope="session")
def mrms_match_file_mhs() -> List[Path]:
    """
    Provides a list containing trainig data files for the GPROF-NN 1D retrieval
    for GMI derived from collocations of MRMS and GMI.
    """
    match_file = MRMS_DATA / "MHS2MRMS_match2019" / "monthly_2021" / "1901_MRMS2MHS_DB1_01.bin.gz"
    return match_file


@pytest.fixture(scope="session")
def l1c_file_mhs(tmpdir_factory) -> Path:
    l1c_path = Path(sensors.MHS.l1c_file_path) / "1901" / "190101"
    l1c_files = sorted(list(
        l1c_path.glob(f"**/{sensors.MHS.l1c_file_prefix}*.HDF5")
    ))
    l1c_path_mhs = tmpdir_factory.mktemp("l1c_mhs")
    l1c_file = l1c.L1CFile(l1c_files[0])
    new_file = l1c_path_mhs / l1c_files[0].name
    l1c_file.extract_scan_range(400, 700, new_file)
    return Path(new_file)


@pytest.fixture(scope="session")
def preprocessor_file_mhs(tmpdir_factory, l1c_file_mhs) -> Path:
    pp_path_mhs = tmpdir_factory.mktemp("pp_mhs")
    pp_file = pp_path_mhs / Path(l1c_file_mhs).with_suffix(".pp").name
    run_preprocessor(l1c_file_mhs, sensors.MHS, output_file=pp_file)
    return pp_file


@pytest.fixture(scope="session")
def training_files_1d_mhs_era5() -> Path:
    """
    Provides a path containing trainig data for the GPROF-NN 1D retrieval for MHS
    derived from collocations of MHS observations and ERA5 precip.
    """
    l1c_file = (
        Path(sensors.MHS.l1c_file_path) /
        "1901/190101/1C.NOAA19.MHS.XCAL2021-V.20190101-S082007-E100207.051014.V07A.HDF5"
    )

    data_path_1d = TEST_DATA_PATH / "mhs" / "era5" / "1d"
    data_path_3d = TEST_DATA_PATH / "mhs" / "era5" / "3d"
    files = list(data_path_1d.glob("*.nc"))
    if len(files) == 0:
        data_path_1d.mkdir(parents=True, exist_ok=True)
        data_path_3d.mkdir(parents=True, exist_ok=True)
        data = era5.process_l1c_file(
            sensors.MHS,
            l1c_file,
            data_path_1d,
            data_path_3d
        )
    return sorted(list(data_path_1d.glob("*.nc")))



@pytest.fixture(scope="session")
def training_files_3d_mhs_era5(training_files_1d_mhs_era5) -> Path:
    """
    Provides a path containing trainig data for the GPROF-NN 3D retrieval for MHS
    derived from collocations of MHS observations and ERA5 precip.
    """
    return sorted(list((training_files_1d_mhs_era5[0].parent.parent / "3d").glob("*.nc")))


@pytest.fixture(scope="session")
def l1c_file_amsr2(tmpdir_factory) -> Path:
    l1c_path = Path(sensors.AMSR2.l1c_file_path) / "1812" / "181201"
    l1c_files = sorted(list(
        l1c_path.glob(f"**/{sensors.AMSR2.l1c_file_prefix}*.HDF5")
    ))
    l1c_path_amsr2 = tmpdir_factory.mktemp("l1c_amsr2")
    l1c_file = l1c.L1CFile(l1c_files[0])
    new_file = l1c_path_amsr2 / l1c_files[0].name
    l1c_file.extract_scan_range(400, 700, new_file)
    return Path(new_file)


@pytest.fixture(scope="session")
def preprocessor_file_amsr2(tmpdir_factory, l1c_file_amsr2) -> Path:
    pp_path_amsr2 = tmpdir_factory.mktemp("pp_amsr2")
    pp_file = pp_path_amsr2 / Path(l1c_file_amsr2).with_suffix(".pp").name
    run_preprocessor(l1c_file_amsr2, sensors.AMSR2, output_file=pp_file)
    return pp_file



@pytest.fixture(scope="session")
def sim_collocations_amsr2() -> xr.Dataset:
    """
    Provides an xarray.Dataset of a .sim file collocated with input
    data from the preprocessor.
    """
    input_file = SIM_DATA / "simV8_amsr2/1810/AMSR2.dbsatTb.20181011.026239.sim"

    data_path = TEST_DATA_PATH / "amsr2" / "sim"
    if not (data_path / "collocations.nc").exists():
        data_path.mkdir(parents=True, exist_ok=True)
        data = sim.collocate_targets(
            input_file,
            sensors.AMSR2,
            None,
        )
        data.to_netcdf(data_path / "collocations.nc")
    return xr.load_dataset(data_path / "collocations.nc")


@pytest.fixture(scope="session")
def training_files_1d_amsr2_sim(
        tmp_path_factory,
        sim_collocations_amsr2: xr.Dataset
) -> List[Path]:
    """
    Provides GPROF-NN 3D training data for AMSR2.
    """
    output_path = tmp_path_factory.mktemp("1d")
    sim.write_training_samples_1d(output_path, "sim", sim_collocations_amsr2)
    return sorted(list(output_path.glob("*.nc")))


@pytest.fixture(scope="session")
def training_files_3d_amsr2_sim(
        tmp_path_factory,
        sim_collocations_amsr2: xr.Dataset
) -> List[Path]:
    """
    Provides GPROF-NN 3D training data for AMSR2.
    """
    output_path = tmp_path_factory.mktemp("3d")
    sim.write_training_samples_3d(
        output_path,
        "sim",
        sim_collocations_amsr2,
        n_scans=221,
        n_pixels=221
    )
    return sorted(list(output_path.glob("*.nc")))


@pytest.fixture(scope="session")
def training_files_1d_amsr2_mrms() -> Path:
    """
    Provides a path containing trainig data for the GPROF-NN 1D retrieval for AMSR2
    derived from collocations of MRMS and AMSR2.
    """
    match_file = MRMS_DATA / "AMSR22MRMS_match2019" / "monthly_2021" / "1812_MRMS2AMSR2_01.bin.gz"
    l1c_file = (
        Path(sensors.AMSR2.l1c_file_path) /
        "1812/181201/1C.GCOMW1.AMSR2.XCAL2016-V.20181201-S060901-E074753.034786.V07A.HDF5"
    )

    data_path_1d = TEST_DATA_PATH / "amsr2" / "mrms" / "1d"
    data_path_3d = TEST_DATA_PATH / "amsr2" / "mrms" / "3d"
    files = list(data_path_1d.glob("*.nc"))
    if len(files) == 0:
        data_path_1d.mkdir(parents=True, exist_ok=True)
        data_path_3d.mkdir(parents=True, exist_ok=True)
        data = mrms.extract_collocations(
            sensors.AMSR2,
            match_file,
            l1c_file,
            data_path_1d,
            data_path_3d
        )
    return sorted(list(data_path_1d.glob("*.nc")))


@pytest.fixture(scope="session")
def training_files_3d_amsr2_mrms(training_files_1d_gmi_mrms) -> Path:
    """
    Provides a path containing trainig data for the GPROF-NN 3D retrieval for GMI
    derived from collocations of MRMS and GMI.
    """
    return sorted(list((training_files_1d_gmi_mrms[0].parent.parent / "3d").glob("*.nc")))


@pytest.fixture(scope="session")
def training_files_1d_amsr2_era5() -> Path:
    """
    Provides a path containing trainig data for the GPROF-NN 1D retrieval for AMSR2
    derived from collocations of AMSR2 observations and ERA5 precip.
    """
    l1c_file = (
        Path(sensors.AMSR2.l1c_file_path) /
        "1812/181201/1C.GCOMW1.AMSR2.XCAL2016-V.20181201-S060901-E074753.034786.V07A.HDF5"
    )

    data_path_1d = TEST_DATA_PATH / "amsr2" / "era5" / "1d"
    data_path_3d = TEST_DATA_PATH / "amsr2" / "era5" / "3d"
    files = list(data_path_1d.glob("*.nc"))
    if len(files) == 0:
        data_path_1d.mkdir(parents=True, exist_ok=True)
        data_path_3d.mkdir(parents=True, exist_ok=True)
        data = era5.process_l1c_file(
            sensors.AMSR2,
            l1c_file,
            data_path_1d,
            data_path_3d
        )
    return sorted(list(data_path_1d.glob("*.nc")))


@pytest.fixture(scope="session")
def training_files_3d_amsr2_era5(training_files_1d_amsr2_era5) -> Path:
    """
    Provides a path containing trainig data for the GPROF-NN 3D retrieval for AMSR2
    derived from collocations of AMSR2 observations and ERA5 precip.
    """
    return sorted(list((training_files_1d_amsr2_era5[0].parent.parent / "3d").glob("*.nc")))


@pytest.fixture(scope="session")
def gprof_nn_1d(tmpdir_factory) -> Path:
    """
    An un-trained GPROF-NN 1D retrieval model.
    """
    model_path = tmpdir_factory.mktemp("gprof_nn_1d")
    training.init("gmi", model_path, "1d", model_path, model_path)
    model = load_and_compile_model(model_path / "model.toml")
    return model


@pytest.fixture(scope="session")
def gprof_nn_3d(tmpdir_factory) -> Path:
    """
    An un-trained GPROF-NN 3D retrieval model.
    """
    model_path = tmpdir_factory.mktemp("gprof_nn_3d")
    training.init("gmi", model_path, "3d", model_path, model_path)
    model = load_and_compile_model(model_path / "model.toml")
    return model
