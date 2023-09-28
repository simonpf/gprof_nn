from pathlib import Path
from typing import List

import pytest

import xarray as xr

from gprof_nn import sensors
from gprof_nn.data.sim import (
    collocate_targets,
    write_training_samples_1d,
    write_training_samples_3d
)

HAS_ARCHIVES = Path(sensors.GMI.l1c_file_path).exists()
NEEDS_ARCHIVES = pytest.mark.skipif(
    not HAS_ARCHIVES, reason="L1C data not available."
)

SIM_DATA = Path("/qdata1/pbrown/dbaseV8")
NEEDS_SIM_DATA = pytest.mark.skipif(
    not SIM_DATA.exists(), reason="Needs sim files."
)

@pytest.fixture(scope="session")
def sim_collocations_gmi() -> xr.Dataset:
    """
    Provides an xarray.Dataset of a .sim file collocated with input
    data from the preprocessor.
    """
    input_file = SIM_DATA / "simV8/1810/GMI.dbsatTb.20181031.026559.sim"
    data = collocate_targets(
        input_file,
        sensors.GMI,
        None,
    )
    return data


@pytest.fixture(scope="session")
def training_files_1d_gmi(
        tmp_path_factory,
        sim_collocations_gmi: xr.Dataset
) -> List[Path]:
    """
    Provides GPROF-NN 3D training data for gmi.
    """

    output_path = tmp_path_factory.mktemp("1d")
    write_training_samples_1d(sim_collocations_gmi, output_path)

    return sorted(list(output_path.glob("*.nc")))


@pytest.fixture(scope="session")
def training_files_3d_gmi(
        tmp_path_factory,
        sim_collocations_gmi: xr.Dataset
) -> List[Path]:
    """
    Provides GPROF-NN 3D training data for gmi.
    """
    output_path = tmp_path_factory.mktemp("3d")
    write_training_samples_3d(sim_collocations_gmi, output_path)

    return sorted(list(output_path.glob("*.nc")))
