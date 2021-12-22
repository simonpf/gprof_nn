"""
=============
gprof_nn.data
=============

The data module provides functionality to dynamically download
data needed by the module.

Its sub-modules provide functionality to read and process various
data formats required for the processing of training data.
"""
from pathlib import Path
import shutil
import subprocess
from tempfile import mkdtemp
import urllib

from appdirs import user_config_dir

_DATA_DIR = Path(user_config_dir("gprof_nn", "gprof_nn"))
if not _DATA_DIR.exists():
    _DATA_DIR.mkdir(parents=True)

_MODEL_DIR = _DATA_DIR / "models"
if not _MODEL_DIR.exists():
    _MODEL_DIR.mkdir(parents=True)

_PROFILE_DIR = _DATA_DIR / "profiles"
if not _PROFILE_DIR.exists():
    _PROFILE_DIR.mkdir(parents=True, exist_ok=True)

_TEST_DIR = _DATA_DIR / "test"
if not _TEST_DIR.exists():
    _TEST_DIR.mkdir(parents=True, exist_ok=True)


_DATA_URL = "http://rain.atmos.colostate.edu/gprof_nn/"


def get_file(path):
    """
    Get file from user's data directory or download it if necessary.

    Args:

        path: Path of the file relative to the root of the data
            directory.

    Return:
        A absolute local path pointing to the file.
    """
    local_path = _DATA_DIR / path
    if not local_path.exists():
        url = _DATA_URL + str(path)
        with urllib.request.urlopen(url) as response:
            with open(local_path, "wb") as output:
                shutil.copyfileobj(response, output)

    return local_path


def get_model_path(kind, sensor, configuration):
    """
    Get path to local GPROF-NN model.

    Args:
        kind: The type of the GPROF-NN model ('1D' or '3D')
        sensor: Sensor object representing the sensor for which
            to retrieve the model.
        configuration: The configuration of the model
            ('ERA5' or 'GANAL').

    Return:
        Local path to the requested model.
    """
    kind = kind.lower()
    if kind not in ["1d", "3d"]:
        raise ValueError("'kind' must be one of: '1D', '3D'")
    configuration = configuration.lower()
    if configuration not in ["era5", "ganal"]:
        raise ValueError("'configuration' must be one of: 'ERA5', 'GANAL'")

    sensor_name = sensor.full_name.lower()
    model_name = f"gprof_nn_{kind}_{sensor_name}_{configuration}.pckl"
    path = Path("models") / model_name

    try:
        return get_file(path)
    except urllib.error.URLError:
        pass

    sensor_name = sensor.sensor_name.lower()
    model_name = f"gprof_nn_{kind}_{sensor_name}_{configuration}.pckl"
    path = Path("models") / model_name
    try:
        return get_file(path)
    except urllib.error.HTTPError:
        raise ValueError(
            f"Couldn't find a model for sensor '{sensor.name}' and "
            f"configuration '{configuration}' neither locally nor on "
            f"the server. Maybe it doesn't exist yet?"
        )


def get_profile_clusters():
    """
    Return local path to the directory containing the profile cluster
    files.

    Return:
        Absolute path of directory containing the profile clusters.
    """
    path_raining = "profiles/GPM_profile_clustersV7.dat"
    get_file(path_raining)
    path_non_raining = "profiles/GPM_profile_clustersNRV7.dat"
    path = get_file(path_non_raining)
    return path.parent


def get_test_data_path():
    """
    Return path to test data and download it if not alread
    available.
    """
    test_data_path = _TEST_DIR / "data"
    if not test_data_path.exists():
        tmp = None
        get_file("test/data.tar.gz")
        subprocess.run(
            ["tar", "-xvf", str(_TEST_DIR / "data.tar.gz")],
            capture_output=True,
            cwd=_TEST_DIR
        )
        (_TEST_DIR / "data.tar.gz").unlink()
    return test_data_path
