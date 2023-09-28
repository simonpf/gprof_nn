import os
from pathlib import Path

from gprof_nn.config import (
    parse_config_file,
    get_config_file,
    set_config
)

CONFIG = """
[preprocessor]
GMI=/gmi/preprocessor
MHS=/mhs/preprocessor
"""

def test_config(tmp_path):
    """
    Test parsing of config file.
    """
    config_file = tmp_path / "config.ini"
    with open(config_file, "w") as hndl:
        hndl.write(CONFIG)

    config = parse_config_file(config_file)
    assert config.preprocessor.GMI == Path("/gmi/preprocessor")
    assert config.preprocessor.MHS == Path("/mhs/preprocessor")


def test_get_config_file(tmp_path):
    """
    Test path of config file.
    """
    config_file = tmp_path / "config.ini"
    os.environ["GPROF_NN_CONFIG"] = str(tmp_path / "config.ini")
    config_file_2 = get_config_file()
    assert config_file == config_file_2


def test_set_config(tmp_path):
    """
    Test setting of config file.
    """
    config_file = tmp_path / "config.ini"
    os.environ["GPROF_NN_CONFIG"] = str(tmp_path / "config.ini")
    set_config("data", "era5_path", tmp_path)

    config = parse_config_file(config_file)
    assert config.data.era5_path == tmp_path
