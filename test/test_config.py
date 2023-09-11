from pathlib import Path

from gprof_nn.config import parse_config_file

CONFIG = """
[preprocessor]
GMI=/gmi/preprocessor
MHS=/mhs/preprocessor
"""

def test_config(tmp_path):

    config_file = tmp_path / "config.ini"
    with open(config_file, "w") as hndl:
        hndl.write(CONFIG)

    config = parse_config_file(config_file)
    assert config.preprocessor.GMI == Path("/gmi/preprocessor")
    assert config.preprocessor.MHS == Path("/mhs/preprocessor")
