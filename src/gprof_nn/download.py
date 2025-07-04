"""
gprof_nn.download
=================

Functionality to download GPROF-NN retrieval models and auxiliary data.
"""

import logging
from pathlib import Path
from typing import Optional

from filelock import FileLock
from huggingface_hub import hf_hub_download, snapshot_download

from gprof_nn.config import CONFIG


LOGGER = logging.getLogger(__name__)


def download_model(sensor: str) -> Path:
    """
    Download model to gprof_nn model path and return local path.

    Args:
        sensor: The name of the sensor for which to download the model.

    Return:
        The local path of the model.
    """
    model = f"gprof_nn_3d_{sensor.name.lower()}.pt"
    model_path = CONFIG.data.model_path

    if not (model_path / model).exists():
        LOGGER.info("Downloading model file %s to model path %s", model, model_path)
        lock = FileLock((model_path / model).with_suffix(".lock"))
        with lock:
            try:
                hf_hub_download("simonpf/gprof_nn", filename=model, local_dir=model_path)
            except Exception:
                LOGGER.warning(
                    "Didn't find a model for sensor '%s'.",
                    sensor.name
                )
    else:
        LOGGER.debug("Found model at %s.", model_path / model)


    return model_path / model


def update_models() -> Path:
    """
    Update all retrieval models to the latest version.
    """
    from gprof_nn import sensors
    sensor_names = []
    for value in vars(sensors).values():
        if isinstance(value, sensors.Sensor):
            sensor_names.append(value.name)

    for sensor_name in sensor_names:
        model = f"gprof_nn_3d_{sensor_name.lower()}.pt"
        model_path = CONFIG.data.model_path

        if (model_path / model).exists():
            LOGGER.info("Downloading model file %s to model path %s", model, model_path)
            lock = FileLock((model_path / model).with_suffix(".lock"))
            with lock:
                try:
                    hf_hub_download("simonpf/gprof_nn", filename=model, local_dir=model_path, force_download=True)
                except Exception:
                    LOGGER.warning(
                        "Didn't find a model for sensor '%s'.",
                        sensor.name
                    )


TEST_FILES = {
    "gmi": {
        "l1c": "1C-R.GPM.GMI.XCAL2016-C.20241008-S051110-E064423.060255.V07B.HDF5",
        "preprocessor": "1C-R.GPM.GMI.XCAL2016-C.20241008-S051110-E064423.060255.V07B.pp",
    },
    "amsr2": {
        "l1c": "1C.GCOMW1.AMSR2.XCAL2016-V.20241008-S064659-E082551.065921.V07A.HDF5",
        "preprocessor": "1C.GCOMW1.AMSR2.XCAL2016-V.20241008-S064659-E082551.065921.V07A.pp",
    },
    "atms": {
        "l1c": "1C.NOAA20.ATMS.XCAL2019-V.20241008-S054520-E072649.035695.V07A.HDF5",
        "preprocessor": "1C.NOAA20.ATMS.XCAL2019-V.20241008-S054520-E072649.035695.V07A.pp",
    },
}


def download_test_file(sensor: str, kind: str, dest: Optional[Path] = None) -> Path:
    """
    Download test input file for a given sensor.

    Args:
        sensor: The name of the sensor for which to download the model.
        kind: 'l1c' or 'preprocessor' to download the L1C or preprocessor input file.
        dest: Directory to which to download the file.

    Return:
        The local path of the input file.
    """
    if dest is None:
        dest = Path(".")

    test_file = TEST_FILES.get(sensor, {}).get(kind, None)
    if test_file is None:
        raise ValueError(f"No known test file for sensor '{sensor}' and kind '{kind}'.")
    subfolder = f"test_data/{sensor}"
    lock_file = (dest / test_file).with_suffix(".lock")
    lock = FileLock(lock_file)
    with lock:
        LOGGER.info("Downloading test  file %s to %s", test_file, dest)
        hf_hub_download(
            "simonpf/gprof_nn",
            filename=test_file,
            subfolder=subfolder,
            local_dir=dest,
        )
    lock_file.unlink()

    return dest / subfolder / test_file
