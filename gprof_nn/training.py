"""
gprof_nn.training
=================

Interface for training of the GPROF-NN retrievals.
"""
import logging
import os
from pathlib import Path
from typing import Optional, List

import click


from gprof_nn import sensors
import gprof_nn.logging


LOGGER = logging.Logger(__name__)


def init(
        sensor: str,
        path: Path,
        config: str,
        training_data_path: Path,
        validation_data_path: Optional[Path] = None,
        targets: List[str] = None,
        ancillary_data: bool = True,
) -> None:
    """
    Initialize a GPROF-NN model in a given location.

    Args:
        sensor: Name of the sensor for which the training is performed. Only relevant for
            the 'sim' configuration.
        path: A path object pointing to the location in which to writhe
            the model and training configuration files.
        config: A string specifying the model configuration: '1d', '3d' or 'sim'.
        training_data_path: The path containing the training data.
        validation_data_path: The path containing the validation data.
        targets: If given, the model will only be trained to retrieve the
             given subset of retrieval targets.
        ancillary_data: Whether or not to include ancillary the model
            should make use of ancillary data.
    """
    config_path = Path(__file__).parent / "config_files"

    training_config = config_path / f"gprof_nn_{config.lower()}_training.toml"
    training_config = open(training_config, "r").read()
    training_config = training_config.format(**{
        "training_data_path": str(training_data_path),
        "validation_data_path": str(validation_data_path)
    })
    with open(path / "training.toml", "w") as output:
        output.write(training_config)

    model_config = config_path / f"gprof_nn_{config.lower()}_model.toml"
    model_config = open(model_config, "r").read()
    if config.lower() == "sim":
        sensor = sensors.get_sensor(sensor)
        n_chans = sensor.n_chans
        if isinstance(sensor, sensors.CrossTrackScanner):
            model_config = model_config.format(
                tb_sim_shape=f"[{sensor.n_angles}, {n_chans}]",
                tb_bias_shape=f"[{n_chans}]",
            )
        else:
            model_config = model_config.format(
                tb_sim_shape=f"[{n_chans}]",
                tb_bias_shape=f"[{n_chans}]",
            )

    with open(path / "model.toml", "w") as output:
        output.write(model_config)


@click.argument("sensor", type=str)
@click.argument("configuration", type=str)
@click.argument("training_data_path", type=str)
@click.argument("validation_data_path", type=str)
def init_cli(
        sensor: str,
        configuration: str,
        training_data_path: str,
        validation_data_path: str,
) -> int:
    """
    Initialize model directory.

    This comand initializes a directory with default model and training
    configuration files that can be used to train the GPROF-NN retrievals.
    """

    configuration = configuration.lower()
    if not configuration in ["1d", "3d", "sim"]:
        LOGGER.error(
            "'configuration' must be one of ['1d', '3d', 'sim']."
        )
        return 1

    training_data_path = Path(training_data_path)
    if not training_data_path.exists() or not training_data_path.is_dir():
        LOGGER.error(
            "'training_data_path' must point to an existing directory."
        )
        return 1

    validation_data_path = Path(validation_data_path)
    if not validation_data_path.exists() or not validation_data_path.is_dir():
        LOGGER.error(
            "'validation_data_path' must point to an existing directory."
        )
        return 1

    init(
        sensor,
        Path(os.getcwd()),
        configuration,
        training_data_path,
        validation_data_path,
        None,
        None
    )
