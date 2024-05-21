"""
gprof_nn.training
=================

Interface for training of the GPROF-NN retrievals.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from pytorch_retrieve.config import InferenceConfig
import toml

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


def load_inference_config(
        config: str,
        output_config: Dict[str, Any],
        ancillary: bool = True) -> InferenceConfig:
    """
    Load inference configuration for GPROF-NN retrieval.

    Args:
        config: A string specifying the retrieval configuration.
        model_config: The model configuration specifying the retrieval model.
        ancillary: A bool specifying whether the model requires ancillary data.

    Return:
        An inference config object specifying the inference settings for the
        GPROF-NN retrievals.
    """
    config_file_path = (
        Path(__file__).parent / "config_files" /
        f"gprof_nn_{config.lower()}_inference.toml"
    )
    config_str = open(config_file_path, "r").read()
    config_str = config_str.format(
        ancillary="true" if ancillary else "false"
    )
    inference_config = toml.loads(config_str)
    inference_config = InferenceConfig.parse(
        output_config,
        inference_config
    )
    return inference_config


@click.argument(
    "sensor",
    type=str,
)
@click.argument(
    "configuration",
    type=click.Choice(["1d", "3d"])
)
@click.argument(
    "training_data_path",
    type=str
)
@click.argument(
    "validation_data_path",
    type=str
)
def init_cli(
        sensor: str,
        configuration: str,
        training_data_path: str,
        validation_data_path: str,
) -> int:
    """
    Create training configurations files in the current working directory
    for training GPROF-NN 1D/3D retrievals for the sensor SENSOR using
    the training and validation data located in TRAININ_DATA_PATH and
    VALIDATION_DATA_PATH, respectively.
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


@click.option(
    "--model_path",
    default=None,
    help="The model directory. Defaults to the current working directory",
)
@click.option(
    "--model_config",
    default=None,
    help=(
        "Path to the model config file. If not provided, pytorch_retrieve "
        " will look for a 'model.toml' or 'model.yaml' file in the current "
        " directory."
    ),
)
@click.option(
    "--training_config",
    default=None,
    help=(
        "Path to the training config file. If not provided, pytorch_retrieve "
        " will look for a 'training.toml' or 'training.yaml' file in the current "
        " directory."
    ),
)
@click.option(
    "--compute_config",
    default=None,
    help=(
        "Path to the compute config file defining the compute environment for "
        " the training."
    ),
)
@click.option(
    "--resume",
    "-r",
    "resume",
    is_flag=True,
    default=False,
    help=("If set, training will continue from a checkpoint file if available."),
)
def run_cli(
    model_path: Optional[Path],
    model_config: Optional[Path],
    training_config: Optional[Path],
    compute_config: Optional[Path],
    resume: bool = False,
) -> int:
    """
    Train retrieval model.

    This command runs the training of the retrieval model specified by the
    model and training configuration files.

    """
    from pytorch_retrieve import load_model
    from pytorch_retrieve.lightning import LightningRetrieval
    from pytorch_retrieve.architectures import compile_architecture, MLP
    from pytorch_retrieve.training import parse_training_config, run_training
    from pytorch_retrieve.eda import run_eda
    from pytorch_retrieve.utils import (
        read_model_config,
        read_training_config,
        read_compute_config,
        find_most_recent_checkpoint
    )
    from pytorch_retrieve.config import (
        InputConfig,
        OutputConfig,
        ComputeConfig,
    )

    if model_path is None:
        model_path = Path(".")

    LOGGER = logging.getLogger(__name__)
    model_config = read_model_config(LOGGER, model_path, model_config)
    module_name = None
    if "name" in model_config:
        module_name = model_config["name"]

    training_config = read_training_config(LOGGER, model_path, training_config)
    if training_config is None:
        LOGGER.error(
            "Failed parsing the training configuration."
        )
        return 1
    training_schedule = parse_training_config(training_config)

    compute_config = read_compute_config(LOGGER, model_path, compute_config)
    if compute_config is not None:
        compute_config = ComputeConfig.parse(compute_config)

    stats_path = model_path / "stats"
    if not stats_path.exists():
        LOGGER.info(
            "Running EDA because current model directory does not contain a stats "
            "directory."
        )
        input_configs = {
            name: InputConfig.parse(name, cfg)
            for name, cfg in model_config["input"].items()
        }
        output_configs = {
            name: OutputConfig.parse(name, cfg)
            for name, cfg in model_config["output"].items()
        }
        run_eda(
            stats_path,
            input_configs,
            output_configs,
            training_schedule["stage_1"],
            compute_config=compute_config
        )

    if model_config is None:
        return 1
    retrieval_model = compile_architecture(model_config)

    module = LightningRetrieval(
        retrieval_model,
        training_schedule=training_schedule,
        name=module_name
    )


    checkpoint = None
    if resume:
        checkpoint = find_most_recent_checkpoint(
            model_path / "checkpoints", module.name
        )

    model_path = run_training(
        model_path,
        module,
        compute_config=compute_config,
        checkpoint=checkpoint,
    )

    # Set inference config
    model = load_model(model_path)
    if isinstance(model, MLP):
        config = "1d"
    else:
        config = "3d"
    output_configs = {
            name: OutputConfig.parse(name, cfg)
            for name, cfg in model_config["output"].items()
    }
    model.inference_config = load_inference_config(
        config,
        output_configs
    )
    model.save(model_path)
