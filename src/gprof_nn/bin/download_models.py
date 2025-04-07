"""
============================
gprof_nn.bin.download_models
============================

This sub-module implements the command line interface to download the
neural-network models for the GPROF-NN retrieval.
"""
import logging
import os
from pathlib import Path

import gprof_nn.logging


LOGGER = logging.getLogger(__name__)

ALL_SENSORS = [
    "GMI",
    "TMIPO",
    "TMIPR",
    "SSMI",
    "SSMIS",
    "AMSRE",
    "AMSR2",
    "MHS",
    "ATMS"
]

ALL_CONFIGS = [
    "ERA5"
]


def add_parser(subparsers):
    """
    Add parser for 'download_models' command to the top-level parser. This
    function is called from the top-level parser defined in 'gprof_nn.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "download_models",
        help="Download neural-network models.",
        description="Downloads neural-network models to given destination.",
    )
    parser.add_argument(
        "path",
        metavar="path",
        type=str,
        help=(
            "Path to the directory in which to store the neural network "
            "models."
        ),
    )
    parser.add_argument(
        "--sensors",
        metavar="sensor",
        type=str,
        nargs="*",
        default=ALL_SENSORS,
        help=(
            "List of sensor names for which to download the models."
            " If not given, all available models will be downloaded."
        ),
    )
    parser.add_argument(
        "--configurations",
        metavar="conf",
        type=str,
        nargs="*",
        default=ALL_CONFIGS,
        help=(
            "List of sensor names for which to download the models."
            " If not given, all available models will be downloaded."
        ),
    )
    parser.set_defaults(func=run)


def run(args):
    """
    Downloads the requested models.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    path = Path(args.path)
    if not (path.is_dir() and path.exists()):
        LOGGER.error(
            "Provided path '%s' must point to an existing directory.",
            path
        )
    (path / "models").mkdir(exist_ok=True)
    (path / "profiles").mkdir(exist_ok=True)
    gprof_nn.data.get_profile_clusters()

    sensors = args.sensors
    if not all([sensor in ALL_SENSORS for sensor in sensors]):
        LOGGER.error(
            "All provided sensors names must be one of %s",
            ALL_SENSORS
        )
        return 1

    configs = args.configurations
    if not all([config in ALL_CONFIGS for config in configs]):
        LOGGER.error(
            "All provided configuration names must be one of %s",
            ALL_CONFIGS
        )
        return 1

    gprof_nn.data._DATA_DIR = path

    for sensor in sensors:
        for config in configs:
            sensor = getattr(gprof_nn.sensors, sensor)
            for kind in ["1D", "3D"]:
                path = gprof_nn.data.get_model_path(kind, sensor, config)
                LOGGER.info(
                    "Downloaded retrieval model for sensor '%s', retrieval"
                    " kind '%s', and configuration '%s' to '%s'.",
                    str(sensor.name),
                    kind,
                    config,
                    str(path)
                )

