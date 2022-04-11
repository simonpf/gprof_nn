"""
=======================
gprof_nn.bin.statistics
=======================

This sub-module implements the 'calculate_statistics' sub-command of the
'gprof_nn' command line application, which can be used to calculate
relevant statistics from training and retrieval data.
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import gprof_nn.logging
from gprof_nn.retrieval import RetrievalDriver, RetrievalGradientDriver
from gprof_nn import sensors


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'statistics' command to top-level parser. This function
    is called from the top-level parser defined in 'gprof_nn.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "calculate_statistics",
        description=(
            """
            Calculates relevant training data statistics.

            This command can be used to calculate statistics of retrieval input
            and output data of the GPROF-NN algorithm.
            """
            )
    )
    parser.add_argument('sensor', metavar="sensor", type=str,
                        help="The sensor corresponding to the data.")
    parser.add_argument('kind', metavar="kind", type=str,
                        help="The type of statistics to calculate.")
    parser.add_argument('input', metavar="input", type=str,
                        help="The path to the directory tree containing"
                        "the input data.",
                        nargs="*")
    parser.add_argument('output', metavar="output", type=str,
                        help="Path to the folder to which to write the "
                        "results.")
    parser.add_argument('--n_processes',
                        metavar="n",
                        type=int,
                        default=4,
                        help='The number of processes to use for the processing.')
    parser.add_argument("--no_correction",
                        help="Disable TB correction of sensor.",
                        action="store_true")
    parser.add_argument("--no_resampling",
                        help="Disable latitude resampling of sensor.",
                        action="store_true")
    parser.add_argument("--latitude_ratios",
                        help=("A latitude-time probability ratio file to use"
                              " for resampling the training data"),
                        type=str,
                        default=None)
    parser.set_defaults(func=run)


def get_stats(kind, latitude_ratios):
    """
    Provide list of statistic types for given statistics type.

    Args:
        kind: The kind of statistics to be computed
        latitude_ratios: Optional array with latitude or latitude-time
            ratios to use to resample the observations.

    Return:
        A list with statistics object to calculate the requested
        statistics type.
    """
    from gprof_nn import statistics
    if kind == "training_1d":
        stats = [
            statistics.TrainingDataStatistics(kind="1d"),
            statistics.ZonalDistribution(),
            statistics.GlobalDistribution()
        ]
    elif kind == "training_3d":
        stats = [
            statistics.TrainingDataStatistics(kind="3d"),
            statistics.ZonalDistribution(),
            statistics.GlobalDistribution()
        ]
    elif kind == "bin":
        stats = [
            statistics.BinFileStatistics(),
            statistics.GlobalDistribution()
        ]
    elif kind == "observations":
        stats = [statistics.ObservationStatistics(statistics=latitude_ratios)]
    elif kind == "retrieval":
        stats = [
            statistics.RetrievalStatistics(statistics=latitude_ratios),
            statistics.ZonalDistribution(statistics=latitude_ratios),
            statistics.GlobalDistribution(),
            statistics.ScanPositionMean()
        ]
    elif kind == "combined":
        stats = [statistics.GPMCMBStatistics(monthly=False)]
    else:
        LOGGER.error(
            "Kind must be one of ['training_1d', 'training_3d', "
            "'bin', 'observations', 'retrieval' 'combined'] not"
            "%s", kind
        )
        return None
    return stats


PATTERNS = {
    "training_1d": "**/*.nc*",
    "training_3d": "**/*.nc*",
    "bin": "**/*.bin",
    "observations": "**/*.nc",
    "retrieval": "**/*.nc.gz",
    "combined": "**/2B.GPM*.HDF5"
}


def run(args):
    """
    Calculate statistics.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from gprof_nn import statistics

    #
    # Check and load inputs.
    #

    sensor = getattr(sensors, args.sensor.upper(), None)
    if sensor is None:
        LOGGER.error(
            "Sensor '%s' is not supported.",
            args.sensor
        )
        return 1
    if args.no_resampling:
        sensor._latitude_ratios = None
    if args.no_correction:
        sensor._correction = None

    latitude_ratios = None
    if args.latitude_ratios is not None:
        sensor._latitude_ratios = args.latitude_ratios
        latitude_ratios = np.load(args.latitude_ratios)

    kind = args.kind.lower().strip()
    if kind == "training":
        kind = "training_1d"

    inputs = [Path(f) for f in args.input]
    for path in inputs:
        if not path.exists():
            LOGGER.error(
                "The given input path '%s' doesn't exist", path
            )
            return 1

    output = Path(args.output)
    if not output.exists():
        LOGGER.error(
            "The given output path '%s' doesn't exist", output
        )
        return 1

    n_procs = args.n_processes

    endings = PATTERNS[kind]

    input_files = []
    for path in inputs:
        input_files += list(path.glob(endings))

    stats = get_stats(kind, latitude_ratios)
    if stats is None:
        return 1

    processor = statistics.StatisticsProcessor(sensor, input_files, stats)
    processor.run(n_procs, output)
