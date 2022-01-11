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

import gprof_nn.logging
from gprof_nn.retrieval import RetrievalDriver, RetrievalGradientDriver
from gprof_nn import sensors
from gprof_nn import statistics


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
    parser.set_defaults(func=run)


STATS = {
    "training_1d": [
        statistics.TrainingDataStatistics(kind="1d"),
        statistics.ZonalDistribution(),
        statistics.GlobalDistribution()
    ],
    "training_3d": [
        statistics.TrainingDataStatistics(kind="3d"),
        statistics.ZonalDistribution(),
        statistics.GlobalDistribution()
    ],
    "bin": [
        statistics.BinFileStatistics(),
        statistics.GlobalDistribution()
    ],
    "observations": [statistics.ObservationStatistics()],
    "retrieval": [
        statistics.RetrievalStatistics(),
        statistics.ZonalDistribution(),
        statistics.GlobalDistribution(),
        statistics.ScanPositionMean()
        ],
    "combined": [statistics.GPMCMBStatistics(monthly=False)]
}

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

    kind = args.kind.lower().strip()
    if kind == "training":
        kind = "training_1d"

    if not kind in STATS.keys():
        LOGGER.error(
            f"'kind' argument must be one of {list(STATS.keys())}."
        )
        return 1

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

    stats = STATS[kind]
    processor = statistics.StatisticsProcessor(sensor, input_files, stats)
    processor.run(n_procs, output)
