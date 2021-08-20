"""
=====================
gprof_nn.bin.retrieve
=====================

This sub-module implements the command line interface to apply the GPROF-NN
to input data.
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from quantnn.qrnn import QRNN
from quantnn.normalizer import Normalizer
from rich.progress import track

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
                        nargs="*",
                        "the input data.")
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
    "training": [
        statistics.TrainingDataStatistics(conditional=1),
        statistics.ZonalDistribution(),
        statistics.GlobalDistribution()
    ],
    "bin": [
        statistics.BinFileStatistics(),
        statistics.ZonalDistribution(),
        statistics.GlobalDistribution()
    ],
    "observations": [statistics.ObservationStatistics(conditional=1)]
}

ENDINGS = {
    "training": "**/*.nc",
    "bin": "**/*.bin",
    "observations": "**/*.nc"
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
            "Sensor '%s' is not supported."
        )
        return 1

    kind = args.kind.lower()
    if not kind in ["training", "bin", "observations"]:
        LOGGER.error(
            "'kind' argument must be one of 'training', 'bin' or"
            "'observations'."
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

    endings = ENDINGS[kind]

    input_files = []
    for path in inputs:
        input_files += list(path.glob(endings))

    stats = STATS[kind]
    processor = statistics.StatisticsProcessor(sensor, input_files, stats)
    processor.run(n_procs, output)
