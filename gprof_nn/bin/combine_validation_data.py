"""
====================================
gprof_nn.bin.combine_validation_data
====================================

This sub-module implements the command line interface to combine
validation results.
"""
import logging
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'extract_data' command to top-level parser. This function
    is called from the top-level parser defined in 'gprof_nn.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "combine_validation_data",
        description=("Combined retrieval results with validation data.")
    )
    parser.add_argument(
        "validation_data",
        metavar="path",
        type=str,
        help="Path to the validation data.",
    )
    parser.add_argument(
        "output_path",
        metavar="path",
        type=str,
        help="Path to write the results to.",
    )
    parser.add_argument(
        "--gprof_v5",
        metavar="path",
        type=str,
        default=None,
        help="Path to the results of the GPROF V5 retrieval.",
    )
    parser.add_argument(
        "--gprof_v7",
        metavar="path",
        type=str,
        help="Path to the results of the GPROF V7 retrieval.",
    )
    parser.add_argument(
        "--gprof_nn_1d",
        metavar="path",
        type=str,
        help="Path to the results of the GPROF-NN 1D retrieval.",
    )
    parser.add_argument(
        "--gprof_nn_3d",
        metavar="path",
        type=str,
        help="Path to the results of the GPROF-NN 3D retrieval.",
    )
    parser.add_argument(
        "--simulator",
        metavar="path",
        type=str,
        help="Path to the simulator files.",
    )
    parser.add_argument(
        "--start",
        metavar="date",
        type=str,
        help="Optional start time to limit range of overpass files.",
        default=None
    )
    parser.add_argument(
        "--end",
        metavar="date",
        type=str,
        help="Optional end time to limit range of overpass files.",
        default=None
    )
    parser.add_argument(
        "--n_processes",
        metavar="N",
        type=int,
        help="How many worker processes to use.",
        default=4,
    )
    parser.set_defaults(func=run)


def run(args):
    """
    This function coordinates the combination of the validation data.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from gprof_nn import sensors
    from gprof_nn.validation import (ResultCollector,
                                     GPROFNN1DResults,
                                     GPROFNN3DResults,
                                     GPROFResults,
                                     GPROFLegacyResults,
                                     SimulatorFiles)

    validation_path = Path(args.validation_data)
    if not validation_path.exists():
        LOGGER.error(
            "The provided path (%s) for the validation data doesn't exist.",
            validation_path
        )
    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    datasets = []

    # GPROF V5
    if args.gprof_v5 is not None:
        gprof_v5 = GPROFLegacyResults(args.gprof_v5)
        if len(gprof_v5) > 0:
            datasets.append(gprof_v5)
        LOGGER.info(f"Found %s GPROF V5 granules.", len(gprof_v5))

    # GPROF V7
    gprof_v7 = GPROFResults(args.gprof_v7)
    if len(gprof_v7) > 0:
        datasets.append(gprof_v7)
    LOGGER.info(f"Found %s GPROF V7 granules.", len(gprof_v7))

    # GPROF NN 1D
    gprof_nn_1d = GPROFNN1DResults(args.gprof_nn_1d)
    if len(gprof_nn_1d) > 0:
        datasets.append(gprof_nn_1d)
    LOGGER.info(f"Found %s GPROF-NN 1D granules.", len(gprof_nn_1d))

    # GPROF NN 3D
    gprof_nn_3d = GPROFNN3DResults(args.gprof_nn_3d)
    if len(gprof_nn_3d) > 0:
        datasets.append(gprof_nn_3d)
    LOGGER.info(f"Found %s GPROF-NN 3D granules.", len(gprof_nn_3d))

    # Simulator files
    if args.simulator is not None:
        simulator = SimulatorFiles(args.simulator)
        if len(simulator) > 0:
            datasets.append(simulator)
        LOGGER.info(f"Found %s simulator granules.", len(simulator))

    start = args.start
    if start is not None:
        start = np.datetime64(start)

    end = args.end
    if end is not None:
        end = np.datetime64(end)

    n_processes = args.n_processes

    collector = ResultCollector(validation_path, datasets)
    collector.run(output_path,
                  start=start,
                  end=end,
                  n_processes=n_processes)
