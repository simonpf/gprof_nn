"""
=============================
gprof_nn.bin.run_preprocessor
=============================

This module implements a CLI interface to run the preprocessor on a
range of L1C files.
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
    Add parser for 'run_preprocessor' command to top-level parser. This
    function should be called from the top-level parser defined in
    'gprof_nn.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "run_preprocessor",
        description=(
            """
            Run preprocessor on L1C file and store results as NetCDF dataset.
            """
        ),
    )
    parser.add_argument(
        "sensor",
        metavar="sensor",
        type=str,
        help="The sensor corresponding to the data.",
    )
    parser.add_argument(
        "configuration",
        metavar="sensor",
        type=str,
        help="The sensor corresponding to the data.",
    )
    parser.add_argument(
        "input",
        metavar="input",
        type=str,
        help="The path to the directory tree containing" "the input data.",
        nargs="*",
    )
    parser.add_argument(
        "output",
        metavar="output",
        type=str,
        help="Path to the folder to which to write the " "results.",
    )
    parser.add_argument(
        "--n_processes",
        metavar="n",
        type=int,
        default=4,
        help="The number of processes to use for the processing.",
    )
    parser.set_defaults(func=run)


def process_file(input_file, sensor, configuration, output, log_queue):
    """
    Run preprocessor on given input file and store results as
    NetCDF files in a given output directory.

    Args:
        input_file: The path pointing to the input file.
        sensor: Sensor object representing the sensor for which
            to run the preprocessor.
        configuration: Which configuration of the preprocessor to
            run.
        destination: The folder to which to store the results using
            the name of the input file with the suffix replaced by
            '.nc'.
        log_queue: Queue to handle the logging from sub processes.
    """
    from gprof_nn.data.preprocessor import run_preprocessor

    input_file = Path(input_file)
    output = Path(output)

    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)

    # If output file ends in .pp, don't convert to NetCDF.
    if output.suffix == ".pp":
        run_preprocessor(
            input_file, sensor, configuration=configuration, output_file=output
        )
        return None

    # Else store output in NetCDF format.
    data = run_preprocessor(input_file, sensor, configuration=configuration)
    data.to_netcdf(output)


def run(args):
    """
    Run preprocessor.

    Args:
        args: The namespace object provided by the top-level parser.
    """

    #
    # Check and load inputs.
    #

    sensor = getattr(sensors, args.sensor.upper(), None)
    if sensor is None:
        LOGGER.error("Sensor '%s' is not supported.", args.sensor)
        return 1

    configuration = args.configuration.upper()
    if not configuration in ["ERA5", "GANAL"]:
        LOGGER.error("Configuration must be one of 'ERA5' or 'GANAL'.")
        return 1

    inputs = [Path(f) for f in args.input]
    for path in inputs:
        if not path.exists():
            LOGGER.error("The given input path '%s' doesn't exist", path)
            return 1

    output = Path(args.output)
    if output.suffix == "" and not output.exists():
        LOGGER.error("The given output path '%s' doesn't exist", output)
        return 1

    n_procs = args.n_processes
    pool = ProcessPoolExecutor(max_workers=n_procs)

    input_files = []
    output_files = []

    for path in inputs:
        if path.is_dir():
            files = list(path.glob("**/*.HDF5"))
            input_files += files
            output_files += [
                output / (str(f.relative_to(path))[:-4] + "nc") for f in files
            ]
        else:
            input_files.append(path)
            if output.is_dir():
                output_files.append(output / (path.stem + ".nc"))
            else:
                output_files.append(output)

    tasks = []
    log_queue = gprof_nn.logging.get_log_queue()
    for f, o in zip(input_files, output_files):
        tasks.append(pool.submit(process_file, f, sensor, configuration, o, log_queue))

    for f, t in track(
        list(zip(input_files, tasks)),
        description="Processing files:",
        console=gprof_nn.logging.get_console(),
    ):
        gprof_nn.logging.log_messages()
        try:
            t.result()
        except Exception as e:
            LOGGER.error(
                "Processing of file '%s' failed with the following" "exception:  %s",
                f,
                e,
            )
