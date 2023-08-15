"""
===================
gprof_nn.bin.legacy
===================

This sub-module implements the command line interface to run the legacy
GPROF algorithm.
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rich.progress import track
import pandas as pd

import gprof_nn.logging
from gprof_nn import sensors
from gprof_nn.legacy import (run_gprof_training_data,
                             run_gprof_standard)
from gprof_nn.definitions import CONFIGURATIONS


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'legacy' command to top-level parser. This function
    is called from the top-level parser defined in 'gprof_nn.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "legacy",
        description=(
            """
            Run the (legacy) GPROF algorithm on given input.

            The input file may be a preprocessor file or a NetCDF4 file in
            the same format as the training data.
            """
            )
    )
    parser.add_argument('sensor', metavar="sensor", type=str,
                        help='Name of the sensor for which to run GPROF.')
    parser.add_argument('configuration', metavar="[ERA5/GANAL]", type=str,
                        help='Which configuration of GPROF to run.')
    parser.add_argument('input', metavar="input", type=str,
                        help='Folder or file containing the input data.')
    parser.add_argument('output',
                        metavar="output",
                        type=str,
                        help='Folder or file to which to write the output.')
    parser.add_argument('--gradients',
                        action='store_true',
                        help='Whether to include gradients in the results.')
    parser.add_argument('--profiles',
                        action='store_true',
                        help="Whether to also retrieval profiles.")
    parser.add_argument('--full_profiles',
                        action='store_true',
                        help="Whether to include full profiles in the results.")
    parser.add_argument('--preserve_structure',
                        action='store_true',
                        help="Whether or not to preserve the spatial structure.")
    parser.add_argument('--n_processes',
                        metavar="n",
                        type=int,
                        default=4,
                        help='The number of processes to use for the processing.')
    parser.add_argument('--ancillary_path',
                        metavar="path",
                        type=str,
                        default=None,
                        help=("Ancillary data path that GPROF will be invoked "
                              "with."))
    parser.set_defaults(func=run)


def process_file(sensor,
                 configuration,
                 input_file,
                 output_file,
                 profiles,
                 log_queue,
                 mode="STANDARD",
                 nedts=None,
                 preserve_structure=False,
                 ancillary_path=None):
    """
    Helper function for distributed processing.

    Args:
        sensor: The senor object for which to run the retrieval.
        input_file: The input file to process.
        output_file: Path to the output file to which ro write the
            results.
        log_queue: Queue object to use for logging.
        mode: The retrieval mode.
        nedts: The retrieval errors to assume.
        preserve_structure: If True the spatial structure of the inputs
            well be preserved.
    """
    gprof_nn.logging.configure_queue_logging(log_queue)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Processing file %s. %s", input_file, output_file)

    if input_file.suffix in [".gz", ".nc"]:
        results = run_gprof_training_data(
            sensor,
            configuration,
            input_file,
            mode,
            profiles,
            nedts=nedts,
            preserve_structure=preserve_structure,
            ancillary_path=ancillary_path
        )
    else:
        results = run_gprof_standard(
            sensor,
            configuration,
            input_file,
            mode,
            profiles,
            nedts=nedts,
            ancillary_path=ancillary_path
        )

    if results is not None:
        if not output_file.parent.exists():
            output_file.parent.mkdir(exist_ok=True, parents=True)

        results.to_netcdf(str(output_file.absolute()))


def run(args):
    """
    Run GPROF algorithm.

    Args:
        args: The namespace object provided by the top-level parser.
    """

    #
    # Check and load inputs.
    #

    sensor = args.sensor
    sensor = sensor.strip().upper()
    sensor = getattr(sensors, sensor, None)
    if sensor is None:
        LOGGER.error(
            "Sensor '%s' is not supported.",
            args.sensor.strip().upper()
        )
        return 1

    configuration = args.configuration
    configuration = configuration.strip().upper()
    if configuration.upper() not in CONFIGURATIONS:
        LOGGER.error(
            "'configuration' should be one of %s.",
            CONFIGURATIONS
        )
        return 1


    input = Path(args.input)
    output = Path(args.output)

    if not input.exists():
        LOGGER.error("Input must be an existing file or folder.")
        return 1

    if input.is_dir() and not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    if args.gradients:
        mode = "SENSITIVITY"
        if args.full_profiles:
            LOGGER.error(
                "Only one of the 'gradients' and 'full_profiles' flags may be"
                " set at a time."
            )
            return 1
    elif args.full_profiles:
        mode = "PROFILES"
    else:
        mode = "STANDARD"

    preserve_structure = args.preserve_structure

    profiles = args.profiles
    n_procs = args.n_processes

    nedts = None

    ancillary_path = args.ancillary_path
    if ancillary_path is not None:
        ancillary_path = Path(ancillary_path)
        if not ancillary_path.exists():
            LOGGER.error(
                "If ancillary path is provided it must point to an existing "
                "directory."
            )
            return 1
        ancillary_path = str(ancillary_path) + "/"

    # Find files and determine output names.
    if input.is_dir():
        if output is None or not output.is_dir():
            LOGGER.error(
                "If the input file is a directory, the 'output_file' argument "
                "must point to a directory as well."
            )
            return 1

        input_files = list(input.glob("**/*.nc"))
        input_files += list(input.glob("**/*.nc.gz"))
        input_files += list(input.glob("**/*.pp"))
        input_files += list(input.glob("**/*.HDF5"))

        output_files = []
        for f in input_files:
            of = f.relative_to(input)
            if of.suffix == ".gz":
                of = of.with_suffix("")
            if of.suffix in [".pp", ".bin"]:
                of = of.with_suffix(".nc")
            output_files.append(output / of)
    else:
        input_files = [input]
        if output.is_dir():
            filename = input.name
            if filename.endswith(".gz"):
                filename = filename[:-3]
            output_files = [output / filename]
        else:
            output_files = [output]

    #
    # Run retrieval.
    #

    gprof_nn.logging.LOGGER.addHandler(gprof_nn.logging._HANDLER)
    pool = ProcessPoolExecutor(max_workers=n_procs)
    log_queue = gprof_nn.logging.get_log_queue()
    tasks = []
    for input_file, output_file in (zip(input_files, output_files)):
        tasks += [pool.submit(
            process_file,
            sensor, configuration, input_file, output_file, profiles, log_queue,
            mode=mode, nedts=None, preserve_structure=preserve_structure,
            ancillary_path=ancillary_path
        )]

    for t in track(tasks, description="Processing files:"):
        gprof_nn.logging.log_messages()
        t.result()
    pool.shutdown()
    gprof_nn.logging.log_messages()
