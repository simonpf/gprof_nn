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

from quantnn.qrnn import QRNN
from quantnn.normalizer import Normalizer
from rich.progress import track

import gprof_nn.logging
from gprof_nn.retrieval import RetrievalDriver, RetrievalGradientDriver
from gprof_nn.definitions import ALL_TARGETS, PROFILE_NAMES


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
    parser.add_argument('--n_processes',
                        metavar="n",
                        type=int,
                        default=4,
                        help='The number of processes to use for the processing.')
    parser.set_defaults(func=run)


def process_file(input_file,
                 output_file,
                 profiles,
                 mode,
                 nedts,
                 log_queue):
    """
    Helper function for distributed processing.
    """
    gprof_nn.logging.configure_queue_logging(log_queue)

    LOGGER.info("Processing file %s.", input_file)

    if input_file.suffix == ".nc":
        results = run_gprof_training_data(input_file,
                                          mode,
                                          profiles,
                                          nedts=nedts)
    else:
        results = run_gprof_standard(input_file,
                                     mode,
                                     profiles,
                                     nedts=nedts)

    results.to_netcdf(output_file)


def run(args):
    """
    Run GPROF algorithm.

    Args:
        args: The namespace object provided by the top-level parser.
    """

    #
    # Check and load inputs.
    #

    input = Path(args.input)
    output = Path(args.output)

    if not input.exists():
        LOGGER.error("Input must be an existing file or folder.")

    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    if args.gradients:
        mode = "SENSITIVITY"
        if args.full_profiles:
            LOGGER.error(
                "Only one of the 'gradients' and 'full_profiles' flags may be"
                " set at a time."
            )
    elif args.full_profiles:
        mode = "PROFILES"
    else:
        mode = "STANDARD"

    profiles = args.profiles
    n_procs = args.n_processes

    nedts = None

    # Find files and determine output names.
    if input.is_dir():
        if output is None or not output.is_dir():
            LOGGER.error(
                "If the input file is a directory, the 'output_file' argument "
                "must point to a directory as well."
            )

        input_files = list(input.glob("**/*.nc"))
        input_files += list(input.glob("**/*.pp"))
        input_files += list(input.glob("**/*.HDF5"))

        output_files = []
        for f in input_files:
            of = f.relative_to(input)
            of = of.with_suffix(".nc")
            output_files.append(output / of)
    else:
        input_files = [input]
        output_files = [output]

    #
    # Run retrieval.
    #

    pool = ProcessPoolExecutor(max_workers=n_procs)
    log_queue = gprof_nn.logging.get_log_queue()
    tasks = []
    for input_file, output_file in (zip(input_files, output_files)):
        tasks += [pool.submit(process_file,
                              input_file,
                              output_file,
                              mode,

                              log_queue)]

    for t in track(tasks, description="Processing files:"):
        gprof_nn.logging.log_messages()
        t.result()

    pool.shutdown()
