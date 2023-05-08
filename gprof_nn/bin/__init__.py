"""
============
gprof_nn.bin
============

This sub-module implements the top-level 'gprof_nn' command line application.
Its task is to delegate the processing to the sub-commands that are defined
in the sub-module of the 'gprof_nn.bin' module.
"""
import argparse
import sys
import warnings

from gprof_nn import logging


def gprof_nn():
    """
    This function implements the top-level command line interface for the
    'gprof_nn' package. It serves as the global entry point to execute
    any of the available sub-commands.
    """
    from gprof_nn.bin import extract_data
    from gprof_nn.bin import retrieve
    from gprof_nn.bin import statistics
    from gprof_nn.bin import train
    from gprof_nn.bin import legacy
    from gprof_nn.bin import run_preprocessor
    from gprof_nn.bin import process
    from gprof_nn.bin import extract_validation_data
    from gprof_nn.bin import combine_validation_data
    from gprof_nn.bin import download_models

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    description = (
        "GPROF-NN: A neural-network based implementation of the Goddard "
        "Profiling algorithm."
    )

    parser = argparse.ArgumentParser(prog="gprof_nn", description=description)
    parser.add_argument(
        "--log_file",
        type=str,
        help="""
        File to write logging output to
        """,
        default=None
    )

    subparsers = parser.add_subparsers(help="Sub-commands")
    extract_data.add_parser(subparsers)
    train.add_parser(subparsers)
    retrieve.add_parser(subparsers)
    run_preprocessor.add_parser(subparsers)
    legacy.add_parser(subparsers)
    statistics.add_parser(subparsers)
    extract_validation_data.add_parser(subparsers)
    combine_validation_data.add_parser(subparsers)
    download_models.add_parser(subparsers)

    process.add_parser(subparsers, "1d")
    process.add_parser(subparsers, "3d")
    process.add_parser(subparsers, "hr")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    log_file = args.log_file
    if log_file is not None:
        logging.enable_file_logging(log_file)

    args.func(args)
