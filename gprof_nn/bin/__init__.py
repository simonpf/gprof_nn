"""
============
gprof_nn.bin
============

This sub-module implements the top-level 'gprof_nn' command line applitcation.
The top-level implementation delegates the processing to the sub-commands
that are defined in the sub-module of the 'gprof_nn.bin' module.
"""
import argparse
import sys
import warnings

import gprof_nn.logging
from gprof_nn.logging import set_log_level


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

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_log_level("INFO")

    description = (
        "GPROF-NN: A neural-network based implementation of the Goddard "
        "Profiling algorithm."
    )

    parser = argparse.ArgumentParser(
            prog='gprof_nn',
            description=description
            )

    subparsers = parser.add_subparsers(help='Sub-commands')

    extract_data.add_parser(subparsers)
    retrieve.add_parser(subparsers)
    statistics.add_parser(subparsers)
    train.add_parser(subparsers)
    legacy.add_parser(subparsers)
    run_preprocessor.add_parser(subparsers)

    process.add_parser(subparsers, "1d")
    process.add_parser(subparsers, "3d")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()
    args.func(args)


