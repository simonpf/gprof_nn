"""
===================================
gprof_nn.bin.extract_valiation_data
===================================

This sub-module implements the command line interface to extract
GPROF validation data.
"""
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'extract_data' command to top-level parser. This function
    is called from the top-level parser defined in 'gprof_nn.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "extract_validation_data",
        description=("Extract MRMS validation data from" "NASA Goddard servers."),
    )
    parser.add_argument(
        "sensor",
        metavar="sensor",
        type=str,
        help=("Name of the sensor for which to extract the" "validation data"),
    )
    parser.add_argument(
        "year",
        metavar="year",
        type=int,
        help=("The year for which to extract the " "validation data."),
    )
    parser.add_argument(
        "months",
        metavar="months",
        type=int,
        nargs="+",
        help=("The month for which to extract the " "validation data."),
    )
    parser.add_argument(
        "mrms_output",
        metavar="mrms_output",
        type=str,
        help="Folder to which to write the MRMS data.",
    )
    parser.add_argument(
        "preprocessor_output",
        metavar="preprocessor_output",
        type=str,
        help=("Folder to which to write the extracted " "preprocessor files."),
    )
    parser.add_argument(
        "--n_processes",
        metavar="N",
        type=int,
        help="How many processes to use to extract the validation data.",
        default=4
    )
    parser.set_defaults(func=run)


def run(args):
    """
    This function coordinates the  extraction of the validation data.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from gprof_nn import sensors
    from gprof_nn.data.validation import ValidationFileProcessor

    # Check sensor
    sensor = getattr(sensors, args.sensor.strip().upper(), None)
    if sensor is None:
        LOGGER.error("The sensor '%s' is not yet supported.", args.sensor)
        return 1

    year = args.year
    months = args.months
    if not year > 2000:
        LOGGER.error(f"Year must be > 2000 not '%s'.", year)
        return 1

    for month in args.months:
        if not (month > 0 and month < 13):
            LOGGER.error(f"Month must be within [1, 12] not '%s'.", year)
            return 1

        mrms_output = Path(args.mrms_output)
        pp_output = Path(args.preprocessor_output)
        n_procs = args.n_processes

        processor = ValidationFileProcessor(sensor, year, month)
        processor.run(mrms_output, pp_output, n_workers=n_procs)
