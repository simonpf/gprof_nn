"""
=========================
gprof_nn.bin.extract_data
=========================

This sub-module implements the command line interface to extract training
data for different sensors from *.sim. files.
"""
import logging
import os
from pathlib import Path

import gprof_nn.logging
from gprof_nn.definitions import TRAINING_DAYS, VALIDATION_DAYS, TEST_DAYS

os.environ["OMP_NUM_THREADS"] = "1"

LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'extract_data' command to top-level parser. This function
    is called from the top-level parser defined in 'gprof_nn.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "extract_data",
        help="Generate training, validation or test data.",
        description="Extract training data from *.sim files.",
    )
    parser.add_argument(
        "sensor",
        metavar="sensor",
        type=str,
        help=("Name of the sensor for which to generate the" "training data"),
    )
    parser.add_argument(
        "configuration",
        metavar="configuration",
        type=str,
        help=("For which configuration to extract the training data:" " ERA5 or GANAL"),
    )
    parser.add_argument(
        "kind",
        metavar="kind",
        type=str,
        help="The type of data to extract: 'TRAIN', 'VAL', 'TEST' or 'ALL'",
    )
    parser.add_argument(
        "output",
        metavar="output",
        type=str,
        help="Folder to which to write the extracted data.",
    )
    parser.add_argument(
        "--era5_path",
        metavar="path",
        type=str,
        help="Folder to which to write the extracted data.",
        default="/qdata2/archive/ERA5",
    )
    parser.add_argument(
        "--no_seaice",
        action="store_true",
        help=(
            "Turns off the extraction of ERA5 collocations over sea ice."
        )
    )
    parser.add_argument(
        "--n_processes",
        metavar="n",
        type=int,
        default=4,
        help="The number of processes to use for the processing.",
    )
    parser.add_argument(
        "--high_resolution", action="store_true", help="Extract high-reslution data."
    )
    parser.add_argument(
        "--combined_path", type=str, default=None, help="Base path of GPM combined data"
    )
    parser.add_argument(
        "--slh_path", type=str, default=None, help="Base path of GPM SLH data"
    )
    parser.add_argument(
        "--sim_file_path",
        type=str,
        default=None,
        help=(
            "Overwrites the sensor's default file path for sim files."
        ),
    )
    parser.add_argument(
        "--mrms_file_path",
        type=str,
        default=None,
        help=(
            "Overwrites the sensor's default file path for MRMS matchups."
        ),
    )
    parser.add_argument(
        "--mrms_file_type",
        type=str,
        default=None,
        help=(
            "Overwrites the sensor's mrms_record_type with the record of "
            "the corresponding MRMS file type. Valid types are: CONICAL,"
            " CONICAL_CONST, XTRACK and SNOW."
        ),
    )
    parser.set_defaults(func=run)


def run(args):
    """
    This function implements the actual extraction of the data from the *.sim
    files.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from gprof_nn import sensors
    from gprof_nn.data.sim import SimFileProcessor
    from gprof_nn.data.combined import CombinedFileProcessor

    # Check sensor
    sensor = getattr(sensors, args.sensor.strip().upper(), None)
    if sensor is None:
        LOGGER.error("The sensor '%s' is not yet supported.", args.sensor)
        return 1

    # Check configuration
    config = args.configuration.lower().strip()
    if not config in ["era5", "ganal"]:
        LOGGER.error(
            "The configuration should be 'era5' or 'ganal' not '%s'.",
            args.configuration,
        )
        return 1

    # Check kind
    kind = args.kind.lower().strip()
    if not kind in ["training", "validation", "test", "all"]:
        LOGGER.error(
            "The kind should be 'training', 'validation', 'test' or 'all'"
            "not '%s'.", args.kind
        )
        return 1

    high_res = args.high_resolution
    combined_path = args.combined_path
    slh_path = args.slh_path
    if high_res and combined_path is None:
        LOGGER.error(
            "Root of the combined path must be provided if high-resolution"
            "data is to be extracted."
        )

    output = Path(args.output)
    if not output.exists() or not output.is_dir():
        LOGGER.error("The 'output' argument must point to a directory.")
        return 1

    # Sim file path
    if args.sim_file_path is not None:
        sf_path = args.sim_file_path
        if sf_path.lower() == "none":
            sensor.sim_file_path = None
        else:
            sensor.sim_file_path = args.sim_file_path

    # MRMS matchup file path
    if args.mrms_file_path is not None:
        sf_path = args.mrms_file_path
        if sf_path.lower() == "none":
            sensor.mrms_file_path = None
        else:
            sensor.mrms_file_path = sf_path

    # Optionally overwrite MRMS file type.
    if args.mrms_file_type is not None:
        from gprof_nn.data.types import get_mrms_file_record
        try:
            sensor.mrms_file_record = get_mrms_file_record(
                sensor.n_chans,
                sensor.n_angles,
                args.mrms_file_type
            )
        except ValueError as e:
            LOGGER.error("%s", e)
            return 1


    era5_path = args.era5_path
    if args.no_seaice:
        era5_path = None

    n_procs = args.n_processes

    if kind == "training":
        days = TRAINING_DAYS
    elif kind == "validation":
        days = VALIDATION_DAYS
    elif kind == "test":
        days = TEST_DAYS
    elif kind == "all":
        days = TRAINING_DAYS + VALIDATION_DAYS + TEST_DAYS

    LOGGER.info("Starting extraction of %s data.", kind)

    # Loop over days.
    for d in days:
        LOGGER.info("Pocessing day %s.", d)
        output_file = output / f"gprof_nn_{sensor.name.lower()}_{config}_{d:02}"
        if not high_res:
            processor = SimFileProcessor(
                output_file,
                sensor,
                config.upper(),
                era5_path=era5_path,
                n_workers=n_procs,
                day=d,
            )
        else:
            processor = CombinedFileProcessor(
                output_file, combined_path, slh_path, n_workers=n_procs, day=d
            )
        processor.run()
