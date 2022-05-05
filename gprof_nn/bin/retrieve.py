"""
=====================
gprof_nn.bin.retrieve
=====================

This sub-module implements the command line interface to apply the GPROF-NN
to input data.
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from pathlib import Path

from quantnn.qrnn import QRNN
from rich.progress import track

import gprof_nn.logging
from gprof_nn import sensors
from gprof_nn.retrieval import RetrievalDriver, RetrievalGradientDriver
from gprof_nn.definitions import ALL_TARGETS, PROFILE_NAMES


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'retrieve' command to top-level parser. This function
    is called from the top-level parser defined in 'gprof_nn.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "retrieve",
        description=(
            """
            Runs the GPROF-NN algorithm on a given input file.

            The input file may be a preprocessor file or a NetCDF4 file in
            the same format as the training data. This command can also be used
            to run a simulator model on a training dataset in order to extend
            the field of view of the simulated observations.
            """
            )
    )
    parser.add_argument('model', metavar="model", type=str,
                        help="The GPROF-NN model to use for the retrieval.")
    parser.add_argument('inputs', metavar="input", type=str, nargs="+",
                        help='Folder or file containing the input data.')
    parser.add_argument('output',
                        metavar="output",
                        type=str,
                        help='Folder or file to which to write the output.')
    parser.add_argument('--gradients', action='store_true')
    parser.add_argument('--no_profiles', action='store_true')
    parser.add_argument('--format', type=str, help='Output file format')
    parser.add_argument(
        '--sensor',
        type=str,
        metavar="sensor",
        help="Name of a sensor object to use to load training data.",
        default=None
    )
    parser.add_argument(
        '--preserve_structure',
        action='store_true',
        help=("Whether or not to preserve the spatial structure"
              " of the 1D retrieval on training data.")
    )
    parser.add_argument(
        '--n_processes',
        metavar="n",
        type=int,
        default=4,
        help='The number of processes to use for the processing.'
    )
    parser.add_argument(
        '--device',
        metavar="name",
        type=str,
        default="cpu",
        help='Name of the PyTorch device to run the retrieval on.'
    )
    parser.set_defaults(func=run)


def process_file(input_file,
                 output_file,
                 model,
                 targets,
                 gradients,
                 device,
                 log_queue,
                 preserve_structure=False,
                 format=format,
                 sensor=None):
    """
    Process input file.

    Args:
        input_file: Path pointing to the input file.
        output_file: Path to the file to which to store the results.
        model: The GPROF-NN model with which to run the retrieval.
        targets: List of the targets to retrieve.
        gradients: Whether or not to do a special run to calculate
            gradients of the retrieval.
        device: The device on which to run the retrieval
        log_queue: Queue object to use for multi process logging.
    """
    gprof_nn.logging.configure_queue_logging(log_queue)

    LOGGER.info("Processing file %s.", input_file)
    xrnn = QRNN.load(model)
    if targets is not None:
        xrnn.set_targets(targets)
    driver = RetrievalDriver
    if gradients:
        driver = RetrievalGradientDriver
    retrieval = driver(input_file,
                       xrnn,
                       output_file=output_file,
                       device=device,
                       preserve_structure=preserve_structure,
                       sensor=sensor,
                       output_format=format,
                       tile=True)
    retrieval.run()


def run(args):
    """
    Run GPROF-NN algorithm.

    Args:
        args: The namespace object provided by the top-level parser.
    """

    #
    # Check and load inputs.
    #

    model = Path(args.model)
    if not model.exists():
        LOGGER.error("Given model is not an existing file.")
        return 1

    inputs = list(map(Path, args.inputs))
    output = Path(args.output)

    for input in inputs:
        if not input.exists():
            LOGGER.error("Input must be an existing file or folder.")
            return 1

    if len(inputs) == 1 and not inputs[0].is_dir() and not output.exists():
        if not output.suffix:
            output.mkdir(parents=True, exist_ok=True)

    preserve_structure = args.preserve_structure

    sensor_name = args.sensor
    if sensor_name is not None:
        try:
            sensor = sensors.get_sensor(sensor_name)
        except KeyError:
            LOGGER.error(
                "If provided, sensor must be a valid sensor name not '%s'.",
                sensor_name
            )
            return 1
    else:
        sensor = None

    gradients = args.gradients
    n_procs = args.n_processes
    device = args.device
    format = args.format
    if device.startswith("cuda"):
        mp.set_start_method("spawn")

    # Find files and determine output names.
    input_files = []
    output_files = []
    for input in inputs:
        if input.is_dir():
            if output is None or not output.is_dir():
                LOGGER.error(
                    "If the input file is a directory, the 'output_file' argument "
                    "must point to a directory as well."
                )

            files = list(input.glob("**/*.nc"))
            files += list(input.glob("**/*.nc.gz"))
            files += list(input.glob("**/*.nc.bin.gz"))
            files += list(input.glob("**/*.pp"))
            files += list(input.glob("**/*.HDF5"))
            input_files += files

            for f in files:
                of = f.relative_to(input)
                if of.suffix in [".nc", ".HDF5"]:
                    of = of.with_suffix(".nc")
                elif of.suffix == ".gz":
                    of = of.with_suffix("")
                else:
                    of = of.with_suffix(".bin")
                output_files.append(output / of)
        else:
            input_files += [input]
            output_files += [output]

    # Try to load the model.
    xrnn = QRNN.load(model)
    if args.no_profiles:
        targets = [t for t in ALL_TARGETS if not t in PROFILE_NAMES]
    else:
        targets = None

    #
    # Run retrieval.
    #

    if args.gradients:
        pool = ThreadPoolExecutor(max_workers=n_procs)
    else:
        pool = ProcessPoolExecutor(max_workers=n_procs)

    log_queue = gprof_nn.logging.get_log_queue()
    tasks = []
    for input_file, output_file in (zip(input_files, output_files)):
        tasks += [pool.submit(process_file,
                              input_file,
                              output_file,
                              model,
                              targets,
                              gradients,
                              device,
                              log_queue,
                              sensor=sensor,
                              format=format,
                              preserve_structure=preserve_structure)]

    for filename, task in track(list(zip(input_files, tasks)),
                                description="Processing files:"):
        gprof_nn.logging.log_messages()
        try:
            task.result()
        except Exception as e:
            LOGGER.error(
                "The following error was encountered during the processing "
                "of file %s:\n %s",
                filename,
                e
            )
    pool.shutdown()
