"""
=====================
gprof_nn.bin.retrieve
=====================

This sub-module implements the command line interface to apply the GPROF-NN
to input data.
"""
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
        help="Run a GPROF-NN retrieval.",
        description=(
            """
            Runs the GPROF-NN algorithm on a given input file.

            The input file may be a preprocessor file or a NetCDF4 file in
            the same format as the training data. This command can also be used
            to run a GPROF-NN simulator on training files. This
            will create new training data files with extended simulated TBs
            and biases.
            """
        ),
    )
    parser.add_argument(
        "model",
        metavar="model",
        type=str,
        help="The GPROF-NN model to use for the retrieval.",
    )
    parser.add_argument(
        "inputs",
        metavar="input",
        type=str,
        nargs="+",
        help=(
            """
        Single input file or a folder containing input files. The input
        must be a preprocessor file, a L1C file (in this case the
        preprocessor will be run automatically) or a NetCDF4 training
        data file. If a folder is given all files with suffixes
        '.pp', '.HDF5' and '.nc' will be processed.
        """
        ),
    )
    parser.add_argument(
        "output",
        metavar="output",
        type=str,
        help="Folder or file to which to write the output.",
    )
    parser.add_argument(
        "--gradients",
        action="store_true",
        help=(
            """
            If set, the gradients of the surface precipitation with
            respect to the inputs will be included in the output.
            """
        ),
    )
    parser.add_argument(
        "--no_profiles",
        action="store_true",
        help="If set, no profiles will be retrieved.",
    )
    parser.add_argument(
        "--format",
        type=str,
        help=(
            """
            The output file format. Should be 'GPROF_BINARY' or 'NETCDF'.
            """
        ),
    )
    parser.add_argument(
        "--sensor",
        type=str,
        metavar="sensor",
        help="Name of a sensor object to use to load training data.",
        default=None,
    )
    parser.add_argument(
        "--preserve_structure",
        action="store_true",
        help=(
            """
            If set and the input file is a training data file, the retrieval
            will be performed on a spatially coherent scene such as those
            used for the training of the GPROF-NN 3D retrieval.
            """
        ),
    )
    parser.add_argument(
        "--n_processes",
        metavar="n",
        type=int,
        default=4,
        help=(
            """
            When the retrieval is run on multiple input files, the processing
            is parallelized across input files. The 'n_processes' argument can
            be used to customize the number of processes used. Defaults to 4.
            """
        ),
    )
    parser.add_argument(
        "--device",
        metavar="name",
        type=str,
        default="cpu",
        help=(
            """
            Name of the device to run the retrieval on. 'cpu' or 'cuda:0',
            'cuda:1', ...
            """
        ),
    )
    parser.set_defaults(func=run)


def process_file(
    input_file,
    output_file,
    model,
    targets,
    gradients,
    device,
    log_queue,
    preserve_structure=False,
    fmt=None,
    sensor=None,
):
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
    logger = logging.getLogger(__name__)
    logger.info("Processing file %s.", input_file)
    xrnn = QRNN.load(model)
    if targets is not None:
        xrnn.set_targets(targets)
    driver = RetrievalDriver
    if gradients:
        driver = RetrievalGradientDriver
    retrieval = driver(
        input_file,
        xrnn,
        output_file=output_file,
        device=device,
        preserve_structure=preserve_structure,
        sensor=sensor,
        output_format=fmt,
        tile=False,
    )
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

    for input_file in inputs:
        if not input_file.exists():
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
                "If provided, sensor must be a valid sensor name not '%s'.", sensor_name
            )
            return 1
    else:
        sensor = None

    gradients = args.gradients
    n_procs = args.n_processes
    device = args.device
    fmt = args.format
    if device.startswith("cuda"):
        mp.set_start_method("spawn")

    # Find files and determine output names.
    input_files = []
    output_files = []
    for inp in inputs:
        if inp.is_dir():
            if output is None or not output.is_dir():
                LOGGER.error(
                    "If the input file is a directory, the 'output_file' argument "
                    "must point to a directory as well."
                )

            files = list(inp.glob("**/*.nc"))
            files += list(inp.glob("**/*.nc.gz"))
            files += list(inp.glob("**/*.nc.bin.gz"))
            files += list(inp.glob("**/*.pp"))
            files += list(inp.glob("**/*.HDF5"))
            input_files += files

            for f in files:
                of = f.relative_to(inp)
                if of.suffix in [".nc", ".HDF5"]:
                    of = of.with_suffix(".nc")
                elif of.suffix == ".gz":
                    of = of.with_suffix("")
                else:
                    of = of.with_suffix(".bin")
                output_files.append(output / of)
        else:
            input_files += [inp]
            output_files += [output]

    # Try to load the model.
    if args.no_profiles:
        targets = [t for t in ALL_TARGETS if t not in PROFILE_NAMES]
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
    for input_file, output_file in zip(input_files, output_files):
        tasks += [
            pool.submit(
                process_file,
                input_file,
                output_file,
                model,
                targets,
                gradients,
                device,
                log_queue,
                sensor=sensor,
                fmt=fmt,
                preserve_structure=preserve_structure,
            )
        ]

    for filename, task in track(
        list(zip(input_files, tasks)),
        description="Processing files:",
        console=gprof_nn.logging.get_console(),
    ):
        gprof_nn.logging.log_messages()
        try:
            task.result()
        except Exception as exc:
            LOGGER.exception(
                "The following error was encountered during the processing "
                "of file %s.",
                filename,
            )
    pool.shutdown()
    gprof_nn.logging.log_messages()
