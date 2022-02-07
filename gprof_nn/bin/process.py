"""
This file implements the 'gprof_nn 1d' and 'gprof_nn 2d' commands


"""
from functools import partial
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def add_parser(subparsers, mode):
    """
    Add parser for '1d' or '2d' commands to top-level parser. This function
    is called from the top-level parser defined in 'gprof_nn.bin'.

    Args: subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        mode,
        description=(
            """
            Runs the GPROF-NN algorithm on a given input file.

            The input file may be a preprocessor file or a NetCDF4 file in
            the same format as the training data.
            """
            )
    )
    parser.add_argument(
        'configuration',
        metavar="configuration",
        type=str,
        help="The retrieval configuration to run: 'ERA5' or 'GANAL'"
    )
    parser.add_argument(
        'input',
        metavar="input", type=str,
        help='The input file.'
    )
    parser.add_argument(
        '-o',
        '--output',
        metavar="output",
        type=str,
        help='File to which to write the retrieval results.'
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
    parser.set_defaults(func=partial(run, mode))


def process_file(
        kind,
        configuration,
        input_file,
        output_file=None
):
    from quantnn.qrnn import QRNN

    from gprof_nn.data import get_model_path
    from gprof_nn.data.l1c import L1CFile
    from gprof_nn.data.preprocessor import PreprocessorFile
    from gprof_nn.retrieval import RetrievalDriver

    # Need to open the file to figure out the sensor.
    try:
        if input_file.suffix in [".pp", ".bin"]:
            input_data = PreprocessorFile(input_file)
        elif input_file.suffix == ".HDF5":
            input_data = L1CFile(input_file)
        else:
            LOGGER.error(
                "Only input files with suffixes '.pp', '.bin', or '.HDF5'"
                " are supported."
            )
            return 1
    except ValueError as err:
        LOGGER.error("%s", err)
        return 1

    # Now try and find the model file for sensor and configuration.
    sensor = input_data.sensor
    try:
        model_path = get_model_path(kind, sensor, configuration)
    except Exception as e:
        LOGGER.error("%s", str(e))
        return 1

    # Finally, run the retrieval:
    model = QRNN.load(model_path)

    driver = RetrievalDriver(
        input_file,
        model,
        output_file=output_file
    )
    driver.run()

def run(kind, args):

    configuration = args.configuration.upper()
    if configuration not in ["ERA5", "GANAL"]:
        LOGGER.error(
            "Configuration should be one of 'ERA5' or 'GANAL', not '%s'",
            configuration
        )
        return 1

    input_file = Path(args.input)
    if not input_file.exists():
        LOGGER.error("The input file '%s' doesn't exist", input_file)
        return 1

    output_file = args.output
    if output_file is None:
        output_file = Path(".")

    process_file(kind, configuration, input_file, output_file=output_file)




