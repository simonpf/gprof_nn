"""
Run GPROF-NN retrieval algorithm.
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


LOGGER = logging.getLogger(__name__)

#
# Parse arguments
#

parser = argparse.ArgumentParser(
    description=(
        "Runs the GPROF-NN algorithm on a given input file. The input file "
        "may be a preprocessor file or a NetCDF4 file in the same format "
        "as the training data."
        )
)
parser.add_argument('model', metavar="model", type=str,
                    help="Stored quantnn model to use for the retrieval.")
parser.add_argument('normalizer', metavar="normalizer", type=str,
                    help="The normalizer to use to normalize the input data.")
parser.add_argument('input_file', metavar="input_file", type=str,
                    help='The NetCDF4 file containing the input data.')
parser.add_argument('--output_file',
                    metavar="output_file",
                    type=str,
                    default=None,
                    help='The file to which to write the retrieval results.')
parser.add_argument('--gradients', action='store_true')
parser.add_argument('--n_processes',
                    metavar="n",
                    type=int,
                    default=4,
                    help='The number of processes to use for the processing.')


args = parser.parse_args()
model = args.model
normalizer = args.normalizer
input_file = Path(args.input_file)
output_file = args.output_file
gradients = args.gradients
n_procs = args.n_processes
if output_file is not None:
    output_file = Path(args.output_file)


#
# Check and load inputs.
#

if input_file.is_dir():
    if output_file is None or not output_file.is_dir():
        raise ValueError(
            "If the input file is a directory, the 'output_file' argument must"
            " point to a directory as well."
        )
    input_files = list(input_file.glob("**/*.nc"))
    input_files += list(input_file.glob("**/*.pp"))
    input_files += list(input_file.glob("**/*.HDF5"))

    output_files = []
    for f in input_files:
        of = f.relative_to(input_file)
        if of.suffix in [".nc", ".HDF5"]:
            of = of.with_suffix(".nc")
        else:
            of = of.with_suffix(".bin")
        output_files.append(output_file / of)
    print(output_files)
else:
    input_files = [input_file]
    output_files = [output_file]

xrnn = QRNN.load(model)
normalizer = Normalizer.load(normalizer)

#
# Run retrieval.
#


def process_file(input_file, output_file, gradients):
    """
    Helper function for distributed processing.
    """
    driver = RetrievalDriver
    if gradients:
        driver = RetrievalGradientDriver
    retrieval = driver(input_file,
                       normalizer,
                       xrnn,
                       output_file=output_file)
    result = retrieval.run()


pool = ProcessPoolExecutor(max_workers=n_procs)
tasks = []
for input_file, output_file in (zip(input_files, output_files)):
    tasks += [pool.submit(process_file, input_file, output_file, gradients)]

for t in track(tasks, description="Processing files:"):
    t.result()
pool.shutdown()
