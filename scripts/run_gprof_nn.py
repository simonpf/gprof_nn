"""
Run GPROF-NN retrieval algorithm.
"""
import argparse
from pathlib import Path

from quantnn.qrnn import QRNN
from quantnn.normalizer import Normalizer
from gprof_nn.retrieval import RetrievalDriver

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

args = parser.parse_args()
model = args.model
normalizer = args.normalizer
input_file = Path(args.input_file)
output_file = args.output_file
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
    input_files = list(input_file.glob("*.nc"))
    output_files = [output_file / f.name for f in input_files]
else:
    input_files = [input_file]
    output_files = [output_file]

xrnn = QRNN.load(model)
normalizer = Normalizer.load(normalizer)

#
# Run retrieval.
#

for input_file, output_file in zip(input_files,
                                   output_files):
    retrieval = RetrievalDriver(input_file,
                                normalizer,
                                xrnn,
                                output_file=output_file)
    retrieval.run_retrieval()
