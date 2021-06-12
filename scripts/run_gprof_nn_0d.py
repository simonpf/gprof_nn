"""
Run GPROF-NN 0D algorithm on input data in GPROF-NN training
data format.
"""
import argparse
from pathlib import Path

from quantnn.qrnn import QRNN
from quantnn.normalizer import Normalizer
from gprof_nn.data.training_data import run_retrieval_0d

#
# Parse arguments
#

parser = argparse.ArgumentParser(
    description=(
        "Run GPROF-NN 0D on GPROF-NN 0D input data and store results as"
        "NetCDF files."
        )
)
parser.add_argument('model', metavar="model", type=str,
                    help="Stored quantnn model to use for the retrieval.")
parser.add_argument('normalizer', metavar="normalizer", type=str,
                    help="The normalizer to use to normalize the input data.")
parser.add_argument('input_file', metavar="input_file", type=str,
                    help='The NetCDF4 file containing the input data.')
parser.add_argument('output_file', metavar="output_file", type=str,
                    help='The file to which to write the retrieval results.')

args = parser.parse_args()
model = args.model
normalizer = args.normalizer
input_file = Path(args.input_file)
output_file = Path(args.output_file)

#
# Check inputs.
#

if input_file.is_dir():
    if not output_file.is_dir():
        raise ValueError(
            "If the input file is a directory, the output file must be a "
            "directory as well."
        )
    input_files = list(input_file.glob("*.nc"))
    output_files = [output_file / f.name for f in input_files]
else:
    input_files = [input_file]
    output_files = [output_file]

xrnn = QRNN.load(model)
normalizer = Normalizer.load(normalizer)
for input_file, output_file in zip(input_files,
                                   output_files):
    run_retrieval_0d(input_file,
                     xrnn,
                     normalizer,
                     output_file)
