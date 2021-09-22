"""
Run legacy GPROF algorithm on input data in GPROF-NN 0D training
data format.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import tempfile

from rich.progress import track
import xarray as xr

import gprof_nn
from gprof_nn.data.training_data import (GPROF0DDataset,
                                         write_preprocessor_file)
from gprof_nn.data.retrieval import RetrievalFile
from gprof_nn.data.preprocessor import PreprocessorFile
from tqdm import tqdm

def reshape_data(dataset):
    """
    Reshape retrieval results to input data form with
    samples, scans and pixel dimensions.

    Args:
        dataset: 'xarray.Dataset' containing the retrieval results.

    Return:
        A copy of 'dataset' reshaped to containg samples of size 221 x 221.
    """
    n_pixels = 221
    n_scans = 221

    new_data = {}
    for v in dataset.variables:
        data = dataset[v].data
        new_shape = (-1, n_pixels, n_scans) + data.shape[2:]
        data = data.reshape(new_shape)
        dims = dataset.variables[v].dims
        new_data[v] = (("samples", ) + dims), data
    return xr.Dataset(new_data)

#
# Parse arguments
#

parser = argparse.ArgumentParser(
    description=(
        "Run legacy GPROF on GPROF-NN 0D input data and store results as"
        "NetCDF files."
        )
)
parser.add_argument('input_file', metavar="input_file", type=str, nargs=1,
                    help='The NetCDF4 file containing the input data.')
parser.add_argument('output_file', metavar="output_file", type=str, nargs=1,
                    help='The file to which to write the retrieval results.')
parser.add_argument('--template_file', metavar="template_file", type=str,
                    help='Preprocessor file to use as template..')
parser.add_argument('--profiles', action="store_true",
                    help='Flag to include profiles in output.')
args = parser.parse_args()
input_file = Path(args.input_file[0])
output_file = Path(args.output_file[0])
profiles = args.profiles
if args.template_file is None:
    template_file = None
else:
    template_file = PreprocessorFile(Path(args.template_file))

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

#
# Run retrieval.
#

for input_file, output_file in zip(input_files, output_files):
    preprocessor_file = None
    retrieval_file = None
    try:
        input_data = xr.open_dataset(input_file)
        results = []
        print(f"Starting processing of {input_file}.")
        print(input_data)
        chunk_size = 32
        i = 0
        n_samples = input_data.samples.size
        n_chunks = n_samples // chunk_size

        if profiles:
            profiles = "1"
        else:
            profiles = "0"
        print("PROFILES: ", profiles)

        if (n_samples % chunk_size) > 0:
            n_chunks += 1
        for i in track(range(n_chunks)):
            i_start = i * chunk_size
            i_end = i_start + chunk_size
            _, preprocessor_file = tempfile.mkstemp(dir="/gdata/simon/tmp")
            _, retrieval_file = tempfile.mkstemp(dir="/gdata/simon/tmp")
            print("Writing preprocessor file.")
            write_preprocessor_file(input_data[{"samples": slice(i_start, i_end)}],
                                    preprocessor_file,
                                    template=template_file)
            print("Running retrieval.")
            subprocess.run(["GPROF_2020_V1",
                            str(preprocessor_file),
                            str(retrieval_file),
                            "log",
                            "/qdata1/pbrown/gpm/ancillary/",
                            profiles])
            print("Storing results.")
            retrieval = RetrievalFile(retrieval_file, has_profiles=profiles).to_xarray_dataset()
            results += [reshape_data(retrieval)]
        results = xr.concat(results, "samples")
        results.to_netcdf(output_file)
    finally:
        if preprocessor_file is not None:
            Path(preprocessor_file).unlink()
        if retrieval_file is not None:
            Path(retrieval_file).unlink()
