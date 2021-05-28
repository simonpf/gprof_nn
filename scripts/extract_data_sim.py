"""
Extracts training data from GPROF GMI bin files.
"""
import argparse
from gprof_nn.data.bin import FileProcessor

# Parse arguments
parser = argparse.ArgumentParser(description="Extract data from GPROF files.")
parser.add_argument('input_path', metavar='path', type=str, nargs=1,
                    help='Path to folder containing input data.')
parser.add_argument('output_file', metavar="output_file", type=str, nargs=1,
                    help='Filename to store extracted data to.')
parser.add_argument('--start', metavar='start_fraction', type=float, nargs=1,
                    help='Fractional start of sample range to extract from each bin file.')
parser.add_argument('--end', metavar='start_fraction', type=float, nargs=1,
                    help='Fractional end of sample range to extract from each bin file.')
parser.add_argument('-n', metavar='n_procs', type=int, nargs=1,
                    help='Number of processes to use.')
args = parser.parse_args()
input_path = args.input_path[0]
output_file = args.output_file[0]
start = args.start[0]
end = args.end[0]
n  = args.n[0]

# Run processing.
processor = FileProcessor(input_path, include_profiles=True)
print(f"\nFound {len(processor.files)} matching files in {input_path}.\n")
print(f"Starting extraction of data:")
processor.run_async(output_file, start, end, n)
