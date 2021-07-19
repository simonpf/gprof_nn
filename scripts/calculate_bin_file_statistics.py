import argparse
from pathlib import Path

from gprof_nn import sensors
from gprof_nn import statistics

#
# Parse arguments
#

parser = argparse.ArgumentParser(
    description=(
        "Run legacy GPROF on GPROF-NN 0D input data and store results as"
        "NetCDF files."
    )
)
parser.add_argument(
    "input_path",
    metavar="input_path",
    type=str,
    help="Root of the directory tree containing the files" "to process",
)
parser.add_argument(
    "output_path",
    metavar="output_path",
    type=str,
    help="The folder to which to write the results.",
)
parser.add_argument('sensor',
                    metavar="sensor",
                    type=str,
                    help='Name of the sensor.')
parser.add_argument('input_paths',
                    metavar="path_1, path_2, ...",
                    type=str,
                    nargs="*",
                    help='Directories containing the bin files.'
                    'to process')
parser.add_argument('output_path',
                    metavar="output_path",
                    type=str,
                    help='The folder to which to write the results.')
parser.add_argument('--n_processes',
                    metavar="n",
                    type=str,
                    default=4,
                    help='The number of processes to use for the processing.')

args = parser.parse_args()
input_paths = Path(args.input_paths)
output_path = Path(args.output_path)

sensor = getattr(sensors, args.sensor, None)
if sensor is None:
    raise ValueError(f"Sensor {args.sensor} is currently not supported.")

stats = [statistics.BinFileStatistics()]
input_files = []
for path in input_paths:
    input_files += list(Path(path).glob("**/*.bin"))
processor = statistics.StatisticsProcessor(sensor, input_files, stats)
processor.run(4, output_path)
