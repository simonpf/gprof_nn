"""
Script to calculate statistics for bin files.
"""
import argparse
from pathlib import Path

from gprof_nn import sensors
from gprof_nn import statistics

#
# Parse arguments
#

parser = argparse.ArgumentParser(
    description=(
        "Calculate statistics for bin files for a given sensor."
    )
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
                    type=int,
                    default=4,
                    help='The number of processes to use for the processing.')

args = parser.parse_args()
input_paths = args.input_paths
output_path = Path(args.output_path)
n_procs = args.n_processes

sensor = getattr(sensors, args.sensor, None)
if sensor is None:
    raise ValueError(f"Sensor {args.sensor} is currently not supported.")

stats = [statistics.BinFileStatistics(),
         statistics.ZonalDistribution(),
         statistics.GlobalDistribution()]
input_files = []
for path in input_paths:
    input_files += list(Path(path).glob("**/*.bin"))
processor = statistics.StatisticsProcessor(sensor, input_files, stats)
processor.run(n_procs, output_path)
