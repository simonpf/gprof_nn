"""
Script to extract brightness temperature statistics from observation
datasets in NetCDF format.
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
        "Calculate statistics for all observation dataset files in the"
        "folder."
    )
)

parser.add_argument('sensor',
                    metavar="sensor",
                    type=str,
                    help='Name of the sensor.')
parser.add_argument('input_path',
                    metavar="path",
                    type=str,
                    help=("Root of the folder tree containing the observation"
                          "dataset files."))
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
n_procs = parser.n_processes
if sensor is None:
    raise ValueError(f"Sensor {args.sensor} is currently not supported.")

stats = [statistics.ObservationStatistics()]
input_files = list(Path(input_path).glob("**/*.nc"))
processor = statistics.StatisticsProcessor(sensor, input_files, stats)
processor.run(n_procs, output_path)
