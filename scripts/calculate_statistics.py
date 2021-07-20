import argparse
from pathlib import Path

import gprof_nn.statistics
from gprof_nn import sensors
#
# Parse arguments
#

parser = argparse.ArgumentParser(
    description=(
        "Run legacy GPROF on GPROF-NN 0D input data and store results as"
        "NetCDF files."
        )
)
parser.add_argument('sensor',
                    metavar="sensor",
                    type=str,
                    help='Name of the sensor.')
parser.add_argument('input_path', metavar="input_path", type=str,
                    help='Root of the directory tree containing the files'
                    'to process')
parser.add_argument('output_path', metavar="output_path", type=str,
                    help='The folder to which to write the results.')
parser.add_argument('statistics', metavar="statistic_1 statistic_2 ...",
                    type=str, nargs="+",
                    help='The names of the statistics to compute.')
parser.add_argument('--pattern', metavar="<glob_pattern>",
                    type=str,
                    default="**/*.nc",
                    help='Glob patter to use to find files to process.')
parser.add_argument('--n_processes',
                    metavar="n",
                    type=int,
                    default=4,
                    help='The number of processes to use for the processing.')

args = parser.parse_args()
input_path = Path(args.input_path)
output_path = Path(args.output_path)
stats = args.statistics
pattern = args.pattern
n_procs = args.n_processes

sensor = getattr(sensors, args.sensor, None)
if sensor is None:
    raise ValueError(f"Sensor {args.sensor} is currently not supported.")

statistics = []
for s in stats:
    print(s)
    statistic = getattr(gprof_nn.statistics, s, None)
    if statistic is None:
        raise ValueError(f"Coud not find statistic '{s}'.")
    statistics.append(statistic())


input_files = list(Path(input_path).glob(pattern))
print(input_files)

processor = gprof_nn.statistics.StatisticsProcessor(sensor,
                                                    input_files,
                                                    statistics)
processor.run(n_procs, output_path)
