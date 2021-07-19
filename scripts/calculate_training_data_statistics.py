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
parser.add_argument('sensor', metavar="sensor", type=str,
                    help='Name of the sensor.')
parser.add_argument('input_path', metavar="input_path", type=str,
                    help='Root of the directory tree containing the files'
                    'to process')
parser.add_argument('output_path', metavar="output_path", type=str,
                    help='The folder to which to write the results.')

args = parser.parse_args()
input_path = Path(args.input_path)
output_path = Path(args.output_path)

sensor = getattr(sensors, args.sensor, None)
if sensor is None:
    raise ValueError(f"Sensor {args.sensor} is currently not supported.")

stats = [statistics.TrainingDataStatistics()]
input_files = list(Path(input_path).glob("**/*.nc"))
print(input_files)
processor = statistics.StatisticsProcessor(sensor, input_files, stats)
processor.run(4, output_path)

