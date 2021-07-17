import argparse
from pathlib import Path

from gprof_nn import evaluation
#
# Parse arguments
#

parser = argparse.ArgumentParser(
    description=(
        "Run legacy GPROF on GPROF-NN 0D input data and store results as"
        "NetCDF files."
        )
)
parser.add_argument('input_path', metavar="input_path", type=str,
                    help='Root of the directory tree containing the files'
                    'to process')
parser.add_argument('output_path', metavar="output_path", type=str,
                    help='The folder to which to write the results.')

args = parser.parse_args()
input_path = Path(args.input_path)
output_path = Path(args.output_path)
stats = args.statistics
pattern = args.pattern

statistics = [TrainingDataStatistics()]
for s in stats:
    statistic = getattr(evaluation, s, None)
    if statistic is None:
        raise ValueError(f"Coud not find statistic '{s}'.")
    statistics.append(statistic())


input_files = list(Path(input_path).glob("**/*.nc"))
print(input_files)
processor = evaluation.StatisticsProcessor(input_files, statistics)
processor.run(4, output_path)

