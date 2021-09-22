"""
Extracts real retrieval input data from L1C files.
"""
import argparse

import gprof_nn.logging
from gprof_nn import sensors
from gprof_nn.data.l1c import ObservationProcessor

#
# Command line arguments
#

parser = argparse.ArgumentParser(
    description=(
        "Extracts training data for the GPROF-NN 2D algorithm from sim files "
        " and MRMS match ups."
        )
)
parser.add_argument('sensor',
                    metavar='sensor',
                    type=str,
                    help=('Name of the sensor for which to generate the'
                         'training data'))
parser.add_argument('day',
                    metavar="day",
                    type=int,
                    nargs=1,
                    help='The day of the month for which to extract the data')
parser.add_argument('output_file',
                    metavar="output_file",
                    type=str,
                    nargs=1,
                    help='File to which to write the extracted data.')

args = parser.parse_args()
sensor = args.sensor
day = args.day[0]
output_file = args.output_file[0]

sensor = getattr(sensors, sensor.strip().upper(), None)
if sensor is None:
    raise ValueError(f"The sensor '{args.sensor}' is not supported yet.")


of = output_file
print("Running processor: ", of)
processor = ObservationProcessor(of,
                                 sensor,
                                 n_workers=4,
                                 day=day)
processor.run()
