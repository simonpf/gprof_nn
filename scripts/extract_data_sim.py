"""
Extracts training data for the GPROF-NN algorithm from sim files and
MRMS match ups.
"""
import argparse

import gprof_nn.logging
from gprof_nn import sensors
from gprof_nn.data.sim import SimFileProcessor

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


SIM_PATH = "/qdata1/pbrown/dbaseV7/simV7"
L1C_PATH = "/pdata4/archive/GPM/1CR_GMI"
MRMS_PATH = "/pdata4/veljko/GMI2MRMS_match2019/db_mrms4GMI/"
ERA5_PATH = "/qdata2/archive/ERA5/"

of = output_file
print("Running processor: ", of)
processor = SimFileProcessor(of,
                             sensor,
                             era5_path=ERA5_PATH,
                             n_workers=32,
                             day=day)
processor.run()
