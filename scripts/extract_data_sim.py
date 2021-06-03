"""
Extracts training data for the GPROF-NN 2D algorithm from sim files and
MRMS match ups.
"""
import argparse

import gprof_nn.logging
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
day = args.day[0]
output_file = args.output_file[0]

SIM_PATH = "/qdata1/pbrown/dbaseV7/simV7"
L1C_PATH = "/pdata4/archive/GPM/1CR_GMI"
MRMS_PATH = "/pdata4/veljko/GMI2MRMS_match2019/db_mrms4GMI/"
ERA5_PATH = "/qdata2/archive/ERA5/"

of = output_file
print("Running processor: ", of)
processor = SimFileProcessor(of,
                             sim_file_path=SIM_PATH,
                             mrms_path=MRMS_PATH,
                             l1c_path=L1C_PATH,
                             era5_path=ERA5_PATH,
                             n_workers=4,
                             day=day)
processor.run()
