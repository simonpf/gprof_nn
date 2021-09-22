#!/bin/bash

export OMP_NUM_THREADS=4
INPUT=/qdata1/pbrown/gpm/MHSbins_ERA5_V7/
OUTPUT=/gdata/simon/gprof_nn/bin/mhs/era5

python extract_data_bin.py $INPUT $OUTPUT/gprof_mhs_era5_bin.nc --start 0.0 --end 0.05 -n 8
