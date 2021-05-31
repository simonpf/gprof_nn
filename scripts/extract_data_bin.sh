#!/bin/bash

export OMP_NUM_THREADS=4
INPUT=/qdata1/pbrown/gpm/GMIbins_ERA5_V7/
OUTPUT=/gdata/simon/gprof_nn

python extract_data_0d.py $INPUT $OUTPUT/gprof_gmi_era5_bin.nc --start 0.0 --end 0.05 -n 4
