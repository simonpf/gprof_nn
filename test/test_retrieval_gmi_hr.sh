#! /bin/bash
wget -q http://rain.atmos.colostate.edu/gprof_nn/test/l1cr_gmi_test.HDF5
echo "Running GPROF-NN HR retrieval."
gprof_nn hr l1cr_gmi_test.HDF5 -o test.nc
python -c "import xarray as xr; import numpy as np; 1/0 if not np.all(xr.load_dataset('test.nc').surface_precip.data >= 0.0) else 0;"
