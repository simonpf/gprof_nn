#! /bin/bash
wget -q http://rain.atmos.colostate.edu/gprof_nn/test/f17_era5.pp
echo "Running GPROF-NN 1D retrieval for SSMIS."
gprof_nn 1d ERA5 f17_era5.pp -o test.bin
python -c "from gprof_nn.data.retrieval import RetrievalFile; import numpy as np; 1/0 if not np.all(RetrievalFile('test.bin').to_xarray_dataset().surface_precip >= 0.0) else 0;"

#echo "Running GPROF-NN 3D retrieval for SSMIS."
#gprof_nn 3d ERA5 f17_era5.pp
#python -c "from gprof_nn.data.retrieval import RetrievalFile; import numpy as np; 1/0 if not np.all(RetrievalFile('test.bin').to_xarray_dataset().surface_precip >= 0.0) else 0;"
