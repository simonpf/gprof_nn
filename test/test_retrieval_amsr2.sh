#! /bin/bash
wget -q http://rain.atmos.colostate.edu/gprof_nn/test/gcomw1_amsr2.pp
echo "Running GPROF-NN 1D retrieval for AMSR 2."
gprof_nn 1d ERA5 gcomw1_amsr2.pp -o test.bin
python -c "from gprof_nn.data.retrieval import RetrievalFile; import numpy as np; 1/0 if not np.mean(RetrievalFile('test.bin').to_xarray_dataset().surface_precip.data >= 0.0) > 0.99 else 0;"

#echo "Running GPROF-NN 3D retrieval for AMSR 2."
#gprof_nn 3d ERA5 gcomw1_amsr2.pp
#python -c "from gprof_nn.data.retrieval import RetrievalFile; import numpy as np; 1/0 if not np.mean(RetrievalFile('test.bin').to_xarray_dataset().surface_precip.data >= 0.0) > 0.99 else 0;"
