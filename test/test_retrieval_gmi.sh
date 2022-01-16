#! /bin/bash
wget -q http://rain.atmos.colostate.edu/gprof_nn/test/gmi_era5_harvey.pp
echo "Runing GPROF-NN 1D retrieval."
gprof_nn 1d ERA5 gmi_era5_harvey.pp -o test.bin
python -c "from gprof_nn.data.retrieval import RetrievalFile; import numpy as np; 1/0 if not np.all(RetrievalFile('test.bin').to_xarray_dataset().surface_precip >= 0.0) else 0;"
#echo "Runing GPROF-NN 3D retrieval."
#gprof_nn 3d ERA5 gmi_era5_harvey.pp
