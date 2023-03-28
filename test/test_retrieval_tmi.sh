#! /bin/bash
wget -q https://rain.atmos.colostate.edu/gprof_nn/test/tmipr_era5.pp
echo "Running GPROF-NN 1D retrieval for TMIPR."
gprof_nn 1d ERA5 tmipr_era5.pp -o test_tmipr.bin
python -c "from gprof_nn.data.retrieval import RetrievalFile; import numpy as np; 1/0 if not np.all(RetrievalFile('test_tmipr.bin').to_xarray_dataset().surface_precip >= 0.0) else 0;"
rm test_tmipr.bin
#echo "Running GPROF-NN 3D retrieval for TMIPR."
#gprof_nn 3d ERA5 tmipr_era5.pp
#python -c "from gprof_nn.data.retrieval import RetrievalFile; import numpy as np; 1/0 if not np.all(RetrievalFile('test.bin').to_xarray_dataset().surface_precip >= 0.0) else 0;"

wget -q https://rain.atmos.colostate.edu/gprof_nn/test/tmipo_era5.pp
echo "Running GPROF-NN 1D retrieval for TMIPO."
gprof_nn 1d ERA5 tmipo_era5.pp -o test_tmipo.bin
python -c "from gprof_nn.data.retrieval import RetrievalFile; import numpy as np; 1/0 if not np.all(RetrievalFile('test_tmipo.bin').to_xarray_dataset().surface_precip >= 0.0) else 0;"
rm test_tmipo.bin
#echo "Running GPROF-NN 3D retrieval for TMIPO."
#gprof_nn 3d ERA5 tmipo_era5.pp
#python -c "from gprof_nn.data.retrieval import RetrievalFile; import numpy as np; 1/0 if not np.all(RetrievalFile('test.bin').to_xarray_dataset().surface_precip >= 0.0) else 0;"
