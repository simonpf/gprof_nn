#! /bin/bash
wget http://rain.atmos.colostate.edu/gprof_nn/test/gmi_era5_harvey.pp
echo "Runing GPROF-NN 1D retrieval."
gprof_nn 1d ERA5 gmi_era5_harvey.pp
echo "Runing GPROF-NN 3D retrieval."
gprof_nn 3d ERA5 gmi_era5_harvey.pp
