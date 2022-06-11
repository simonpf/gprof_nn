#! /bin/bash
wget http://rain.atmos.colostate.edu/gprof_nn/test/mhs_era5_harvey.pp
echo "Running GPROF-NN 1D retrieval."
gprof_nn 1d ERA5 mhs_era5_harvey.pp
echo "Running GPROF-NN 3D retrieval."
gprof_nn 3d ERA5 mhs_era5_harvey.pp
