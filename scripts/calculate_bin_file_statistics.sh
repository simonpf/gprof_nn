export OMP_NUM_THREADS=1
gprof_nn calculate_statistics GMI bin /qdata1/pbrown/gpm/GMIbins_ERA5_V7/ /qdata1/pbrown/gpm/GMIbins_SI_ERA5_V7/ ~/src/gprof_nn/statistics/bin/gmi/era5 --n_processes 32
