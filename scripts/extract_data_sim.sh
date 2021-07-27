export OMP_NUM_THREADS=4
for DAY in {01..03}
do
    python extract_data_sim.py GMI ${DAY} /gdata/simon/gprof_nn/test_data/gmi/era5/gprof_nn_gmi_era5_${DAY}.nc
done
for DAY in {04..05}
do
    python extract_data_sim.py GMI ${DAY} /gdata/simon/gprof_nn/validation_data/gmi/era5/gprof_nn_gmi_era5_${DAY}.nc
done
for DAY in {06..30}
do
    python extract_data_sim.py GMI ${DAY} /gdata/simon/gprof_nn/training_data/gmi/era5/gprof_nn_gmi_era5_${DAY}.nc
