export OMP_NUM_THREADS=4
for DAY in {08..30}
do
    python extract_data_sim.py MHS ${DAY} /gdata/simon/gprof_nn/gprof_nn_mhs_era5_${DAY}.nc
done
