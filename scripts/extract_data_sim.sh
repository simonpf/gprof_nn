export OMP_NUM_THREADS=4
for DAY in {01..02}
do
    python extract_data_sim.py GMI ${DAY} /gdata/simon/gprof_nn/gprof_nn_gmi_era5_${DAY}.nc
done
