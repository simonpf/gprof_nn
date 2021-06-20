export OMP_NUM_THREADS=32
for DAY in 12
do
    python extract_data_sim.py ${DAY} /gdata/simon/gprof_nn/gprof_gmi_era5_${DAY}.nc
done
