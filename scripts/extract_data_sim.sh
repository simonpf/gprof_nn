export OMP_NUM_THREADS=4
for DAY in {20..31}
do
    python extract_data_sim.py ${DAY} /gdata/simon/gprof_nn/gprof_gmi_era5_${DAY}.nc
done
