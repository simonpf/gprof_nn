export OMP_NUM_THREADS=32
for DAY in 12
do
    python extract_data_sim.py MHS ${DAY} /gdata/simon/gprof_nn/gprof_mhs_era5_${DAY}.nc
done
