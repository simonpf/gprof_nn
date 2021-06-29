export OMP_NUM_THREADS=32
for DAY in 5
do
    python extract_data_sim.py MHS ${DAY} /gdata/simon/gprof_nn/gprof_mhs_era5_${DAY}.nc
done
for DAY in 6
do
    python extract_data_sim.py MHS ${DAY} /gdata/simon/gprof_nn/gprof_mhs_era5_${DAY}.nc
done
for DAY in 7
do
    python extract_data_sim.py MHS ${DAY} /gdata/simon/gprof_nn/gprof_mhs_era5_${DAY}.nc
done
for DAY in 8
do
    python extract_data_sim.py MHS ${DAY} /gdata/simon/gprof_nn/gprof_mhs_era5_${DAY}.nc
done
