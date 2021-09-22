TRAINING_DATA_SOURCE=/gdata/simon/gprof_nn/training_data_small/mhs/era5
VALIDATION_DATA_SOURCE=/gdata/simon/gprof_nn/validation_data/mhs/era5
TRAINING_DATA_TARGET=/gdata/simon/gprof_nn/observations/noaa_19/
VALIDATION_DATA_TARGET=/gdata/simon/gprof_nn/observations/noaa_19/

MODEL_PATH=${HOME}/src/gprof_nn/models/

export OMP_NUM_THREADS=4

python train_noise_estimator.py MHS  ${TRAINING_DATA_SOURCE} ${TRAINING_DATA_TARGET} ${VALIDATION_DATA_SOURCE} ${VALIDATION_DATA_TARGET} ${MODEL_PATH} --device cuda:0 --batch_size 4096 --kind GAN
