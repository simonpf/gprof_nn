#TRAINING_DATA=/home/simonpf/src/gprof_nn/data/training_data/mhs/era5
TRAINING_DATA=/gdata1/simon/gprof_nn/training_data/mhs/era5
#VALIDATION_DATA=/home/simonpf/src/gprof_nn/data/validation_data/mhs/era5
VALIDATION_DATA=/gdata1/simon/gprof_nn/validation_data/mhs/era5

MODEL_PATH=${HOME}/src/gprof_nn/models/

export QUANTNN_LOG_LEVEL=INFO
export OMP_NUM_THREADS=4
 
gprof_nn train SIM MHS ERA5 ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_neurons_body 256 --n_neurons_head 256 --n_layers_head 4  --device cuda:1 --batch_size 8
