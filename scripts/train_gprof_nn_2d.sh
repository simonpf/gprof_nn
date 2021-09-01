TRAINING_DATA=/gdata/simon/gprof_nn/training_data/gmi/era5
#TRAINING_DATA=/home/simonpf/src/gprof_nn/data/training_data/gmi/era5
VALIDATION_DATA=/gdata/simon/gprof_nn/validation_data/gmi/era5
#VALIDATION_DATA=/home/simonpf/src/gprof_nn/data/validation_data/gmi/era5

MODEL_PATH=${HOME}/src/gprof_nn/models/
TARGETS="surface_precip convective_precip rain_water_path ice_water_path cloud_water_path cloud_water_content snow_water_content rain_water_content latent_heat"

export QUANTNN_LOG_LEVEL=INFO
export OMP_NUM_THREADS=1

gprof_nn train 2D  GMI ERA5 ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_neurons_head 128 --n_neurons_body  256 --n_blocks 4 --n_layers_head 4 --device cuda:0 --targets ${TARGETS} --type qrnn_exp --batch_size 8 --n_epochs 10 20 20 --learning_rate 0.0005 0.0005 0.00005
