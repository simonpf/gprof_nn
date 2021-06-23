#TRAINING_DATA=/gdata/simon/gprof_nn/training_data_small/gmi/era5
#TRAINING_DATA=/gdata/simon/gprof_nn/training_data_small
TRAINING_DATA=/home/simonpf/src/gprof_nn/data/validation_data/gmi/era5
#VALIDATION_DATA=/gdata/simon/gprof_nn/validation_data_small/gmi/era5
VALIDATION_DATA=/home/simonpf/src/gprof_nn/data/validation_data/gmi/era5
#VALIDATION_DATA=/gdata/simon/gprof_nn/validation_data_small

MODEL_PATH=${HOME}/src/gprof_nn/models/
TARGETS="surface_precip convective_precip rain_water_path ice_water_path cloud_water_path cloud_water_content snow_water_content rain_water_content latent_heat"
#TARGETS="surface_precip rain_water_content"
#TARGETS="surface_precip"

export QUANTNN_LOG_LEVEL=INFO
export OMP_NUM_THREADS=4

python train_gprof_nn_0d.py  ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_neurons_body 512 --n_layers_body 6 --n_neurons_head 256 --n_layers_head 2 --activation GELU --residuals hyper --device cuda:1 --targets ${TARGETS} --type qrnn_exp --batch_size 1024
