TRAINING_DATA=/gdata/simon/gprof_nn/training_data_small/gmi/era5
#TRAINING_DATA=/home/simonpf/src/gprof_nn/data/training_data/gmi/era5
VALIDATION_DATA=/gdata/simon/gprof_nn/validation_data_small/gmi/era5
#VALIDATION_DATA=/home/simonpf/src/gprof_nn/data/validation_data/gmi/era5

MODEL_PATH=${HOME}/src/gprof_nn/models/
TARGETS="surface_precip convective_precip rain_water_path ice_water_path cloud_water_path cloud_water_content snow_water_content rain_water_content latent_heat"

export QUANTNN_LOG_LEVEL=INFO
export OMP_NUM_THREADS=4

python train_gprof_nn_2d.py  ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_features_head 256 --n_features_body  256 --n_blocks 8 --n_layers_head 4 --device cuda:0 --targets ${TARGETS} --type qrnn_exp --batch_size 2
