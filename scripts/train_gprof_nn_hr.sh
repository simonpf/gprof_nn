TRAINING_DATA=/gdata1/simon/gprof_nn/training_data/gmi_hr
VALIDATION_DATA=/gdata1/simon/gprof_nn/validation_data/gmi_hr

MODEL_PATH=/gdata1/simon/gprof_nn/models/gprof_nn_hr.pckl
TARGETS="surface_precip rain_water_path ice_water_path cloud_water_path cloud_water_content snow_water_content rain_water_content latent_heat"

export QUANTNN_LOG_LEVEL=INFO
export OMP_NUM_THREADS=2

gprof_nn train HR  GMI ERA5 ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_neurons_head 256 --n_neurons_body  512 --n_blocks 5 --n_layers_head 4 --device cuda:0 --targets ${TARGETS} --type qrnn_exp --batch_size 4 --n_epochs 10 --learning_rate 0.0005 --activation GELU --no_validation
