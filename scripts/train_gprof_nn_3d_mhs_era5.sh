TRAINING_DATA=/gdata1/simon/gprof_nn/training_data/mhs
VALIDATION_DATA=/gdata1/simon/gprof_nn/validation_data/mhs

MODEL_PATH=/gdata1/simon/gprof_nn/models/gprof_nn_3d_mhs_test.pckl
TARGETS="surface_precip convective_precip rain_water_path ice_water_path cloud_water_path cloud_water_content snow_water_content rain_water_content latent_heat"


export QUANTNN_LOG_LEVEL=INFO
export OMP_NUM_THREADS=2

gprof_nn train 3D  MHS ERA5 ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_neurons_head 256 --n_neurons_body  512 --n_blocks 5 --n_layers_head 4 --device cuda:0 --targets ${TARGETS} --type qrnn_exp --batch_size 8 --n_epochs 10 --learning_rate 0.0005 --activation GELU --no_validation
