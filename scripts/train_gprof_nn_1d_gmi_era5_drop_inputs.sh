TRAINING_DATA=/gdata1/simon/gprof_nn/training_data/gmi/
VALIDATION_DATA=/gdata1/simon/gprof_nn/validation_data/gmi

MODEL_PATH=/home/simon/src/gprof_nn_clean/models/gprof_nn_1d_gmi_era5_dropped_[15].pckl
TARGETS="surface_precip convective_precip rain_water_path ice_water_path cloud_water_path cloud_water_content snow_water_content rain_water_content latent_heat"

export QUANTNN_LOG_LEVEL=INFO
export OMP_NUM_THREADS=2
 
gprof_nn train 1D GMI ERA5 ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_neurons_body 512 --n_layers_body 8 --n_neurons_head 256 --n_layers_head 2 --activation GELU --residuals hyper --device cuda:0 --targets ${TARGETS} --type qrnn_exp --batch_size 2048 --n_epochs 20 20 --learning_rate 0.0005 0.0005 --no_validation --drop_inputs 15
