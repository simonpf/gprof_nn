TRAINING_DATA=/gdata/simon/gprof_nn/training_data/gmi/era5
VALIDATION_DATA=/gdata/simon/gprof_nn/validation_data/gmi/era5

MODEL_PATH=${HOME}/src/gprof_nn/models/gprof_nn_0d_gmi_era5_na.pckl
TARGETS="surface_precip convective_precip rain_water_path ice_water_path cloud_water_path cloud_water_content snow_water_content rain_water_content latent_heat"
#TARGETS="surface_precip rain_water_content"
#TARGETS="surface_precip"

export QUANTNN_LOG_LEVEL=INFO
export OMP_NUM_THREADS=1
 
gprof_nn train 0D GMI ERA5 ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_neurons_body 512 --n_layers_body 8 --n_neurons_head 256 --n_layers_head 2 --activation GELU --residuals hyper --device cuda:0 --targets ${TARGETS} --type qrnn_exp --batch_size 2048 --n_epochs 10 20 --learning_rate 0.0005 0.0005 --no_ancillary
