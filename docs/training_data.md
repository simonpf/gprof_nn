# Preparing the training data

This section describes how to generate the sensor-specific training data for the GPROF-NN retrieval. The training data for each sensor is comprised of subsets extracted from different sources using the ``gprof_nn extract_training_data`` command. 

## Training data sources

Training data for GPROF-NN retrievals can be derived from four sources:

1. Simulator files
2. MRMS collocations
3. ERA5 collocations
4. GMI collocations (for finetuning)

Sources 1, 2, and 3 are the same as the ones used to create the a priori databases of the
conventional GPROF retrieval. Matched observations from simulator files are used
over most surface types except sea ice and snow-covered, non-mountainous
surfaces. MRMS and ERA5 collocations are used over snow-covered surfaces and sea
ice (and very cold and dry environments), respectively.

Sources 1 through 3 are the same as those used to construct the a priori databases of the conventional GPROF retrieval:

- **Simulator files** provide matched observations over most surface types, except for sea ice and snow-covered, non-mountainous regions.
- **MRMS collocations** are used over snow-covered surfaces.
- **ERA5 collocations** are used over sea ice and in very cold and dry environments.

The **GMI collocations** are a new addition in GPROF-NN version 8. These collocations with reference retrievals from the GMI instrument are not used in the conventional GPROF algorithm. They are employed to fine-tune the model trained on ERA5-based data and help correct biases arising from simulation errors in the GPROF simulator-based training data.


## Simulator files

Training-data extraction from sim files is implemented by the ``gprof_nn extract_training_data sim`` sub-command.
This training data uses the GPROF-NN simulator model to simulate observations from the target sensor from GMI input
observations. Since the simulator is implemented using a neural network, the extraction should be run on a machine that has a GPU.


Training data can be extracted from simulator files using the `gprof_nn extract_training_data sim` sub-command. This command uses the GPROF-NN simulator model to simulate target-sensor observations from GMI input to generate synthetic training data. Because the simulator is implemented as a neural network, GPU acceleration is strongly recommended.

To extract training data for a specific sensor, run:

````
gprof_nn extract_training_data sim <sensor_name> /path/to/sim_file_folder/ training 1d 3d --simulator_model /path/to/simulator_model
````

This will extract training data from the simulator files located in
``/path/to/sim_file_folder/`` and store the resulting training data for the
GPROF-NN 1D and GPROF-NN 3D retrievals in the ``1d`` and ``3d`` directories,
respectively. For a complete list of options to customize the behavior of the
command, run ``gprof_nn extract_training_data sim --help``.


```{note}
The output directories `1d` and `3d` must exist before running the command.
```

## MRMS collocations

Training data from MRMS collocations can be extracted using the ``gprof_nn extract_training_data mrms`` sub-command.

````
gprof_nn extract_training_data mrms <sensor_name> /path/to/mrms_collocation_folder/ /path/to/l1c_files training 1d 3d
````

## ERA5 collocations

Similarly, training data from ERA5 collocations are extracted using the ``gprof_nn extract_training_data era5`` sub-command.

````
gprof_nn extract_training_data era5 <sensor_name> path/to/l1c_files training 1d 3d
````

## Finetuning data

Finally, training data from GMI reference retrievals can be extract as follows:

````
gprof_nn extract_training_data finetuning gmi <target_sensor> <year> <month> 1d 3d --retrieval_path /edata2/simon/gprof_v8/results/gmi/gprof_nn_3d
````

This will extract training samples from collocations with GMI for the given year and month. To extract training samples for the full database period it will thus be necessary to invoke the command 12 times.

