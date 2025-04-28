# Preparing the training data

Training retrieval models for the GPROF-NN retrieval requires the preparation of sensor-specific training data. This training data can be extracted using the ``gprof_nn extract_training_data`` command.

## Training data sources

Training data for GPROF-NN retrievals can be derived from four sources:

1. Simulator files
2. MRMS collocations
3. ERA5 collocations
4. GMI collocations (finetuning data)

Sources 1, 2, and 3 are also used to create the a priori databases of the
conventional GPROF retrieval. Matched observations from simulator files are used
over most surface types except sea ice and snow-covered, non-mountainous
surfaces. MRMS and ERA5 collocations are used over snow-covered surfaces and sea
ice (and very cold and dry environments), respectively. Collocations with
reference retrievals from GMI are a new feature of GPROF-NN V8 retrieval and are
not used in the conventional GPROF algorithm.


## Simulator files

Training-data extraction from sim files is implemented by the ``gprof_nn extract_training_data sim`` sub-command.
Extracting training data for a given sensor  is performed as follows:

````
gprof_nn extract_training_data sim <sensor_name> /path/to/sim_file_folder/ training 1d 3d --simulator_model /path/to/simulator_model
````

This will extract training data from the sim files located in ``/path/to/sim_file_folder/`` and store the resulting
training data for the GPROF-NN 1D and GPROF-NN 3D retrievals in the ``1d`` and ``3d`` directories  in the current working directory.
For available options to customize the data extraction invoke the command using the ``--help`` option: ``gprof_nn extract_training_data sim --help``.

```{note}
The target directories for the GPROF-NN 1D and GPROF-NN 3D training data must exist prior to invoking the
command.
```

## MRMS collocations

Training data from MRMS collocations are extracted using the ``gprof_nn extract_training_data mrms`` sub-command.

````
gprof_nn extract_training_data mrms <sensor_name> /path/to/mrms_collocation_folder/ /path/to/l1c_files training 1d 3d
````

## ERA5 collocations

Training data from ERA5 collocations are extracted using the ``gprof_nn extract_training_data era5`` sub-command.

````
gprof_nn extract_training_data ERA <sensor_name> path/to/l1c_files training 1d 3d
````

## Finetuning data

Training data from CMB collocations are extracted using the ``gprof_nn extract_training_data finetuning`` sub-command.

````
gprof_nn extract_training_data finetuning /path/to/collocations/ training 1d 3d
````
