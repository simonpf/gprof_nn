# Preparing the training data

Training retrieval models for the GPROF-NN retrieval requires the preparation of sensor-specific training data. This training data can be extracted using the ``gprof_nn extract_training_data`` command.

## Training data sources

Training data for GPROF-NN retrievals can be derived from four sources:

1. Simulator files
2. MRMS collocations
3. ERA5 collocations
4. CMB collocations

Sources 1, 2, and 3 are also used to create the a priori databases of the
conventional GPROF retrieval. Matched observations from simulator files are used
over most surface types except sea ice and snow-covered, non-mountainous
surfaces. MRMS and ERA5 collocations are used over snow-covered surfaces and sea ice,
respectively. Collocations with GPM combined radar/PMW retrievals (CMB) are a new
feature  of  GPROF-NN V8 retrieval and are not used in the conventional GPROF algorithm.

For global GPROF-NN retrievals that are equivalent to conventional GPROF
retrievals, training data must be extracted from sources 1, 2, and 3. For retrievals
that are limited to observations of ice-free oceans, snow-free non-mountainous surfaces
and potentially snow-covered mountains it is sufficient to extract training data
solely from source 1.

## Simulator files

Training-data extraction from sim files is implemented by the ``gprof_nn extract_training_data sim`` sub-command.
Extracting training data for TROPICS TMS, for example, is performed as follows:

````
gprof_nn extract_training_data sim TMS /path/to/sim_file_folder/ training 1d 3d
````

This will extract training data from the sim files located in ``/path/to/sim_file_folder/`` and store the resulting
training data for the GPROF-NN 1D and GPROF-NN 3D retrievals in the ``1d`` and ``3d`` directories. 
For available options to customize the data extraction invoke the command using the ``--help`` option: ``gprof_nn extract_training_data sim --help``.

```{note}
The target directories for the GPROF-NN 1D and GPROF-NN 3D training data must exist prior to invoking the
command.
```

## MRMS collocations

Training data from MRMS collocations are extracted using the ``gprof_nn extract_training_data mrms`` sub-command.

````
gprof_nn extract_training_data mrms GMI /path/to/mrms_collocation_folder/ /path/to/l1c_files training 1d 3d
````


## ERA5 collocations

Training data from ERA5 collocations are extracted using the ``gprof_nn extract_training_data era5`` sub-command.

````
gprof_nn extract_training_data ERA GMI path/to/l1c_files training 1d 3d
````

## CMB collocations

Training data from CMB collocations are extracted using the ``gprof_nn extract_training_data finetuning`` sub-command.

````
gprof_nn extract_training_data finetuning /path/to/collocations/ training 1d 3d
````
