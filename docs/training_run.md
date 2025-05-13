# Running the training

In this step, the training, validation and test datasets generated in the previous step are used to train the neural network models that implement the retrieval. The output from this step are the model files that are required to run GPROF-NN retrievals.

## Organization

The training in turn involves three steps:

1. Creation of the training configuration using ``gprof_nn train init``
2. Running the training using ``gprof_nn train run``
3. Evaluating the model using ``gprof_nn train evaluate``

The parameters defining the training of a GPROF-NN retrieval model is defined using configuration files in ``.toml`` format. Since the training produces several artifacts required for training and the monitoring of the training progress, it is recommended to organize the training of different models into separate directories.

## Creating training configuration files

The configurations files required for training can be created using the ``gprof_nn training init`` command.

As an example, the following command creates the configuration files to train a GPROF-NN 1D retrieval model for the TROPICS TMS sensor using training and validation files located in ``path/to/trainin_data`` and ``path/to/validation_data``, respectively:

```
gprof_nn training init TMS 1d /path/to/training_data /path/to/validation_data
```

This will create model and training configuration files names ``model.toml`` and ``training.toml`` defining the standard GPROF training scheme.

### Multiple training and validation folders

In order to simplify combining training data from the different sources, the ``gprof_nn training init`` command accepts lists of paths for both training and validation data. To pass multiple training and/or validation data folder simply separate them using  ``,``:


```
gprof_nn training init <sensor> 3d /path/to/training_data_1,/path/to/training_data_2 /path/to/validation_data_1
```

## Running the training

Finally, the training can be run using the ``gprof_nn training run`` command.  By default this will look for ``model.toml`` and ``training.toml`` files in the current working directory and train the model defined by the two files. For ways to customize the behavior of the ``gprof_nn training run`` command use ``gprof_nn training run --help``.

### Exploratory data analysis

The ``gprof_nn training run`` command will perform an brief exploratory data anlysis (EDA) if it is run for the first time. During this EDA it will record basic statistics of the retrieval input and output data. This data is required for the normalization of the retrieval input data. The results of the EDA are stored in a folder named ``stats``. The EDA is skipped when a stats folder with the expected statistics is present in the current working directory.

### Monitoring training progress

Training progress can be monitored using ``tensorboard``. To start the tensorboard server invoke
```
tensorboard --logdir logs
```
from the directory the training is run from.

