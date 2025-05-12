# Installation


## Basic installation


The basic installation of gprof_nn provides all the necessary functionality to run the GPROF-NN retrieval on observations from the GPM constellation. This is the easiest and recommended method for most users.

To install, run:

```
pip install git+https://github.com/simonpf/gprof_nn@gprof_v8
```

## Downloading the Retrieval Models


The trained neural network models used for retrievals with each GPM sensor are maintained in [a separate repository](https://hf.co/simonpf/gprof_nn). The ``gprof_nn`` package automatically downloads the required models on demand and stores them in the local model directory.

If you'd prefer to download all available models in advance, you can do so with the following command:


```
gprof_nn download model all

```

To change the directory ``gprof_nn`` uses to store the model files follow the instructions available [here](configuration:model_path).

## Development installation

The development installation includes additional dependencies required for training and evaluating GPROF-NN models. Follow these steps if you plan to contribute to development or train models.


### Obtaining the code

First, obtain the source code by cloning the repository:

```
git clone -b gprof_v8 https://github.com/simonpf/gprof_nn
```


### Dependencies

The recommended way to install the external depencies required for using
``gprof_nn`` is through the conda environment provided in the base directory of
the source code.

```
# Swith to folder if not already done
cd gprof_nn

conda env create --file conda_environment.yml
```

After successful create of the ``gprof_v8`` conda environment, it must be activated using
```
conda activate gprof_v8
```


### Installing the ``gprof_nn`` package

Finally, the ``gprof_nn`` package can be installed using

```
pip install -e .
```
For now, it is recommended to install the package in editable mode, i.e. using the ``-e`` option, as GPROF-NN V8 remains under development and it may therefore be necessary to update the code in the future.


