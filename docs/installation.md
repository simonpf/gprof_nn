# Installation


## Obtaining the code

The first step in installing ``gprof_nn`` consists of cloning the source code from GitHub.

```
git clone -b gprof_v8 git@github.com/simonpf/gprof_nn
```

This will clone the source code for the V8 version of GPROF-NN into a new folder named ``gprof_nn``.


## Dependencies

The recommended way to install the external depencies required for using ``gprof_nn`` is through
the conda environment provided in the base directory of the source code.

```
# Swith to folder if not already done
cd gprof_nn

conda env create --file conda_environment.yml
```

After successful create of the ``gprof_v8`` conda environment, it must be activated using
```
conda activate gprof_v8
```


## Installation

Finally, the ``gprof_nn`` package can be installed using

```
pip install -e .
```
For now, it is recommended to install the package in editable mode, i.e. using the ``-e`` option, as GPROF-NN V8 remains under development and it may therefore be necessary to update the code in the future.
