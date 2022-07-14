# GPROF-NN

This Python package implements the GPROF-NN retrieval algorithms for the passive
microwave observations of GPM.

## Overview

This package provides a command line application, which implements the principal
data preparation, training and retrieval functionality of GPROF-NN. In addition
to that, the ``gprof_nn`` python package provides utility functions for the
processing of GPM-related data.

## Quick start

Running any of the released GPROF-NN retrievals should by as easy as
````
pip install -U gprof_nn                # Install gprof_nn
gprof_nn hr l1c_file.HDF5 -o output.nc # Run retrieval
````

## Documentation

Detailed documentation is available on
[readthedocs](https://gprof-nn.readthedocs.io/en/latest/).
