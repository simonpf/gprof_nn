# GPROF-NN

This repository contains the prototype implementation of the GPROF-NN retrieval originally described in Pfreundschuh (2022). It has now been superseded by the updated version of GPROF-NN used to implement the operational GPROF V08 retrieval.

Pfreundschuh, S., Brown, P. J., Kummerow, C. D., Eriksson, P., and Norrestad​​​​​​​, T.: GPROF-NN: a neural-network-based implementation of the Goddard Profiling Algorithm, Atmos. Meas. Tech., 15, 5033–5060, https://doi.org/10.5194/amt-15-5033-2022, 2022.

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
