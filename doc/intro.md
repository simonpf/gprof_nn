# GPROF-NN

GPROF-NN is a neural-network-based precipitation and hydrometeor profile retrieval for
passive microwave observations from the Global Precipitation Measurement (GPM) mission.
This document describes the usage of the ``gprof_nn`` Python package, which implements
the GPROF-NN retrieval.

## Overview

The GPROF-NN retrieval comes in two configurations and different versions for each sensor of
the GPM constellation. The two retrieval configurations are called *GPROF-NN 1D* and
*GPROF-NN 3D* for the retrieval based on a fully-connected network and the retrieval based on
a CNN, respectively. A specific retrieval for a given sensor is implemented and fully defined by
the underlying neural network, the *retrieval model*. 


The ``gprof_nn`` Python package implements all functionality required to train retrieval
models for specific sensors and using these retrieval model to retrieve precipitation from
GPM observations.

The typical workflow of training and using a GPROF-NN retrieval consists of the following steps:

1. Installation of the ``gprof_nn`` package
2. Generation of the training, validation, and testing datasets
3. Training the retrieval.
4. Running the retrieval.

Instructions for performing each of these four steps are provided in this documentation. For simply running retrievals only steps 1. and 4. are required.



