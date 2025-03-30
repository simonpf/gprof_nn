# GPROF-NN

GPROF-NN is the neural-network-based implementation of the Goddard Profiling
Algorithm (GPROF), the operational precipitation retrieval for the passive
micorwave observations from the Global Precipitation Measurement (GPM) Mission.
The GPROF-NN algorithm produces estimates of surface precipitation and
hydrometeor profiles for all sensors of the GPM constellation.


## Overview

The ``gprof_nn`` python package implements the GPROF-NN retrieval. The package contains the code for the full training, evaluation, and inference pipeline. Trained retrieval models for all major sensor types are available from the corresponding [Hugging Face repository](https://hf.co/simonpf/gprof_nn).
