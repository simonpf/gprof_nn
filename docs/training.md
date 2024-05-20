# Training

The next step after generating training, validation and test datasets is training the neural network models. 
The output from this step are the model files that are required to run GPROF-NN retrievals.

## Organization

The training process consists of three steps:

1. Creation of the training configuration using ``gprof_nn train init``
2. Running the training using ``gprof_nn train run``
3. Evaluating the model using ``gprof_nn train evaluate``

## Creating training configuration files

