# Running retrievals

Finally, retrieval models produced by in the training step can be retrieve
precipitation and hydrometeor profiles from GPM observations.

## Command-line interface

GPROF-NN retrieval on one or multiple input files can be run using the ``gprof_nn retrieve`` command. 

```
gprof_nn retrieve /path/to/retrieval_model.pt /path/to/input_data --output_path /path/to_output_data
```

The command can be used to run the trained retrieval model on L1C, preprocessor or training data files. 
