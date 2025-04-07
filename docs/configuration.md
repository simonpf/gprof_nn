# Configuration

The gprof_nn package uses configurable parameters to determine how it interacts with the local system. This section explains how to view and modify these settings.

## The ``gprof_nn`` config file

To keep track of the local configuration, the ``gprof_nn`` package keeps a configuration file in the current user's configuration directory. To display the location of the config file used by ``gprof_nn`` used

```
gprof_nn config file
```

You can modify the configuration either by editing this file directly or by using the ``gprof_nn config set`` command.

## Inspecting the current configuration

To inspect the current configuration, use the following command:

```
gprof_nn config show
```

(configuration:model_path)=
## Setting the Model Path

The **model path** defines where gprof_nn looks for retrieval models. By default, this is set to the user's application data directory. To change it, use:

```
gprof_nn config set data model_path /path/to/new/model_path
```

## Specifying the Preprocessor Executable


To preprocess Level 1C (L1C) files, gprof_nn requires the path to the appropriate preprocessor executable for each sensor. By default, it attempts to run the preprocessor by invoking the command using the name of the corresponding executable. To specify a custom executable for a particular sensor, run:


```
gprof_nn config set preprocessor <sensor> <preprocessor_executable>
```

Replace ``<sensor>`` with the sensor name and ``<preprocessor_executable>`` with the full path to the executable.
