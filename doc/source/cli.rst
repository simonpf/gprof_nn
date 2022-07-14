Command line interface
======================

After successfully installing the gprof_nn package, executing the ``gprof_nn``
command should produce a list of sub-commands provided by the application.
Detailed information on the options available for each sub-command can be
printed by issuing the following command.

.. code-block:: console
  
   gprof_nn <command> --help


Running the retrieval
---------------------

There are two ways of running ``gprof_nn`` retrievals. The first one uses the
publicly available, released models. The second one runs the retrieval using an
arbitrary retrieval model.

Operational models
^^^^^^^^^^^^^^^^^^

``gprof_nn`` provides the ``1d`` and ``3d`` sub-commands for running the
operational gprof-nn 1d and 3d retrievals. These commands automatically download
the publicly released retrieval models and run them on a given input file. The
following command runs the GPROF-NN 1D retrieval in the ``ERA5`` configuration
on an input file ``input.pp`` in the GPROF preprocessor format. The results
are written to a file ``output.bin`` in GPM binary output format.

.. code-block:: console
  
   gprof_nn 1d ERA5 input.pp -o output.bin


The ``gprof_nn`` command can also be used to write the output directly to NetCDF4 format by simply changing the suffix of the output file.

.. code-block:: console

   gprof_nn 1d ERA5 input.pp -o output.nc


Experimental models
^^^^^^^^^^^^^^^^^^^

To run the GPROF-NN retrieval using an arbitrary ``gprof_nn`` neural-network
model, the ``gprof_nn retrieve`` command can be used as follow:

.. code-block:: console

   gprof_nn retrieve model.pckl input.pp output.nc

This command will process the input file ``input.pp`` using the model
``model.pckl`` and write the results in NetCDF4 format to the file
``output.nc``.

Input files
^^^^^^^^^^^

The retrieval input can be a file in preprocessor format ending in ``.pp``,
a L1C file ending in ``.HDF5`` or a NetCDF file in the same format as the
training data. If the input is a L1C file, the preprocessor will be run
automatically.

If a directory is given as the input, all files with suffixes ``.pp``, ``.HDF5``
and ``.nc`` are processed. The processing is parallelized and the number
of processes used can be customized using the ``--n_processes`` flag.


Output format
^^^^^^^^^^^^^

The output format can be set explicitly for all processed files
by setting  the ``--format`` flag to ``GPROF_BINARY`` or ``NETCDF``.
Otherwise the output format will be inferred from the name of the
output file. If the output file ends in ``.bin`` the binary
GPROF format will be used. Otherwise the outputs will be stored
in NetCDF4 format. If no explicit output filename is give, the
output will be in GPROF binary format only if the input file
is a preprocessor file.


Generation of training data
---------------------------

Training data for the training of the GPROF-NN retrieval models must be generated for each sensor and preprocessor configuration. The command below generates training data for the ``GMI`` sensor and the ``ERA5`` preprocessor configuration:

.. code-block:: console

   gprof_nn extract_data GMI ERA5 training output_folder

Calculation of training data and observations statistics
--------------------------------------------------------

The ``calculate_statistics`` sub-command provides functionality to calculate relevant statistics from the training data as well as L1 observations. The statistics are used to calculate corrrections for the


Training data statistics
^^^^^^^^^^^^^^^^^^^^^^^^

The following command calculates statistics of the training data for the GPROF-NN 1D retrieval
located in ``training_data_folder`` and writes NetCDF4 containing the results to the ``destination_folder``.

.. code-block:: console

   gprof_nn calculate_statistics GMI training_1d training_data_folder destination_folder

Observation statistics
^^^^^^^^^^^^^^^^^^^^^^

The following command calculates observation statistics for L1C observations
located in ``observation_data_folder`` and writes NetCDF4 containing the results to the ``destination_folder``.


.. code-block:: console

   gprof_nn calculate_statistics GMI observations observation_data_folder destination_folder
