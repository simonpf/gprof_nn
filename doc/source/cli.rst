Command line interface
======================

After successful installation of the ``gprof_nn`` package executing the ``gprof_nn`` command should produce a list sub-commands that are provided by the application. Their usage is described below.

Running the retrieval
---------------------

``gprof_nn`` provides the ``1d`` and ``3d`` sub-commands For running the operational GPROF-NN 1D and 3D retrievals.
These commands will run the operational GPROF-NN neural network models.
The following command runs the GPROF-NN 1D retrieval in the ``ERA5``  on an input file in the GPROF preprocessor ``input.pp`` format and writes the results to a file ``output.bin`` in GPM binary output format.

.. code-block:: console
  
   gprof_nn 1d ERA5 input.pp -o output.bin


The ``gprof_nn`` command can also be used to write the output directly to NetCDF4 format by simply changing the suffix of the output file.

.. code-block:: console

   gprof_nn 1d ERA5 input.pp -o output.nc

To run the GPROF-NN retrieval using an experimental ``gprof_nn`` neural-network model, the ``gprof_nn retrieve`` command can be used as follow:

.. code-block:: console

   gprof_nn retrieve model.pckl input.pp output.nc

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
