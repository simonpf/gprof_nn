Installation
============

There are two ways of installing ``gprof_nn``:

1. Installing an official (pre-)release from `PyPI <https://pypi.org/project/gprof-nn>`_ using PIP
2. Installing the current development version from `GitHub <https://github.com/simonpf/gprof_nn>`_.

The first option is recommended for those who only want to run one of the
released ``gprof_nn`` retrievals. The second option is recommended for those who
want to develop new retrievals.

PIP
---

To install the latest, official release of ``gprof_nn`` simply run

.. code-block:: console
  
   pip install -U gprof_nn



Development version
-------------------

The development version of ``gprof_nn`` is required to train new GPROF retrievals.

External dependencies
^^^^^^^^^^^^^^^^^^^^^

Unfortunately, not all dependencies for the development verions of ``gprof_nn``
are available through PyPI and therefore are not installed automatically with
``pip``. The dependencies are listed in the ``conda_environment.yml`` file
in the root directory of the repository.

The following command creates a conda environment called ``gprof_nn`` and
installs the required packages. After activating the environment, your system
will be set up with all external dependencies of ``gprof_nn``.

.. code-block:: console
  
   conda env create -f conda_environment.yml

.. note ::
   Don't forget to install the environment using ``conda activate gprof_nn`` after installing
   it.

Getting or updating the code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The central repository containing the source code for ``gprof_nn`` is located
at `github.com/simonpf/gprof_nn <https://github.com/simonpf/gprof_nn>`_. To
obtain a local copy of the code, run

.. code-block:: console
  
   git clone https://github.com/simonpf/gprof_nn

If you already have a local copy of the code and want to update it to the most
recent version, use the following command from the folder you have copied the
code to.

.. code-block:: console

   git pull

Installing ``gprof_nn``
^^^^^^^^^^^^^^^^^^^^^^^

Finally, you can install ``gprof_nn`` by issuing the following command from the folder
you have cloned the code into:
  
.. code-block:: console

   pip install -e .[development]

.. note ::
   Passing the ``-e`` flag to the ``pip`` command installs the package in editable mode.
   This ensures that changes to the source code in this folder propagate to the ``gprof_nn``
   package installed in your Python environment.
   

Preprocessor binaries
^^^^^^^^^^^^^^^^^^^^^

Generating the ``gprof_nn`` training data requires the GPROF preprocessor
binaries to be available on the binary search path. The preprocessor binary for
the sensor ``SENSOR`` is expected to follow the naming convention


.. code-block:: console
  
   gprof2021pp_<SENSOR>_L1C

That is, the binary for GMI should be called ``gprof2021pp_GMI_L1C``.


For sensors other than GMI an additional preprocessor binary must be available.
This should be a modified version of the GMI preprocessor, which loads the
surface type map for the sensor. For a given sensor ``SENSOR``, the executable
is expected to be named
``gprof2021pp_GMI_<SENSOR>_L1C``.
