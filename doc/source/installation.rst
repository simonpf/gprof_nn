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

To install the latest release of ``gprof_nn`` simply run

.. code-block:: console
  
   pip install -U gprof_nn



Development version
-------------------

The development version of ``gprof_nn`` is required to train new GPROF retrievals.
To install it, first clone the source code from GitHub:

.. code-block:: console
  
   git clone https://github.com/simonpf/gprof_nn

Then, install the Python package in editable mode using

.. code-block:: console
  
   cd gprof_nn && pip install -e . [development]


Additional dependencies
^^^^^^^^^^^^^^^^^^^^^^^

Unfortunately, not all dependencies for the development verions of ``gprof_nn`` are available
through PyPI. These additional dependencies are listed in the ``conda_environment.yml`` file
in the root directory of the repository. The corresponding ``gprof_nn`` conda environment can be
installed using 

.. code-block:: console
  
   conda env create -f conda_environment.yml

.. note ::
   Don't forget to install the environment using ``conda activate gprof_nn`` after installing
   it.

Preprocessor binaries
^^^^^^^^^^^^^^^^^^^^^

Generating the ``gprof_nn`` training data requires the GPROF preprocessor binaries
to be found on the binary search path. The preprocessor binary for the sensor ``SENSOR``
is expected to follow the naming convention


.. code-block:: console
  
   gprof2020pp_<SENSOR>_L1C

That is, the binary for GMI should be called ``gprof2020pp_GMI_L1C``.


For sensors other than GMI and additional preprocessor binary must be available. This
should be a modified version, which loads the surface type map for the sensor.
For a given sensor ``SENSOR``, the executable is expected to be named
``gprof2020pp_GMI_<SENSOR>_L1C``.
