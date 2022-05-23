Installation
============

There are two ways of installing ``gprof_nn``:

1. Installing an official (pre-)release from `PyPI <https://pypi.org/project/gprof-nn>`_ using PIP
2. Installing the current development version from `GitHub <https://github.com/simonpf/gprof_nn>`_.

The first option is recommended for those who only want to run one of the ``gprof_nn`` retrievals.
The second option is recommended for those who want to develop new retrievals.

PIP
---

To install the latest release of ``gprof_nn`` simply run

.. code-block:: console
  
   pip install -U gprof_nn



From source
-----------

The development version of ``gprof_nn`` is required to train new GPROF retrievals. To install
it first clone the source code from GitHub:

.. code-block:: console
  
   git clone https://github.com/simonpf/gprof_nn

Then install the Python package in editable mode using

.. code-block:: console
  
   cd gprof_nn && pip install -e .
