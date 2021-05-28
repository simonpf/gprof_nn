"""
================
gprof_nn.logging
================

Sets up the Python logging to use rich for formatting. This can be used in
scripts based on the 'gprof_nn' package to get fancier output.
"""
import logging
import multiprocessing
import os

from rich.logging import RichHandler

_LOG_LEVEL = os.environ.get('GPROF_NN_LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
_MP_LOGGER = multiprocessing.get_logger()
_MP_LOGGER.setLevel(_LOG_LEVEL)