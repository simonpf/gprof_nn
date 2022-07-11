"""
================
gprof_nn.logging
================

Sets up the Python logging to use rich for formatting. This can be used in
scripts based on the 'gprof_nn' package to get fancier output.
"""
import logging
import logging.handlers
import multiprocessing
import os
import threading

from rich.logging import RichHandler
from rich.console import Console

#
# Basic logging
#

_LOG_LEVEL = os.environ.get("GPROF_NN_LOG_LEVEL", "INFO").upper()
_CONSOLE = Console()
_HANDLER = RichHandler(console=_CONSOLE)

# The parent logger for the module.
LOGGER = logging.getLogger("gprof_nn")
LOGGER.setLevel(_LOG_LEVEL)
LOGGER.addHandler(_HANDLER)


def get_console():
    """
    Return the console to use for live logging.
    """
    return _CONSOLE


def set_log_level(level):
    """
    Args:
        level: String defining the log level.
    """
    # logging.basicConfig(
    #    level=level.upper(),
    #    format="%(message)s",
    #    datefmt="[%X]",
    #    handlers=[_HANDLER]
    # )


#
# Multi-process logging.
#

_LOG_QUEUE = None


def get_log_queue():
    """
    Return global logging queue.
    """
    global _LOG_QUEUE
    if _LOG_QUEUE is None:
        _LOG_QUEUE = multiprocessing.Manager().Queue()
    return _LOG_QUEUE


def configure_queue_logging(log_queue):
    """
    Configure logging to queue in a subprocesses.

    Args:
        log_queue: The log queue provided from the parent process.
    """
    logger = logging.getLogger("gprof_nn")
    handler = logging.handlers.QueueHandler(log_queue)
    logger.propagate = False
    logger.handlers = [handler]


def log_messages():
    """
    Log messages from queue.
    """
    if _LOG_QUEUE is not None:
        while _LOG_QUEUE.qsize():
            record = _LOG_QUEUE.get()
            _HANDLER.emit(record)
