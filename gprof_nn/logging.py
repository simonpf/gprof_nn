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
import threading

from rich.logging import RichHandler
from rich.console import Console

_LOG_QUEUE = None


def get_log_queue():
    """
    Return global logging queue.
    """
    global _LOG_QUEUE
    if _LOG_QUEUE is None:
        _LOG_QUEUE = multiprocessing.Queue()
    return _LOG_QUEUE


def configure_queue_logging(log_queue):
    """
    Configure logging to queue in a subprocesses.

    Args:
        log_queue: The log queue provided from the parent process.
    """
    h = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    root.addHandler(h)


def log_messages():
    """
    Log messages from queue.
    """
    if _LOG_QUEUE is not None:
        while _LOG_QUEUE.qsize():
            record = self.log_queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)


_LOG_LEVEL = os.environ.get('GPROF_NN_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
_MP_LOGGER = multiprocessing.get_logger()
_MP_LOGGER.setLevel(_LOG_LEVEL)

console = Console()


def set_log_level(level):
    """
    Args:
        level: String defining the log level.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )

