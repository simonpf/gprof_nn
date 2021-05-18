"""
================
gprof_nn.logging
================

Sets up the Python logging to use rich for formatting. This can be used in
scripts based on the 'gprof_nn' package to get fancier output.
"""
from rich.logging import RichHandler

_LOG_LEVEL = os.environ.get('QUANTNN_LOG_LEVEL', 'WARNING').upper()
_logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
