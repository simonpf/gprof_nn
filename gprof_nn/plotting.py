"""
=================
gprof_nn.plotting
=================

Utility functions for plotting.
"""
import pathlib

from matplotlib import rc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


_STYLE_FILE = pathlib.Path(__file__).parent / "files" / "matplotlib_style.rc"


def set_style(latex=False):
    """
    Sets matplotlib style to a style file that I find visually more pleasing
    then the default settings.

    Args:
        latex: Whether or not to use latex to render text.
    """
    plt.style.use(str(_STYLE_FILE))
    rc("text", usetex=latex)
