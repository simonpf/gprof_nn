"""
Test for the tiling and assembling of input.
"""
import numpy as np
import matplotlib.pyplot as plt

from gprof_nn.tiling import Tiler

def upsample(x):
    m, n = x.shape
    x_new = np.zeros((3 * m - 2, n))
    x_new[0::3, :] = x
    x_new[1::3, :] = (2 * x[:-1] / 3 + 1 * x[1:] / 3)
    x_new[2::3, :] = (1 * x[:-1] / 3 + 2 * x[1:] / 3)
    return x_new


def test_assembling():
    """
    Tests assembling of tiles. Also tests that the assemble of upsampled
    tiles works as expected.
    """
    x_3 = np.tile(np.arange(128 * 3 - 2)[..., None], (1, 128))
    x_3 = x_3.astype(np.float32)
    x = x_3[::3]

    tiler = Tiler(x, tile_size=(64, 64), overlap=(16, 16))

    tiles = [
        [tiler.get_tile(i, j) for j in range(tiler.N)]
        for i in range(tiler.M)
    ]
    x_assembled = tiler.assemble(tiles)
    assert np.all(np.isclose(x, x_assembled))

    tiles = [
        [upsample(tiler.get_tile(i, j)) for j in range(tiler.N)]
        for i in range(tiler.M)
    ]
    tiler_3 = Tiler(x_3, tile_size=(3 * 64 - 2, 64), overlap=(3 * 16 - 2, 16))
    x_3_assembled = tiler_3.assemble(tiles)
    assert np.all(np.isclose(x_3, x_3_assembled))
