import numpy as np

def get_start_and_clips(n, tile_size, overlap):
    """ Calculate start indices and numbers of clipped pixels for a given
    side length, tile size and overlap.

    Args:
        n: The image size to tile in pixels.
        tile_size: The size of each tile
        overlap: The number of pixels of overlap.

    Rerturn:
        A tuple ``(start, clip)`` containing the start indices of each tile
        and the number of pixels to clip between each neighboring tiles.
    """
    start = []
    clip = []
    j = 0
    while j + tile_size < n:
        start.append(j)
        if j > 0:
            clip.append(overlap // 2)
        j = j + tile_size - overlap
    start.append(max(n - tile_size, 0))
    if len(start) > 1:
        clip.append((start[-2] + tile_size - start[-1]) // 2)
    start = start
    clip = clip
    return start, clip


class Tiler:
    """
    Helper class that performs two-dimensional tiling of retrieval inputs and
    calculates clipping ranges for the reassembly of tiled predictions.

    Attributes:
        M: The number of tiles along the first image dimension (rows).
        N: The number of tiles along the second image dimension (columns).
    """

    def __init__(self, x, tile_size=512, overlap=32):
        """
        Args:
            x: List of input tensors for the hydronn retrieval.
            tile_size: The size of a single tile.
            overlap: The overlap between two subsequent tiles.
        """

        self.x = x
        m, n = x.shape[-2:]
        self.m = m
        self.n = n

        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        if len(tile_size) == 1:
            tile_size = tile_size * 2
        self.tile_size = (min(m, tile_size[0]), min(n, tile_size[1]))

        if isinstance(overlap, int):
            overlap = (overlap, overlap)
        if len(overlap) == 1:
            overlap = overlap * 2

        self.overlap = overlap

        i_start, i_clip = get_start_and_clips(self.m, tile_size[0], overlap[0])
        self.i_start = i_start
        self.i_clip = i_clip

        j_start, j_clip = get_start_and_clips(self.n, tile_size[1], overlap[1])
        self.j_start = j_start
        self.j_clip = j_clip

        self.M = len(i_start)
        self.N = len(j_start)

    def get_tile(self, i, j):
        """
        Get tile in the 'i'th row and 'j'th column of the two
        dimensional tiling.

        Args:
            i: The 0-based row index of the tile.
            j: The 0-based column index of the tile.

        Return:
            List containing the tile extracted from the list
            of input tensors.
        """
        i_start = self.i_start[i]
        i_end = i_start + self.tile_size[0]
        j_start = self.j_start[j]
        j_end = j_start + self.tile_size[1]
        return self.x[..., i_start:i_end, j_start:j_end]

    def get_slices(self, i, j):
        """
        Return slices for the clipping of the result tensors.

        Args:
            i: The 0-based row index of the tile.
            j: The 0-based column index of the tile.

        Return:
            Tuple of slices that can be used to clip the retrieval
            results to obtain non-overlapping tiles.
        """
        if i == 0:
            i_clip_l = 0
        else:
            i_clip_l = self.i_clip[i - 1]
        if i >= self.M - 1:
            i_clip_r = self.tile_size[0]
        else:
            i_clip_r = self.tile_size[0] - self.i_clip[i]
        slice_i = slice(i_clip_l, i_clip_r)

        if j == 0:
            j_clip_l = 0
        else:
            j_clip_l = self.j_clip[j - 1]
        if j >= self.N - 1:
            j_clip_r = self.tile_size[1]
        else:
            j_clip_r = self.tile_size[0] - self.j_clip[j]
        slice_j = slice(j_clip_l, j_clip_r)

        return (slice_i, slice_j)

    def get_weights(self, i, j):
        """
        Get weights to reassemble results.

        Args:
            i: Row-index of the tile.
            j: Column-index of the tile.

        Return:
            Numpy array containing weights for the corresponding tile.
        """
        sl_i, sl_j = self.get_slices(i, j)

        m, n  = self.tile_size
        w_i = np.ones((m, n))
        if i > 0:
            trans_start = self.i_start[i]
            trans_end = self.i_start[i - 1] + self.tile_size[0]
            l_trans = trans_end - trans_start
            start = l_trans
            w_i[:start] = np.linspace(0, 1, l_trans)[..., np.newaxis]
        if i < self.M - 1:
            trans_start = self.i_start[i + 1]
            trans_end = self.i_start[i] + self.tile_size[0]
            l_trans = trans_end - trans_start
            start = (self.tile_size[0] - l_trans)
            w_i[start:] = np.linspace(1, 0, l_trans)[..., np.newaxis]

        w_j = np.ones((m, n))
        if j > 0:
            trans_start = self.j_start[j]
            trans_end = self.j_start[j - 1] + self.tile_size[1]
            l_trans = trans_end - trans_start
            start = l_trans
            w_j[:, :start] = np.linspace(0, 1, l_trans)[np.newaxis]
        if j < self.N - 1:
            trans_start = self.j_start[j + 1]
            trans_end = self.j_start[j] + self.tile_size[1]
            l_trans = trans_end - trans_start
            start = (self.tile_size[1] - l_trans)
            w_j[:, start:] = np.linspace(1, 0, l_trans)[np.newaxis]

        return w_i * w_j

    def assemble(self, slices):
        """
        Assemble slices back to original shape using linear interpolation in
        overlap regions.

        Args:
            slices: List of lists of slices.

        Return:
            ``numpy.ndarray`` containing the data from the slices reconstructed
            to the original shape.
        """
        slice_0 = slices[0][0]

        shape = slice_0.shape[:-2] + (self.m, self.n)
        results = np.zeros(shape, dtype=slice_0.dtype)

        for i, row in enumerate(slices):
            for j, slc in enumerate(row):

                i_start = self.i_start[i]
                i_end = i_start + self.tile_size[0]
                row_slice = slice(i_start, i_end)
                j_start = self.j_start[j]
                j_end = j_start + self.tile_size[1]
                col_slice = slice(j_start, j_end)

                output = results[..., row_slice, col_slice]
                weights = self.get_weights(i, j)
                output += weights * slc

        return results

    def __repr__(self):
        return f"Tiler(tile_size={self.tile_size}, overlap={self.overlap})"
