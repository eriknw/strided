import itertools
import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture
from strided import gather, scatter_sparse
from .utils import as_strided


def test_4d_moveaxis():
    wsize = 2
    xsize = 3
    ysize = 5
    zsize = 7

    zstride = 1
    ystride = zsize
    xstride = ysize * ystride
    wstride = xsize * xstride

    A = np.arange(wsize * xsize * ysize * zsize, dtype=np.uint8)

    for combo in itertools.permutations(
        [
            (wstride, wsize, "W", 0),
            (xstride, xsize, "X", 1),
            (ystride, ysize, "Y", 2),
            (zstride, zsize, "Z", 3),
        ],
        4,
    ):
        strides, shape, names, axes = zip(*combo)
        expected = np.moveaxis(
            A.reshape((wsize, xsize, ysize, zsize)), axes, list(range(len(axes)))
        )
        B = as_strided(A, shape, strides)
        assert_array_equal(B, expected)

        G = gather(A, shape, strides)
        assert_array_equal(G, expected)

        SS = scatter_sparse(G, shape, strides)
        assert_array_equal(SS.reshape(A.shape), A)
