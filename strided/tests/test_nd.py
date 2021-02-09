import itertools
import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture
from strided import gather, scatter_sparse
from .utils import as_strided


def test_4d_moveaxis():
    N = 4

    wsize = 2
    xsize = 3
    ysize = 5
    zsize = 7

    zstride = 1
    ystride = zsize
    xstride = ysize * ystride
    wstride = xsize * xstride

    base_shape = (wsize, xsize, ysize, zsize)
    A = np.arange(wsize * xsize * ysize * zsize, dtype=np.uint8).reshape(base_shape)
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
        expected = np.moveaxis(A, axes, list(range(N)))
        B = as_strided(A, shape, strides)
        assert_array_equal(B, expected)

        G = gather(A, shape, strides)
        assert_array_equal(G, expected)

        SS = scatter_sparse(G, shape, strides)
        assert_array_equal(SS.reshape(A.shape), A)

        # Now use scatter to go from A to expected!  Since scatter goes
        # in reverse, instead of going from shape (W, X, Y, Z) -> shape,
        # we need to cast the operation as going from shape -> (W, X, Y, Z).
        # Hence, we need to compute the strides according the the new shape.

        # Strides as if we are already in `shape`
        base_strides = [1] * N
        for i in reversed(range(N - 1)):
            base_strides[i] = base_strides[i + 1] * base_shape[axes[i + 1]]

        # Now put the strides in the right order.  This needs to "go in reverse" as well.
        # >>> for i, j in enumerate(axes): reverse_axis[j] = i
        reverse_strides = [0] * N
        for i, j in enumerate(axes):
            reverse_strides[j] = base_strides[i]

        SSF = scatter_sparse(A, base_shape, reverse_strides, output_shape=expected.shape)
        assert_array_equal(SSF, expected)

        BR = as_strided(G, base_shape, reverse_strides)
        assert_array_equal(BR, A)

        GR = as_strided(G, base_shape, reverse_strides)
        assert_array_equal(GR, A)


def test_4d_moveaxis_twice():
    # Make sure we know how to combine operations
    N = 4

    wsize = 2
    xsize = 3
    ysize = 5
    zsize = 7

    zstride = 1
    ystride = zsize
    xstride = ysize * ystride
    wstride = xsize * xstride

    A = np.arange(wsize * xsize * ysize * zsize, dtype=np.uint8).reshape(
        (wsize, xsize, ysize, zsize)
    )
    combos = [
        (wstride, wsize, "W", 0),
        (xstride, xsize, "X", 1),
        (ystride, ysize, "Y", 2),
        (zstride, zsize, "Z", 3),
    ]
    for combo1 in itertools.permutations(combos, 4):
        strides1, shape1, names1, axes1 = zip(*combo1)
        AA = np.moveaxis(A, axes1, list(range(N)))

        for combo2 in itertools.permutations(combos, 4):
            strides2, shape2, names2, axes2 = zip(*combo2)
            expected = np.moveaxis(AA, axes2, list(range(N)))

            strides3 = [strides1[i] for i in axes2]
            shape3 = [shape1[i] for i in axes2]
            names3 = [names1[i] for i in axes2]
            axes3 = [axes1[i] for i in axes2]

            B = as_strided(A, shape3, strides3)
            assert_array_equal(B, expected)

            G = gather(A, shape3, strides3)
            assert_array_equal(G, expected)

            SS = scatter_sparse(G, shape3, strides3)
            assert_array_equal(SS.reshape(A.shape), A)
