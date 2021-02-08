import numpy as np
from numpy.testing import assert_array_equal
from strided import gather, gather_sparse, scatter_sparse
from .utils import as_strided, zeros_nearby


def test_upper_band():
    A = np.arange(1, 4, dtype=np.uint8)
    A = zeros_nearby(A, 4)
    strides = (-1, 1)
    shape = (5, 5)
    expected = np.array(
        [
            [1, 2, 3, 0, 0],
            [0, 1, 2, 3, 0],
            [0, 0, 1, 2, 3],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 1],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)


def test_lower_band():
    A = np.arange(1, 4, dtype=np.uint8)
    A = zeros_nearby(A, 4)
    strides = (1, -1)
    shape = (5, 5)
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [3, 2, 1, 0, 0],
            [0, 3, 2, 1, 0],
            [0, 0, 3, 2, 1],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)


def test_upper_band_offset():
    A = np.arange(1, 4, dtype=np.uint8)
    A = zeros_nearby(A, 5)
    strides = (-1, 1)
    shape = (5, 5)
    offset = -1
    expected = np.array(
        [
            [0, 1, 2, 3, 0],
            [0, 0, 1, 2, 3],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)


def test_triband_offset():
    A = np.arange(1, 4, dtype=np.uint8)
    A = zeros_nearby(A, 3)
    strides = (-1, 1)
    shape = (5, 5)
    offset = 1
    expected = np.array(
        [
            [2, 3, 0, 0, 0],
            [1, 2, 3, 0, 0],
            [0, 1, 2, 3, 0],
            [0, 0, 1, 2, 3],
            [0, 0, 0, 1, 2],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)


def test_lower_band_offset():
    A = np.arange(1, 4, dtype=np.uint8)
    A = zeros_nearby(A, 5)
    strides = (1, -1)
    shape = (5, 5)
    offset = -1
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [3, 2, 1, 0, 0],
            [0, 3, 2, 1, 0],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)


def test_lower_band_other():
    A = np.arange(1, 4, dtype=np.uint8)
    A = zeros_nearby(A, 4)
    strides = (1, 1)
    shape = (5, 5)
    offset = -4
    expected = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 2],
            [0, 0, 1, 2, 3],
            [0, 1, 2, 3, 0],
            [1, 2, 3, 0, 0],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)


def test_diagonal_vector():
    A = np.arange(1, 5, dtype=np.uint8)
    A = zeros_nearby(A, 9)
    strides = (-3, 4)
    shape = (4, 4)
    expected = np.array(
        [
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 4],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)


def test_diagonal_scalar():
    A = np.arange(1, 2, dtype=np.uint8)
    A = zeros_nearby(A, 4)
    strides = (-1, 1)
    shape = (5, 5)
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)


def test_double_diagonal():
    A = np.arange(1, 4, dtype=np.uint8)
    A = zeros_nearby(A, 13)
    strides = (-9, -2, 3)
    shape = (2, 3, 6)
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 3],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides).reshape((6, 6))
    assert_array_equal(B, expected)

    G = gather(A, shape, strides).reshape((6, 6))
    assert_array_equal(G, expected)


def test_matrix_zeropad_top():
    A = np.arange(1, 10, dtype=np.uint8).reshape((3, 3))
    A = zeros_nearby(A, 3)
    strides = (3, 1)
    shape = (5, 3)
    offset = -3
    expected = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [0, 0, 0],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)


def test_matrix_checkers1():
    A = np.arange(1, 17, dtype=np.uint8).reshape((4, 4))
    strides = (8, 3, 2)
    shape = (2, 2, 2)
    offset = 1
    expected = np.array(
        [
            [2, 4],
            [5, 7],
            [10, 12],
            [13, 15],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides, offset=offset).reshape((4, 2))
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset).reshape((4, 2))
    assert_array_equal(G, expected)


def test_matrix_checkers2():
    A = np.arange(1, 17, dtype=np.uint8).reshape((4, 4))
    strides = (8, 5, 2)
    shape = (2, 2, 2)
    expected = np.array(
        [
            [1, 3],
            [6, 8],
            [9, 11],
            [14, 16],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides).reshape((4, 2))
    assert_array_equal(B, expected)

    G = gather(A, shape, strides).reshape((4, 2))
    assert_array_equal(G, expected)


def test_matrix_checkers_full1():
    A = np.arange(1, 17, dtype=np.uint8).reshape((4, 4))
    A = zeros_nearby(A, 15)
    strides = (8, 19, 2, 16)
    shape = (2, 2, 2, 2)
    offset = -15
    expected = np.array(
        [
            [0, 2, 0, 4],
            [5, 0, 7, 0],
            [0, 10, 0, 12],
            [13, 0, 15, 0],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides, offset=offset).reshape((4, 4))
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset).reshape((4, 4))
    assert_array_equal(G, expected)


def test_matrix_checkers_full2():
    A = np.arange(1, 17, dtype=np.uint8).reshape((4, 4))
    A = zeros_nearby(A, 16)
    strides = (8, 21, 2, -16)
    shape = (2, 2, 2, 2)
    offset = 0
    expected = np.array(
        [
            [1, 0, 3, 0],
            [0, 6, 0, 8],
            [9, 0, 11, 0],
            [0, 14, 0, 16],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides, offset=offset).reshape((4, 4))
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset).reshape((4, 4))
    assert_array_equal(G, expected)


def test_matrix_mirror():
    A = np.arange(1, 21, dtype=np.uint8).reshape((4, 5))
    strides = (5, -1)
    shape = (4, 5)
    offset = 4
    expected = np.array(
        [
            [5, 4, 3, 2, 1],
            [10, 9, 8, 7, 6],
            [15, 14, 13, 12, 11],
            [20, 19, 18, 17, 16],
        ],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)

    SS = scatter_sparse(G, shape, strides, offset=offset)
    assert_array_equal(SS.reshape(A.shape), A)


def test_matrix_flip():
    A = np.arange(1, 13, dtype=np.uint8).reshape((3, 4))
    strides = (-4, 1)
    shape = (3, 4)
    offset = 8
    expected = np.array(
        [[9, 10, 11, 12], [5, 6, 7, 8], [1, 2, 3, 4]],
        dtype=A.dtype,
    )
    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)

    SS = scatter_sparse(G, shape, strides, offset=offset)
    assert_array_equal(SS.reshape(A.shape), A)


def test_broadcast_rows():
    # This is like exercise 12, but broadcasts other axis
    A = np.arange(1, 26, dtype=np.uint8).reshape((5, 5))
    strides = (0, 1)
    shape = (4, 5)
    expected = np.repeat(A[0, :], 4).reshape((5, 4)).T

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)


def test_broadcast_scalar():
    A = np.arange(1, 5, dtype=np.uint8)
    strides = (0, 0)
    shape = (3, 4)
    offset = 2
    expected = A[offset] * np.ones(shape, dtype=A.dtype)

    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)
