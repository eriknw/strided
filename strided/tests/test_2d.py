import numpy as np
from numpy.testing import assert_array_equal
from strided import gather_strided
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

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


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

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


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

    G = gather_strided(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)


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

    G = gather_strided(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)


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

    G = gather_strided(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)


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

    G = gather_strided(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)


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

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


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

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


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

    G = gather_strided(A, shape, strides).reshape((6, 6))
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

    G = gather_strided(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)


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

    G = gather_strided(A, shape, strides, offset=offset).reshape((4, 2))
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

    G = gather_strided(A, shape, strides).reshape((4, 2))
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

    G = gather_strided(A, shape, strides, offset=offset).reshape((4, 4))
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

    G = gather_strided(A, shape, strides, offset=offset).reshape((4, 4))
    assert_array_equal(G, expected)
