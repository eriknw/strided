import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture
from strided import gather, gather_sparse, scatter_sparse
from .utils import as_strided, zeros_nearby


@fixture(scope="module")
def A():
    return zeros_nearby(np.arange(1, 13, dtype=np.uint8), 20)


def test_1d_offset(A):
    strides = (1,)
    shape = (3,)
    offset = 2
    expected = A[2:5]

    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)

    SS = scatter_sparse(G, shape, strides, offset=offset, output_shape=A.shape)
    expected2 = np.zeros(A.shape, dtype=A.dtype)
    expected2[2:5] = A[2:5]
    assert_array_equal(SS, expected2)


def test_1d_offset_negative(A):
    strides = (1,)
    shape = (3,)
    offset = -1
    expected = np.array([0, 1, 2], dtype=A.dtype)

    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)

    SS = scatter_sparse(G, shape, strides, offset=offset, output_shape=A.shape)
    expected2 = np.zeros(A.shape, dtype=A.dtype)
    expected2[:2] = A[:2]
    assert_array_equal(SS, expected2)


def test_1d_offset_reverse(A):
    strides = (-1,)
    shape = (3,)
    offset = 5
    expected = A[5:2:-1]

    B = as_strided(A, shape, strides, offset=offset)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides, offset=offset)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides, offset=offset)
    assert_array_equal(S, expected)

    SS = scatter_sparse(G, shape, strides, offset=offset, output_shape=A.shape)
    expected2 = np.zeros(A.shape, dtype=A.dtype)
    expected2[3:6] = A[3:6]
    assert_array_equal(SS, expected2)
