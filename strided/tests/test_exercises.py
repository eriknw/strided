"""  Test the 25 exercises from here:

https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20

"""
import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture
from strided import gather_strided
from .utils import as_strided


@fixture(scope="module")
def A():
    return np.arange(1, 26, dtype=np.uint8).reshape(5, 5)


def test_exercise01(A):
    strides = (1,)
    shape = (3,)
    expected = A[0, :3]

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise02(A):
    strides = (1,)
    shape = (8,)
    expected = A.flat[:8]

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise03(A):
    strides = (1,)
    shape = (25,)
    expected = A.flatten()

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise04(A):
    strides = (2,)
    shape = (3,)
    expected = A[0, ::2]

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise05(A):
    strides = (5,)
    shape = (4,)
    expected = A[:4, 0]

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise06(A):
    strides = (6,)
    shape = (5,)
    expected = A.diagonal()

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise07(A):
    strides = (0,)
    shape = (5,)
    expected = np.repeat(A[0, 0], 5)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise08(A):
    strides = (5, 1)
    shape = (3, 4)
    expected = A[:3, :4]

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise09(A):
    strides = (6, 1)
    shape = (4, 2)
    expected = np.array([[1, 2], [7, 8], [13, 14], [19, 20]], dtype=A.dtype)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise10(A):
    strides = (10, 2)
    shape = (3, 3)
    expected = A[::2, ::2]

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise11(A):
    strides = (1, 5)
    shape = (3, 3)
    expected = A[:3, :3].T

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise12(A):
    strides = (5, 0)
    shape = (5, 4)
    expected = np.repeat(A[:, 0], 4).reshape((5, 4))

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise13():
    A = np.arange(1, 13, dtype=np.uint8)
    strides = (3, 1)
    shape = (4, 3)
    expected = A.reshape(shape)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise14():
    A = np.arange(1, 11, dtype=np.uint8)
    strides = (1, 1)
    shape = (8, 3)
    expected = np.array([[i, i + 1, i + 2] for i in range(1, 9)], dtype=A.dtype)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise15():
    A = np.array([[10 * i, 10 * i + 1] for i in range(6)], dtype=np.uint8)
    strides = (2, 1)
    shape = (4, 6)
    expected = np.array(
        [
            [10 * i, 10 * i + 1, 10 * (i + 1), 10 * (i + 1) + 1, 10 * (i + 2), 10 * (i + 2) + 1]
            for i in range(4)
        ],
        dtype=A.dtype,
    )

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise16():
    A = np.arange(1, 13, dtype=np.uint8).reshape((3, 2, 2))
    strides = (4, 1)
    shape = (3, 4)
    expected = A.reshape((3, 4))

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise17(A):
    strides = (15, 5, 1)
    shape = (2, 2, 2)
    expected = np.array([[[1, 2], [6, 7]], [[16, 17], [21, 22]]], dtype=A.dtype)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise18(A):
    strides = (10, 6, 1)
    shape = (2, 2, 3)
    expected = np.array([[[1, 2, 3], [7, 8, 9]], [[11, 12, 13], [17, 18, 19]]], dtype=A.dtype)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise19(A):
    strides = (0, 5, 1)
    shape = (3, 2, 4)
    expected = np.array(3 * [[[1, 2, 3, 4], [6, 7, 8, 9]]], dtype=A.dtype)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise20():
    A = np.arange(1, 13, dtype=np.uint8).reshape((3, 2, 2))
    strides = (4, 1, 2)
    shape = (3, 2, 2)
    expected = np.swapaxes(A, 1, 2)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise21():
    A = np.arange(1, 21, dtype=np.uint8).reshape((4, 5))
    strides = (5, 5, 1)
    shape = (3, 2, 5)
    expected = (
        np.array(
            [
                [list(range(5 * i, 5 * (i + 1))), list(range(5 * (i + 1), 5 * (i + 2)))]
                for i in range(3)
            ],
            dtype=np.uint8,
        )
        + 1
    )

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise22():
    A = np.arange(1, 13, dtype=np.uint8)
    strides = (6, 3, 1)
    shape = (2, 2, 3)
    expected = A.reshape(shape)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise23(A):
    strides = (10, 2, 5, 1)
    shape = (2, 2, 3, 3)
    expected = np.array(
        [
            [[[1, 2, 3], [6, 7, 8], [11, 12, 13]], [[3, 4, 5], [8, 9, 10], [13, 14, 15]]],
            [
                [[11, 12, 13], [16, 17, 18], [21, 22, 23]],
                [[13, 14, 15], [18, 19, 20], [23, 24, 25]],
            ],
        ],
        dtype=A.dtype,
    )

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise24():
    A = np.arange(1, 13, dtype=np.uint8).reshape((2, 2, 3))
    strides = (6, 0, 3, 1)
    shape = (2, 2, 2, 3)
    expected = np.stack([A, A], axis=1)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)


def test_exercise25():
    A = np.arange(1, 17, dtype=np.uint8)
    strides = (8, 4, 2, 1)
    shape = (2, 2, 2, 2)
    expected = A.reshape(shape)

    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather_strided(A, shape, strides)
    assert_array_equal(G, expected)
