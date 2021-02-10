"""  Test the 25 exercises from here:

https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20

"""
import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture
from strided import gather, gather_sparse, scatter_sparse
from .utils import as_strided, zeros_nearby


@fixture(scope="module")
def A():
    return np.arange(1, 26, dtype=np.uint8).reshape(5, 5)


def run(
    *,
    A,
    strides,
    shape,
    expected,
    reverse_shape=None,
    reverse_strides=None,
    reverse_expected=None,
    reverse_pad=None,
    skip_reverse=False,
    skip_gather_sparse=False,
    skip_gather_sparse_reverse=False,
    **kwargs
):
    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    if not skip_gather_sparse:
        S = gather_sparse(A, shape, strides)
        assert_array_equal(S, expected)

    if not skip_reverse:
        SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
        assert_array_equal(SSF, expected)

    # Reverse
    if reverse_pad is not None:
        AA = zeros_nearby(expected, reverse_pad)
    else:
        AA = expected.copy()
    if not skip_reverse:
        BR = as_strided(AA, reverse_shape, reverse_strides, output_shape=A.shape)
        assert_array_equal(BR, reverse_expected)

        GR = gather(AA, reverse_shape, reverse_strides, output_shape=A.shape)
        assert_array_equal(GR, reverse_expected)
        if not skip_gather_sparse_reverse:
            SR = gather_sparse(AA, reverse_shape, reverse_strides, output_shape=A.shape)
            assert_array_equal(SR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise01(A):
    strides = (1,)
    shape = (3,)
    expected = A[0, :3]

    reverse_pad = 14
    reverse_shape = A.shape
    reverse_strides = (3, 1)  # (n, 1) where n >= 3
    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[0, :3] = expected

    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = zeros_nearby(expected, 14)
    BR = as_strided(AA, reverse_shape, reverse_strides)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides)
    assert_array_equal(GR, reverse_expected)

    SR = gather_sparse(AA, reverse_shape, reverse_strides)
    assert_array_equal(SR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise02(A):
    strides = (1,)
    shape = (8,)
    expected = A.flat[:8]

    reverse_pad = 17
    reverse_shape = A.shape
    reverse_strides = (5, 1)
    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected.flat[:8] = expected

    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = zeros_nearby(expected, 17)
    BR = as_strided(AA, reverse_shape, reverse_strides)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides)
    assert_array_equal(GR, reverse_expected)

    SR = gather_sparse(AA, reverse_shape, reverse_strides)
    assert_array_equal(SR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise03(A):
    strides = (1,)
    shape = (25,)
    expected = A.flatten()

    reverse_shape = A.shape
    reverse_strides = (5, 1)
    reverse_expected = A

    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = expected.copy()
    BR = as_strided(AA, reverse_shape, reverse_strides)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides)
    assert_array_equal(GR, reverse_expected)

    SR = gather_sparse(AA, reverse_shape, reverse_strides)
    assert_array_equal(SR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise04(A):
    strides = (2,)
    shape = (3,)
    expected = A[0, ::2]

    reverse_pad = 12
    reverse_shape = (13, 2)
    reverse_strides = (1, 3)
    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[0, ::2] = expected

    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = zeros_nearby(expected, 12)
    BR = as_strided(AA, reverse_shape, reverse_strides, output_shape=A.shape)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides, output_shape=A.shape)
    assert_array_equal(GR, reverse_expected)

    SR = gather_sparse(AA, reverse_shape, reverse_strides, output_shape=A.shape)
    assert_array_equal(SR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise05(A):
    strides = (5,)
    shape = (4,)
    expected = A[:4, 0]

    reverse_pad = 21
    reverse_shape = A.shape
    reverse_strides = (1, 5)
    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[:4, 0] = A[:4, 0]

    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = zeros_nearby(expected, 21)
    BR = as_strided(AA, reverse_shape, reverse_strides)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides)
    assert_array_equal(GR, reverse_expected)

    SR = gather_sparse(AA, reverse_shape, reverse_strides)
    assert_array_equal(SR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise06(A):
    strides = (6,)
    shape = (5,)
    expected = A.diagonal()

    reverse_pad = 16
    reverse_shape = A.shape
    reverse_strides = (-4, 5)
    reverse_expected = np.diag(A.diagonal())

    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = zeros_nearby(expected, 16)
    BR = as_strided(AA, reverse_shape, reverse_strides)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides)
    assert_array_equal(GR, reverse_expected)

    SR = gather_sparse(AA, reverse_shape, reverse_strides)
    assert_array_equal(SR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise07(A):
    strides = (0,)
    shape = (5,)
    expected = np.repeat(A[0, 0], 5)

    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[0, 0] = 5

    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise08(A):
    strides = (5, 1)
    shape = (3, 4)
    expected = A[:3, :4]

    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[:3, :4] = A[:3, :4]

    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise09(A):
    strides = (6, 1)
    shape = (4, 2)
    expected = np.array([[1, 2], [7, 8], [13, 14], [19, 20]], dtype=A.dtype)

    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    for i in range(4):
        reverse_expected[i, i : i + 2] = A[i, i : i + 2]

    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise10(A):
    strides = (10, 2)
    shape = (3, 3)
    expected = A[::2, ::2]

    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[::2, ::2] = A[::2, ::2]

    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise11(A):
    strides = (1, 5)
    shape = (3, 3)
    expected = A[:3, :3].T

    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[:3, :3] = A[:3, :3]

    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise12(A):
    strides = (5, 0)
    shape = (5, 4)
    expected = np.repeat(A[:, 0], 4).reshape((5, 4))

    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[:, 0] = 4 * A[:, 0]

    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise13():
    A = np.arange(1, 13, dtype=np.uint8)
    strides = (3, 1)
    shape = (4, 3)
    expected = A.reshape(shape)

    reverse_shape = A.shape
    reverse_strides = (1,)
    reverse_expected = A

    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = expected.copy()
    BR = as_strided(AA, reverse_shape, reverse_strides)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides)
    assert_array_equal(GR, reverse_expected)

    SR = gather_sparse(AA, reverse_shape, reverse_strides)
    assert_array_equal(SR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise14():
    A = np.arange(1, 11, dtype=np.uint8)
    strides = (1, 1)
    shape = (8, 3)
    expected = np.array([[i, i + 1, i + 2] for i in range(1, 9)], dtype=A.dtype)

    reverse_expected = 3 * A
    for i in range(2):
        reverse_expected[i] -= (2 - i) * A[i]
        reverse_expected[-i - 1] -= (2 - i) * A[-i - 1]

    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


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

    reverse_expected = A.copy()
    reverse_expected[1] *= 2
    reverse_expected[2] *= 3
    reverse_expected[3] *= 3
    reverse_expected[4] *= 2

    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise16():
    A = np.arange(1, 13, dtype=np.uint8).reshape((3, 2, 2))
    strides = (4, 1)
    shape = (3, 4)
    expected = A.reshape((3, 4))

    reverse_shape = A.shape
    reverse_strides = (4, 2, 1)
    reverse_expected = A

    skip_gather_sparse_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    S = gather_sparse(A, shape, strides)
    assert_array_equal(S, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = expected.copy()
    BR = as_strided(AA, reverse_shape, reverse_strides)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides)
    assert_array_equal(GR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise17(A):
    strides = (15, 5, 1)
    shape = (2, 2, 2)
    expected = np.array([[[1, 2], [6, 7]], [[16, 17], [21, 22]]], dtype=A.dtype)

    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[:2, :2] = A[:2, :2]
    reverse_expected[-2:, :2] = A[-2:, :2]

    skip_gather_sparse = True
    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise18(A):
    strides = (10, 6, 1)
    shape = (2, 2, 3)
    expected = np.array([[[1, 2, 3], [7, 8, 9]], [[11, 12, 13], [17, 18, 19]]], dtype=A.dtype)

    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[0, :3] = A[0, :3]
    reverse_expected[1, 1:4] = A[1, 1:4]
    reverse_expected[2, :3] = A[2, :3]
    reverse_expected[3, 1:4] = A[3, 1:4]

    skip_gather_sparse = True
    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise19(A):
    strides = (0, 5, 1)
    shape = (3, 2, 4)
    expected = np.array(3 * [[[1, 2, 3, 4], [6, 7, 8, 9]]], dtype=A.dtype)

    reverse_expected = np.zeros(A.shape, dtype=A.dtype)
    reverse_expected[:2, :4] = 3 * A[:2, :4]

    skip_gather_sparse = True
    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise20():
    A = np.arange(1, 13, dtype=np.uint8).reshape((3, 2, 2))
    strides = (4, 1, 2)
    shape = (3, 2, 2)
    expected = np.swapaxes(A, 1, 2)

    reverse_shape = A.shape
    reverse_strides = strides
    reverse_expected = A

    skip_gather_sparse = True
    skip_gather_sparse_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = expected.copy()
    BR = as_strided(AA, reverse_shape, reverse_strides)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides)
    assert_array_equal(GR, reverse_expected)

    SS = scatter_sparse(G, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


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

    reverse_expected = A.copy()
    reverse_expected[1:3, :] *= 2

    skip_gather_sparse = True
    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise22():
    A = np.arange(1, 13, dtype=np.uint8)
    strides = (6, 3, 1)
    shape = (2, 2, 3)
    expected = A.reshape(shape)

    reverse_shape = A.shape
    reverse_strides = (1,)
    reverse_expected = A

    skip_gather_sparse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = expected.copy()
    BR = as_strided(AA, reverse_shape, reverse_strides)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides)
    assert_array_equal(GR, reverse_expected)

    SR = gather_sparse(AA, reverse_shape, reverse_strides)
    assert_array_equal(SR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


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

    reverse_expected = A.copy()
    reverse_expected[2, :] *= 2
    reverse_expected[:, 2] *= 2

    skip_gather_sparse = True
    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise24():
    A = np.arange(1, 13, dtype=np.uint8).reshape((2, 2, 3))
    strides = (6, 0, 3, 1)
    shape = (2, 2, 2, 3)
    expected = np.stack([A, A], axis=1)

    reverse_expected = 2 * A.copy()

    skip_gather_sparse = True
    skip_reverse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    # Reverse
    AA = expected.copy()
    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)


def test_exercise25():
    A = np.arange(1, 17, dtype=np.uint8)
    strides = (8, 4, 2, 1)
    shape = (2, 2, 2, 2)
    expected = A.reshape(shape)

    reverse_shape = A.shape
    reverse_strides = (1,)
    reverse_expected = A

    skip_gather_sparse = True
    run(**locals())

    # Forward
    B = as_strided(A, shape, strides)
    assert_array_equal(B, expected)

    G = gather(A, shape, strides)
    assert_array_equal(G, expected)

    SSF = scatter_sparse(A, reverse_shape, reverse_strides, output_shape=shape)
    assert_array_equal(SSF, expected)

    # Reverse
    AA = expected.copy()
    BR = as_strided(AA, reverse_shape, reverse_strides)
    assert_array_equal(BR, reverse_expected)

    GR = gather(AA, reverse_shape, reverse_strides)
    assert_array_equal(GR, reverse_expected)

    SR = gather_sparse(AA, reverse_shape, reverse_strides)
    assert_array_equal(SR, reverse_expected)

    SS = scatter_sparse(AA, shape, strides, output_shape=A.shape)
    assert_array_equal(SS, reverse_expected)
