import numpy as np
from functools import reduce
from operator import mul
from numpy.lib.stride_tricks import as_strided as np_as_strided, DummyArray


def array_offset(array, offset):
    """ Recreate an array with the pointer moved by an offset"""
    # Haha, use with caution
    d = array.__array_interface__
    d["data"] = (d["data"][0] + offset * array.dtype.itemsize, False)
    return np.array(DummyArray(d, base=array), copy=False)


def as_strided(array, shape, strides, *, offset=None, output_shape=None):
    """ Modified ``numpy.lib.stride_tricks.as_strided`` to include offset"""
    if offset is not None:
        array = array_offset(array, offset)
    rv = np_as_strided(array, shape, strides)
    if output_shape is not None:
        size = reduce(mul, output_shape)
        rv = rv.flat[:size].reshape(output_shape)
    return rv


def zeros_nearby(array, num_zeros):
    """ Create an array that has neighboring zeros in memory"""
    padded = np.zeros(array.size + 2 * num_zeros, dtype=array.dtype)
    padded[num_zeros:-num_zeros] = array.flat
    return padded[num_zeros:-num_zeros].reshape(array.shape)
