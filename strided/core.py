import numpy as np


def gather_strided(array, shape, strides, *, offset=0):
    rv = np.empty(shape, dtype=array.dtype)
    array_flat = array.flat
    rv_flat = rv.flat
    for dest_idx in range(rv.size):
        val = dest_idx
        source_idx = offset
        for size, stride in zip(reversed(shape), reversed(strides)):
            val, index = divmod(val, size)
            source_idx += index * stride
        if 0 <= source_idx < array.size:
            rv_flat[dest_idx] = array_flat[source_idx]
        else:
            rv_flat[dest_idx] = 0
    return rv
