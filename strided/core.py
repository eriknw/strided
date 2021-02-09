import numpy as np


def gather(array, shape, strides, *, offset=0):
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


def gather_sparse(array, shape, strides, *, offset=0):
    """Iterate over input array to "gather" into output array.

    This requires solving a linear diophantine equation of the form

        stride[0] * x_0 + ... + stride[m] * x_m = source_index + offset

    with

        0 <= x_i < shape[i], and x_i is integer for all i

    and not including stride values of 0.  Then compute dest_index from x_i's:

        dest_index = x_0 * prod(shape[1:]) + x_1 * prod(shape[2:]) + ... + x_m

    """
    if len(shape) == 1:
        return _gather_sparse_1d(array, shape, strides, offset=offset)
    if len(shape) == 2:
        return _gather_sparse_2d(array, shape, strides, offset=offset)
    raise NotImplementedError()


def _gather_sparse_1d(array, shape, strides, *, offset=0):
    rv = np.zeros(shape, dtype=array.dtype)
    array_flat = array.flat
    [size] = shape
    [stride] = strides
    if stride == 0:
        if 0 <= offset < array.size:
            rv[:] = array_flat[offset]
        return rv
    for source_idx in range(array.size):
        dest_idx, remainder = divmod(source_idx - offset, stride)
        if dest_idx >= size:
            if stride > 0:
                break
            continue
        if dest_idx < 0:
            if -dest_idx <= offset:
                continue
            break
        if remainder == 0:
            rv[dest_idx] = array_flat[source_idx]
    return rv


def _gather_sparse_2d(array, shape, strides, *, offset=0):
    rv = np.zeros(shape, dtype=array.dtype)
    array_flat = array.flat
    rv_flat = rv.flat

    if strides[0] == 0:
        rv1d = _gather_sparse_1d(array, (shape[1],), (strides[1],), offset=offset)
        rv[:, :] = rv1d[:]
        return rv
    elif strides[1] == 0:
        rv1d = _gather_sparse_1d(array, (shape[0],), (strides[0],), offset=offset)
        rv[:, :] = rv1d[:, None]
        return rv

    # this is hacky (for now) to, uh, help develop tests ;)
    import sympy
    from functools import reduce
    from itertools import starmap

    xs = sympy.symbols(f"x_:{len(strides)}", integer=True)
    ts = sympy.symbols(f"t_:{len(strides) - 1}", integer=True)
    base_eq = reduce(sympy.Add, starmap(sympy.Mul, zip(strides, xs))) + offset

    for source_idx in range(array.size):
        eq = base_eq - source_idx
        soln = sympy.diophantine(eq)
        if not soln:
            continue
        [soln] = soln

        dest_indices = []
        if soln[0] == ts[0]:
            for i in range(shape[0]):
                expr = soln[1].subs({ts[0]: i})
                if expr < 0:
                    if -expr <= offset:
                        continue
                    if strides[1] < 0:
                        continue
                    break
                if expr < shape[1]:
                    idx = i * shape[1] + expr
                    dest_indices.append(idx)
        elif soln[1] == -ts[0]:
            for i in range(shape[1]):
                expr = soln[0].subs({ts[0]: -i})
                if expr < 0:
                    if -expr <= offset:
                        1 / 0
                        continue
                    if strides[0] < offset:
                        1 / 0
                        continue
                    break
                if expr < shape[0]:
                    idx = expr * shape[1] + i
                    dest_indices.append(idx)
        else:
            for t in range(10000000):  # XXX
                i, j = [expr.subs({ts[0]: t}) for expr in soln]
                if i >= shape[0]:
                    break
                if j >= shape[1]:
                    break
                if i >= 0 and j >= 0:
                    idx = i * shape[1] + j
                    dest_indices.append(idx)
            else:
                raise RuntimeError("oops")

        for dest_idx in dest_indices:
            rv_flat[dest_idx] = array_flat[source_idx]
    return rv


def scatter_sparse(array, shape, strides, *, offset=0, output_shape=None):
    """Iterate over the input array to scatter into the output array.

    This is designed to "undo" a gather operation with the same arguments.

    Accumulate values scattered to the same destination.

    """
    if output_shape is None:
        output_shape = shape
    rv = np.zeros(output_shape, dtype=array.dtype)
    array_flat = array.flat
    rv_flat = rv.flat
    for source_idx in range(array.size):
        val = source_idx
        dest_idx = offset
        for size, stride in zip(reversed(shape), reversed(strides)):
            val, index = divmod(val, size)
            dest_idx += index * stride
        if 0 <= dest_idx < rv.size:
            rv_flat[dest_idx] += array_flat[source_idx]
    return rv
