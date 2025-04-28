"""
This file is part of bLUe software.
"""

import numpy as np


def cartesianProduct(arrays):
    """
    Builds the cartesian product of
    N one dimensional array-like objects as a numpy array
    with shape (a_1.shape,...,a_N.shape, N).
    All arrays must have the same dtype.

    :param arrays : list or tuple of 1-D array-like objects, with identical dtype
    :type arrays: list or tuple of 1-D array-like objects
    :return: ndarray representing the Cartesian product
    :rtype: ndarray
    :raises ValueError: if input arrays do not share the same dtype
    """
    if not arrays:
        return np.array([])
    arrays = [np.asarray(arr) for arr in arrays]
    dtype = arrays[0].dtype
    if any(arr.dtype != dtype for arr in arrays):
        raise ValueError("cartesianProduct : all arrays must have the same dtype")

    def recursive_cartesian(arrays, out=None):
        total_combinations = np.prod([arr.size for arr in arrays])
        if total_combinations == 0:
            return np.array([])

        if out is None:  # unique allocation of the whole array
            # all axes but the last are flattened
            out = np.zeros((total_combinations, len(arrays)), dtype=dtype)

        repeat_factor = total_combinations // arrays[0].size
        # copy 1st array
        out[:, 0] = np.repeat(arrays[0], repeat_factor)

        if len(arrays) > 1:
            # copy remaining arrays
            recursive_cartesian(arrays[1:], out=out[0:repeat_factor, 1:])
            for i in range(1, arrays[0].size):
                out[i * repeat_factor:(i + 1) * repeat_factor, 1:] = out[0:repeat_factor, 1:]

        return out

    result = recursive_cartesian(arrays)
    final_shape = tuple(arr.size for arr in arrays) + (len(arrays),)
    return result.reshape(final_shape)
