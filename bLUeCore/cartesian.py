"""
This file is part of bLUe software.
"""

# the code below is taken from the site https://gist.github.com/hernamesbarbara/68d073f551565de02ac5
# We gratefully acknowledge the contribution of the author.

import numpy as np


def cartesianProduct(arrayList, out=None):
    """
    Builds the cartesian product of
    N one dimensional array-like objects as a numpy array
    with shape (a1.shape,...,aN.shape, N). All arrays must have identical
    dtypes, otherwise a ValueError exception is raised.

    :param arrayList : list or tuple of 1-D array-like objects, with identical dtype
    :type arrayList: list or tuple
    :param out : used only for recursive calls
    :return: the cartesian product of the arrays
    :rtype: ndarray
    :raise ValueError
    """
    arrayList = [np.asarray(x) for x in arrayList]
    n = np.prod([x.size for x in arrayList])
    # empty product
    if n == 0:
        return np.array([])
    itemSize = len(arrayList)
    dtype = arrayList[0].dtype
    for a in arrayList[1:]:
        if a.dtype != dtype:
            raise ValueError("cartesianProduct : all arrays must have the same dtype")
    m = n // arrayList[0].size
    if out is None:
        out = np.zeros([n, itemSize], dtype=dtype)
    out[:, 0] = np.repeat(arrayList[0], m)
    if arrayList[1:]:
        cartesianProduct(arrayList[1:], out=out[0:m, 1:])
        for j in range(1, arrayList[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    dims = tuple([x.size for x in arrayList] + [len(arrayList)])
    return np.reshape(out, dims)
