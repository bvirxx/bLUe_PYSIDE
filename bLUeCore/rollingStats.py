"""
This File is part of bLUe software.

Copyright (C) 2017  Bernard Virot <bernard.virot@libertysurf.fr>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Lesser Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

hasOpenCV = False
try:
    import cv2

    hasOpenCV = True
except ImportError:
    pass


def rolling_window(a, winsize):
    """
    Add a last axis to an array, filled with the values of a
    1-dimensional sliding window

    @param a: array
    @type a: ndarray, dtype= any numeric type
    @param winsize: size of the moving window
    @type winsize: int
    @return: array with last axis added
    @rtype: ndarray
    """
    shape = a.shape[:-1] + (a.shape[-1] - winsize + 1, winsize)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def strides_2d(a, r, linear=True):
    """
    Compute the 2D moving windows of an array. The size of the windows
    is h=2*r[0]+1, w=2*r[1]+1, they are centered at the given array item
    and completed by reflection at borders if needed. If linear is True, the
    windows are reshaped as 1D arrays, otherwise they are left unchanged.
    The output array keeps the shape and dtype of a.
    The original idea is taken from

    U{https://gist.github.com/thengineer/10024511}

    @param a: 2D array
    @type a: ndarray, ndims=2
    @param r: window sizes
    @type r: 2-uple of int
    @param linear:
    @type linear: boolean
    @return: array of moving windows
    @rtype: ndarray, shape=a.shape, dtype=a.dtype
    """
    ax = np.zeros(shape=(a.shape[0] + 2 * r[0], a.shape[1] + 2 * r[1]), dtype=a.dtype)
    ax[r[0]:ax.shape[0] - r[0], r[1]:ax.shape[1] - r[1]] = a
    # reflection mode for rows:  ...2,1,0,1,2...
    for i in range(r[0]):
        imod = (i + 1) % a.shape[0] - 1  # cycle through rows if r[0] >= a.shape[0]
        ax[r[0] - i - 1, r[1]:-r[1]] = a[imod + 1, :]  # copy rows a[1,:]... to  ax[r[0]-1,:]...
        ax[i - r[0], r[1]:-r[1]] = a[-imod - 2, :]  # copy rows a[-2,:]... to  ax[-r[0],:]...
    # ax[:r[0],r[1]:-r[1]] = a[::-1,:][-r[0]-1:-1,:]
    # ax[-r[0]:,r[1]:-r[1]] = a[::-1,:][1:r[0]+1,:]
    # reflection mode for cols: cf rows above
    ax[:, :r[1]] = ax[:, ::-1][:, -2 * r[1] - 1:-r[1] - 1]
    ax[:, -r[1]:] = ax[:, ::-1][:, r[1] + 1:2 * r[1] + 1]
    # add two axes and strides for the windows
    shape = a.shape + (1 + 2 * r[0], 1 + 2 * r[1])  # concatenate t-uples
    strides = ax.strides + ax.strides  # concatenate t-uples
    s = as_strided(ax, shape=shape, strides=strides)
    # reshape
    return s.reshape(a.shape + (shape[2] * shape[3],)) if linear else s


def movingAverage(a, winsize, version='kernel'):
    """
    Compute the moving averages of a 1D or 2D array.
    For 1D arrays, the borders are not handled : the dimension of
    the returned array is a.shape[0] - winsize//2.
    For 2D arrays, the window is square (winsize*winsize), the
    borders are handled by reflection and the returned array
    keeps the shape of a. For 2D arrays, if version='kernel' (default)
    we use the opencv function filter2D to compute the moving average. It is
    fast but suffers from a lack of precision. If version = 'strides',
    we perform a direct and more precise computation,
    using 64 bits floating numbers.
    @param a: array
    @type a: ndarray ndims = 1 or 2
    @param winsize: size of moving window
    @type winsize: int
    @param version: 'kernel' or 'strides'
    @type version: str
    @return: array of moving averages
    @rtype: ndarray, dtype = np.float32 if a.ndims==2 and version=='kernel', otherwise
            a.dtype (int types are cast to np.float64)
    """
    n = a.ndim
    if n == 1:
        return np.mean(rolling_window(a, winsize), axis=-1)
    elif n == 2:
        if hasOpenCV and version == 'kernel':
            kernel = np.ones((winsize, winsize), dtype=np.float32) / (winsize * winsize)
            return cv2.filter2D(a.astype(np.float32), -1, kernel.astype(np.float32))
        else:
            r = int((winsize - 1) / 2)
            b = strides_2d(a, (r, r), linear=False)
            m = np.mean(b, axis=(-2, -1))
            return m
    else:
        raise ValueError('array ndims must be 1 or 2')


def movingVariance(a, winsize, version='kernel'):
    """
    Compute the moving variance of a 1D or 2D array.
    For 1D arrays, the borders are not handled : the dimension of
    the returned array is a.shape[0] - winsize//2.
    For 2D arrays, the window is square (winsize*winsize), the
    borders are handled by reflection and the returned array
    keeps the shape of a.
    @param a: array
    @type a: ndarray ndims = 1 or 2
    @param winsize: size of moving window
    @type winsize: int
    @param version:
    @type version: str
    @return: array of moving variances
    @rtype: ndarray, dtype = np.float32 or np.float64
    """
    if a.ndim > 2:
        raise ValueError('array ndims must be 1 or 2')
    if hasOpenCV and version == 'kernel':
        a = a.astype(np.float32)
        """
        f1 = movingAverage(a, winsize, version=version)
        f2 = movingAverage(a * a, winsize, version=version)
        return f2 - f1 * f1
        """
    else:
        a = a.astype(np.float64)
    # faster than np.var !!!
    f1 = movingAverage(a, winsize, version=version)
    f2 = movingAverage(a * a, winsize, version=version)
    return f2 - f1 * f1
