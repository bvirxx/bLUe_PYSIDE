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
from PySide6.QtCore import QPointF


#####################
# displacement spline
#####################

def displacementSpline(X, Y, V, period=0, clippingInterval=None):
    """
     Calculate the y-coordinates corresponding to the x-coordinates V for
    the dsiplacement spline defined by the control points zip(X,Y).
    A ValueError exception is raised if the X values are not distinct.

    :param X: x-coordinates of control points, sorted in increasing order
    :type  X: ndarray, dtype=float
    :param Y: y-coordinates of control points
    :type  Y: ndarray, dtype=float
    :param V: x-coordinates of spline points
    :type  V: ndarray, dtype=float
    :param period:
    :type  period: int
    :param clippingInterval:
    :type  clippingInterval:
    :return: y-coordinates of spline points
    :rtype: ndarray, dtype=np.float
    """

    def bumpVec(V, t1, t2, b, period=period):
        # return the y-coordinates corresponding to the x-coordinates
        # given by V, for the (periodic if period > 0) bump spline (t1, t2, b)
        M = (t1 + t2) / 2
        if period > 0:
            # find the nearest neighbor of M congruent with V modulo period
            K = (V - M) / period
            K = np.round(K)
            V = V - period * K
        return np.where(V <= t1, 0,
                        np.where(V >= t2, 0, np.where(V < M, b * (V - t1) / (M - t1), b * (t2 - V) / (t2 - M))))

    left, right, bump = X[::2], X[1::2], Y
    tmp = np.vstack(([bumpVec(V, left[i], right[i], bump[i], period=period) for i in range(left.shape[0])]))
    fPlus = np.amax(tmp, axis=0, initial=0)
    fMoins = np.amax(-tmp, axis=0, initial=0)
    return fPlus - fMoins


###############################
# Quadratic interpolation spline
###############################

def interpolationQuadSpline(a, b, d):
    """
    Builds a monotonic transformation curve T from [0,1] onto b[0], b[-1],
    as a piecewise quadratic interpolation spline to a set of (a[k], b[k]) 2D points.
    Coefficients d[k] are the slopes at nodes. The function returns a tabulation of T as
    a list of T[k/255] for k in range(256).
    a and b must be non decreasing sequences in range 0..1, with a strictly increasing.
    We assume d >=0 and d[k] = d[k-1] = 0 if b[k]=b[k-1]. These conditions are
    necessary conditions for the existence of a non decreasing interpolation spline
    cf https://pdfs.semanticscholar.org/1fee/3b0eab9828dd772cc4e735d132bd153b007f.pdf
    The 3 arrays must have the same length.
    Note. for k < a[0], T[k]=b[0] and, for k > a[-1], T[k]=b[-1]

    :param a: x-coordinates of nodes
    :type  a: ndarray, dtype float
    :param b: y-coordinates of nodes
    :type  b: ndarray, dtype=float
    :param d: slopes at nodes
    :type  d: ndarray, dtype=float
    :return: Spline table
    :rtype: ndarray, dtype=float
    """
    assert len(a) == len(b) == len(d)
    if np.min(a[1:] - a[:-1]) <= 0:
        raise ValueError('histogram.InterpolationSpline : a must be strictly increasing')
    # x-coordinate range
    x = np.arange(256, dtype=float) / 255
    x = np.clip(x, a[0], a[-1])
    # find  node intervals containing x : for each i, get smallest j s.t. a[j] > x[i]
    tmp = np.fromiter(((a[j] > x[i]) for j in range(len(a)) for i in range(len(x))), dtype=bool)
    tmp = tmp.reshape(len(a), len(x))
    k = np.argmax(tmp, axis=0)  # a[k[i]-1]<= x[i] < a[k[i]] if k[i] > 0, and x[i] out of a[0],..a[-1] otherwise
    k = np.where(x >= a[-1], len(a) - 1, k)
    r = (b[1:] - b[:-1]) / (a[1:] - a[:-1])  # r[k] = (b[k] - b[k-1]) / (a[k] - a[k-1])
    r = np.concatenate(([0], r))
    t = (x - a[k - 1]) / (a[k] - a[k - 1])  # t[k] = (x - a[k-1]) / (a[k] - a[k-1]) for x in a[k-1]..a[k]
    # sanity check
    assert np.all(t >= 0)
    assert np.all(t <= 1)
    # tabulate spline
    t1 = (1 - t) * t
    with np.errstate(divide='ignore', invalid='ignore'):
        T = b[k - 1] + (r[k] * t * t + d[k - 1] * t1) * (b[k] - b[k - 1]) / (r[k] + (d[k] + d[k - 1] - 2 * r[k]) * t1)
    # T should be constant in intervals where r[k] = 0 : we replace nan by the preceding (non NaN) value in T.
    # To enable arithmetic comparisons, we use  a value < b[0] to mark the components to be replaced.
    T = np.where(np.isnan(T), b[0] - 100, T)
    T[0] = max(b[0], T[0])
    return np.maximum.accumulate(T)


#################################################################################################
# Cubic interpolation spline for a set of N 2D-points.
# Interpolation is done through N-1 cubic polynomials. First and second derivatives of successive
# polynomials at the boundary points must be equal.
# See https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation for more information.
###################################################################################################

def coeff(X, Y):
    """
    Given two arrays of X and Y coordinates with the same size N, the method computes the coefficients
    deltaX1 and R of the N-1 cubic polynomials.
    Polynomial Pi is Pi(t) = t * Y[i+1] + (1-t)*Y[i] + deltaX1[i] * deltaX1[i] * (P(t) * R[i+1] + P(1-t) * R[i])/6.0,
    where t = (x - X[i]) / deltaX1[i] and P(t) = t**3 - t.
    If 2 points have identical x-coordinates, a ValueError exception is raised.

    :param X: X ccordinates of points
    :type  X: ndarray, dtype=float
    :param Y: Y coordinates of points
    :type  Y: ndarray, dtype=float
    :return: deltaX1 and R arrays of size N-1 and N respectively
    :rtype: ndarray, dtype=float
    """
    old_settings = np.seterr(all='ignore')
    deltaX1 = X[1:] - X[:-1]
    deltaY1 = Y[1:] - Y[:-1]
    D1 = deltaY1 / deltaX1
    np.seterr(**old_settings)
    if not np.isfinite(D1).all():
        raise ValueError()
    deltaX2 = np.zeros((X.shape[0] - 1,))
    deltaX2[1:] = 2.0 * (X[2:] - X[:-2])  # tangent
    W = np.zeros((X.shape[0] - 1,))
    W[1:] = 6 * (D1[1:] - D1[:-1])  # second derivative * deltaX1 *6
    W[2:] = W[2:] - W[1:-1] * deltaX1[1:-1] / deltaX2[1:-1]
    deltaX2[2:] = deltaX2[2:] - deltaX1[1:-1] * deltaX1[1:-1] / deltaX2[1:-1]
    N = X.shape[0]
    R = np.zeros(X.shape)
    for i in range(N - 2, 0, -1):
        R[i] = (W[i] - deltaX1[i] * R[i + 1]) / deltaX2[i]
    return deltaX1, R


def cubicSpline(X, Y, V):
    """
    Calculates the y-coordinates corresponding to the x-coordinates V for
    the cubic spline interpolating the control points zip(X,Y).
    A ValueError exception is raised if the X values are not distinct.

    :param X: x-coordinates of control points, sorted in increasing order
    :type  X: ndarray, dtype=float
    :param Y: y-coordinates of control points
    :type  Y: ndarray, dtype=float
    :param V: x-coordinates of spline points
    :type  V: ndarray, dtype=float
    :return: y-coordinates of the spline points
    :rtype: ndarray, dtype=float
    """

    def P(t):
        return t ** 3 - t

    deltaX1, R = coeff(X, Y)  # raises ValueError if two X values are equal
    i = np.searchsorted(X, V, side='right') - 1
    isave = i
    i = np.clip(i, 0, len(Y) - 2)
    t = (V - X[i]) / deltaX1[i]
    values = t * Y[i + 1] + (1 - t) * Y[i] + deltaX1[i] * deltaX1[i] * (P(t) * R[i + 1] + P(1 - t) * R[i]) / 6.0
    values = np.where(isave > len(Y) - 2, Y[-1], values)
    values = np.where(isave < 0, Y[0], values)
    return values


def interpolationCubSpline(X, Y, clippingInterval=None):
    """
    Interpolates a set of 2D points by a cubic spline.
    The spline has exactly 256 sampling points.
    X and Y must have equal sizes.
    A ValueError exception is raised if the X values are not distinct.

    :param X: x-coordinates of points
    :type  X: list of float
    :param Y: y-coordinates of points
    :type  Y: list of float
    :param clippingInterval: min and max values for the spline y-values
    :type  clippingInterval: 2-uple of float
    :return: the interpolated cubic spline
    :rtype: list of QPointF (length 256)
    """
    m, M = np.min(X), np.max(X)
    step = (M - m) / 255.0
    xValues = np.arange(256) * step + m
    yValues = cubicSpline(X, Y, xValues)
    if clippingInterval is not None:
        minY, maxY = clippingInterval[0], clippingInterval[1]
        yValues = np.clip(yValues, minY, maxY)
    return [QPointF(x, y) for x, y in zip(xValues, yValues)]
