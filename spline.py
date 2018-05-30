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
import bisect

import numpy as np
from PySide2.QtCore import QPointF

###############################
# Quadratic interpolation spline
################################

def interpolationQuadSpline(a, b, d, plot=False):
    """
    Builds a monotonic transformation curve T from [0,1] onto b[0], b[-1],
    as a piecewise rational quadratic interpolation spline to a set of (a[k], b[k]) 2D nodes.
    Coefficients d[k] are the slopes at nodes. The function returns a tabulation of T as
    a list of T[k/255] for k in range(256).
    a and b must be non decreasing sequences in range 0..1, with a strictly increasing.
    The 3 arrays must have the same length.
    Note. for k < a[0], T[k]=b[0] and, for k > a[-1], T[k]=b[-1]
    @param a: x-coordinates of nodes
    @type a: ndarray, dtype np.float
    @param b: y-coordinates of nodes
    @type b: ndarray, dtype=np.float
    @param d: slopes at nodes
    @type d: ndarray, dtype=np.float
    @return: Spline table
    @rtype: ndarray, dtype=np.float
    """
    if np.min(a[1:] - a[:-1]) <= 0:
        raise ValueError('histogram.InterpolationSpline : a must be strictly increasing')
    # x-coordinate range
    x = np.arange(256, dtype=np.float)/255
    x = np.clip(x, a[0], a[-1])
    #find  node intervals containing x : for each i, get smallest j s.t. a[j] > x[i]
    tmp = np.fromiter(((a[j]> x[i]) for j in range(len(a)) for i in range(len(x))), dtype=bool)
    tmp = tmp.reshape(len(a), len(x))
    k = np.argmax(tmp, axis=0)                     # a[k[i]-1]<= x[i] < a[k[i]] if k[i] > 0, and x[i] out of a[0],..a[-1] otherwise
    k = np.where(x >= a[-1], len(a) - 1, k)
    r = (b[1:] - b[:-1]) / (a[1:] - a[:-1])        # r[k] = (b[k] - b[k-1]) / (a[k] - a[k-1])
    r = np.concatenate(([0], r))
    t = (x-a[k-1])/(a[k]-a[k-1])                   # t[k] = (x - a[k-1]) / (a[k] - a[k-1]) for x in a[k-1]..a[k]
    # sanity check
    assert np.all(t>=0)
    assert np.all(t<=1)
    # tabulate spline
    T = b[k-1] + (r[k]*t*t + d[k-1]*(1-t)*t)*(b[k]-b[k-1]) / (r[k]+(d[k]+d[k-1] - 2*r[k])*(1-t)*t)
    """
    if plot:
        #import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        M = np.int(a[-1]*255)
        ax.plot(T*255)
        ax.plot([i for i in range(256)])
        ax.plot(a*255, b*255, 'ro')
        #ax.plot(a*255, a*255, 'bo')
        fig.savefig('temp.png')
        plt.close()
    """
    return T


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
    @param X: X ccordinates of points
    @type X: ndarray, dtype=float
    @param Y: Y coordinates of points
    @type Y: ndarray, dtype=np.float
    @return: deltaX1 and R arrays of size N-1 and N respectively
    @rtype: ndarray, dtype=np.float
    """
    old_settings = np.seterr(all='ignore')
    deltaX1 = X[1:] - X[:-1]
    deltaY1 = Y[1:] - Y[:-1]
    D1= deltaY1 / deltaX1
    np.seterr(**old_settings)
    if np.isnan(D1).any() or (D1==np.inf).any() or (D1==-np.inf).any():
        raise ValueError()
    deltaX2 = np.zeros((X.shape[0]-1,))
    deltaX2[1:] = 2.0 * (X[2:] - X[:-2])  #tangent
    W = np.zeros((X.shape[0] - 1,))
    W[1:] = 6 * (D1[1:] - D1[:-1]) # second derivative * deltaX1 *6
    W[2:] = W[2:] - W[1:-1] * deltaX1[1:-1] / deltaX2[1:-1]
    deltaX2[2:] =  deltaX2[2:] - deltaX1[1:-1] * deltaX1[1:-1] / deltaX2[1:-1]
    N = X.shape[0]
    R = np.zeros(X.shape)
    for i in range(N-2, 0, -1):
        R[i] = (W[i] - deltaX1[i] * R[i+1]) / deltaX2[i]
    return deltaX1, R

def cubicSpline(X, Y, V):
    """
    Calculates the y-coordinates corresponding to the x-coordinates V for
    the cubic spline defined by the control points zip(X,Y).
    A ValueError exception is raised if the X values are not distinct.
    @param X: x-coordinates of control points, sorted in increasing order
    @type X: ndarray, dtype=np.float
    @param Y: y-coordinates of control points
    @type Y: ndarray, dtype=np.float
    @param V: x-coordinates of spline points
    @type V: ndarray, dtype=np.float
    @return: y-coordinates of the spline points
    @rtype: ndarray, dtype=np.float
    """
    def P(t):
        return t**3 - t
    deltaX1, R = coeff(X,Y)  # raises ValueError if two X values are equal
    #i = bisect.bisect(X, v)
    i = np.searchsorted(X, V, side='right') - 1
    isave = i
    i = np.clip(i,0, len(Y)-2)
    t = (V - X[i]) / deltaX1[i]
    values = t * Y[i+1] + (1-t)*Y[i] + deltaX1[i] * deltaX1[i] * (P(t) * R[i+1] + P(1-t) * R[i])/6.0
    values = np.where(isave>len(Y)-2, Y[-1], values)
    values = np.where(isave<0, Y[0], values)
    return values
    """
    if i < 0 :
        return Y[0]
    elif i > len(Y)-2:
        return Y[-1]
    #  0<=i<=N-2
    t = (v - X[i]) / deltaX1[i]
    value = t * Y[i+1] + (1-t)*Y[i] + deltaX1[i] * deltaX1[i] * (P(t) * R[i+1] + P(1-t) * R[i])/6.0
    return value
    """

def interpolationCubSpline(X, Y, clippingInterval=None):
    """
    Interpolates a set of 2D points by a cubic spline.
    The returned list has exactly 256 sampling points.
    A ValueError exception is raised if the X values are not distinct.
    @param X: x-coordinates of points
    @type X: list of float
    @param Y: y-coordinates of points
    @type Y: list of float
    @param clippingInterval: min and max values for spline y-values
    @type clippingInterval: 2-uple-like of float values
    @return: the interpolated cubic spline
    @rtype: list of QPointF (length 256)
    """
    m, M = np.min(X) , np.max(X)
    step = (M - m) / 255.0
    xValues = np.arange(256) * step + m
    yValues = cubicSpline(X, Y, xValues)
    if clippingInterval is not None:
        minY, maxY = clippingInterval[0], clippingInterval[1]
        yValues = np.clip(yValues, minY, maxY)
    return [QPointF(x,y) for x,y in zip(xValues, yValues)]






