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
"""
Cubic spline interpolation of a set of N 2D-points.
Interpolation is done through N-1 cubic polynomials. First and second derivatives of successive
polynomials at the boundary points must be equal.
See https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation for details.
"""

def coeff(X, Y):
    """
    Given the arrays of X and Y coordinates with the same size N, we compute the coefficients
    deltaX1 and R of the N-1 cubic polynoms.
    :param X: X ccordinates of points (array)
    :param y: Y coordinates of points (array)
    :return: deltaX1 and R arrays of size N-1 and N respectively
    """

    # deltaX1 is X[1] - X[0],.., X[N-1]-X[N-2]
    deltaX1 = X[1:] - X[:-1]
    deltaY1 = Y[1:] - Y[:-1]

    deltaX2 = np.zeros((X.shape[0]-1,))
    deltaX2[1:] = 2.0 * (X[2:] - X[:-2])



    Z = deltaY1 / deltaX1

    W = np.zeros((X.shape[0] - 1,))
    W[1:] = 6 * (Z[1:] - Z[:-1])

    W[2:] = W[2:] - W[1:-1] * deltaX1[1:-1] / deltaX2[1:-1]
    deltaX2[2:] =  deltaX2[2:] - deltaX1[1:-1] * deltaX1[1:-1] / deltaX2[1:-1]

    N = X.shape[0]
    R = np.zeros(X.shape)
    R[0]=0.0
    for i in xrange(N-2, 1, -1):
        R[i] = (W[i] - deltaX1[i] * R[i+1]) / deltaX2[i]
    R[N-1]=0.0

    return deltaX1, R

def cubicSpline(X, Y, v):
    def P(t):
        return t**3 - t
    deltaX1, R = coeff(X,Y)
    i = bisect.bisect(X, v)
    i = i-1
    if i < 0 :
        return Y[0]
    elif i > len(Y)-2:
        return Y[-1]
    t = (v - X[i]) / deltaX1[i]
    value = t * Y[i+1] + (1-t)*Y[i] + deltaX1[i] * deltaX1[i] * (P(t) * R[i+1] + P(1-t) * R[i])/6.0
    return value

def cubicSplineCurve(X,Y):
    xValues = np.arange(np.min(X),np.max(X))
    def f(v):
        return cubicSpline(X,Y,v)
    yValues = np.array(map(f, list(xValues)))
    return xValues, yValues





X= np.array(range(5))
cubicSpline(X, X, 3.5)





