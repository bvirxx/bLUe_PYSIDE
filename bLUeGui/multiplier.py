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

###############################################################
# The functions in this module establish the correspondence between
# (temperature, tint) and the RGB mutipliers mR, mG, mB.
# The idea is as follow :
# The planes mB/mR = constant in the RGB color space correspond to lines y=kx
# in the xy color space. In this later space, consider the point of intersection m1 of
# the locus (white points) with the line y = kx. It gives the temperature T and
# the tint corresponds to an homothety applied to m1, which gives the final point m.
####################################################################

import numpy as np
from .colorCIE import temperatureAndTint2xy, conversionMatrix, temperature2xyWP

def temperatureAndTint2RGBMultipliers(temp, tint, XYZ2RGBMatrix):
    """
    Converts temperature and tint to RGB multipliers, as White Point RGB coordinates,
    modulo tint green correction (mG = WP_G * tint)
    We compute the xy coordinates of the white point WP(T) by the Robertson's method.
    Next, we transform these coordinates to RGB values (mR,mG,mB), using the
    conversion matrix XYZ2RGBMatrix.
    Multipliers are m1 = mR, m2 = mG*tint, m3 = mB. For convenience
    the function returns the 4 values m1, m2, m3, m2, scaled to min(m1,m2,m3)=1.
    The tint factor should be between 0.2 and 2.5
    @param temp: temperature
    @type temp: float
    @param tint: Tint factor
    @type tint: float
    @param XYZ2RGBMatrix: conversion matrix from XYZ to linear RGB
    @type XYZ2RGBMatrix: 3x3 array
    @return: 4 multipliers (RGBG)
    @rtype: 4-uple of float
    """
    # WP coordinates for temp
    x, y = temperatureAndTint2xy(temp, 0)
    # transform to XYZ coordinates
    X, Y, Z = x / y, 1.0, (1.0 - x - y) / y
    # WP RGB coordinates
    m1, m2, m3 = np.dot(XYZ2RGBMatrix, [X, Y, Z])
    # apply tint correction (green-magenta shift) to G channel.
    m2 = m2 * tint
    mi = min((m1, m2, m3))
    m1, m2, m3 = m1 / mi, m2 / mi, m3 / mi
    return m1, m2, m3, m2


def convertMultipliers(Tdest, Tsource, tint, m):
    M = conversionMatrix(Tdest, Tsource)
    m1 = M[0, 0] / m[0]
    m2 = M[1, 1] / m[1] * tint
    m3 = M[2, 2] / m[2]
    mi = min((m1, m2, m3))
    m1, m2, m3 = m1 / mi, m2 / mi, m3 / mi
    return m1, m2, m3, m2


def RGBMultipliers2TemperatureAndTint(mR, mG, mB, XYZ2RGBMatrix):
    """
    Evaluation of the temperature and tint correction corresponding to a
    set of 3 RGB multipliers. They are interpreted as the RGB coordinates of a white point.
    The aim is to find a temperature T with a
    corresponding white point WP(T), and a factor tint, such that mB/mR = WPb/WPr
    and mG*tint/mR = WpG/WPR. As mutipliers are invariant by scaling, this
    function can be seen as the inverse function
    of temperatureAndTint2RGBMultipliers.
    We consider the function f(T) = WPb/WPr giving
    the ratio of blue over red coordinates for the white point WP(T). Assuming  f is monotonic,
    we solve the equation f(T) = mB/mR by a simple dichotomous search.
    Then, the tint is simply defined as the scaling factor mu verifying tint * mG/mR = WPG/WPR
    The RGB space used is defined by the matrix XYZ2RGBMatrix.
    Note that to be inverse functions, RGBMultipliers2Temperature and temperatureAndTint2RGBMultipliers
    must use the same XYZ2RGBMatrix.
    @param mR:
    @type mR:
    @param mG:
    @type mG:
    @param mB:
    @type mB:
    @param XYZ2RGBMatrix:
    @type XYZ2RGBMatrix:
    @return: the evaluated temperature and the tint correction
    @rtype: 2-uple of float
    """
    # search for T
    Tmin, Tmax = 1667.0, 15000.0
    while (Tmax - Tmin) > 10:
        T = (Tmin + Tmax) / 2.0
        x, y = temperature2xyWP(T)  # TODO temperature2xyWP(T) = temperatureAndTint2xy(T,0) ???
        X, Y, Z = x / y, 1, (1 - x - y) / y
        r, g, b = np.dot(XYZ2RGBMatrix, [X, Y, Z])
        if (b / r) > (mB / mR):
            Tmax = T
        else:
            Tmin = T
    # get tint correction
    green = (r / g) * (mG / mR)
    if green < 0.2:
        green = 0.2
    if green > 2.5:
        green = 2.5
    return round(T / 10) * 10, green

