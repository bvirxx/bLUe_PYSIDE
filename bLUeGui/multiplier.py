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
# The functions in this module establish the back abd forth correspondence
# between (temperature, tint) and the camera mutipliers (mR, mG, mB).
# The camera is characterized by a color matrix XYZ2CameraMatrix.
# Currently, the matrix is supposed constant for each camera model :
# we do not use interpolation between the ColorMatrix1 and ColorMatrix2 tags
# (cf. Adobe dng spec. p. 80 )
####################################################################

import numpy as np
from .colorCIE import temperatureAndTint2xy, temperature2xyWP

def CIExyY2XYZ(x, y):
    """
    Based on CIE xyY color space, return the XYZ expansion
    of (x,y),  assuming Y=1.
    cf https://en.wikipedia.org/wiki/CIE_1931_color_space
    @param x: x coordinate
    @type x: float
    @param y: y coordinate
    @type y: float
    @return: XYZ coordinates
    @rtype: 3-uple of float
    """
    return x / y, 1, (1 - x - y) / y

def XYZ2CIExyY(X, Y, Z):
    """
    Return the two first coordinates
    of (X,Y,Z) in CIE xyY
    cf https: // en.wikipedia.org / wiki / CIE_1931_color_space

    """
    s = X + Y + Z
    return X / s, Y / s

def temperatureAndTint2RGBMultipliers(temp, tint, XYZ2CameraMatrix):
    """
    Convert temperature and tint to RGB multipliers and apply a
    tint shift : mG = WP_G * tint.
    The matrix XYZ2CameraMatrix should the interpolated Camera Color Matrix for the
    temperature T.
    We compute the xy coordinates of the white point WP(T) by the Robertson's method.
    Next, we transform these coordinates to camera RGB values (mR,mG,mB), using the
    conversion matrix XYZ2RGBMatrix.
    Multipliers are m1 = mR, m2 = mG*tint, m3 = mB. For convenience
    the function returns the 4 values m1, m2, m3, m2, scaled to min(m1,m2,m3)=1.
    The tint factor should be between 0.2 and 2.5
    @param temp: temperature
    @type temp: float
    @param tint: Tint factor
    @type tint: float
    @param XYZ2CameraMatrix: conversion matrix from XYZ to camera RGB
    @type XYZ2CameraMatrix: 3x3 array
    @return: 4 multipliers (RGBG)
    @rtype: 4-uple of float
    """
    # get the coordinates of WP(temp) in xy color space.
    # We use temperatureAndTint2xy(temp, 0).
    # We could also call temperature2xyWP(temp).
    # As the two methods use different methods of approximation,
    # they return slightly different results.
    WP_x, WP_y = temperatureAndTint2xy(temp, 0)
    # expand to XYZ color space
    WP_X, WP_Y, WP_Z = CIExyY2XYZ(WP_x, WP_y)  # x / y, 1.0, (1.0 - x - y) / y
    # Convert to camera neutral
    m1, m2, m3 = np.dot(XYZ2CameraMatrix, [WP_X, WP_Y, WP_Z])
    # apply tint shift (green-magenta shift) to G channel.
    m2 = m2 * tint
    mi = min((m1, m2, m3))
    m1, m2, m3 = m1 / mi, m2 / mi, m3 / mi
    return m1, m2, m3, m2

def RGBMultipliers2TemperatureAndTint(mR, mG, mB, XYZ2CameraMatrix):
    """
    Inverse function of temperatureAndTint2RGBMultipliers.
    Compute the temperature and tint correction corresponding to a
    set of 3 RGB multipliers.
    The matrix XYZ2CameraMatrix should the interpolated Camera Color Matrix for the
    temperature T.
    The aim is to find a temperature T with a corresponding white point WP(T),
    and a factor tint, such that mB/mR = WPb/WPr and mG*tint/mR = Wpg/WPr.
    We consider the function f(T) = WPb/WPr giving
    the ratio of blue over red coordinates for the white point WP(T). Assuming  f is monotonic,
    we solve the equation f(T) = mB/mR by a simple dichotomous search.
    Then, the tint is simply defined as the scaling factor mu verifying tint * mG/mR = WPG/WPR
    Note that to be inverse functions, RGBMultipliers2Temperature and temperatureAndTint2RGBMultipliers
    must use the same XYZ2CameraMatrix.
    @param mR:
    @type mR:
    @param mG:
    @type mG:
    @param mB:
    @type mB:
    @param XYZ2CameraMatrix:
    @type XYZ2CameraMatrix:
    @return: temperature and tint correction
    @rtype: 2-uple of float
    """
    # search for T
    Tmin, Tmax = 1667.0, 15000.0
    while (Tmax - Tmin) > 10:
        T = (Tmin + Tmax) / 2.0
        x, y = temperature2xyWP(T)  # ~ temperatureAndTint2xy(T,0)
        # expand to the XYZ color space
        X, Y, Z = CIExyY2XYZ(x, y) #= x / y, 1, (1 - x - y) / y
        # Convert to camera neutral
        r, g, b = np.dot(XYZ2CameraMatrix, [X, Y, Z])
        if (b / r) > (mB / mR):
            Tmax = T
        else:
            Tmin = T
    # compute tint shift
    green = (r / g) * (mG / mR)
    if green < 0.2:
        green = 0.2
    if green > 2.5:
        green = 2.5
    return round(T / 10) * 10, green

