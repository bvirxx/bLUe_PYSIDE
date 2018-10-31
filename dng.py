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
import os
from os.path import basename

import exiftool
import numpy as np

#########################################################################################
# Functions and classes related to profile tags
# compliant with the Adobe DNG specification
# cf. https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/dng_spec_1.4.0.0.pdf
########################################################################################
from bLUeGui.spline import cubicSpline
from settings import DNG_PROFILES_DIR2, DNG_PROFILES_DIR1


def getDngProfileDict(filename):
    """
    Read profile related binary tags from a .dng or .dcp file
    @param filename:
    @type filename: str
    @return:
    @rtype: dict of decoded str
    """
    #if filename[-3:].lower() not in ['dng', 'dcp']:
        #raise ValueError("getProfileDict : wrong file type")
    with exiftool.ExifTool() as e:
        profileDict = e.readBinaryDataAsDict(filename, taglist=['LinearizationTable', 'ProfileLookTableData', 'ProfileToneCurve'])
    return profileDict

def getDngProfileList(cameraName):
    plist = []
    if cameraName == '':
        return plist
    cameraName = cameraName.lower()
    for folder in [DNG_PROFILES_DIR1, DNG_PROFILES_DIR2]:
        for filename in os.listdir(folder):
            if cameraName in basename(filename.lower()):
                plist.append(folder + filename)
    return plist

class dngProfileToneCurve:
    #arrays of x - coordinates and y - coordinates of
    # the tone curve.All coordinates are floats in the interval [0, 1].
    dataX, dataY = None, None
    def __init__(self, buf):
        """
        Init the coordinates from a bytes buffer.
        @param buf: as read by exiftool -b and decoded
        @type buf: str
        """
        try:
            buf = buf.split(' ')
            buf = np.array([float(x) for x in buf])
            self.dataX, self.dataY = buf[::2], buf[1::2]
        except (ValueError, AttributeError):
            # identity curve
            self.dataX, self.dataY = np.array([0,1]), np.array([0,1])

    def toLUTXY(self, range=8):
        """
        interpolate the data points by a cubic spline (cf adobe dng specification p. 56).
        @param range: 8 (8 bits images) or 16 (16 bits images)
        @type range: int
        @return: interpolated cubic spline 0..255 ---> 0..255
        @rtype: ndarray
        """
        coeff = 255 if range == 8 else 65535
        LUTXY = cubicSpline(self.dataX * coeff, self.dataY * coeff, np.arange(coeff+1))
        return LUTXY

