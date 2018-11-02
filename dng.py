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
        profileDict = e.readBinaryDataAsDict(filename, taglist=['LinearizationTable', 'ProfileLookTableData', 'ProfileLookTableDims',
                                                                'ProfileLookTableEncoding', 'ProfileToneCurve'])
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
        Init the coordinates from a bytes buffer as read by exiftool -b and decoded.
        @param buf:
        @type buf: str
        """
        try:
            buf = buf.split(' ')
            buf = np.array([float(x) for x in buf])
            self.dataX, self.dataY = buf[::2], buf[1::2]
        except (ValueError, AttributeError):
            # identity curve
            self.dataX, self.dataY = np.array([0,1]), np.array([0,1])

    def toLUTXY(self, maxrange=255):
        """
        interpolate the data points by a cubic spline (cf adobe dng specification p. 56).
        @param maxrange: max of data range (identical for input and output)
        @type range: int
        @return: interpolated cubic spline : [0, maxrange] ---> [0;;maxrange]
        @rtype: ndarray
        """
        LUTXY = cubicSpline(self.dataX * maxrange, self.dataY * maxrange, np.arange(maxrange + 1))
        return LUTXY

class dngProfileLookTable:
    """
    hue, saturation, value mapping table
    """

    def __init__(self, dngDict):
        """
        Init a profile look table from a dictionary of (tagname, str) pairs
        Dictionary values are decoded following the Adobe dng spec.
        @param dngDict:
        @type dngDict:
        """
        self.isValid = False
        dims, encoding, data = dngDict.get('ProfileLookTableDims', None), dngDict.get('ProfileLookTableEncoding', None), dngDict.get('ProfileLookTableData', None)
        if dims is None  or data is None:  # encoding not used yet : it seems to be missing in some dng files
            return
        try:
            # raed dims of the LookTable
            dims = [int(x) for x in dims.split(' ')]
            self.dims = tuple(dims)  # h, s, v counts of division points
            # read encoding
            try:
                self.encoding = int(encoding)  # 0: linear, 1 : sRGb
            except TypeError:
                self.encoding = 0
            # allocate data array
            buf = np.zeros((dims[0]+1, dims[1], dims[2], 3), dtype = np.float) + (0, 1, 1)
            # read data h inices start from 0, s, v indice
            # the table is stored in v, h, s loops ordering (cf. the dng specification)
            data = np.array([float(x) for x in data.split(' ')]).reshape(dims[2], dims[0], dims[1], 3) # v, h, s
            # move to h, s, v ordering for axes
            data = np.moveaxis(data, (0,1,2), (2, 0, 1)) # h, s, v
            # h, s, v start from index 0
            buf[0:-1, :, :, : ] = data[:, :, :, ]
            buf[-1,:,:,0] = buf[0,:,:,0]
            self.data = buf
        except (ValueError, TypeError) as e:
            print(str(e))
        self.isValid = True


