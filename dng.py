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
# Functions and classes related to profile tags.
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
    with exiftool.ExifTool() as e:
        profileDict = e.readBinaryDataAsDict(filename,
                                             taglist=['LinearizationTable',
                                                      'ProfileLookTableData',
                                                      'ProfileLookTableDims',
                                                      'ProfileLookTableEncoding',
                                                      'ProfileToneCurve'
                                                      ])
    return profileDict

def getDngProfileList(cameraName):
    """
    Search for paths to profiles for a camera model.
    @param cameraName: camera model
    @type cameraName: str
    @return: list of paths to profiles
    @rtype: list of str
    """
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
    # THe attributes datX and dataY are the arrays of x-coordinates
    # and y-coordinates for the tone curve. They share the same length.
    # All coordinates are floats in the interval [0, 1].
    def __init__(self, buf):
        """
        Init the coordinates from a bytes buffer of interleaved x,y coordinates
        as read by exiftool -b and decoded.
        @param buf: decoded buffer
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
        @type maxrange: int
        @return: interpolated cubic spline : [0, maxrange] ---> [0, maxrange]
        @rtype: ndarray
        """
        return cubicSpline(self.dataX * maxrange, self.dataY * maxrange, np.arange(maxrange + 1))

class dngProfileLookTable:
    """
    (hue, saturation, value) 3D LUT.
    Property data gives the table array
    Property divs gives the number of division points for each axis.
    Due to modulo arithmetic for hue and to the presence of sentinels,
    divs and data.shape are different.
    Input values for axis=i must be mapped to the (closed) interval  [0, divs[i]]
    """
    def __init__(self, dngDict):
        """
        Init a profile look table from a dictionary of (tagname, str) pairs.
        Tags are 'ProfileLookTableDims', 'ProfileLookTableEncoding', 'ProfileLookTableData'.
        Values are decoded following the Adobe dng spec.
        @param dngDict:
        @type dngDict: dict
        """
        self.isValid = False
        divs, encoding, data = dngDict.get('ProfileLookTableDims', None), dngDict.get('ProfileLookTableEncoding', None), dngDict.get('ProfileLookTableData', None)
        if divs is None  or data is None:  # encoding not used yet : it seems to be missing in dng files
            return
        try:
            # read encoding : may be missing
            try:
                self.encoding = int(encoding)  # 0: linear, 1 : sRGb
            except TypeError:
                self.encoding = 0
            # read the number of division points for each axis.
            divs = [int(x) for x in divs.split(' ')]
            # read data. Tthe table is stored in v, h, s loops ordering (cf. the dng specification)
            data = np.array([float(x) for x in data.split(' ')]).reshape(divs[2], divs[0], divs[1], 3)  # v, h, s
            # add a division point for hue = 360 (cf. dng spec p. 82)
            divs[0] += 1
            self.__divs = tuple(divs)
            # allocate data array.
            # adding sentinels, so all
            # dims are increased by +1 (Sentinels allow to
            # use closed intervals instead of right-opened intervals
            # as input ranges).
            buf = np.zeros((divs[0] + 1, divs[1] + 1, divs[2] + 1, 3), dtype = np.float) + (0, 1, 1)
            # move axes to h, s, v ordering
            data = np.moveaxis(data, (0,1,2), (2, 0, 1))
            # put values into table, starting from index 0.
            buf[0:-2, :-1, :-1, : ] = data[:, :, :, ]
            # modulo arithmetic for hue
            buf[-2,:,:,0] = buf[0,:,:,0]
            # interpolation does not use the values of sentinels faces, so don't care
            self.__data = buf
        except (ValueError, TypeError) as e:
            print('dngProfileLooktable : ', str(e))
        self.isValid = True

    @property
    def divs(self):
        """
        Count of division points for each axis.

        @return:
        @rtype: 3-uple of int
        """
        return self.__divs

    @property
    def data(self):
        return self.__data


