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

from bLUeTop import exiftool
import numpy as np
from bLUeGui.spline import cubicSpline
from bLUeTop.settings import DNG_PROFILES_DIR2, DNG_PROFILES_DIR1


#########################################################################################
# Functions and classes related to dng/dcp profile tags.
# Compliant with the Adobe DNG specification.
# cf. https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/dng_spec_1.4.0.0.pdf
########################################################################################


def getDngProfileDict(filename):
    """
    Read profile related tags from a dng or dcp file or folder.
    Return a dictionary of (str) decoded {tagname : tagvalue} pairs.
    @param filename: path
    @type filename: str
    @return: dictionary
    @rtype: dict
    """
    with exiftool.ExifTool() as e:
        profileDict = e.readBinaryDataAsDict(filename,
                                             taglist=['LinearizationTable',
                                                      'ProfileLookTableData',
                                                      'ProfileLookTableDims',
                                                      'ProfileLookTableEncoding',
                                                      'ProfileToneCurve',
                                                      'CalibrationIlluminant1',
                                                      'CalibrationIlluminant2',
                                                      'ColorMatrix1',
                                                      'ColorMatrix2',
                                                      'CameraCalibration1',
                                                      'CameraCalibration2',
                                                      'ForwardMatrix1',
                                                      'ForwardMatrix2',
                                                      'AnalogBalance'
                                                      ])
    return profileDict


def getDngProfileList(cameraName):
    """
    Return the list of paths to profiles for a camera model.
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
        try:
            for entry in os.scandir(folder):
                if cameraName in entry.name.lower():
                    if entry.is_file():
                        # camera file : add
                        plist.append(entry.path)
                    elif entry.is_dir():
                        # camera folder : add all files
                        for ent in os.scandir(entry.path):
                            plist.append(ent.path)
        except OSError:
            pass
    return plist


class dngProfileToneCurve:
    """
    Tone curve class.
    Attributes dataX and dataY are the arrays of x-coordinates
    and y-coordinates of the tone curve. They share the same length.
    All coordinates are floats in the interval [0, 1].
    """

    def __init__(self, buf):
        """
        Init the coordinates from a (str) decoded buffer of
        interleaved x and y coordinates. If the tone curve cannot
        be initialized from the buffer, it is set to identity.
        @param buf: decoded buffer
        @type buf: str
        """
        try:
            buf = buf.split(' ')
            buf = np.array([float(x) for x in buf])
            self.dataX, self.dataY = buf[::2], buf[1::2]
        except (ValueError, AttributeError):
            # identity curve
            self.dataX, self.dataY = np.array([0, 1]), np.array([0, 1])

    def toLUTXY(self, maxrange=255):
        """
        interpolate the tone curve by a cubic spline (cf adobe dng specification p. 56).
        @param maxrange: max of data range (identical for input and output)
        @type maxrange: int
        @return: interpolated cubic spline : [0, maxrange] ---> [0, maxrange]
        @rtype: ndarray
        """
        return cubicSpline(self.dataX * maxrange, self.dataY * maxrange, np.arange(maxrange + 1))


class dngProfileLookTable:
    """
    (hue, saturation, value) 3D LUT class.
    Property data holds the table array.
    Property divs holds the number of division points for each axis.
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
        divs, encoding, data = dngDict.get('ProfileLookTableDims', None), dngDict.get('ProfileLookTableEncoding', None), \
                               dngDict.get('ProfileLookTableData', None)
        if divs is None or data is None:  # encoding not used yet : it seems to be missing in dng files
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
            self.__divs = tuple(divs)
            # allocate data array.
            # Adding sentinels, so all
            # dims are increased by +1 (Sentinels allow to
            # use closed intervals instead of right-opened intervals
            # as input ranges).
            # Adding a division point for hue = 360 (cf. dng spec p. 82) : total increment for divs[0] is +2.
            buf = np.zeros((divs[0] + 2, divs[1] + 1, divs[2] + 1, 3), dtype=np.float) + (0, 1, 1)
            # move axes to h, s, v ordering
            data = np.moveaxis(data, (0, 1, 2), (2, 0, 1))
            # put values into table, starting from index 0.
            buf[0:-2, :-1, :-1, :] = data[:, :, :, ]
            # modulo arithmetic for hue
            buf[-2, :, :, 0] = buf[0, :, :, 0]
            # interpolation does not use the values of sentinel sides, so don't care
            self.__data = buf
            self.isValid = True
        except (ValueError, TypeError) as e:
            print('dngProfileLooktable : ', str(e))

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
        """
        (hue, sat, value) 3D look up table.
        Output values are shifts (additive shift for hue, multiplicative
        shifts for saturation and value).
        @return: 3D look up table
        @rtype: ndarray shape=(divs[0] + 2, divs[1] + 1, divs[2] + 1, 3), dtype=float
        """
        return self.__data


class dngProfileIlluminants:
    """
    Wrapper for the two illuminant temperatures
    """
    ExifTemperatureDict = {  # TODO 16/11/18 some conversions from EXIF to temperatures need review
        0: 0,  # Unknown
        1: 5600,  # Daylight
        2: 3600,  # Fluorescent
        3: 3200,  # Tungsten(incandescent light)
        4: 6000,  # Flash
        9: 5600,  # Fine weather
        10: 6500,  # Cloudy weather
        11: 8000,  # Shade
        12: 5700,  # Daylight fluorescent(D 5700 - 7100K)
        13: 4600,  # Day white fluorescent(N 4600 - 5400K)
        14: 3900,  # Cool white fluorescent(W 3900 - 4500K)
        15: 3200,  # White fluorescent(WW3200 - 3700K)
        17: 2856,  # Standard light A
        18: 4874,  # Standard light B
        19: 6774,  # Standard light C
        20: 5500,  # D55
        21: 6500,  # D65
        22: 7500,  # D75
        23: 5000,  # D50
        24: 3200,  # ISO studio tungsten
        255: 6500  # Other light source
    }

    def __init__(self, dngDict):
        try:
            illuminant1, illuminant2 = int(dngDict['CalibrationIlluminant1']), int(dngDict['CalibrationIlluminant2'])
            self.temperature1, self.temperature2 = self.ExifTemperatureDict[illuminant1], self.ExifTemperatureDict[
                illuminant2]
        except (ValueError, KeyError) as e:
            print('dngProfileIlluminants : ', str(e))
            raise e


class dngProfileColorMatrices:
    """
    Wrapper for the two color matrices
    """

    def __init__(self, dngDict):
        try:
            for tag in ['ColorMatrix1', 'ColorMatrix2']:
                M = dngDict.get(tag, None)
                M = np.array([float(x) for x in M.split(' ')]).reshape(3, 3)
                setattr(self, '_' + tag, M)  # a single _ , as setattr does no mangling

        except (ValueError, KeyError) as e:
            print('dngProfileColorMatrices : ', str(e))
            raise e

    @property
    def colorMatrix1(self):
        return self._ColorMatrix1

    @property
    def colorMatrix2(self):
        return self._ColorMatrix2


class dngProfileForwardMatrices:
    """
    Wrapper for the two color matrices
    """

    def __init__(self, dngDict):
        try:
            for tag in ['ForwardMatrix1', 'ForwardMatrix2']:
                M = dngDict.get(tag, None)
                M = np.array([float(x) for x in M.split(' ')]).reshape(3, 3)
                setattr(self, '_' + tag, M)  # a single _ , as setattr does no mangling
        except (ValueError, KeyError) as e:
            print('dngProfileForwardMatrices : ', str(e))
            raise e

    @property
    def forwardMatrix1(self):
        return self._ForwardMatrix1

    @property
    def forwardMatrix2(self):
        return self._ForwardMatrix2


class dngProfileDual:
    """
    Main class for dual illuminant profile.
    An invalid or missing profile dictionary sets the
    property dngProfileDual.isValid to False.
    """

    def __init__(self, dngDict):
        self.__isValid = False
        try:
            illuminants = dngProfileIlluminants(dngDict)
            self.__T1, self.__T2 = illuminants.temperature1, illuminants.temperature2
            matrices = dngProfileColorMatrices(dngDict)
            self.__colorMatrix1, self.__colorMatrix2 = matrices.colorMatrix1, matrices.colorMatrix2
            matrices = dngProfileForwardMatrices(dngDict)
            self.__forwardMatrix1, self.__forwardMatrix2 = matrices.forwardMatrix1, matrices.forwardMatrix2
            self.__isValid = True
        except (ValueError, KeyError, AttributeError) as e:
            print('dngProfileDual : ', str(e))

    @property
    def isValid(self):
        return self.__isValid

    @property
    def colorMatrix1(self):
        return self.__colorMatrix1

    @property
    def colorMatrix2(self):
        return self.__colorMatrix2

    @property
    def forwardMatrix1(self):
        return self.__forwardMatrix1

    @property
    def forwardMatrix2(self):
        return self.__forwardMatrix2

    @property
    def T1(self):
        return self.__T1

    @property
    def T2(self):
        return self.__T2


def interpolate(T, M1, M2, T1, T2):
    """
    Return the interpolated color matrix
    for temperature T, using the two calibration
    illuminants (M1, T1) and (M2, T2).
    Following the Adobe dng spec.(p. 79), we apply
    linear interpolation to the inverse of the temperatures.
    @param T: temperature of interpolation
    @type T: float
    @param M1: ColorMatrix1
    @type M1: ndarray
    @param M2: ColorMatrix2
    @type M2: ndArray
    @param T1: 1st illuminant temperature
    @type T1: float
    @param T2: 2nd illuminant temperature
    @type T2: float
    @return: interpolated matrix
    @rtype: ndarray
    """
    T, T1, T2 = 1 / T, 1 / T1, 1 / T2
    # now T2 < T1
    if T >= T1:
        return M1
    if T <= T2:
        return M2
    return (M1 * (T2 - T) + M2 * (T - T1)) / (T2 - T1)


def interpolatedColorMatrix(T, dngDict):
    """
    Return the interpolated matrix for temperature T, using the
    two illuminants from dngDict.
    Raise a ValueError exception if dngDict is not a valid
    dual illuminant profile.
    @param T: temperature
    @type T: float
    @param dngDict: dng profile tag values dict
    @type dngDict: dict
    @return: interpolated matrix
    @rtype: ndarray, shape=(3,3)
    """
    calibration = dngProfileDual(dngDict)
    if calibration.isValid:
        T1, T2 = calibration.T1, calibration.T2
        colorMatrix1, colorMatrix2 = calibration.colorMatrix1, calibration.colorMatrix2
        return interpolate(T, colorMatrix1, colorMatrix2, T1, T2)
    else:
        raise ValueError("interpolatedColorMatrix : invalid profile")


def interpolatedForwardMatrix(T, dngDict):
    """
    Return the interpolated matrix for temperature T, using the
    two illuminants from dngDict.
    Raise a ValueError exception if dngDict is not a valid
    dual illuminant profile.
    @param T: temperature
    @type T: float
    @param dngDict: dng profile tag values dict
    @type dngDict: dict
    @return: interpolated matrix
    @rtype: ndarray, shape=(3,3)
    """
    calibration = dngProfileDual(dngDict)
    if calibration.isValid:
        T1, T2 = calibration.T1, calibration.T2
        forwardMatrix1, forwardMatrix2 = calibration.forwardMatrix1, calibration.forwardMatrix2
        return interpolate(T, forwardMatrix1, forwardMatrix2, T1, T2)
    else:
        raise ValueError("interpolatedForwardMatrix : invalid profile")
