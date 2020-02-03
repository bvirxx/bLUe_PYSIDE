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
from .cartesian import cartesianProduct
import numpy as np


class HaldArray(object):
    """
    hald image wrapper, recording the size of the corresponding 3D LUT.
    """
    def __init__(self, haldBuffer, size):
        """
        @param haldBuffer: 2D array
        @type haldBuffer: ndarray, shape (w,h,3)
        @param size: size of the 3D LUT
        @type size: int
        """
        self.size = size
        self. haldBuffer = haldBuffer
        super().__init__()


class LUT3D (object):
    """
    Standard RGB 3D LUT, following the Adobe cube LUT specification :
    cf. http://wwwimages.adobe.com/content/dam/Adobe/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf

    This class implements a RGB 3D LUT as a cubic array with shape (s, s, s, 3). The size s should be s = 2**n + 1,
    where n is a postive integer. Most common values are s=17 or s=33.
    The input role (R or G or B) of the LUT axes must follow the ordering
    of the output color channels.

    A 3D LUT can also be represented as a 2D image, called a hald. To build the hald, the LUT is
    flattened, padded with 0,and next reshaped as a two dimensional array.

    Another 3D LUT class, following the Adobe dng spec. and suitable for arbitrary color spaces,
    can be found in the module dng.py.
    """
    ####################################
    # MaxRange defines the maximum input value
    # that can be interpolated from the LUT.
    # It should be 2**n with integer n.
    # For standard 3D LUTs it is always 256
    standardMaxRange = 256
    #####################################

    ####################
    # default LUT size
    defaultSize = 33  # 17
    ####################

    @staticmethod
    def HaldBuffer2LUT3D(haldBuff):
        """
        Convert a HaldArray instance to a LUT3D object.
        The role (R or G or B) of the LUT axes follows the ordering of the color channels.

        @param haldBuff: hald image
        @type haldBuff: HaldArray
        @return: 3D LUT
        @rtype: LUT3D object
        """
        buf = haldBuff.haldBuffer[:, :, :3].ravel()
        size = haldBuff.size
        count = (size ** 3) * 3
        if count > buf.shape[0]:
            raise ValueError('haldBuffer2LUT3D : LUT3D size and hald dimensions do not match')
        buf = buf[:count].reshape((size, size, size, 3))
        LUT = np.zeros((size, size, size, 3), dtype=float)
        LUT[:, :, :, :] = buf
        return LUT3D(LUT, size=size)

    @staticmethod
    def readFromTextStream(inStream):
        """
        Read a 3D LUT from a text stream in format .cube.
        Values read should be between 0 and 1. They are
        multiplied by 255 and converted to int.
        The channels of the LUT and the axes of the cube are both in BGR order.
        Raises a ValueError exception if the method fails.
        @param inStream:
        @type inStream: TextIoWrapper
        @return: 3D LUT
        @rtype: LUT3D object
        """
        ##########
        # read header
        #########
        # We expect exactly 2 uncommented lines
        # where the second is LUT_3D_SIZE xxx
        i = 0
        for line in inStream:
            # skip comments
            if line.startswith('#') or (len(line.lstrip()) == 0):
                continue
            i += 1
            if i < 2:
                continue
            # get LUT size (second line format should be : Size xx)
            token = line.split()
            if len(token) >= 2:
                _, size = token
                break
            else:
                raise ValueError('Cannot find LUT size')
        # LUT size
        size = int(size)
        bufsize = (size ** 3) * 3
        buf = np.zeros(bufsize, dtype=float)
        #######
        # LUT
        ######
        i = 0
        # restarting from current position
        for line in inStream:
            if line.startswith('#') or (len(line.lstrip()) == 0):
                continue
            token = line.split()
            if len(token) >= 3:
                a, b, c = token
            else:
                raise ValueError('Wrong file format')
            # BGR order for channels
            buf[i:i+3] = float(c), float(b), float(a)
            i += 3
        # sanity check
        if i != bufsize:
            raise ValueError('LUT size does not match line count')
        buf *= 255.0
        buf = buf.astype(int)
        buf = buf.reshape(size, size, size, 3)
        # the specification of the .cube format
        # gives BGR order for the cube axes (R-axis changing most rapidly)
        # So, no transposition is needed.
        # buf = buf.transpose(2, 1, 0, 3)
        return LUT3D(buf, size=size)

    @classmethod
    def readFromTextFile(cls, filename):
        """
        Read a 3D LUT from a file in format .cube.
        Values read should be between 0 and 1. They are
        multiplied by 255 and converted to int.
        The channels of the LUT and the axes of the cube are both in order BGR.
        Raise a IOError exception.
        @param filename: path to file
        @type filename: str
        @return: LUT3D
        @rtype: LUT3D class instance
        """
        with open(filename) as textStream:
            lut = cls.readFromTextStream(textStream)
        return lut

    def __init__(self, LUT3DArray, size=defaultSize, maxrange=standardMaxRange, dtype=np.int16, alpha=False):
        """
        Initializes a LUT3D object with shape (size, size, size, d), d = 3 or 4.
        size should be 2**n +1. Most common values are 17 and 33.

        maxrange defines the maximum value which can be interpolated from the LUT.

        LUT3DArray is the array of color values, with shape (size, size, size, 3).
        By convention, the orderings of input and output color channels are identical.

        If LUT3DArray is None, we construct an "identity" LUT3D :
        it holds 3-uples (r,g,b) of numbers of type dtype, evenly
        distributed in the range 0..standardMaxRange (edges included).
        When used to interpolate an image, this "identity" LUT3D
        keeps it unchanged.

        The parameter dtype has no effect when LUT3Darray is not None.

        If alpha is True a fourth channel alpha, initialized to 0, is added to LUT3DArray.
        Interpolated as usual, it is used to build selection masks from sets of 3D LUT vertices.
        @param LUT3DArray: cubic array of LUT3D values
        @type LUT3DArray: ndarray, dtype float or int, shape (size, size, size, 3)
        @param size: size of the axes of the LUT3D
        @type size: int
        @param maxrange: max value that can be interpolated from the LUT
        @type maxrange: int
        @param dtype: type of array data
        @type dtype: numeric type
        @param alpha:
        @type alpha: boolean
        """
        # sanity check
        if ((size - 1) & (size - 2)) != 0:
            raise ValueError("LUT3D : size should be 2**n+1, found %d" % size)

        self.LUT3DArray = LUT3DArray
        self.size = size

        # interpolation step
        self.step = maxrange / (size - 1)
        if not self.step.is_integer():
            raise ValueError('LUT3D : wrong size')

        if LUT3DArray is None:
            # build default (identity) LUT3DArray
            a = np.arange(size, dtype=dtype) * self.step
            self.LUT3DArray = cartesianProduct((a, a, a))
        else:
            s = LUT3DArray.shape
            s0 = (size, size, size, 3)
            if s != s0:
                raise ValueError("LUT3D : array shape should be (%d,%d,%d,%d)" % s0)
        if alpha:
            self.LUT3DArray = np.concatenate((
                                     self.LUT3DArray,
                                     np.zeros(self.LUT3DArray.shape[:3] + (1,), dtype=self.LUT3DArray.dtype)),
                                     axis=-1)
        super().__init__()

    def toHaldArray(self, w, h):
        """
        Convert a LUT3D object to a haldArray object with shape (w,h,3).
        The 3D LUT is clipped to 0..255, flattened, padded with 0, and reshaped
        to a 2D array. The product w * h must be greater than (self.size)**3
        Hald channels, LUT channels and LUT axes must follow the same ordering (BGR or RGB).
        To simplify, we only handle halds and LUTs of type BGR.
        @param w: image width
        @type w: int
        @param h: image height
        @type h: int
        @return: hald image
        @rtype: HaldArray
        """
        s = self.size
        if (s ** 3) > w * h:
            raise ValueError("toHaldArray : incorrect sizes)")
        buf = np.zeros((w * h * 3), dtype=np.uint8)
        count = (s ** 3) * 3
        buf[:count] = np.clip(self.LUT3DArray.ravel(), 0, 255)
        buf = buf.reshape(h, w, 3)
        return HaldArray(buf, s)

    def writeToTextStream(self, outStream):
        """
        Writes a 3D LUT to a text stream in format .cube.
        Values are divided by 255.
        The 3D LUT must be in BGR or BGRA order.
        @param outStream:
        @type outStream: TextIoWrapper
        """
        LUT = self.LUT3DArray
        outStream.write('bLUe 3D LUT\n')
        outStream.write('Size %d\n' % self.size)
        coeff = 255.0
        for b in range(self.size):
            for g in range(self.size):
                for r in range(self.size):
                    # r1, g1, b1 = LUT[r, g, b]  # order RGB
                    b1, g1, r1 = LUT[b, g, r][:3]  # order BGR; BGRA values are allowed, so [:3] is mandatory
                    outStream.write("%.7f %.7f %.7f\n" % (r1 / coeff, g1 / coeff, b1 / coeff))

    def writeToTextFile(self, filename):
        """
        Writes 3D LUT to QTextStream in format .cube.
        Values are divided by 255.
        The 3D LUT must be in BGR order.
        @param filename:
        @type filename: str
        Raise IOError
        """
        with open(filename, 'w') as textStream:
            self.writeToTextStream(textStream)


class DeltaLUT3D(object):
    """
    Versatile displacement 3D LUT. First dim is meant for hue
    (additive shift and modulo arithmetic)
    and remaining dims can be used for any type of input (multiplicative shifts).
    """
    def __init__(self, divs):
        """
        Init an identity displacement 3D LUT with
        shape (divs[0] + 2, divs[1] + 1, divs[2] + 1, 3)
        @param divs: division count for each axis
        @type divs: 3-uple of int

        """
        self.__divs = divs
        self.__data = np.zeros((divs[0] + 2, divs[1] + 1, divs[2] + 1, 3), dtype=np.float) + (0, 1, 1)

    @property
    def divs(self):
        """
        Count of dividing intervals for each axis.

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
