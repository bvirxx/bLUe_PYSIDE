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

class haldArray(object):
    """
    hald array wrapper, recording the size of the corresponding LUT.
    """
    def __init__(self, haldBuffer, size):
        """
        Inits a hald array.
        @param haldBuffer: 2D array
        @type haldBuffer: ndarray, shape (w,h,3)
        @param size: size of the LUT
        @type size: int
        """
        self.size = size
        self. haldBuffer = haldBuffer
        super().__init__()

class LUT3D (object):
    """
    Implements a 3D LUT as a cubic array with shape (s, s, s, 3). The size s should be s=2**n + 1,
    where n is a postive integer. Most common values are s=17 or s=33.

    The role (R or G or B) of the LUT axes follows the ordering of the color channels.

    A 3D LUT can also be represented as an array or image, called a hald. To build the hald, the LUT is
    flattened, padded with 0,and next reshaped as a a two dimensional array. The 3D LUT can easily be
    reconstructed from the hald.
    """
    ####################################
    # MaxRange defines the maximum input value
    # that can be interpolated from the LUT.
    # It should be 2**n with integer n.
    # For standard 3D LUT formats it is always 256
    standardMaxRange = 256
    #####################################

    ####################
    # default LUT size
    defaultSize = 33
    ####################

    @classmethod
    def HaldBuffer2LUT3D(cls, haldBuff):
        """
        Converts a hald array to a LUT3D object.

        A hald image or hald array can be viewed as a 3D LUT flattened and reshaped
        as a 2D array. The (self.size-1)**3 first pixels
        of the flattened image are taken from the LUT; remainings bytes are padded with 0.

        The role (R or G or B) of the LUT axes is given by the color channels ordering
        in haldBuf.

        @param haldBuff: hald array
        @type haldBuff: haldArray
        @return: 3D LUT
        @rtype: LUT3D object
        """
        buf = haldBuff.haldBuffer[:, :, :3].ravel()
        size = haldBuff.size
        count = (size ** 3) * 3
        if count > buf.shape[0]:
            raise ValueError('HaldImage2LUT3D : LUT3D size and hald dimensions do not match')
        buf = buf[:count].reshape((size, size, size, 3))
        LUT = np.zeros((size, size, size, 3), dtype=float)  # TODO 18/10/18 changed dtype int to float : validate
        LUT[:, :, :, :] = buf # [:, :, :, ::-1]
        return LUT3D(LUT, size=size)

    @classmethod
    def readFromTextStream(cls, inStream):
        """
        Read a 3D LUT from a text stream in format .cube.
        Values read should be between 0 and 1. They are
        multiplied by 255 and converted to int.
        The channels of the LUT and the axes of the cube are both in order BGR.
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
            i+=3
        # sanity check
        if i != bufsize:
           raise ValueError('LUT size does not match line count')
        buf *= 255.0
        buf = buf.astype(int)
        buf = buf.reshape(size,size,size,3)
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

    def __init__(self, LUT3DArray, size=defaultSize, maxrange=standardMaxRange, dtype=np.int16):
        """
        Initializes a LUT3D object with shape (size, size,size, 3).
        size should be 2**n +1. Most common values are 17 and 33.

        maxrange defines the maximum value which can be interpolated from the LUT.

        LUT3DArray is the array of color values, with shape (size, size, size, 3).
        By convention, the role (R or G or B) of the three first axes follows the ordering
        of the color channels..

        If LUT3DArray is None, we construct an "identity" LUT3D :
        it holds 3-uples (r,g,b) of numbers of type dtype, evenly
        distributed in the range 0..standardMaxRange (edges included).
        When used to interpolate an image, this "identity" LUT3D
        should keep it unchanged.

        The parameter dtype has no effect when LUT3Darray is not None.

        Note that an "identity" LUT is invariant when the color
        channel ordering changes.
        @param LUT3DArray: cubic array of LUT3D values
        @type LUT3DArray: ndarray, dtype float or int, shape (size, size, size, 3)
        @param size: size of the axes of the LUT3D
        @type size: int
        @param maxrange: max value that can be interpolated from the LUT
        @type maxrange: int
        @param dtype: type of array data
        @type dtype: numeric type
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
            # build default LUT3DArray
            a = np.arange(size, dtype=dtype) * self.step  #
            c = cartesianProduct((a, a, a))
            self.LUT3DArray = c
        else:
            s = LUT3DArray.shape
            s0 = (size, size, size, 3)
            if s != s0:
                raise ValueError("LUT3D : array shape should be (%d,%d,%d,%d)" % s0)
        super().__init__()

    def toHaldArray(self, w, h):
        """
        Convert the LUT3D object to a hald array with shape (w,h,3).

        The 3D LUT is flattened, padded with 0, and reshaped
        to a 2D array (or image).

        w*h should be greater than (self.size)**3

        IMPORTANT : Hald channels, LUT channels and LUT axes must follow the same ordering (BGR or RGB).
        To simplify, we only handle halds and LUTs of type BGR.
        @param w: image width
        @type w: int
        @param h: image height
        @type h: int
        @return: hald array
        @rtype: haldArray object
        """
        s = self.size
        if (s ** 3) > w * h:
            raise ValueError("toHaldArray : incorrect sizes)")
        buf = np.zeros((w * h * 3), dtype=np.uint8)
        count = (s ** 3) * 3  # TODO may be clip LUT array to 0,255 ?
        buf[:count] = self.LUT3DArray.ravel()
        buf = buf.reshape(h, w, 3)
        return haldArray(buf, s)

    def writeToTextStream(self, outStream):
        """
        Writes a 3D LUT to a text stream in format .cube.
        Values are divided by 255.
        The 3D LUT must be in BGR order.
        @param outStream:
        @type outStream: TextIoWrapper
        """
        LUT=self.LUT3DArray
        outStream.write('bLUe 3D LUT\n')
        outStream.write('Size %d\n' % self.size)
        coeff = 255.0
        for b in range(self.size):
            for g in range(self.size):
                for r in range(self.size):
                    #r1, g1, b1 = LUT[r, g, b]  # order RGB
                    b1, g1, r1 = LUT[b, g, r]  # order BGR
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
