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

import numpy as np
from PySide2.QtGui import QImage, QPixmap

class bImage(QImage):
    """
    Base class for all bLUe images.
    Inherits from QImage and adds a bunch
    of caches encapsulated as properties.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rPixmap = None
        self.hspbBuffer = None
        self.LabBuffer = None
        self.HSVBuffer = None

    @property
    def rPixmap(self):
        return self.__rPixmap

    @rPixmap.setter
    def rPixmap(self, pixmap):
        self.__rPixmap = pixmap

    @property
    def hspbBuffer(self):
        return self.__hspbBuffer

    @hspbBuffer.setter
    def hspbBuffer(self, buffer):
        self.__hspbBuffer = buffer

    @property
    def LabBuffer(self):
        return self.__LabBUffer

    @LabBuffer.setter
    def LabBuffer(self, buffer):
        self.__LabBuffer = buffer

    @property
    def HSVBuffer(self):
        return self.__HSVBUffer

    @HSVBuffer.setter
    def HSVBuffer(self, buffer):
        self.__HSVBuffer = buffer

    def updatePixmap(self, maskOnly=False):
        """
        To respect the Substitutability Principle of Liskov
        for subtypes, we should keep identical signatures for all
        overriding methods, so we define here an unused parameter
        maskOnly.
        @param maskOnly: not used
        @type maskOnly: boolean
        """
        self.rPixmap = QPixmap.fromImage(self)

QImageFormats = {0:'invalid', 1:'mono', 2:'monoLSB', 3:'indexed8', 4:'RGB32', 5:'ARGB32',6:'ARGB32 Premultiplied',
                 7:'RGB16', 8:'ARGB8565 Premultiplied', 9:'RGB666',10:'ARGB6666 Premultiplied', 11:'RGB555', 12:'ARGB8555 Premultiplied',
                 13: 'RGB888', 14:'RGB444', 15:'ARGB4444 Premultiplied'}

def ndarrayToQImage(ndimg, format=QImage.Format_ARGB32):
    """
    Converts a 3D numpy ndarray to a QImage. No sanity check is
    done regarding the compatibility of the ndarray shape with
    the QImage format.
    @param ndimg: The ndarray to be converted
    @type ndimg: ndarray
    @param format: The QImage format (default ARGB32)
    @type format:
    @return: The converted image
    @rtype: QImage
    """
    if ndimg.ndim != 3 or ndimg.dtype != 'uint8':
        raise ValueError("ndarray2QImage : array must be 3D with dtype=uint8, found ndim=%d, dtype=%s" %(ndimg.ndim, ndimg.dtype))
    bytePerLine = ndimg.shape[1] * ndimg.shape[2]
    if len(np.ravel(ndimg).data)!=ndimg.shape[0]*bytePerLine :  # TODO added ravel 5/11/17 needed by vImage.resize
        raise ValueError("ndarrayToQImage : conversion error")
    # build QImage from buffer
    qimg = QImage(ndimg.data, ndimg.shape[1], ndimg.shape[0], bytePerLine, format)
    if qimg.format() == QImage.Format_Invalid:
        raise ValueError("ndarrayToQImage : wrong conversion")
    return qimg

def QImageBuffer(qimg):
    """
    Returns the buffer of a QImage as a numpy ndarray, dtype=uint8. The size of the
    3rd axis (raw pixels) depends on the image type. Pixels are in
    BGRA order (little endian arch. (intel)) or ARGB (big  endian arch.)
    Format 1 bit per pixel is not supported.
    Performance : 20 ms for a 15 Mpx image.
    @param qimg:
    @type qimg: QImage
    @return: The buffer array
    @rtype: numpy ndarray, shape = (h,w, bytes_per_pixel), dtype=uint8
    """
    # pixel depth
    bpp = qimg.depth()
    if bpp == 1:
        raise ValueError("QImageBuffer : unsupported image format 1 bit per pixel")
    # Bytes per pixel
    Bpp = bpp // 8
    # Get image buffer
    # Calling bits() performs a deep copy of the buffer,
    # suppressing all dependencies due to implicit data sharing.
    # To avoid deep copy use constBits() instead (Caution : it returns a read-only buffer).
    ptr = qimg.bits()  # type memoryview, items are bytes : ptr.itemsize = 1
    #convert buffer to ndarray and reshape
    h,w = qimg.height(), qimg.width()
    return np.asarray(ptr, dtype=np.uint8).reshape(h, w, Bpp)  # specifying dtype is mandatory to prevent copy of data
