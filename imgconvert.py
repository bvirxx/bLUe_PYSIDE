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
from PySide.QtGui import QImage
from PIL import Image

QImageFormats = {0:'invalid', 1:'mono', 2:'monoLSB', 3:'indexed8', 4:'RGB32', 5:'ARGB32',6:'ARGB32 Premultiplied',
                 7:'RGB16', 8:'ARGB8565 Premultiplied', 9:'RGB666',10:'ARGB6666 Premultiplied', 11:'RGB555', 12:'ARGB8555 Premultiplied',
                 13: 'RGB888', 14:'RGB444', 15:'ARGB4444 Premultiplied'}

def ndarrayToQImage(ndimg, format=QImage.Format_ARGB32):
    """
    Convert a 3D numpy ndarray to a QImage. No sanity check is
    done concerning the compatibility between the ndarray shape and
    the QImage format. Although the doc is unclear, it seems that the
    buffer is copied when needed.
    @param ndimg: The ndarray to be converted
    @param format: The QImage format (default ARGB32)
    @return: The converted image
    """
    if ndimg.ndim != 3 or ndimg.dtype != 'uint8':
        raise ValueError("ndarray2QImage : array must be 3D with dtype=uint8, found ndim=%d, dtype=%s" %(ndimg.ndim, ndimg.dtype))

    bytePerLine = ndimg.shape[1] * ndimg.shape[2]
    if len(ndimg.data)!=ndimg.shape[0]*bytePerLine :
        raise ValueError("ndarrayToQImage : conversion error")
    # build QImage from buffer
    qimg = QImage(ndimg.data, ndimg.shape[1], ndimg.shape[0], bytePerLine, format)
    if qimg.format() == QImage.Format_Invalid:
        raise ValueError("ndarrayToQImage : wrong conversion")
    return qimg

def QImageBuffer(qimg):
    """
    Get the QImage buffer as a numpy ndarray with dtype uint8. The size of the
    3rd axis depends on the image type. Pixel color is
    in BGRA order (little endian arch. (intel)) or ARGB (big  endian arch.)
    Format 1 bit per pixel is not supported
    @rtype: numpy.ndarray
    @param qimg: QImage
    @return: The buffer array
    """
    # pixel depth
    bpp = qimg.depth()
    if bpp == 1:
        print "Qimage2array : unsupported image format 1 bit per pixel"
        return None
    Bpp = bpp / 8

    # image buffer (sip.array of Bytes)
    # Calling bits() performs a deep copy of the buffer,
    # suppressing dependencies due to implicit data sharing.
    # To avoid deep copy use constBits() instead.
    ptr = qimg.bits()
    #ptr.setsize(qimg.byteCount())

    #convert sip array to ndarray and reshape
    h,w = qimg.height(), qimg.width()
    return np.asarray(ptr).reshape(h, w, Bpp)

def PilImageToQImage(pilimg) :
    """
    Convert a PIL image to a QImage
    @param pilimg: The PIL image, mode RGB
    @type pilimg: PIL image
    @return: QImage object, format QImage.Format_ARGB32
    @rtype: PySide.QtGui.QImage
    """
    w, h = pilimg.width, pilimg.height
    mode = pilimg.mode

    if mode != 'RGB':
        raise ValueError("PilImageToQImage : wrong mode : %s" % mode)

    # get data buffer (type str)
    data = pilimg.tobytes('raw', mode)

    if len(data) != w * h * 3:
        raise ValueError("PilImageToQImage : incorrect buffer length : %d, should be %d" % (len(data), w * h * 3))

    BytesPerLine = w * 3
    qimFormat = QImage.Format_RGB888
    img888 = QImage(data, w, h, BytesPerLine, qimFormat)
    return img888.convertToFormat(QImage.Format_ARGB32)


def QImageToPilImage(qimg) :
    """
    Convert a QImage to a PIL image
    @param qimg: The Qimage
    @return: PIL image  object, mode RGB
    """
    a = QImageBuffer(qimg)

    if (qimg.format() == QImage.Format_ARGB32) or (qimg.format() == QImage.Format_RGB32):
        # convert pixels from BGRA or BGRX to RGB
        a=a[:,:,:3][:,:,::-1]
        a = np.ascontiguousarray(a)
    else :
        raise ValueError("QImageToPilImage : unrecognized format : %s" %qimg.Format())

    w, h = qimg.width(), qimg.height()

    return Image.frombytes('RGB', (w,h), a.data)