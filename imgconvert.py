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
from time import time

import numpy as np

from bLUeGui.bLUeImage import QImageBuffer
from compat import PilImgToRaw
from PySide2.QtGui import QImage
from PIL import Image

from debug import tdec

def PilImageToQImage(pilimg) :
    """
    Converts a PIL image (mode RGB) to a QImage (format RGB32)
    @param pilimg: The PIL image, mode RGB
    @type pilimg: PIL image
    @return: the converted image
    @rtype: QImage
    """
    ############################################
    # CAUTION: PIL ImageQt causes a memory leak!!!
    # return ImageQt(pilimg)
    ############################################
    im_data = PilImgToRaw(pilimg)
    Qimg = QImage(im_data['im'].size[0], im_data['im'].size[1], im_data['format'])
    buf = QImageBuffer(Qimg).ravel()
    buf[:] = np.frombuffer(im_data['data'], dtype=np.uint8)
    return Qimg

def QImageToPilImage(qimg) :
    """
    Converts a QImage (format ARGB32or RGB32) to a PIL image
    @param qimg: The Qimage to convert
    @type qimg: Qimage
    @return: PIL image  object, mode RGB
    @rtype: PIL Image
    """
    a = QImageBuffer(qimg)
    if (qimg.format() == QImage.Format_ARGB32) or (qimg.format() == QImage.Format_RGB32):
        # convert pixels from BGRA or BGRX to RGB
        a = np.ascontiguousarray(a[:,:,:3][:,:,::-1]) #ascontiguousarray is mandatory to speed up Image.fromArray (x3)
    else :
        raise ValueError("QImageToPilImage : unrecognized format : %s" %qimg.Format())
    return Image.fromarray(a)