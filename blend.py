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

from LUT3D import hsv2rgbVec, rgb2hsBVec, hsp2rgbVec
from MarkedImg import imImage
from imgconvert import QImageBuffer


def blendLuminosity(dest, source):
    """
    Implements blending with luminosity mode, which is missing
    in Qt. We use the HSpB color model for the intermediate color space.
    The blended image retains the hue and saturation of dest, with the
    luminosity (i.e. perceptive brightness) of source.
   
    @param dest: destination QImage
    @type dest QImage
    @param source: source QImage
    @type source QImage
    @return: The blended image
    @rtype: QImage
    
    """
    sourceBuf = QImageBuffer(source)[:,:,:3]
    destBuf = QImageBuffer(dest)[:,:,:3]

    hsvSourceBuf = rgb2hsBVec(sourceBuf[:,:,::-1], perceptual=True)
    hsvDestBuf = rgb2hsBVec(destBuf[:,:,::-1], perceptual=True)
    hsvDestBuf[:, :, 2] = hsvSourceBuf[:,:,2]
    blendBuf = hsp2rgbVec(hsvDestBuf)

    img = QImage(source.size(), source.format())
    tmp=QImageBuffer(img)
    (tmp[:,:,:3])[:,:,::-1] = blendBuf
    tmp[:, :, 3]= 255

    return img

def blendColor(dest, source):
    """
    Implements blending using color mode, which is missing
    in Qt. We use the HSpB color model as intermediate color space.
    The blended image retains the hue and saturation of source, with the
    luminosity (i.e. perceptive brightness) of dest.
    @param dest: destination QImage
    @type dest: QImage
    @param source: source QImage
    @type source: QImage
    @return: The blended image
    @rtype: QImage
    """
    sourceBuf = QImageBuffer(source)[:, :, :3]
    destBuf = QImageBuffer(dest)[:, :, :3]

    hsvSourceBuf = rgb2hsBVec(sourceBuf[:, :, ::-1], perceptual=True)
    hsvDestBuf = rgb2hsBVec(destBuf[:, :, ::-1], perceptual=True)
    hsvDestBuf[:, :, :2] = hsvSourceBuf[:, :, :2]
    blendBuf = hsp2rgbVec(hsvDestBuf)

    img = QImage(source.size(), source.format())
    tmp = QImageBuffer(img)
    (tmp[:, :, :3])[:, :, ::-1] = blendBuf
    tmp[:, :, 3] = 255

    return img




