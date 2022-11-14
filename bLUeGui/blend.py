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

from collections import OrderedDict

from PySide2.QtGui import QImage, QPainter

from bLUeGui.colorCube import rgb2hlsVec, hls2rgbVec
from bLUeGui.bLUeImage import QImageBuffer

# Type of QPainter.XXXX modes is QPainter.CompositionMode (Shiboken enum-type).
# Additional modes are not implemented by QPainter.

# TODO :  blending functions should handle pixel opacity

compositionModeDict = OrderedDict([('Normal', QPainter.CompositionMode_SourceOver),
                                   ('Plus', QPainter.CompositionMode_Plus),
                                   ('Multiply', QPainter.CompositionMode_Multiply),
                                   ('Screen', QPainter.CompositionMode_Screen),
                                   ('Overlay', QPainter.CompositionMode_Overlay),
                                   ('Darken', QPainter.CompositionMode_Darken),
                                   ('Lighten', QPainter.CompositionMode_Lighten),
                                   ('Color Dodge', QPainter.CompositionMode_ColorDodge),
                                   ('Linear Dodge', -3),
                                   ('Color Burn', QPainter.CompositionMode_ColorBurn),
                                   ('Linear Burn', -4),
                                   ('Hard Light', QPainter.CompositionMode_HardLight),
                                   ('Soft Light', QPainter.CompositionMode_SoftLight),
                                   ('Linear Light', -5),
                                   ('Difference', QPainter.CompositionMode_Difference),
                                   ('Exclusion', QPainter.CompositionMode_Exclusion),
                                   ('Luminosity', -1),
                                   ('Color', -2)
                                   ])

compositionModeDict_names = {v: k.replace(' ', '') for k, v in compositionModeDict.items()}


def _applyBlendBufFunc(dest, source, func):
    sourceBuf = QImageBuffer(source)[:, :, :3]
    destBuf = QImageBuffer(dest)[:, :, :3]
    blendBuf = func(destBuf[..., ::-1], sourceBuf[..., ::-1])
    img = QImage(source.size(), source.format())
    tmp = QImageBuffer(img)
    tmp[:, :, :3][:, :, ::-1] = blendBuf
    tmp[:, :, 3] = 255
    return tmp


def blendLuminosityBuf(destBuf, sourceBuf):
    """
    Important : Buffer channels should be in r,g,b order.
    The method implements blending in luminosity mode,
    which is missing in Qt.
    The blended image retains the hue and saturation of dest,
    with the luminosity of source.
    We use the HLS color model:
    see https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Note blendColor and blendLuminosity are commuted versions of each other:
    blendLuminosity(img1, img2) = blendColor(img2, img1)

    :param destBuf: destination r,g,b image buffer
    :type  destBuf: ndarray
    :param sourceBuf: source r,g,b image buffer
    :type  sourceBuf: ndarray
    :return: the blended buffer
    :rtype: ndarray
    """
    hlsSourceBuf = rgb2hlsVec(sourceBuf)
    hlsDestBuf = rgb2hlsVec(destBuf)
    # copy source luminosity to dest
    hlsDestBuf[:, :, 1] = hlsSourceBuf[:, :, 1]
    blendBuf = hls2rgbVec(hlsDestBuf)
    return blendBuf


def blendLuminosity(dest, source):
    """
    Implements blending in luminosity mode,
    which is missing in Qt.
    The blended image retains the hue and saturation of dest,
    with the luminosity of source.
    We use the HLS color model:
    see https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Note blendColor and blendLuminosity are commuted versions of each other:
    blendLuminosity(img1, img2) = blendColor(img2, img1)

    :param dest: destination QImage
    :type  dest: QImage
    :param source: source QImage
    :type  source: QImage
    :return: The blended image
    :rtype: QImage same size and format as source

    """
    return _applyBlendBufFunc(dest, source, blendLuminosityBuf)


def blendColorBuf(destBuf, sourceBuf):
    """
    Important : Buffer channels should be in r,g,b order.
    The method implements blending in color mode, which is missing
    in Qt. We use the HLS color model as intermediate color space.
    The blended image retains the hue and saturation of source, with the
    luminosity of dest. We use the HLS color model:
    see https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Note blendColor and blendLuminosity are commuted versions of each other:
    blendLuminosity(img1, img2) = blendColor(img2, img1)

    :param destBuf: destination r,g,b image buffer
    :type  destBuf: ndarray
    :param sourceBuf: source r,g,b image buffer
    :type  sourceBuf: ndarray
    :return: the blended buffer
    :rtype: ndarray
    """
    return blendLuminosityBuf(sourceBuf, destBuf)


def blendColor(dest, source):
    """
    Implements blending in color mode, which is missing
    in Qt. We use the HLS color model as intermediate color space.
    The blended image retains the hue and saturation of source, with the
    luminosity of dest. We use the HLS color model:
    see https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Note blendColor and blendLuminosity are commuted versions of each other:
    blendLuminosity(img1, img2) = blendColor(img2, img1)

    :param dest: destination QImage
    :type  dest: QImage
    :param source: source QImage
    :type  source: QImage
    :return: The blended image
    :rtype: QImage QImage same size and format as source
    """
    return blendLuminosity(source, dest)


def blendLinearDodgeBuf(destBuf, sourceBuf):
    """
    Linear Dodge blending mode:
    result = source + dest

    :param destBuf: destination r,g,b image buffer
    :type destBuf: ndarray
    :param sourceBuf: source r,g,b image buffer
    :type sourceBuf: ndarray
    :return: the blended buffer
    :rtype: ndarray
    """
    tmp = np.add(sourceBuf, destBuf, dtype=np.int16)
    np.clip(tmp, None, 255, out=tmp)
    return tmp.astype(np.uint8)


def blendLinearDodge(dest, source):
    """

    :param dest: destination QImage
    :type  dest: QImage
    :param source: source QImage
    :type  source: QImage
    :return: The blended image
    :rtype: QImage same size and format as source
    """
    return _applyBlendBufFunc(dest, source, blendLinearDodgeBuf)


def blendLinearBurnBuf(destBuf, sourceBuf):
    """
    Linear Burn blending mode:
    result = source + dest - 255

    :param destBuf: destination r,g,b image buffer
    :type destBuf: ndarray
    :param sourceBuf: source r,g,b image buffer
    :type sourceBuf: ndarray
    :return: the blended buffer
    :rtype: ndarray
    """
    tmp = np.add(sourceBuf, destBuf, dtype=np.int16)
    tmp -= 255
    np.clip(tmp, 0, None, out=tmp)
    return tmp.astype(np.uint8)


def blendLinearBurn(dest, source):
    """

    :param dest: destination QImage
    :type  dest: QImage
    :param source: source QImage
    :type  source: QImage
    :return: The blended image
    :rtype: QImage same size and format as source
    """
    return _applyBlendBufFunc(dest, source, blendLinearBurnBuf)


def blendLinearLightBuf(destBuf, sourceBuf):
    """
    Linear Light blending mode:
    result = linearBurn(source, 2 * dest) if dest < 128 else LinearDodge(source, 2 * (dest - 128))

    :param destBuf: destination r,g,b image buffer
    :type destBuf: ndarray
    :param sourceBuf: source r,g,b image buffer
    :type sourceBuf: ndarray
    :return: the blended buffer
    :rtype: ndarray
    """
    destBuf2 = np.multiply(2, destBuf, dtype=np.int16)
    tmp1 = blendLinearBurnBuf(destBuf2, sourceBuf)
    tmp2 = blendLinearDodgeBuf(destBuf2 - 256, sourceBuf)
    tmp = np.where(destBuf < 128, tmp1, tmp2)
    return tmp


def blendLinearLight(dest, source):
    """

    :param dest: destination QImage
    :type  dest: QImage
    :param source: source QImage
    :type  source: QImage
    :return: The blended image
    :rtype: QImage same size and format as source
    """
    return _applyBlendBufFunc(dest, source, blendLinearLightBuf)
