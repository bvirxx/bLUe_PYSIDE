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
from PySide6.QtGui import QImage

from bLUeGui.colorCube import rgb2hlsVec, hls2rgbVec
from bLUeGui.bLUeImage import QImageBuffer


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
    :return:the blended buffer
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
    :type  dest QImage
    :param source: source QImage
    :type  source QImage
    :return:The blended image
    :rtype: QImage same size and format as source

    """
    sourceBuf = QImageBuffer(source)[:, :, :3]
    destBuf = QImageBuffer(dest)[:, :, :3]
    blendBuf = blendLuminosityBuf(destBuf[..., ::-1], sourceBuf[..., ::-1])
    img = QImage(source.size(), source.format())
    tmp = QImageBuffer(img)
    tmp[:, :, :3][:, :, ::-1] = blendBuf
    tmp[:, :, 3] = 255
    return img


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
    :return:the blended buffer
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
    :return:The blended image
    :rtype: QImage QImage same size and format as source
    """
    return blendLuminosity(source, dest)
