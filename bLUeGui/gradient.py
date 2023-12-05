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
import cv2
from PySide6.QtGui import QColor, QImage
from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.dialog import dlgWarn
from bLUeGui.colorCIE import rgb2rgbLinear, rgbLinear2rgb

"""
the code in this file comes in part from 
https://stackoverflow.com/questions/22607043/color-gradient-algorithm
"""
def all_channels2(func):
    def wrapper(channel1, channel2, *args, **kwargs):
        try:
            return func(channel1, channel2, *args, **kwargs)
        except TypeError:
            return tuple(func(c1, c2, *args, **kwargs) for c1,c2 in zip(channel1, channel2))
    return wrapper


@all_channels2
def lerp(color1, color2, frac):
    return color1 * (1 - frac) + color2 * frac


def getGradient(color1, color2, steps):
    """
    Builds a generator object for steps interpolated colors between color1 and color2 (linear gradient).
    Interpolated colors are RGB 3-uples of integers in range 0..255
    :param color1: RGB color
    :type color1: QColor
    :param color2: RGB color
    :type color2: QColor
    :param steps: interpolation steps
    :type steps: int
    :return: Steps interpolated RGB colors
    :rtype: Generator object
    """
    gamma = 0.43
    r, g, b, _ = color1.getRgb()
    color1_lin = rgb2rgbLinear((r, g, b))
    bright1 = sum(color1_lin)**gamma
    r, g, b, _ = color2.getRgb()
    color2_lin = rgb2rgbLinear((r, g, b))
    bright2 = sum(color2_lin)**gamma
    for step in range(steps):
        intensity = lerp(bright1, bright2, step / steps) ** (1/gamma)
        color = lerp(color1_lin, color2_lin, step / steps)
        if sum(color) != 0:
            color = [c * intensity / sum(color) for c in color]
        color = rgbLinear2rgb(color)
        yield color

def gradient2Img(grad, height=50):
    """
    Builds an image and fills it with gradient.
    :param grad: iterable of RGB 3-uples, range 0..255
    :type grad: iterable
    :param height: image height
    :type height:
    :return: Gradient image
    :rtype: QImage
    """
    s = grad.shape[0]
    img = QImage(s, height,  QImage.Format_RGBA8888)
    imgBuffer = QImageBuffer(img)
    imgBuffer[..., :3][::-1] = grad
    imgBuffer[..., 3] = 255
    return img

def gradientArray(colorList, stepList):
    """
    Concatenates linear gradients between consecutive items in colorList into an array of RGB 3-uples.

    :param colorList: colors
    :type colorList: list of QColors
    :param stepList: Interpolation steps between consecutive colors in colorList
    :type stepList: list of int
    :return: gradient
    :rtype: array of 3-uples of RGB colors, range 0..255
    """
    try:
        if len(colorList) < 2 or len(stepList) != len(colorList) - 1:
            raise ValueError
        size = sum(stepList)
        grad = np.empty((size,), dtype=np.dtype((float, 3)))
        grad_current = 0
        for i in range(len(colorList) - 1):
            buf = np.fromiter(getGradient(colorList[i],
                                          colorList[i + 1],
                                          stepList[i],
                                          ),
                              dtype=np.dtype((float, 3))
                              )
            grad[grad_current: grad_current + stepList[i]] = buf
            if i == 0:
                grad[0] = buf[0]
            grad_current += stepList[i]
        return grad
    except ValueError:
        dlgWarn('hsvGradientArray : invalid gradient')

def hsvGradientArray(grad):
    """
    Converts gradient from RGB to HSV.
    :param grad: gradient array of RGB colors, range 0..255
    :type grad: array of 3-uples of int or float
    :return: gradient of HSV colors, range 0..360, 0..1, 0..1
    :rtype:
    """

    grad = grad[np.newaxis,...]
    bufhsv = cv2.cvtColor((grad / 255).astype(np.float32), cv2.COLOR_RGB2HSV)

    return bufhsv[0]

def setLUTfromGradient(lut, grad, ori2hsv, brCoeffs):
    """
    Builds a LUT3D instance for Color Grading : Image pixel hues  are replaced by corresponding gradient hues,
    saturations and brightnesses are kept.
    :param lut:
    :type lut:  LUT3D instance
    :param grad: HSV gradient array
    :type grad: array of HSV 3-uples
    :param ori2hsv: Identity 3D LUT array preconverted to HSV values
    :type ori2hsv: same as lut.LUT3DArray
    :param brCoeffs brightness corrections, range 0..1
    :type brCoeffs : array of size 255, dtype float
    """

    lut = lut.LUT3DArray
    steps = grad.shape[0] - 1
    """
    Pure Python implementation. THe numpy implementation below is 100 times faster.
    for i in range(0, lut.shape[0]):
        for j in range(0, lut.shape[1]):
            for k in range(0, lut.shape[2]):
                ind = min(int(ori2hsv[i, j, k, 2] * steps), 254)
                value = grad[ind]
                coeff = brCoeffs[ind]
                if value[1] <= 0:  # keep grey tones
                    lut[i, j, k] = hsv2rgb(value[0], value[1], ori2hsv[i, j, k, 2]*coeff)[::-1]
                else:
                    lut[i, j, k] = hsv2rgb(value[0], ori2hsv[i, j, k, 1], ori2hsv[i, j, k, 2]*coeff)[::-1]
    """
    # use current brightness as index
    ind = np.minimum(ori2hsv[..., 2] * steps, 254).astype(int)
    # get hue from gradient
    lut[..., 0] = grad[ind][..., 0]
    # get saturation from gradient, but keep grey tones
    lut[..., 1] = np.where(grad[ind][..., 1] <= 0, grad[ind][..., 1], ori2hsv[..., 1])
    # keep (corrected) brightness
    lut[..., 2] = ori2hsv[..., 2] * brCoeffs[ind]
    aux = lut.reshape(33*33, 33, 3).astype(np.float32)
    aux = cv2.cvtColor(aux, cv2.COLOR_HSV2BGR).reshape(33, 33, 33, 3)
    lut[...] = aux * 255






