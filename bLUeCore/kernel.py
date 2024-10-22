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
from math import erf


class filterIndex():
    IDENTITY, UNSHARP, SHARPEN, BLUR1, BLUR2, SURFACEBLUR, GRADUALFILTER = range(7)


def phi(x, mu, sigma):
    """
    Cumulative distribution function (CDF) of the
    normal distribution N(mu, sigma).

    :param x: parameter of the CDF
    :type x: float
    :param mu : Gaussian mean value
    :type mu: float
    :param sigma: Gaussian standard deviation
    :type sigma: float
    :return: CDF value at x
    :rtype: numpy.float64
    """
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0


def gaussianKernel(w):
    """
    2D gaussian kernel of radius w (size = 2*w + 1), and
    standard deviation sigma = w / 4.0

    :param w: kernel radius
    :type w: int
    :return: gaussian kernel, size s=2*w + 1
    :rtype: 2D array, shape (s,s), dtype numpy.float64
    """
    sigma = w / 4.0  # 8.0
    points = np.arange( 2 * w + 2) - (w + 0.5)  # [ -w - 0.5, ..., w + 0.5]
    # gaussian CDF
    points = map(lambda x: phi(x, 0, sigma), points)  # 0
    # 1D kernel
    kern1d = np.diff(list(points))
    kern1d /= np.sum(kern1d)
    ##### opencv version
    # cv2k1d = cv2.getGaussianKernel(2*w+1, sigma)[:,0]
    # cv2k2d = np.outer(cv2k1d, cv2k1d)
    #####
    # 2D kernel
    kernel = np.outer(kern1d, kern1d)  # k[i, j] = k[i] * k[j]
    return kernel


def kernelGaussianBlur(radius):
    return gaussianKernel(radius)


def kernelUnsharpMask(radius, amount):
    amount = amount / 100.0
    kernel = - kernelGaussianBlur(radius) * amount
    w = kernel.shape[0]
    # sum of coeff. must be 1.0
    kernel[w // 2, w // 2] += 1.0 + amount  # python 3 integer quotient
    return kernel


def kernelSharpen():
    kernel = np.array([[0.0, -1.0, 0.0],
                       [-1.0, 5.0, -1.0],
                       [0.0, -1.0, 0.0]])
    return kernel


def getKernel(category, radius=1, amount=1.0):
    if category == filterIndex.UNSHARP:
        return kernelUnsharpMask(radius, amount)
    elif category == filterIndex.SHARPEN:
        return kernelSharpen()
    elif category == filterIndex.BLUR1:
        return kernelGaussianBlur(2 * radius)
    else:
        return np.array([[1]])

