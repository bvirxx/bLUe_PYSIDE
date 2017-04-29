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
    IDENTITY, UNSHARP, SHARPEN, BLUR1, BLUR2 = range(5)

def phi(x, mu, sigma):
    """
    Cumulative distribution function (CDF) of the
    normal distribution N(mu, sigma).
    @param x: parameter of the CDF
    @type x: float
    @param mu : Gaussian mean value
    @type mu: float
    @param sigma: Gaussian standard deviation
    @type sigma: float
    @return: CDF value at x
    @rtype: numpy.float64
    """
    return (1.0 + erf((x-mu)/(sigma*np.sqrt(2)))) / 2.0

def gaussianKernel(mu, w):
    """
    2D gaussian kernel of size w and mean mu. 
    The standard deviation sigma and w are bound by the relation w = 2.0 * int(4.0 * sigma + 0.5)
    @param mu: gaussian mean
    @type mu: float
    @param w: kernel size, should be odd
    @type w: int 
    @return: gaussian kernel, size w
    @rtype: 2D array, shape (w,w), dtype numpy.float64
    """
    sigma = (w - 1.0) / 8.0
    interval = 4.0 * sigma
    points = np.linspace(-interval, interval, num=w + 1)
    # gaussian CDF
    points = map(lambda x : phi(x,0, sigma), points)
    #1D kernel
    kern1d = np.diff(points)
    # 2D kernel
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    # normalize
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def kernelGaussianBlur(radius):

    gblur_kernel = gaussianKernel(0.0, radius + 2)

    return gblur_kernel

def kernelUnsharpMask(radius, amount):

    amount = amount / 100.0
    kernel = - kernelGaussianBlur(radius) * amount

    w = kernel.shape[0]

    kernel[w/2, w/2] += 1.0 + amount

    return kernel

def kernelSharpen():

    kernel = np.array([[0.0, -1.0, 0.0],
                        [-1.0, 5.0, -1.0],
                        [0.0,-1.0, 0.0]])
    return kernel

def getKernel(category, radius=1, amount=1.0):
    if category == filterIndex.UNSHARP:
        return kernelUnsharpMask(radius, amount)
    elif category == filterIndex.SHARPEN:
        return kernelSharpen()
    elif category == filterIndex.BLUR1:
        return kernelGaussianBlur(radius)
    else:
        return np.array([[1]])


if __name__ == '__main__':
    print gaussianKernel(0, 5)*256