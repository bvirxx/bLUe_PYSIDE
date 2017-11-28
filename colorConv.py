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
import cv2
import numpy as np

#######################################################################
# This module implements temperature dependent                        #
# conversion functions for color spaces.                              #
# sRGB color space (illuminant D65)                                   #
# is assumed for all input and output images.                         #
#######################################################################

################
# Conversion Matrices
#################

# Conversion from CIE XYZ to LMS-like color space for chromatic adaptation
# see http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html

sRGBWP = 6500

Von_Kries =  [[0.4002400,  0.7076000, -0.0808100],
              [-0.2263000, 1.1653200,  0.0457000],
              [0.0000000,  0.0000000,  0.9182200]]

Von_KriesInverse =  [[1.8599364, -1.1293816,  0.2198974],
                     [0.3611914,  0.6388125, -0.0000064],
                     [0.0000000,  0.0000000,  1.0890636]]

Bradford =  [[0.8951000,  0.2664000, -0.1614000],  #photoshop and best
             [-0.7502000, 1.7135000,  0.0367000],
             [0.0389000,  -0.0685000, 1.0296000]]

BradfordInverse =  [[0.9869929, -0.1470543,  0.1599627],
                    [0.4323053,  0.5183603,  0.0492912],
                    [-0.0085287, 0.0400428,  0.9684867]]

#########################
# conversion from LINEAR sRGB (D65) to XYZ and back.
# see http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
# and https://en.wikipedia.org/wiki/SRGB
########################

sRGB2XYZ = [[0.4124564,  0.3575761,  0.1804375],
            [0.2126729,  0.7151522,  0.0721750],
            [0.0193339,  0.1191920,  0.9503041]]

sRGB2XYZInverse = [[3.2404542, -1.5371385, -0.4985314],
                   [-0.9692660, 1.8760108,  0.0415560],
                    [0.0556434, -0.2040259, 1.0572252]]

######################
# XYZ/Lab conversion :
# D65 illuminant Xn, Yn, Zn
# conversion constants Ka, Kb
# See https://en.wikipedia.org/wiki/Lab_color_space
####################
Xn, Yn, Zn = 0.95047, 1.0, 1.08883 #95.02, 100.0, 108.82 #95.047, 100.0, 108.883
Ka, Kb = 172.355, 67.038 #172.30, 67.20

################
# Constants and precomputed tables for the
# sRGB linearizing functions
# rgbLinear2rgbVec and rgb2rgbLinearVec.
# See https://en.wikipedia.org/wiki/SRGB
################

a = 0.055
alpha = 2.4
beta = 1.0 / alpha
b = (a / (1.0 + a)) ** alpha
d = 12.92
c = 255.0 * d
# tabulation of x**beta
# e is the size of the table
e = 255 #255*255
F = e**beta #255.0**(2*beta)
# tabulated functions
table0 = np.arange(256, dtype=np.float64)
table1 = table0 / 255.0
table2 = table0 / c
table3 = np.power(table1, alpha)  # (i/255)**alpha
# tabulation of x**beta
table4 = np.arange(e + 1, dtype = np.float64)
table5 = np.power(table4, beta) *(1.0+a)/F # i**beta

def rgbLinear2rgb(r,g,b):
    """
    Conversion from linear sRGB to sRGB.
    All values are in range 0..1.
    
    @param r:
    @param g:
    @param b:
    @return: The converted values
    """
    def cl2c(c):
        if c <= 0.0031308:
            c = d * c
        else:
            c = (1.0 + a) * (c**beta) - a
        return c
    return cl2c(r)*255, cl2c(g)*255, cl2c(b)*255

def rgbLinear2rgbVec(img):
    """
    Converts image from linear sRGB to sRGB.
    See https://en.wikipedia.org/wiki/SRGB
    @param img: linear sRGB image, range 0..1
    @type img: numpy array, dtype=float
    @return: converted RGB image
    @rtype: numpy array, dtype=float, range 0..255
    """
    img2 = img * d
    imgDiscretized = (img * e).astype(int)
    imgDiscretized = np.clip(imgDiscretized, 0, e)
    img3 = table5[imgDiscretized] #* ((1.0+a)/F)
    return np.where(img <=  0.0031308, img2, img3) * 255

gammaLinearTreshold = 0.04045
def rgb2rgbLinear(r,g,b):
    """
       Conversion from sRGB to LINEAR sRGB.
       All values are in range 0..1.
       See https://en.wikipedia.org/wiki/SRGB
       @param r:
       @param g:
       @param b:
       @return: The converted values
       """
    def c2cl(c):
        if c <= gammaLinearTreshold:
            # consider linear
            c =  c / d
        else:
            c = ((c+a)/(1+a))**alpha
        return c
    return c2cl(r), c2cl(g), c2cl(b)

def rgb2rgbLinearVec(img):
    """
    Converts image from sRGB to linear sRGB.
    See https://en.wikipedia.org/wiki/SRGB
    @param img: RGB image, range 0..255
    @type img: numpy array, dtype=uint8 or int or float
    @return: converted linear RGB image, range 0..1
    @rtype: numpy array, dtype=float
    """
    img2 = table2[img[...]]  # equivalent to img2 = img / c, faster
    img3 = table3[img[...]]  # img3 = power(img, alpha)
    tr = gammaLinearTreshold * 255.0
    return np.where(img <= tr, img2, img3)

def sRGB2XYZVec(imgBuf):
    """
    Conversion from sRGB to XYZ
    @param imgBuf: Array of RGB values, range 0..255
    @return: 
    """
    #buf = QImageBuffer(img)[:, :, :3][:,:,::-1]
    bufLinear = rgb2rgbLinearVec(imgBuf)
    bufXYZ = np.tensordot(bufLinear, sRGB2XYZ, axes=(-1, -1))
    return bufXYZ

def XYZ2sRGBVec(imgBuf):
    """
    Vectorized XYZ to sRGB conversion.
    @param imgBuf: image buffer, mode XYZ
    @type imgBuf: ndarray
    @return: image buffer, mode sRGB, range 0..255
    @rtype: ndarray, dtype numpy.float64
    """
    #buf = QImageBuffer(img)[:, :, :3]
    bufsRGBLinear = np.tensordot(imgBuf, sRGB2XYZInverse, axes=(-1, -1))
    bufsRGB = rgbLinear2rgbVec(bufsRGBLinear)
    return bufsRGB

def sRGB2LabVec(bufsRGB, useOpencv = True) :
    """
    Vectorized sRGB to Lab conversion.  No clipping
    is performed.
    
    See U{https://en.wikipedia.org/wiki/Lab_color_space}
    
    range for Lab coordinates is L:0..1, a:-86.185..98.254, b:-107.863..94.482
    
    See U{http://stackoverflow.com/questions/19099063/what-are-the-ranges-of-coordinates-in-the-cielab-color-space}
    @param bufsRGB: image buffer, mode sRGB, range 0..255
    @type bufsRGB: ndarray
    @return: bufLab Image buffer mode Lab
    @rtype: ndarray, dtype numpy.float64
    """
    if useOpencv :
        bufLab = cv2.cvtColor(bufsRGB, cv2.COLOR_RGB2Lab)
        bufLab = bufLab.astype(np.float)
        # for 8 bits per channel images opencv uses L,a,b range 0..255
        bufLab[:,:,0] = bufLab[:,:,0] / 255.0
        bufLab[:,:,1:] = bufLab[:,:,1:] - 128
        # bufLab[:, :, 2] = bufLab[:, :, 2] - 128
    else :
        oldsettings = np.seterr(all='ignore')
        bufXYZ = sRGB2XYZVec(bufsRGB) # * 100.0
        YoverYn = bufXYZ[:,:,1] / Yn
        bufL = np.sqrt(YoverYn)
        bufa = Ka * ( bufXYZ[:,:,0] / Xn - YoverYn) / bufL
        bufb = Kb * (YoverYn - bufXYZ[:,:,2]/Zn) / bufL
        np.seterr(**oldsettings)
        bufLab = np.dstack((bufL, bufa, bufb))
        # converting invalid values to int gives indeterminate results
        bufLab[np.isnan(bufLab)] = 0.0  # TODO np.inf
    return bufLab


def Lab2sRGBVec(bufLab, useOpencv = True):
    """
    Vectorized Lab to sRGB conversion. No clipping
    is performed.
    
    See U{https://en.wikipedia.org/wiki/Lab_color_space}
    @param bufLab: image buffer, mode Lab, range 0..1
    @return: bufLab Image buffer mode sRGB, range 0..255, 
    @rtype: ndarray, dtype numpy.float64
    """
    if useOpencv:
        tmp = bufLab.copy()
        # for 8 bits per channel images opencv uses L,a,b range 0..255
        tmp[:,:,0] = tmp[:,:,0] * 255.0
        tmp[:,:,1:] = tmp[:,:,1:] + 128
        #tmp[:, :, 2] = tmp[:, :, 2] + 128
        bufsRGB = cv2.cvtColor(tmp.astype(np.uint8), cv2.COLOR_Lab2RGB)
    else:
        bufL, bufa, bufb = bufLab[:,:,0], bufLab[:,:,1], bufLab[:,:,2]
        bufL2 = bufL* bufL
        bufY = bufL2 * Yn
        bufX = Xn * ((bufa/ Ka) * bufL + bufL2)
        bufZ = Zn * (bufL2 - ((bufb / Kb)) * bufL)
        bufXYZ = np.dstack((bufX, bufY, bufZ)) # /100.0
        bufsRGB = XYZ2sRGBVec(bufXYZ)
        # converting invalid values to int gives indeterminate results
        bufsRGB[np.isnan(bufsRGB)] = 0.0  # TODO np.inf
    return bufsRGB

def bbTemperature2RGB(temperature):
    """
    Converts Kelvin temperature to rgb values.
    See http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code
    @param temp: Kelvin temperature
    @type temp: float
    @return: r, g, b values in  range 0..255
    @rtype: 3-uple of int
    """
    temperature = temperature / 100.0
    if temperature <= 66 :
        red = 255
        green = temperature
        green = 99.4708025861 * np.log(green) - 161.1195681661
    else:
        red = temperature - 60
        red = 329.698727446 * (red**-0.1332047592)
        red = min(max(0, red), 255)
        green = temperature - 60
        green = 288.1221695283 * (green**-0.0755148492)
    green = min(max(0, green), 255)
    if temperature >= 66:
        blue = 255
    else:
        if temperature <= 19:
            blue = 0
        else:
            blue = temperature - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
            blue = min(max(0,blue), 255)
    return int(red), int(green), int(blue)

##############################################
# Chromatic adaptation.
# We use the approximation of the Planckian Locus by
# a cubic spline as described in http://en.wikipedia.org/wiki/Planckian_locus#Approximation
# combined with the cone response matrix method.
# See https://web.stanford.edu/~sujason/ColorBalancing/adaptation.html for details

def temperature2xyWP(T):
    """
    Calculates the CIE chromaticity coordinates xc, yc
    of white point from temperature (cubic spline approximation).
    see http://en.wikipedia.org/wiki/Planckian_locus#Approximation
    @param T: temperature in Kelvin, range 1667..25000
    @type T: float
    @return: xc, yc
    @rtype: 2-uple of float
    """

    if T <= 4000:
        xc = -0.2661239 *(10**9) / (T**3) - 0.2343580 *(10**6) / (T**2) + 0.8776956 * (10**3) / T + 0.179910  # 1667<T<4000
    else:
        xc = - 3.0258469 *(10**9) / (T**3) + 2.1070379 *(10**6) / (T**2) + 0.2226347 * (10**3) / T + 0.240390 # 4000<T<25000
    if T <= 2222:
        yc = -1.1063814 * (xc**3) - 1.34811020 * (xc**2) + 2.18555832 * xc - 0.20219683  #1667<T<2222
    elif T<= 4000:
        yc = -0.9549476 *(xc**3) - 1.37418593 * (xc**2) + 2.09137015 * xc - 0.16748867  # 2222<T<4000
    else:
        yc = 3.0817580 * (xc**3) - 5.87338670 *(xc**2) + 3.75112997  * xc - 0.37001483  # 4000<T<25000
    return xc, yc

def temperature2Rho(T):
    """
    Returns the cone responses for temperature T (Kelvin).
    see https://web.stanford.edu/~sujason/ColorBalancing/adaptation.html for details.
    @param T: temperature (Kelvin)
    @type T: float
    @return: 3-uple of cone responses
    @rtype: 3-uple of floats
    """
    # get CIE chromaticity coordinates of white point
    x, y = temperature2xyWP(T)
    L = 1.0 # arbitrary non zero constant
    # transform in XYZ coordinates
    X, Y , Z = L * x / y, L, L * (1.0 - x - y ) / y
    rho1, rho2, rho3 = np.dot(np.array(Bradford), np.array([X,Y,Z]).T)
    return rho1, rho2, rho3

def conversionMatrix(Tdest, Tsource):
    """
    Returns the conversion matrix in the XYZ color space, from
    Tsource to Tdest. We apply the method described in
    https://web.stanford.edu/~sujason/ColorBalancing/adaptation.html.
    @param Tdest: destination temperature (Kelvin)
    @type Tdest: float
    @param Tsource: Source temperature (Kelvin)
    @type Tsource: float
    @return: np array
    @rtype: shape=(3,3), dtype=float
    """
    rhos1, rhos2, rhos3  = temperature2Rho(Tsource)
    rhod1, rhod2, rhod3 = temperature2Rho(Tdest)
    D = np.diag((rhod1/rhos1, rhod2/rhos2, rhod3/rhos3))
    N = np.dot(np.array(BradfordInverse), D)  # N= (MA**-1) D
    P = np.dot(N, np.array(Bradford))         # P = N MA = (MA**-1) D MA
    return P


if __name__ == '__main__':
    T=4000.0
    r,g,b = bbTemperature2RGB(T)
    x,y = temperature2xyWP(T)
    L=0.7
    r1, g1, b1 = np.dot(sRGB2XYZInverse, np.array([L*x/y, L, L* (1.0 -x - y)/y]).T)
    r2, g2, b2 = rgbLinear2rgb(r1,g1,b1)
    #print r,g,b
    #print r2, g2, b2

