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

#############################################################################
# Temperature dependent conversion functions are located in this module:    #
# sRGB2LabVec, lab2sRGBVec, XYZ2sRGB, sRGB2XYZ, XYZ2sRGBLinear.             #
# Other conversion functions are change of coordinates in                   #
# a single RGB color space. They are located in the module                  #
# colorCube.py :                                                            #
# rgb2hlsVec,hls2RGBVec, hsv2RGBVec, hsp2RGBVec, hsp2RGBVecSmall,           #
# rgb2hsBVec, rgb2hspVec                                                    #
#############################################################################

sRGBWP = 6500

#####################
# Conversion Matrices
#####################
# According to Python and Numpy conventions, the following definitions of matrix
# constants M as lists and/or arrays correspond to
# M[row_index, col_index] = M[row_index][col_index] values.

# The Bradford matrix converts from
# CIE XYZ to the cone response domain.
# Cf. http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html

Bradford = [[0.8951000, 0.2664000, -0.1614000],
            [-0.7502000, 1.7135000, 0.0367000],
            [0.0389000, -0.0685000, 1.0296000]]

BradfordInverse = [[0.9869929, -0.1470543, 0.1599627],
                   [0.4323053, 0.5183603, 0.0492912],
                   [-0.0085287, 0.0400428, 0.9684867]]

# conversion matrices from LINEAR sRGB (D65) to XYZ and back.
# Cf. http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
# and https://en.wikipedia.org/wiki/SRGB

sRGB_lin2XYZ = [[0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]]

sRGB_lin2XYZInverse = [[3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660, 1.8760108, 0.0415560],
                       [0.0556434, -0.2040259, 1.0572252]]

# conversion matrices from LINEAR sRGB (D50) to XYZ  and back.
# Cf. http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
# and https://en.wikipedia.org/wiki/SRGB

sRGB_lin2XYZ_D50 = [[0.4360747,  0.3850649,  0.1430804],
                    [0.2225045,  0.7168786,  0.0606169],
                    [0.0139322,  0.0971045,  0.7141733]]

sRGB_lin2XYZ_D50Inverse = [[3.1338561, -1.6168667, -0.4906146],
                           [-0.9787684,  1.9161415,  0.0334540],
                           [0.0719453, -0.2289914,  1.4052427]]

# XYZ/sRGB/Lab conversion :
# D65 illuminant Xn, Yn, Zn and
# conversion constants Ka, Kb
# See https://en.wikipedia.org/wiki/Lab_color_space

Xn, Yn, Zn = 0.95047, 1.0, 1.08883
Ka, Kb = 172.355, 67.038


USE_BT_709 = False

class gammaTables:
    """
    # Constants, methods and precomputed tables for
    # RGB gamma adaptation.
    # The sRGB and REC BT 709 versions are available.
    # Cf. https://en.wikipedia.org/wiki/SRGB
    # Use the USE_BT_709 flag to switch between version.
    """
    if USE_BT_709:
        # REC BT 709
        gamma = 2.2
        a = 0.099
        d = 4.5
        gammaLinearTreshold1 = 0.018
        gammaLinearTreshold2 = 0.081
    else:
        # sRGB
        gamma = 2.4
        a = 0.055
        d = 12.92
        gammaLinearTreshold1 = 0.0031308
        gammaLinearTreshold2 = 0.04045

    beta = 1.0 / gamma

    @classmethod
    def getInstance(cls):
        """
        Returns a unique instance : a new instance
        is created only if there exists no instance yet.
        @return: gammaTables instance
        @rtype: gammaTables
        """
        if getattr(cls, 'instance', None) is None:
            cls.instance = gammaTables()
        return cls.instance

    def __init__(self):
        F = 255.0 ** self.beta
        table0 = np.arange(256, dtype=np.float64)
        # tabulate ( (x + a) / (1 + a) )**gamma
        # inputs should be integers in range 0..255.They are scaled to[0..1].
        self.__table3 = np.power((table0 / 255.0 + self.a) / (1 + self.a), self.gamma)
        trh = int(self.gammaLinearTreshold2 * 255)
        self.__table3[0: trh + 1] = np.arange(trh + 1) / ((255.0) * self.d)
        # tabulate (1 + a) * x**(1/gamma) - a :
        # inputs should be integers in range 0..255. They are scaled to [0..1].
        self.__table5 = np.power(table0, self.beta) * (1.0 + self.a) / F - self.a
        trh = int(self.gammaLinearTreshold1 * 255)
        self.__table5[0: trh + 1] = np.arange(trh + 1) * self.d
        self.__table5 *= 255

    @property
    def table3(self):
        """
        Gamma-Linearizing table.
        Values (outputs) are in range 0..1
        @return:
        @rtype: ndarray shape (255,)
        """
        return self.__table3

    @property
    def table5(self):
        """
        Gamma-adaptation table.
        Values (outputs) are in range 0..255
        @return:
        @rtype: ndarray shape (255,)
        """
        return self.__table5

def rgbLinear2rgb(rgbColors):
    """
    conversion from linear RGB to sRGB.
    See U{https://en.wikipedia.org/wiki/SRGB}
    Linear RGB values should be in range 0..1.
    Output values are in range 0..255.
    @param rgbColors: RGB color or array of RGB colors
    @type rgbColors: tuple or list of 3 scalars, or ndarray with last_dim = 3
    @return: gamma adapted RGB  colors
    @rtype: identical to type of input
    """
    try:
        vectorize = (rgbColors.ndim > 1)
    except AttributeError:
        vectorize = False
    if vectorize:
        gt = gammaTables.getInstance()
        imgDiscretized = (rgbColors * 255.0).astype(int)
        # np.clip(imgDiscretized, 0, 255, out=imgDiscretized)
        return gt.table5[imgDiscretized]
    else:
        gt = gammaTables
        def cl2c(c):
            if c <= gt.gammaLinearTreshold1:
                c = gt.d * c
            else:
                c = (1.0 + gt.a) * (c ** gt.beta) - gt.a
            return c
        r, g, b = rgbColors
        return cl2c(r) * 255, cl2c(g) * 255, cl2c(b) * 255


def rgb2rgbLinear(rgbColors):
    """
    Conversion from RGB to linear RGB.
    See https://en.wikipedia.org/wiki/SRGB
    Input Input values should be integers in range 0..255.
    Output values are in range 0..1.
    @param rgbColors: color or array of RGB colors
    @type rgbColors: tuple or list of 3 scalars, or ndarray with last_dim = 3
    @return: linear RGB colors
    @rtype: identical to type of input
    """
    try:
        vectorize = (rgbColors.ndim > 1)
    except AttributeError:
        vectorize = False
    if vectorize:
        gt = gammaTables.getInstance()
        return gt.table3[rgbColors]
    else:
        gt = gammaTables
        def c2cl(c):
            if c <= gt.gammaLinearTreshold2:
                # consider linear
                c = c / gt.d
            else:
                c = ((c + gt.a) / (1 + gt.a)) ** gt.gamma
            return c
        r, g, b = rgbColors
        return c2cl(r), c2cl(g), c2cl(b)


def sRGB2XYZ(rgbColors):
    """
    Conversion from sRGB (D65) to XYZ.
    Input values should be in integers in range 0..255 AND
    gamma-corrected.
    Output values are in range 0..1.
    opencv cvtColor does NOT perform gamma conversion
    for RGB<-->XYZ cf.
    U{http://docs.opencv.org/trunk/de/d25/imgproc_color_conversions.html#color_convert_rgb_xyz}.
    Moreover, RGB-->XYZ and XYZ-->RGB matrices are not inverse transformations!
    This yields incorrect results.
    As a workaround, we first convert to rgbLinear,
    and next use the conversion matrices from
    U{http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html}
    @param rgbColors: color or array of RGB colors
    @type rgbColors: tuple or list of 3 scalars, or ndarray with last_dim = 3
    @return: colors converted to the XYZ color space
    @rtype: ndarray, dtype numpy float64
    """
    bufLinear = rgb2rgbLinear(rgbColors)  # TODO modified 19/07/19 validate rgb2rgbLinearVec(imgBuf)
    bufXYZ = np.tensordot(bufLinear, sRGB_lin2XYZ, axes=(-1, -1))
    return bufXYZ


def XYZ2sRGBLinear(imgBuf):
    """
    vectorized conversion from XYZ to LINEAR sRGB (D65)
    @param imgBuf: image buffer,  XYZ color space
    @type imgBuf: list or 3-uple of scalars or ndarray shape (w, h, 3)
    @return: image buffer mode sRGB LINEAR, range 0..1
    @rtype: ndarray, shape (w, h, 3), dtype numpy.float64
    """
    return np.tensordot(imgBuf, sRGB_lin2XYZInverse, axes=(-1, -1))


def XYZ2sRGB(XYZColors):
    """
    Converts XYZ colors to gamma-adapted sRGB (D65) colors.
    Input values should be in range 0..1.
    Output values are in range 0..255
    opencv cvtColor does NOT perform gamma correction
    for RGB<-->XYZ conversion, cf.
    U{http://docs.opencv.org/trunk/de/d25/imgproc_color_conversions.html#color_convert_rgb_xyz}.
    Moreover, RGB-->XYZ and XYZ-->RGB matrices are not inverse transformations!
    This yields incorrect results. As a workaround, we first convert to rgbLinear,
    and next use the sRGB <--> XYZ conversion matrices from
    U{http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html}
    @param XYZColors: color, mode XYZ, range 0..1
    @type XYZColors: tuple or list of 3 scalars, or ndarray with last_dim = 3
    @return: RGB colors
    @rtype:  identical to type of input
    """
    try:
        vectorize = (XYZColors.ndim > 1)
    except AttributeError:
        vectorize = False
    if vectorize:
        M = np.max(XYZColors[..., 1])
    else:
        M = XYZColors[1]
        XYZColors = np.array(XYZColors)
    if M > 1:
        XYZColors /= M       # TODO a better approach is to to clip c1 ?
        print('XYZ2sRGBVec warning : Y channel max %.5f' % M)
    c1 = XYZ2sRGBLinear(XYZColors)
    c2 = rgbLinear2rgb(c1)  # TODO modified 19/07/19 validate rgbLinear2rgb(*c1)
    return c2


def sRGB2LabVec(bufsRGB, useOpencv=True):
    """
    Vectorized sRGB to Lab conversion for 8 bits images only.  No clipping
    is performed. If useOpencv is True (default, faster),
    we use opencv cvtColor (Note that it seems to perform
    linearizations, in contrast to sRGB <---> XYZ conversions)
    See U{https://en.wikipedia.org/wiki/Lab_color_space}
    The range for Lab coordinates is L:0..1, a:-86.185..98.254, b:-107.863..94.482
    See U{http://stackoverflow.com/questions/19099063/what-are-the-ranges-of-coordinates-in-the-cielab-color-space}
    @param bufsRGB: image buffer, mode sRGB, range 0..255
    @type bufsRGB: ndarray, dtype=np.uint8
    @param useOpencv:
    @type useOpencv: boolean
    @return: bufLab Image buffer, mode Lab
    @rtype: ndarray, dtype numpy float64
    """
    if useOpencv:
        bufLab = cv2.cvtColor(bufsRGB, cv2.COLOR_RGB2Lab)
        bufLab = bufLab.astype(np.float)
        # for 8 bits per channel images opencv uses L,a,b range 0..255
        bufLab[:, :, 0] /= 255.0
        bufLab[:, :, 1:] -= 128
    else:
        oldsettings = np.seterr(all='ignore')
        bufXYZ = sRGB2XYZ(bufsRGB)  # * 100.0
        YoverYn = bufXYZ[:, :, 1] / Yn
        bufL = np.sqrt(YoverYn)
        bufa = Ka * (bufXYZ[:, :, 0] / Xn - YoverYn) / bufL
        bufb = Kb * (YoverYn - bufXYZ[:, :, 2] / Zn) / bufL
        np.seterr(**oldsettings)
        bufLab = np.dstack((bufL, bufa, bufb))
        # converting invalid values to int gives indeterminate results
        bufLab[np.isnan(bufLab)] = 0.0  # TODO should be np.inf ?
    return bufLab


def Lab2sRGBVec(bufLab, useOpencv=True):
    """
    Vectorized Lab to sRGB conversion. No clipping
    is performed. If useOpencv is True (default, faster),
    we use opencv cvtColor.

    See U{https://en.wikipedia.org/wiki/Lab_color_space}
    @param bufLab: image buffer, mode Lab, range 0..1
    @type bufLab: ndarray, dtype numpy float
    @param useOpencv:
    @type useOpencv: boolean
    @return: Image buffer mode sRGB, range 0..255,
    @rtype: ndarray, dtype=np.uint8
    """
    if useOpencv:
        # for 8 bits per channel images opencv uses L,a,b range 0..255
        tmp = bufLab + [0.0, 128.0, 128.0]
        tmp[:, :, 0] *= 255.0
        bufsRGB = cv2.cvtColor(tmp.astype(np.uint8), cv2.COLOR_Lab2RGB)
    else:
        bufL, bufa, bufb = bufLab[:, :, 0], bufLab[:, :, 1], bufLab[:, :, 2]
        bufL2 = bufL * bufL
        bufY = bufL2 * Yn
        bufX = Xn * ((bufa / Ka) * bufL + bufL2)
        bufZ = Zn * (bufL2 - (bufb / Kb) * bufL)
        bufXYZ = np.dstack((bufX, bufY, bufZ))  # /100.0
        bufsRGB = XYZ2sRGB(bufXYZ)  # TODO modified 19/07/19 validate XYZ2sRGBVec(bufXYZ)
        # converting invalid values to int gives indeterminate results
        bufsRGB[np.isnan(bufsRGB)] = 0.0  # TODO should be np.inf ?
    return bufsRGB


def bbTemperature2RGB(temperature):
    """
    Converts black body Kelvin temperature to rgb values.
    Cf. http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code
    @param temperature: Kelvin temperature
    @type temperature: float
    @return: r, g, b values in  range 0..255
    @rtype: 3-uple of int
    """
    temperature = temperature / 100.0
    if temperature <= 66:
        red = 255
        green = temperature
        green = 99.4708025861 * np.log(green) - 161.1195681661
    else:
        red = temperature - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = min(max(0, red), 255)
        green = temperature - 60
        green = 288.1221695283 * (green ** -0.0755148492)
    green = min(max(0, green), 255)
    if temperature >= 66:
        blue = 255
    else:
        if temperature <= 19:
            blue = 0
        else:
            blue = temperature - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
            blue = min(max(0, blue), 255)
    return int(red), int(green), int(blue)


##############################################
# Chromatic adaptation.
# We use the approximation of the Planckian Locus by
# a cubic spline as described in http://en.wikipedia.org/wiki/Planckian_locus#Approximation
# combined with the cone response matrix method.
# Cf. https://web.stanford.edu/~sujason/ColorBalancing/adaptation.html for details
##################################################
def xyWP2temperature(x, y):
    """
    Calculate the temperature from White point coordinates in the color space xy
    We use spline approximation (see https://en.wikipedia.org/wiki/Color_temperature#Approximation)
    @param x:
    @type x: float
    @param y:
    @type y: float
    @return: Temperature in Kelvin
    @rtype: float
    """
    xe, ye = 0.3320, 0.1858
    n = (x - xe) / (y - ye)
    T = - 449.0 * (n ** 3) + 3525.0 * (n ** 2) - 6823.3 * n + 5520.33
    return T


def temperature2xyWP(T):
    """
    Calculate the (CIE chromaticity) coordinates xc, yc
    of WP(T) in the xy color space (use cubic spline approximation,
    accurate for 1667<T<25000).
    Cf. http://en.wikipedia.org/wiki/Planckian_locus#Approximation
    @param T: temperature in Kelvin, range 1667..25000
    @type T: float
    @return: xc, yc
    @rtype: 2-uple of float
    """
    # get xc
    if T <= 4000:
        xc = -0.2661239 * (10 ** 9) / (T ** 3) - 0.2343580 * (10 ** 6) / (T ** 2) + 0.8776956 * (
                    10 ** 3) / T + 0.179910  # 1667<T<4000
    else:
        xc = -3.0258469 * (10 ** 9) / (T ** 3) + 2.1070379 * (10 ** 6) / (T ** 2) + 0.2226347 * (
                    10 ** 3) / T + 0.240390  # 4000<T<25000
    # get yc
    if T <= 2222:
        yc = -1.1063814 * (xc ** 3) - 1.34811020 * (xc ** 2) + 2.18555832 * xc - 0.20219683  # 1667<T<2222
    elif T <= 4000:
        yc = -0.9549476 * (xc ** 3) - 1.37418593 * (xc ** 2) + 2.09137015 * xc - 0.16748867  # 2222<T<4000
    else:
        yc = 3.0817580 * (xc ** 3) - 5.87338670 * (xc ** 2) + 3.75112997 * xc - 0.37001483  # 4000<T<25000
    return xc, yc

###################################################################################################
# The next table records lines [(10**6/T), u, v, slope], with T = temperature, (u,v) = WP coordinates in CIEYUV,
# slope = isotherm slope, for temperatures from 1666.66K to infinity.
# We use it as an interpolation table for converting u,v coordinates to and
# # from (Temperature, Tint) (Robertson's method)
# References
#   1) Wyszecki and Stiles book "Color Science", 2nd edition, p 228.
#   2) http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_T.html
###################################################################################################


uvtTable = [
    [0, 0.18006, 0.26352, -0.24341],
    [10, 0.18066, 0.26589, -0.25479],
    [20, 0.18133, 0.26846, -0.26876],
    [30, 0.18208, 0.27119, -0.28539],
    [40, 0.18293, 0.27407, -0.30470],
    [50, 0.18388, 0.27709, -0.32675],
    [60, 0.18494, 0.28021, -0.35156],
    [70, 0.18611, 0.28342, -0.37915],
    [80, 0.18740, 0.28668, -0.40955],
    [90, 0.18880, 0.28997, -0.44278],
    [100, 0.19032, 0.29326, -0.47888],
    [125, 0.19462, 0.30141, -0.58204],
    [150, 0.19962, 0.30921, -0.70471],
    [175, 0.20525, 0.31647, -0.84901],
    [200, 0.21142, 0.32312, -1.0182],
    [225, 0.21807, 0.32909, -1.2168],
    [250, 0.22511, 0.33439, -1.4512],
    [275, 0.23247, 0.33904, -1.7298],
    [300, 0.24010, 0.34308, -2.0637],
    [325, 0.24792, 0.34655, -2.4681],
    [350, 0.25591, 0.34951, -2.9641],
    [375, 0.26400, 0.35200, -3.5814],
    [400, 0.27218, 0.35407, -4.3633],
    [425, 0.28039, 0.35577, -5.3762],
    [450, 0.28863, 0.35714, -6.7262],
    [475, 0.29685, 0.35823, -8.5955],
    [500, 0.30505, 0.35907, -11.324],
    [525, 0.31320, 0.35968, -15.628],
    [550, 0.32129, 0.36011, -23.325],
    [575, 0.32931, 0.36038, -40.770],
    [600, 0.33724, 0.36051, -116.45]
]


def xy2uv(x, y):
    """
    convert from xy to Yuv color space
    @param x:
    @type x: float
    @param y:
    @type y: float
    @return: u, v coordinates
    @rtype: 2-uple of float
    """
    d = 1.5 - x + 6.0 * y
    u, v = 2.0 * x / d, 3.0 * y / d
    return u, v


def uv2xy(u, v):
    d = u - 4.0 * v + 2.0
    x, y = 1.5 * u / d, v / d
    return x, y


def xy2TemperatureAndTint(x, y):
    """
    Convert xy coordinates of a color point to Temperature and Tint.
    The conversion is based on the Robertson's method
    of interpolation in the uv space.
    Tint is a shift in the uv space.
    @param x:
    @type x: float
    @param y:
    @type y: float
    @return: Temperature and Tint
    @rtype: 2-uple of float
    """
    # convert to uv
    u, v = xy2uv(x, y)
    last_dt, last_dv, last_du = 0.0, 0.0, 0.0

    temp, tint = 0, 0
    for index in range(31):
        # get unit vector of current isotherm
        du, dv = 1.0, uvtTable[index][3]
        n = np.sqrt(1.0 + dv * dv)
        du, dv = du / n, dv / n
        # get vector from current WP to u, v
        uu, vv = u - uvtTable[index][1], v - uvtTable[index][2]
        # get algebraic distance from (u,v) to current isotherm
        dt = - uu * dv + vv * du  # (-dv, du) is a unit vector orthogonal to the isotherm
        if dt <= 0 or (index == 30):
            if index == 0:
                raise ValueError('xy2TemperatureAndTint : Temp should not be infinity')
            if index == 30:
                if dt > 0:
                    raise ValueError('xy2TemperatureAndTint : Temp should be >= 1667 K')
            dt = -dt
            # interpolate 1/Temp between index-1 and index
            w = dt / (last_dt + dt)

            temp = 10 ** 6 / (w * uvtTable[index - 1][0] + (1.0 - w) * uvtTable[index][0])

            # interpolate unit vectors along isotherms
            du, dv = du * (1.0 - w) + last_du * w, dv * (1.0 - w) + last_dv * w
            n = np.sqrt(du * du + dv * dv)
            du, dv = du / n, dv / n
            tint = (u * du + v * dv)  # * TintScale
            break
        last_dt, last_du, last_dv = dt, du, dv
    return temp, tint


def temperatureAndTint2xy(temp, tint):
    """
    Convert temperature and tint to xy coordinates.
    The conversion is based on the Robertson's method of interpolation.
    Tint is a shift : for tint=0.0, the function gives the xy coordinates of the white point WP(T) :
    Cf. also temperature2xyWP(T).

    @param temp:
    @type temp: float
    @param tint:
    @type tint: float
    @return: x, y coordinates
    @rtype: 2-uple of float
    """
    r = (10 ** 6) / temp
    result = (0.0, 0.0)
    for index in range(30):
        if (r < uvtTable[index + 1][0]) or (r == 29):
            if r >= uvtTable[index + 1][0]:
                raise ValueError('TemperatureAndTint2xy: Temp should be >= 1667 K')
            w = (uvtTable[index + 1][0] - r) / (uvtTable[index + 1][0] - uvtTable[index][0])
            # interpolate WP coordinates
            WPu = uvtTable[index][1] * w + uvtTable[index + 1][1] * (1.0 - w)
            WPv = uvtTable[index][2] * w + uvtTable[index + 1][2] * (1.0 - w)
            # interpolate isotherms
            uu1, vv1 = 1.0, uvtTable[index][2]
            uu2, vv2 = 1.0, uvtTable[index + 1][2]
            n1, n2 = np.sqrt(uu1 * uu1 + vv1 * vv1), np.sqrt(uu2 * uu2 + vv2 * vv2)
            uu1, vv1 = uu1 / n1, vv1 / n1
            uu2, vv2 = uu2 / n2, vv2 / n2
            uu3, vv3 = w * uu1 + (1.0 - w) * uu2, w * vv1 + (1.0 - w) * vv2
            n3 = np.sqrt(uu3 * uu3 + vv3 * vv3)
            uu3, vv3 = uu3 / n3, vv3 / n3
            # shift WP along the isotherm according to tint
            u, v = WPu + uu3 * tint, WPv + vv3 * tint
            result = uv2xy(u, v)
            break
    return result


def wp2Rho(x, y):
    """
    Returns the cone responses (multipliers) for the (x, y) white point
    see https://web.stanford.edu/~sujason/ColorBalancing/adaptation.html for details.
    @param x: white point x-coord
    @type x: float
    @param y: white point y-coord
    @type y: float
    @return: cones responses
    @rtype: 3-uple of float
    """
    # expand x, y to XYZ coordinates
    X, Y, Z = x / y, 1.0, (1.0 - x - y) / y
    rho1, rho2, rho3 = np.dot(np.array(Bradford), np.array([X, Y, Z]))
    return rho1, rho2, rho3


def temperature2Rho(T):
    """
    Returns the cone responses (multipliers) for temperature T (Kelvin).
    see https://web.stanford.edu/~sujason/ColorBalancing/adaptation.html for details.
    @param T: temperature (Kelvin)
    @type T: float
    @return: cone responses
    @rtype: 3-uple of floats
    """
    # get CIE chromaticity coordinates of white point
    x, y = temperature2xyWP(T)
    return wp2Rho(x, y)


def rho2BAM(rhod, rhos):
    """
    Build the Bradford adaptation matrix from source multipliers rhos
    to destination multipliers rhod
    @param rhod:
    @type rhod: 3-uple of float
    @param rhos:
    @type rhos: 3-uple of float
    @return: adaptation matrix
    @rtype: ndarray, shape=(3,3), dtype=float
    """
    rhos1, rhos2, rhos3 = rhos
    rhod1, rhod2, rhod3 = rhod
    D = np.diag((rhod1 / rhos1, rhod2 / rhos2, rhod3 / rhos3))
    N = np.dot(np.array(BradfordInverse), D)  # N = (B**-1) D
    P = np.dot(N, np.array(Bradford))  # P = N B = (B**-1) D B
    return P


def bradfordAdaptationMatrix_wp(wpd, wps):
    """
    Returns the conversion matrix in the XYZ color space, from
    wps to wpd. We apply the method described in
    https://web.stanford.edu/~sujason/ColorBalancing/adaptation.html.
    @param wpd: destination white point xy coordinates
    @type wpd: 2-uple of float
    @param wps: source white point xy coordinates
    @type wps: 2-uple of float
    @return: adaptation matrix
    @rtype: ndarray, shape=(3,3), dtype=float
    """
    rhos1, rhos2, rhos3 = wp2Rho(*wps)
    rhod1, rhod2, rhod3 = wp2Rho(*wpd)
    return rho2BAM((rhod1, rhod2, rhod3), (rhos1, rhos2, rhos3))


def bradfordAdaptationMatrix(Tdest, Tsource):
    """
    Returns the conversion matrix in the XYZ color space, from
    Tsource to Tdest. We apply the method described in
    https://web.stanford.edu/~sujason/ColorBalancing/adaptation.html.
    @param Tdest: destination temperature (Kelvin)
    @type Tdest: float
    @param Tsource: Source temperature (Kelvin)
    @type Tsource: float
    @return: adaptation matrix
    @rtype: ndarray, shape=(3,3), dtype=float
    """
    rhos1, rhos2, rhos3 = temperature2Rho(Tsource)
    rhod1, rhod2, rhod3 = temperature2Rho(Tdest)
    return rho2BAM((rhod1, rhod2, rhod3), (rhos1, rhos2, rhos3))


def xyY2XYZ(xyYColors):
    """
    Converts a xyY color or an array of xyY colors into the XYZ color space.
    @param xyYColors: color or array of colors in the xyY color space
    @type xyYColors: tuple or list of 3 scalars, or ndarray with last_dim = 3
    @return: color or array of colors in the XYZ color space
    @rtype: identical to the input type
    """
    try:
        vectorize = (xyYColors.ndim > 1)
    except AttributeError:
        vectorize = False
    if vectorize:
        x, y, Y = xyYColors[..., :]
    else:
        x, y, Y = xyYColors
    Y_over_y = Y / y
    r = x * Y_over_y, Y, (1 - x - y) * Y_over_y
    return np.dstack(r) if vectorize else r


"""
if __name__ == '__main__':
    T = 4000.0
    r, g, b = bbTemperature2RGB(T)
    x, y = temperature2xyWP(T)
    L = 0.7
    r1, g1, b1 = np.dot(sRGB_lin2XYZInverse, np.array([L * x / y, L, L * (1.0 - x - y) / y]).T)
    r2, g2, b2 = rgbLinear2rgb(r1, g1, b1)
    # print r,g,b
    # print r2, g2, b2
"""
