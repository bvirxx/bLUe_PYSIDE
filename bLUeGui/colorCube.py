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

###############################################
# Weights for perceptual brightness calculation
###############################################
# usual values
"""
Perc_R=0.2126
Perc_G=0.7152
Perc_B=0.0722
"""
# interpolated values
Perc_R = 0.2338
Perc_G = 0.6880
Perc_B = 0.0782


def rgb2hsp(r, g, b):
    return rgb2hsB(r, g, b, perceptual=True)


def rgb2hspVec(rgbImg):
    return rgb2hsBVec(rgbImg, perceptual=True)


def rgb2hsB(r, g, b, perceptual=False):
    """
    transforms the r, g, b components of a color
    to hue, saturation, brightness h, s, v.
    The r, g, b components are integers in range 0..255. If perceptual is False
    (default) v = max(r,g,b)/255.0, else v = sqrt(Perc_R*r*r + Perc_G*g*g + Perc_B*b*b)

    :param r: range 0..255
    :type  r: int
    :param g: range 0..255
    :type  g: int
    :param b: range 0..255
    :type  b: int
    :param perceptual:
    :type  perceptual: boolean
    :return: h, s, v values : 0<=h<360, 0<=s<=1, 0<=v<=1
    :rtype: float
    """
    cMax = max(r, g, b)
    cMin = min(r, g, b)
    if cMax > 255 or cMin < 0:
        raise ValueError('rgb2hsB : r, g, b must be in range 0..255')
    delta = cMax - cMin
    # hue
    if delta == 0:
        H = 0.0
    elif cMax == r:
        H = 60.0 * float(g - b) / delta if g >= b else 360 + 60.0 * float(g - b) / delta
    elif cMax == g:
        H = 60.0 * (2.0 + float(b - r) / delta)
    elif cMax == b:
        H = 60.0 * (4.0 + float(r - g) / delta)
    # saturation
    S = 0.0 if cMax == 0.0 else float(delta) / cMax
    # brightness
    if perceptual:
        V = np.sqrt(Perc_R * r * r + Perc_G * g * g + Perc_B * b * b)
    else:
        V = cMax
    V = V / 255.0
    assert 0 <= H <= 360 and 0 <= S <= 1 and 0 <= V <= 1, "rgb2hsv conversion error r=%d, g=%d, b=%d, h=%f, s=%f, v=%f" % (
    r, g, b, H, S, V)
    return H, S, V


def rgb2hsBVec(rgbImg, perceptual=False):
    """
    Vectorized version of rgb2hsB.
    RGB-->HSV color space transformation
    The r, g, b components are integers in range 0..255. If perceptual is False
    (default) V = max(r,g,b)/255.0, else V = sqrt(Perc_R*r*r + Perc_G*g*g + Perc_B*b*b) / 255

    :param rgbImg: RGB image range 0..255
    :type  rgbImg: (n,m,3) array, , dtype=uint8 or dtype=int or dtype=float
    :param perceptual:
    :type  perceptual: boolean
    :return: identical shape array of hue,sat,brightness values (0<=h<=360, 0<=s<=1, 0<=v<=1)
    :rtype: (n,m,3) array, dtype=float
    """
    buf = cv2.cvtColor(rgbImg.astype(np.uint8), cv2.COLOR_RGB2HSV)
    buf = buf.astype(float)
    buf *= [2, 1.0 / 255.0, 1.0 / 255.0]  # scale to 0..360, 0..1, 0..1
    if perceptual:
        rgbImg2 = rgbImg.astype(float)
        rgbImg2 *= rgbImg2
        pB = np.tensordot(rgbImg2, [Perc_R, Perc_G, Perc_B], axes=(-1, -1))
        pB /= 255.0 * 255
        # pB = np.sqrt(pB)
        # buf[:, :, 2] = pB
        np.sqrt(pB, out=buf[:, :, 2])
    return buf


def rgb2hlsVec(rgbImg):
    """
    Converts RGB color space to HLS.
    With M = max(r,g,b) and m = min(r,g,b) the HLS color space uses
    luminosity L= (M+m)/2 and saturation S = (M-m)/(M+m) if l < 0.5 and
    S =  (M-m) / (2-(M+m)) otherwise, (0<=h<=360, 0<=l<=1, 0<=s<=1).
    We do not follow the opencv convention for HLS value ranges.

    :param rgbImg: rgbImg: array of r,g, b values
    :type  rgbImg: rgbImg: (n,m,3) array, , dtype=uint8 or dtype=int or dtype=float
    :return: identical shape array of hue,luma, chroma values (0<=h<=360, 0<=l<=1, 0<=s<=1)
    :rtype: (n,m,3) array, dtype=float
    """
    buf = cv2.cvtColor(rgbImg.astype(np.uint8), cv2.COLOR_RGB2HLS)
    buf = buf.astype(float)
    buf *= [2, 1.0 / 255.0, 1.0 / 255.0]  # scale to 0..360, 0..1, 0..1
    return buf


def hls2rgbVec(hlsImg, cvRange=False):
    """
    With M = max(r,g,b) and m = min(r,g,b) the HLS color space uses
    luminosity L= (M+m)/2 and saturation S = (M-m)/(M+m) if l < 0.5 and
    S =  (M-m) / (2-(M+m)) otherwise.
    If cvRange is True the input array must follow the
    opencv conventions : ranges 0..180, 0..255, 0..255, otherwise ranges are 0..360, 0..1, 0..1

    :param hlsImg: hlsImg: hls image array
    :type  hlsImg: dtype = float
    :param cvRange:
    :type  cvRange:
    :return: identical shape array of r, g, b values in range 0..255
    :rtype: dtype = uint8
    """
    # scale to 0..180, 0..255, 0..255 (opencv convention)
    if not cvRange:
        buf = hlsImg * [1.0 / 2.0, 255.0, 255.0]
    else:
        buf = hlsImg
    # convert to rgb
    buf = cv2.cvtColor(buf.astype(np.uint8), cv2.COLOR_HLS2RGB)
    return buf


def hsv2rgb(h, s, v):
    """
    Transforms the hue, saturation, brightness h, s, v components of a color
    into red, green, blue values. (Cf. schema in file colors.docx)
    Note : here, brightness is v= max(r,g,b)/255.0. For perceptual brightness use
    hsp2rgb()

    :param h: float value in range 0..360
    :param s: float value in range 0..1
    :param v: float value in range 0..1
    :return: r,g,b integers between 0 and 255
    """
    h = h / 60.0
    i = np.floor(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0:  # r > g > b,  r=v and s=(r-b)/r, thus b = v(1-s)=p, h = (g-b)/(r-b) gives g=v(1-s(1-h))=t
        r1, g1, b1 = v, t, p
    elif i == 1:  # g > r > b
        r1, g1, b1 = q, v, p
    elif i == 2:  # g > b > r
        r1, g1, b1 = p, v, t
    elif i == 3:  # b > g >r
        r1, g1, b1 = p, q, v
    elif i == 4:  # b > r > g
        r1, g1, b1 = t, p, v
    else:  # r > b >g
        r1, g1, b1 = v, p, q
    r = int(r1 * 255.0)
    g = int(g1 * 255.0)
    b = int(b1 * 255.0)
    return r, g, b


def hsv2rgbVec(hsvImg, cvRange=False):
    """
    Vectorized version of hsv2rgb.
    Transforms the hue, saturation and brightness h, s, v components of a color
    into red, green, blue values. If cvRange is True the input array must follow the
    opencv conventions : ranges 0..180, 0..255, 0..255, otherwise ranges are 0..360, 0..1, 0..1

    :param hsvImg: hsv image array
    :type  hsvImg: ndarray dtype=np.float
    :param cvRange:
    :type  cvRange:
    :return: rgb image array range 0..255
    :rtype: ndarray dtype=np.uint8
    """
    flatten = (hsvImg.ndim > 3)
    if flatten:
        s = hsvImg.shape
        hsvImg = hsvImg.reshape(np.prod(s[:-1]), 1, s[-1])
    if not cvRange:
        # scale to 0..180, 0..255, 0..255 (opencv convention)
        hsvImg = hsvImg * [0.5, 255.0, 255.0]  # make sure a copy is done : do not use *=
    rgbImg = cv2.cvtColor(hsvImg.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if flatten:
        rgbImg = rgbImg.reshape(s)
    return rgbImg


def hsp2rgb(h, s, p):
    return hsp2rgb_ClippingInd(h, s, p)[:3]


def hsp2rgb_ClippingInd(h, s, p, trunc=True):
    """
    Transforms the hue, saturation and perceptual brightness components of a color
    into red, green, blue values.

    :param h: float value in range 0..360
    :param s: float value in range 0..1
    :param p: float value in range 0..1
    :param trunc:
    :type  trunc: boolean
    :return: r,g,b integers between 0 and 255
    """

    h = h / 60.0
    i = np.floor(h)
    f = h - i

    if s == 1.0:
        if h < 1.0:  # r > g > b=0
            r = np.sqrt(p * p / (Perc_R + Perc_G * f * f))
            g = r * f
            b = 0.0
        elif h < 2.0:  # g>r>b=0
            g = np.sqrt(p * p / (Perc_G + Perc_R * (1 - f) * (1 - f)))
            r = g * (1 - f)
            b = 0.0
        elif h < 3.0:  # g>b>r=0
            g = np.sqrt(p * p / (Perc_G + Perc_B * f * f))
            b = g * f
            r = 0.0
        elif h < 4.0:  # b>g>r=0
            b = np.sqrt(p * p / (Perc_B + Perc_G * (1 - f) * (1 - f)))
            g = b * (1 - f)
            r = 0.0
        elif h < 5.0:  # b>r>g=0
            b = np.sqrt(p * p / (Perc_B + Perc_R * f * f))
            r = b * f
            g = 0.0
        else:  # r>b>g=0
            r = np.sqrt(p * p / (Perc_R + Perc_B * (1 - f) * (1 - f)))
            b = r * (1 - f)
            g = 0.0
    else:  # s !=1
        Mm = 1.0 / (1.0 - s)  # Mm >= 1
        if h < 1.0:  # r > g > b
            part = 1.0 + f * (Mm - 1.0)  # part >=1 part = g/b
            b = p / np.sqrt(Perc_R * Mm * Mm + Perc_G * part * part + Perc_B)  # b<=p
            r = b * Mm
            g = b + f * (r - b)
        elif h < 2.0:  # g>r>b
            part = 1.0 + (1 - f) * (Mm - 1.0)  # part = r/b
            b = p / np.sqrt(Perc_G * Mm * Mm + Perc_R * part * part + Perc_B)
            g = b * Mm
            r = b + (1 - f) * (g - b)
        elif h < 3.0:  # g>b>r
            part = 1.0 + f * (Mm - 1.0)  # part = b/r
            r = p / np.sqrt(Perc_G * Mm * Mm + Perc_B * part * part + Perc_R)
            g = r * Mm
            b = r + f * (g - r)
        elif h < 4.0:  # b>g>r
            part = 1.0 + (1 - f) * (Mm - 1.0)
            r = p / np.sqrt(Perc_B * Mm * Mm + Perc_G * part * part + Perc_R)
            b = r * Mm
            g = r + (1 - f) * (b - r)
        elif h < 5.0:  # b>r>g
            part = 1.0 + f * (Mm - 1.0)
            g = p / np.sqrt(Perc_B * Mm * Mm + Perc_R * part * part + Perc_G)
            b = g * Mm
            r = g + f * (b - g)
        else:  # r>b>g
            part = 1.0 + (1 - f) * (Mm - 1.0)
            g = p / np.sqrt(Perc_R * Mm * Mm + Perc_B * part * part + Perc_G)
            r = g * Mm
            b = g + (1 - f) * (r - g)
    pc = Perc_R * r * r + Perc_G * g * g + Perc_B * b * b
    # sanity check
    assert abs(p * p - pc) <= 0.000001, 'colorCube.py:  hsp2rgb conversion error'
    M = max(r, g, b)
    if trunc:
        if M > 1:
            r, g, b = r / M, g / M, b / M
    return int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0)), (M > 1)


def hsp2rgbVec(hspImg):
    """
    Vectorized version of hsp2rgb.
    We first convert to HSV and next we use cv2.cvtColor()
    to convert from HSV to RGB

    :param hspImg: (n,m,3) array of H, S, pB values, range H:0..360, S:0..1, pB:0..1
    :type  hspImg: ndarray dtype=float
    :return: identical shape array of RGB values
    :rtype: ndarray dtype=float
    """
    h, s, p = hspImg[..., 0], hspImg[..., 1], hspImg[..., 2]

    shape = h.shape

    h = np.ravel(h)
    s = np.ravel(s)
    p = np.ravel(p)
    h /= 60.0
    i = np.floor(h).astype(int)
    f = h - i

    old_settings = np.seterr(all='ignore')
    Mm = 1.0 / (1.0 - s)
    Mm2 = Mm * Mm
    f2 = f * f
    oneMinusf = 1 - f
    oneMinusMm = 1.0 - Mm
    oneMinusf2 = oneMinusf * oneMinusf
    p2 = p * p
    bMask = (Mm == np.inf)

    part1 = (-f) * oneMinusMm
    part1 += 1.0  # 1.0 - f * oneMinusMm
    part1[bMask] = f[
        bMask]  # np.where(Mm == np.inf, f, part1)  s == 1 corresponds to Mm == np.inf. However this supposes s clipped to 1
    part2 = (-oneMinusf) * oneMinusMm
    part2 += 1.0  # 1.0 - oneMinusf * oneMinusMm
    part2[bMask] = oneMinusf[bMask]  # np.where(Mm == np.inf, oneMinusf, part2)

    ###################################################
    # Auxiliary functions for optimized computations of array expressions
    def pOverABplusCDplusE(A, B, C, D, E, P):
        """
        X = P / (A * B + C * D + E)
        """
        X = A * B
        X += C * D
        X += E
        X = np.reciprocal(X, out=X)
        X *= P
        return X

    def pOverAplusCD(A, C, D, P):
        """
        X = P / (A + C * D)
        """
        X = C * D
        X += A
        X = np.reciprocal(X, out=X)
        X *= P
        return X

    def region(A, B, C, D, E, F, P):
        """
        X = np.where(Mm == np.inf, 0, p2 / (Perc_R * Mm2 + Perc_G * part1 + Perc_B))
        Y = np.where(Mm == np.inf, p2 / (Perc_R + Perc_G * f2), X * Mm2)
        """
        X = pOverABplusCDplusE(A, B, C, D, E, P)
        X[bMask] = 0
        Y = pOverAplusCD(A, C, F, P)
        X *= B
        Y[~bMask] = X[~bMask]
        return Y

    ######################################################

    part1 *= part1
    part2 *= part2

    Y1 = region(Perc_R, Mm2, Perc_G, part1, Perc_B, f2, p2)
    Y2 = region(Perc_G, Mm2, Perc_R, part2, Perc_B, oneMinusf2, p2)
    Y3 = region(Perc_G, Mm2, Perc_B, part1, Perc_R, f2, p2)
    Y4 = region(Perc_B, Mm2, Perc_G, part2, Perc_R, oneMinusf2, p2)
    Y5 = region(Perc_B, Mm2, Perc_R, part1, Perc_G, f2, p2)
    Y6 = region(Perc_R, Mm2, Perc_B, part2, Perc_G, oneMinusf2, p2)

    np.seterr(**old_settings)
    # stack as rows
    clistMax = np.vstack((Y1, Y2, Y3, Y4, Y5, Y6))

    orderMax = np.arange(6)
    tmp = np.arange(np.prod(shape))

    rgbMax = clistMax[orderMax[i], tmp]
    np.sqrt(rgbMax, out=rgbMax)
    np.clip(rgbMax, 0, 1, out=rgbMax)
    hsv = np.dstack((h, s, rgbMax))
    hsv *= [30.0, 255.0, 255.0]
    rgb1 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb1.reshape(shape + (3,))


def hsp2rgbVecSmall(hspImg):
    """
    Vectorized version of hsp2rgb. Optimized for small images - currently unused -
    Very bad performances for big images : time = 11,11 s  and memory > 6 Go
    for a 15 Mpx image

    :param hspImg: (n,m,3) array of hsp values
    :return: identical shape array of rgb values
    """
    h, s, p = hspImg[..., 0], hspImg[..., 1], hspImg[..., 2]

    shape = h.shape
    # we compute the array rgbMax of brightness=max(r,g,b) values, and
    # we apply cvtColor(hsvImg, cv2.COLOR_HSV2RGB). This mixed approach
    # gives better performances for small images.
    h = np.ravel(h)
    s = np.ravel(s)
    p = np.ravel(p)
    h = h / 60.0
    i = np.floor(h).astype(int)
    f = h - i

    old_settings = np.seterr(all='ignore')
    Mm = 1.0 / (1.0 - s)
    Mm2 = Mm * Mm

    part1 = 1.0 + f * (Mm - 1.0)
    part1 = np.where(Mm == np.inf, f,
                     part1)  # s == 1 corresponds to Mm == np.inf. However this supposes s clipped to 1.
    part2 = 1.0 + (1 - f) * (Mm - 1.0)
    part2 = np.where(Mm == np.inf, 1 - f, part2)

    part1 = part1 * part1
    part2 = part2 * part2

    X1 = np.where(Mm == np.inf, 0, p / np.sqrt(Perc_R * Mm2 + Perc_G * part1 + Perc_B))  # b
    Y1 = np.where(Mm == np.inf, p / np.sqrt(Perc_R + Perc_G * f * f), X1 * Mm)  # r

    X2 = np.where(Mm == np.inf, 0, p / np.sqrt(Perc_G * Mm2 + Perc_R * part2 + Perc_B))  # b
    Y2 = np.where(Mm == np.inf, p / np.sqrt(Perc_G + Perc_R * (1 - f) * (1 - f)), X2 * Mm)  # g

    X3 = np.where(Mm == np.inf, 0, p / np.sqrt(Perc_G * Mm2 + Perc_B * part1 + Perc_R))  # r
    Y3 = np.where(Mm == np.inf, p / np.sqrt(Perc_G + Perc_B * f * f), X3 * Mm)  # g

    X4 = np.where(Mm == np.inf, 0, p / np.sqrt(Perc_B * Mm2 + Perc_G * part2 + Perc_R))  # r
    Y4 = np.where(Mm == np.inf, p / np.sqrt(Perc_B + Perc_G * (1 - f) * (1 - f)), X4 * Mm)  # b

    X5 = np.where(Mm == np.inf, 0, p / np.sqrt(Perc_B * Mm2 + Perc_R * part1 + Perc_G))  # g
    Y5 = np.where(Mm == np.inf, p / np.sqrt(Perc_B + Perc_R * f * f), X5 * Mm)  # b

    X6 = np.where(Mm == np.inf, 0, p / np.sqrt(Perc_R * Mm2 + Perc_B * part2 + Perc_G))  # g
    Y6 = np.where(Mm == np.inf, p / np.sqrt(Perc_R + Perc_B * (1 - f) * (1 - f)), X6 * Mm)  # r

    np.seterr(**old_settings)

    clistMax = np.vstack((Y1, Y2, Y3, Y4, Y5, Y6))
    orderMax = np.array([[0], [1], [2], [3], [4], [5]])

    tmp = np.arange(np.prod(shape))[:, None]

    rgbMax = clistMax[orderMax[i], tmp][:, 0]
    rgbMax = rgbMax.clip(0, 1)
    hsv = np.dstack((h * 60, s, rgbMax)).reshape(shape + (3,)) * [0.5, 255, 255]

    rgb1 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb1.reshape(shape + (3,))


def rgb2cmyk(r, g, b):
    """
    Convert r, g, b values in range 0..255 to
    CMYK colors as percents (range 0..100)

    :param r:
    :type  r: int
    :param g:
    :type  g: int
    :param b:
    :type  b: int
    :return: CMYK colors
    :rtype: 4 uple of ints
    """
    r, g, b = r / 255, g / 255, b / 255
    K = 1 - max(r, g, b)
    if K == 1:
        C, M, Y = (0,) * 3
    else:
        C = (1 - r - K) / (1 - K)
        M = (1 - g - K) / (1 - K)
        Y = (1 - b - K) / (1 - K)
    return int(C * 100), int(M * 100), int(Y * 100), int(K * 100)


def cmyk2rgb(c, m, y, k):
    """
    Convert CMYK values in range 0..100 to RGB colors in range 0..255

    :param c:
    :type  c: int
    :param m:
    :type  m: int
    :param y:
    :type  y: int
    :param k:
    :type  k: int
    :return:
    :rtype: 3-uple of int
    """
    c, m, y, k = c / 100.0, m / 100.0, y / 100.0, k / 100.0
    k = 1 - k
    if k == 0:
        return (0,) * 3
    else:
        return int(255 * (1 - c) * k), int(255 * (1 - m) * k), int(255 * (1 - y) * k)
