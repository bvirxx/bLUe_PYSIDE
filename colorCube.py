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
from functools import partial

import cv2
import gc
import numpy as np
from PySide2.QtCore import QFile, QIODevice, QTextStream
from PySide2.QtGui import QImage

from cartesian import cartesianProduct
from imgconvert import QImageBuffer

###########################################
# 3D LUT constants
# LUTSIZE is the length of LUT axes.
# Interpolation range is 0 <= x < MAXLUTRGB
###########################################

MAXLUTRGB = 256
# LUTSIZE = 17
LUTSIZE = 33
LUTSTEP = MAXLUTRGB / (LUTSIZE - 1)

######################################
# Weights for perceptual brightness
#######################################
# usual values
"""
Perc_R=0.2126
Perc_G=0.7152
Perc_B=0.0722
"""
# interpolated values
Perc_R=0.2338
Perc_G=0.6880
Perc_B=0.0782

class LUT3D (object):
    """
    Implement 3D LUT. Size should be be s=2**n + 1,
    array shape is (s, s, s, 3) and array values
    are positive integers in range 0..255
    """
    def __init__(self, LUT3DArray, size=LUTSIZE):
        # sanity check
        if ((size - 1) & (size - 2)) != 0 :
            raise ValueError("LUT3D : size should be 2**n+1, found %d" % size)
        self.LUT3DArray = LUT3DArray
        self.size = size
        self.contrast = lambda p : p #np.power(p,1.2)
        self.step = MAXLUTRGB // (size - 1)  # python 3 : integer quotient

    @classmethod
    def LUT3DFromFactory(cls, size=LUTSIZE):
        """
        Init a LUT3D object with shape (size, size,size, 3).
        size should be 2**n +1. Most common values are 17 and 33.
        The 4th axis holds 3-uples (r,g,b) of integers evenly
        distributed in the range 0..MAXLUTRGB (edges included) :
        with step = MAXLUTRGB / (size - 1), we have
        LUT3DArray(i, j, k) = (i*step, j*step, k*step).
        Note that, with that initial LUT array, trilinear interpolation boils
        down to identity : for all i,j,k in the range 0..255,
        trilinear(i//step, j//step, k//step) = (i,j,k).
        @param size: integer value (should be 2**n+1)
        @return: 3D LUT table
        @rtype: LUT3D object shape (size, size, size, 3), dtype=int
        """
        step = MAXLUTRGB / (size - 1)
        a = np.arange(size, dtype=np.float) * step  # TODO 09/04/18 validate dtype
        c = cartesianProduct((a, a, a))
        # clip for the sake of consistency with Hald images
        c = np.clip(c, 0, 255)
        return LUT3D(c, size=size)

    @classmethod
    def HaldImage2LUT3D(cls, hald, size=LUTSIZE):
        """
        Converts a hald image to a LUT3D object.
        @param hald: image
        @type hald: QImage
        @param size: LUT size
        @type size: int
        @return: 3D LUT
        @rtype: LUT3D object
        """
        buf = QImageBuffer(hald)
        buf = buf[:,:,:3].ravel()
        count = (size ** 3) * 3 # ((size - 1) ** 3) * 3
        buf = buf[:count]
        buf = buf.reshape((size, size, size, 3))
        LUT = np.zeros((size, size, size, 3), dtype=int) + 255
        LUT[:,:,:,:] = buf[:,:,:,::-1]
        return LUT3D(LUT, size=size)

    @classmethod
    def HaldBuffer2LUT3D(cls, haldBuff, size=LUTSIZE):
        buf = haldBuff
        buf = buf[:, :, :3].ravel()
        count = (size ** 3) * 3  # ((size - 1) ** 3) * 3
        buf = buf[:count]
        buf = buf.reshape((size, size, size, 3))
        LUT = np.zeros((size, size, size, 3), dtype=int) + 255
        LUT[:, :, :, :] = buf[:, :, :, ::-1]
        return LUT3D(LUT, size=size)

    def getHaldImage(self, w, h):
        """
        Converts a LUT3D object to an image buffer (numpy array). 
        The (self.size-1)**3 *3 first bytes of the buffer are
        
        [ LUT3DArray[r,g,b][::-1] for r in range(self.size-1) for g in range(self.size-1) for b in range(self.size-1) ]
        
       The remainings bytes are padded with 0.
       Example: LUT3DIdentity.getHaldImage(w, h) returns an identity hald image of size (w,h) padded with 0.
       w*h must be greater than LUTSIZE^^3
       
        Note that the buffer is in BGR order, suitable for direct conversion to a QImage and that 
        the b value increases fastest (i.e. from a byte to the next). 
        @param w: image width
        @type w: int
        @param h: image height
        @type h: int
        @return: numpy array shape=(h,w,3), dtype=np.uint8
        """
        buf = np.zeros((w*h*3), dtype = np.uint8) # + 255
        s = self.size# - 1
        count = (s**3) *3 # TODO clip LUT array to 0,255 ?
        buf[:count] = self.LUT3DArray[:,:,:,::-1].ravel() #self.LUT3DArray[:s,:s,:s,::-1].ravel()
        buf = buf.reshape(h, w, 3)
        return buf

    @classmethod
    def readFromTextStream(cls, inStream):
        #header
        for i in range(2):
            line = inStream.readLine()
        # get LUT size (second line format should be : Size xx)
        token = line.split()
        if len(token) >= 2:
            _, size = token
        else:
            raise ValueError('Cannot find LUT size')
        size = int(size)
        bufsize = (size**3)*3
        buf = np.zeros(bufsize, dtype=float)
        i = 0
        while True:
            line = inStream.readLine()
            if len(line) == 0:
                break
            token = line.split()
            if len(token) >= 3:
                a, b, c = token
            else:
                raise ValueError('Wrong file format')
            buf[i], buf[i+1], buf[i+2] = float(a), float(b), float(c)
            i+=3
        if i != bufsize:
           raise ValueError('LUT size does not match line count')
        buf = buf.reshape(size,size,size,3)
        return buf, size

    @classmethod
    def readFromTextFile(cls, filename):
        """
        Reads 3D LUT from text file in format .cube.
        Values are multiplied by 255.
        @param filename: 
        @type filename: str
        @return: 
        """
        qf = QFile(filename)
        if not qf.open(QIODevice.ReadOnly):
            raise IOError('cannot open file %s' % filename)
        textStream = QTextStream(qf)
        LUT3DArray, size = cls.readFromTextStream(textStream)
        qf.close()
        LUT3DArray = LUT3DArray *255.0
        LUT3DArray = LUT3DArray.astype(int)
        LUT3DArray = LUT3DArray.transpose(2,1,0, 3)
        return LUT3DArray, size

    @classmethod
    def readFromHaldFile(cls, filename):
        img = QImage(filename)
        buf = QImageBuffer(img)[:,:,:3][:,:,::-1]
        LUT = LUT3D.LUT3DFromFactory()
        s = LUT.size
        buf = buf.reshape(s-1, s-1,s-1,3)
        LUT.LUT3DArray[:s-1,:s-1,:s-1] = buf
        return LUT.LUT3DArray

    def writeToTextStream(self, outStream):
        """
        Writes 3D LUT to QTextStream in format .cube.
        Values are divided by 255.
        @param outStream: 
        @type outStream: QTextStream
        @return: 
        """
        LUT=self.LUT3DArray
        outStream << ('bLUe 3D LUT')<<'\n'
        outStream << ('Size %d' % self.size)<<'\n'
        coeff = 255.0
        for b in range(self.size):
            for g in range(self.size):
                for r in range(self.size):
                    r1, g1, b1 = LUT[r, g, b]
                    outStream << ("%.7f %.7f %.7f" % (r1 / coeff, g1 / coeff, b1 / coeff)) << '\n'

    def writeToTextFile(self, filename):
        qf = QFile(filename)
        if not qf.open(QIODevice.WriteOnly):
            raise IOError('cannot open file %s' % filename)
        textStream = QTextStream(qf)
        self.writeToTextStream(textStream)
        qf.close()


LUT3DIdentity = LUT3D.LUT3DFromFactory()
LUT3D_ORI = LUT3DIdentity.LUT3DArray
a,b,c,d = LUT3D_ORI.shape
LUT3D_SHADOW = np.zeros((a,b,c,d+1))
LUT3D_SHADOW[:,:,:,:3] = LUT3D_ORI

def rgb2hsp(r, g, b):
    return rgb2hsB(r,g,b, perceptual=True)

def rgb2hspVec(rgbImg):
    return rgb2hsBVec(rgbImg, perceptual=True)

def rgb2hsB(r, g, b, perceptual=False):
    """
    transforms the red, green ,blue r, g, b components of a color
    into hue, saturation, brightness h, s, v. (Cf. schema in file colors.docx)
    The r, g, b components are integers in range 0..255. If perceptual is False
    (default) v = max(r,g,b)/255.0, else v = sqrt(Perc_R*r*r + Perc_G*g*g + Perc_B*b*b)
    @param r:
    @param g:
    @param b:
    @param perceptual:
    @type perceptual: boolean
    @return: h, s, v values : 0<=h<360, 0<=s<=1, 0<=v<=1
    @rtype: float
    """
    cMax = max(r, g, b)
    cMin = min(r, g, b)
    delta = cMax - cMin
    # hue
    if delta == 0:
        H = 0.0
    elif cMax == r:
        H = 60.0 * float(g-b)/delta if g >= b else 360 + 60.0 * float(g-b)/delta
    elif cMax == g:
        H = 60.0 * (2.0 + float(b-r)/delta)
    elif cMax == b:
        H = 60.0 * (4.0 + float(r-g)/delta)
    # saturation
    S = 0.0 if cMax == 0.0 else float(delta) / cMax
    # brightness
    if perceptual:
        V = np.sqrt(Perc_R * r * r + Perc_G * g * g + Perc_B * b * b)
    else:
        V = cMax
    V = V / 255.0
    assert 0<=H and H<=360 and 0<=S and S<=1 and 0<=V and V<=1, "rgb2hsv conversion error r=%d, g=%d, b=%d, h=%f, s=%f, v=%f" %(r,g,b,H,S,V)
    return H,S,V

def rgb2hsBVec(rgbImg, perceptual=False):
    """
    Vectorized version of rgb2hsB.
    RGB-->HSV color space transformation
    The r, g, b components are integers in range 0..255. If perceptual is False
    (default) V = max(r,g,b)/255.0, else V = sqrt(Perc_R*r*r + Perc_G*g*g + Perc_B*b*b) / 255
    @param rgbImg: RGB image range 0..255
    @type rgbImg: (n,m,3) array, , dtype=uint8 or dtype=int or dtype=float
    @return: identical shape array of hue,sat,brightness values (0<=h<=360, 0<=s<=1, 0<=v<=1)
    @rtype: (n,m,3) array, dtype=float
    """
    buf = cv2.cvtColor(rgbImg.astype(np.uint8), cv2.COLOR_RGB2HSV)
    buf = buf.astype(np.float) * [2, 1.0 / 255.0, 1.0 / 255.0]  # scale to 0..360, 0..1, 0..1
    if perceptual:
        rgbImg2 = rgbImg.astype(float) * rgbImg
        pB = np.tensordot(rgbImg2, [Perc_R, Perc_G, Perc_B] , axes=(-1,-1)) / (255.0*255)
        pB = np.sqrt(pB)
        buf[:,:,2] = pB
    return buf

def rgb2hlsVec(rgbImg):
    """
    Converts RGB color space to HLS.
    With M = max(r,g,b) and m = min(r,g,b) the HLS color space uses
    luminosity L= (M+m)/2 and saturation S = (M-m)/(M+m) if l < 0.5 and
    S =  (M-m) / (2-(M+m)) otherwise, (0<=h<=360, 0<=l<=1, 0<=s<=1).
    We do not follow the opencv convention for HLS value ranges.
    @param rgbImg: rgbImg: array of r,g, b values
    @type rgbImg: rgbImg: (n,m,3) array, , dtype=uint8 or dtype=int or dtype=float
    @return: identical shape array of hue,luma, chroma values (0<=h<=360, 0<=l<=1, 0<=s<=1)
    @rtype: (n,m,3) array, dtype=float
    """
    buf = cv2.cvtColor(rgbImg.astype(np.uint8), cv2.COLOR_RGB2HLS)
    buf = buf.astype(np.float) * [2, 1.0 / 255.0, 1.0 / 255.0]  # scale to 0..360, 0..1, 0..1
    return buf

def hls2rgbVec(hlsImg):
    """
    With M = max(r,g,b) and m = min(r,g,b) the HLS color space uses
    luminosity L= (M+m)/2 and saturation S = (M-m)/(M+m) if l < 0.5 and
    S =  (M-m) / (2-(M+m)) otherwise.
    @param hlsImg: hlsImg: hls image array range 0..360, 0..1, 0..1
    @type hlsImg: dtype = float
    @return: identical shape array of r, g, b values in range 0..255
    @rtype: dtype = uint8
    """
    # scale to 0..360/2, 0..255, 0..255 (opencv convention)
    buf = hlsImg * [1.0 / 2.0, 255.0, 255.0]
    # convert to rgb
    buf = cv2.cvtColor(buf.astype(np.uint8), cv2.COLOR_HLS2RGB)
    return buf

def hsv2rgb(h,s,v):
    """
    Transforms the hue, saturation, brightness h, s, v components of a color
    into red, green, blue values. (Cf. schema in file colors.docx)
    Note : here, brightness is v= max(r,g,b)/255.0. For perceptual brightness use
    hsp2rgb()

    @param h: float value in range 0..360
    @param s: float value in range 0..1
    @param v: float value in range 0..1
    @return: r,g,b integers between 0 and 255
    """
    h = h/60.0
    i = np.floor(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * ( 1.0 - s * ( 1.0 - f))
    if  i == 0 : # r > g > b,  r=v and s=(r-b)/r, thus b = v(1-s)=p, h = (g-b)/(r-b) gives g=v(1-s(1-h))=t
        r1, g1, b1 = v, t, p
    elif i == 1: # g > r > b
        r1, g1, b1 = q, v, p
    elif i == 2: # g > b > r
        r1, g1, b1 = p, v, t
    elif i == 3: # b > g >r
        r1, g1, b1 = p, q, v
    elif i == 4: # b > r > g
        r1, g1, b1 = t, p, v
    else : # r > b >g
        r1, g1, b1 = v, p, q
    r = int(r1 * 255.0)
    g = int(g1 * 255.0)
    b = int(b1 * 255.0)
    return r,g,b

def hsv2rgbVec(hsvImg):
    """
    Vectorized version of hsv2rgb.
    Transform the hue, saturation and brightness h, s, v components of a color
    into red, green, blue values.
    @param hsvImg: hsv image array range 0..360, 0..1, 0..1
    @return: rgb image array
    """
    flatten = (hsvImg.ndim > 3)
    if flatten :
        s = hsvImg.shape
        hsvImg = hsvImg.reshape(np.prod(s[:-1]), 1, s[-1])
    hsvImg = hsvImg * [1.0 / 2.0, 255.0, 255.0]  # scale to 0..360/2, 0..255, 0..255 (opencv convention)
    rgbImg = cv2.cvtColor(hsvImg.astype(np.uint8), cv2.COLOR_HSV2RGB )
    if flatten:
        rgbImg = rgbImg.reshape(s)
    return rgbImg

def hsp2rgb(h, s, p):
    return hsp2rgb_ClippingInd(h,s,p)[:3]

def hsp2rgb_ClippingInd(h,s,p, trunc=True):
    """
    Transform the hue, saturation and perceptual brightness components of a color
    into red, green, blue values.
    @param h: float value in range 0..360
    @param s: float value in range 0..1
    @param p: float value in range 0..1
    @return: r,g,b integers between 0 and 255
    """

    h = h /60.0
    i = np.floor(h)
    f = h - i

    if s == 1.0:
        if h < 1.0 :  # r > g > b=0
            r = np.sqrt(p * p / (Perc_R + Perc_G * f * f))
            g = r * f
            b = 0.0
        elif h < 2.0:  # g>r>b=0
            g = np.sqrt(p * p / (Perc_G + Perc_R * (1-f) * (1-f)))
            r = g * (1-f)
            b = 0.0
        elif h < 3.0: # g>b>r=0
            g = np.sqrt(p * p / (Perc_G + Perc_B * f * f))
            b = g * f
            r = 0.0
        elif h < 4.0 : # b>g>r=0
            b = np.sqrt(p * p / (Perc_B + Perc_G * (1-f) * (1-f)))
            g = b * (1-f)
            r = 0.0
        elif h < 5.0 :  # b>r>g=0
            b = np.sqrt(p * p / (Perc_B + Perc_R * f * f))
            r = b * f
            g = 0.0
        else : # r>b>g=0
            r = np.sqrt(p * p / (Perc_R + Perc_B * (1-f) * (1-f)))
            b = r * (1-f)
            g = 0.0
    else:  # s !=1
        Mm = 1.0 / (1.0 - s)  #Mm >= 1
        if h < 1.0 :  # r > g > b
            part = 1.0 + f * (Mm - 1.0)  # part >=1 part = g/b
            b = p / np.sqrt(Perc_R * Mm * Mm + Perc_G * part * part + Perc_B)  # b<=p
            r = b * Mm
            g = b + f * (r - b)
        elif h < 2.0: #g>r>b
            part = 1.0 + (1-f) * (Mm - 1.0) #part = r/b
            b = p / np.sqrt(Perc_G * Mm * Mm + Perc_R * part * part + Perc_B)
            g = b * Mm
            r = b + (1-f) * (g - b)
        elif h < 3.0: # g>b>r
            part = 1.0 + f * (Mm - 1.0) # part = b/r
            r = p / np.sqrt(Perc_G * Mm * Mm + Perc_B * part * part + Perc_R)
            g = r * Mm
            b = r + f * (g - r)
        elif h < 4.0: # b>g>r
            part = 1.0 + (1-f) * (Mm - 1.0)
            r = p / np.sqrt(Perc_B * Mm * Mm + Perc_G * part * part + Perc_R)
            b = r * Mm
            g = r  + (1-f) * (b - r)
        elif h < 5.0: # b>r>g
            part = 1.0 + f * (Mm - 1.0)
            g = p / np.sqrt(Perc_B * Mm * Mm + Perc_R * part * part + Perc_G)
            b = g * Mm
            r = g + f * (b - g)
        else: # r>b>g
            part = 1.0 + (1-f) * (Mm - 1.0)
            g = p / np.sqrt(Perc_R * Mm * Mm + Perc_B * part * part + Perc_G)
            r = g * Mm
            b = g + (1-f) * (r - g)
    pc= Perc_R * r * r + Perc_G * g * g + Perc_B * b * b
    # sanity check
    assert abs(p*p - pc)<= 0.000001, 'colorCube.py:  hsp2rgb conversion error'
    M = max(r, g, b)
    if trunc:
        if M > 1:
            r, g, b = r / M, g / M, b / M
    return int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0)), (M > 1)

def hsp2rgbVec(hspImg):
    """
    Vectorized version of hsp2rgb.
    We convert to HSV and we use cv2.cvtColor()
    to convert from HSV to RGB
    @param hspImg: (n,m,3) array of hsp values
    @return: identical shape array of rgb values
    """
    """
    # use faster version for small images
    if hspImg.shape[0] * hspImg.shape[1] < 1000:
        return hsp2rgbVecSmall(hspImg)
    """
    h, s, p = hspImg[..., 0], hspImg[..., 1], hspImg[..., 2]

    shape = h.shape

    h = np.ravel(h)
    s = np.ravel(s)
    p = np.ravel(p)
    h = h / 60.0
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

    part1 = 1.0 - f * oneMinusMm
    part1 = np.where(Mm == np.inf, f, part1)  # TODO some invalid values remain for s = 1
    part2 = 1.0 - oneMinusf * oneMinusMm
    part2 = np.where(Mm == np.inf, oneMinusf, part2)

    part1 = part1 * part1
    part2 = part2 * part2


    #X1 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_R * Mm2 + Perc_G * part1 + Perc_B))   # b
    X1 = np.where(Mm == np.inf, 0, p2 / (Perc_R * Mm2 + Perc_G * part1 + Perc_B))  # b
    #Y1 = np.where(Mm==np.inf, p / np.sqrt(Perc_R+ Perc_G * f *f), X1 * Mm)              # r
    Y1 = np.where(Mm == np.inf, p2 / (Perc_R + Perc_G * f2), X1 * Mm2)  # r
    X1=None
    #Z1 = np.where(Mm==np.inf, Y1 * f, X1 + f * (Y1 - X1))                               # g

    #X2 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_G * Mm2 + Perc_R * part2 + Perc_B))   # b
    X2 = np.where(Mm == np.inf, 0, p2 / (Perc_G * Mm2 + Perc_R * part2 + Perc_B))  # b
    #Y2 = np.where(Mm==np.inf, p / np.sqrt(Perc_G + Perc_R * (1-f) * (1-f)), X2 * Mm)    # g
    Y2 = np.where(Mm == np.inf, p2 / (Perc_G + Perc_R * oneMinusf2), X2 * Mm2)  # g
    X2=None
    #Z2 = np.where(Mm==np.inf, Y2 * (1-f), X2 + (1 - f) * (Y2 - X2))                     # r

    #X3 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_G * Mm2 + Perc_B * part1 + Perc_R))   # r
    X3 = np.where(Mm == np.inf, 0, p2 / (Perc_G * Mm2 + Perc_B * part1 + Perc_R))  # r
    #Y3 = np.where(Mm==np.inf, p / np.sqrt(Perc_G + Perc_B * f * f), X3 * Mm)            # g
    Y3 = np.where(Mm == np.inf, p2 / (Perc_G + Perc_B * f2), X3 * Mm2)  # g
    X3=None
    #Z3 = np.where(Mm==np.inf, Y3 * f, X3 + f * (Y3 - X3))                               # b

    gc.collect()
    #X4 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_B * Mm2 + Perc_G * part2 + Perc_R))   # r
    X4 = np.where(Mm == np.inf, 0, p2 / (Perc_B * Mm2 + Perc_G * part2 + Perc_R))  # r
    #Y4 = np.where(Mm==np.inf, p / np.sqrt(Perc_B + Perc_G * (1-f) * (1-f)), X4 * Mm)    # b
    Y4 = np.where(Mm == np.inf, p2 / (Perc_B + Perc_G * oneMinusf2), X4 * Mm2)  # b
    X4=None
    #Z4 = np.where(Mm==np.inf, Y4 * (1 - f), X4 + (1 - f) * (Y4 - X4))                   # g

    #X5 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_B * Mm2 + Perc_R * part1 + Perc_G))   # g
    X5 = np.where(Mm == np.inf, 0, p2 / (Perc_B * Mm2 + Perc_R * part1 + Perc_G))  # g
    #Y5 = np.where(Mm==np.inf, p / np.sqrt(Perc_B + Perc_R * f * f), X5 * Mm)            # b
    Y5 = np.where(Mm == np.inf, p2 / (Perc_B + Perc_R * f2), X5 * Mm2)  # b
    X5=None
    #Z5 = np.where(Mm==np.inf, Y5 * f, X5 + f * (Y5 - X5))                               # r

    #X6 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_R * Mm2 + Perc_B * part2 + Perc_G))   # g
    X6 = np.where(Mm == np.inf, 0, p2 / (Perc_R * Mm2 + Perc_B * part2 + Perc_G))  # g
    #Y6 = np.where(Mm==np.inf, p / np.sqrt(Perc_R + Perc_B * (1-f) * (1-f)), X6 * Mm)    # r
    Y6 = np.where(Mm == np.inf, p2 / (Perc_R + Perc_B * oneMinusf2), X6 * Mm2)  # r
    X6=None
    #Z6 = np.where(Mm==np.inf, Y6 * (1 - f), X6 + (1 - f) * (Y6 - X6))                   # b

    np.seterr(**old_settings)
    # stacked as lines
    clistMax = np.vstack((Y1, Y2, Y3, Y4, Y5,  Y6))

    orderMax = np.array([[0],[1],[2],[3],[4],[5]])

    tmp = np.arange(np.prod(shape))[:, None]

    rgbMax = clistMax[orderMax[i], tmp][:,0]
    rgbMax = np.sqrt(rgbMax)
    rgbMax=rgbMax.clip(0,1)
    #hsv = np.dstack((h*60,s,rgbMax)).reshape(shape + (3,)) * [0.5, 255, 255]
    hsv = np.dstack((h * 60, s, rgbMax)) * [0.5, 255, 255]
    rgb1=cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return rgb1.reshape(shape + (3,))

    """
    # advanced array indexing
    rgb = clist[order[i], tmp]  # order[i] has shape (w*h, 3), tmp is broadcast to the same shape, thus rgb has (shape w*h, 3)

    # converting invalid values to int gives indeterminate results
    rgb[np.isnan(rgb)] = 0.0  # TODO np.inf

    # for uint8 image buffer, int values > 255 are truncated to lower 8 bits
    # rgbMask = (rgb > 1)
    clipped = np.amax(rgb, axis=1)
    clipped = np.dstack((clipped, clipped, clipped))
    rgb = np.where(clipped > 1, rgb / clipped, rgb)
    # rgb = np.clip(rgb, 0, 1.0)

    rgb = (rgb * 255.0).astype(int)

    return rgb.reshape(shape + (3,))
    """

def hsp2rgbVecSmall(hspImg):
    """
    Vectorized version of hsp2rgb. Optimized for small images.
    Very bad performances for big images : time = 11,11 s  and memory > 6 Go
    for a 15 Mpx image
    @param hspImg: (n,m,3) array of hsp values
    @return: identical shape array of rgb values
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
    part1 = np.where(Mm == np.inf, f, part1)  # TODO some invalid values remain for s = 1
    part2 = 1.0 + (1 - f) * (Mm - 1.0)
    part2 = np.where(Mm == np.inf, 1 - f, part2)

    part1 = part1 * part1
    part2 = part2 * part2


    X1 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_R * Mm2 + Perc_G * part1 + Perc_B))   # b
    Y1 = np.where(Mm==np.inf, p / np.sqrt(Perc_R+ Perc_G * f *f), X1 * Mm)              # r
    #Z1 = np.where(Mm==np.inf, Y1 * f, X1 + f * (Y1 - X1))                               # g  # TODO 12/04/18 validate removal of Zi, clist, order 

    X2 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_G * Mm2 + Perc_R * part2 + Perc_B))   # b
    Y2 = np.where(Mm==np.inf, p / np.sqrt(Perc_G + Perc_R * (1-f) * (1-f)), X2 * Mm)    # g
    #Z2 = np.where(Mm==np.inf, Y2 * (1-f), X2 + (1 - f) * (Y2 - X2))                     # r

    X3 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_G * Mm2 + Perc_B * part1 + Perc_R))   # r
    Y3 = np.where(Mm==np.inf, p / np.sqrt(Perc_G + Perc_B * f * f), X3 * Mm)            # g
    #Z3 = np.where(Mm==np.inf, Y3 * f, X3 + f * (Y3 - X3))                               # b

    X4 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_B * Mm2 + Perc_G * part2 + Perc_R))   # r
    Y4 = np.where(Mm==np.inf, p / np.sqrt(Perc_B + Perc_G * (1-f) * (1-f)), X4 * Mm)    # b
    #Z4 = np.where(Mm==np.inf, Y4 * (1 - f), X4 + (1 - f) * (Y4 - X4))                   # g

    X5 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_B * Mm2 + Perc_R * part1 + Perc_G))   # g
    Y5 = np.where(Mm==np.inf, p / np.sqrt(Perc_B + Perc_R * f * f), X5 * Mm)            # b
    #Z5 = np.where(Mm==np.inf, Y5 * f, X5 + f * (Y5 - X5))                               # r

    X6 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_R * Mm2 + Perc_B * part2 + Perc_G))   # g
    Y6 = np.where(Mm==np.inf, p / np.sqrt(Perc_R + Perc_B * (1-f) * (1-f)), X6 * Mm)    # r
    #Z6 = np.where(Mm==np.inf, Y6 * (1 - f), X6 + (1 - f) * (Y6 - X6))                   # b

    np.seterr(**old_settings)

    # stack as lines
    #clist = np.vstack((X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4, X5, Y5, Z5, X6, Y6, Z6))

    clistMax = np.vstack((Y1, Y2, Y3, Y4, Y5,  Y6))
    orderMax = np.array([[0],[1],[2],[3],[4],[5]])

    # for hue slices 0,..,5, the corresponding 3-uple gives the line indices in clist for the r,g,b values
    #order = np.array([[1, 2, 0], [5, 4, 3], [6, 7, 8], [9, 11, 10], [14, 12, 13], [16, 15, 17]])

    tmp = np.arange(np.prod(shape))[:, None]

    rgbMax = clistMax[orderMax[i], tmp][:,0]
    rgbMax = rgbMax.clip(0,1)
    hsv = np.dstack((h*60,s,rgbMax)).reshape(shape + (3,)) * [0.5, 255, 255]

    rgb1=cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb1.reshape(shape + (3,))


def interpMulti(LUT, ndImg, pool=None):
    """
    parallel version of trilinear interpolation.
    Should be compared to the vectorized version interpVec_
    @param LUT:
    @param ndImg:
    @return:
    """
    w, h = ndImg.shape[1], ndImg.shape[0]
    SLF = 4
    sl_w = [slice((w * i) // SLF, (w * (i+1)) // SLF) for i in range(SLF)]  # python 3 // for integer quotient
    sl_h = [slice((h * i) // SLF, (h * (i + 1)) // SLF) for i in range(SLF)]

    slices = [ (s1, s2) for s1 in sl_w for s2 in sl_h]
    imgList = [ndImg[s2, s1] for s1, s2 in slices]
    if pool is None:
        raise ValueError('interpVec: no processing pool')
    # get vectorized interpolation as partial function
    partial_f = partial(interpVec_, LUT)
    # parallel interpolation
    res = pool.map(partial_f, imgList)
    outImg = np.empty(ndImg.shape)
    # collect results
    for i, (s1, s2) in enumerate(slices):
            outImg[s2, s1] = res[i]
    return outImg

def interpVec_(LUT, ndImg, pool=None):
    """
    Vectorized version of trilinear interpolation
    Cf. file trilinear.docx for details
    Convert an RGB image using a 3D LUT. The output image is interpolated from the LUT.
    It has the same dimensions and type as the input image.
    @param LUT: 3D LUT array
    @param ndImg: image array
    @return: RGB image with same dimensions as the input image
    """
    # bounding unit cube coordinates for (r, g, b)/LUTSTEP
    ndImgF = ndImg / float(LUTSTEP)
    #a=np.floor(ndImgF).astype(int)  TODO validate modif 09/04/18
    a = ndImgF.astype(int)
    aplus1 = a + 1


    # RGB channels
    r0, g0, b0 = a[:,:,0], a[:,:,1], a[:,:,2]
    r1, g1, b1 = aplus1[:,:,0], aplus1[:,:,1], aplus1[:,:,2]

    ndImg00 = LUT[r0, g0, b0]
    ndImg01 = LUT[r1, g0, b0]
    ndImg02 = LUT[r0, g1, b0]
    ndImg03 = LUT[r1, g1, b0]

    ndImg10 = LUT[r0, g0, b1]
    ndImg11 = LUT[r1, g0, b1]
    ndImg12 = LUT[r0, g1, b1]
    ndImg13 = LUT[r1, g1, b1]

    # interpolate
    alpha =  ndImgF[:,:,1] - g0
    alpha=np.dstack((alpha, alpha, alpha))

    I11Value = ndImg11 + alpha * (ndImg13 - ndImg11)  #oneMinusAlpha * ndImg11 + alpha * ndImg13
    I12Value = ndImg10 + alpha * (ndImg12 - ndImg10)  #oneMinusAlpha * ndImg10 + alpha * ndImg12
    I21Value = ndImg01 + alpha * (ndImg03 - ndImg01)  #oneMinusAlpha * ndImg01 + alpha * ndImg03
    I22Value = ndImg00 + alpha * (ndImg02 - ndImg00)  # oneMinusAlpha * ndImg00 + alpha * ndImg02

    beta = ndImgF[:,:,0] - r0
    beta = np.dstack((beta, beta, beta))

    I1Value = I12Value + beta * (I11Value - I12Value) #oneMinusBeta * I12Value + beta * I11Value
    I2Value = I22Value + beta * (I21Value - I22Value) #oneMinusBeta * I22Value + beta * I21Value

    gamma = ndImgF[:,:,2] - b0
    gamma = np.dstack((gamma, gamma, gamma))

    IValue =  I2Value + gamma * (I1Value - I2Value)  #(1 - gamma) * I2Value + gamma * I1Value

    return IValue


if __name__=='__main__':
    # random ints in range 0 <= x < 256
    b = np.random.randint(0,256, size=500*500*3, dtype=np.uint8)
    testImg = np.reshape(b, (500,500,3))
    interpImg = LUT3D.LUT3DFromFactory(size=33)
    interpImg = interpMulti(LUT3D_ORI, testImg)
    d = testImg - interpImg
    if (d != 0.0).any():
        pass
        #print "interpolation error"






