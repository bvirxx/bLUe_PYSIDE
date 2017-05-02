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
from PySide.QtCore import QDataStream, QFile, QIODevice, QTextStream

from cartesian import cartesianProduct


# 3D LUT init.

"""
Each axis of the LUT has length LUTSIZE.
r,g,b values are between 0 and 256.
""";
LUTSIZE = 17
LUTSIZE = 33
LUTSTEP = 256 / (LUTSIZE - 1)
a = np.arange(LUTSIZE)
LUT3D_ORI = cartesianProduct((a,a,a)) * LUTSTEP
a,b,c,d = LUT3D_ORI.shape
LUT3D_SHADOW = np.zeros((a,b,c,d+1))
LUT3D_SHADOW[:,:,:,:3] = LUT3D_ORI

"""
v = QVector3D(1.0,1.0,1.0) * 256.0 / 3.0
LUTPROJ = [QVector3D(i/(i+j+k),j/(i+j+k),k/(i+j+k))*256 - v  for i in range(LUTSIZE) for j in range(LUTSIZE) for k in range(LUTSIZE) if gcd(i,gcd(j,k))==1]
# LUTPROJ +v gives r,g,b colors
orig_theta = QVector3D(0.33 - 1, 0.33,0.33).normalized()
orth = QVector3D(0, 1.0, -1.0).normalized()
LUTPROJ_x = [ int(QVector3D.dotProduct(V, orig_theta)) for V in LUTPROJ]
LUTPROj_y = [ int(QVector3D.dotProduct(V, orth)) for V in LUTPROJ]

LUTPROJTUPLE = [(i*LUTSTEP,j*LUTSTEP,k*LUTSTEP)  for i in range(LUTSIZE) for j in range(LUTSIZE) for k in range(LUTSIZE) if gcd(i,gcd(j,k))==1]
LUTPROJSET = set(LUTPROJ)
"""
# perceptual brightness constants

Perc_R = 0.299
Perc_G = 0.587
Perc_B = 0.114
"""
Perc_R=0.2126
Perc_G=0.7152
Perc_B=0.0722
"""
# interpolated values
Perc_R=0.2338
Perc_G=0.6880
Perc_B=0.0782

"""
Perc_R=0.79134178
Perc_G=2.31839104
Perc_B=0.25510923
"""
"""
Perc_R=1.0/3.0
Perc_G=1.0/3.0
Perc_B=1.0/3.0
"""
class LUT3D (object):
    """
    Implements 3D LUT. Size must be 2**n.
    Array shape should be (size, size, size, 3)
    """
    def __init__(self, LUT3DArray, size=LUTSIZE):
        # consistency check
        if not (size & (size - 1)):
            raise ValueError("LUT3D : size must be 2**n, found %d" % size)
        self.LUT3DArray = LUT3DArray
        self.size = size
        # for convenience
        self.step = 256 / (size - 1)
        self.contrast = lambda p : p #np.power(p,1.2)

    @classmethod
    def readFromTextStream(self, inStream):
        #header
        for i in range(2):
            line = inStream.readLine()
        buf = np.zeros((33**3)*3, dtype=float)
        i=0
        while True:
            line = inStream.readLine()
            if len(line) == 0:
                break
            a, b, c = line.split(" ")
            buf[i], buf[i+1], buf[i+2] = float(a), float(b), float(c)
            i+=3
        buf = buf.reshape(33,33,33,3)
        return buf

    @classmethod
    def readFromTextFile(self, filename):
        qf = QFile(filename)
        qf.open(QIODevice.ReadOnly)
        textStream = QTextStream(qf)
        LUT3DArray = self.readFromTextStream(textStream)
        LUT3DArray = LUT3DArray *255.0
        LUT3DArray = LUT3DArray.astype(int)
        LUT3DArray = LUT3DArray.transpose(2,1,0, 3)
        return LUT3DArray


def LUT3DFromFactory(size=LUTSIZE):
    """
    Inits a LUT3D array of shape ( size, size,size, 3).
    size should be 2**n +1. Most common values are 17 and 33.
    The 4th axis holds 3-uples (r,g,b) of integers evenly
    distributed in the range 0..256 (edges inclusive) :
    let step = 256 / (size - 1), then
    LUT3DArray(i, j, k) = (i*step, j*step, k*step).
    Note that, with these initialization, trilinear interpolation boils down to identity :
    for all i,j,k in the range 0..256,
    trilinear(i//step, j//step, k//step) = (i,j,k).
    @param size: integer value (should be 2**n+1)
    @return: 3D LUT table
    @rtype: LUT3D object shape (size, size, size, 3), dtype=int
    """
    step = 256 / (size - 1)
    a = np.arange(size)
    return LUT3D(cartesianProduct((a, a, a)) * step, size=size)

def redistribute_rgb(r, g, b):
    """
     To keep the hue, we want to maintain the ratio of (middle-lowest)/(highest-lowest).
    @param r:
    @param g:
    @param b:
    @return:
    """

    threshold = 255.999
    m = max(r, g, b)

    if m <= threshold:
        return int(r), int(g), int(b)

    total = r + g + b
    if total >= 3 * threshold:
        return int(threshold), int(threshold), int(threshold)

    # 3.0*m > 3.0*treshold > total
    x = (3.0 * threshold - total) / (3.0 * m - total)
    gray = threshold - x * m
    return int(gray + x * r), int(gray + x * g), int(gray + x * b)

def rgb2hsp(r, g, b):
    return rgb2hsB(r,g,b, perceptual=True)

def rgb2hspVec(rgbImg):
    return rgb2hsBVec(rgbImg, perceptual=True)

def rgb2hsB(r, g, b, perceptual=False):
    """
    transform the red, green ,blue r, g, b components of a color
    into hue, saturation, brightness h, s, v. (Cf. schema in file colors.docx)
    The r, g, b components are integers in range 0..255. If perceptual is False
    (default) v = max(r,g,b)/255.0, else v = sqrt(Perc_R*r*r + Perc_G*g*g + Perc_B*b*b)
    @param r:
    @param g:
    @param b:
    @param perceptual: boolean
    @return: h, s, v float values : 0<=h<360, 0<=s<=1, 0<=v<=1
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
    Vectorized version of rgb2hsB
    @param rgbImg: (n,m,3) array of rgb values
    @return: identical shape array of hsB values 0<=h<=360, 0<=s<=1, 0<=v<=1
    """
    r, g, b = rgbImg[:, :, 0].astype(float), rgbImg[:, :, 1].astype(float), rgbImg[:, :, 2].astype(float)

    cMax = np.maximum.reduce([r, g, b]) #.astype(float)
    cMin = np.minimum.reduce([r, g, b]) #.astype(float)
    delta = cMax - cMin

    H1 = 1.0 / delta
    H1 = np.where( delta==0.0, 0.0, H1)
    H2 = 60.0 * (g - b) * H1
    H3 = np.where(g>=b, H2, 360 + H2)
    H4 = 60.0 * (2.0 + (b-r)*H1)
    H5 = 60.0 * (4.0 + (r-g)*H1)

    H = np.where(delta==0.0, 0.0, np.where(cMax==r, H3, np.where(cMax==g, H4, H5)))
    """
    # hue
    if delta == 0:
        H = 0.0
    elif cMax == r:
        H = 60.0 * float(g-b)/delta if g >= b else 360 + 60.0 * float(g-b)/delta
    elif cMax == g:
        H = 60.0 * (2.0 + float(b-r)/delta)
    elif cMax == b:
        H = 60.0 * (4.0 + float(r-g)/delta)
    """
    #saturation
    S = np.where(cMax==0.0, 0.0, delta / cMax)

    #S = 0.0 if cMax == 0.0 else float(delta)/cMax

    # brightness
    if perceptual:
        V = np.sqrt(Perc_R * r * r + Perc_G * g * g + Perc_B * b * b)
        V = V / 255.0
    else:
        V = cMax/255.0
    #assert 0<=H and H<=360 and 0<=S and S<=1 and 0<=V and V<=1, "rgb2hsv conversion error r=%d, g=%d, b=%d" %(r,g,b)
    return np.dstack((H,S,V))

def hsv2rgb(h,s,v):
    """
    Transform the hue, saturation, brightness h, s, v components of a color
    into red, green, blue values. (Cf. schema in file colors.docx)
    Note : here, brightness is v= max(r,g,b)/255.0. For perceptual brightness use
    hsp2rgb()

    @param h: float value in range 0..360
    @param s: float value in range 0..1
    @param v: float value in range 0..1
    @return: r,g,b integers between 0 and 255
    """
    assert h>=0 and h<=360 and s>=0 and s<=1 and v>=0 and v<=1

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
    @param hsvImg: hsv image array
    @return: rgb image array
    """

    h,s,v = hsvImg[:,:,0], hsvImg[:,:,1], hsvImg[:,:,2]

    shape = h.shape
    i = (h / 60.0).astype(int)
    f = h / 60.0 - i

    q = f
    t = 1.0 - f
    i = np.ravel(i)
    f = np.ravel(f)
    i %= 6

    t = np.ravel(t)
    q = np.ravel(q)

    clist = (1 - np.ravel(s) * np.vstack([np.zeros_like(f), np.ones_like(f), q, t])) * np.ravel(v)

    # 0:v 1:p=v(1-s) 2:q 3:t
    order = np.array([[0, 3, 1], [2, 0, 1], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]])
    rgb = clist[order[i], np.arange(np.prod(shape))[:, None]]

    """
    clipped = np.amax(rgb, axis=1)
    clipped = np.dstack((clipped, clipped, clipped))
    rgb = np.where(clipped > 1, rgb / clipped, rgb)
    """
    rgb = (rgb * 255.0).astype(int)

    return rgb.reshape(shape + (3,))


def hsp2rgbOld(h,s,p, trunc=True):
    """
    Transform the hue, saturation and perceptual brightness h, s, p components of a color
    into red, green, blue values.
    @param h: float value in range 0..360
    @param s: float value in range 0..1
    @param p: float value in range 0..1
    @return: r,g,b integers between 0 and 255
    """

    h = h /60.0
    i = np.floor(h) # TODO put into code
    f = h - i

    if s == 1.0:
        if h < 1.0 :  # r > g > b=0
            h = h # f
            r = np.sqrt(p * p / (Perc_R + Perc_G * h * h))
            g = r * h
            b = 0.0
        elif h < 2.0:  # g>r>b=0
            h = (-h + 2.0) # 1-f
            g = np.sqrt(p * p / (Perc_G + Perc_R * h * h))
            r = g * h
            b = 0.0
        elif h < 3.0: # g>b>r=0
            h = (h - 2.0) # f
            g = np.sqrt(p * p / (Perc_G + Perc_B * h * h))
            b = g * h
            r = 0.0
        elif h < 4.0 : # b>g>r=0
            h = (-h + 4.0) #1 - f
            b = np.sqrt(p * p / (Perc_B + Perc_G * h * h))
            g = b * h
            r = 0.0
        elif h < 5.0 :  # b>r>g=0
            h = (h - 4.0) # f
            b = np.sqrt(p * p / (Perc_B + Perc_R * h * h))
            r = b * h
            g = 0.0
        else : # r>b>g=0
            h = (-h + 6.0) # 1 -f
            r = np.sqrt(p * p / (Perc_R + Perc_B * h * h))
            b = r * h
            g = 0.0
    else:  # s !=1
        Mm = 1.0 / (1.0 - s)  #Mm >= 1
        if h < 1.0 :  # r > g > b
            h = h#/60.0
            part = 1.0 + h * (Mm - 1.0)  # part >=1 part = g/b
            b = p / np.sqrt(Perc_R * Mm * Mm + Perc_G * part * part + Perc_B)  # b<=p
            r = b * Mm
            g = b + h * (r - b)
        elif h < 2.0: #g>r>b
            h = (-h + 2.0)
            part = 1.0 + h * (Mm - 1.0) #part = r/b
            b = p / np.sqrt(Perc_G * Mm * Mm + Perc_R * part * part + Perc_B)
            g = b * Mm
            r = b + h * (g - b)
        elif h < 3.0: # g>b>r
            h = (h - 2.0)
            part = 1.0 + h * (Mm - 1.0) # part = b/r
            r = p / np.sqrt(Perc_G * Mm * Mm + Perc_B * part * part + Perc_R)
            g = r * Mm
            b = r + h * (g - r)
        elif h < 4.0: # b>g>r
            h = (-h + 4.0)
            part = 1.0 + h * (Mm - 1.0)
            r = p / np.sqrt(Perc_B * Mm * Mm + Perc_G * part * part + Perc_R)
            b = r * Mm
            g = r  + h * (b - r)
        elif h < 5.0: # b>r>g
            h = (h - 4.0)
            part = 1.0 + h * (Mm - 1.0)
            g = p / np.sqrt(Perc_B * Mm * Mm + Perc_R * part * part + Perc_G)
            b = g * Mm
            r = g + h * (b - g)
        else: # r>b>g
            h = (-h + 6.0)
            part = 1.0 + h * (Mm - 1.0)
            g = p / np.sqrt(Perc_R * Mm * Mm + Perc_B * part * part + Perc_G)
            r = g * Mm
            b = g + h * (r - g)

    if r<0 or g<0 or b<0 :
        print 'neg value found', h,s,p, r,g,b

    pc= Perc_R * r * r + Perc_G * g * g + Perc_B * b * b
    assert abs(p*p - pc)<= 0.000001, 'hsp2rgb error'
    if trunc:
        return min(255,int(round(r*255))), min(255,int(round(g*255))), min(255, int(round(b*255)))
    else:
        return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))

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

    if r<0 or g<0 or b<0 :
        print 'neg value found', h,s,p, r,g,b

    pc= Perc_R * r * r + Perc_G * g * g + Perc_B * b * b
    assert abs(p*p - pc)<= 0.000001, 'hsp2rgb error'
    M = max(r, g, b)
    #clippingInd = (max(r,g,b) > 1.0)
    if trunc:
        if M > 1:
            r, g, b = r / M, g / M, b / M
        #return min(255,int(round(r*255.0))), min(255,int(round(g*255.0))), min(255, int(round(b*255.0))), (M > 1)
    return int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0)), (M > 1)
    #else:
        #return int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), ( M > 1)

def hsp2rgbVec(hspImg):
    """
    Vectorized version of hsp2rgb

    @param hspImg: (n,m,3) array of hsp values
    @return: identical shape array of rgb values
    """
    h, s, p = hspImg[:, :, 0], hspImg[:, :, 1], hspImg[:, :, 2]

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

    part1 = 1.0 + f * (Mm - 1.0)
    part1 = np.where(Mm == np.inf, f, part1)  # TODO some invalid values remain for s = 1
    part2 = 1.0 + (1 - f) * (Mm - 1.0)
    part2 = np.where(Mm == np.inf, 1 - f, part2)

    part1 = part1 * part1
    part2 = part2 * part2


    X1 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_R * Mm2 + Perc_G * part1 + Perc_B))   # b
    Y1 = np.where(Mm==np.inf, p / np.sqrt(Perc_R+ Perc_G * f *f), X1 * Mm)              # r
    Z1 = np.where(Mm==np.inf, Y1 * f, X1 + f * (Y1 - X1))                               # g

    X2 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_G * Mm2 + Perc_R * part2 + Perc_B))   # b
    Y2 = np.where(Mm==np.inf, p / np.sqrt(Perc_G + Perc_R * (1-f) * (1-f)), X2 * Mm)    # g
    Z2 = np.where(Mm==np.inf, Y2 * (1-f), X2 + (1 - f) * (Y2 - X2))                     # r

    X3 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_G * Mm2 + Perc_B * part1 + Perc_R))   # r
    Y3 = np.where(Mm==np.inf, p / np.sqrt(Perc_G + Perc_B * f * f), X3 * Mm)            # g
    Z3 = np.where(Mm==np.inf, Y3 * f, X3 + f * (Y3 - X3))                               # b

    X4 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_B * Mm2 + Perc_G * part2 + Perc_R))   # r
    Y4 = np.where(Mm==np.inf, p / np.sqrt(Perc_B + Perc_G * (1-f) * (1-f)), X4 * Mm)    # b
    Z4 = np.where(Mm==np.inf, Y4 * (1 - f), X4 + (1 - f) * (Y4 - X4))                   # g

    X5 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_B * Mm2 + Perc_R * part1 + Perc_G))   # g
    Y5 = np.where(Mm==np.inf, p / np.sqrt(Perc_B + Perc_R * f * f), X5 * Mm)            # b
    Z5 = np.where(Mm==np.inf, Y5 * f, X5 + f * (Y5 - X5))                               # r

    X6 = np.where(Mm==np.inf, 0, p / np.sqrt(Perc_R * Mm2 + Perc_B * part2 + Perc_G))   # g
    Y6 = np.where(Mm==np.inf, p / np.sqrt(Perc_R + Perc_B * (1-f) * (1-f)), X6 * Mm)    # r
    Z6 = np.where(Mm==np.inf, Y6 * (1 - f), X6 + (1 - f) * (Y6 - X6))                   # b

    np.seterr(**old_settings)

    # stacked as lines
    clist = np.vstack((X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4, X5, Y5, Z5, X6, Y6, Z6))

    # for hue slices 0,..,5, the corresponding 3-uple gives the line indices in clist for the r,g,b values
    order = np.array([[1, 2, 0], [5, 4, 3], [6, 7, 8], [9, 11, 10], [14, 12, 13], [16, 15, 17]])

    tmp = np.arange(np.prod(shape))[:, None]

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
def colorPicker(w,h):

    img = QImage(w,h, QImage.Format_ARGB32)
    cx=w/2
    cy=h/2
    for i in range(w):
        for j in range(h):
            i1=i-cx
            j1=-j+cy
            m = max(abs(i1), abs(j1))
            hue = np.arctan2(j1,i1)*180.0/np.pi + 315
            hue = hue - floor(hue/360.0)*360.0
            #sat = np.sqrt(i1*i1 + j1*j1)/np.sqrt(w*w/2.0)
            #sat = float(m) /cx
            sat = sat = np.sqrt(i1*i1 + j1*j1)/cx
            sat = min(1.0, sat)
            c = QColor(*hsp2rgb(hue,sat,0.45))
            #if i == w-1 and j == 0:#0:#h - 90:
            if hue==0.0:
                r=c.red()
                g=c.green()
                b=c.blue()
                print hue,sat, r,g,b, i, j
                print np.sqrt((Pr*r*r+Pg*g*g+Pb*b*b))/255.0
            img.setPixel(i,j,c.rgb())
    img = imImage(QImg=img)
    return img
"""

def interpVec(LUT, ndImg):
    """
    Convert an RGB image using a 3D LUT. The output image is interpolated from the LUT.
    It has the same dimensions and type as the input image.
    We use a vectorized version of trilinear interpolation.
    Cf. file trilinear.docx for details

    @param LUT: 3D LUT array
    @param ndImg: image array
    @return: RGB image with same dimensions as the input image
    """

    # bounding unit cube coordinates for (r, g, b)/LUTSTEP
    ndImgF = ndImg/float(LUTSTEP)
    a=np.floor(ndImgF).astype(int)
    aplus1 = a + 1


    # RGB channels
    r0, g0, b0 = a[:,:,0], a[:,:,1], a[:,:,2]
    r1, g1, b1 = aplus1[:,:,0], aplus1[:,:,1], aplus1[:,:,2]

    #apply LUT to cube vertices
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
    #alpha = alpha[..., np.newaxis]  slower !
    oneMinusAlpha = 1 - alpha

    I11Value = oneMinusAlpha * ndImg11 + alpha * ndImg13
    I12Value = oneMinusAlpha * ndImg10 + alpha * ndImg12
    I21Value = oneMinusAlpha * ndImg01 + alpha * ndImg03
    I22Value = oneMinusAlpha * ndImg00 + alpha * ndImg02
    """
    I11Value = ndImg11 + alpha * (ndImg13 - ndImg11)
    I12Value = ndImg10 + alpha * (ndImg12 - ndImg10)
    I21Value = ndImg01 + alpha * (ndImg03 - ndImg01)
    I22Value = ndImg00 + alpha * (ndImg02 - ndImg00)
    """
    beta = ndImgF[:,:,0] - r0
    beta = np.dstack((beta, beta, beta))
    #beta = beta[..., np.newaxis] slower !
    oneMinusBeta = 1 - beta
    I1Value = oneMinusBeta * I12Value + beta * I11Value
    I2Value = oneMinusBeta * I22Value + beta * I21Value

    #I1Value = oneMinusBeta * oneMinusAlpha * ndImg10 + oneMinusBeta * alpha * ndImg12 + beta * oneMinusAlpha * ndImg11 + beta * alpha * ndImg13
    #I2Value = oneMinusBeta * oneMinusAlpha * ndImg00 + oneMinusBeta * alpha * ndImg02 + beta * oneMinusAlpha * ndImg01 + beta * alpha * ndImg03

    gamma = ndImgF[:,:,2] - b0
    gamma = np.dstack((gamma, gamma, gamma))
    #gamma = gamma[..., np.newaxis] slower !
    IValue = (1 - gamma) * I2Value + gamma * I1Value

    return IValue


if __name__=='__main__':
    # random ints in range 0 <= x < 256
    b = np.random.randint(0,256, size=500*500*3, dtype=np.uint8)
    testImg = np.reshape(b, (500,500,3))
    interpImg = LUT3DFromFactory(33)
    interpImg = interpVec(LUT3D_ORI, testImg)
    d = testImg - interpImg
    if (d != 0.0).any():
        print "interpolation error"






