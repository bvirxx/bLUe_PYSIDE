import numpy as np
from PyQt4.QtCore import Qt, QPoint, QPointF
from PyQt4.QtGui import QImage, QColor

#from MarkedImg import mImage, imImage

# 3D LUT init.
"""
Each axis has length LUTSIZE.
r,g,b values are between 0 and 256.
"""
LUTSIZE = 17
#LUTSIZE = 129
LUTSTEP = 256 / (LUTSIZE - 1)
LUT3D = np.array([[[(i * LUTSTEP, j * LUTSTEP, k * LUTSTEP) for k in range(LUTSIZE)] for j in range(LUTSIZE)] for i in range(LUTSIZE)])


# perceptual brightness constants

Perc_R = 0.299
Perc_G = 0.587
Perc_B = 0.114
"""
Perc_R=0.2126
Perc_G=0.7152
Perc_B=0.0722
"""
# interpolated from 3D lUT Creator
Perc_R=0.2338
Perc_G=0.6880
Perc_B=0.0782

"""
Perc_R=0.79134178
Perc_G=2.31839104
Perc_B=0.25510923
"""
def reset_LUT3D():
    return np.array([[[(i * LUTSTEP, j * LUTSTEP, k * LUTSTEP) for k in range(LUTSIZE)] for j in range(LUTSIZE)] for i in range(LUTSIZE)])

class QPoint3D(object):
    def __init__(self, x,y,z):
        self.x_ =x
        self.y_=y
        self.z_=z

    def x(self):
        return self.x_
    def y(self):
        return self.y_
    def z(self):
        return self.z_

    def __add__(self, other):
        return QPoint3D(self.x_ + other.x_, self.y_ + other.y_, self.z_ + other.z_)

    def __radd__(self, other):
        return QPoint3D(self.x_ + other.x_, self.y_ + other.y_, self.z_ + other.z_)

    def __mul__(self, scalar):
        return QPoint3D(scalar*self.x_, scalar.self.y_, scalar.self.z_)

    def __rmul__(self, scalar):
        return QPoint3D(scalar*self.x_, scalar.self.y_, scalar.self.z_)

def rgb2hsB(r, g, b, perceptual=False):
    """
    transform the red, green ,blue r, g, b components of a color
    into hue, saturation, brightness h, s, v. (Cf. schema in file colors.docx)
    The r, g, b components are integers in range 0..255. If perceptual is False
    (default) v = max(r,g,b)/255.0, else v = sqrt(Perc_R*r*r + Perc_G*g*g + Perc_B*b*b)
    :param r:
    :param g:
    :param b:
    :param perceptual: boolean
    :return: h, s, v float values : 0<=h<360, 0<=s<=1, 0<=v<=1
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

    #saturation
    S = 0.0 if cMax == 0.0 else float(delta)/cMax

    # brightness
    if perceptual:
        V = np.sqrt(Perc_R * r * r + Perc_G * g * g + Perc_B * b * b)
        V = V / 255.0
    else:
        V = cMax/255.0
    assert 0<=H and H<=360 and 0<=S and S<=1 and 0<=V and V<=1, "rgb2hsv conversion error"
    return H,S,V

def hsv2rgb(h,s,v):
    """
    Transform the hue, saturation, brightness h, s, v components of a color
    into red, green, blue values. (Cf. schema in file colors.docx)
    Note : here, v= max(r,g,b)/255.0. For perceptual brightness use
    hsp2rgb()

    :param h: float value in range 0..360
    :param s: float value in range 0..1
    :param v: float value in range 0..1
    :return: r,g,b integers between 0 and 255
    """
    assert h>=0 and h<=360 and s>=0 and s<=1 and v>=0 and v<=1

    h = h/60.0
    i = np.floor(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * ( 1 - s * ( 1 - f))

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

    r = int(r1 * 255)
    g = int(g1 * 255)
    b = int(b1 * 255)

    return r,g,b

def hsp2rgbOld(h,s,p, trunc=True):
    """
    Transform the hue, saturation and perceptual brightness h, s, p components of a color
    into red, green, blue values.
    :param h: float value in range 0..360
    :param s: float value in range 0..1
    :param p: float value in range 0..1
    :return: r,g,b integers between 0 and 255
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

def hsp2rgb(h,s,p):
    return hsp2rgb_ClippingInd(h,s,p)[:3]

def hsp2rgb_ClippingInd(h,s,p, trunc=True):
    """
    Transform the hue, saturation and perceptual brightness components of a color
    into red, green, blue values.
    :param h: float value in range 0..360
    :param s: float value in range 0..1
    :param p: float value in range 0..1
    :return: r,g,b integers between 0 and 255
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
    clippingInd = max(r,g,b) > 1.0
    if trunc:
        return min(255,int(round(r*255))), min(255,int(round(g*255))), min(255, int(round(b*255))), clippingInd
    else:
        return int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), clippingInd


def hsp2rgbVec(hspImg):
    """

    :param hspImg:
    :return:
    """
    h, s, p = hspImg[:, :, 0], hspImg[:, :, 1], hspImg[:, :, 2]

    shape = h.shape

    h = np.ravel(h)
    s = np.ravel(s)
    p=np.ravel(p)
    h = h / 60.0
    i = np.floor(h).astype(int)
    f = h - i

    Mm = 1.0 / (1.0 - s) # where s=1 Mm=inf
    part1 = 1.0 + f * (Mm - 1.0)
    part1 = np.where(Mm==np.inf, f, part1 )             # TODO some invalid values remain for s = 1
    part2 = 1.0 + (1-f) * (Mm - 1.0)
    part2 = np.where(Mm==np.inf, 1-f, part2 )

    part1 = part1 * part1
    part2 = part2 * part2
    Mm2 = Mm * Mm

    X1 = p / np.sqrt(Perc_R * Mm2  + Perc_G * part1  + Perc_B)
    Y1 = X1 * Mm
    Z1 = X1 + f * (Y1 - X1)
    X2 = p / np.sqrt(Perc_G * Mm2  + Perc_R * part2  + Perc_B)
    Y2 = X2 * Mm
    Z2 = X2 + (1-f) * (Y2 - X2)
    X3 = p / np.sqrt(Perc_G * Mm2  + Perc_B * part1  + Perc_R)
    Y3 = X3 * Mm
    Z3 = X3 + f * (Y3 - X3)
    X4 = p / np.sqrt(Perc_B * Mm2  + Perc_G * part2  + Perc_R)
    Y4 = X4 * Mm
    Z4 = X4 + (1-f) * (Y4 - X4)
    X5 = p / np.sqrt(Perc_B * Mm2  + Perc_R * part1  + Perc_G)
    Y5 = X5 * Mm
    Z5 = X5 + f * (Y5 - X5)
    X6 = p / np.sqrt(Perc_R * Mm2  + Perc_B * part2  + Perc_G)
    Y6 = X6 *Mm
    Z6 = X6 + (1-f) * (Y6 - X6)

    clist = np.vstack((X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4, X5,Y5,Z5,X6,Y6,Z6))

    # each 3-uple gives line numbers in clist for hue slice i = 0,..5
    order = np.array([[1, 2, 0], [5, 4, 3], [6, 7, 8], [9, 11, 10], [14, 12, 13], [16, 15,17]])

    tmp = np.arange(np.prod(shape))[:, None]

    # we use advanced array indexing
    rgb = clist[order[i], tmp]  # order[i] has shape (w*h, 3), tmp is broadcast to the same shape, thus rgb has (shape w*h, 3)

    # converting invalid values to int gives indeterminate results
    rgb[np.isnan(rgb)]=0.0

    # for uint8 image buffer, int values > 255 are truncated to lower 8 bits
    #rgbMask = (rgb > 1)
    rgb = np.clip(rgb, 0, 1.0)

    rgb = (rgb * 255).astype(int)

    return rgb.reshape(shape + (3,))


def hsv2rgbVec(hsvImg, trunc=True):
    """
    Transform the hue, saturation and brightness h, s, v components of a color
    into red, green, blue values.
    :param hspImg: hsv image array
    :param trunc : if True, the rgb values of the output image are truncated to range 0..255
    :return: rgb image array
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
    rgb = (rgb * 255).astype(int)

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

    :param LUT: 3D LUT array
    :param ndImg: image array
    :return: RGB image with same dimensions as the input image
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
"""
def interp(LUT, i,j,k):
    Trilinear interpolation in a 3D LUT

                                              k    I12
                                           F1 |---------- D1
                                              |           |
                                       E1 |   |   I11    |
                                          |   |           |
                                          |C0/--------------j
                                          | /      /I22    E0
                                          |/      /
                                          /      /I2
                                      D0 /------/-------F0
                                        /        I21
                                        i


    :param LUT 3D LUT array
    :param i,j,k: coordinates to interpolate
    :return: interpolated value

    i16, j16, k16 = i/16, j/16, k/16

    C0 = (i16 , j16, k16)
    D0 = (i16 +1 , j16, k16)    #C0 + QPoint3D(1,0,0)
    E0 = (i16 , j16+1, k16)     #C0 + QPoint3D(0,1,0)
    F0 = (i16+1, j16+1, k16)    # C0 + QPoint3D(1, 1, 0)

    C1= (i16 +1 , j16+1, k16+1) #C0 + QPoint3D(1,1,1)
    D1 = (i16, j16+1, k16+1)    #C1 - QPoint3D(1, 0, 0)
    E1 = (i16+1, j16, k16+1)    #C1 -  QPoint3D(0, 1, 0)
    F1 =  (i16, j16, k16+1)     #C1 - QPoint3D(1, 1, 0)

    iP = float(i)/16
    jP = float(j)/16
    kP = float(k) / 16

    I1 = (iP,jP,C1[2])
    I2= (iP,jP, C0[2])
    I11 = (C1[0], jP, C1[2])  # C1.x(), C0.y(),  C1.z()  and C1
    I12 = (C0[0], jP, C1[2])  # C0.x(), C0.y(), C1.z() and C0.x(), C1.y(),C1.z()
    I21 = (C1[0], jP, C0[2])  # C1.x(), C0.y(),  C0.z()  and C1.x(), C1.y(),  C0.z()
    I22 = (C0[0], jP, C0[2])  # C0.x(), C0.y(),  C0.z()  and C0.x(), C1.y(),  C0.z()

    alpha = float(jP-E1[1])/(C1[1]-E1[1])
    I11Value= LUT[E1] + alpha*(LUT[C1]-LUT[E1])

    alpha = float(jP - F1[1]) / (D1[1] - F1[1])
    I12Value = LUT[F1] + alpha * (LUT[D1] - LUT[F1])

    alpha = float(jP - D0[1]) / (F0[1] - D0[1])
    I21Value = LUT[D0] + alpha * (LUT[F0] - LUT[D0])

    alpha = float(jP - C0[1]) / (E0[1] - C0[1])
    I22Value = LUT[C0] + alpha * (LUT[E0] - LUT[C0])

    alpha = float(iP - I11[0]) / (I12[0] - I11[0])
    I1Value = I11Value + alpha * (I12Value - I11Value)

    alpha = float(iP - I21[0]) / (I22[0] - I21[0])
    I2Value = I21Value + alpha * (I22Value - I21Value)

    alpha = float(kP - I1[2]) / (I2[2] - I1[2])
    IValue = I1Value + alpha * (I2Value - I1Value)

    #print "ivalue", IValue, i,j,k
    return IValue
"""
def lutNN(LUT, r,g,b):
    """
    Get the nearest neighbor vertex of a (r,g,b) value in 3D LUT.
    :param LUT: 3D LUT array
    :param r:
    :param g:
    :param b:
    :return: 3-uple index of the NN vertex.
    """

    x = 0 if r % LUTSTEP < LUTSTEP / 2 else 1
    y = 0 if g % LUTSTEP < LUTSTEP / 2 else 1
    z = 0 if b % LUTSTEP < LUTSTEP / 2 else 1

    NN = (r / LUTSTEP + x, g / LUTSTEP + y , b / LUTSTEP + z)

    return NN

if __name__=='__main__':

    pass

"""

#define  Pr  .299
#define  Pg  .587
#define  Pb  .114



//  public domain function by Darel Rex Finley, 2006
//
//  This function expects the passed-in values to be on a scale
//  of 0 to 1, and uses that same scale for the return values.
//
//  See description/examples at alienryderflex.com/hsp.html

void RGBtoHSP(
double  R, double  G, double  B,
double *H, double *S, double *P) {

  //  Calculate the Perceived brightness.
  *P=sqrt(R*R*Pr+G*G*Pg+B*B*Pb);

  //  Calculate the Hue and Saturation.  (This part works
  //  the same way as in the HSV/B and HSL systems???.)
  if      (R==G && R==B) {
    *H=0.; *S=0.; return; }
  if      (R>=G && R>=B) {   //  R is largest
    if    (B>=G) {
      *H=6./6.-1./6.*(B-G)/(R-G); *S=1.-G/R; }
    else         {
      *H=0./6.+1./6.*(G-B)/(R-B); *S=1.-B/R; }}
  else if (G>=R && G>=B) {   //  G is largest
    if    (R>=B) {
      *H=2./6.-1./6.*(R-B)/(G-B); *S=1.-B/G; }
    else         {
      *H=2./6.+1./6.*(B-R)/(G-R); *S=1.-R/G; }}
  else                   {   //  B is largest
    if    (G>=R) {
      *H=4./6.-1./6.*(G-R)/(B-R); *S=1.-R/B; }
    else         {
      *H=4./6.+1./6.*(R-G)/(B-G); *S=1.-G/B; }}}



//  public domain function by Darel Rex Finley, 2006
//
//  This function expects the passed-in values to be on a scale
//  of 0 to 1, and uses that same scale for the return values.
//
//  Note that some combinations of HSP, even if in the scale
//  0-1, may return RGB values that exceed a value of 1.  For
//  example, if you pass in the HSP color 0,1,1, the result
//  will be the RGB color 2.037,0,0.
//
//  See description/examples at alienryderflex.com/hsp.html

void HSPtoRGB(
double  H, double  S, double  P,
double *R, double *G, double *B) {

  double  part, minOverMax=1.-S ;

  if (minOverMax>0.) {
    if      ( H<1./6.) {   //  R>G>B
      H= 6.*( H-0./6.); part=1.+H*(1./minOverMax-1.);
      *B=P/sqrt(Pr/minOverMax/minOverMax+Pg*part*part+Pb);
      *R=(*B)/minOverMax; *G=(*B)+H*((*R)-(*B)); }
    else if ( H<2./6.) {   //  G>R>B
      H= 6.*(-H+2./6.); part=1.+H*(1./minOverMax-1.);
      *B=P/sqrt(Pg/minOverMax/minOverMax+Pr*part*part+Pb);
      *G=(*B)/minOverMax; *R=(*B)+H*((*G)-(*B)); }
    else if ( H<3./6.) {   //  G>B>R
      H= 6.*( H-2./6.); part=1.+H*(1./minOverMax-1.);
      *R=P/sqrt(Pg/minOverMax/minOverMax+Pb*part*part+Pr);
      *G=(*R)/minOverMax; *B=(*R)+H*((*G)-(*R)); }
    else if ( H<4./6.) {   //  B>G>R
      H= 6.*(-H+4./6.); part=1.+H*(1./minOverMax-1.);
      *R=P/sqrt(Pb/minOverMax/minOverMax+Pg*part*part+Pr);
      *B=(*R)/minOverMax; *G=(*R)+H*((*B)-(*R)); }
    else if ( H<5./6.) {   //  B>R>G
      H= 6.*( H-4./6.); part=1.+H*(1./minOverMax-1.);
      *G=P/sqrt(Pb/minOverMax/minOverMax+Pr*part*part+Pg);
      *B=(*G)/minOverMax; *R=(*G)+H*((*B)-(*G)); }
    else               {   //  R>B>G
      H= 6.*(-H+6./6.); part=1.+H*(1./minOverMax-1.);
      *G=P/sqrt(Pr/minOverMax/minOverMax+Pb*part*part+Pg);
      *R=(*G)/minOverMax; *B=(*G)+H*((*R)-(*G)); }}
  else {
    if      ( H<1./6.) {   //  R>G>B
      H= 6.*( H-0./6.); *R=sqrt(P*P/(Pr+Pg*H*H)); *G=(*R)*H; *B=0.; }
    else if ( H<2./6.) {   //  G>R>B
      H= 6.*(-H+2./6.); *G=sqrt(P*P/(Pg+Pr*H*H)); *R=(*G)*H; *B=0.; }
    else if ( H<3./6.) {   //  G>B>R
      H= 6.*( H-2./6.); *G=sqrt(P*P/(Pg+Pb*H*H)); *B=(*G)*H; *R=0.; }
    else if ( H<4./6.) {   //  B>G>R
      H= 6.*(-H+4./6.); *B=sqrt(P*P/(Pb+Pg*H*H)); *G=(*B)*H; *R=0.; }
    else if ( H<5./6.) {   //  B>R>G
      H= 6.*( H-4./6.); *B=sqrt(P*P/(Pb+Pr*H*H)); *R=(*B)*H; *G=0.; }
    else               {   //  R>B>G
      H= 6.*(-H+6./6.); *R=sqrt(P*P/(Pr+Pb*H*H)); *B=(*R)*H; *G=0.; }}}


"""

"""
 v = cv2.calcHist([image],     #list of images
                    [0, 1, 2], # list of channels
                    None,       # mask
                    [8, 8, 8], # hist size for each channel
                    [0, 256, 0, 256, 0, 256]) # bound values
        v = v.flatten()
        hist = v / sum(v)
        histograms[fname] = hist

"""