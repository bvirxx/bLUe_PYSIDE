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
from PySide.QtCore import Qt
from PySide.QtGui import QColor
from PySide.QtGui import QFontMetrics
from PySide.QtGui import QHBoxLayout
from PySide.QtGui import QImage
from PySide.QtGui import QLabel
from PySide.QtGui import QPainter
from PySide.QtGui import QPushButton
from PySide.QtGui import QSizePolicy
from PySide.QtGui import QSlider
from PySide.QtGui import QVBoxLayout
from PySide.QtGui import QWidget

from blend import blendLuminosity
from imgconvert import QImageBuffer

################
# Conversion Matrices
#################

# Conversion from CIE XYZ to LMS-like color space for chromatic adaptation
# see http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
from utils import optionsWidget

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
# conversion from LINEAR sRGB (D65) to XYZ and backwards.
# see http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
# and https://en.wikipedia.org/wiki/SRGB
########################

sRGB2XYZ = [[0.4124564,  0.3575761,  0.1804375],
            [0.2126729,  0.7151522,  0.0721750],
            [0.0193339,  0.1191920,  0.9503041]]

sRGB2XYZInverse = [[3.2404542, -1.5371385, -0.4985314],
                   [-0.9692660, 1.8760108,  0.0415560],
                    [0.0556434, -0.2040259, 1.0572252]]

################
# Constants and precomputed tables for the
# sRGB linearizing functions
# rgbLinear2rgbVec,
# rgb2rgbLinearVec.
# See https://en.wikipedia.org/wiki/SRGB
################

a = 0.055
alpha = 2.4
beta = 1.0 / alpha
b = (a / (1.0 + a)) ** alpha
d = 12.92
c = 255.0 * d
e = 255*255
F = 255.0**(2*beta)

table0 = np.arange(256, dtype=np.float64)
table1 = table0 / 255.0  # np.array([i/255.0 for i in xrange(256)])
table2 = table0 / c  # np.array([i/c for i in xrange(256)])
table3 = np.power(table1, alpha)
table4 = np.arange(e + 1, dtype = np.float64)
table5 = np.power(table4, beta)

def rgbLinear2rgb(r,g,b):

    """
    Conversion from linear sRGB to sRGB.
    All values are in range 0..1.
    
    :param r:
    :param g:
    :param b:
    :return: The converted values
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
    :param img: linear sRGB image (RGB range 0..1, type numpy array, dtype=np.float64)
    :return: converted image (RGB range 0..255)
    See https://en.wikipedia.org/wiki/SRGB
    """
    img2 = d * img
    #img3 = (1.0 + a) * np.power(img, beta) - a
    imgDiscretized = (img * e )
    imgDiscretized = np.clip(imgDiscretized, 0, e)
    imgDiscretized = np.floor(imgDiscretized)
    imgDiscretized = imgDiscretized.astype(float)
    imgDiscretized = imgDiscretized.astype(int)
    imgDiscretized = np.clip(imgDiscretized, 0, e)
    img3 = (1.0 + a) * table5[imgDiscretized] / F
    return np.where(img <=  0.0031308, img2, img3) * 255

def rgb2rgbLinear(r,g,b):
    """
       Conversion from sRGB to LINEAR sRGB.
       All values are in range 0..1.
       See https://en.wikipedia.org/wiki/SRGB
       :param r:
       :param g:
       :param b:
       :return: The converted values
       """
    def c2cl(c):
        if c <= 0.04045:
            c =  c / d
        else:
            c = ((c+a)/(1+a))**alpha
        return c
    return c2cl(r), c2cl(g), c2cl(b)

def rgb2rgbLinearVec(img):
    """
    Converts image from sRGB to linear sRGB.
    THe image colors are first converted to float values in range 0..1
    See https://en.wikipedia.org/wiki/SRGB
    :param img: sRGB image (RGB range 0..255, type numpy array)
    :return: converted image (RGB range 0..1, type numpy array dtype=np.float64)
    """
    img1 =  table1[img[...]]
    img2 = table2[img[...]]
    img3 = table3[img[...]]
    return np.where(img1 <= 0.04045, img2, img3)

def sRGB2XYZVec(imgBuf):
    """
    Conversion from sRGB to XYZ
    :param imgBuf: Array of RGB values, range 0..255
    :return: 
    """
    #buf = QImageBuffer(img)[:, :, :3][:,:,::-1]
    bufLinear = rgb2rgbLinearVec(imgBuf)
    bufXYZ = np.tensordot(bufLinear, sRGB2XYZ, axes=(-1, -1))
    return bufXYZ

def XYZ2sRGBVec(imgBuf):
    """
    
    :param imgBuf: XYZ buffer
    :return: sRGB buffer range 0..255
    """
    #buf = QImageBuffer(img)[:, :, :3]
    bufsRGBLinear = np.tensordot(imgBuf, sRGB2XYZInverse, axes=(-1, -1))
    bufsRGB = rgbLinear2rgbVec(bufsRGBLinear)
    return bufsRGB

def sRGB2LabVec(bufsRGB) :
    """
    https://en.wikipedia.org/wiki/Lab_color_space
    :param bufsRGB: 
    :return: bufLab Image buffer mode Lab, range 0..1, dtype float64
    """
    Xn, Yn, Zn = 95.02, 100.0, 108.82 #95.02, 100.0, 108.82
    Ka, Kb = 172.30, 67.20
    bufXYZ = sRGB2XYZVec(bufsRGB)
    YoverYn = bufXYZ[:,:,1] / Yn
    bufL = np.sqrt(YoverYn)
    bufa = Ka * ( bufXYZ[:,:,0] / Xn - YoverYn) / bufL
    bufb = Kb * (YoverYn - bufXYZ[:,:,2]/Zn) / bufL

    bufLab = np.dstack((bufL, bufa, bufb))

    return bufLab


def Lab2sRGBVec(bufLab):
    """
    https://en.wikipedia.org/wiki/Lab_color_space
    :param bufLab: 
    :return: bufsRGB
    """
    Xn, Yn, Zn = 95.02, 100.0, 108.82
    Ka, Kb = 172.30, 67.20
    #buf =  QImageBuffer(bufLab)
    bufL, bufa, bufb = bufLab[:,:,0], bufLab[:,:,1], bufLab[:,:,2]
    bufL2 = bufL* bufL
    bufY = bufL2 * Yn
    bufX = Xn * ((bufa/ Ka) * bufL + bufL2)
    bufZ = Zn * (bufL2 - ((bufb / Kb)) * bufL)

    bufXYZ = np.dstack((bufX, bufY, bufZ))

    return XYZ2sRGBVec(bufXYZ)


def bbTemperature2RGB(temperature):
    """
    Converts Kelvin temperature to rgb values.
    The temperature is Kelvin.
    See http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code
    :param temp: Kelvin temperature
    :return: r, g, b vlaues in  range 0..255 (type int)
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

    return red, green, blue

"""
def applyTemperature(qImg, temperature, coeff, version=2):
    

    :param qImg:
    :param temperature:
    :param coeff:
    :return:
    
    if version == 0:
        r, g, b = bbTemperature2RGB(temperature)
        filter = QImage(qImg)
        filter.fill(QColor(r,g,b, 255))
        qp = QPainter(filter)
        #qp.setOpacity(coeff)
        qp.setCompositionMode(QPainter.CompositionMode_Multiply)
        qp.drawImage(0,0,qImg)
        qp.end()
        img = blendLuminosity(filter, qImg)
        buf = QImageBuffer(img)
        buf[:,:,3] = int(coeff * 255)
        return img
    else:
        M = conversionMatrix(temperature, 6500)
        img = QImage(qImg)
        buf =QImageBuffer(img)[:,:,:3]
        bufLinear = rgb2rgbLinearVec(buf)
        resLinear = np.tensordot(bufLinear[:,:,::-1], M, axes= (-1,-1))
        res = rgbLinear2rgbVec(resLinear)
        res=np.clip(res, 0, 255)
        buf[:, :, ::-1] = res
    return img
"""
def temperature2xyWP(T):
    """
    Calculates the CIE chromaticity coordinates xc, yc
    of white point from temperature (cubic spline approximation).
    see http://en.wikipedia.org/wiki/Planckian_locus#Approximation

    :param T: temperature in Kelvin
    :return: xc, yc
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

    x,y = temperature2xyWP(T)
    L = 1.0
    X, Y , Z = L * x / y, L, L * (1.0 - x -y ) / y

    rho1, rho2, rho3 = np.dot(np.array(Bradford), np.array([X,Y,Z]).T)

    return rho1, rho2, rho3

def conversionMatrix(Tdest, Tsource):
    rhos1, rhos2, rhos3  = temperature2Rho(Tsource)
    rhod1, rhod2, rhod3 = temperature2Rho(Tdest)
    D = np.diag((rhod1/rhos1, rhod2/rhos2, rhod3/rhos3))
    N = np.dot(np.array(BradfordInverse), D)
    P = np.dot(N, np.array(Bradford))
    Q = np.dot(np.array(sRGB2XYZInverse), P)
    R = np.dot(Q , sRGB2XYZ)
    return R

class temperatureForm (QWidget):
    @classmethod
    def getNewWindow(cls, targetImage=None, size=500, layer=None, parent=None):
        wdgt = temperatureForm(targetImage=targetImage, size=size, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        """
        pushButton = QPushButton('apply', parent=wdgt)
        hLay = QHBoxLayout()
        wdgt.setLayout(hLay)
        hLay.addWidget(pushButton)
        pushButton.clicked.connect(lambda: wdgt.execute())
        """
        return wdgt

    def __init__(self, targetImage=None, size=500, layer=None, parent=None):
        super(temperatureForm, self).__init__(parent=parent)
        self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(size, size)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.img = targetImage
        self.layer = layer
        self.defaultTemp = 6500
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)

        # options
        self.options = {'use Chromatic Adaptation': True}
        self.listWidget1 = optionsWidget(options=['use Photo Filter', 'use Chromatic Adaptation'], exclusive=True)
        self.listWidget1.select(self.listWidget1.items['use Chromatic Adaptation'])
        def onSelect1(item):
            self.options['use Chromatic Adaptation'] = item is self.listWidget1.items['use Chromatic Adaptation']
        self.listWidget1.onSelect = onSelect1
        l.addWidget(self.listWidget1)

        # opcity slider
        self.sliderTemp = QSlider(Qt.Horizontal)
        self.sliderTemp.setTickPosition(QSlider.TicksBelow)
        self.sliderTemp.setRange(1000, 9000)
        self.sliderTemp.setSingleStep(100)
        opacityLabel = QLabel()
        opacityLabel.setMaximumSize(150, 30)
        opacityLabel.setText("Color temperature")
        l.addWidget(opacityLabel)
        hl = QHBoxLayout()
        self.opacityValue = QLabel()
        font = self.opacityValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.opacityValue.setMinimumSize(w, h)
        self.opacityValue.setMaximumSize(w, h)

        #self.opacityValue.setText('6500 ')
        self.opacityValue.setStyleSheet("QLabel {background-color: white;}")
        hl.addWidget(self.opacityValue)
        hl.addWidget(self.sliderTemp)
        l.addLayout(hl)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom


        self.setLayout(l)

        # opacity value done event handler
        def f():
            self.sliderTemp.setEnabled(False)
            self.opacityValue.setText(str('%d ' % self.sliderTemp.value()))
            #QImg=applyTemperature(self.layer.inputImg(), self.sliderTemp.value(), 0.25)
            #self.layer.setImage(QImg)
            #self.img.onImageChanged()
            self.onUpdateTemperature(self.sliderTemp.value())
            self.sliderTemp.setEnabled(True)

        # opacity value changed event handler
        def g():
            self.opacityValue.setText(str('%d ' % self.sliderTemp.value()))
            #self.previewWindow.setPixmap()

        self.sliderTemp.valueChanged.connect(g)
        self.sliderTemp.sliderReleased.connect(f)

        self.sliderTemp.setValue(self.defaultTemp)


if __name__ == '__main__':
    T=4000.0
    r,g,b = bbTemperature2RGB(T)
    x,y = temperature2xyWP(T)
    L=0.7
    r1, g1, b1 = np.dot(sRGB2XYZInverse, np.array([L*x/y, L, L* (1.0 -x - y)/y]).T)
    r2, g2, b2 = rgbLinear2rgb(r1,g1,b1)
    print r,g,b
    print r2, g2, b2

