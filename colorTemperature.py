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

# conversion from sRGB (D65) to XYZ
# see http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
sRGB2XYZ = [[0.4124564,  0.3575761,  0.1804375],
            [0.2126729,  0.7151522,  0.0721750],
            [0.0193339,  0.1191920,  0.9503041]]

sRGB2XYZInverse = [[3.2404542, -1.5371385, -0.4985314],
                   [-0.9692660, 1.8760108,  0.0415560],
                    [0.0556434, -0.2040259, 1.0572252]]

################
#
################

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
        blue = temperature - 10
        blue = 138.5177312231 * np.log(blue) - 305.0447927307
        blue = min(max(0,blue), 255)

    return red, green, blue

def applyTemperature(qImg, temperature, coeff, version=2):
    """

    :param qImg:
    :param temperature:
    :param coeff:
    :return:
    """
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
        res = np.tensordot(buf[:,:,::-1].astype(np.float64), M, axes= (-1,-1))
        res=np.clip(res, 0, 255)
        buf[:, :, ::-1] = res
    return img

def temperature2xyWP(T):
    """
    Calculates the CIE chromaticity coordinates xc, yc
    of white point from temperature (cubic spline approximation).
    see http://en.wikipedia.org/wiki/Planckian_locus#Approximation

    :param T: temperature in Kelvin
    :return: xc, yc
    """

    if T <= 4000:
        xc = -0.2661239 *(10**9) / (T**3) - 0.2343580 *10**6 / (T**2) + 0.8776956 * 10**3 / T + 0.179910  # 1667<T<4000
    else:
        xc = - 3.0258469 *(10**9) / (T**3) + 2.1070379 *10**6 / (T**2) + 0.2226347 * 10**3 / T + 0.240390 # 4000<T<25000

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

        # opcity slider
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
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
            QImg=applyTemperature(self.layer.inputImg, self.sliderTemp.value(), 0.25)
            self.layer.setImage(QImg)
            self.img.onImageChanged()
            self.sliderTemp.setEnabled(True)

        # opacity value changed event handler
        def g():
            self.opacityValue.setText(str('%d ' % self.sliderTemp.value()))
            #self.previewWindow.setPixmap()

        self.sliderTemp.valueChanged.connect(g)
        self.sliderTemp.sliderReleased.connect(f)

        self.sliderTemp.setValue(self.defaultTemp)


if __name__ == '__main__':
    T=5000.0
    r,g,b = bbTemperature2RGB(T)
    x,y = temperature2xyWP(T)
    L=200
    r1, g1, b1 = np.dot(np.array(sRGB2XYZInverse), np.array([L*x/y, L, L* (1.0 -x - y)/y]).T)
    print 'ici',  r,g,b ,r1, g1,b1

