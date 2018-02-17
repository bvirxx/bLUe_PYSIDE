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
from PySide2 import QtCore

from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics, QFont
from PySide2.QtWidgets import QGraphicsView, QSizePolicy, QVBoxLayout, QLabel, QHBoxLayout, QSlider

from colorConv import xyWP2temperature, sRGB2XYZ, temperature2xyWP, sRGB2XYZInverse, Bradford, BradfordInverse, \
    temperatureAndTint2RGBMultipliers, xy2TemperatureAndTint, RGBMultipliers2TemperatureAndTint, temperatureAndTint2xy
from utils import optionsWidget, UDict

# cf https://github.com/LibRaw/LibRaw/blob/master/src/libraw_cxx.cpp

class rawForm (QGraphicsView):
    """
    GUI for postprocessing of raw files
    """
    dataChanged = QtCore.Signal()
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        wdgt = rawForm(axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        """
        pushButton = QPushButton('apply', parent=wdgt)
        hLay = QHBoxLayout()
        wdgt.setLayout(hLay)
        hLay.addWidget(pushButton)
        pushButton.clicked.connect(lambda: wdgt.execute())
        """
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        self.expCorrection = 0.0
        self.contCorrection = 0.0
        self.noiseCorrection = 1.0
        self.satCorrection = 0.5
        super(rawForm, self).__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize+200)  # +200 to prevent scroll bars in list Widgets
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.layer = layer
        # get rawpy object
        rawpyObj = layer.parentImage.rawImage
        # get multipliers
        self.rawMultipliers = rawpyObj.camera_whitebalance
        daylight = rawpyObj.daylight_whitebalance
        #self.rawMultipliers = [daylight[i] / multipliers[i] for i in range(3)] + [daylight[1] / multipliers[1]]
        # get Camera RGB - XYZ conversion matrix.
        # From rawpy doc, this matrix is constant for each camera model
        # Last row is zero for RGB cameras and non-zero for different color models (CMYG and so on) : type ndarray of shape (4,3)
        rgb_xyz_matrix = rawpyObj.rgb_xyz_matrix[:3,:]
        rgb_xyz_matrix_inverse = np.linalg.inv(rgb_xyz_matrix)
        # Color_matrix, read from file for some cameras, calculated for others, type ndarray of shape (3,4), seems to be 0.
        # color_matrix = rawpyObj.color_matrix
        #################################
        # Libraw correspondances:
        # rgb_xyz_matrix is libraw cam_xyz
        # camera_whitebalance is libraw cam_mul
        # daylight_whitebalance is libraw pre_mul
        ##################################
        multipliers = [daylight[i] / self.rawMultipliers[i] for i in range(3)]
        self.tempCorrection, self.tintCorrection = RGBMultipliers2TemperatureAndTint(*multipliers, rgb_xyz_matrix_inverse)
        # options
        optionList = ['Auto Brightness', 'Preserve Highlights', 'Auto Scale']
        self.listWidget1 = optionsWidget(options=optionList, exclusive=False, changed=self.dataChanged)
        self.listWidget1.checkOption(optionList[0])
        optionList = ['Auto WB', 'Camera WB', 'User WB']
        self.listWidget2 = optionsWidget(options=optionList, exclusive=True, changed=self.dataChanged)
        self.listWidget2.checkOption(optionList[1])
        self.options = UDict(self.listWidget1.options, self.listWidget2.options)
        text = self.listWidget2.item(1).text()

        # WB sliders
        self.sliderTemp = QSlider(Qt.Horizontal)
        self.sliderTemp.setTickPosition(QSlider.TicksBelow)
        self.sliderTemp.setRange(0,130)
        def slider2Temp(v):
            return 2000 + v*v
        def temp2Slider(T):
            return np.sqrt(T - 2000)
        self.sliderTemp.setSingleStep(1)

        tempLabel = QLabel()
        #tempLabel.setFixedSize(40, 20)
        tempLabel.setText("Temp")

        self.tempValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000000")
        h = metrics.height()
        self.tempValue.setMinimumSize(w, h)
        self.tempValue.setMaximumSize(w, h)
        self.tempValue.setStyleSheet("QLabel {color : gray;}")

        # temp done event handler
        def tempUpdate():
            self.sliderTemp.setEnabled(False)
            self.tempCorrection = slider2Temp(self.sliderTemp.value())
            self.tempValue.setText(str("{:.0f}".format(self.tempCorrection)))
            multipliers = temperatureAndTint2RGBMultipliers(self.tempCorrection, self.tintCorrection, rgb_xyz_matrix_inverse)
            self.rawMultipliers = [daylight[i] / multipliers[i] for i in range(3)] + [daylight[1] / multipliers[1]]
            m = min(self.rawMultipliers[:3])
            self.rawMultipliers = [self.rawMultipliers[i] / m for i in range(4)]
            self.dataChanged.emit()
            self.sliderTemp.setEnabled(True)
        # temp value changed event handler
        def sliderTempUpdate():
            self.tempValue.setText(str("{:.0f}".format(slider2Temp(self.sliderTemp.value()))))

        self.sliderTemp.valueChanged.connect(sliderTempUpdate)
        self.sliderTemp.sliderReleased.connect(tempUpdate)
        # slider Temp init
        self.sliderTemp.setValue(round(temp2Slider(self.tempCorrection)))
        sliderTempUpdate()
        self.sliderTemp.setEnabled(False)  # initially we use camera WB

        self.sliderTint = QSlider(Qt.Horizontal)
        self.sliderTint.setTickPosition(QSlider.TicksBelow)
        self.sliderTint.setRange(0, 100)
        def slider2Tint(v):
            return 0.2 + 0.025 * v
        def tint2Slider(t):
            return (t - 0.2) / 0.025
        def slider2User(v):
            return ((slider2Tint(v) - 1)*100)
        self.sliderTint.setSingleStep(1)

        tintLabel = QLabel()
        #tintLabel.setFixedSize(40, 20)
        tintLabel.setText("Tint")

        self.tintValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("10000")
        h = metrics.height()
        self.tintValue.setMinimumSize(w, h)
        self.tintValue.setMaximumSize(w, h)
        self.tintValue.setStyleSheet("QLabel {color : gray;}")

        # tint done event handler
        def tintUpdate():
            self.sliderTint.setEnabled(False)
            #self.tintValue.setText(str("{:+.0f}".format((self.sliderTint.value() - 1.0/self.DefaultTintStep))))
            #self.tintCorrection = self.sliderTint.value() * self.DefaultTintStep
            self.tintCorrection = slider2Tint(self.sliderTint.value())
            self.tintValue.setText(str("{:+.0f}".format(slider2User(self.sliderTint.value()))))
            multipliers = temperatureAndTint2RGBMultipliers(self.tempCorrection, self.tintCorrection, rgb_xyz_matrix_inverse)
            self.rawMultipliers = [daylight[i] / multipliers[i] for i in range(3)] + [daylight[1] / multipliers[1]]
            m = min(self.rawMultipliers[:3])
            self.rawMultipliers = [self.rawMultipliers[i] / m for i in range(4)]
            self.dataChanged.emit()
            self.sliderTint.setEnabled(True)

        # tint value changed event handler
        def sliderTintUpdate():
            self.tintValue.setText(str("{:+.0f}".format(slider2User(self.sliderTint.value()))))

        self.sliderTint.valueChanged.connect(sliderTintUpdate)
        self.sliderTint.sliderReleased.connect(tintUpdate)
        # slider Tint init
        self.sliderTint.setValue(round(tint2Slider(self.tintCorrection)))
        sliderTintUpdate()
        self.sliderTint.setEnabled(False)  # initially we use camera WB

        # exp slider
        self.sliderExp = QSlider(Qt.Horizontal)
        self.sliderExp.setTickPosition(QSlider.TicksBelow)
        self.sliderExp.setRange(0, 100)
        def slider2Exp(v):
            return v / 20.0 - 2.0
        def exp2Slider(e):
            return round((e + 2.0) * 20.0)
        self.sliderExp.setSingleStep(1)

        expLabel = QLabel()
        #expLabel.setFixedSize(40, 20)
        expLabel.setText("Exp")

        self.expValue = QLabel()
        font = self.expValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("10000")
        h = metrics.height()
        self.expValue.setMinimumSize(w, h)
        self.expValue.setMaximumSize(w, h)
        self.expValue.setStyleSheet("QLabel {color : gray;}")

        # exp done event handler
        def expUpdate():
            self.sliderExp.setEnabled(False)
            # rawpy: expCorrection range is -2.0...3.0 boiling down to exp_shift range 2**(-2)=0.25...2**3=8.0
            self.expCorrection = slider2Exp(self.sliderExp.value())
            self.expValue.setText(str("{:+.1f}".format(self.expCorrection)))
            self.dataChanged.emit()
            self.sliderExp.setEnabled(True)
        # exp value changed event handler
        def sliderExpUpdate():
            self.expValue.setText(str("{:+.1f}".format(slider2Exp(self.sliderExp.value()))))

        self.sliderExp.valueChanged.connect(sliderExpUpdate)
        self.sliderExp.sliderReleased.connect(expUpdate)
        # exp init
        self.sliderExp.setValue(exp2Slider(0.0))
        self.sliderExp.setEnabled(False)  # initially  we use auto brightness
        sliderExpUpdate()

        # contrast slider
        self.sliderCont = QSlider(Qt.Horizontal)
        self.sliderCont.setTickPosition(QSlider.TicksBelow)
        self.sliderCont.setRange(0, 20)

        def slider2Cont(v):
            return v

        def cont2Slider(e):
            return e

        self.sliderCont.setSingleStep(1)

        contLabel = QLabel()
        #contLabel.setFixedSize(55, 20)
        contLabel.setText("Contrast")

        self.contValue = QLabel()
        font = self.contValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("10000")
        h = metrics.height()
        self.contValue.setMinimumSize(w, h)
        self.contValue.setMaximumSize(w, h)
        #self.contValue.setStyleSheet("QLabel {color : gray;}")

        # cont done event handler
        def contUpdate():
            self.sliderCont.setEnabled(False)
            self.contCorrection = slider2Cont(self.sliderCont.value())
            self.contValue.setText(str("{:+d}".format(self.contCorrection)))
            self.dataChanged.emit()
            self.sliderCont.setEnabled(True)

        # cont value changed event handler
        def sliderContUpdate():
            self.contValue.setText(str("{:+d}".format(slider2Cont(self.sliderCont.value()))))

        self.sliderCont.valueChanged.connect(sliderContUpdate)
        self.sliderCont.sliderReleased.connect(contUpdate)
        # cont init
        self.sliderCont.setValue(cont2Slider(0.0))
        #self.sliderCont.setEnabled(False)
        sliderContUpdate()

        # noise reduction slider
        self.sliderNoise = QSlider(Qt.Horizontal)
        self.sliderNoise.setTickPosition(QSlider.TicksBelow)
        self.sliderNoise.setRange(0, 20)

        def slider2Noise(v):
            return v

        def noise2Slider(e):
            return e

        self.sliderNoise.setSingleStep(1)

        noiseLabel = QLabel()
        #noiseLabel.setFixedSize(110, 20)
        noiseLabel.setText("Noise Reduction")

        self.noiseValue = QLabel()
        font = self.noiseValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("10000")
        h = metrics.height()
        self.noiseValue.setMinimumSize(w, h)
        self.noiseValue.setMaximumSize(w, h)

        # self.contValue.setStyleSheet("QLabel {color : gray;}")

        # noise done event handler
        def noiseUpdate():
            self.sliderNoise.setEnabled(False)
            self.noiseCorrection = slider2Noise(self.sliderNoise.value())
            self.noiseValue.setText(str("{:+d}".format(slider2Noise(self.sliderNoise.value()))))
            self.dataChanged.emit()
            self.sliderNoise.setEnabled(True)

        # noise value changed event handler
        def sliderNoiseUpdate():
            self.noiseValue.setText(str("{:+d}".format(slider2Noise(self.sliderNoise.value()))))

        self.sliderNoise.valueChanged.connect(sliderNoiseUpdate)
        self.sliderNoise.sliderReleased.connect(noiseUpdate)
        # noise init
        self.sliderNoise.setValue(noise2Slider(1.0))
        # self.sliderCont.setEnabled(False)
        sliderNoiseUpdate()

        # saturation slider
        self.sliderSat = QSlider(Qt.Horizontal)
        self.sliderSat.setTickPosition(QSlider.TicksBelow)
        self.sliderSat.setRange(0, 100)

        def slider2Sat(v):
            return v - 50 #np.math.pow(10, v / 50)

        def sat2Slider(e):
            return e + 50 #50 * np.math.log10(e)

        self.sliderSat.setSingleStep(1)

        satLabel = QLabel()
        satLabel.setText("Sat")

        self.satValue = QLabel()
        font = self.satValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("10000")
        h = metrics.height()
        self.satValue.setMinimumSize(w, h)
        self.satValue.setMaximumSize(w, h)
        # self.contValue.setStyleSheet("QLabel {color : gray;}")

        # sat done event handler
        def satUpdate():
            self.sliderSat.setEnabled(False)
            self.satCorrection = slider2Sat(self.sliderSat.value())
            self.satValue.setText(str("{:+d}".format(slider2Sat(self.sliderSat.value()))))
            self.dataChanged.emit()
            self.sliderSat.setEnabled(True)

        # sat value changed event handler
        def sliderSatUpdate():
            self.satValue.setText(str("{:+d}".format(slider2Sat(self.sliderSat.value()))))

        self.sliderSat.valueChanged.connect(sliderSatUpdate)
        self.sliderSat.sliderReleased.connect(satUpdate)
        # slider sat init
        self.sliderSat.setValue(sat2Slider(1.0))
        # self.sliderCont.setEnabled(False)
        sliderSatUpdate()

        # data changed event handler
        def updateLayer():
            useUserWB = self.listWidget2.options["User WB"]
            useUserExp = not self.listWidget1.options["Auto Brightness"]
            self.sliderTemp.setEnabled(useUserWB)
            self.sliderTint.setEnabled(useUserWB)
            self.sliderExp.setEnabled(useUserExp)
            color = "black" if useUserWB else "gray"
            self.tempValue.setStyleSheet("color: %s;" % color)
            self.tintValue.setStyleSheet("color: %s;" % color)
            color = "black" if useUserExp else "gray"
            self.expValue.setStyleSheet("color: %s;" % color)
            self.layer.applyToStack()
            self.layer.parentImage.onImageChanged()
        self.dataChanged.connect(updateLayer)
        self.setStyleSheet("QListWidget, QLabel {font : 7pt;}")
        #layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
        hl1 = QHBoxLayout()
        hl1.addWidget(expLabel)
        hl1.addWidget(self.expValue)
        hl1.addWidget(self.sliderExp)
        l.addWidget(self.listWidget1)
        l.addWidget(self.listWidget2)
        hl2 = QHBoxLayout()
        hl2.addWidget(tempLabel)
        hl2.addWidget(self.tempValue)
        hl2.addWidget(self.sliderTemp)
        hl3 = QHBoxLayout()
        hl3.addWidget(tintLabel)
        hl3.addWidget(self.tintValue)
        hl3.addWidget(self.sliderTint)
        hl4 = QHBoxLayout()
        hl4.addWidget(contLabel)
        hl4.addWidget(self.contValue)
        hl4.addWidget(self.sliderCont)

        hl7 = QHBoxLayout()
        hl7.addWidget(satLabel)
        hl7.addWidget(self.satValue)
        hl7.addWidget(self.sliderSat)
        hl5 = QHBoxLayout()
        hl5.addWidget(self.noiseValue)
        hl5.addWidget(self.sliderNoise)

        hl6 = QHBoxLayout()
        hl6.setAlignment(Qt.AlignHCenter)
        hl6.addWidget(noiseLabel)
        l.addLayout(hl2)
        l.addLayout(hl3)
        l.addLayout(hl1)
        l.addLayout(hl4)
        l.addLayout(hl7)
        l.addLayout(hl6)
        l.addLayout(hl5)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        l.addStretch(1)
        self.setLayout(l)
        self.adjustSize()

        def writeToStream(self, outStream):
            layer = self.layer
            outStream.writeQString(layer.actionName)
            outStream.writeQString(layer.name)
            outStream.writeQString(self.listWidget1.selectedItems()[0].text())
            outStream.writeInt32(self.sliderExp.value())
            return outStream

        def readFromStream(self, inStream):
            actionName = inStream.readQString()
            name = inStream.readQString()
            sel = inStream.readQString()
            temp = inStream.readInt32()
            for r in range(self.listWidget1.count()):
                currentItem = self.listWidget1.item(r)
                if currentItem.text() == sel:
                    self.listWidget.select(currentItem)
            self.sliderExp.setValue(temp)
            self.update()
            return inStream