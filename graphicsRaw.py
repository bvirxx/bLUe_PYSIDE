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
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QGraphicsView, QSizePolicy, QVBoxLayout, QLabel, QHBoxLayout, QSlider

from colorConv import xyWP2temperature, sRGB2XYZ, temperature2xyWP, sRGB2XYZInverse, Bradford, BradfordInverse, \
    RGBMultipliers, xy2TemperatureAndTint, RGBMultipliers2Temperature
from utils import optionsWidget, UDict


class rawForm (QGraphicsView):
    """
    GUI for postprocessing of raw files
    """
    defaultExpCorrection = 0.0
    DefaultExpStep = 0.1
    defaultTempCorrection = 6500
    DefaultTempStep = 100
    defaultTintCorrection = 0.0
    DefaultTintStep = 1
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
        self.expCorrection = self.defaultExpCorrection
        super(rawForm, self).__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.layer = layer
        rawpyObj = layer.parentImage.rawImage
        ashot = rawpyObj.camera_whitebalance
        d1 = rawpyObj.daylight_whitebalance

        print ("rgbtotemp", RGBMultipliers2Temperature(515,256,369))
        # rgb_xyz_matrix : Camera RGB - XYZ conversion matrix. This matrix is constant (different for different models).
        # Last row is zero for RGB cameras and non-zero for different color models (CMYG and so on).
        # type:	ndarray of shape (4,3)

        # rgb_xyz_matrix is libraw cam_xyz
        # camera_whitebalance is libraw cam_mul
        # daylight_whitebalance is libraw pre_mul

        # color_matrix Color matrix, read from file for some cameras, calculated for others, type ndarray of shape (3,4)
        """
        In general you need 2 steps: (1) you need to convert temp/tint values to/from chromaticities (e.g., xy coordinates) 
        and (2) you need to associate these white point values to linear camera coordinates. 

        To do the 1st task you can look at the DNG sdk, dng_temperature.cpp. 

        For the 2nd task you generally need scene-referred colorimetry -- i.e., 
        do some measurements with an actual camera, such as shoot a neutral sample 
        (flat spectrum) under a known illuminant. That would tell you the camera neutral 
        (the actual linear RGB camera coordinates corresponding to neutral for that illuminant) as well 
        as the white point (since the illuminant is known), thereby giving you a correspondence between the two.
        """
        self.tempCorrection, self.tintCorrection = RGBMultipliers2Temperature(ashot[0], ashot[1], ashot[2]) #xy2TemperatureAndTint(x, y)
        #############
        cam_RGB2XYZ = rawpyObj.rgb_xyz_matrix[:3,:]

        self.multipliers = RGBMultipliers(self.tempCorrection, self.tintCorrection)
        # options
        optionList = ['Auto Brightness', 'Auto Scale']
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
        self.sliderTemp.setRange(int(2000 / self.DefaultTempStep), int(25000 / self.DefaultTempStep))
        self.sliderTemp.setSingleStep(1)

        tempLabel = QLabel()
        tempLabel.setFixedSize(40, 20)
        tempLabel.setText("Temp")

        self.tempValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000000")
        h = metrics.height()
        self.tempValue.setMinimumSize(w, h)
        self.tempValue.setMaximumSize(w, h)
        self.tempValue.setStyleSheet("QLabel {background-color: white;}")

        # temp done event handler
        def tempUpdate():
            self.sliderTemp.setEnabled(False)
            self.tempValue.setText(str("{:.0f}".format(self.sliderTemp.value() * self.DefaultTempStep)))
            # rawpy: expCorrection range is -2.0...3.0 boiling down to exp_shift range 2**(-2)=0.25...2**3=8.0
            self.tempCorrection = self.sliderTemp.value() * self.DefaultTempStep
            self.multipliers = RGBMultipliers(self.tempCorrection, self.tintCorrection)
            self.dataChanged.emit()
            self.sliderTemp.setEnabled(True)
        # temp value changed event handler
        def sliderTempUpdate():
            self.tempValue.setText(str("{:.0f}".format(self.sliderTemp.value() * self.DefaultTempStep)))

        self.sliderTemp.valueChanged.connect(sliderTempUpdate)
        self.sliderTemp.sliderReleased.connect(tempUpdate)
        # slider Temp init
        self.sliderTemp.setValue(round(self.tempCorrection / self.DefaultTempStep))
        sliderTempUpdate()

        self.sliderTint = QSlider(Qt.Horizontal)
        self.sliderTint.setTickPosition(QSlider.TicksBelow)
        self.sliderTint.setRange(int(0.0 / self.DefaultTintStep), int(20. / self.DefaultTintStep))
        self.sliderTint.setSingleStep(1)

        tintLabel = QLabel()
        tintLabel.setFixedSize(40, 20)
        tintLabel.setText("Tint")

        self.tintValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("10000")
        h = metrics.height()
        self.tintValue.setMinimumSize(w, h)
        self.tintValue.setMaximumSize(w, h)
        self.tintValue.setStyleSheet("QLabel {background-color: white;}")

        # tint done event handler
        def tintUpdate():
            self.sliderTint.setEnabled(False)
            self.tintValue.setText(str("{:+.0f}".format(self.sliderTint.value() * self.DefaultTintStep - 10)))
            self.tintCorrection = (self.sliderTint.value() * self.DefaultTintStep - 10)
            self.multipliers = RGBMultipliers(self.tempCorrection, self.tintCorrection)
            self.dataChanged.emit()
            self.sliderTint.setEnabled(True)

        # tint value changed event handler
        def sliderTintUpdate():
            self.tintValue.setText(str("{:+.0f}".format(self.sliderTint.value() * self.DefaultTintStep - 10)))

        self.sliderTint.valueChanged.connect(sliderTintUpdate)
        self.sliderTint.sliderReleased.connect(tintUpdate)
        # slider Tint init
        self.sliderTint.setValue(round(10 + self.tintCorrection / self.DefaultTintStep))
        sliderTintUpdate()

        # exp slider
        self.sliderExp = QSlider(Qt.Horizontal)
        self.sliderExp.setTickPosition(QSlider.TicksBelow)
        self.sliderExp.setRange(int(0.0 / self.DefaultExpStep), int(5.0 / self.DefaultExpStep))
        self.sliderExp.setSingleStep(1)

        expLabel = QLabel()
        expLabel.setFixedSize(40, 20)
        expLabel.setText("Exp")

        self.expValue = QLabel()
        font = self.expValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("10000")
        h = metrics.height()
        self.expValue.setMinimumSize(w, h)
        self.expValue.setMaximumSize(w, h)
        self.expValue.setStyleSheet("QLabel {background-color: white;}")

        # exp done event handler
        def expUpdate():
            self.sliderExp.setEnabled(False)
            self.expValue.setText(str("{:+.1f}".format(self.sliderExp.value() * self.DefaultExpStep - 2.0)))
            # rawpy: expCorrection range is -2.0...3.0 boiling down to exp_shift range 2**(-2)=0.25...2**3=8.0
            self.expCorrection = self.sliderExp.value() * self.DefaultExpStep - 2.0
            self.dataChanged.emit()
            self.sliderExp.setEnabled(True)
        # exp value changed event handler
        def sliderExpUpdate():
            self.expValue.setText(str("{:+.1f}".format(self.sliderExp.value() * self.DefaultExpStep - 2.0)))
        self.sliderExp.valueChanged.connect(sliderExpUpdate)
        self.sliderExp.sliderReleased.connect(expUpdate)
        # slider init
        self.sliderExp.setValue(int((2.0 + self.defaultExpCorrection) / self.DefaultExpStep))
        sliderExpUpdate()
        # data changed event handler
        def updateLayer():
            useUserWB = self.listWidget2.options["User WB"]
            self.sliderTemp.setEnabled(useUserWB)
            self.sliderTint.setEnabled(useUserWB)
            self.layer.applyToStack()
            self.layer.parentImage.onImageChanged()
        self.dataChanged.connect(updateLayer)

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
        l.addLayout(hl2)
        l.addLayout(hl3)
        l.addLayout(hl1)
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