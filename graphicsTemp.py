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
import weakref

from PySide2 import QtCore
from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QLabel, QHBoxLayout

from bLUeGui.colorCIE import sRGBWP
from bLUeGui.graphicsForm import baseForm
from bLUeGui.memory import weakProxy
from utils import optionsWidget, QbLUeSlider, QbLUeLabel


class temperatureForm (baseForm):

    # dataChanged = QtCore.Signal()

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        wdgt = temperatureForm(axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        self.tempCorrection = 6500
        self.tintCorrection = 1.0
        self.setStyleSheet('QRangeSlider * {border: 0px; padding: 0px; margin: 0px}')
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        # link back to image layer
        self.layer = weakProxy(layer)
        """
        # using weak ref for back links
        if type(layer) in weakref.ProxyTypes:
            self.layer = layer
        else:
            self.layer = weakref.proxy(layer)
        """
        self.defaultTemp = sRGBWP  # ref temperature D65
        self.defaultTint = 0

        # options
        optionList, optionNames = ['Photo Filter', 'Chromatic Adaptation'], ['Photo Filter', 'Chromatic Adaptation']
        self.listWidget1 = optionsWidget(options=optionList, optionNames=optionNames, exclusive=True, changed=lambda: self.dataChanged.emit())
        self.listWidget1.checkOption(self.listWidget1.intNames[1])
        self.options = self.listWidget1.options

        # temp slider
        self.sliderTemp = QbLUeSlider(Qt.Horizontal)
        self.sliderTemp.setStyleSheet(QbLUeSlider.bLueSliderDefaultIColorStylesheet)
        self.sliderTemp.setRange(17, 100) # 250) # valid range for spline approximation is 1667..25000, cf. colorConv.temperature2xyWP
        self.sliderTemp.setSingleStep(1)

        tempLabel = QbLUeLabel()
        tempLabel.setMaximumSize(150, 30)
        tempLabel.setText("Color temperature")
        tempLabel.doubleClicked.connect(lambda: self.sliderTemp.setValue(self.temp2Slider(self.defaultTemp)))

        self.tempValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("0000")
        h = metrics.height()
        self.tempValue.setMinimumSize(w, h)
        self.tempValue.setMaximumSize(w, h)
        self.tempValue.setText(str("{:d}".format(self.sliderTemp2User(self.sliderTemp.value()))))

        # tint slider
        self.sliderTint = QbLUeSlider(Qt.Horizontal)
        self.sliderTint.setStyleSheet(QbLUeSlider.bLueSliderDefaultMGColorStylesheet)
        self.sliderTint.setRange(0, 100)  # 250) # valid range for spline approximation is 1667..25000, cf. colorConv.temperature2xyWP
        self.sliderTint.setSingleStep(1)

        tintLabel = QbLUeLabel()
        tintLabel.setMaximumSize(150, 30)
        tintLabel.setText("Tint")
        tintLabel.doubleClicked.connect(lambda: self.sliderTint.setValue(self.tint2Slider(self.defaultTint)))

        self.tintValue = QLabel()
        font = self.tintValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("0000")
        h = metrics.height()
        self.tintValue.setMinimumSize(w, h)
        self.tintValue.setMaximumSize(w, h)
        self.tintValue.setText(str("{:d}".format(self.sliderTint2User(self.sliderTint.value()))))

        # temp change event handler
        def tempUpdate(value):
            self.tempValue.setText(str("{:d}".format(self.sliderTemp2User(value))))
            # move not yet terminated or values not modified
            if self.sliderTemp.isSliderDown() or self.slider2Temp(value) == self.tempCorrection :
                return
            self.sliderTemp.valueChanged.disconnect()
            self.sliderTemp.sliderReleased.disconnect()
            self.tempCorrection = self.slider2Temp(value)
            self.dataChanged.emit()
            self.sliderTemp.valueChanged.connect(tempUpdate)
            self.sliderTemp.sliderReleased.connect(lambda: tempUpdate(self.sliderTemp.value()))

        # tint change event handler
        def tintUpdate(value):
            self.tintValue.setText(str("{:d}".format(self.sliderTint2User(value))))
            # move not yet terminated or values not modified
            if self.sliderTint.isSliderDown() or  self.slider2Tint(value) == self.tintCorrection:
                return
            self.sliderTint.valueChanged.disconnect()
            self.sliderTint.sliderReleased.disconnect()
            self.tintCorrection = self.slider2Tint(value)
            self.dataChanged.emit()
            self.sliderTint.valueChanged.connect(tintUpdate)
            self.sliderTint.sliderReleased.connect(lambda: tintUpdate(self.sliderTint.value()))

        self.sliderTemp.valueChanged.connect(tempUpdate)
        self.sliderTemp.sliderReleased.connect(lambda: tempUpdate(self.sliderTemp.value()))
        self.sliderTint.valueChanged.connect(tintUpdate)
        self.sliderTint.sliderReleased.connect(lambda: tintUpdate(self.sliderTint.value()))

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        l.addWidget(self.listWidget1)
        l.addWidget(tempLabel)
        hl = QHBoxLayout()
        hl.addWidget(self.tempValue)
        hl.addWidget(self.sliderTemp)
        l.addLayout(hl)
        l.addWidget(tintLabel)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.tintValue)
        hl1.addWidget(self.sliderTint)
        l.addLayout(hl1)
        self.setLayout(l)
        self.adjustSize()
        self.dataChanged.connect(self.updateLayer)
        self.setStyleSheet("QListWidget, QLabel {font : 7pt;}")
        self.setDefaults()
        self.setWhatsThis(
"""<b>Color Temperature</b><br>
<b>Photo Filter</b> uses the multiply blending mode to mimic a color filter
in front of the camera lens.<br>
<b>Chromatic Adaptation</b> uses multipliers in the linear sRGB
color space to adjust <b>temperature</b> and <b>tint</b>.
"""
                        )  # end of setWhatsThis

    def enableSliders(self):
        self.sliderTemp.setEnabled(True)
        self.sliderTint.setEnabled(self.options['Chromatic Adaptation'])

    def setDefaults(self):
        self.listWidget1.unCheckAll()
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.enableSliders()
        self.sliderTemp.setValue(round(self.temp2Slider(self.tempCorrection)))
        self.sliderTint.setValue(round(self.tint2Slider(self.defaultTint)))
        # self.dataChanged.emit() # removed 30/10/18

    def updateLayer(self):
        """
        data changed event handler.
        """
        self.enableSliders()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def slider2Temp(self, v):
        return v * 100

    def temp2Slider(self, v):
        return int(v /100)

    def sliderTemp2User(selfself, v):
        return v * 100

    def slider2Tint(self, v):
        return (v - 50) / 50

    def tint2Slider(self, v):
        return int( (1.0 + v) *50.0)

    def sliderTint2User(selfself, v):
        return int((v - 50) / 5.0)

    def writeToStream(self, outStream):
        layer = self.layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        outStream.writeQString(self.listWidget1.selectedItems()[0].text())
        outStream.writeInt32(self.sliderTemp.value()*100)
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
        self.sliderTemp.setValue(temp//100)
        self.update()
        return inStream

