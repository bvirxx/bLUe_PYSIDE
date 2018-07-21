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
from PySide2 import QtCore
from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QLabel, QHBoxLayout, QWidget

from colorConv import sRGBWP
from utils import optionsWidget, QbLUeSlider


class temperatureForm (QWidget): #(QGraphicsView): TODO modified 25/06/18 validate

    dataChanged = QtCore.Signal()

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        wdgt = temperatureForm(axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        self.tempCorrection = 6500
        self.setStyleSheet('QRangeSlider * {border: 0px; padding: 0px; margin: 0px}')
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.layer = layer

        self.defaultTemp = sRGBWP  # ref temperature D65

        # options
        optionList, optionNames = ['Photo Filter', 'Chromatic Adaptation'], ['Photo Filter', 'Chromatic Adaptation']
        self.listWidget1 = optionsWidget(options=optionList, optionNames=optionNames, exclusive=True, changed=lambda: self.dataChanged.emit())
        self.listWidget1.checkOption(self.listWidget1.intNames[1])
        self.options = self.listWidget1.options

        # temp slider
        self.sliderTemp = QbLUeSlider(Qt.Horizontal)
        self.sliderTemp.setStyleSheet(QbLUeSlider.bLueSliderDefaultColorStylesheet)
        self.sliderTemp.setRange(17, 250)  # valid range for spline approximation is 1667..25000, cf. colorConv.temperature2xyWP
        self.sliderTemp.setSingleStep(1)

        tempLabel = QLabel()
        tempLabel.setMaximumSize(150, 30)
        tempLabel.setText("Color temperature")

        self.tempValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("000000")
        h = metrics.height()
        self.tempValue.setMinimumSize(w, h)
        self.tempValue.setMaximumSize(w, h)
        self.tempValue.setText(str("{:d}".format(self.sliderTemp2User(self.sliderTemp.value()))))

        # temp change event handler
        def tempUpdate(value):
            self.tempValue.setText(str("{:d}".format(self.sliderTemp2User(value))))
            # move not yet terminated or value not modified
            if self.sliderTemp.isSliderDown() or self.slider2Temp(value) == self.tempCorrection:
                return
            self.sliderTemp.valueChanged.disconnect()
            self.sliderTemp.sliderReleased.disconnect()
            self.tempCorrection = self.slider2Temp(value)
            self.dataChanged.emit()
            self.sliderTemp.valueChanged.connect(tempUpdate)
            self.sliderTemp.sliderReleased.connect(lambda: tempUpdate(self.sliderTemp.value()))
        self.sliderTemp.valueChanged.connect(tempUpdate)
        self.sliderTemp.sliderReleased.connect(lambda: tempUpdate(self.sliderTemp.value()))

        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
        l.addWidget(self.listWidget1)
        l.addWidget(tempLabel)
        hl = QHBoxLayout()
        hl.addWidget(self.tempValue)
        hl.addWidget(self.sliderTemp)
        l.addLayout(hl)
        self.setLayout(l)
        self.adjustSize()
        self.dataChanged.connect(self.updateLayer)
        self.setStyleSheet("QListWidget, QLabel {font : 7pt;}")
        self.setDefaults()
        self.setWhatsThis(
"""<b>Color Temperature</b><br>
<b>Photo Filter</b> uses the multiply blending mode to mimic a color filter
in front of the camera lens.<br>
<b>Chromatic Adaptation</b> uses a linear transformation in the XYZ color space.<br>
"""
                        )  # end of setWhatsThis

    def enableSliders(self):
        self.sliderTemp.setEnabled(True)

    def setDefaults(self):
        self.listWidget1.unCheckAll()
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.enableSliders()
        self.sliderTemp.setValue(round(self.temp2Slider(self.tempCorrection)))
        self.dataChanged.emit()

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

