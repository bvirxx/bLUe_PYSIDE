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

from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout

from bLUeGui.colorCIE import sRGBWP
from bLUeGui.graphicsForm import baseForm
from utils import optionsWidget, QbLUeSlider, QbLUeLabel


class temperatureForm (baseForm):

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = temperatureForm(axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.tempCorrection = 6500
        self.tintCorrection = 1.0
        self.defaultTemp = sRGBWP  # ref temperature D65
        self.defaultTint = 0

        # options
        optionList, optionNames = ['Photo Filter', 'Chromatic Adaptation'], ['Photo Filter', 'Chromatic Adaptation']
        self.listWidget1 = optionsWidget(options=optionList, optionNames=optionNames, exclusive=True,
                                         changed=lambda: self.dataChanged.emit())
        self.listWidget1.checkOption(self.listWidget1.intNames[1])
        self.options = self.listWidget1.options

        # temp slider
        self.sliderTemp = QbLUeSlider(Qt.Horizontal)
        self.sliderTemp.setStyleSheet(QbLUeSlider.bLueSliderDefaultIColorStylesheet)
        self.sliderTemp.setRange(17, 100)  # 250)  # valid range for spline approximation is 1667..25000, cf. colorConv.temperature2xyWP
        self.sliderTemp.setSingleStep(1)

        tempLabel = QbLUeLabel()
        tempLabel.setMaximumSize(150, 30)
        tempLabel.setText("Filter Temperature")
        tempLabel.doubleClicked.connect(lambda: self.sliderTemp.setValue(self.temp2Slider(self.defaultTemp)))

        self.tempValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("00000")
        h = metrics.height()
        self.tempValue.setMinimumSize(w, h)
        self.tempValue.setMaximumSize(w, h)
        self.tempValue.setText(str("{:d}".format(self.sliderTemp2User(self.sliderTemp.value()))))

        # tint slider
        self.sliderTint = QbLUeSlider(Qt.Horizontal)
        self.sliderTint.setStyleSheet(QbLUeSlider.bLueSliderDefaultMGColorStylesheet)
        self.sliderTint.setRange(0, 100)  # 250) # valid range for spline approximation is 1667..25000, cf. colorConv.temperature2xyWP
        self.sliderTint.setSingleStep(1)

        self.tintLabel = QbLUeLabel()
        self.tintLabel.setMaximumSize(150, 30)
        self.tintLabel.setText("Tint")
        self.tintLabel.doubleClicked.connect(lambda: self.sliderTint.setValue(self.tint2Slider(self.defaultTint)))

        self.tintValue = QLabel()
        font = self.tintValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("0000")
        h = metrics.height()
        self.tintValue.setMinimumSize(w, h)
        self.tintValue.setMaximumSize(w, h)
        self.tintValue.setText(str("{:d}".format(self.sliderTint2User(self.sliderTint.value()))))

        # temp change slot
        def tempUpdate(value):
            self.tempValue.setText(str("{:d}".format(self.sliderTemp2User(value))))
            # move not yet terminated or values not modified
            if self.sliderTemp.isSliderDown() or self.slider2Temp(value) == self.tempCorrection:
                return
            try:
                self.sliderTemp.valueChanged.disconnect()
                self.sliderTemp.sliderReleased.disconnect()
            except RuntimeError:
                pass
            self.tempCorrection = self.slider2Temp(value)
            self.dataChanged.emit()
            self.sliderTemp.valueChanged.connect(tempUpdate)
            self.sliderTemp.sliderReleased.connect(lambda: tempUpdate(self.sliderTemp.value()))

        # tint change slot
        def tintUpdate(value):
            self.tintValue.setText(str("{:d}".format(self.sliderTint2User(value))))
            # move not yet terminated or values not modified
            if self.sliderTint.isSliderDown() or self.slider2Tint(value) == self.tintCorrection:
                return
            try:
                self.sliderTint.valueChanged.disconnect()
                self.sliderTint.sliderReleased.disconnect()
            except RuntimeError:
                pass
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
        l.addWidget(self.tintLabel)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.tintValue)
        hl1.addWidget(self.sliderTint)
        l.addLayout(hl1)
        self.setLayout(l)
        self.adjustSize()
        self.setDefaults()
        self.setWhatsThis(
                        """<b>Color Temperature</b><br>
                        <b>Photo Filter</b> uses the multiply blending mode to mimic a warming or cooling filter
                        put in front of the camera lens. The luminosity of the resulting image is corrected.<br>
                        <b>Chromatic Adaptation</b> uses multipliers in the linear sRGB
                        color space to adjust <b>temperature</b> and <b>tint</b>.
                        """
                        )  # end of setWhatsThis

    def enableSliders(self):
        self.sliderTemp.setEnabled(True)
        for item in [self.sliderTint, self.tintLabel, self.tintValue]:
            item.setEnabled(self.options['Chromatic Adaptation'])

    def setDefaults(self):
        # prevent multiple updates
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.listWidget1.unCheckAll()
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.enableSliders()
        self.sliderTemp.setValue(round(self.temp2Slider(self.tempCorrection)))
        self.sliderTint.setValue(round(self.tint2Slider(self.defaultTint)))
        self.dataChanged.connect(self.updateLayer)

    def updateLayer(self):
        """
        data changed slot
        """
        self.enableSliders()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    @staticmethod
    def slider2Temp(v):
        return v * 100

    @staticmethod
    def temp2Slider(v):
        return int(v / 100)

    @staticmethod
    def sliderTemp2User(v):
        return v * 100

    @staticmethod
    def slider2Tint(v):
        return (v - 50) / 50

    @staticmethod
    def tint2Slider(v):
        return int((1.0 + v) * 50.0)

    @staticmethod
    def sliderTint2User(v):
        return int((v - 50) / 5.0)
    """
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
    """

