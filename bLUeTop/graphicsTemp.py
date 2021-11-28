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
from PySide2.QtGui import QFontMetrics, QColor
from PySide2.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout

from bLUeGui.colorCIE import sRGBWP
from bLUeGui.graphicsForm import baseForm
from bLUeTop.utils import optionsWidget, QbLUeSlider, QbLUeLabel, QbLUePushButton


class temperatureForm (baseForm):

    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = temperatureForm(axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.tempCorrection = 6500
        self.tintCorrection = 1.0
        self.filterColor = Qt.white
        self.defaultTemp = sRGBWP  # ref temperature D65
        self.defaultTint = 0

        # options
        optionList, optionNames = ['Color Filter', 'Photo Filter', 'Chromatic Adaptation'], ['Color Filter', 'Photo Filter', 'Chromatic Adaptation']
        self.listWidget1 = optionsWidget(options=optionList, optionNames=optionNames, exclusive=True,
                                         changed=self.dataChanged)
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.options = self.listWidget1.options
        # link to app color dialog
        self.colorChooser = self.parent().colorChooser
        # color viewer
        self.colorLabel = QLabel()
        self.colorLabel.setMaximumSize(50, 50)
        # color chooser button
        self.colorChooserBtn = QbLUePushButton('Select Filter Color')
        self.colorChooserBtn.clicked.connect(self.showColorChooser)

        # temp slider
        self.sliderTemp = QbLUeSlider(Qt.Horizontal)
        self.sliderTemp.setStyleSheet(QbLUeSlider.bLueSliderDefaultIColorStylesheet)
        self.sliderTemp.setRange(17, 100)  # 250)  # valid range for spline approximation is 1667..25000, cf. colorConv.temperature2xyWP
        self.sliderTemp.setSingleStep(1)

        self.tempLabel = QbLUeLabel()
        self.tempLabel.setMaximumSize(150, 30)
        self.tempLabel.setText("Filter Temperature")
        self.tempLabel.doubleClicked.connect(lambda: self.sliderTemp.setValue(self.temp2Slider(self.defaultTemp)))

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

        self.sliderTemp.valueChanged.connect(self.tempUpdate)
        self.sliderTemp.sliderReleased.connect(lambda: self.tempUpdate(self.sliderTemp.value()))
        self.sliderTint.valueChanged.connect(self.tintUpdate)
        self.sliderTint.sliderReleased.connect(lambda: self.tintUpdate(self.sliderTint.value()))

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        l.addWidget(QLabel('Filter Type'))
        l.addWidget(self.listWidget1)
        l.addStretch(1)
        hl2 = QHBoxLayout()
        hl2.addWidget(self.colorLabel)
        hl2.addWidget(self.colorChooserBtn)
        l.addLayout(hl2)
        l.addStretch(1)
        l.addWidget(self.tempLabel)
        hl = QHBoxLayout()
        hl.addWidget(self.tempValue)
        hl.addWidget(self.sliderTemp)
        l.addLayout(hl)
        l.addWidget(self.tintLabel)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.tintValue)
        hl1.addWidget(self.sliderTint)
        l.addLayout(hl1)
        l.addStretch(1)
        self.setLayout(l)
        self.adjustSize()
        self.setDefaults()
        self.setWhatsThis(
                        """<b> Color Filter</b> and <b>Photo Filter</b> use the multiply blending mode
                        to mimic a warming or cooling filter put in front of the camera lens. 
                        The luminosity of the resulting image is corrected.<br>
                        <b>Chromatic Adaptation</b> uses multipliers in the linear RGB
                        color space to adjust <b>temperature</b> and <b>tint</b>.
                        """
                        )  # end of setWhatsThis

    def enableSliders(self):
        for item in [self.colorLabel, self.colorChooserBtn]:
            item.setEnabled(self.options['Color Filter'])
        for item in [self.sliderTemp, self.tempLabel]:
            item.setEnabled(not self.options['Color Filter'])
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

        self.filterColor = self.colorChooser.currentColor()
        # set colorLabel background
        self.colorLabel.setAutoFillBackground(True)
        colorstr = ''.join('%02x'% i for i in self.filterColor.getRgb()[:3])
        self.colorLabel.setStyleSheet("background:#%s" % colorstr)

    def colorUpdate(self, color):
        """
        color Changed slot
        @param color:
        @type color: QColor
        """
        self.dataChanged.emit()

    def tempUpdate(self, value):
        """
        temp change slot
        @param value:
        @type value: int
        """
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
        self.sliderTemp.valueChanged.connect(self.tempUpdate)
        self.sliderTemp.sliderReleased.connect(lambda: self.tempUpdate(self.sliderTemp.value()))

    def tintUpdate(self, value):
        """
        tint change slot
        @param value:
        @type value: int
        """
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
        self.sliderTint.valueChanged.connect(self.tintUpdate)
        self.sliderTint.sliderReleased.connect(lambda: self.tintUpdate(self.sliderTint.value()))

    def updateLayer(self):
        """
        data changed slot
        """
        self.enableSliders()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def setFilterColor(self, color):
        """
        currentColorChanged slot
        @param color:
        @type color: QColor
        """
        self.filterColor = color
        self.colorLabel.setAutoFillBackground(True)
        colorstr = ''.join('%02x' % i for i in color.getRgb()[:3])
        self.colorLabel.setStyleSheet("background:#%s" % colorstr)

    def showColorChooser(self):
        self.colorChooser.show()
        try:
            self.colorChooser.currentColorChanged.disconnect()  # TODO conflict with other usages of app colorchooser
        except RuntimeError:
            pass
        self.colorChooser.setCurrentColor(self.filterColor)
        self.colorChooser.currentColorChanged.connect(self.setFilterColor)
        self.colorChooser.colorSelected.connect(self.colorUpdate)  # TODO conflict with other usages of app colorchooser

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

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [optionsWidget, QbLUeSlider]:
                d[a] = obj.__getstate__()
        c = self.filterColor
        d['filterColor'] = (c.red(), c.green(), c.blue(), c.alpha())
        return d

    def __setstate__(self, d):
        # prevent multiple updates
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        for name in d['state']:
            obj = getattr(self, name, None)
            if type(obj) in [optionsWidget, QbLUeSlider]:
                obj.__setstate__(d['state'][name])
        r, g, b, a = d['state']['filterColor']
        self.setFilterColor(QColor(r, g, b, a=a))
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()


