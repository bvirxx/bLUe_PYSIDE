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
from PySide2.QtWidgets import QGraphicsView, QSizePolicy, QVBoxLayout, QSlider, QLabel, QHBoxLayout

from colorConv import sRGBWP
from utils import optionsWidget


class temperatureForm (QGraphicsView):
    @classmethod
    def getNewWindow(cls, targetImage=None, size=500, layer=None, parent=None, mainForm=None):
        wdgt = temperatureForm(targetImage=targetImage, size=size, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, size=500, layer=None, parent=None, mainForm=None):
        super(temperatureForm, self).__init__(parent=parent)
        self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(size, size)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.img = targetImage
        self.layer = layer
        self.defaultTemp = sRGBWP  # ref temperature D65
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
        # options
        options = ['Photo Filter', 'Chromatic Adaptation']
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        self.options = self.listWidget1.options
        # set initial selection
        self.listWidget1.checkOption(options[1])

        l.addWidget(self.listWidget1)

        # temp slider
        self.sliderTemp = QSlider(Qt.Horizontal)
        self.sliderTemp.setTickPosition(QSlider.TicksBelow)
        self.sliderTemp.setRange(17, 250)  # valid range for spline approximation is 1667..25000, cf. colorConv.temperature2xyWP
        self.sliderTemp.setSingleStep(1)

        tempLabel = QLabel()
        tempLabel.setMaximumSize(150, 30)
        tempLabel.setText("Color temperature")
        l.addWidget(tempLabel)
        hl = QHBoxLayout()
        self.tempValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("000000")
        h = metrics.height()
        self.tempValue.setMinimumSize(w, h)
        self.tempValue.setMaximumSize(w, h)
        self.tempValue.setStyleSheet("QLabel {background-color: white;}")
        hl.addWidget(self.tempValue)
        hl.addWidget(self.sliderTemp)
        l.addLayout(hl)
        l.addStretch(1)
        #l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        l.setContentsMargins(10, 10, 10, 10)  # left, top, right, bottom
        self.setLayout(l)
        self.adjustSize()
        # temp done event handler
        def f():
            self.sliderTemp.setEnabled(False)
            temp = self.sliderTemp.value()*100
            self.tempValue.setText(str('%d ' % temp))
            self.onUpdateTemperature(self.layer, temp)
            self.sliderTemp.setEnabled(True)
        # temp value changed event handler
        def g():
            self.tempValue.setText(str('%d ' % (self.sliderTemp.value()*100)))

        def h(lay, temperature):
            lay.temperature = temperature
            lay.applyToStack()
            mainForm.label.img.onImageChanged()

        self.onUpdateTemperature = h

        self.sliderTemp.valueChanged.connect(g)
        self.sliderTemp.sliderReleased.connect(f)
        self.listWidget1.onSelect = lambda item: f()

        self.sliderTemp.setValue(self.defaultTemp//100)

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

