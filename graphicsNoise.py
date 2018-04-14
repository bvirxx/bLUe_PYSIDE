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
from PySide2.QtWidgets import QGraphicsView, QSizePolicy, QVBoxLayout, QSlider, QLabel, QHBoxLayout

from utils import optionsWidget, QbLUeSlider


class noiseForm (QGraphicsView):
    dataChanged = QtCore.Signal(bool)
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        wdgt = noiseForm(axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None): # TODO 01/12/17 remove param targetImage
        super(noiseForm, self).__init__(parent=parent)
        #self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        #self.img = targetImage
        self.layer = layer

        # options
        optionList= ['Wavelets', 'Bilateral', 'NLMeans']
        self.listWidget1 = optionsWidget(options=optionList, exclusive=True, changed=lambda: self.dataChanged.emit(True))
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.options = self.listWidget1.option

        # threshold slider
        self.sliderThr = QbLUeSlider(Qt.Horizontal)
        self.sliderThr.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)
        self.sliderThr.setTickPosition(QSlider.TicksBelow)
        self.sliderThr.setRange(0,10)
        self.sliderThr.setSingleStep(1)

        self.sliderThr.valueChanged.connect(self.thrUpdate)
        self.sliderThr.sliderReleased.connect(lambda: self.thrUpdate(self.sliderThr.value()))  # signal has no parameter)

        self.thrLabel = QLabel()
        self.thrLabel.setMaximumSize(150, 30)
        self.thrLabel.setText("level")

        self.thrValue = QLabel()
        font = self.thrValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("0000")
        h = metrics.height()
        self.thrValue.setMinimumSize(w, h)
        self.thrValue.setMaximumSize(w, h)
        self.thrValue.setText(str("{:.0f}".format(self.slider2Thr(self.sliderThr.value()))))
        self.dataChanged.connect(self.updateLayer)
        self.setStyleSheet("QListWidget, QLabel {font : 7pt;}")

        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
        l.addWidget(self.listWidget1)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.thrLabel)
        hl1.addWidget(self.thrValue)
        hl1.addWidget(self.sliderThr)
        l.addLayout(hl1)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        #l.setContentsMargins(10, 10, 10, 10)  # left, top, right, bottom
        self.setLayout(l)
        self.adjustSize()
        self.setDefaults()

    def setDefaults(self):
        self.listWidget1.unCheckAll()
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.noiseCorrection = 0
        self.dataChanged.disconnect()
        self.sliderThr.setValue(round(self.thr2Slider(self.noiseCorrection)))
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit(True)

    def updateLayer(self, invalidate):
        """
        data changed event handler
        """
        if invalidate:
            self.layer.postProcessCache = None
        #self.enableSliders()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def slider2Thr(self, v):
        return v

    def thr2Slider(self, t):
        return t

    def thrUpdate(self, value):
        self.thrValue.setText(str("{:.0f}".format(self.slider2Thr(self.sliderThr.value()))))
        # move not yet terminated or value not modified
        if self.sliderThr.isSliderDown() or self.slider2Thr(value) == self.noiseCorrection:
            return
        self.sliderThr.valueChanged.disconnect()
        self.sliderThr.sliderReleased.disconnect()
        self.noiseCorrection = self.slider2Thr(self.sliderThr.value())
        self.thrValue.setText(str("{:+d}".format(self.noiseCorrection)))
        self.dataChanged.emit(False)
        self.sliderThr.valueChanged.connect(self.thrUpdate)  # send new value as parameter
        self.sliderThr.sliderReleased.connect(lambda: self.thrUpdate(self.sliderThr.value()))  # signal has no parameter

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

