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
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QLabel, QHBoxLayout

from bLUeGui.graphicsForm import baseForm
from bLUeGui.memory import weakProxy
from bLUeGui.qrangeslider import QRangeSlider
from utils import optionsWidget

class blendFilterIndex:
    GRADUALBT, GRADUALTB, GRADUALNONE= range(3)

class blendFilterForm (baseForm):  # TODO updateLayer()
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = blendFilterForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(parent=parent)
        self.defaultFilterStart = 0
        self.defaultFilterEnd = 99
        self.filterStart = self.defaultFilterStart
        self.filterEnd = self.defaultFilterEnd

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # link back to image layer
        self.targetImage = weakProxy(targetImage)
        self.img = weakProxy(targetImage)
        self.layer = weakProxy(layer)
        self.kernelCategory = blendFilterIndex.GRADUALNONE  # TODO kernelCategory should be renamed as filterIndex 5/12/18
        # options
        optionList, optionNames = ['Gradual Top', 'Gradual Bottom'], ['Top To Bottom', 'Bottom To Top']
        filters = [blendFilterIndex.GRADUALTB, blendFilterIndex.GRADUALBT] #, blendFilterIndex.GRADUALNONE]
        filterDict = dict(zip(optionList, filters))

        self.listWidget1 = optionsWidget(options=optionList, optionNames=optionNames, exclusive=True, changed=self.dataChanged)
        # set initial selection to gradual top
        self.listWidget1.checkOption(optionList[0])

        rs = QRangeSlider()
        rs.setMaximumSize(16000, 10)

        rs.tail.setStyleSheet(
            'background: white; /*qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #222, stop:1 #888); margin 3px;*/')
        rs.handle.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 black, stop:1 white);')
        rs.head.setStyleSheet(
            'background: black; /*qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #999, stop:1 #222);*/')
        self.sliderFilterRange = rs
        frLabel = QLabel('Range')

        # filter range done event handler
        def frUpdate(start, end):
            if self.sliderFilterRange.isSliderDown() or (start == self.filterStart and end == self.filterEnd):
                return
            self.sliderFilterRange.startValueChanged.disconnect()
            self.sliderFilterRange.endValueChanged.disconnect()
            self.sliderFilterRange.rangeDone.disconnect()
            self.filterStart, self.filterEnd = self.sliderFilterRange.getRange()
            self.dataChanged.emit()
            self.sliderFilterRange.startValueChanged.connect(frUpdate)  # send new value as parameter
            self.sliderFilterRange.endValueChanged.connect(frUpdate)  # send new value as parameter
            self.sliderFilterRange.rangeDone.connect(frUpdate)

        self.sliderFilterRange.startValueChanged.connect(frUpdate)  # send new value as parameter
        self.sliderFilterRange.endValueChanged.connect(frUpdate)  # send new value as parameter
        self.sliderFilterRange.rangeDone.connect(frUpdate)

        # data changed event handler
        def updateLayer():
            for key in self.listWidget1.options:
                if self.listWidget1.options[key]:
                    self.kernelCategory = filterDict[key]
                    break
            self.layer.applyToStack()
            self.layer.parentImage.onImageChanged()

        self.dataChanged.connect(updateLayer)  # TODO 3/12/18 move to setDefaults
        # layout
        l = QVBoxLayout()
        l.addWidget(self.listWidget1)
        hl8 = QHBoxLayout()
        hl8.addWidget(frLabel)
        hl8.addWidget(self.sliderFilterRange)
        l.addLayout(hl8)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(l)

        self.setWhatsThis(
"""<b>Gradual neutral filter.</b><br> 
   It mimics the classical gradual gray filter often used by photographers to darken the sky.<br>
   To control the regions of maximum and minimum intensities use the Range slider.
"""
                        ) # end setWhatsThis

        self.setDefaults()

    def setDefaults(self, name1='Gradual Top', start=0, end=0, orientation=0):
        self.defaultFilterStart = start
        self.defaultFilterEnd = end
        self.sliderFilterRange.setRange(start, end)
        self.listWidget1.checkOption(name1)

    def writeToStream(self, outStream):
        layer = self.layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        outStream.writeQString(self.listWidget1.selectedItems()[0].text())
        outStream.writeFloat32(self.sliderRadius.value())
        outStream.writeFloat32(self.sliderAmount.value())
        return outStream

    def readFromStream(self, inStream):
        actionName = inStream.readQString()
        name = inStream.readQString()
        sel = inStream.readQString()
        radius = inStream.readFloat32()
        amount = inStream.readFloat32()
        for r in range(self.listWidget1.count()):
            currentItem = self.listWidget1.item(r)
            if currentItem.text() == sel:
                self.listWidget.select(currentItem)
        self.sliderRadius.setValue(radius)
        self.sliderAmount.setValue(amount)
        self.repaint()
        return inStream
