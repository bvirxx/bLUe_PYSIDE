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
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QSlider, QLabel, QHBoxLayout
from PySide2.QtGui import QFontMetrics

from bLUeGui.graphicsSpline import baseForm
from bLUeCore.kernel import filterIndex, getKernel
from utils import optionsWidget, QbLUeSlider


class filterForm (baseForm):
    dataChanged = QtCore.Signal()
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        wdgt = filterForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        defaultRadius = 10
        defaultTone = 100.0
        defaultAmount = 50.0
        self.radius = defaultRadius
        self.tone = defaultTone
        self.amount = defaultAmount
        self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.img = targetImage
        # link back to image layer
        # using weak ref for back links
        if type(layer) in weakref.ProxyTypes:
            self.layer = layer
        else:
            self.layer = weakref.proxy(layer)
        self.mainForm = mainForm
        self.kernelCategory = filterIndex.UNSHARP

        # options
        self.optionList = ['Unsharp Mask', 'Sharpen', 'Gaussian Blur', 'Surface Blur']
        filters = [ filterIndex.UNSHARP, filterIndex.SHARPEN, filterIndex.BLUR1, filterIndex.SURFACEBLUR]
        self.filterDict = dict(zip(self.optionList, filters))

        self.listWidget1 = optionsWidget(options=self.optionList, exclusive=True, changed=self.dataChanged)
        # set initial selection to unsharp mask
        item = self.listWidget1.checkOption(self.optionList[0])

        # sliders
        self.sliderRadius = QbLUeSlider(Qt.Horizontal)
        #self.sliderRadius.setTickPosition(QSlider.TicksBelow)
        self.sliderRadius.setRange(1, 50)
        self.sliderRadius.setSingleStep(1)
        self.radiusLabel = QLabel()
        self.radiusLabel.setMaximumSize(150, 30)
        self.radiusLabel.setText("Radius")

        self.radiusValue = QLabel()
        font = self.radiusValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.radiusValue.setMinimumSize(w, h)
        self.radiusValue.setMaximumSize(w, h)

        self.sliderAmount = QbLUeSlider(Qt.Horizontal)
        #self.sliderAmount.setTickPosition(QSlider.TicksBelow)
        self.sliderAmount.setRange(0, 100)
        self.sliderAmount.setSingleStep(1)
        self.amountLabel = QLabel()
        self.amountLabel.setMaximumSize(150, 30)
        self.amountLabel.setText("Amount")
        self.amountValue = QLabel()
        font = self.radiusValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.amountValue.setMinimumSize(w, h)
        self.amountValue.setMaximumSize(w, h)

        self.toneValue = QLabel()
        self.toneLabel = QLabel()
        self.toneLabel.setMaximumSize(150, 30)
        self.toneLabel.setText("Sigma")
        self.sliderTone = QbLUeSlider(Qt.Horizontal)
        #self.sliderTone.setTickPosition(QSlider.TicksBelow)
        self.sliderTone.setRange(0, 100)
        self.sliderTone.setSingleStep(1)
        font = self.radiusValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.toneValue.setMinimumSize(w, h)
        self.toneValue.setMaximumSize(w, h)

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
        l.addWidget(self.listWidget1)
        hl = QHBoxLayout()
        hl.addWidget(self.radiusLabel)
        hl.addWidget(self.radiusValue)
        hl.addWidget(self.sliderRadius)
        l.addLayout(hl)
        hl = QHBoxLayout()
        hl.addWidget(self.amountLabel)
        hl.addWidget(self.amountValue)
        hl.addWidget(self.sliderAmount)
        l.addLayout(hl)
        hl = QHBoxLayout()
        hl.addWidget(self.toneLabel)
        hl.addWidget(self.toneValue)
        hl.addWidget(self.sliderTone)
        l.addLayout(hl)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(l)

        # value changed event handler
        def sliderUpdate():
            self.radiusValue.setText(str('%d ' % self.sliderRadius.value()))
            self.amountValue.setText(str('%d ' % self.sliderAmount.value()))
            self.toneValue.setText(str('%d ' % self.sliderTone.value()))
        # value done event handler
        def formUpdate():
            sR, sA, sT = self.sliderRadius.isEnabled(), self.sliderAmount.isEnabled(), self.sliderTone.isEnabled()
            self.sliderRadius.setEnabled(False)
            self.sliderAmount.setEnabled(False)
            self.sliderTone.setEnabled(False)
            self.tone = self.sliderTone.value()
            self.radius = self.sliderRadius.value()
            self.amount = self.sliderAmount.value()
            sliderUpdate()
            self.dataChanged.emit()
            self.sliderRadius.setEnabled(sR)
            self.sliderAmount.setEnabled(sA)
            self.sliderTone.setEnabled(sT)

        self.sliderRadius.valueChanged.connect(sliderUpdate)
        self.sliderRadius.sliderReleased.connect(formUpdate)
        self.sliderAmount.valueChanged.connect(sliderUpdate)
        self.sliderAmount.sliderReleased.connect(formUpdate)
        self.sliderTone.valueChanged.connect(sliderUpdate)
        self.sliderTone.sliderReleased.connect(formUpdate)

        self.dataChanged.connect(self.updateLayer)
        # init
        self.sliderRadius.setValue(defaultRadius)
        self.sliderAmount.setValue(defaultAmount)
        self.sliderTone.setValue(defaultTone)
        self.enableSliders()
        sliderUpdate()
        self.setWhatsThis(
"""
   <b>Unsharp Mask</b> and <b>Sharpen Mask</b> are used to sharpen an image.<br>
   <b>Gaussian Blur</b> and <b>Surface Blur</b> are used to blur an image.<br>
   In contrast to Gaussian Blur, Surface Blur preserves edges and reduces noise.<br>
   It is possible to <b>limit the effect of a filter to a rectangular region of the image</b> by
   drawing a selection rectangle on the layer with the marquee tool.<br>
   Ctrl-Click to <b>clear the selection</b><br>
   
"""
                        )  # end setWhatsThis

    def updateLayer(self):
        """
        dataChanged Slot
        """
        self.enableSliders()
        for key in self.listWidget1.options:
            if self.listWidget1.options[key]:
                self.kernelCategory = self.filterDict[key]
                break
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def enableSliders(self):
        opt = self.listWidget1.options
        useRadius = opt[self.optionList[0]] or opt[self.optionList[2]] or opt[self.optionList[3]]
        useAmount = opt[self.optionList[0]] or opt[self.optionList[2]]
        useTone = opt[self.optionList[3]]
        self.sliderRadius.setEnabled(useRadius)
        self.sliderAmount.setEnabled(useAmount)
        self.sliderTone.setEnabled(useTone)
        self.radiusValue.setEnabled(self.sliderRadius.isEnabled())
        self.amountValue.setEnabled(self.sliderAmount.isEnabled())
        self.toneValue.setEnabled(self.sliderTone.isEnabled())
        self.radiusLabel.setEnabled(self.sliderRadius.isEnabled())
        self.amountLabel.setEnabled(self.sliderAmount.isEnabled())
        self.toneLabel.setEnabled(self.sliderTone.isEnabled())

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

