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
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QGraphicsView
from PySide2.QtGui import QFontMetrics

from kernel import filterIndex, getKernel
from utils import optionsWidget


class filterForm (QGraphicsView):
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        wdgt = filterForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super(filterForm, self).__init__(parent=parent)
        defaultRadius = 1
        defaultColorSigma2 = 100.0
        defaultPositionSigma2 = 100.0
        defaultAmount = 50.0
        self.radius = defaultRadius
        self.colorSigma2 = defaultColorSigma2
        self.positionSigma2 = defaultPositionSigma2
        self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.img = targetImage
        self.layer = layer
        self.mainForm = mainForm
        self.onUpdateFilter = lambda *args: 0
        self.kernel = getKernel(filterIndex.UNSHARP)
        self.kernelCategory = filterIndex.UNSHARP
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)

        # options
        options = ['Unsharp Mask', 'Sharpen', 'Gaussian Blur', 'Surface Blur']
        filters = [ filterIndex.UNSHARP, filterIndex.SHARPEN, filterIndex.BLUR1, filterIndex.BLUR2, filterIndex.SURFACEBLUR]
        filterDict = dict(zip(options, filters))
        self.options={}
        for op in options:
            self.options[op] = False
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        sel = options[0]
        self.listWidget1.select(self.listWidget1.items[sel])
        self.options[sel] = True

        # selection event handler
        def onSelect1(item):
            for key in self.options:
                self.options[key] = item is self.listWidget1.items[key]
                if self.options[key]:
                    selkey = key
            self.kernelCategory = filterDict[selkey]
            self.sliderRadius.setEnabled(selkey==options[0] or selkey==options[2] )
            self.radiusValue.setEnabled(self.sliderRadius.isEnabled())
            self.sliderAmount.setEnabled(selkey==options[0])
            self.amountValue.setEnabled(self.sliderAmount.isEnabled())
            self.radius = self.sliderRadius.value()
            self.colorSigma2 = self.sliderTone.value()
            #self.positionSigma2 = self.sliderPositionSigma2.value()
            self.kernel = getKernel(self.kernelCategory, self.radius, self.sliderAmount.value())
            self.onUpdateFilter(self.kernelCategory, self.sliderRadius.value(), self.sliderAmount.value())

        self.listWidget1.onSelect = onSelect1

        # sliders
        self.sliderRadius = QSlider(Qt.Horizontal)
        self.sliderRadius.setTickPosition(QSlider.TicksBelow)
        self.sliderRadius.setRange(1, 100)
        self.sliderRadius.setSingleStep(1)
        radiusLabel = QLabel()
        radiusLabel.setMaximumSize(150, 30)
        radiusLabel.setText("Radius")

        self.radiusValue = QLabel()
        font = self.radiusValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.radiusValue.setMinimumSize(w, h)
        self.radiusValue.setMaximumSize(w, h)
        self.radiusValue.setStyleSheet("QLabel {background-color: white;}")

        self.sliderAmount = QSlider(Qt.Horizontal)
        self.sliderAmount.setTickPosition(QSlider.TicksBelow)
        self.sliderAmount.setRange(0, 100)
        self.sliderAmount.setSingleStep(1)
        amountLabel = QLabel()
        amountLabel.setMaximumSize(150, 30)
        amountLabel.setText("Amount")
        self.amountValue = QLabel()
        font = self.radiusValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.amountValue.setMinimumSize(w, h)
        self.amountValue.setMaximumSize(w, h)
        self.amountValue.setStyleSheet("QLabel {background-color: white;}")

        self.toneValue = QLabel()
        toneLabel = QLabel()
        toneLabel.setMaximumSize(150, 30)
        toneLabel.setText("Tone")
        self.sliderTone = QSlider(Qt.Horizontal)
        self.sliderTone.setTickPosition(QSlider.TicksBelow)
        self.sliderTone.setRange(0, 100)
        self.sliderTone.setSingleStep(1)
        font = self.radiusValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.toneValue.setMinimumSize(w, h)
        self.toneValue.setMaximumSize(w, h)
        self.toneValue.setStyleSheet("QLabel {background-color: white;}")

        hl = QHBoxLayout()
        hl.addWidget(radiusLabel)
        hl.addWidget(self.radiusValue)
        hl.addWidget(self.sliderRadius)
        l.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(amountLabel)
        hl.addWidget(self.amountValue)
        hl.addWidget(self.sliderAmount)
        l.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(toneLabel)
        hl.addWidget(self.toneValue)
        hl.addWidget(self.sliderTone)
        l.addLayout(hl)

        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom

        self.setLayout(l)

        # value done event handler
        def f():
            self.sliderRadius.setEnabled(False)
            self.radiusValue.setText(str('%d ' % self.sliderRadius.value()))
            self.amountValue.setText(str('%d ' % self.sliderAmount.value()))
            self.kernel = getKernel(self.kernelCategory, self.sliderRadius.value(), self.sliderAmount.value())
            self.onUpdateFilter(self.kernelCategory, self.sliderRadius.value(), self.sliderAmount.value())
            self.sliderRadius.setEnabled(True)

        # value changed event handler
        def g():
            self.radiusValue.setText(str('%d ' % self.sliderRadius.value()))
            self.amountValue.setText(str('%d ' % self.sliderAmount.value()))

        self.sliderRadius.valueChanged.connect(g)
        self.sliderRadius.sliderReleased.connect(f)
        self.sliderAmount.valueChanged.connect(g)
        self.sliderAmount.sliderReleased.connect(f)

        self.sliderRadius.setValue(defaultRadius)
        self.sliderAmount.setValue(defaultAmount)
        self.radiusValue.setText(str('%d ' % self.sliderRadius.value()))
        self.amountValue.setText(str('%d ' % self.sliderAmount.value()))

        # set initial selection to unsharp mask
        item = self.listWidget1.items[options[0]]
        item.setCheckState(Qt.Checked)
        self.listWidget1.select(item)
        l.addWidget(self.listWidget1)

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

