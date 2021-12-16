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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout
from PySide6.QtGui import QFontMetrics

from bLUeGui.graphicsForm import baseForm
from bLUeCore.kernel import filterIndex
from bLUeTop.utils import optionsWidget, QbLUeSlider


class filterForm(baseForm):
    defaultRadius = 10
    defaultTone = 100.0
    defaultAmount = 50.0

    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = filterForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        # connect layer selectionChanged signal
        self.layer.selectionChanged.sig.connect(self.updateLayer)
        self.kernelCategory = filterIndex.UNSHARP
        # options
        self.optionList = ['Unsharp Mask', 'Sharpen', 'Gaussian Blur', 'Surface Blur']
        filters = [filterIndex.UNSHARP, filterIndex.SHARPEN, filterIndex.BLUR1, filterIndex.SURFACEBLUR]
        self.filterDict = dict(zip(self.optionList, filters))  # filters is not a dict: don't use UDict here

        self.listWidget1 = optionsWidget(options=self.optionList, exclusive=True, changed=self.dataChanged)
        # set initial selection to unsharp mask
        self.listWidget1.checkOption(self.optionList[0])

        # sliders
        self.sliderRadius = QbLUeSlider(Qt.Horizontal)
        self.sliderRadius.setRange(1, 50)
        self.sliderRadius.setSingleStep(1)
        self.radiusLabel = QLabel()
        self.radiusLabel.setMaximumSize(150, 30)
        self.radiusLabel.setText("Radius")

        self.radiusValue = QLabel()
        font = self.radiusValue.font()
        metrics = QFontMetrics(font)
        w = metrics.horizontalAdvance("1000 ")
        h = metrics.height()
        self.radiusValue.setMinimumSize(w, h)
        self.radiusValue.setMaximumSize(w, h)

        self.sliderAmount = QbLUeSlider(Qt.Horizontal)
        self.sliderAmount.setRange(0, 100)
        self.sliderAmount.setSingleStep(1)
        self.amountLabel = QLabel()
        self.amountLabel.setMaximumSize(150, 30)
        self.amountLabel.setText("Amount")
        self.amountValue = QLabel()
        font = self.radiusValue.font()
        metrics = QFontMetrics(font)
        w = metrics.horizontalAdvance("1000 ")
        h = metrics.height()
        self.amountValue.setMinimumSize(w, h)
        self.amountValue.setMaximumSize(w, h)

        self.toneValue = QLabel()
        self.toneLabel = QLabel()
        self.toneLabel.setMaximumSize(150, 30)
        self.toneLabel.setText("Sigma")
        self.sliderTone = QbLUeSlider(Qt.Horizontal)
        self.sliderTone.setRange(0, 100)
        self.sliderTone.setSingleStep(1)
        font = self.radiusValue.font()
        metrics = QFontMetrics(font)
        w = metrics.horizontalAdvance("1000 ")
        h = metrics.height()
        self.toneValue.setMinimumSize(w, h)
        self.toneValue.setMaximumSize(w, h)

        # value change/done slot
        def formUpdate():
            self.radiusValue.setText(str('%d ' % self.sliderRadius.value()))
            self.amountValue.setText(str('%d ' % self.sliderAmount.value()))
            self.toneValue.setText(str('%d ' % self.sliderTone.value()))
            if self.sliderRadius.isSliderDown() or self.sliderAmount.isSliderDown() or self.sliderTone.isSliderDown():
                return
            try:
                for slider in [self.sliderRadius, self.sliderAmount, self.sliderTone]:
                    slider.valueChanged.disconnect()
                    slider.sliderReleased.disconnect()
            except RuntimeError:
                pass
            self.tone = self.sliderTone.value()
            self.radius = self.sliderRadius.value()
            self.amount = self.sliderAmount.value()
            self.dataChanged.emit()
            for slider in [self.sliderRadius, self.sliderAmount, self.sliderTone]:
                slider.valueChanged.connect(formUpdate)
                slider.sliderReleased.connect(formUpdate)

        for slider in [self.sliderRadius, self.sliderAmount, self.sliderTone]:
            slider.valueChanged.connect(formUpdate)
            slider.sliderReleased.connect(formUpdate)

        # layout
        l = QVBoxLayout()
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

        self.setDefaults()
        self.setWhatsThis(
            """
               <b>Unsharp Mask</b> and <b>Sharpen Mask</b> are used to sharpen an image.
               Unsharp Mask usually gives best results.<br>
               <b>Gaussian Blur</b> and <b>Surface Blur</b> are used to blur an image.<br>
               In contrast to Gaussian Blur, Surface Blur preserves edges and reduces noise,
               but it may be slow.<br>
               It is possible to <b>limit the effect of a filter to a rectangular region of the image</b> by
               drawing a selection rectangle on the layer with the marquee (rectangle) tool.<br>
               Ctrl Click <b>clears the selection</b><br>
               
            """
        )  # end setWhatsThis

    def setDefaults(self):
        self.enableSliders()
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.sliderRadius.setValue(self.defaultRadius)
        self.sliderAmount.setValue(self.defaultAmount)
        self.sliderTone.setValue(self.defaultTone)
        self.dataChanged.connect(self.updateLayer)

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

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [optionsWidget, QbLUeSlider]:
                d[a] = obj.__getstate__()
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
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()
