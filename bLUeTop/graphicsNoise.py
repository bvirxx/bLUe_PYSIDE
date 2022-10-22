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
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QVBoxLayout, QSlider, QLabel, QHBoxLayout

from bLUeGui.graphicsForm import baseForm
from bLUeTop.utils import optionsWidget, QbLUeSlider


class noiseForm(baseForm):
    noiseCorrection = 0

    @staticmethod
    def slider2Thr(v):
        return v

    @staticmethod
    def thr2Slider(t):
        return t

    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = noiseForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.layer.selectionChanged.sig.connect(self.updateLayer)
        # options
        optionList = ['Wavelets', 'Bilateral', 'NLMeans']
        self.listWidget1 = optionsWidget(options=optionList, exclusive=True, changed=self.dataChanged)
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.options = self.listWidget1.options

        # threshold slider
        self.sliderThr = QbLUeSlider(Qt.Horizontal)
        self.sliderThr.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)
        self.sliderThr.setTickPosition(QSlider.TicksBelow)
        self.sliderThr.setRange(0, 10)
        self.sliderThr.setSingleStep(1)

        self.sliderThr.valueChanged.connect(self.thrUpdate)
        self.sliderThr.sliderReleased.connect(
            lambda: self.thrUpdate(self.sliderThr.value()))  # signal has no parameter)

        self.thrLabel = QLabel()
        self.thrLabel.setMaximumSize(150, 30)
        self.thrLabel.setText("level")

        self.thrValue = QLabel()
        font = self.thrValue.font()
        metrics = QFontMetrics(font)
        w = metrics.horizontalAdvance("0000")
        h = metrics.height()
        self.thrValue.setMinimumSize(w, h)
        self.thrValue.setMaximumSize(w, h)
        self.thrValue.setText(str("{:.0f}".format(self.slider2Thr(self.sliderThr.value()))))

        # layout
        l = QVBoxLayout()
        l.addWidget(self.listWidget1)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.thrLabel)
        hl1.addWidget(self.thrValue)
        hl1.addWidget(self.sliderThr)
        l.addLayout(hl1)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(l)
        self.adjustSize()

        self.setDefaults()
        self.setWhatsThis(
            """<b>Noise Reduction</b><br>
               <b>Bilateral Filtering</b> is the fastest method.<br>
               <b>NLMeans</b> (Non Local Means) and <b>Wavelets</b> are slower,
               but they usually give better results.<br>
               To <b>limit the action of any method to a 
               rectangular region of the image</b>
               draw a selection rectangle on the layer with the marquee tool.<br>
               Ctrl-Click to <b>clear the selection</b><br>
            """
        )  # end of setWhatsThis

    def setDefaults(self):
        self.listWidget1.unCheckAll()
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.noiseCorrection = 0
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.sliderThr.setValue(round(self.thr2Slider(self.noiseCorrection)))
        self.dataChanged.connect(self.updateLayer)

    def updateLayer(self):
        """
        data changed slot
        """
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def thrUpdate(self, value):
        """
        Slidet thr slot.

        :param value:
        :type value:
        :return:
        :rtype:
        """
        self.thrValue.setText(str("{:.0f}".format(self.slider2Thr(self.sliderThr.value()))))
        # move not yet terminated or value unchanged
        if self.sliderThr.isSliderDown() or self.slider2Thr(value) == self.noiseCorrection:
            return
        try:
            self.sliderThr.valueChanged.disconnect()
            self.sliderThr.sliderReleased.disconnect()
        except RuntimeError:
            pass
        self.noiseCorrection = self.slider2Thr(self.sliderThr.value())
        self.dataChanged.emit()
        self.sliderThr.valueChanged.connect(self.thrUpdate)  # send new value as parameter
        self.sliderThr.sliderReleased.connect(lambda: self.thrUpdate(self.sliderThr.value()))  # signal has no parameter

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
