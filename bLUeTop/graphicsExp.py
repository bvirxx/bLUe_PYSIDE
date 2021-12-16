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
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout

from bLUeGui.graphicsForm import baseForm
from bLUeTop.utils import QbLUeSlider, QbLUeLabel


class ExpForm(baseForm):
    defaultExpCorrection = 0.0
    defaultStep = 0.1

    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = ExpForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        # options
        self.options = None
        # exposure slider
        self.sliderExp = QbLUeSlider(Qt.Horizontal)
        self.sliderExp.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)
        self.sliderExp.setRange(-20, 20)
        self.sliderExp.setSingleStep(1)

        expLabel = QbLUeLabel()
        expLabel.setMaximumSize(150, 30)
        expLabel.setText("Exposure Correction")
        expLabel.doubleClicked.connect(lambda: self.sliderExp.setValue(self.defaultExpCorrection))

        self.expValue = QbLUeLabel()
        font = self.expValue.font()
        metrics = QFontMetrics(font)
        w = metrics.horizontalAdvance("1000 ")
        h = metrics.height()
        self.expValue.setMinimumSize(w, h)
        self.expValue.setMaximumSize(w, h)

        # exp change/released slot
        def f():
            self.expValue.setText(str("{:+.1f}".format(self.sliderExp.value() * self.defaultStep)))
            if self.sliderExp.isSliderDown() or (self.expCorrection == self.sliderExp.value() * self.defaultStep):
                return
            try:
                self.sliderExp.valueChanged.disconnect()
                self.sliderExp.sliderReleased.disconnect()
            except RuntimeError:
                pass
            self.expCorrection = self.sliderExp.value() * self.defaultStep
            self.dataChanged.emit()
            self.sliderExp.valueChanged.connect(f)
            self.sliderExp.sliderReleased.connect(f)

        self.sliderExp.valueChanged.connect(f)
        self.sliderExp.sliderReleased.connect(f)

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        l.addWidget(expLabel)
        hl = QHBoxLayout()
        hl.addWidget(self.expValue)
        hl.addWidget(self.sliderExp)
        l.addLayout(hl)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(l)
        self.adjustSize()
        self.setWhatsThis(
            """<b>Exposure Correction</b>
            Multiplicative correction in the linear sRGB color space.<br>
            Unit is the diaphragm stop.<br>
            """
        )  # end setWhatsThis

        self.setDefaults()

    def updateLayer(self):
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def setDefaults(self):
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.sliderExp.setValue(self.defaultExpCorrection)
        self.expValue.setText(str("{:+.1f}".format(self.defaultExpCorrection)))
        self.expCorrection = self.defaultExpCorrection * self.defaultStep
        self.dataChanged.connect(self.updateLayer)

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [QbLUeSlider]:
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
            if type(obj) in [QbLUeSlider]:
                obj.__setstate__(d['state'][name])
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()
