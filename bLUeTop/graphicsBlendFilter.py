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
from PySide6.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout

from bLUeGui.graphicsForm import baseForm
from bLUeGui.qrangeslider import QRangeSlider
from bLUeTop.utils import optionsWidget


class blendFilterIndex:
    GRADUALBT, GRADUALTB, GRADUALNONE = range(3)


class blendFilterForm(baseForm):

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.defaultFilterStart = 0
        self.defaultFilterEnd = 99
        self.filterStart = self.defaultFilterStart
        self.filterEnd = self.defaultFilterEnd

        self.kernelCategory = blendFilterIndex.GRADUALNONE  # TODO kernelCategory should be renamed as filterIndex 5/12/18
        # options
        optionList, optionNames = ['Gradual Top', 'Gradual Bottom'], ['Top To Bottom', 'Bottom To Top']
        filters = [blendFilterIndex.GRADUALTB, blendFilterIndex.GRADUALBT]  # , blendFilterIndex.GRADUALNONE]
        self.filterDict = dict(zip(optionList, filters))

        self.listWidget1 = optionsWidget(options=optionList, optionNames=optionNames, exclusive=True,
                                         changed=self.dataChanged)
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
            try:
                self.sliderFilterRange.startValueChanged.disconnect()
                self.sliderFilterRange.endValueChanged.disconnect()
                self.sliderFilterRange.rangeDone.disconnect()
            except RuntimeError:
                pass
            self.filterStart, self.filterEnd = self.sliderFilterRange.getRange()
            self.dataChanged.emit()
            self.sliderFilterRange.startValueChanged.connect(frUpdate)  # send new value as parameter
            self.sliderFilterRange.endValueChanged.connect(frUpdate)  # send new value as parameter
            self.sliderFilterRange.rangeDone.connect(frUpdate)

        self.sliderFilterRange.startValueChanged.connect(frUpdate)  # send new value as parameter
        self.sliderFilterRange.endValueChanged.connect(frUpdate)  # send new value as parameter
        self.sliderFilterRange.rangeDone.connect(frUpdate)

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
               It mimics the classical gradual gray filter often used by
               photographers to darken the sky.<br>
               To control the regions of maximum and minimum intensities use the Range slider.
            """
        )  # end setWhatsThis

        self.setDefaults()

    def setDefaults(self, name1='Gradual Top', start=0, end=0, orientation=0):
        self.defaultFilterStart = start
        self.defaultFilterEnd = end
        self.sliderFilterRange.setRange(start, end)
        self.listWidget1.checkOption(name1)
        self.dataChanged.connect(self.updateLayer)

    def updateLayer(self):
        """
        datachanged slot
        """
        for key in self.listWidget1.options:
            if self.listWidget1.options[key]:
                self.kernelCategory = self.filterDict[key]
                break
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [optionsWidget, QRangeSlider]:
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
            if type(obj) in [optionsWidget, QRangeSlider]:
                obj.__setstate__(d['state'][name])
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()
