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
from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QVBoxLayout, QLabel, QWidget, QHBoxLayout

from bLUeTop.Gui import window


class bTablet:

    valuator = Enum('valuator', ['PressureValuator', 'TangentialPressureValuator',
                                 'TiltValuator', 'VTiltValuator', 'HTiltValuator', 'NoValuator'], start=0)

    ########################################################################################
    # the stylus controls Brush size, opacity and saturation.
    # The corresponding functions are found in imageLabel.tabletEvent()
    # and can be freely modified.
    try:
        # restore valuators from settings
        __widthValuator = valuator[window.settings.value('tablet/widthvaluator', 'NoValuator')]
        __alphaValuator = valuator[window.settings.value('tablet/alphavaluator', 'NoValuator')]
        __satValuator = valuator[window.settings.value('tablet/satvaluator', 'NoValuator')]
    except KeyError:
        __widthValuator = valuator['NoValuator']
        __alphaValuator = valuator['NoValuator']
        __satValuator = valuator['NoValuator']
    ########################################################################################

    @classmethod
    def getWidthValuator(cls):
        return cls.__widthValuator

    @classmethod
    def setWidthValuator(cls, v):
        cls.__widthValuator = v

    @classmethod
    def getSatValuator(cls):
        return cls.__satValuator

    @classmethod
    def setSatValuator(cls, v):
        cls.__satValuator = v

    @classmethod
    def getAlphaValuator(cls):
        return cls.__alphaValuator

    @classmethod
    def setAlphaValuator(cls, v):
        cls.__alphaValuator = v


class bTabletSettings(QWidget):
    wdg = None
    @classmethod
    def showSettings(cls):
        if cls.wdg:
            cls.wdg.show()
            return
        cls.wdg = bTabletSettings(parent=window)
        cls.wdg.setWindowTitle('Tablet settings')
        cls.wdg.show()

    def __init__(self, parent=None):
        super().__init__(parent=parent, f=Qt.WindowType.Window)
        box1 = QComboBox()
        box2 = QComboBox()
        box3 = QComboBox()
        box4 = QComboBox()
        for box in [box1, box2, box3]:
            box.addItems([v.name for v in bTablet.valuator])
            box.setEditable(False)
            box.model().item(1).setEnabled(False)  #tangentialPressureValuator is not available for stylus
        box1.setCurrentText(bTablet.getWidthValuator().name)
        box1.currentIndexChanged.connect(self.changeWidthValuator)
        box2.setCurrentText(bTablet.getAlphaValuator().name)
        box2.currentIndexChanged.connect(self.changeAlphaValuator)
        box3.setCurrentText(bTablet.getSatValuator().name)
        box3.currentIndexChanged.connect(self.changeSatValuator)

        layout = QVBoxLayout()
        hl1 = QHBoxLayout()
        hl1.addWidget(QLabel('Bruhs Width'))
        hl1.addWidget(box1)
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel('Brush Alpha'))
        hl2.addWidget(box2)
        hl3 = QHBoxLayout()
        hl3.addWidget(QLabel('Brush Saturation'))
        hl3.addWidget(box3)
        for hl in [hl1, hl2, hl3]:
            layout.addLayout(hl)
        self.setLayout(layout)
        self.setWhatsThis(
            """
            <b>Choose how stylus parameters control brush properties.</b>
            
            """
        )

    def changeWidthValuator(self, i):
        bTablet.setWidthValuator(bTablet.valuator(i))
        window.settings.setValue('tablet/widthvaluator', bTablet.getWidthValuator().name)

    def changeSatValuator(self, i):
        bTablet.setSatValuator(bTablet.valuator(i))
        window.settings.setValue('tablet/satvaluator', bTablet.getSatValuator().name)

    def changeAlphaValuator(self, i):
        bTablet.setAlphaValuator(bTablet.valuator(i))
        window.settings.setValue('tablet/alphavaluator', bTablet.getAlphaValuator().name)

