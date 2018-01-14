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
from PySide2.QtWidgets import QGraphicsView, QSizePolicy, QVBoxLayout, QSlider

from utils import optionsWidget


class transForm (QGraphicsView):

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=200, layer=None, parent=None, mainForm=None):
        wdgt = transForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt


    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super(transForm, self).__init__(parent=parent)
        self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.img = targetImage
        self.layer = layer
        self.mainForm = mainForm
        self.onUpdateFilter = lambda *args: 0

        # options
        optionList = ['Perspective', 'Rotation']
        self.listWidget1 = optionsWidget(options=optionList, exclusive=True)
        self.options = self.listWidget1.options
        # set initial selection
        self.listWidget1.checkOption(optionList[0])

        # rotation slider
        self.sliderRot = QSlider(Qt.Horizontal)
        self.sliderRot.setTickPosition(QSlider.TicksBelow)
        self.sliderRot.setRange(-180, 180)
        self.sliderRot.setSingleStep(1)

        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
        l.addWidget(self.listWidget1)
        l.addWidget(self.sliderRot)

        self.setLayout(l)
        self.adjustSize()

