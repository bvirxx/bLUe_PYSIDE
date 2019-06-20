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
from PySide2.QtWidgets import QVBoxLayout, QPushButton, QHBoxLayout

from bLUeGui.graphicsForm import baseForm
from bLUeTop.utils import optionsWidget, UDict


class transForm (baseForm):
    """
    Geometric transformation form
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=200, layer=None, parent=None):
        wdgt = transForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        # options
        optionList1, optionNames1 = ['Free', 'Rotation', 'Translation'], ['Free Transformation', 'Rotation', 'Translation']
        self.listWidget1 = optionsWidget(options=optionList1, optionNames=optionNames1, exclusive=True, changed=self.dataChanged)
        optionList2, optionNames2 = ['Transparent'], ['Set Transparent Pixels To Black']
        self.listWidget2 = optionsWidget(options=optionList2, optionNames=optionNames2, exclusive=False, changed=self.dataChanged)
        self.options = UDict((self.listWidget1.options, self.listWidget2.options))
        # set initial selection to Perspective
        self.listWidget1.checkOption(optionList1[0])

        pushButton1 = QPushButton(' Reset Transformation ')
        pushButton1.adjustSize()

        pushButton1.clicked.connect(self.reset)

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        l.addWidget(self.listWidget1)
        l.addWidget(self.listWidget2)
        hl = QHBoxLayout()
        hl.setAlignment(Qt.AlignHCenter)
        hl.addWidget(pushButton1)
        l.addLayout(hl)
        self.setLayout(l)
        self.adjustSize()
        self.setDefaults()
        self.setWhatsThis(
                        """
                        <b>Geometric transformation :</b><br>
                          Choose a transformation type and drag either corner of the image 
                          using the small square red buttons.<br>
                          Ctrl+Alt+Drag to change the <b>initial positions</b> of buttons.
                        """
                        )  # end of setWhatsThis

    def setDefaults(self):
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.dataChanged.connect(self.updateLayer)

    def updateLayer(self):
        """
        dataChanged slot
        """
        l = self.layer
        l.tool.setBaseTransform()
        l.applyToStack()
        l.parentImage.onImageChanged()

    def reset(self):
        self.layer.tool.resetTrans()


class imageForm(transForm):
    pass

