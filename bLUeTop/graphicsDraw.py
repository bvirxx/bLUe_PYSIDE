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
from PySide2.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout

from bLUeGui.graphicsForm import baseForm


class drawForm (baseForm):
    """
    Drawing form
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=200, layer=None, parent=None):
        wdgt = drawForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.options = None
        """
        # options
        optionList1, optionNames1 = ['Free', 'Rotation', 'Translation', 'Align'], ['Free Transformation', 'Rotation', 'Translation', 'Align']
        self.listWidget1 = optionsWidget(options=optionList1, optionNames=optionNames1, exclusive=True, changed=self.dataChanged)
        optionList2, optionNames2 = ['Transparent'], ['Set Transparent Pixels To Black']
        self.listWidget2 = optionsWidget(options=optionList2, optionNames=optionNames2, exclusive=False, changed=self.dataChanged)
        self.options = UDict((self.listWidget1.options, self.listWidget2.options))
        # set initial selection to Perspective
        self.listWidget1.checkOption(optionList1[0])
        """
        pushButton1 = QPushButton(' Undo ')
        pushButton1.adjustSize()
        pushButton2 = QPushButton(' Redo ')
        pushButton2.adjustSize()

        pushButton1.clicked.connect(self.undo)
        pushButton2.clicked.connect(self.redo)

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        hl = QHBoxLayout()
        hl.setAlignment(Qt.AlignHCenter)
        hl.addWidget(pushButton1)
        hl.addWidget(pushButton2)
        l.addLayout(hl)
        self.setLayout(l)
        self.adjustSize()

        self.setDefaults()
        self.setWhatsThis(
                        """
                        <b>Drawing :</b><br>
                          Choose a brush family, flow, hardness and opacity.
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
        # l.tool.setBaseTransform()
        l.applyToStack()
        l.parentImage.onImageChanged()

    def undo(self):
        try:
            self.layer.sourceImg = self.layer.history.undo(saveitem=self.layer.sourceImg.copy()).copy()  # copy is mandatory
            self.updateLayer()
        except ValueError:
            pass

    def redo(self):
        try:
            self.layer.sourceImg = self.layer.history.redo().copy()  # copy is mandatory
            self.updateLayer()
        except ValueError:
            pass

    def reset(self):
        self.layer.tool.resetTrans()
