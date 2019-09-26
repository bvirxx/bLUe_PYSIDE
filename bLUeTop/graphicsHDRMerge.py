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
from PySide2.QtWidgets import QVBoxLayout, QPushButton

from bLUeGui.graphicsForm import baseForm

class HDRMergeForm(baseForm):
    defaultExpCorrection = 0.0
    defaultStep = 0.1

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = HDRMergeForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        # options
        self.options = None
        self.mergeButton = QPushButton('Refresh')

        # button slot
        def f():
            self.dataChanged.emit()

        self.mergeButton.pressed.connect(f)

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        l.addSpacing(30)
        l.addWidget(self.mergeButton)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(l)
        self.adjustSize()
        self.setWhatsThis(
                        """<b>Exposure Fusion</b>
                        In the layer stack use the context menu to select the layers to be merged. Next, press
                        the <i>Refresh</i> button.<br>
                        """
                         )  # end setWhatsThis

        self.setDefaults()

    def updateLayer(self):
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def setDefaults(self):
        pass


