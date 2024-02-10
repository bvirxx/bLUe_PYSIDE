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
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QLabel

from bLUeGui.graphicsForm import baseForm


class HDRMergeForm(baseForm):
    defaultExpCorrection = 0.0
    defaultStep = 0.1

    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = HDRMergeForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        # options
        self.options = None
        self.mergeButton = QPushButton('Refresh')
        self.mergeButton.setMaximumWidth(80)
        self.warn = QLabel(
                             """
                             To select image layers to merge, check <i>Fusion Flag</i> in their context menu
                             and next, press <i>Refresh</i>'
                             """
                          )
        self.warn.setWordWrap(True)
        self.warn.setMaximumWidth(300)

        # self.warn.setStyleSheet("QLabel {color : yellow; }")

        # button slot
        def f():
            self.dataChanged.emit()

        self.mergeButton.pressed.connect(f)

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        l.addSpacing(30)
        l.addWidget(self.mergeButton)
        l.addSpacing(100)
        l.addWidget(self.warn)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(l)
        self.adjustSize()
        self.setWhatsThis(
            """<b>Exposure Fusion</b>
            Computes exposure fusion for a list of images with identical sizes.<br>
            To <b>select the images</b> to merge, right-click on each image in the 
            layer stack and check <i>Fusion Flag</i> in the context menu which opens. 
            A letter <i>F</i> will appear next to the layer name.<br>
            To <b>merge</b> the images, press the <i>Refresh</i> button.<br>
            <b>Note.</b> Only selected images <b>below</b> the <i>Merge</i> layer will be merged.
            """
        )  # end setWhatsThis

        self.setDefaults()

    def updateLayer(self):
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def setDefaults(self):
        pass
