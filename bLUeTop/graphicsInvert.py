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
from PySide2.QtWidgets import QVBoxLayout

from bLUeGui.graphicsForm import baseForm
from bLUeTop.utils import optionsWidget


class invertForm(baseForm):
    """
    Form for negative inversion
    """
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        newWindow = invertForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        newWindow.setWindowTitle(layer.name)
        return newWindow
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)

        # options
        optionList, optionNames = ['Auto'], ['Auto Orange Mask Removing']
        self.listWidget1 = optionsWidget(options=optionList, optionNames=optionNames, exclusive=False,
                                         changed=self.dataChanged)
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.options = self.listWidget1.options

        # layout
        vl = QVBoxLayout()
        vl.addWidget(self.listWidget1)
        self.setLayout(vl)

        self.setDefaults()
        self.setWhatsThis(
            """
            <b> Negative Inversion</b><br>
            Negative films show an orange mask that must be corrected.<br>
            Automatic correction try to sample the mask color
            from an unexposed area of the negative film.<br>
            To do a <b>manual correction</b>, uncheck 'Auto Orange Mask Removing' and, next,
            Ctrl+Click the dark border of the image or, otherwise, a dark gray area.<br>
            """
        )  # end of setWhatsThis

    def setDefaults(self):
        self.Rmask, self.Gmask, self.Bmask = (128,) * 3

    def colorPickedSlot(self, x, y, modifiers):
        """
        Overriding method.
        Set the invert mask to the color picked on the
        active layer.
        (x,y) coordinates are relative to the full size image.

        :param x:
        :type x: int
        :param y:
        :type y: int
        :param modifiers:
        :type modifiers: Qt.KeyboardModifiers
        """
        if modifiers == Qt.ControlModifier:
            r, g, b = self.layer.parentImage.getActivePixel(x, y)
            self.setInvertMask(r, g, b)

    def setInvertMask(self, r, g, b):
        """
        Set the invert mask to color (r, g, b)
        and update.

        :param r:
        :type r: int
        :param g:
        :type g: int
        :param b:
        :type b: int
        """
        self.Rmask, self.Gmask, self.Bmask = r, g, b
        self.dataChanged.emit()

    def updateLayer(self):
        """
        overriding dataChanged slot
        """
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()
