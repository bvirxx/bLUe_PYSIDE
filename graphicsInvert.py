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

from bLUeGui.graphicsForm import baseForm


class invertForm(baseForm) :  # TODO setDefaults(), updateLayer()
    """
    Form for interactive RGB curves
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        newWindow = invertForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        newWindow.setWindowTitle(layer.name)
        return newWindow
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, parent=parent)
        self.Rmask, self.Bmask, self.Gmask = (255,) * 3

    def colorPickedSlot(self, x, y, modifiers):
        """
        Set the invert mask to the color picked on the
        active layer.
        (x,y) coordinates are relative to the full size image.
        @param x:
        @type x:
        @param y:
        @type y:
        @param modifiers:
        @type modifiers:
        """
        if modifiers == Qt.ControlModifier:
            r, g, b = self.layer.parentImage.getActivePixel(x, y)
            self.setMask(r, g, b)

    def setMask(self, r, g, b):
        self.Rmask, self.Gmask, self.Bmask = r, g, b
        self.layer.applyToStack()
        self.layer.onImageChanged()