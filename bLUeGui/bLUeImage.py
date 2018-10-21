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

from PySide2.QtGui import QImage, QPixmap

class bImage(QImage):
    """
    Base class for all bLUe images.
    Inherits from QImage and adds a bunch
    of caches encapsulated as properties.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rPixmap = None
        self.hspbBuffer = None
        self.LabBuffer = None
        self.HSVBuffer = None

    @property
    def rPixmap(self):
        return self.__rPixmap

    @rPixmap.setter
    def rPixmap(self, pixmap):
        self.__rPixmap = pixmap

    @property
    def hspbBuffer(self):
        return self.__hspbBuffer

    @hspbBuffer.setter
    def hspbBuffer(self, buffer):
        self.__hspbBuffer = buffer

    @property
    def LabBuffer(self):
        return self.__LabBUffer

    @LabBuffer.setter
    def LabBuffer(self, buffer):
        self.__LabBuffer = buffer

    @property
    def HSVBuffer(self):
        return self.__HSVBUffer

    @HSVBuffer.setter
    def HSVBuffer(self, buffer):
        self.__HSVBuffer = buffer

    def updatePixmap(self, maskOnly=False):
        """
        To respect the Substitutability Principle of Liskov
        for subtypes, we should keep identical signatures for all
        overriding methods, so we define here an unused parameter
        maskOnly.
        @param maskOnly: not used
        @type maskOnly: boolean
        """
        self.rPixmap = QPixmap.fromImage(self)
