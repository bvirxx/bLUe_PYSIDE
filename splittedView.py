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

#########
# Before/After View
#########
from itertools import cycle

from PySide2.QtCore import Qt


class splittedWindow:
    """
    before/afetr viewing
    """
    splittedViews = cycle(('H','V','B'))

    def __init__(self, win):
        self.mainWin = win

    def setSplittedView(self):
        """
        Initialize a Before/After view
        @return:
        @rtype:
        """
        self.mainWin.label.hide()
        self.mainWin.splitter.show()
        self.mainWin.label_2.show()
        self.mainWin.label_3.show()
        # wait for size updates
        watchDog = 0
        from QtGui1 import app
        while not (self.mainWin.label_2.width() > 0 and self.mainWin.label_3.width() > 0):
            if watchDog >= 3:
                break
            watchDog += 1
            app.processEvents()
        # sync before (label_2) with after (label_3)
        self.mainWin.label_2.img.Zoom_coeff = self.mainWin.label_3.img.Zoom_coeff
        if self.mainWin.splitter.currentState == 'H':
            self.mainWin.label_2.img.xOffset = self.mainWin.label_3.img.xOffset - self.mainWin.label_3.width()
            self.mainWin.label_2.img.yOffset = self.mainWin.label_3.img.yOffset
        elif self.mainWin.splitter.currentState == 'V':
            self.mainWin.label_2.img.yOffset = self.mainWin.label_3.img.yOffset - self.mainWin.label_3.height()
            self.mainWin.label_2.img.xOffset = self.mainWin.label_3.img.xOffset
        else:
            # Only Before window
            self.mainWin.label_2.img.xOffset, self.mainWin.label_2.img.yOffset = self.mainWin.label_3.img.xOffset, self.mainWin.label_3.img.yOffset
            self.mainWin.label_3.hide()
        self.mainWin.label_2.update()
        self.mainWin.label_3.update()

    def nextSplittedView(self):
        """
        Jump to next Before/After Mode
        @return:
        @rtype:
        """
        self.mainWin.splitter.currentState = next(self.splittedViews)
        if self.mainWin.splitter.currentState == 'H':
            self.mainWin.splitter.setOrientation(Qt.Horizontal)
        elif self.mainWin.splitter.currentState == 'V':
            self.mainWin.splitter.setOrientation(Qt.Vertical)
        else:
            # Only Before window
            self.mainWin.label_3.hide()
        self.setSplittedView()

    def syncSplittedView(self, widg1, widg2, linked):
        """
        Sync Before/After views.
        Called by the mouse event handler
        @param widg1:
        @type widg1:
        @param widg2:
        @type widg2:
        @param linked:
        @type linked:
        @return:
        @rtype:
        """
        if not linked:
            return
        widg1.img.Zoom_coeff = widg2.img.Zoom_coeff
        if self.mainWin.splitter.currentState == 'H':
            if widg1.objectName() == 'label_2': # dest is right
                widg1.img.xOffset = widg2.img.xOffset - widg2.width()
            else: # dest is left
                widg1.img.xOffset = widg2.img.xOffset + widg1.width()
            widg1.img.yOffset = widg2.img.yOffset
        else:
            if widg1.objectName() == 'label_2':
                widg1.img.yOffset = widg2.img.yOffset - widg2.height()
            else:
                widg1.img.yOffset = widg2.img.yOffset + widg1.height()
            widg1.img.xOffset = widg2.img.xOffset
        self.mainWin.label_2.update()
        self.mainWin.label_3.update()
