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
import os

from PySide2 import QtWidgets, QtCore
from PySide2.QtCore import QSettings
import sys

from PySide2.QtWidgets import QApplication, QLabel, QMainWindow

import resources_rc   # MANDATORY - DO NOT REMOVE !!!!
from graphicsHist import histForm
from layerView import QLayerView
from pyside_dynamicLoader import loadUi
from utils import hideConsole, showConsole

class Form1(QMainWindow):
    """
    Main form class.
    The form is loaded from bLUe.ui.
    """
    # screen changed signal
    screenChanged = QtCore.Signal(int)
    def __init__(self): # app, parent=None):  # TODO 05/07/18 validate app and parent removing
        super(Form1, self).__init__()
        # load .ui file
        self.settings = QSettings("bLUe.ini", QSettings.IniFormat)
        # we presume that the form will be shown first on screen 0;
        # No detection possible before it is effectively shown !
        self.currentScreenIndex = 0
        loadUi('bLUe.ui', baseinstance=self, customWidgets= {'QLayerView': QLayerView, 'QLabel': QLabel, 'histForm': histForm})
        # hook called by event slots
        # should be redefined later
        self.onWidgetChange = lambda b : None
        # State recording.
        self.slidersValues = {}
        self.btnValues = {}
        # connect slider and button signals to slots
        for slider in self.findChildren(QtWidgets.QSlider):
            slider.valueChanged.connect(
                            lambda value, slider=slider : self.handleSliderMoved(value, slider)
                            )
            self.slidersValues [str(slider.accessibleName())] = slider.value()

        for button in self.findChildren(QtWidgets.QPushButton) :
            # signal clicked has a default argument checked=False,
            # so we consume all args passed
            button.clicked.connect(
                            lambda *args, button=button : self.handlePushButtonClicked(button)
                            )
            self.btnValues[str(button.accessibleName())] = button.isChecked()

        for button in self.findChildren(QtWidgets.QToolButton) :
            button.toggled.connect(
                            lambda state, button=button : self.handleToolButtonClicked(button)
                            )
            if not button.isCheckable():
                # signal clicked has a default argument checked=False
                # so we consume all args passed.
                button.clicked.connect(
                                lambda *args, button=button : self.handleToolButtonClicked(button)
                                )
            self.btnValues[str(button.accessibleName())] = button.isChecked()

    def handlePushButtonClicked(self, button):
        """
        button clicked/toggled slot.
        @param button:
        @type button:
        """
        self.onWidgetChange(button)

    def handleToolButtonClicked(self, button):
        """
        button clicked/toggled slot.
        The toggled signal is triggered only by checkable buttons,
        when the button state changes. Thus, the method is called
        by all auto exclusive buttons in a group to correctly update
        the btnValues dictionary.
        @param button:
        @type button: QButton
        """
        self.btnValues[str(button.accessibleName())] = button.isChecked()
        self.onWidgetChange(button)

    def handleSliderMoved (self, value, slider) :
        """
        Slider valueChanged slot
        @param value:
        @param slider:
        @type slider : QSlider
        """
        self.slidersValues[slider.accessibleName()] = value
        self.onWidgetChange(slider)

    def moveEvent(self, event):
        """
        Dragging the window to another screen
        does not change screenNumber : only
        a call to move() updates screenNumber.
        As a workaround we override moveEvent.
        The signal screenChanged is emitted
        when a screen change is detected.
        @param event:
        @type event:
        """
        super(Form1,self).moveEvent(event)
        # detect screen changes
        c = self.frameGeometry().center()
        sn = rootWidget.screenNumber(c)
        if sn != self.currentScreenIndex:
            # screen changed detected
            self.currentScreenIndex = sn
            self.screenChanged.emit(sn)

    def closeEvent(self, event):
        if self.onCloseEvent():
            #close
            event.accept()
        else:
            # don't close
            event.ignore()
            return
        self.settings.sync()
        if getattr(sys, 'frozen', False):
            showConsole()
        super(Form1, self).closeEvent(event)

def enumerateMenuActions(menu):
    """
    Recursively builds the list of actions contained in a menu
    and in its submenus.
    @param menu: Qmenu object
    @return: list of actions
    """
    actions = []
    for action in menu.actions():
        #subMenu
        if action.menu():
            actions.extend(enumerateMenuActions(action.menu()))
            action.menu().parent()
        else:
            actions.append(action)
    return actions

########################
# Add plugin path to library path : mandatory to enable
# the loading of imageformat dlls for reading and writing QImage objects.
#######################
plugin_path = os.path.join(os.path.dirname(QtCore.__file__), "plugins")
QtCore.QCoreApplication.addLibraryPath(plugin_path)

######################
# Hide console for frozen app
# Pass an argument to program to keep console showing
#####################
if getattr(sys, 'frozen', False) and len(sys.argv) <= 1:
    hideConsole()

############
# launch app
############
app = QApplication(sys.argv)
# get root widget for screen management
rootWidget = app.desktop()

################3
# init main form
window = Form1()
#################


