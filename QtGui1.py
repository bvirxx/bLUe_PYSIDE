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
from PySide2.QtCore import QSettings, Qt
import sys

from PySide2.QtWidgets import QApplication, QLabel, QMainWindow

import resources_rc   # DO NOT REMOVE !!!!
from graphicsHist import histForm
from layerView import QLayerView
from pyside_dynamicLoader import loadUi
from utils import hideConsole, showConsole


class Form1(QMainWindow):#, Ui_MainWindow): #QtGui.QMainWindow):
    """
    Main window class.
    The layout is loaded from the ui form bLUe.ui.
    """
    screenChanged = QtCore.Signal(int)
    def __init__(self, app, parent=None):
        super(Form1, self).__init__()
        # load UI
        loadUi('bLUe.ui', baseinstance=self, customWidgets= {'QLayerView': QLayerView, 'QLabel': QLabel, 'histForm': histForm})
        #self = QtUiTools.QUiLoader().load("bLUe.ui", self)
        # hooks added to event handlers
        self.updateStatus = lambda : 0
        self.onWidgetChange = lambda : 0
        self.onShowContextMenu = lambda : 0
        self.onExecFileOpen = lambda : 0
        self.onUpdateMenuAssignProfile = lambda : 0
        # State recording.
        self.slidersValues = {}
        self.btnValues = {}
        # connect slider and button signals to handlers
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
        button clicked/toggled signal slot.
        @param button:
        @type button:
        """
        self.onWidgetChange(button)

    def handleToolButtonClicked(self, button):
        """
        button clicked/toggled signal slot.
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

    def readSettings(self):
        # init a Qsettings instance bound to the file bLUe.ini
        self.settings = QSettings("bLUe.ini", QSettings.IniFormat)

    def writeSettings(self):
        self.settings.sync()

    def moveEvent(self, event):
        super(Form1,self).moveEvent(event)
        # detect screen changes
        # CAUTION : dragging the window to another screen
        # does not change screenNumber. Only
        # a call to move() updates screenNumber value.
        c = self.frameGeometry().center()
        id =self.dktp.screenNumber(c)
        if id != self.currentScreenIndex:
            # screen changed
            self.currentScreenIndex = id
            self.screenChanged.emit(id)

    def closeEvent(self, event):
        if self.onCloseEvent():
            #close
            event.accept()
        else:
            # don't close
            event.ignore()
            return
        self.writeSettings()
        if getattr(sys, 'frozen', False):
            showConsole()
        super(Form1, self).closeEvent(event)

def enumerateMenuActions(menu):
    """
    recursively builds the list of actions contained in a menu
    and all its submenus.
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

######################
# launch app and init main form
#####################
app = QApplication(sys.argv)
window = Form1(app)


