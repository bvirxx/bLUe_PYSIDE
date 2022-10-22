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
from PySide2.QtCore import QSettings, Qt, QtMsgType
import sys

from PySide2.QtGui import QScreen
from PySide2.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy

import bLUeTop.Gui

from bLUeTop.pyside_dynamicLoader import loadUi
from bLUeTop.splitView import splitWindow
from bLUeTop.utils import hideConsole, showConsole, QbLUeColorDialog, colorInfoView


class Form1(QMainWindow):
    """
    Main form class.
    """
    # screen changed signal
    screenChanged = QtCore.Signal(QScreen)

    def __init__(self):
        super(Form1, self).__init__()
        self.settings = QSettings("bLUe.ini", QSettings.IniFormat)
        # we presume that the form will be shown first on screen 0;
        # No detection possible before it is effectively shown !
        self.currentScreenIndex = 0
        self.__colorChooser, self.__infoView = (None,) * 2

    @property
    def colorChooser(self):
        if self.__colorChooser is None:
            self.__colorChooser = QbLUeColorDialog(parent=self)
        return self.__colorChooser

    @property
    def infoView(self):
        if self.__infoView is None:
            self.__infoView = colorInfoView()
            self.__infoView.label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
            self.__infoView.label.setMaximumSize(400, 80)
        return self.__infoView

    def init(self):
        """
        Load the form from the .ui file

        """
        from bLUeTop.graphicsHist import histForm
        from bLUeTop.layerView import QLayerView
        from bLUeTop.imLabel import imageLabel
        loadUi('bLUe.ui', baseinstance=self,
               customWidgets={'QLayerView': QLayerView, 'QLabel': QLabel, 'histForm': histForm,
                              'imageLabel': imageLabel})
        # hook called by event slots
        # should be redefined later
        self.onWidgetChange = lambda b: None
        # State recording.
        self.slidersValues = {}
        self.btnValues = {}
        self.btns = {}
        # connect slider and button signals to slots
        for slider in self.findChildren(QtWidgets.QSlider):
            slider.valueChanged.connect(
                lambda value, slider=slider: self.handleSliderMoved(value, slider)
            )
            self.slidersValues[str(slider.accessibleName())] = slider.value()

        for button in self.findChildren(QtWidgets.QPushButton):
            # signal clicked has a default argument checked=False,
            # so we consume all passed args
            button.clicked.connect(
                lambda *args, button=button: self.handlePushButtonClicked(button)
            )
            self.btnValues[str(button.accessibleName())] = button.isChecked()
            self.btns[str(button.accessibleName())] = button

        for button in self.findChildren(QtWidgets.QToolButton):
            button.toggled.connect(
                lambda state, button=button: self.handleToolButtonClicked(button)
            )
            if not button.isCheckable():
                # signal clicked has a default argument checked=False
                # so we consume all args passed.
                button.clicked.connect(
                    lambda *args, button=button: self.handleToolButtonClicked(button)
                )
            self.btnValues[str(button.accessibleName())] = button.isChecked()
            self.btns[str(button.accessibleName())] = button

    def handlePushButtonClicked(self, button):
        """
        button clicked/toggled slot.

        :param button:
        :type button:
        """
        self.onWidgetChange(button)

    def handleToolButtonClicked(self, button):
        """
        button clicked/toggled slot.
        The toggled signal is triggered only by checkable buttons,
        when the button state changes. Thus, the method should be called
        by all auto exclusive buttons in a group to correctly update
        the btnValues dictionary.

        :param button:
        :type button: QButton
        """
        checked = button.isChecked()
        self.btnValues[str(button.accessibleName())] = checked
        self.onWidgetChange(button)

    def handleSliderMoved(self, value, slider):
        """
        Slider valueChanged slot.

        :param value:
        :type value:
        :param slider:
        :type slider : QSlider
        """
        self.slidersValues[slider.accessibleName()] = value
        self.onWidgetChange(slider)

    def moveEvent(self, event):
        """
        Overriding moveEvent to emit
        a screenChanged signal
        when a screen change is detected.

        :param event:
        :type event:
        """
        super(Form1, self).moveEvent(event)
        # detecting screen changes :
        # getting current QScreen instance
        sn = self.windowHandle().screen()
        if sn is not self.currentScreenIndex:
            # screen changed detected
            self.currentScreenIndex = sn
            self.screenChanged.emit(sn)

    def closeEvent(self, event):
        if self.onCloseEvent():
            # close
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

    :param menu: Qmenu object
    :return: list of actions
    """
    actions = []
    for action in menu.actions():
        # subMenu
        if action.menu():
            actions.extend(enumerateMenuActions(action.menu()))
            action.menu().parent()
        else:
            actions.append(action)
    return actions


#######################
# app message handler
#######################
def message_handler(kind, context, msg):
    if kind != QtMsgType.QtWarningMsg:
        print(msg, file=sys.stderr)


QtCore.qInstallMessageHandler(message_handler)

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

##################
# constructing app
##################

QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)  # needed when a plugin initializes a web engine
bLUeTop.Gui.app = QApplication(sys.argv)

#################
# init main form
# the UI is not loaded yet
bLUeTop.Gui.window = Form1()
#################

# Before/After view
bLUeTop.Gui.splitWin = splitWindow(bLUeTop.Gui.window)
