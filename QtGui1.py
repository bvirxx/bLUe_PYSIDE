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

from PySide import QtGui, QtUiTools #, uic
from PySide.QtCore import QObject
from PySide.QtCore import QSettings, QSize
from PIL.ImageCms import getProfileDescription
import sys
from PySide.QtGui import QApplication, QMessageBox
from PySide.QtGui import QLabel
from PySide.QtGui import QMainWindow

import resources_rc   # DO NOT REMOVE !!!!
from layerView import QLayerView
from pyside_dynamicLoader import loadUi


class Form1(QMainWindow):#, Ui_MainWindow): #QtGui.QMainWindow):
    """
    Main window class.
    The layout is loaded from the ui form bLUe.ui.
    """
    def __init__(self, parent=None):
        super(Form1, self).__init__()
        #self.setupUi(self)
        # load UI
        loadUi('bLUe.ui', baseinstance=self, customWidgets= {'QLayerView': QLayerView, 'QLabel': QLabel})
        #self = QtUiTools.QUiLoader().load("bLUe.ui", self)
        # Slot hooks : they make the GUI independent
        # from the underlying application.
        self.onWidgetChange = lambda : 0
        self.onShowContextMenu = lambda : 0
        self.onExecMenuFile = lambda : 0
        self.onExecFileOpen = lambda : 0
        self.onExecMenuWindow = lambda : 0
        self.onExecMenuImage = lambda : 0
        self.onExecMenuLayer = lambda: 0
        self.onUpdateMenuAssignProfile = lambda : 0

        # button state recording.
        self.slidersValues = {}
        self.btnValues = {}

        self._recentFiles = []

        # connections to SLOTS
        for slider in self.findChildren(QtGui.QSlider):
            slider.valueChanged.connect(
                            lambda value, slider=slider : self.handleSliderMoved(value, slider)
                            )
            self.slidersValues [str(slider.accessibleName())] = slider.value()

        for button in self.findChildren(QtGui.QPushButton) :
            button.pressed.connect(
                            lambda button=button : self.handlePushButtonClicked(button)
                            )
            self.btnValues[str(button.accessibleName())] = button.isChecked()

        for button in self.findChildren(QtGui.QToolButton) :
            button.pressed.connect(
                            lambda button=button : self.handleToolButtonClicked(button)
                            )
            self.btnValues[str(button.accessibleName())] = button.isChecked()

        for widget in self.findChildren(QtGui.QLabel):
            widget.customContextMenuRequested.connect(lambda pos, widget=widget : self.showContextMenu(pos,widget))

        for action in enumerateMenuActions(self.menu_File) : # replace by enumerateMenu
            action.triggered.connect(lambda x=0, actionName=action.objectName(): self.execMenuFile(x, actionName))
            #action.triggered.connect(lambda actionName=action.objectName(): self.execMenuFile(0, actionName))

        for action in enumerateMenuActions(self.menuWindow) :
            action.triggered.connect(lambda x=0, actionName=action.objectName(): self.execMenuWindow(x, actionName))

        tmp = enumerateMenuActions(self.menuImage)
        for action in tmp :# enumerateMenuActions(self.menuImage) :
            action.triggered.connect(lambda x=0, actionName=action.objectName(): self.execMenuImage(x, actionName))

        for action in enumerateMenuActions(self.menuLayer) :
            action.triggered.connect(lambda x=0, actionName=action.objectName(): self.execMenuLayer(x, actionName))

        # mouse hovered event Slots
        #self.menuOpen_recent.menuAction().hovered.connect(lambda : self.updateMenuOpenRecent())
        #self.menuColor_settings.menuAction().hovered.connect(lambda : self.updateMenuAssignProfile())


    def execAssignProfile(self, x):
        self.onExecAssignProfile(x)

    def execMenuFile(self,x, name):
        self.onExecMenuFile(name)

    def execFileOpen(self, f):
        self.onExecFileOpen(f)

    def execMenuWindow(self, x, name):
        self.onExecMenuWindow(x, name)

    def execMenuImage(self, x, name):
        self.onExecMenuImage(x, name)

    def execMenuLayer(self, x, name):
        self.onExecMenuLayer(x, name)

    def showContextMenu(self, pos, widget):
        self.onShowContextMenu(widget)

    def handlePushButtonClicked(self, button):
        self.onWidgetChange(button)

    def handleToolButtonClicked(self, button):
        """
        connected to button.pressed signal
        mutually exclusive selection
        :param button:
        """
        for k in self.btnValues :
            self.btnValues[k] = False
        self.btnValues[str(button.accessibleName())] = True
        self.onWidgetChange(button)

    def handleSliderMoved (self, value, slider) :
        """
        connected to slider.valueChanged signal
        :param value:
        :param slider:
        """
        self.slidersValues[str(slider.accessibleName())] = value
        self.onWidgetChange(slider)

    def readSettings(self):
        self.settings = QSettings("bLUe.ini", QSettings.IniFormat)
        self.resize(self.settings.value("mainwindow/size", QSize(250, 200))) #.toSize())

    def writeSettings(self):
        self.settings.sync()
        """
        settings = QSettings("bLUe.ini", QSettings.IniFormat);

        print settings.value('paths/dlgdir', 'novalue').toString()
        return
        settings = QSettings("bLUe.ini", QSettings.IniFormat);

        settings.beginGroup("mainwindow")
        settings.setValue("size", self.size())
        settings.endGroup()

        settings.beginGroup("text")
        settings.setValue("font", 1 )
        settings.setValue("size", 2)
        settings.endGroup()
        """

    def closeEvent(self, event):
        if self.onCloseEvent(event):
            #close
            event.accept()
        else:
            event.ignore()
            return
        """
        quit_msg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message', quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
            return
        """
        self.writeSettings()
        super(Form1, self).closeEvent(event)

def enumerateMenuActions(menu):
    """
    Build  recursively the list of actions contained in a menu
    and all its submenus.
    :param menu: Qmenu object
    :return: list of actions
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

########
# QApplication and mainWindow init
# A Python module is a singleton : the initialization code below
# is executed only once, regardless the number of module imports.
#######
app = QApplication(sys.argv)
window = Form1()


