from PyQt4 import QtGui, uic
from PyQt4.QtCore import QSettings, QSize


# Python loader for the UI Form built by Qt Designer.


#import sys
import resources_rc   # don't remove

class Form1(QtGui.QMainWindow):

    def __init__(self, parent=None):
        #QtGui.QMainWindow.__init__(self, parent)
        super(Form1, self).__init__(parent)
        # load UI
        ui=uic.loadUi('essai1.ui', self)
        """
        color = QtGui.QColorDialog(ui.window())
        color.setWindowFlags(Qt.Widget)
        color.setOption(QtGui.QColorDialog.NoButtons, True)
        color.show()
        ui.horizontalLayout.addWidget(color)
        """
        self.onWidgetChange = lambda : 0          # triggered by button or slider change
        self.onShowContextMenu = lambda : 0
        self.onExecFileMenu = lambda : 0
        self.onExecFileOpen = lambda : 0
        self.onExecMenuWindow = lambda : 0

        self.slidersValues = {}
        self.btnValues = {}
        self._recentFiles = []

        for slider in self.findChildren(QtGui.QSlider):
            segment = slider.objectName()
            slider.valueChanged.connect(
                            lambda value, slider=slider : self.handleSliderMoved(value, slider)
                            )
            self.slidersValues [str(slider.accessibleName())] = slider.value()

        for button in self.findChildren(QtGui.QPushButton) :
            segment = button.objectName()
            button.pressed.connect(
                            lambda button=button : self.handleButtonClicked(button)
                            )
            self.btnValues[str(button.accessibleName())] = button.isChecked()

        for button in self.findChildren(QtGui.QToolButton) :
            segment = button.objectName()
            #button.setStyleSheet("QToolButton#DCButton:checked {color:black; background-color: green;}")
            button.pressed.connect(
                            lambda button=button : self.handleButtonClicked(button)
                            )
            self.btnValues[str(button.accessibleName())] = button.isChecked()

        for widget in self.findChildren(QtGui.QLabel):
            widget.customContextMenuRequested.connect(lambda pos, widget=widget : self.showContextMenu(pos,widget))

        #for action in self.findChildren(QtGui.QAction) :
        for action in enumerateMenuActions(self.menu_File) : # replace by enumerateMenu
            action.triggered.connect(lambda x, actionName=action.objectName(): self.execFileDialog(x, actionName))

        for action in enumerateMenuActions(self.menuWindow) :
            action.triggered.connect(lambda x, actionName=action.objectName(): self.execMenuWindow(x, actionName))

    def updateMenuOpenRecent(self):
        self.menuOpen_recent.clear()
        for f in self._recentFiles :
            self.menuOpen_recent.addAction(f, lambda x=f: self.execFileOpen(x))

    def execFileDialog(self,x, name):
        self.onExecFileDialog(name)

    def execFileOpen(self, f):
        self.onExecFileOpen(f)

    def execMenuWindow(self, x, name):
        print 'salut'
        self.onExecMenuWindow(x, name)

    def showContextMenu(self, pos, widget):
        self.onShowContextMenu(widget)

    def handleButtonClicked(self, button):   # connected to button.pressed
        for k in self.btnValues :
            self.btnValues[k]=0
        self.btnValues[str(button.accessibleName())] = 1
        self.onWidgetChange(button)

    def handleSliderMoved (self, value, slider) :   # connected to slider.valueChanged
        self.slidersValues[str(slider.accessibleName())] = value
        self.onWidgetChange(slider)

    def readSettings(self):
        self.settings = QSettings("qsettingsexample.ini", QSettings.IniFormat);

        self.resize(self.settings.value("mainwindow/size", QSize(250, 200)).toSize());


    def writeSettings(self):
        self.settings.sync()
        return
        settings = QSettings("qsettingsexample.ini", QSettings.IniFormat);

        print settings.value('paths/dlgdir', 'novalue').toString()
        return
        settings = QSettings("qsettingsexample.ini", QSettings.IniFormat);

        settings.beginGroup("mainwindow")
        settings.setValue("size", self.size())
        settings.endGroup()

        settings.beginGroup("text")
        settings.setValue("font", 1 )
        settings.setValue("size", 2)
        settings.endGroup()

    def closeEvent(self, event):
        self.writeSettings()
        super(Form1, self).closeEvent(event)

def enumerateMenuActions(menu):
    """
    Build  recursively the list of actions contained in menu
    and all submenus.
    :param menu: Qmenu object
    :return: The list of actions
    """
    actions = []
    for action in menu.actions():
        #subMenu
        if action.menu() :
            actions.extend(enumerateMenuActions(action.menu()))
        else:
            actions.append(action)
    return actions
"""
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = Form1()
    import cv2
    img = cv2.imread('E:\orch2-2-2.png')
    height, width, bpc = img.shape
    image = QtGui.QImage(img.data, width, height, bpc*width, QtGui.QImage.Format_RGB888)
    myPixmap=QtGui.QPixmap.fromImage(image)


    window.show()
    myScaledPixmap = myPixmap.scaled(window.label.size(), QtCore.Qt.KeepAspectRatio)

    window.label.setPixmap(myScaledPixmap)
    sys.exit(app.exec_())
"""
