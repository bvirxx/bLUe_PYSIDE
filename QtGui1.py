from PyQt4 import QtCore, QtGui, uic
#from PyQt4 import QtGui

# Python wrapper for the PyQt4 Form UI_MainWindow
# built by Qt Designer.
#
# The Class UI_MainWindow is built automatically from
# essai1.ui : pyuic4 essai1.ui -o essai1.py
#
# Here, we mainly set the widget events handlers

#from essai1 import *
import sys
import resources_rc  # mandatory!!!

#class Form1(QtGui.QMainWindow, Ui_MainWindow):
class Form1(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        uic.loadUi('essai1.ui', self)
        #self.setupUi(self)                  # inherited from UI_MainWindow
        self.onChange = lambda : 0          # triggered by button or slider change
        self.slidersValues = {}
        self.btnValues = {}

        for slider in self.groupBox.findChildren(QtGui.QSlider):
            segment = slider.objectName()
            slider.valueChanged.connect(
                            lambda value, slider=slider : self.handleSliderMoved(value, slider)
                            )
            self.slidersValues [str(slider.accessibleName())] = slider.value()

        for button in self.groupBox.findChildren(QtGui.QPushButton) :
            segment = button.objectName()
            button.pressed.connect(
                            lambda button=button : self.handleButtonClicked(button)
                            )
            self.btnValues [str(button.accessibleName())] = button.isChecked()

        for button in self.groupBox.findChildren(QtGui.QToolButton) :
            segment = button.objectName()
            #button.setStyleSheet("QToolButton#DCButton:checked {color:black; background-color: green;}")
            button.pressed.connect(
                            lambda button=button : self.handleButtonClicked(button)
                            )
            self.btnValues[str(button.accessibleName())] = button.isChecked()

    def handleButtonClicked(self, button):   # connected to button.toggled
        for k in self.btnValues :
            self.btnValues[k]=0
        self.btnValues[str(button.accessibleName())] = 1
        print "called", self.sender().accessibleName()
        print '*********', self.btnValues['drawFG']
        self.onChange(button)

    def handleSliderMoved (self, value, slider) :   # connected to slider.valueChanged
        self.slidersValues[str(slider.accessibleName())] = value
        self.onChange(slider)

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
