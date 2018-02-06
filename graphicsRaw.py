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
from PySide2 import QtCore

from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QGraphicsView, QSizePolicy, QVBoxLayout, QLabel, QHBoxLayout, QSlider

from utils import optionsWidget, UDict


class rawForm (QGraphicsView):
    """
    GUI for postprocessing of raw files
    """
    defaultExpCorrection = 1.0
    DefaultStep = 0.01
    dataChanged = QtCore.Signal()
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        wdgt = rawForm(axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        """
        pushButton = QPushButton('apply', parent=wdgt)
        hLay = QHBoxLayout()
        wdgt.setLayout(hLay)
        hLay.addWidget(pushButton)
        pushButton.clicked.connect(lambda: wdgt.execute())
        """
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        self.expCorrection = self.defaultExpCorrection
        super(rawForm, self).__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.layer = layer
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
        # signal cannot be directly connected to self.layer.applyToStack
        # due to "QLayer unhashable type" Python error. May be a Pyside bug.
        def dummy():
            self.layer.applyToStack()
            self.layer.parentImage.onImageChanged()
        self.dataChanged.connect(dummy)
        # options
        optionList = ['Auto Brightness', 'Auto Scale']
        self.listWidget1 = optionsWidget(options=optionList, exclusive=False, changed=self.dataChanged)
        self.listWidget1.checkOption(optionList[0])
        optionList = ['Auto White Balance', 'Camera White Balance', 'User White Balance']
        self.listWidget2 = optionsWidget(options=optionList, exclusive=True, changed=self.dataChanged)
        self.options = UDict(self.listWidget1.options, self.listWidget2.options)
        # f is redefined later, but we need to declare it right now
        def f():
            pass

        # clipLimit slider
        self.sliderClip = QSlider(Qt.Horizontal)
        self.sliderClip.setTickPosition(QSlider.TicksBelow)
        self.sliderClip.setRange(25, 800)  #0.25 to 8
        self.sliderClip.setSingleStep(10)

        expLabel = QLabel()
        expLabel.setMaximumSize(150, 30)
        expLabel.setText("Exposure Correction")
        l.addWidget(expLabel)
        hl = QHBoxLayout()
        self.expValue = QLabel()
        font = self.expValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.expValue.setMinimumSize(w, h)
        self.expValue.setMaximumSize(w, h)
        self.expValue.setStyleSheet("QLabel {background-color: white;}")
        hl.addWidget(self.expValue)
        hl.addWidget(self.sliderClip)
        l.addWidget(self.listWidget1)
        l.addWidget(self.listWidget2)
        l.addLayout(hl)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        l.addStretch(1)
        self.setLayout(l)
        self.adjustSize()

        # exp done event handler
        def f():
            self.sliderClip.setEnabled(False)
            self.expValue.setText(str("{:+.2f}".format(self.sliderClip.value() * self.DefaultStep)))
            #self.onUpdateDevelop(self.layer, self.sliderClip.value() * self.DefaultStep)
            self.expCorrection = self.sliderClip.value() * self.DefaultStep
            self.dataChanged.emit()
            self.sliderClip.setEnabled(True)

        # exp value changed event handler
        def g():
            self.expValue.setText(str("{:+.2f}".format(self.sliderClip.value() * self.DefaultStep)))

        self.sliderClip.valueChanged.connect(g)
        self.sliderClip.sliderReleased.connect(f)

        self.sliderClip.setValue(self.defaultExpCorrection / self.DefaultStep)
        self.expValue.setText(str("{:+.2f}".format(self.defaultExpCorrection)))

        def writeToStream(self, outStream):
            layer = self.layer
            outStream.writeQString(layer.actionName)
            outStream.writeQString(layer.name)
            outStream.writeQString(self.listWidget1.selectedItems()[0].text())
            outStream.writeInt32(self.sliderClip.value())
            return outStream

        def readFromStream(self, inStream):
            actionName = inStream.readQString()
            name = inStream.readQString()
            sel = inStream.readQString()
            temp = inStream.readInt32()
            for r in range(self.listWidget1.count()):
                currentItem = self.listWidget1.item(r)
                if currentItem.text() == sel:
                    self.listWidget.select(currentItem)
            self.sliderClip.setValue(temp)
            self.update()
            return inStream