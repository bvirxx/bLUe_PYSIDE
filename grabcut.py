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
from PySide2.QtWidgets import QHBoxLayout, QPushButton, QWidget, QSizePolicy, QVBoxLayout, QSpinBox, QLabel

from QtGui1 import window
from utils import optionsWidget

class segmentForm(QWidget):
    """
    Form for segmentation (grabcut)
    """
    dataChanged = QtCore.Signal()
    layerTitle = "Segmentation"
    iterDefault = 3
    contourMarginDefault = 1
    @classmethod
    def getNewWindow(cls, targetImage=None, layer=None, mainForm=None):
        wdgt = segmentForm(targetImage=targetImage, layer=layer, mainForm=mainForm)
        return wdgt

    def __init__(self, targetImage=None, layer=None, mainForm=None):
        super(segmentForm, self).__init__()
        self.setWindowTitle('grabcut')
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(200, 200)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.layer = layer
        self.targetImage = targetImage
        self.mainForm = mainForm
        self.dataChanged.connect(self.updateLayer)
        def f():
            self.targetImage.getActiveLayer().applyToStack()
            window.label.img.onImageChanged()
        pushButton = QPushButton('apply')
        pushButton.clicked.connect(f)
        self.spBox = QSpinBox()
        self.spBox.setRange(1,10)
        def f(iter):
            self.spBox.valueChanged.disconnect()
            self.dataChanged.emit()
            self.spBox.valueChanged.connect(f)
        self.spBox.valueChanged.connect(f)
        spBoxLabel = QLabel()
        spBoxLabel.setText('Iterations')
        self.spBox1 = QSpinBox()
        self.spBox1.setRange(0, 20)
        spBox1Label = QLabel()
        spBox1Label.setText('Contour Margin')
        def f1(iter):
            self.spBox1.valueChanged.disconnect()
            self.dataChanged.emit()
            self.spBox1.valueChanged.connect(f1)
        self.spBox1.valueChanged.connect(f1)
        # options
        optionList1, optionNames1 = ['Clipping Layer'], ['Clipping Layer']
        self.listWidget1 = optionsWidget(options=optionList1, optionNames=optionNames1, exclusive=False)
        self.options = self.listWidget1.options
        # option changed handler
        def g(item):
            self.layer.isClipping = self.options['Clipping Layer']
            self.layer.applyToStack()
            self.layer.parentImage.onImageChanged()
        self.listWidget1.onSelect = g
        # layout
        hLay = QHBoxLayout()
        hLay.addWidget(spBoxLabel)
        hLay.addWidget(self.spBox)
        hLay.addStretch(1)
        hLay1 = QHBoxLayout()
        hLay1.addWidget(spBox1Label)
        hLay1.addWidget(self.spBox1)
        hLay1.addStretch(1)
        h2 = QHBoxLayout()
        h2.addWidget(self.listWidget1)
        vLay = QVBoxLayout()
        vLay.setAlignment(Qt.AlignTop)
        vLay.setContentsMargins(20, 8, 20, 25)  # left, top, right, bottom
        vLay.addLayout(hLay)
        vLay.addLayout(hLay1)
        vLay.addLayout(h2)
        vLay.addWidget(pushButton)
        self.setLayout(vLay)
        self.setDefaults()
        self.setWhatsThis(
""" Object extraction  
  Select the object to extract with the rectangle Marquee Tool. Next, click the Apply button. 
  Correct (roughly) if needed the foreground (FG) and the background (BG) regions using the FG and BG tools (Ctrl to undo) and click again the Apply button.
  To get a smoother contour increase the value of the Contour Margin and click the Apply Button.
  By default the mask is displayed as a color mask. To view it as an opacity mask, right click on the Segmentation layer row in the right pane and check Enable Mask As > Opacity Mask in the context menu.
  To copy the object to a new image layer use the right pane Context Menu : Copy Image to Clipboard and, next, Paste Image.
  When done, toggle off the visibility of the segmentation layer.
"""
                        )  # end setWhatsThis

    def setDefaults(self): #TODO 13/06/18 put all initial connect in setDefaults to minimize updates
        self.listWidget1.unCheckAll()
        self.spBox.setValue(self.iterDefault)
        self.spBox1.setValue(self.contourMarginDefault)
        self.start = True
        self.dataChanged.emit()
        # initially the layer is not ciipping to show the image to segment.
        # self.listWidget1.checkOption(self.listWidget1.intNames[0])
        # self.layer.isClipping = True

    def updateLayer(self):
        self.nbIter = self.spBox.value()
        self.contourMargin = self.spBox1.value()




