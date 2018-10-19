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
import weakref

from PySide2 import QtCore
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QHBoxLayout, QPushButton, QWidget, QSizePolicy, QVBoxLayout, QSpinBox, QLabel

from versatileImg import vImage
from QtGui1 import window
from bLUeGui.graphicsSpline import baseForm
from utils import optionsWidget

class segmentForm(baseForm):
    """
    Segmentation (grabcut) form

        Methods                          Attributes
            getNewWindow                     contourMargin
            __init__                         contourMarginDefault
            setDefaults                      dataChanged
            updateLayer                      iterDefault
            reset                            layer
                                             layerTitle
                                             listWidget1
                                             nbIter
                                             options
                                             spBox
                                             spBox1
                                             start

    """
    dataChanged = QtCore.Signal()
    layerTitle = "Segmentation"
    iterDefault = 3
    contourMarginDefault = 1
    @classmethod
    def getNewWindow(cls, targetImage=None, layer=None, mainForm=None):
        wdgt = segmentForm(layer=layer)
        return wdgt

    def __init__(self, layer=None):
        super(segmentForm, self).__init__()
        self.setWindowTitle('grabcut')
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(200, 200)
        self.setAttribute(Qt.WA_DeleteOnClose)
        # link back to image layer
        # using weak ref for back links
        if type(layer) in weakref.ProxyTypes:
            self.layer = layer
        else:
            self.layer = weakref.proxy(layer)

        pushButton = QPushButton('apply')
        # apply button slot
        def f():
            self.layer.noSegment = False
            self.layer.applyToStack()
            window.label.img.onImageChanged()
            # do manual segmentation only
            layer.noSegment = True
        pushButton.clicked.connect(f)

        pushButton1 = QPushButton('Reset')
        pushButton1.clicked.connect(lambda : self.reset())

        self.spBox = QSpinBox()
        self.spBox.setRange(1,10)
        # spBox Slot
        def f2(iterCount):
            self.spBox.valueChanged.disconnect()
            self.dataChanged.emit()
            self.spBox.valueChanged.connect(f2)
        self.spBox.valueChanged.connect(f2)
        spBoxLabel = QLabel()
        spBoxLabel.setText('Iterations')

        self.spBox1 = QSpinBox()
        self.spBox1.setRange(0, 20)
        spBox1Label = QLabel()
        spBox1Label.setText('Contour Margin')
        # spBox1 slot
        def f1(margin):
            self.spBox1.valueChanged.disconnect()
            self.dataChanged.emit()
            self.spBox1.valueChanged.connect(f1)
        self.spBox1.valueChanged.connect(f1)

        # options
        optionList1, optionNames1 = ['Clipping Layer'], ['Clipping Layer']
        self.listWidget1 = optionsWidget(options=optionList1, optionNames=optionNames1, exclusive=False)
        self.options = self.listWidget1.options
        # option changed slot
        def g(item):
            self.layer.isClipping = self.options['Clipping Layer']
            self.layer.applyToStack()
            self.layer.parentImage.onImageChanged()
        self.listWidget1.onSelect = g

        # dataChanged must be connected to updateLayer in __init__
        # otherwise disconnecting in setDefaults raises an exception
        self.dataChanged.connect(self.updateLayer)

        # attributes initialized in setDefaults, declared here
        # for the sake of correctness
        self.start = None
        self.nbIter = None
        self.contourMargin = None

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
        h3 = QHBoxLayout()
        h3.addWidget(pushButton)
        h3.addWidget(pushButton1)
        vLay.addLayout(h3)
        self.setLayout(vLay)
        self.setDefaults()
        self.setWhatsThis(
""" <b>Object extraction</b><br>  
  Select the object to extract with the rectangle Marquee Tool. Next, click the Apply button.<br>
  Correct (roughly) if needed the foreground (FG) and the background (BG) regions using the FG and BG tools (Ctrl to undo) and click again the Apply button.<br>
  To get a smoother contour increase the value of the Contour Margin and click the Apply Button.<br>
  By default the mask is displayed as a color mask. To view it as an opacity mask, right click on the Segmentation layer row in the right pane and check Enable Mask As > Opacity Mask in the context menu.
  Use the same context menu to copy/paste the object to a new image layer or the mask to another layer.<br>
  
"""
                        )  # end setWhatsThis

    def setDefaults(self):
        self.dataChanged.disconnect(self.updateLayer)
        self.listWidget1.unCheckAll()
        self.spBox.setValue(self.iterDefault)
        self.spBox1.setValue(self.contourMarginDefault)
        self.start = True
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()

    def updateLayer(self):
        self.nbIter = self.spBox.value()
        self.contourMargin = self.spBox1.value()

    def reset(self):
        layer = self.layer
        layer.maskIsEnabled = True
        layer.maskIsSelected = True
        # mask pixels are not yet painted as FG or BG
        # so we mark them as invalid
        layer.mask.fill(vImage.defaultColor_Invalid)
        layer.paintedMask = layer.mask.copy()
        layer.isClipping = False
        self.setDefaults()
        layer.updatePixmap()





