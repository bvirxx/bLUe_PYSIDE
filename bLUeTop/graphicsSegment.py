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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QLabel

from bLUeTop.versatileImg import vImage
from bLUeTop.QtGui1 import window
from bLUeGui.graphicsForm import baseForm
from bLUeTop.utils import optionsWidget, QbLUeSpinBox


class segmentForm(baseForm):
    """
    Segmentation (grabcut) form
    """
    layerTitle = "Segmentation"
    iterDefault = 3
    contourMarginDefault = 0

    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = segmentForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        return wdgt
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.setWindowTitle('grabcut')
        button0 = QPushButton('Segment')

        # button slot
        def f():
            self.layer.noSegment = False
            self.layer.applyToStack()
            window.label.img.onImageChanged()
            # manual segmentation only
            self.layer.noSegment = True

        button0.clicked.connect(f)

        button1 = QPushButton('Reset')
        button1.clicked.connect(self.reset)

        self.spBox = QbLUeSpinBox()
        self.spBox.setRange(1, 10)

        # spBox slot
        def f2(iterCount):
            self.nbIter = self.spBox.value()
            self.dataChanged.emit()

        self.spBox.valueChanged.connect(f2)
        spBoxLabel = QLabel()
        spBoxLabel.setText('Iterations')

        self.spBox1 = QbLUeSpinBox()
        self.spBox1.setRange(0, 20)
        spBox1Label = QLabel()
        spBox1Label.setText('Contour Redo Radius')

        # spBox1 slot
        def f1(margin):
            self.contourMargin = self.spBox1.value()
            self.dataChanged.emit()

        self.spBox1.valueChanged.connect(f1)

        # options
        optionList1, optionNames1 = ['Clipping Layer'], ['Clipping Layer']
        self.listWidget1 = optionsWidget(options=optionList1, optionNames=optionNames1, exclusive=False,
                                         changed=self.dataChanged)
        self.options = self.listWidget1.options

        # option changed slot
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
        h3 = QHBoxLayout()
        h3.addWidget(button0)
        h3.addWidget(button1)
        vLay.addLayout(h3)
        self.setLayout(vLay)

        self.setDefaults()
        self.setWhatsThis(
            """ <b>Segmentation (Object extraction)</b><br>  
              Select the object to extract with the rectangle Marquee Tool. Next, press the <i>Segment</i> button.<br>
              The background of the segmented image is transparent : to <b>mask the underlying layers</b> check the
              option <i>Clipping Layer.</i><br>
              To <b>redo the segmentation of a region</b> (e.g. a border area) hold down the Ctrl key while painting the area
              with the foreground (FG) or background (BG) tool and next press again <i>Segment.</i><br>
              To <b>manually correct the selection</b>, paint eventual misclassed pixels with the foreground (FG) or background (BG) tool.<br>
              To <b>redo the segmentation of the whole contour</b> set <i>Contour Redo Radius</i> to a value >= 1 and
              press <i>Segment</i>. Note that setting <i>Contour Redo Radius</i> to a value >= 1 may undo some manual corrections.<br>
              To <b>smooth the contour</b> right click the layer row in the <i>Layers</i> panel
              and choose <i>Smooth Mask</i> from the context menu.<br>
            """
        )  # end setWhatsThis

    def setDefaults(self):
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.listWidget1.unCheckAll()
        self.spBox.setValue(self.iterDefault)
        self.spBox1.setValue(self.contourMarginDefault)
        # self.start = True
        self.contourMargin = self.contourMarginDefault
        self.nbIter = self.iterDefault
        self.dataChanged.connect(self.updateLayer)

    def updateLayer(self):
        self.layer.isClipping = self.options['Clipping Layer']
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def reset(self):
        layer = self.layer
        layer.noSegment = True
        layer.maskIsEnabled = True
        layer.maskIsSelected = True
        # mask pixels are not yet painted as FG or BG
        # so we mark them as invalid
        layer.mask.fill(vImage.defaultColor_Invalid)
        # layer.paintedMask = layer.mask.copy()
        layer.isClipping = False
        self.setDefaults()
        self.dataChanged.emit()
        layer.updatePixmap()

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [optionsWidget, QbLUeSpinBox]:
                d[a] = obj.__getstate__()
        return d

    def __setstate__(self, d):
        for name in d['state']:
            obj = getattr(self, name, None)
            if type(obj) in [optionsWidget, QbLUeSpinBox]:
                obj.__setstate__(d['state'][name])
