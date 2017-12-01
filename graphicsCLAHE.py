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
from time import time

import cv2
import numpy as np
from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QGraphicsView, QSizePolicy, QVBoxLayout, QSlider, QLabel, QHBoxLayout

from colorConv import Lab2sRGBVec
from imgconvert import QImageBuffer

# Contrast Limited Adaptive Histogram Equalization.
from utils import optionsWidget

"""
def Clahe(imgLBuf):
    #UNUSED
    start = time()
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe.setClipLimit(0.8)
    res = clahe.apply((imgLBuf[:,:,0]*255.0).astype(np.uint8))
    imgLBuf[:,:,0] = res.astype(np.float) / 255
    return imgLBuf
    ndsRGBImg1 = Lab2sRGBVec(imgLBuf)
    # clipping is mandatory here : numpy bug ?
    ndsRGBImg1 = np.clip(ndsRGBImg1, 0, 255)
    print("clahe %.2f" % (time() - start))
    return ndsRGBImg
"""
class CLAHEForm (QGraphicsView):
    defaultClipLimit = 0.25
    @classmethod
    def getNewWindow(cls, targetImage=None, size=500, layer=None, parent=None, mainForm=None):
        wdgt = CLAHEForm(size=size, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        """
        pushButton = QPushButton('apply', parent=wdgt)
        hLay = QHBoxLayout()
        wdgt.setLayout(hLay)
        hLay.addWidget(pushButton)
        pushButton.clicked.connect(lambda: wdgt.execute())
        """
        return wdgt

    def __init__(self, targetImage=None, size=500, layer=None, parent=None, mainForm=None): # TODO 01/12/17 remove param targetImage
        super(CLAHEForm, self).__init__(parent=parent)
        #self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(size, size)  # default width 200 doesn't fit the length of option names
        self.setAttribute(Qt.WA_DeleteOnClose)
        #self.img = targetImage
        self.layer = layer
        #self.defaultClip = self.defaultClipLimit
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)

        # f is defined later, but we need to declare it righjt now
        def f():
            pass

        # options
        self.options = None
        """
        self.options = {'use Chromatic Adaptation': True}
        options = ['use Photo Filter', 'use Chromatic Adaptation']
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        self.listWidget1.select(self.listWidget1.items['use Chromatic Adaptation'])
        self.listWidget1.setMaximumSize(self.listWidget1.sizeHintForColumn(0) + 5, self.listWidget1.sizeHintForRow(0) * len(options) + 5)
        def onSelect1(item):
            self.options['use Chromatic Adaptation'] = item is self.listWidget1.items['use Chromatic Adaptation']
            f()
        self.listWidget1.onSelect = onSelect1
        l.addWidget(self.listWidget1)
        """

        # clipLimit slider
        self.sliderClip = QSlider(Qt.Horizontal)
        self.sliderClip.setTickPosition(QSlider.TicksBelow)
        self.sliderClip.setRange(1, 50)
        self.sliderClip.setSingleStep(1)

        tempLabel = QLabel()
        tempLabel.setMaximumSize(150, 30)
        tempLabel.setText("Clip Limit")
        l.addWidget(tempLabel)
        hl = QHBoxLayout()
        self.tempValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.tempValue.setMinimumSize(w, h)
        self.tempValue.setMaximumSize(w, h)
        self.tempValue.setStyleSheet("QLabel {background-color: white;}")
        hl.addWidget(self.tempValue)
        hl.addWidget(self.sliderClip)
        l.addLayout(hl)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        l.addStretch(1)
        self.setLayout(l)
        self.adjustSize()

        # temp done event handler
        def f():
            self.sliderClip.setEnabled(False)
            self.tempValue.setText(str("{:d}".format(self.sliderClip.value())))
            self.onUpdateContrast(self.layer, self.sliderClip.value() / 20.0)
            self.sliderClip.setEnabled(True)

        # temp value changed event handler
        def g():
            self.tempValue.setText(str("{:d}".format(self.sliderClip.value())))
            #self.previewWindow.setPixmap()

        self.sliderClip.valueChanged.connect(g)
        self.sliderClip.sliderReleased.connect(f)

        self.sliderClip.setValue(self.defaultClipLimit * 20)
        self.tempValue.setText(str("{:d}".format(int(self.defaultClipLimit * 20))))

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