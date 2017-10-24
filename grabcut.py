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

import numpy as np
import cv2
from time import time

from PySide2.QtCore import Qt
#from PySide2.QtGui import QHBoxLayout, QMessageBox, QPushButton, QWidget, QSizePolicy, QVBoxLayout, QColor, QPainter
from PySide2.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QWidget, QSizePolicy, QVBoxLayout, QSpinBox, QLabel
from imgconvert import QImageBuffer

class segmentForm(QWidget):
    """
    Segmentation layer form
    """
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

        self.targetImage = targetImage
        self.mainForm = mainForm
        self.nbIter = 1
        layer.maskIsEnabled = True
        layer.maskIsSelected = True
        buf = QImageBuffer(layer.mask)
        buf[:, :, :] = 255

        pushButton = QPushButton('apply')
        pushButton.clicked.connect(lambda: self.targetImage.getActiveLayer().applyToStack())

        spBox = QSpinBox()
        spBox.setRange(1,10)
        def f(iter):
            self.nbIter = iter
        spBox.valueChanged.connect(f)
        spBoxLabel = QLabel()
        spBoxLabel.setText('Iterations')

        hLay = QHBoxLayout()
        hLay.addWidget(spBoxLabel)
        hLay.addWidget(spBox)
        hLay.addStretch(1)

        vLay = QVBoxLayout()
        vLay.setAlignment(Qt.AlignTop)
        vLay.setContentsMargins(20, 8, 20, 25)  # left, top, right, bottom
        vLay.addLayout(hLay)
        vLay.addWidget(pushButton)

        self.setLayout(vLay)




