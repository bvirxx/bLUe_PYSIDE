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

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QHBoxLayout, QPushButton, QWidget, QSizePolicy, QVBoxLayout, QSpinBox, QLabel

from QtGui1 import window
from utils import optionsWidget


class segmentForm(QWidget):
    """
    Form for segmentation (grabcut)
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
        self.layer = layer
        self.targetImage = targetImage
        self.mainForm = mainForm
        self.nbIter = 1

        def f():
            self.targetImage.getActiveLayer().applyToStack()
            window.label.img.onImageChanged()
        pushButton = QPushButton('apply')
        pushButton.clicked.connect(f)

        spBox = QSpinBox()
        spBox.setRange(1,10)
        def f(iter):
            self.nbIter = iter
        spBox.valueChanged.connect(f)
        spBoxLabel = QLabel()
        spBoxLabel.setText('Iterations')

        optionList1, optionNames1 = ['Clipping Layer'], ['Clipping Layer']
        self.listWidget1 = optionsWidget(options=optionList1, optionNames=optionNames1, exclusive=False)
        self.options = self.listWidget1.options
        # set initial selection to Clipping
        self.listWidget1.checkOption(optionList1[0])
        # option changed handler
        def g(item):
            self.layer.isClipping = self.options['Clipping Layer']
            self.layer.applyToStack()
        self.listWidget1.onSelect = g

        hint = 'Select some background and/or\nforeground pixels with the selection tools\nand apply'
        self.statusLabel = QLabel(text=hint)
        hLay = QHBoxLayout()
        hLay.addWidget(spBoxLabel)
        hLay.addWidget(spBox)
        hLay.addStretch(1)

        h2 = QHBoxLayout()
        h2.addWidget(self.listWidget1)

        vLay = QVBoxLayout()
        vLay.setAlignment(Qt.AlignTop)
        vLay.setContentsMargins(20, 8, 20, 25)  # left, top, right, bottom
        vLay.addLayout(hLay)
        vLay.addLayout(h2)
        vLay.addWidget(self.statusLabel)
        vLay.addWidget(pushButton)
        self.setLayout(vLay)




