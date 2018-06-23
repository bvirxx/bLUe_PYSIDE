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
        # option changed handler
        def g(item):
            self.layer.isClipping = self.options['Clipping Layer']
            self.layer.applyToStack()
        self.listWidget1.onSelect = g
        # layout
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
        vLay.addWidget(pushButton)
        self.setLayout(vLay)
        self.setDefaults()
        self.setWhatsThis(
""" Object extraction  
  Select the object to extract with the rectangle Marquee Tool. Next, click the Apply button. 
  Correct (roughly) if needed the foreground (FG) and the background (BG) regions using the FG and BG tools (Ctrl to undo) and click again the Apply button.
  By default the mask is displayed as a color mask. To view it as an opacity mask, right click on the Segmentation layer row in the right pane and check Enable Mask As > Opacity Mask in the context menu.
  Note. To keep the transparent background save the image in format png or tiff.
"""
                        )

    def setDefaults(self): #TODO 13/06/18 put all initial connect in setDefaults to minimize updates
        self.listWidget1.unCheckAll()
        # initially the layer is not ciipping to show the image to segment.
        # self.listWidget1.checkOption(self.listWidget1.intNames[0])
        # self.layer.isClipping = True





