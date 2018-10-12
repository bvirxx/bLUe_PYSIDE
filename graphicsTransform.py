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

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

from graphicsLUT import baseForm
from utils import optionsWidget, UDict


class transForm (baseForm):
    """
    Geometric transformation form
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=200, layer=None, parent=None, mainForm=None):
        wdgt = transForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super(transForm, self).__init__(parent=parent)
        self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.img = targetImage
        # link back to image layer
        # using weak ref for back links
        if type(layer) in weakref.ProxyTypes:
            self.layer = layer
        else:
            self.layer = weakref.proxy(layer)
        self.mainForm = mainForm
        # options
        optionList1, optionNames1 = ['Free', 'Rotation', 'Translation'], ['Free Transformation', 'Rotation', 'Translation']
        self.listWidget1 = optionsWidget(options=optionList1, optionNames=optionNames1, exclusive=True)
        optionList2, optionNames2 = ['Transparent'], ['Set Transparent Pixels To Black']
        self.listWidget2 = optionsWidget(options=optionList2, optionNames=optionNames2, exclusive=False)
        self.options = UDict(self.listWidget1.options, self.listWidget2.options)
        # set initial selection to Perspective
        self.listWidget1.checkOption(optionList1[0])
        # option changed handler
        def g(item):
            l = self.layer
            l.tool.setBaseTransform()
            #self.layer.tool.setVisible(True)
            l.applyToStack()
            l.parentImage.onImageChanged()
        self.listWidget1.onSelect = g
        self.listWidget2.onSelect = g
        pushButton1 = QPushButton(' Reset Transformation ')
        pushButton1.adjustSize()
        def f():
            self.layer.tool.resetTrans()
        pushButton1.clicked.connect(f)
        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        l.addWidget(self.listWidget1)
        l.addWidget(self.listWidget2)
        hl = QHBoxLayout()
        hl.setAlignment(Qt.AlignHCenter)
        hl.addWidget(pushButton1)
        l.addLayout(hl)
        self.setLayout(l)
        self.adjustSize()
        self.setWhatsThis(
"""
<b>Geometric transformation :</b><br>
  Choose a transformation type and drag either corner of the image using the small square red buttons.<br>
  Ctrl+Alt+Drag to change the <b>initial positions</b> of buttons.
""")

class imageForm(transForm):
    pass

