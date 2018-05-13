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
from PySide2.QtGui import QFontMetrics, QTransform
from PySide2.QtWidgets import QGraphicsView, QSizePolicy, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QPushButton

from utils import optionsWidget


class transForm (QGraphicsView):
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
        self.layer = layer
        self.mainForm = mainForm
        # options
        optionList, optionNames = ['Free', 'Rotation', 'Translation'], ['Free Transformation', 'Rotation', 'Translation']
        self.listWidget1 = optionsWidget(options=optionList, optionNames=optionNames, exclusive=True)
        self.options = self.listWidget1.options
        # set initial selection to Perspective
        self.listWidget1.checkOption(optionList[0])

        # option changed handler
        def g(item):
            self.tool.setBaseTransform()
            #self.tool.setVisible(True)
            self.layer.applyToStack()

        self.listWidget1.onSelect = g

        pushButton1 = QPushButton('Reset')
        def f():
            self.tool.resetTrans()
        pushButton1.clicked.connect(f)

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        l.addWidget(self.listWidget1)
        l.addWidget(QLabel('Ctrl+Alt+Drag to change the\ninitial positions of buttons'))
        l.addWidget(pushButton1)
        self.setLayout(l)
        self.adjustSize()

class imageForm(transForm):
    pass

