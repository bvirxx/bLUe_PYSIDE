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

from PySide2.QtCore import Qt, QSize
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QLabel, QHBoxLayout

from graphicsLUT import baseForm
from utils import optionsWidget

class histForm (baseForm):
    """
    Form for displaying histograms
    """
    def __init__(self, targetImage=None, size=200, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumSize(size, 100)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.img = targetImage
        # link back to image layer
        # using weak ref for back links
        self.layer = layer if (layer is None  or type(layer) in weakref.ProxyTypes) else weakref.proxy(layer)
        self.Label_Hist = QLabel()
        self.Label_Hist.setScaledContents(True)
        self.Label_Hist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setStyleSheet("QListWidget{background-color: rgb(200,200,200); selection-background-color: rgb(200,200,200); border: 0px; font-size: 9px}")

        # options. We want the items displayed horizontally, so we make 2 lists
        options1 = ['Original Image']
        self.listWidget1 = optionsWidget(options=options1, exclusive=False)
        self.listWidget1.item(0).setSizeHint(QSize(100,10))
        self.listWidget1.setMaximumSize(self.listWidget1.sizeHintForColumn(0) + 5, self.listWidget1.sizeHintForRow(0) * len(options1))

        options2 = ['Color Chans']
        self.listWidget2 = optionsWidget(options=options2, exclusive=False)
        self.listWidget2.item(0).setSizeHint(QSize(100, 10))
        self.listWidget2.setMaximumSize(self.listWidget2.sizeHintForColumn(0) + 5, self.listWidget2.sizeHintForRow(0) * len(options2))
        # default: show color hists
        self.listWidget2.item(0).setCheckState(Qt.Checked)

        self.options = {option : True for option in options1 + options2}
        def onSelect1(item):
            self.options[options1[0]] = item.checkState() is Qt.Checked
            self.targetImage.onImageChanged()
            self.Label_Hist.update()

        def onSelect2(item):
            self.options[options2[0]] = item.checkState() is Qt.Checked
            self.targetImage.onImageChanged()
            self.Label_Hist.update()

        self.listWidget1.onSelect = onSelect1
        self.listWidget2.onSelect = onSelect2

        # layout
        h = QHBoxLayout()
        h.addWidget(self.listWidget1)
        h.addWidget(self.listWidget2)
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        l.addWidget(self.Label_Hist)
        l.addLayout(h)
        l.setContentsMargins(0, 0, 0, 2)  # left, top, right, bottom
        self.setLayout(l)




