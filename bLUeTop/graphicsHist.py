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
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QLabel, QHBoxLayout

from bLUeGui.graphicsForm import baseForm
from bLUeTop.utils import optionsWidget


class histForm (baseForm):
    """
    Form for histogram viewing
    """
    def __init__(self, targetImage=None, size=200, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumSize(size, 100)
        self.Label_Hist = QLabel()
        self.Label_Hist.setFocusPolicy(Qt.ClickFocus)
        self.Label_Hist.setScaledContents(True)
        self.Label_Hist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setStyleSheet("QListWidget{border: 0px; font-size: 12px}")

        # options
        options1, optionNames1 = ['Original Image'], ['Source Image']
        self.listWidget1 = optionsWidget(options=options1, optionNames=optionNames1, exclusive=False)
        self.listWidget1.setMaximumSize(self.listWidget1.sizeHintForColumn(0) + 5,
                                        self.listWidget1.sizeHintForRow(0) * len(options1))
        options2 = ['Color Chans']
        self.listWidget2 = optionsWidget(options=options2, exclusive=False)
        self.listWidget2.setMaximumSize(self.listWidget2.sizeHintForColumn(0) + 5,
                                        self.listWidget2.sizeHintForRow(0) * len(options2))
        # default: show color hists
        self.listWidget2.item(0).setCheckState(Qt.Checked)

        self.options = {option: True for option in options1 + options2}
        self.setWhatsThis("""
        <b>Image Histogram</b><br>
        The histogram shows the color ditributions for the final image unless
        the <I>Source Image</I> option is checked. 
        """)

        def onSelect1(item):
            self.options[options1[0]] = item.checkState() is Qt.Checked
            try:
                self.targetImage.onImageChanged()
                self.Label_Hist.update()
            except AttributeError:
                return

        def onSelect2(item):
            self.options[options2[0]] = item.checkState() is Qt.Checked
            try:
                self.targetImage.onImageChanged()
                self.Label_Hist.update()
            except AttributeError:
                return

        self.listWidget1.onSelect = onSelect1
        self.listWidget2.onSelect = onSelect2

        # layout
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 2)
        h.addWidget(self.listWidget1)
        h.addWidget(self.listWidget2)
        vl = QVBoxLayout()
        vl.setAlignment(Qt.AlignTop)
        vl.addWidget(self.Label_Hist)
        vl.addLayout(h)
        vl.setContentsMargins(0, 0, 0, 2)  # left, top, right, bottom
        self.setLayout(vl)




