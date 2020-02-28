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
from bLUeTop.utils import optionsWidget, UDict


class histForm (baseForm):
    """
    Form for histogram viewing
    """
    def __init__(self, targetImage=None, size=200, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.mode = 'Luminosity'
        self.chanColors = [Qt.gray]  # [Qt.red, Qt.green,Qt.blue]
        self.setWindowTitle('Histogram')
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(size, 100)
        self.Label_Hist = QLabel()
        self.Label_Hist.setFocusPolicy(Qt.ClickFocus)
        self.Label_Hist.setMaximumSize(140000, 140000)
        self.Label_Hist.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setStyleSheet("QListWidget{border: 0px; font-size: 12px}")

        # options
        options1, optionNames1 = ['Original Image'], ['Source']
        self.listWidget1 = optionsWidget(options=options1, optionNames=optionNames1, exclusive=False)
        self.listWidget1.setMaximumSize(self.listWidget1.sizeHintForColumn(0) + 5,
                                        self.listWidget1.sizeHintForRow(0) * len(options1))
        options2, optionNames2 = ['R', 'G', 'B', 'L'], ['R', 'G', 'B', 'L']
        self.listWidget2 = optionsWidget(options=options2, optionNames=optionNames2, exclusive=False, flow=optionsWidget.LeftToRight)
        self.listWidget2.setFixedSize((self.listWidget2.sizeHintForRow(0) + 15) * len(options2), 20)

        # default: show color hists only
        for i in range(3):
            self.listWidget2.checkOption(self.listWidget2.intNames[i])
        self.options = UDict((self.listWidget1.options, self.listWidget2.options))
        self.setWhatsThis("""
        <b>Histogram</b><br>
        The histogram shows the color ditributions for the edited image, unless
        the <I>Source</I> option is checked. 
        """)

        def onSelect(item):
            try:
                self.targetImage.onImageChanged()
                self.Label_Hist.update()
            except AttributeError:
                return

        self.listWidget1.onSelect = onSelect
        self.listWidget2.onSelect = onSelect

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
        self.adjustSize()




