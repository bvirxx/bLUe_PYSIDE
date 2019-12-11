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
        options2, optionNames2 = ['R'], ['R']
        self.listWidget2 = optionsWidget(options=options2, optionNames=optionNames2, exclusive=False)
        self.listWidget2.setMaximumSize(self.listWidget2.sizeHintForColumn(0) + 5,
                                        self.listWidget2.sizeHintForRow(0) * len(options2))
        options3, optionNames3 = ['G'], ['G']
        self.listWidget3 = optionsWidget(options=options3, optionNames=optionNames3, exclusive=False)
        self.listWidget3.setMaximumSize(self.listWidget3.sizeHintForColumn(0) + 5,
                                        self.listWidget3.sizeHintForRow(0) * len(options3))
        options4, optionNames4 = ['B'], ['B']
        self.listWidget4 = optionsWidget(options=options4, optionNames=optionNames4, exclusive=False)
        self.listWidget4.setMaximumSize(self.listWidget4.sizeHintForColumn(0) + 5,
                                        self.listWidget4.sizeHintForRow(0) * len(options4))

        # default: show color hists
        for w in [self.listWidget2, self.listWidget3, self.listWidget4]:
            w.checkOption(w.intNames[0])
        self.options = UDict((self.listWidget1.options, self.listWidget2.options, self.listWidget3.options, self.listWidget4.options))
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
        self.listWidget3.onSelect = onSelect
        self.listWidget4.onSelect = onSelect

        # layout
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 2)
        h.addWidget(self.listWidget1)
        h.addWidget(self.listWidget2)
        h.addWidget(self.listWidget3)
        h.addWidget(self.listWidget4)
        vl = QVBoxLayout()
        vl.setAlignment(Qt.AlignTop)
        vl.addWidget(self.Label_Hist)
        vl.addLayout(h)
        vl.setContentsMargins(0, 0, 0, 2)  # left, top, right, bottom
        self.setLayout(vl)
        self.adjustSize()




