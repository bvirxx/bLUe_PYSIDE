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

import cv2
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox

from bLUeGui.graphicsForm import baseForm
from bLUeGui.dialog import dlgWarn
from bLUeGui.memory import weakProxy

from versatileImg import vImage
from utils import optionsWidget

class patchForm (baseForm):
    """
    Seamless cloning form
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=200, layer=None, parent=None):
        wdgt = patchForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        # back links to image
        self.targetImage = weakProxy(targetImage)
        self.img = weakProxy(targetImage)
        self.layer = weakProxy(layer)
        # options
        options_dict = {'Normal Clone':cv2.NORMAL_CLONE, 'Mixed Clone':cv2.MIXED_CLONE, 'Monochrome Transfer':cv2.MONOCHROME_TRANSFER}
        options = list(options_dict.keys())
        self.layer.cloningMethod = options_dict['Normal Clone']
        self.layer.maskIsEnabled = True
        self.layer.maskIsSelected = True
        # mask all pixels, use a semi transparent mask
        self.layer.resetMask(maskAll=True, alpha=128)
        self.layer.autoclone = False
        self.layer.cloningMethod = cv2.NORMAL_CLONE
        self.options={}
        for op in options:
            self.options[op] = False
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        sel = options[0]
        self.listWidget1.select(self.listWidget1.items[sel])
        self.options[sel] = True
        # select event handler
        def onSelect1(item):
            for key in self.options:
                self.options[key] = item is self.listWidget1.items[key]
                if self.options[key]:
                    self.layer.cloningMethod = options_dict[key]
        self.listWidget1.onSelect = onSelect1
        # set initial selection to normal cloning
        item = self.listWidget1.items[options[0]]
        item.setCheckState(Qt.Checked)
        self.listWidget1.select(item)
        pushButton1 = QPushButton('Clone')
        # button clicked event handler
        def f():
            layer = self.layer
            if vImage.isAllMasked(layer.mask):
                dlgWarn('Nothing to clone: unmask some pixels')
                return
            if layer.xAltOffset == 0.0 and layer.yAltOffset == 0.0:
                dlgWarn('Nothing to clone: Ctr+Alt+Drag the image ')
                return
            layer.applyCloning(seamless=True)
            layer.parentImage.onImageChanged()
        pushButton1.clicked.connect(f)
        pushButton2 = QPushButton('Reset')
        def g():
            layer = self.layer
            # mask all pixels
            layer.resetMask(maskAll=True, alpha=128)
            layer.maskIsSelected = True
            # reset clone layer
            layer.xAltOffset, layer.yAltOffset = 0.0, 0.0
            layer.AltZoom_coeff = 1.0
            layer.applyCloning(seamless=False)
            layer.parentImage.onImageChanged()
        pushButton2.clicked.connect(g)
        # layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(layout)
        layout.addWidget(self.listWidget1)
        hl = QHBoxLayout()
        hl.addWidget(pushButton1)
        hl.addWidget(pushButton2)
        layout.addLayout(hl)
        self.setWhatsThis(
"""
<b>Cloning</b> :
Seamless replacement of a region of the image by another region from the same image (e.g. to erase an object):<br>
   &nbsp; 1) Select the Unmask/FG tool and paint the pixels to erase (use the Mask/BG tool to adjust if needed); <br>
   &nbsp; 2) Select the drag tool and while pressing <b>Ctrl-Alt</b>, drag or zoom the image shown in the
             painted region with the mouse;<br>
   &nbsp; 3) Click the Clone button to start the cloning.<br>
Redo steps 1 to 3 until the result is satisfactory. Eventually use <b>Mask Erode</b> from the layer context menu
to smooth mask edges.<br>
<b> While executing steps 1 to 4 above, make sure that the cloning layer is the topmost visible layer</b><br>

"""
                        )

