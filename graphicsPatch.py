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
import cv2
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QGraphicsView, QSizePolicy, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox

from MarkedImg import vImage
from imgconvert import QImageBuffer
from utils import optionsWidget, dlgWarn


class patchForm (QGraphicsView):
    """
    Seamless cloning form
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=200, layer=None, parent=None, mainForm=None):
        wdgt = patchForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.img = targetImage
        self.layer = layer
        self.mainForm = mainForm
        # options
        options_dict = {'Normal Clone':cv2.NORMAL_CLONE, 'Mixed Clone':cv2.MIXED_CLONE, 'Monochrome Transfer':cv2.MONOCHROME_TRANSFER}
        options = list(options_dict.keys())
        self.layer.cloningMethod = options_dict['Normal Clone']
        self.layer.maskIsEnabled = True
        self.layer.maskIsSelected = True
        # mask all pixels
        self.layer.resetMask(maskAll=True)
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
        """
        opList2 = ['Auto Cloning', 'Press Button To Clone']
        self.listWidget2 = optionsWidget(options=opList2)
        def onSelect2(item):
            keepCloned = self.listWidget2.options[opList2[0]]
            self.layer.keepCloned = keepCloned
            self.layer.maskIsEnabled = not keepCloned
            self.layer.maskIsSelected = not keepCloned
        self.listWidget2.onSelect = onSelect2
        sel = opList2[1]
        self.listWidget2.select(self.listWidget2.items[sel])
        """
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
            #layer.maskIsEnabled = False # done in onselect2
            #layer.maskIsSelected = False
        pushButton1.clicked.connect(f)
        pushButton2 = QPushButton('Reset')
        def g():
            layer = self.layer
            # mask all pixels
            layer.resetMask(maskAll=True)
            layer.maskIsSelected = True
            # reset clone layer
            layer.xAltOffset, layer.yAltOffset = 0.0, 0.0
            layer.AltZoom_coeff = 1.0
            layer.applyCloning(seamless=False)
            layer.parentImage.onImageChanged()
        pushButton2.clicked.connect(g)
        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
        hl = QHBoxLayout()
        l.addLayout(hl)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(l)
        l.addWidget(self.listWidget1)
        # l.addWidget(self.listWidget2)
        l.addWidget(pushButton1)
        l.addWidget(pushButton2)
        self.setWhatsThis(
"""
Seamless replacement of a region of the image by another region from the same image (e.g. to erase an object):
   1) Select the Unmask/FG tool and paint the pixels to erase (Use the Mask/BG tool to adjust if needed);
   2) Select the drag tool and while pressing Ctrl-Alt on the keyboard drag the image shown in the painted area;
When done click the Clone button to start the cloning. 
"""
                        )
class maskForm (QGraphicsView):
    """
    Knitting form (cloning an imported image)
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=200, layer=None, parent=None, mainForm=None):
        wdgt = maskForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.img = targetImage
        self.layer = layer
        self.mainForm = mainForm
        # options
        options_dict = {'Normal Clone':cv2.NORMAL_CLONE, 'Mixed Clone':cv2.MIXED_CLONE, 'Monochrome Transfer':cv2.MONOCHROME_TRANSFER}
        options = list(options_dict.keys())
        self.layer.cloningMethod = options_dict['Normal Clone']
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
        self.layer.sourceIndex = 0
        label1 = QLabel('Source')
        self.sourceCombo = QComboBox()
        for i, item  in enumerate(self.img.layersStack):
            self.sourceCombo.addItem(item.name, i)
        # combo box item chosen event handler
        def g(ind):
            self.layer.sourceIndex = ind
            self.img.getActiveLayer().applyToStack()
            self.img.onImageChanged()
        hl = QHBoxLayout()
        hl.addWidget(label1)
        hl.addWidget(self.sourceCombo)
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
        l.addLayout(hl)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(l)
        # set initial selection to normal cloning
        item = self.listWidget1.items[options[0]]
        item.setCheckState(Qt.Checked)
        self.listWidget1.select(item)
        l.addWidget(self.listWidget1)
        pushButton1 = QPushButton('Seamless Knit')
        def f():
            self.targetImage.getActiveLayer().applyKnitting()
        pushButton1.clicked.connect(f)
        l.addWidget(pushButton1)

