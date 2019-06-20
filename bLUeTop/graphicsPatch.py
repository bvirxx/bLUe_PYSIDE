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
from PySide2.QtCore import Qt, QRect
from PySide2.QtGui import QImage, QPixmap, QPainter
from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog

from bLUeGui.graphicsForm import baseForm
from bLUeGui.dialog import IMAGE_FILE_EXTENSIONS

from bLUeTop.utils import optionsWidget, UDict


class patchForm (baseForm):
    """
    Seamless cloning form.
    """
    # positioning window size
    pwSize = 200

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        # positioning window
        self.widgetImg = QLabel(parent=parent)
        # stay on top and center
        self.widgetImg.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Dialog )
        # source pixmap
        self.sourcePixmap = None
        self.sourcePixmapThumb = None
        # flag indicating where source pixmap come from
        self.sourceFromFile = False
        # opencv flags
        cv2Flag_dict = {'Normal Clone': cv2.NORMAL_CLONE,
                        'Mixed Clone': cv2.MIXED_CLONE,
                        'Monochrome Transfer': cv2.MONOCHROME_TRANSFER}
        cv2Flags = list(cv2Flag_dict.keys())

        self.layer.cloningMethod = cv2Flag_dict['Normal Clone']
        self.layer.maskIsEnabled = True
        self.layer.maskIsSelected = True
        # mask all pixels, use a semi transparent mask
        self.layer.resetMask(maskAll=True, alpha=128)
        self.layer.autoclone = True

        self.listWidget1 = optionsWidget(options=cv2Flags, exclusive=True, changed=self.dataChanged)
        # init flags
        for i in range(self.listWidget1.count()):
            item = self.listWidget1.item(i)
            item.setData(Qt.UserRole, cv2Flag_dict[item.text()])
        self.options = self.listWidget1.options
        self.listWidget1.checkOption(self.listWidget1.intNames[0])

        optionList2, optionListNames2 = ['opencv', 'blue'], ['OpenCV Cloning', 'bLUe Cloning']
        self.listWidget2 = optionsWidget(options=optionList2, optionNames=optionListNames2,
                                         exclusive=True, changed=self.dataChanged)
        self.listWidget2.checkOption(self.listWidget2.intNames[1])

        self.options = UDict((self.options, self.listWidget2.options))

        pushButton1 = QPushButton('Load Image From File')

        # Load Image button clicked slot
        def f():
            from bLUeTop.QtGui1 import window
            lastDir = str(window.settings.value('paths/dlgdir', '.'))
            dlg = QFileDialog(window, "select", lastDir, " *".join(IMAGE_FILE_EXTENSIONS))
            if dlg.exec_():
                filenames = dlg.selectedFiles()
                newDir = dlg.directory().absolutePath()
                window.settings.setValue('paths/dlgdir', newDir)
                img = QImage(filenames[0])
                # scale img while keeping its aspect ratio
                # into a QPixmap having the same size as self layer
                sourcePixmap = QPixmap.fromImage(img).scaled(self.layer.size(), Qt.KeepAspectRatio)
                self.sourcePixmap = QPixmap(self.layer.size())
                self.sourcePixmap.fill(Qt.black)
                qp = QPainter(self.sourcePixmap)
                qp.drawPixmap(QRect(0, 0, sourcePixmap.width(), sourcePixmap.height()), sourcePixmap)
                qp.end()
                self.sourcePixmapThumb = self.sourcePixmap.scaled(self.pwSize, self.pwSize, aspectMode=Qt.KeepAspectRatio)
                self.widgetImg.setPixmap(self.sourcePixmapThumb)
                self.widgetImg.setFixedSize(self.sourcePixmapThumb.size())
                self.sourceFromFile = True
            self.widgetImg.show()

        pushButton1.clicked.connect(f)
        pushButton2 = QPushButton('Reset')

        # reset button clicked slot
        def g():
            layer = self.layer
            # mask all pixels
            layer.resetMask(maskAll=True, alpha=128)
            layer.setMaskEnabled(color=True)
            # reset clone layer
            layer.xAltOffset, layer.yAltOffset = 0.0, 0.0
            layer.AltZoom_coeff = 1.0
            layer.applyCloning(seamless=False, showTranslated=True)
            layer.parentImage.onImageChanged()
        pushButton2.clicked.connect(g)

        # layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(layout)
        layout.addWidget(self.listWidget2)
        layout.addWidget(self.listWidget1)
        hl = QHBoxLayout()
        hl.addWidget(pushButton1)
        hl.addWidget(pushButton2)
        layout.addLayout(hl)

        self.setDefaults()

        self.setWhatsThis(
                            """
                            <b>Cloning</b> :
                            Seamless replacement of a region of the image by another region from the same image 
                            or by another image (e.g. to erase an object):<br>
                               &nbsp; 1) <b> make sure that the cloning layer is the topmost visible layer</b><br>
                               &nbsp; 2) Select the Unmask/FG tool and paint the pixels to erase 
                                         (use the Mask/BG tool to adjust the mask if needed); <br>
                               &nbsp; 3) Select the drag tool and while pressing <b>Ctrl-Alt</b> use
                                         the mouse to drag or zoom the image shown in the painted region;<br>
                            Eventually use <b>Mask Erode</b> from the layer context menu to smooth mask edges.<br>
                            """
                        )  # end of setWhatsthis

    def setDefaults(self):
        self.enableOptions()
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.layer.cloningMethod = self.listWidget1.checkedItems[0].data(Qt.UserRole)
        #self.widgetImg.setPixmap(QPixmap.fromImage(self.layer.inputImg().scaled(200, 200, aspectMode=Qt.KeepAspectRatio)))
        # init positioning window
        img = self.layer.inputImg()
        if img.rPixmap is None:
            img.rPixmap = QPixmap.fromImage(img)
        self.sourcePixmap = img.rPixmap
        self.sourcePixmapThumb = self.sourcePixmap.scaled(200, 200, aspectMode=Qt.KeepAspectRatio)
        self.widgetImg.setPixmap(self.sourcePixmapThumb)
        self.widgetImg.setFixedSize(self.sourcePixmapThumb.size())
        self.widgetImg.show()
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.dataChanged.connect(self.updateLayer)

    def enableOptions(self):
        if self.options['blue']:
            # make sure disabled options are unchecked
            self.listWidget1.checkOption('Normal Clone')
        for item in [self.listWidget1.item(i) for i in (1, 2) ]:  # mixed clone, monochrome transfer
            if self.options['opencv']:
                item.setFlags(item.flags() | Qt.ItemIsEnabled)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)

    def updateLayer(self):
        """
        data changed slot
        """
        self.enableOptions()
        layer = self.layer
        layer.cloningMethod = self.listWidget1.checkedItems[0].data(Qt.UserRole)
        layer.applyToStack()
        layer.parentImage.onImageChanged()
