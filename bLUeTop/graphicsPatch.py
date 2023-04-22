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
from os.path import basename

import cv2
from PySide6.QtCore import Qt, QPoint, QPointF, QRect
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog

from bLUeGui.graphicsForm import baseForm
from bLUeGui.dialog import IMAGE_FILE_EXTENSIONS, dlgWarn

from bLUeTop.utils import optionsWidget, UDict, QImageFromFile


class BWidgetImg(QLabel):
    """
    Pointing window for cloning source image. It manages
    the pointing cursor.
    """

    def __init__(self, *args, **kwargs):
        """

       :param args:
       :type args:
       :param kwargs: parent should be the graphic form
       :type kwargs:
        """
        super().__init__(*args, **kwargs)
        if 'parent' in kwargs:
            self.grForm = kwargs['parent']

    def __del__(self):
        print('widgetImg deleted')

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if ev.modifiers() != Qt.ControlModifier | Qt.AltModifier:
            return
        grForm = self.grForm
        pos = ev.position()
        # set source starting point
        grForm.layer.sourceX = pos.x() * grForm.layer.width() / grForm.sourcePixmapThumb.width()
        grForm.layer.sourceY = pos.y() * grForm.layer.height() / grForm.sourcePixmapThumb.height()
        grForm.layer.cloningState = 'start'

    def paintEvent(self, e):
        super().paintEvent(e)
        qp = QPainter(self)
        grForm = self.grForm
        if grForm.layer.marker is not None:
            x, y = grForm.layer.marker.x() * grForm.sourcePixmapThumb.width() / grForm.sourcePixmap.width(), \
                   grForm.layer.marker.y() * grForm.sourcePixmapThumb.height() / grForm.sourcePixmap.height()
            qp.drawEllipse(x, y, 10, 10)


class patchForm(baseForm):
    """
    Seamless cloning form.
    """
    # positioning window size
    pwSize = 200

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        # source window
        self.widgetImg = BWidgetImg(parent=self)
        self.widgetImg.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.widgetImg.setAttribute(Qt.WA_DeleteOnClose, on=False)
        self.widgetImg.setWindowTitle("Source")
        self.widgetImg.optionName = 'source'  # needed by subcontrol visibility manager
        self.dockT = None
        # source
        self.sourceImage = None
        self.sourcePixmap = None
        self.sourcePixmapThumb = None
        # opencv flags
        cv2Flag_dict = {'Normal Clone': cv2.NORMAL_CLONE,
                        'Mixed Clone': cv2.MIXED_CLONE,
                        'Monochrome Transfer': cv2.MONOCHROME_TRANSFER}
        cv2Flags = list(cv2Flag_dict.keys())

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

        def h():
            if self.dockT is None:
                return
            self.dockT.setVisible(self.listWidget3.options['source'])

        optionList3, optionListNames3 = ['source'], ['Show Source Image']
        self.listWidget3 = optionsWidget(options=optionList3, optionNames=optionListNames3,
                                         exclusive=False, changed=h)

        self.listWidget3.unCheckAll()
        self.listWidget3.setEnabled(False)

        self.options = UDict((self.options, self.listWidget2.options, self.listWidget3.options))

        pushButton1 = QPushButton('Load Source')
        pushButton2 = QPushButton('Reset')

        # Load Image button clicked slot
        def f():
            window = self.mainForm
            if self.sourceImage is not None:
                dlgWarn('A source image is already open', 'Reset the cloning layer before\nloading a new image')
            lastDir = str(window.settings.value('paths/dlgdir', '.'))
            filter = "Images ( *" + " *".join(IMAGE_FILE_EXTENSIONS) + ")"
            dlg = QFileDialog(window.__repr__.__self__, "select", lastDir,
                              filter)  # QFileDialog does not accept weakProxy arg
            if dlg.exec():
                filenames = dlg.selectedFiles()
                newDir = dlg.directory().absolutePath()
                window.settings.setValue('paths/dlgdir', newDir)
                filename = filenames[0]
                im = QImageFromFile(filename)
                im = im.scaled(self.layer.size(), Qt.KeepAspectRatio)
                self.sourceImage = im.copy(QRect(QPoint(0, 0), self.layer.size()))
                self.widgetImg.setWindowTitle(f"Source : {basename(filename)}")
                self.updateSource()
            self.widgetImg.show()

        pushButton1.clicked.connect(f)

        # reset button clicked slot
        def g():
            layer = self.layer
            # mask all pixels
            layer.resetMask(maskAll=True)
            layer.setMaskEnabled(color=False)
            # reset cloning layer
            layer.xAltOffset, layer.yAltOffset = 0.0, 0.0
            layer.AltZoom_coeff = 1.0
            layer.cloningState = ''
            layer.applyCloning(seamless=False, showTranslated=True)
            layer.parentImage.onImageChanged()

        pushButton2.clicked.connect(g)

        # layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(layout)
        layout.addWidget(self.listWidget2)
        layout.addWidget(self.listWidget1)
        layout.addWidget(self.listWidget3)
        hl = QHBoxLayout()
        hl.addWidget(pushButton1)
        # hl.addWidget(self.listWidget3)
        hl.addWidget(pushButton2)
        layout.addLayout(hl)

        self.setDefaults()

        self.setWhatsThis(
            """
            <b>Cloning/healing brush</b><br>
            Seamless replacement of a region of the image by another region of the same image 
            or by another image (e.g. to erase an object):<br>
               &nbsp; 1) <b> Make sure that the cloning layer is the topmost visible layer and
                         that Normal Blend mode is selected.</b><br>
               &nbsp; 2) With the <i>Pointer Tool</i> selected, <b>Ctrl+Alt+Click</b>
                          on the layer or the source window to mark the source starting point;<br> 
               &nbsp; 3) Select the <i>Unmask/FG Tool</i> and paint the destination region to copy and clone pixels. 
                         Use <i>the Mask/BG Tool</i> to adjust the mask if needed. <br>
            Use <b>Ctrl+Alt+Mouse Wheel</b> to zoom in or out the cloned region.<br>
            Eventually use <b>Mask Erode</b> from the layer context menu to smooth the contour of the mask.<br>
            """
        )  # end of setWhatsthis

    def __del__(self):
        print('patchForm deleted')

    def setDefaults(self):
        self.enableOptions()
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.layer.cloningMethod = self.listWidget1.checkedItems[0].data(Qt.UserRole)
        self.layer.setMaskEnabled()
        self.layer.resetMask(maskAll=True)  # , alpha=128)
        # self.widgetImg.setPixmap(QPixmap.fromImage(self.layer.inputImg().scaled(200, 200, aspectMode=Qt.KeepAspectRatio)))
        # init positioning window
        img = self.layer.inputImg(drawTranslated=False)
        if img.rPixmap is None:
            img.rPixmap = QPixmap.fromImage(img)
        self.sourcePixmap = img.rPixmap
        self.sourcePixmapThumb = self.sourcePixmap.scaled(200, 200, aspectMode=Qt.KeepAspectRatio)
        self.widgetImg.setPixmap(self.sourcePixmapThumb)
        self.widgetImg.setFixedSize(self.sourcePixmapThumb.size())
        # show positioning window
        self.widgetImg.hide()
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.dataChanged.connect(self.updateLayer)

    def updateSource(self):
        """
        sets the pointing window, using self.sourceImage
        """
        if self.sourceImage is None:
            return
        # scale img while keeping its aspect ratio
        # into a QPixmap having the same size than self.layer
        sourcePixmap = QPixmap.fromImage(self.sourceImage).scaled(self.layer.size(), Qt.KeepAspectRatio)
        self.sourceSize = sourcePixmap.size()
        self.sourcePixmap = QPixmap(self.layer.size())
        self.sourcePixmap.fill(Qt.black)
        qp = QPainter(self.sourcePixmap)
        qp.drawPixmap(QPointF(), sourcePixmap)
        qp.end()
        self.sourcePixmapThumb = self.sourcePixmap.scaled(self.pwSize, self.pwSize, aspectMode=Qt.KeepAspectRatio)
        self.widgetImg.setPixmap(self.sourcePixmapThumb)
        self.widgetImg.setFixedSize(self.sourcePixmapThumb.size())
        # add subcontrol if needed
        if self.dockT is None:
            window = self.mainForm
            dockT = self.addSubcontrol(None)
            dockT.setWindowFlags(self.widgetImg.windowFlags())
            dockT.setWindowTitle(self.widgetImg.windowTitle())
            window.addDockWidget(Qt.LeftDockWidgetArea, dockT)
            self.dockT = dockT
            dockT.setWidget(self.widgetImg)
        #  set option "show source image"
        self.options.dictionaries[2]['source'] = True
        self.widgetImg.show()
        self.layer.sourceFromFile = True
        self.listWidget3.checkAll()
        self.listWidget3.setEnabled(True)

    def enableOptions(self):
        if self.options['blue']:
            # make sure disabled options are unchecked
            self.listWidget1.checkOption('Normal Clone')
        for item in [self.listWidget1.item(i) for i in (1, 2)]:  # mixed clone, monochrome transfer
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

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [optionsWidget]:
                d[a] = obj.__getstate__()

        return d

    def __setstate__(self, d):
        # prevent multiple updates
        d1 = d['state']
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        for name in d1:
            obj = getattr(self, name, None)
            if type(obj) in [optionsWidget]:
                obj.__setstate__(d1[name])
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()
