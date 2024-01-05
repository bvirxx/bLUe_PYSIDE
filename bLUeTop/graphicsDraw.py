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
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPixmap, QColor, QPainterPath, QTransform
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QLabel

from bLUeGui.graphicsForm import baseForm
from bLUeGui.dialog import dlgWarn
from bLUeTop.drawing import brushFamily
from bLUeTop.utils import QbLUeSlider


class drawForm(baseForm):
    """
    Drawing form
    """
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=200, layer=None, parent=None):
        wdgt = drawForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.options = None
        pushButton1 = QPushButton(' Undo ')
        pushButton1.adjustSize()
        pushButton2 = QPushButton(' Redo ')
        pushButton2.adjustSize()

        pushButton1.clicked.connect(self.undo)
        pushButton2.clicked.connect(self.redo)

        spacingSlider = QbLUeSlider(Qt.Horizontal)
        spacingSlider.setObjectName('spacingSlider')
        spacingSlider.setRange(1, 60)
        spacingSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        spacingSlider.setSliderPosition(10)
        spacingSlider.sliderReleased.connect(self.parent().label.brushUpdate)
        self.spacingSlider = spacingSlider

        jitterSlider = QbLUeSlider(Qt.Horizontal)
        jitterSlider.setObjectName('jitterSlider')
        jitterSlider.setRange(0, 100)
        jitterSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        jitterSlider.setSliderPosition(0)
        jitterSlider.sliderReleased.connect(self.parent().label.brushUpdate)
        self.jitterSlider = jitterSlider

        orientationSlider = QbLUeSlider(Qt.Horizontal)
        orientationSlider.setObjectName('orientationSlider')
        orientationSlider.setRange(0, 360)
        orientationSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        orientationSlider.setSliderPosition(180)
        orientationSlider.sliderReleased.connect(self.parent().label.brushUpdate)
        self.orientationSlider = orientationSlider

        # self.brushFamilyList = self.mainForm.brushes
        self.colorChooser = self.parent().colorChooser

        # sample
        self.sample = QLabel()
        # self.sample.setMinimumSize(200, 100)
        pxmp = QPixmap(250, 100)
        pxmp.fill(QColor(255, 255, 255, 255))
        self.sample.setPixmap(pxmp)
        qpp = QPainterPath()
        qpp.moveTo(QPointF(20, 50))
        qpp.cubicTo(QPointF(80, 25), QPointF(145, 70), QPointF(230, 60))  # c1, c2, endPoint
        self.samplePoly = qpp.toFillPolygon(QTransform())
        # we want an unclosed polygon
        self.samplePoly.removeLast()

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        hl = QHBoxLayout()
        hl.setAlignment(Qt.AlignHCenter)
        hl.addWidget(pushButton1)
        hl.addWidget(pushButton2)
        l.addLayout(hl)
        l.addWidget(QLabel('Brush Dynamics'))
        hl1 = QHBoxLayout()
        hl1.addWidget(QLabel('Spacing'))
        hl1.addWidget(spacingSlider)
        l.addLayout(hl1)
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel('Jitter'))
        hl2.addWidget(jitterSlider)
        l.addLayout(hl2)
        hl3 = QHBoxLayout()
        hl3.addWidget(QLabel('Orientation'))
        hl3.addWidget(self.orientationSlider)
        l.addLayout(hl3)
        l.addWidget(self.sample)
        self.setLayout(l)
        self.adjustSize()

        self.setDefaults()
        self.setWhatsThis(
            """
            <b>Drawing :</b><br>
              Select the <b>Brush</b> tool button.</b>. Choose a brush family (default <i>Round</i>), 
              size, flow, hardness and opacity.<br>
              To <b>change the brush color</b> use one of the following :
               <br> - Ctrl-C key
               <br> - menu <i>View->Color Chooser</i> 
               <br> - <i>Ctrl+Click</i> on an image pixel.<br>
              To <b>load preset brushes</b> use menu <i>File->Load Preset.</i><br>
              To <b>move</b> the layer select the <i>Drag Tool</i> and do <i>Ctrl+Drag</i>.<br><br>
              For faster operations use the <i>Preview</i> mode (the drawing will still be done using the 
              full resolution image).<br><br>
              <b>Warning :</b> All upper layers (drawing layers excepted) must be made non visible.
              Otherwise drawing operations will not be rendered until next layer stack update.<br> 
            """
        )  # end of setWhatsThis

    def setDefaults(self):
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.dataChanged.connect(self.updateLayer)
        self.updateSample()

    def updateLayer(self):
        """
        dataChanged slot
        """
        l = self.layer
        # l.tool.setBaseTransform()
        l.applyToStack()
        l.parentImage.onImageChanged()

    def updateSample(self):
        pxmp = self.sample.pixmap()
        pxmp.fill(QColor(0, 0, 0, 0))
        brushFamily.brushStrokePoly(pxmp, self.samplePoly, self.layer.brushDict)
        self.sample.setPixmap(pxmp)

    def undo(self):
        try:
            self.layer.sourceImg = self.layer.history.undo(
                saveitem=self.layer.sourceImg.copy()).copy()  # copy is mandatory
            self.updateLayer()
        except ValueError:
            pass

    def redo(self):
        try:
            self.layer.sourceImg = self.layer.history.redo().copy()  # copy is mandatory
            self.updateLayer()
        except ValueError:
            pass

    def reset(self):
        self.layer.tool.resetTrans()

    def colorPickedSlot(self, x, y, modifiers):
        """
        (x,y) coordinates are relative to the full size image.
        Ctrl+click on a drawing layer picks new brush color.

        :param x:
        :type x:
        :param y:
        :type y:
        :param modifiers:
        :type modifiers:
        """
        if modifiers == Qt.ControlModifier:
            r, g, b = self.layer.parentImage.getPrPixel(x, y)  # getActivePixel(x, y, fromInputImg=False, qcolor=True)
            clr = QColor(r, g, b)
            self.mainForm.label.brushUpdate(color=clr)
            self.mainForm.colorChooser.setCurrentColor(clr)

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [QbLUeSlider]:
                d[a] = obj.__getstate__()
        brushDict = self.layer.brushDict
        d['brush'] = {name: brushDict[name] for name in
                      ['name', 'size', 'color', 'opacity', 'hardness', 'flow', 'spacing', 'jitter', 'orientation']}
        return d

    def __setstate__(self, d):
        # prevent multiple updates
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        for name in d['state']:
            obj = getattr(self, name, None)
            if type(obj) in [QbLUeSlider]:
                obj.__setstate__(d['state'][name])

        bdict = d['state']['brush']
        brushFamilyNames = [family.name.lower() for family in self.mainForm.brushes]
        try:
            ind = brushFamilyNames.index(bdict['name'].lower())
            family = self.mainForm.brushes[ind]
        except ValueError:
            dlgWarn('Cannot restore brush', 'Reload presets')
            family = None
        bSize = bdict['size']
        bOpacity = bdict['opacity']
        bColor = bdict['color']
        bHardness = bdict['hardness']
        bFlow = bdict['flow']
        bSpacing = bdict['spacing']
        bJitter = bdict['jitter']
        bOrientation = bdict['orientation']
        # pattern = bdict['pattern']
        if family is not None:
            self.layer.brushDict = family.getBrush(bSize, bOpacity, bColor, bHardness, bFlow, spacing=bSpacing,
                                                   jitter=bJitter, orientation=bOrientation)  # pattern=pattern
            self.mainForm.label.State['brush'] = self.layer.brushDict

        if self.layer.brushDict is None:  # no brush set yet
            self.mainForm.label.brushUpdate()
            self.layer.brushDict = self.mainForm.label.State['brush']

        self.mainForm.label.img.onActiveLayerChanged()  # alias to restorebrush()

        self.updateSample()
        self.colorChooser.setCurrentColor(bColor)
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()
