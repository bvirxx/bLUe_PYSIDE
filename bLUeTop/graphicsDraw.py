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
from PySide2.QtCore import Qt, QPointF
from PySide2.QtGui import QPixmap, QColor, QPainterPath, QTransform
from PySide2.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QLabel

from bLUeGui.graphicsForm import baseForm
from bLUeTop.drawing import brushFamily


class drawForm (baseForm):
    """
    Drawing form
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=200, layer=None, parent=None):
        wdgt = drawForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.options = None
        pushButton1 = QPushButton(' Undo ')
        pushButton1.adjustSize()
        pushButton2 = QPushButton(' Redo ')
        pushButton2.adjustSize()

        pushButton1.clicked.connect(self.undo)
        pushButton2.clicked.connect(self.redo)

        spacingSlider = QSlider(Qt.Horizontal)
        spacingSlider.setObjectName('spacingSlider')
        spacingSlider.setRange(1,60)
        spacingSlider.setTickPosition(QSlider.TicksBelow)
        spacingSlider.setSliderPosition(10)
        spacingSlider.sliderReleased.connect(self.parent().label.brushUpdate)
        self.spacingSlider = spacingSlider

        jitterSlider = QSlider(Qt.Horizontal)
        jitterSlider.setObjectName('jitterSlider')
        jitterSlider.setRange(0, 100)
        jitterSlider.setTickPosition(QSlider.TicksBelow)
        jitterSlider.setSliderPosition(0)
        jitterSlider.sliderReleased.connect(self.parent().label.brushUpdate)
        self.jitterSlider = jitterSlider

        orientationSlider = QSlider(Qt.Horizontal)
        orientationSlider.setObjectName('orientationSlider')
        orientationSlider.setRange(0, 360)
        orientationSlider.setTickPosition(QSlider.TicksBelow)
        orientationSlider.setSliderPosition(180)
        orientationSlider.sliderReleased.connect(self.parent().label.brushUpdate)
        self.orientationSlider = orientationSlider

        # sample
        self.sample = QLabel()
        #self.sample.setMinimumSize(200, 100)
        pxmp = QPixmap(250,100)
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
                          Choose a brush family, flow, hardness and opacity.
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
        pxmp.fill(QColor(0,0,0,0))
        brushFamily.brushStrokePoly(pxmp, self.samplePoly, self.layer.brushDict)
        self.sample.repaint()

    def undo(self):
        try:
            self.layer.sourceImg = self.layer.history.undo(saveitem=self.layer.sourceImg.copy()).copy()  # copy is mandatory
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
