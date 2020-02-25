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
import numpy as np
from PySide2.QtCore import Qt, QSize, QPointF
from PySide2.QtGui import QImage, QColor, QPixmap, QPainter, QBrush, QPen
from PySide2.QtWidgets import QGraphicsPixmapItem, QGridLayout, QLabel

from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.graphicsForm import baseGraphicsForm
from bLUeGui.graphicsSpline import activePoint
from bLUeTop.utils import optionsWidget


class activeMixerPoint(activePoint):

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        grForm = self.scene().layer.view.widget()
        if grForm.options['Monochrome']:
            for p in [grForm.rPoint, grForm.gPoint, grForm.bPoint]:
                if p is not self:
                    p.setPos(self.pos())
        self.scene().update()
        grForm.dataChanged.emit()

    def paint(self, qpainter, options, widget):
        """
        Overrides QGraphicsPathItem paint
        @param qpainter:
        @type qpainter: QPainter
        @param options:
        @type options:  QStyleOptionGraphicsItem
        @param widget:
        @type widget: QWidget
        """
        # draw point
        super().paint(qpainter, options, widget)
        # draw connecting lines
        qpainter.save()
        qpainter.setBrush(QBrush(QColor(255, 255, 255)))
        qpainter.setPen(QPen(Qt.white, 1, Qt.DotLine, Qt.RoundCap))
        # local coordinates
        qpainter.drawLine(self.source - self.pos(), QPointF())
        qpainter.restore()


class mixerForm(baseGraphicsForm):

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = mixerForm(axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(parent=parent, targetImage=targetImage, layer=layer)
        self.setMinimumSize(axeSize, axeSize + 100)
        # color wheel size
        self.cwSize = axeSize * 0.95
        self.setAttribute(Qt.WA_DeleteOnClose)
        # options
        optionList = ['Monochrome']
        listWidget1 = optionsWidget(options=optionList, exclusive=False, changed=self.dataChanged)
        # listWidget1.setMinimumWidth(listWidget1.sizeHintForColumn(0) + 5)
        # listWidget1.setMaximumHeight(listWidget1.sizeHintForRow(0) * len(optionList))
        self.options = listWidget1.options
        # barycentric coordinate basis : the 3 base points form an equilateral triangle
        h = self.cwSize - 50
        s = h * 2 / np.sqrt(3)
        self.R, self.G, self.B = QPointF(10, h + 20), QPointF(10 + s, h + 20), QPointF(10 + s / 2, 20)
        # Conversion matrix from cartesian coordinates (x, y, 1) to barycentric coordinates (alpha, beta, gamma)
        self.M = np.array([[self.R.x(), self.G.x(), self.B.x()],
                           [self.R.y(), self.G.y(), self.B.y()],
                           [1,            1,           1      ]])
        self.invM = np.linalg.inv(self.M)
        self.setBackgroundImage()
        # active points
        self.rPoint = activeMixerPoint(self.R.x(), self.R.y(), color=Qt.red, fillColor=Qt.white)
        self.rPoint.source = self.R
        self.gPoint = activeMixerPoint(self.G.x(), self.G.y(), color=Qt.green, fillColor=Qt.white)
        self.gPoint.source = self.G
        self.bPoint = activeMixerPoint(self.B.x(), self.B.y(), color=Qt.blue, fillColor=Qt.white)
        self.bPoint.source = self.B
        graphicsScene = self.scene()
        for point in [self.rPoint, self.gPoint, self.bPoint]:
            graphicsScene.addItem(point)
        gl = QGridLayout()
        container = self.addCommandLayout(gl)
        self.values = QLabel()
        self.values.setMaximumSize(120, 60)
        gl.addWidget(self.values, 0, 0, 4, 8)
        gl.addWidget(listWidget1, 4, 0, 4, 2)
        container.adjustSize()
        self.setViewportMargins(0, 0, 0, container.height() + 15)
        self.setDefaults()
        self.setWhatsThis(
                        """<b>Channel Mixer</b><br>
                        The triangle vertices and the control points correspond to the R, G, B channels .<br>
                        To <b>mix the channels</b>, drag the 3 control points inside the triangle.
                        The closer a control point is to a vertex, the greater the corresponding channel contribution. <br>
                        Checking the option <i>Monochrome</i> gives monochrome images only.
                        """
                        )  # end of setWhatsThis

    def updateLayer(self):
        baryCoordR = self.invM @ [self.rPoint.x(), self.rPoint.y(), 1]
        baryCoordG = self.invM @ [self.gPoint.x(), self.gPoint.y(), 1]
        baryCoordB = self.invM @ [self.bPoint.x(), self.bPoint.y(), 1]
        self.mixerMatrix = np.vstack((baryCoordR, baryCoordG, baryCoordB))
        with np.printoptions(precision=2, suppress=True):
            self.values.setText(self.getChannelValues())
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def setDefaults(self):
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.mixerMatrix = np.identity(3, dtype=np.float)
        with np.printoptions(precision=2, suppress=True):
            self.values.setText(self.getChannelValues())
        self.dataChanged.connect(self.updateLayer)

    def setBackgroundImage(self):
        img = QImage(QSize(256, 256), QImage.Format_ARGB32)
        img.fill(QColor(100, 100, 100))
        a = np.arange(256)
        buf = np.meshgrid(a, a)
        buf1 = QImageBuffer(img)[:, :, :3][:, :, ::-1]
        buf1[:, :, 0], buf1[:, :, 1] = buf
        buf1[:, :, 2] = 1
        buf2 = np.tensordot(buf1, self.invM, axes=(-1, -1)) * 255
        np.clip(buf2, 0, 255, out=buf2)
        buf1[...] = buf2
        img = img.scaled(self.cwSize, self.cwSize)
        qp = QPainter(img)
        # draw edges
        qp.drawLine(self.R, self.G)
        qp.drawLine(self.G, self.B)
        qp.drawLine(self.B, self.R)
        # draw center
        b = (self.B + self.R + self.G) / 3.0
        qp.drawLine(b-QPointF(10, 0), b + QPointF(10, 0))
        qp.drawLine(b - QPointF(0, 10), b + QPointF(0, 10))
        qp.end()
        self.scene().addItem(QGraphicsPixmapItem(QPixmap.fromImage(img)))

    def getChannelValues(self):
        return "\n".join(("      R      G      B",
                          " R : %.2f  %.2f  %.2f" % tuple(self.mixerMatrix[0]),
                          " G : %.2f  %.2f  %.2f" % tuple(self.mixerMatrix[1]),
                          " B : %.2f  %.2f  %.2f" % tuple(self.mixerMatrix[2])))
