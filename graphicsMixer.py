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
from PySide2.QtWidgets import QGraphicsPixmapItem

from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.graphicsForm import baseGraphicsForm
from bLUeGui.graphicsSpline import activePoint


class activeMixerPoint(activePoint):

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self.scene().update()
        grForm = self.scene().layer.view.widget()
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
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.options = None
        # barycentric coordinate basis : the 3 base points form an equilateral triangle
        self.R, self.G, self.B = QPointF(20, 235), QPointF(235, 235), QPointF(128, 13)
        # Conversion matrix from cartesian coordinates (x, y, 1) to barycentric coordinates (alpha, beta, gamma)
        self.M = np.array([[self.R.x(), self.G.x(), self.B.x()],
                           [self.R.y(), self.G.y(), self.B.y()],
                           [1,            1,           1      ]])
        self.invM = np.linalg.inv(self.M)
        self.setBackgroundImage()
        # active points
        self.rPoint = activeMixerPoint(20, 235)
        self.rPoint.source = self.R
        self.gPoint = activeMixerPoint(235, 235)
        self.gPoint.source = self.G
        self.bPoint = activeMixerPoint(128, 13)
        self.bPoint.source = self.B
        graphicsScene = self.scene()
        for point in [self.rPoint, self.gPoint, self.bPoint]:
            graphicsScene.addItem(point)
        self.setDefaults()
        self.setWhatsThis(
                        """
                        <b>Channel Mixer</b><br>
                        To <b>mix the R, G, B channels</b>, drag the 3 control points inside the triangle.<br>
                        """
                        )  # end of setWhatsThis

    def updateLayer(self):
        baryCoordR = self.invM @ [self.rPoint.x(), self.rPoint.y(), 1]
        baryCoordG = self.invM @ [self.gPoint.x(), self.gPoint.y(), 1]
        baryCoordB = self.invM @ [self.bPoint.x(), self.bPoint.y(), 1]
        self.mixerMatrix = np.vstack((baryCoordR, baryCoordG, baryCoordB))
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def setDefaults(self):
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.mixerMatrix = np.identity(3, dtype=np.float)
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
        qp = QPainter(img)
        qp.drawLine(self.R, self.G)
        qp.drawLine(self.G, self.B)
        qp.drawLine(self.B, self.R)
        b = (self.B + self.R + self.G) / 3.0
        qp.drawLine(b-QPointF(10, 0), b + QPointF(10, 0))
        qp.drawLine(b - QPointF(0, 10), b + QPointF(0, 10))
        qp.end()
        self.scene().addItem(QGraphicsPixmapItem(QPixmap.fromImage(img)))
