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
from PySide2.QtGui import QImage, QColor, QPixmap, QPainter, QBrush, QPen, QFontMetrics
from PySide2.QtWidgets import QGraphicsPixmapItem, QLabel, QVBoxLayout

from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.dialog import dlgWarn
from bLUeGui.graphicsForm import baseGraphicsForm
from bLUeGui.graphicsSpline import activePoint
from bLUeTop.utils import optionsWidget


class activeMixerPoint(activePoint):

    def __init__(self, x, y, color=Qt.white, fillColor=None, parentItem=None, grForm=None):
        super().__init__(x, y, color=color, fillColor=fillColor, parentItem=parentItem)
        self.grForm = grForm

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        if self.grForm.options['Monochrome']:
            for p in [self.grForm.rPoint, self.grForm.gPoint, self.grForm.bPoint]:
                if p is not self:
                    p.setPos(self.pos())
        self.scene().update()

    def mouseReleaseEvent(self, e):
        self.grForm.dataChanged.emit()

    def paint(self, qpainter, options, widget):
        """
        Overrides QGraphicsPathItem paint.

        :param qpainter:
        :type qpainter: QPainter
        :param options:
        :type options:  QStyleOptionGraphicsItem
        :param widget:
        :type widget: QWidget
        """
        # draw point
        super().paint(qpainter, options, widget)
        # draw connecting lines
        qpainter.save()
        qpainter.setBrush(QBrush(Qt.white))
        qpainter.setPen(QPen(Qt.white, 1, Qt.DotLine, Qt.RoundCap))
        # local coordinates
        qpainter.drawLine(self.source - self.pos(), QPointF())
        qpainter.restore()


class mixerForm(baseGraphicsForm):
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = mixerForm(axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(parent=parent, targetImage=targetImage, layer=layer)
        self.setMinimumSize(axeSize, axeSize + 100)
        # color wheel size
        self.cwSize = axeSize * 0.95
        self.setAttribute(Qt.WA_DeleteOnClose)
        # options
        optionList = ['Monochrome', 'Luminosity']
        listWidget1 = optionsWidget(options=optionList, exclusive=False, changed=self.dataChanged,
                                    flow=optionsWidget.LeftToRight)
        listWidget1.setMaximumHeight(
            listWidget1.sizeHintForRow(0) + 5)  # mandatory although sizePolicy is set to minimum !
        listWidget1.onSelect = lambda x: dlgWarn('Channel Mixer',
                                                 'Option Luminosity will be removed in the future.' \
                                                 'Use Luminosity blending mode instead')
        self.listWidget1 = listWidget1  # mandatory for __getstate__()
        self.options = listWidget1.options
        # barycentric coordinate basis : the 3 base points form an equilateral triangle
        h = self.cwSize - 50
        s = h * 2 / np.sqrt(3)
        self.R, self.G, self.B = QPointF(10, h + 20), QPointF(10 + s, h + 20), QPointF(10 + s / 2, 20)
        # Conversion matrix from cartesian coordinates (x, y, 1) to barycentric coordinates (alpha, beta, gamma)
        self.M = np.array([[self.R.x(), self.G.x(), self.B.x()],
                           [self.R.y(), self.G.y(), self.B.y()],
                           [1, 1, 1]])
        self.invM = np.linalg.inv(self.M)
        self.setBackgroundImage()
        # active points
        self.rPoint = activeMixerPoint(self.R.x(), self.R.y(), color=Qt.red, fillColor=Qt.white, grForm=self)
        self.rPoint.source = self.R
        self.gPoint = activeMixerPoint(self.G.x(), self.G.y(), color=Qt.green, fillColor=Qt.white, grForm=self)
        self.gPoint.source = self.G
        self.bPoint = activeMixerPoint(self.B.x(), self.B.y(), color=Qt.blue, fillColor=Qt.white, grForm=self)
        self.bPoint.source = self.B
        graphicsScene = self.scene()
        for point in [self.rPoint, self.gPoint, self.bPoint]:
            graphicsScene.addItem(point)
        gl = QVBoxLayout()
        gl.setAlignment(Qt.AlignTop)
        container = self.addCommandLayout(gl)
        self.values = QLabel()
        vh = QFontMetrics(self.values.font()).height()
        self.values.setMaximumSize(150, vh * 4)  # 4 lines
        gl.addWidget(self.values)
        gl.addWidget(listWidget1)
        # don't commute the 3 next lines !
        self.setDefaults()
        self.adjustSize()
        self.setViewportMargins(0, 0, 0, container.height())
        self.setWhatsThis(
            """<b>Channel Mixer</b><br>
            The triangle vertices and the three control points correspond to the R, G, B channels.<br>
            To <b>mix the channels</b>, drag the 3 control points inside the triangle.
            The closer a control point is to a vertex, the greater the corresponding channel contribution. <br>
            To obtain <b>monochrome images</b> only, check the option <i>Monochrome.</i><br>
            To modify the <b>luminosity channel</b> only (volume mode), check the option <i>Luminosity.</i><br>
            """
        )  # end of setWhatsThis

    def updateLayer(self):
        """
        dataChanged slot
        """
        if self.options['Monochrome']:
            for p in [self.gPoint, self.bPoint]:
                p.setPos(self.rPoint.pos())
            self.scene().update()
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
        qp.drawLine(b - QPointF(10, 0), b + QPointF(10, 0))
        qp.drawLine(b - QPointF(0, 10), b + QPointF(0, 10))
        qp.end()
        self.scene().addItem(QGraphicsPixmapItem(QPixmap.fromImage(img)))

    def getChannelValues(self):
        return "\n".join(("         R      G      B",
                          " R <- %.2f  %.2f  %.2f" % tuple(self.mixerMatrix[0]),
                          " G <- %.2f  %.2f  %.2f" % tuple(self.mixerMatrix[1]),
                          " B <- %.2f  %.2f  %.2f" % tuple(self.mixerMatrix[2])))

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [optionsWidget]:
                d[a] = obj.__getstate__()
        d['rPoint'] = self.rPoint.pos()
        d['gPoint'] = self.gPoint.pos()
        d['bPoint'] = self.bPoint.pos()
        return d

    def __setstate__(self, d):
        # prevent multiple updates
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        for name in d['state']:
            obj = getattr(self, name, None)
            if type(obj) in [optionsWidget]:
                obj.__setstate__(d['state'][name])
            elif type(obj) in [activeMixerPoint]:
                p = d['state'][name]
                obj.setPos(p)
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()
