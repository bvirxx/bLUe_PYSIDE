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
import os

import numpy as np
import cv2

from PySide2.QtCore import QRect, QPointF, QPoint
from PySide2.QtGui import QPixmap, QColor, QPainter, QRadialGradient, QBrush, QPainterPath, QImage

from bLUeGui.bLUeImage import QImageBuffer
from bLUeTop.settings import BRUSHES_PATH


class brushFamily:
    """
    A brush family is a set of brushes sharing a common shape.
    The shape is defined by a QPainterPath instance and scaled
    by individual brushes.
    A preset can be defined for each brush family. A preset is a mask
    describing the alpha channel of the brush. It is built from the luminosity
    channel of a png or jpg image.
    A cursor pixmap corresponding to the shape is associated with the brush family.
    To build individual brushes from the family use the method getBrush.
    """
    def __init__(self, name, baseSize, contourPath, presetFilename=None):
        """

        @param name:
        @type name: str
        @param baseSize:
        @type baseSize: int
        @param contourPath: base shape of the brush family
        @type contourPath: QPainterPath
        @param presetFilename: preset file
        @type presetFilename: str
        """
        self.name = name
        self.baseSize = baseSize
        # init the brush pixmap
        self.basePixmap = QPixmap(self.baseSize, self.baseSize)
        # to get an alpha channel, we must fill the pixmap a first time with an opacity < 255
        self.basePixmap.fill(QColor(0, 0, 0, 0))
        if self.name == 'eraser':
            self.basePixmap.fill(QColor(0, 0, 0, 255))
        self.contourPath = contourPath
        # init brush cursor
        self.baseCursor = QPixmap(self.baseSize, self.baseSize)
        self.baseCursor.fill(QColor(0, 0, 0, 0))
        qp = QPainter(self.baseCursor)
        pen = qp.pen()
        pen.setWidth(self.baseSize / 20)
        qp.setPen(pen)  # needed!!
        qp.drawPath(contourPath)
        qp.end()
        self.__pxmp = None
        self.bOpacity = 1.0
        self.bFlow = 1.0
        self.bHardness = 1.0
        self.preset = None
        if presetFilename is not None:
            img = QImage(presetFilename)
            img = img.convertToFormat(QImage.Format_ARGB32)
            buf = QImageBuffer(img)
            b = np.sum(buf[...,:3], axis=-1, dtype=np.float)
            b /= 3
            buf[..., 3] = b
            self.preset = QPixmap.fromImage(img)

    @property
    def pxmp(self):
        return self.__pxmp

    @pxmp.setter
    def pxmp(self, pixmap):
        self.__pxmp = pixmap

    def getBrush(self, size, opacity, color, hardness, flow):
        """
        initializes and returns a brush as a dictionary
        @param size: brush size
        @type size: int
        @param opacity: brush opacity, range 0..1
        @type opacity: float
        @param color:
        @type color: QColor
        @param hardness: brush hardness, range 0..1
        @type hardness: float
        @param flow: brush flow, range 0..1
        @type flow: float
        @return:
        @rtype: dict
        """
        # set brush color
        s = float(self.baseSize) / 2
        if self.name == 'eraser':
            color = QColor(0, 0, 0, 0)
        else:
            color = QColor.fromRgb(color.red(), color.green(), color.blue(), int(64 * flow))
        gradient = QRadialGradient(QPointF(s, s), s)
        gradient.setColorAt(0, color)
        gradient.setColorAt(hardness, color)
        if self.name == 'eraser':
            gradient.setColorAt(1, QColor.fromRgb(0, 0, 0, 255))
        else:
            gradient.setColorAt(1, QColor.fromRgb(0, 0, 0, 0))
        pxmp = self.basePixmap.copy()
        qp = QPainter(pxmp)
        qp.setCompositionMode(qp.CompositionMode_Source)
        qp.fillPath(self.contourPath, QBrush(gradient))
        if self.preset is not None:
            qp.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            qp.drawPixmap(QRect(QPoint(0,0), pxmp.size()), self.preset)
        qp.end()
        self.pxmp = pxmp.scaled(size, size)
        return {'family': self, 'name': self.name, 'pixmap': self.pxmp, 'size': size, 'color': color, 'opacity': opacity,
                'hardness': hardness, 'flow': flow, 'cursor': self.baseCursor}


def initBrushes():
    """
    returns a list of brush families.
    Eraser is the last item of the list.
    @return:
    @rtype: list of brushFamily instances
    """
    brushes = []
    ######################
    # standard round brush
    ######################
    baseSize = 400 # 25
    qpp = QPainterPath()
    qpp.addEllipse(QRect(0, 0, baseSize, baseSize))
    roundBrushFamily = brushFamily('Round', baseSize, qpp, presetFilename=None)
    brushes.append(roundBrushFamily)
    ######################
    # preset brushes
    ######################
    try:
        rank = 1
        for entry in os.scandir(BRUSHES_PATH):
            if entry.is_file():
                if entry.name[-4:].lower() in ['.png', '.jpg']:
                    try:
                        qpp = QPainterPath()
                        qpp.addEllipse(QRect(0, 0, baseSize, baseSize))
                        presetBrushFamily = brushFamily('Preset ' + str(rank), baseSize, qpp, presetFilename=os.getcwd() + '\\' + BRUSHES_PATH + '\\' + entry.name)
                        brushes.append(presetBrushFamily)
                        rank += 1
                    except IOError:
                        pass
    except IOError:
        pass
    ##########
    # eraser
    ##########
    qpp = QPainterPath()
    qpp.addEllipse(QRect(0, 0, baseSize, baseSize))
    eraserFamily = brushFamily('Eraser', baseSize, qpp)
    # eraser must be added last
    brushes.append(eraserFamily)
    return brushes


def bLUeFloodFill(layer, x, y, color):
    """
    Flood fills a region of a drawing layer
    x, y are the source coordinates
    @param layer:
    @type layer: QLayerImage
    @param x:
    @type x: float
    @param y:
    @type y: float
    @param color: filling color
    @type color: QColor
    """
    img = layer.sourceImg
    w, h = img.width(), img.height()
    buf0 = QImageBuffer(img)
    # preparing opencv data
    buf = np.ascontiguousarray(buf0[..., :3][..., ::-1])
    mask = np.zeros((h+2, w+2), dtype=np.uint8)
    # flood filling
    cv2.floodFill(buf, mask, (x, y), (color.red(), color.green(), color.blue()))
    buf0[..., :3] = buf[..., ::-1]
    # set the alpha channel of the filled region
    buf0[mask[1:-1, 1:-1] == 1, 3] = color.alpha()


