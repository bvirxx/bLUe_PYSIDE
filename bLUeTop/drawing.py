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
from math import sqrt
from random import choice

import numpy as np
import cv2

from PySide2.QtCore import QRect, QPointF
from PySide2.QtGui import QPixmap, QColor, QPainter, QRadialGradient, QBrush, QPainterPath, QImage, QTransform

from bLUeGui.bLUeImage import QImageBuffer, ndarrayToQImage
from bLUeTop.presetReader import aParser
from bLUeTop.settings import BRUSHES_PATH

class pattern:
    """
    Brus pattern
    """
    def __init__(self, name, im=None, pxmp=None):
        self.name = name
        self.im = im
        self.pxmp = pxmp

class brushFamily:
    """
    A brush family is a set of brushes sharing a common shape.
    The shape is defined by a QPainterPath instance and scaled
    by individual brushes.
    A preset can be defined for each brush family. A preset is a mask
    describing the alpha channel of the brush. It is built from the luminosity
    channel of a png or jpg image.
    A cursor pixmap corresponding to the shape is associated with the brush family.
    Individual brushes from the family are dictionaries. They are instantiated by
    the method getBrush().
    """
    # range of brush stroke jittering
    jitterRange = range(-5, 5)

    @staticmethod
    def brushStrokeSeg(qp, x0, y0, x, y, brush):
        """
        Base method for brush painting.
        The method paints the straight line ((x0,y0), (x, y)) with the brush defined by brush,
        using an active QPainter instance qp,
        It returns the last painted position, or the initial position if
        not any painting occurs, due to spacing constraints.
        @param x0: image x-coord
        @type x0: float
        @param y0: image y-coord
        @type y0: float
        @param x: image x-coord
        @type x: float
        @param y: image y-coord
        @type y: float
        @param brush: painting brush or eraser
        @type brush: dict
        @return: last painted position
        @rtype: 2-uple of float
        """
        tmp_x = x
        tmp_y = y
        # vector of the move
        a_x, a_y = tmp_x - x0, tmp_y - y0
        # move length : use 1-norm for performance
        d = sqrt(a_x * a_x + a_y * a_y)
        spacing, jitter, radius, pxmp = brush['spacing'], brush['jitter'], brush['size'] / 2, brush['pixmap']
        step = 1 if d == 0 else radius * 0.3 * spacing / d  # 0.25
        sinBeta, cosBeta = 0.0, 1.0
        if jitter != 0.0:
            step *= (1.0 + choice(brushFamily.jitterRange) * jitter / 100.0)
            sinBeta = choice(brushFamily.jitterRange) * jitter / 100
            cosbeta = sqrt(1 - sinBeta * sinBeta)
        p_x, p_y = x0, y0
        if d != 0.0:
            cosTheta, sinTheta = a_x / d, a_y / d
            if jitter != 0.0:
                cosTheta = cosTheta * cosBeta + sinTheta * sinBeta
                sinTheta = sinTheta * cosBeta - cosTheta * sinBeta
            transform = QTransform(cosTheta, sinTheta, -sinTheta, cosTheta, 0, 0)  # Caution: angles > 0 correspond to counterclockwise rotations of pxmp
            pxmp = pxmp.transformed(transform)
        count = 0
        maxCount = int(1.0 / step)
        while count < maxCount:
            count += 1
            if pxmp is None:
                qp.drawEllipse(QPointF(p_x, p_y), radius, radius)
            else:
                qp.drawPixmap(QPointF(p_x - radius, p_y - radius), pxmp)  # TODO radius added 19/03/20 validate
            p_x, p_y = p_x + a_x * step, p_y + a_y * step
        # return last brush position
        return p_x, p_y

    @staticmethod
    def brushStrokePoly(pixmap, poly, brush):
        """
        Draws the brush stroke defined by a QPolygon
        @param layer:
        @type layer:
        @param x:
        @type x:
        @param y:
        @type y:
        @param r:
        @type r:
        """
        # draw the stroke
        if brush['name'] == 'eraser':
            return
        # drawing into stroke intermediate layer
        pxmp_temp = pixmap.copy()
        qp = QPainter()
        qp.begin(pxmp_temp)
        qp.setCompositionMode(qp.CompositionMode_SourceOver)
        # draw lines
        x_last, y_last = poly.first().x(), poly.first().y()
        for i in range(poly.length() - 1):
            x0, y0 = poly.at(i).x(), poly.at(i).y()
            x, y = poly.at(i + 1).x(), poly.at(i + 1).y()
            x_last, y_last = brushFamily.brushStrokeSeg(qp,
                                                        x_last,
                                                        y_last,
                                                        x, y,
                                                        brush)
        qp.end()
        # draw texture aligned with image
        strokeTex = pxmp_temp
        p = brush['pattern']
        if p is not None:
            if p.pxmp is not None:
                strokeTex = pxmp_temp.copy()
                qp1 = QPainter(strokeTex)
                qp1.setCompositionMode(qp.CompositionMode_DestinationIn)
                qp1.setBrush(QBrush(p.pxmp))
                qp1.fillRect(QRect(0, 0, strokeTex.width(), strokeTex.height()), QBrush(p.pxmp))
                qp1.end()
        # restore source image and paint
        # the whole stroke with current brush opacity
        qp.begin(pixmap)
        # qp.setCompositionMode(qp.CompositionMode_Source)
        # qp.drawImage(QPointF(), layer.strokeDest)
        qp.setOpacity(brush['opacity'])
        qp.setCompositionMode(qp.CompositionMode_SourceOver)
        qp.drawPixmap(QPointF(), strokeTex )  # pxmp_temp)
        qp.end()

    def __init__(self, name, baseSize, contourPath, presetFilename=None, image=None):
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
        elif image is not None:
            img = image
        else:
            return
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

    def getBrush(self, size, opacity, color, hardness, flow, spacing=1.0, jitter=0.0, orientation=0, pattern=None):
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
        s = float(self.baseSize) / 2
        # set brush color
        if self.name == 'eraser':
            color = QColor(0, 0, 0, 0)
        else:
            op_max = 255  # 64
            color = QColor(color.red(), color.green(), color.blue(), int(op_max * flow))
        gradient = QRadialGradient(QPointF(s, s), s)
        gradient.setColorAt(0, color)
        gradient.setColorAt(hardness, color)
        if hardness < 1.0:
            # fade action to 0, starting from hardness to 1
            if self.name == 'eraser':
                gradient.setColorAt(1, QColor(0, 0, 0, 255))
            else:
                gradient.setColorAt(1, QColor(0, 0, 0, 0))
        pxmp = self.basePixmap.copy()
        qp = QPainter(pxmp)
        # fill brush contour with gradient (pxmp color is (0,0,0,0)
        # outside of contourPath)
        qp.setCompositionMode(qp.CompositionMode_Source)
        qp.fillPath(self.contourPath, QBrush(gradient))
        if self.preset is not None:
            ################################################
            # we adjust the preset pixmap to pxmp size while keeping
            # its aspect ratio and we center it into pxmp
            ################################################
            w, h = self.preset.width(), self.preset.height()
            # get the bounding rect of the scaled and centered preset
            # and the 2 complementary rects
            if w > h:
                rh = int(self.baseSize * h / w)   # height of bounding rect
                m = int((self.baseSize - rh) / 2.0)  # top and bottom margins
                r = QRect(0, m, self.baseSize, rh)
                r1 = QRect(0, 0, self.baseSize, m)
                r2 = QRect(0, rh + m, self.baseSize, m)
            else:
                rw = int(self.baseSize * w / h)  # width of bounding rect
                m = int((self.baseSize - rw) / 2.0)  # left and right margins
                r = QRect(m, 0, rw, self.baseSize)
                r1 = QRect(0, 0, m, self.baseSize)
                r2 = QRect(rw + m, 0, m, self.baseSize)
            # set opacity of r to that of preset
            qp.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            qp.drawPixmap(r, self.preset)
            # paint the outside of r with transparent color
            pxmp1 = QPixmap(pxmp.size())
            pxmp1.fill(QColor(0, 0, 0, 0))
            qp.drawPixmap(r1, pxmp1)
            qp.drawPixmap(r2, pxmp1)
        qp.end()
        s = size / self.baseSize
        self.pxmp = pxmp.transformed(QTransform().scale(s, s).rotate(orientation)) # pxmp.scaled(size, size)
        pattern = pattern
        return {'family': self, 'name': self.name, 'pixmap': self.pxmp, 'size': size, 'color': color, 'opacity': opacity,
                'hardness': hardness, 'flow': flow, 'spacing': spacing, 'jitter': jitter, 'orientation': orientation,
                'pattern': pattern, 'cursor': self.baseCursor}


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
    ##########
    # eraser
    ##########
    qpp = QPainterPath()
    qpp.addEllipse(QRect(0, 0, baseSize, baseSize))
    eraserFamily = brushFamily('Eraser', baseSize, qpp)
    # eraser must be added last
    brushes.append(eraserFamily)
    return brushes

def loadPresets(filename, first=1):
    """
    Loads brush preset from file
    @param filename:
    @type filename: str
    @return:
    @rtype:  list of brushFamily instances
    """
    brushes = []
    patterns = []
    baseSize = 400  # 25
    try:
        rank = first
        entry = os.path.basename(filename)
        if entry[-4:].lower() in ['.png', '.jpg']:
            try:
                qpp = QPainterPath()
                qpp.addEllipse(QRect(0, 0, baseSize, baseSize))
                presetBrushFamily = brushFamily('Preset ' + str(rank), baseSize, qpp,
                                                presetFilename=os.getcwd() + '\\' + BRUSHES_PATH + '\\' + entry)
                brushes.append(presetBrushFamily)
                rank += 1
            except IOError:
                pass
        elif entry[-4:].lower() in ['.abr']:
            sImages, pImages = aParser.readFile(os.getcwd() + '\\' + BRUSHES_PATH + '\\' + entry)
            for im in sImages:
                qpp = QPainterPath()
                qpp.addEllipse(QRect(0, 0, baseSize, baseSize))
                alpha = np.full_like(im, 255)
                im = np.dstack((im, im, im, im) ) # alpha))
                qim = ndarrayToQImage(im, format=QImage.Format_ARGB32)
                presetBrushFamily = brushFamily('Preset ' + str(rank), baseSize, qpp, image=qim)
                brushes.append(presetBrushFamily)
                rank += 1
            rank = first
            for im in pImages:
                #alpha = np.full_like(im, 255)
                im = np.dstack((im, im, im, im))
                qim = ndarrayToQImage(im, format=QImage.Format_ARGB32)
                p = pattern('pattern ' + str(rank), im=qim, pxmp=QPixmap.fromImage(qim))
                patterns.append(p)
                rank += 1
    except IOError:
        pass
    return brushes, patterns


def bLUeFloodFill(layer, x, y, color):
    """
    Flood fills a region of a drawing layer
    x, y are the seed coordinates
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
    if 0 <= x < w  and 0 <= y < h:
        cv2.floodFill(buf, mask, (x, y), (color.red(), color.green(), color.blue()))
    buf0[..., :3] = buf[..., ::-1]
    # set the alpha channel of the filled region
    buf0[mask[1:-1, 1:-1] == 1, 3] = color.alpha()


