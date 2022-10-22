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
from PySide2.QtGui import QPainterPathStroker, QBrush, QPixmap
from PySide2.QtCore import QRect, QPointF, QPoint
from PySide2.QtWidgets import QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsPolygonItem, \
    QGraphicsSceneMouseEvent, QHBoxLayout
from PySide2.QtGui import QColor, QPen, QPainterPath, QPolygonF
from PySide2.QtCore import Qt, QRectF

from bLUeGui.graphicsForm import graphicsCurveForm
from bLUeTop.utils import optionsWidget, QbLUePushButton
from .baseSignal import baseSignal_No
from .spline import interpolationCubSpline, interpolationQuadSpline, displacementSpline
from .const import channelValues


##########################################################################################
# GUI for interactive construction of 1D LUTs and 2D LUTs.
# 1D LUT are calculated as interpolation splines defined by a set of control points.
# 2D LUTs are calculated as displacement (delta) splines defined by a set of bumps.
# Control points and splines are represented by QGraphicsPathItem instances in a QGraphicsScene.
# The origin of the scene is at the bottom left and the Y axis points downwards.
# Mouse event handlers are reimplemented to provide full control (move range, removing,...)
###########################################################################################


class activePoint(QGraphicsPathItem):
    """
    Interactive point
    """

    def __init__(self, x, y, color=Qt.white, fillColor=None, persistent=False, rect=None, parentItem=None):
        super().__init__(parent=parentItem)
        self.color = color
        self.setAcceptHoverEvents(True)
        self.persistent = persistent
        self.rect = rect
        if self.rect is not None:
            self.xmin, self.xmax, self.ymin, self.ymax = rect.left(), rect.right(), rect.top(), rect.bottom()
            x = min(max(x, self.xmin), self.xmax)
            y = min(max(y, self.ymin), self.ymax)
        self.setPos(QPointF(x, y))
        self.clicked = False
        self.setPen(QPen(color, 2))
        # filling brush
        if fillColor is not None:
            self.setBrush(QBrush(fillColor))
        qpp = QPainterPath()
        # coordinates are relative to activePoint
        qpp.addEllipse(-4, -4, 8, 8)
        self.setPath(qpp)

    def mousePressEvent(self, e):
        self.clicked = True

    def mouseMoveEvent(self, e):
        self.clicked = False
        x, y = e.scenePos().x(), e.scenePos().y()
        if self.rect is not None:
            x = min(max(x, self.xmin), self.xmax)
            y = min(max(y, self.ymin), self.ymax)
        self.setPos(x, y)

    def hoverEnterEvent(self, *args, **kwargs):
        self.setPen(QPen(QColor(0, 255, 0), 2))
        self.update()

    def hoverLeaveEvent(self, *args, **kwargs):
        self.setPen(QPen(self.color, 2))
        self.update()


class activeTriangle(QGraphicsPathItem):
    """
    interactive bump triangle
    """

    def __init__(self, x, y, bump, persistent=False, rect=None, parentItem=None):
        super().__init__(parent=parentItem)
        self.setAcceptHoverEvents(True)
        self.persistent = persistent
        self.rect = rect
        self.setPos(QPointF(x, y))
        self.clicked = False
        # coordinates are relative to activeTriangle
        self.B, self.C, self.A = QPointF(0, 0), QPointF(50, 0), QPointF(25, -bump)
        self.setPen(QPen(QColor(255, 255, 255), 2))
        self.update()

    def update(self):
        qpp = QPainterPath()
        trans = QPointF(-4, -4)
        # coordinates are relative to activeTriangle
        for p in [self.A, self.B, self.C]:
            qpp.addEllipse(p + trans, 4, 4)
        self.setPath(qpp)
        super().update()

    def mousePressEvent(self, e):
        self.clicked = True
        xt, yt = e.pos().x(), e.pos().y()
        p = QPointF(xt, yt)
        self.moving = None
        for p in [self.A, self.B, self.C]:
            if abs(xt - p.x()) + abs(yt - p.y()) < 15:
                self.moving = p
                break

    def mouseMoveEvent(self, e):
        if self.moving is None:
            return
        self.clicked = False
        xt, yt = e.pos().x(), e.pos().y()
        if self.moving is self.A:
            self.A.setY(yt)
        else:
            if self.moving is self.B:
                x1, x2 = xt, self.C.x()
            elif self.moving is self.C:
                x1, x2 = xt, self.B.x()
            self.A.setX((self.B.x() + self.C.x()) / 2)
            self.B.setX(min(x1, x2))
            self.C.setX(max(x1, x2))
        self.update()
        self.parentItem().updatePath()

    def mouseReleaseEvent(self, e):
        # get scene current spline
        sc = self.scene()
        # get parent spline
        activeSpline = self.parentItem()  # sc.cubicItem
        # click event : remove point
        if self.clicked:
            if self.persistent:
                return
            activeSpline.fixedPoints.remove(self)
            sc.removeItem(self)
            return
        activeSpline.updateLUTXY()
        activeSpline.curveChanged.sig.emit()

    def hoverEnterEvent(self, *args, **kwargs):
        self.setPen(QPen(QColor(0, 255, 0), 2))
        self.update()

    def hoverLeaveEvent(self, *args, **kwargs):
        self.setPen(QPen(QColor(255, 255, 255), 2))
        self.update()


class activeMarker(QGraphicsPolygonItem):
    """
    Movable marker
    """

    size = 10
    triangle = QPolygonF()
    triangle.append(QPointF(-size, size))
    triangle.append(QPointF(0, 0))
    triangle.append(QPointF(size, size))

    cross = QPolygonF()
    cross.append(QPointF(-size / 2, -size / 2))
    cross.append(QPointF(0, 0))
    cross.append(QPointF(size / 2, size / 2))
    cross.append(QPointF(0, 0))
    cross.append(QPointF(-size / 2, size / 2))
    cross.append(QPointF(0, 0))
    cross.append(QPointF(size / 2, -size / 2))
    cross.append(QPointF(0, 0))

    @classmethod
    def fromTriangle(cls, *args, **kwargs):
        color = QColor(255, 255, 255)
        item = cls(*args, **kwargs)
        item.setPolygon(cls.triangle)
        item.setPen(QPen(color))
        item.setBrush(QBrush(color))
        # set move range to parent bounding rect
        item.moveRange = item.parentItem().boundingRect()
        return item

    @classmethod
    def fromCross(cls, *args, **kwargs):
        color = QColor(0, 0, 0)
        item = cls(*args, **kwargs)
        item.setPolygon(cls.cross)
        item.setPen(QPen(color))
        item.setBrush(QBrush(color))
        # set move range to parent bounding rect
        item.moveRange = item.parentItem().boundingRect()
        return item

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.onMouseMove, self.onMouseRelease = lambda e, x, y: 0, lambda e, x, y: 0
        self.moveRange = QRectF(0.0, 0.0, 0.0, 0.0)

    @property  # read only
    def currentColor(self):
        return self.scene().slider2D.QImg.pixelColor((self.pos() - self.parentItem().offset()).toPoint())

    def setMoveRange(self, rect):
        self.moveRange = rect

    def mousePressEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        # event position relative to parent
        pos = e.pos() + self.pos()  # e.scenePos() - self.parentItem().scenePos()
        x, y = pos.x(), pos.y()
        # limit move to moveRange
        xmin, ymin = self.moveRange.left(), self.moveRange.top()
        xmax, ymax = self.moveRange.right(), self.moveRange.bottom()
        x, y = xmin if x < xmin else xmax if x > xmax else x, ymin if y < ymin else ymax if y > ymax else y
        self.setPos(x, y)
        self.onMouseMove(e, x, y)

    def mouseReleaseEvent(self, e):
        # event position relative to parent
        pos = e.pos() + self.pos()  # e.scenePos() - self.parentItem().scenePos()
        x, y = pos.x(), pos.y()
        # limit move to (0,0) and moveRange
        xmin, ymin = self.moveRange.left(), self.moveRange.top()
        xmax, ymax = self.moveRange.right(), self.moveRange.bottom()
        x, y = xmin if x < xmin else xmax if x > xmax else x, ymin if y < ymin else ymax if y > ymax else y
        self.onMouseRelease(e, x, y)


class activeRsMarker(activeMarker):
    """
    Marker for range slider. role is 'min' or 'max'
    """

    def __init__(self, parent=None, role=''):
        super().__init__(parent=parent)
        self.role = role

    def sceneEventFilter(self, target, e):
        """
        Filtering of mouse events for range slider markers :
        maintains marker order.

        :param target:
        :type  target:
        :param e:
        :type  e:
        """
        if isinstance(e, QGraphicsSceneMouseEvent):
            if self.role == 'min':
                return e.scenePos().x() <= self.scenePos().x() + self.size
            else:
                return e.scenePos().x() >= self.scenePos().x() - self.size
        return False


class activeSplinePoint(activePoint):
    """
    Interactive (movable and removable) control point
    for a spline in a QGraphicsScene.
    """

    def __init__(self, x, y, persistent=False, rect=None, parentItem=None):
        """
        Interactive control point for the scene current spline.
        Persistent activePoints cannot be removed
        by mouse click (default is non persistent). If rect is not None,
        the moves of the point are restricted to rect.

        :param x: initial x-coordinate
        :type  x: float
        :param y: initial y-coordinate
        :type  y: float
        :param persistent: persistent flag
        :type  persistent: boolean
        :param parentItem:
        :type  parentItem: object
        """
        super().__init__(x, y, persistent=persistent, rect=rect, parentItem=parentItem)
        self.tangent = None  # link to tangent : used only by qudratic spline

    def mouseMoveEvent(self, e):
        self.clicked = False
        item = self.parentItem()
        if item is None:
            return
        p = e.pos() + self.pos()
        x, y = p.x(), p.y()  # e.scenePos().x(), e.scenePos().y()
        if self.rect is not None:
            x = min(max(x, self.xmin), self.xmax)
            y = min(max(y, self.ymin), self.ymax)
        self.setPos(x, y)
        if self.tangent is not None:
            controlPoint, contactPoint = self.tangent.controlPoint, self.tangent.contactPoint
            v = controlPoint - contactPoint
            contactPoint = QPointF(x, y)
            self.tangent.contactPoint = contactPoint
            self.tangent.controlPoint = contactPoint + v
            self.tangent.setPos(contactPoint)
        item.fixedPoints.sort(key=lambda p: p.scenePos().x())
        item.updatePath()

    def mouseReleaseEvent(self, e):
        # get scene current spline
        item = self.parentItem()  # self.scene().cubicItem
        if item is None:
            return
        p = e.pos() + self.pos()
        x, y = p.x(), p.y()  # e.scenePos().x(), e.scenePos().y()
        if self.rect is not None:
            x = min(max(x, self.xmin), self.xmax)
            y = min(max(y, self.ymin), self.ymax)
        self.setPos(x, y)
        sc = self.scene()
        # click event : remove point
        if self.clicked:
            if self.persistent:
                return
            item.fixedPoints.remove(self)
            # remove tangent if any
            fxdtg = getattr(item, 'fixedTangents', None)
            if fxdtg:
                fxdtg.remove(self.tangent)
                sc.removeItem(self.tangent)
            sc.removeItem(self)
            return
        item.updatePath()
        item.updateLUTXY()
        item.curveChanged.sig.emit()


class activeTangent(QGraphicsPathItem):
    """
    Interactive tangent
    """
    strokeWidth = 2
    penWidth = 2
    brushColor = Qt.darkGray

    def __init__(self, controlPoint=QPointF(), contactPoint=QPointF(), parentItem=None):
        super().__init__(parent=parentItem)
        self.savedPen = self.pen()
        self.setAcceptHoverEvents(True)
        self.controlPoint = controlPoint
        self.contactPoint = contactPoint
        self.setPos(contactPoint)
        qpp = QPainterPath()
        # coordinates are relative to activeTangent object
        qpp.moveTo(0, 0)
        qpp.lineTo((controlPoint - contactPoint))
        qpp.addEllipse(controlPoint - contactPoint, 5.0, 5.0)
        self.setPath(qpp)
        self.setZValue(-1)
        # set item pen
        self.setPen(QPen(QBrush(self.brushColor), self.penWidth))

    def updatePath(self):
        qpp = QPainterPath()
        # coordinates are relative to activeTangent object
        qpp.moveTo(0, 0)
        qpp.lineTo((self.controlPoint - self.contactPoint))
        qpp.addEllipse(self.controlPoint - self.contactPoint, 5.0, 5.0)
        self.setPath(qpp)

    def mousePressEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        newPos = e.scenePos()
        if abs(self.contactPoint.x() - newPos.x()) < 5:
            return
        slope = - (self.contactPoint.y() - newPos.y()) / (self.contactPoint.x() - newPos.x())
        if slope > 20 or slope < 0:
            return
        self.controlPoint = newPos
        # update tangent path
        self.updatePath()
        # update spline
        self.scene().cubicItem.updatePath()

    def mouseReleaseEvent(self, e):
        cb = self.scene().cubicItem
        cb.updatePath()
        cb.updateLUTXY()
        cb.curveChanged.sig.emit()

    def hoverEnterEvent(self, *args, **kwargs):
        self.setPen(QPen(QColor(0, 255, 0), 2))
        self.update()

    def hoverLeaveEvent(self, *args, **kwargs):
        self.setPen(self.savedPen)
        self.update()


class activeSpline(QGraphicsPathItem):
    """
    Base class for interactive splines.
    The attribute self.spline holds the list of
    QPointF instances to display (scene coordinates).
    It defines the path to display; sync with the scene is
    achieved by updatePath().
    """
    strokeWidth = 2
    penWidth = 2
    brushColor = Qt.darkGray

    def __init__(self, size, fixedPoints=None, baseCurve=None, parentItem=None):
        """
        Init an interactive spline with an empty set of control points and
        an empty curve.

        :param size: initial path size
        :type  size: int
        :param fixedPoints:
        :type  fixedPoints:
        :param baseCurve: starting (initial) curve
        :type  baseCurve: 2-uple of QPoint
        :param parentItem:
        :type  parentItem: object
        """
        self.curveChanged = baseSignal_No()
        super().__init__(parent=parentItem)
        if fixedPoints is None:
            fixedPoints = []
        self.size = size
        # default initial curve : diagonal
        if baseCurve is None:
            baseCurve = (QPoint(0, 0), QPoint(size, -size))
        qpp = QPainterPath()
        qpp.moveTo(baseCurve[0])
        qpp.lineTo(baseCurve[1])
        # stroke curve
        stroker = QPainterPathStroker()
        stroker.setWidth(self.strokeWidth)
        self.mboundingPath = stroker.createStroke(qpp)
        self.setPath(self.mboundingPath)
        self.clicked = False
        self.setVisible(False)
        self.fixedPoints = fixedPoints
        self.__spline = []
        # self.LUTXY is the 1D LUT : range 0..255 --> 0..255, type ndarray, dtype=int, size=256
        self.LUTXY = np.arange(256)
        self.channel = channelValues.RGB
        self.histImg = None
        # set item pen
        self.setPen(QPen(QBrush(self.brushColor), self.penWidth))

    @property
    def spline(self):
        return self.__spline

    @spline.setter
    def spline(self, data):
        """

        :param data: curve data (scene coordinates)
        :type  data: list of QPointF instances
        """
        self.__spline = data

    def mousePressEvent(self, e):
        self.clicked = True

    def mouseMoveEvent(self, e):
        self.clicked = False

    def mouseReleaseEvent(self, e):
        """
        if clicked, add a control point to the curve
        """
        # click event
        if self.clicked:
            # add point
            p = e.pos()
            a = activeSplinePoint(p.x(), p.y(), parentItem=self)
            self.fixedPoints.append(a)
            self.fixedPoints.sort(key=lambda z: z.scenePos().x())
            self.updatePath()

    def initFixedPoints(self):
        """
        Add 2 boundary control points and a central one.
        """
        axeSize = self.size
        rect = QRectF(0.0, -axeSize, axeSize, axeSize)
        self.fixedPoints = [activeSplinePoint(0, 0, persistent=True, rect=rect, parentItem=self),
                            activeSplinePoint(axeSize / 2, -axeSize / 2, rect=rect, parentItem=self),
                            activeSplinePoint(axeSize, -axeSize, persistent=True, rect=rect, parentItem=self)]

    def setFixedPoints(self, points):
        for p in self.fixedPoints:
            sc = p.scene()
            if sc is not None:
                sc.removeItem(p)
        for p in points:
            p.setParentItem(self)
        self.fixedPoints = points
        self.updatePath()
        self.updateLUTXY()

    def updatePath(self):
        """
        Update and display the spline. Derived classes should
        override and call it after each control
        point or tangent modification: see
        activePoint and activeTangent mouse event handlers
        """
        pass

    def updateLUTXY(self):
        """
        Sync the LUT with the spline
        """
        scale = 255.0 / self.size
        self.LUTXY = np.array([int((-p.y()) * scale) for p in self.spline])

    def __getstate__(self):
        s = self.size
        return {'fixedpoints': [(p.x() / s, p.y() / s) for p in self.fixedPoints]}

    def __setstate__(self, state):
        s = self.size
        fixedPoints = [activeSplinePoint(item[0] * s, item[1] * s) for item in state['fixedpoints']]
        self.setFixedPoints(fixedPoints)

    def writeToStream(self, outStream):
        outStream.writeInt32(self.size)
        outStream.writeInt32(len(self.fixedPoints))
        for point in self.fixedPoints:
            outStream << point.pos()
        return outStream

    def readFromStream(self, inStream):
        size = inStream.readInt32()
        count = inStream.readInt32()
        for point in self.childItems():
            self.scene().removeItem(point)
        self.fixedPoints = []
        for i in range(count):
            point = QPointF()
            inStream >> point
            self.fixedPoints.append(activeSplinePoint(point.x(), point.y(), parentItem=self))
        self.updatePath()
        self.updateLUTXY()
        return self


class activeBSpline(activeSpline):
    """
    Interactive displacement spline.
    """
    # To display periodic splines, the interval [0, period] is
    # represented by [0, axeSize] and the curve is enlarged
    # by periodViewing on both sides.
    periodViewing = 50

    def __init__(self, size, period=0, yZero=0):
        """
        :param size: Spline size (scene coords)
        :type  size: int
        :param period: Spline period (scene coords.) or 0
        :type  period: int
        :param yZero: curve origin is (0, yZero)
        :type  yZero: int
        """
        super().__init__(size, fixedPoints=None, parentItem=None, baseCurve=(QPoint(0, yZero), QPoint(size, yZero)))
        self.period = period
        self.yZero = yZero
        # x-coordinates of the  curve
        self.xCoords = np.arange(size + 2 * self.periodViewing) - self.periodViewing
        self.spline = [QPointF(x, yZero) for x in self.xCoords]  # scene coord.

    def mouseReleaseEvent(self, e):
        """
        if clicked, add a bump triangle to the curve
        """
        # click event
        if self.clicked:
            # add point
            p = e.pos()
            a = activeTriangle(p.x(), p.y(), 50, parentItem=self)
            self.fixedPoints.append(a)
            self.fixedPoints.sort(key=lambda z: z.scenePos().x())
            self.updatePath()

    def initFixedPoints(self):
        """
        Add 2 boundary control points.
        Coordinates are relative to the scene.
        """
        axeSize = self.size
        rect = QRectF(0.0, -axeSize, axeSize, axeSize)
        self.fixedPoints = [activeTriangle(50, self.yZero, 25, persistent=True, rect=rect, parentItem=self)]

    def updatePath(self):
        axeSize = self.size
        yZero = self.yZero
        try:
            X = []
            for item in self.fixedPoints:
                X.extend([item.B.x() + item.x(), item.C.x() + item.x()])
            X = np.array(X)
            Y = np.array([-(item.A.y() + item.y()) for item in self.fixedPoints]) + yZero
            T = displacementSpline(X, Y, self.xCoords,
                                   clippingInterval=[-self.scene().axeSize, 0], period=self.period)
            self.spline = [QPointF(x, y + yZero) for x, y in zip(self.xCoords, -T)]  # scene coord.
            # build path
            polygon = QPolygonF(self.spline)
            qpp = QPainterPath()
            qpp.addPolygon(polygon)
            # stroke path
            stroker = QPainterPathStroker()
            stroker.setWidth(self.strokeWidth)
            mboundingPath = stroker.createStroke(qpp)
            self.setPath(mboundingPath)
        except ValueError:
            pass


class activeCubicSpline(activeSpline):
    """
    Interactive cubic spline. Control points can be :
        - added by a mouse click on the curve,
        - moved with the mouse (cf. activePoint.mouseMoveEvent())
        - removed by a mouse click on the point (cf. activePoint.mouseReleaseEvent())
    """

    def updatePath(self):
        """
        Update and display the spline. Should be called after
        each control point or tangent modification : see
        activePoint and activeTangent mouse event handlers
        """
        # add ending control points, if needed,
        # to get full range 0..self.size
        X = [item.x() for item in self.fixedPoints]
        Y = [item.y() for item in self.fixedPoints]
        X0, X1 = X[0], X[-1]
        Y0, Y1 = Y[0], Y[-1]
        Y2 = Y0 - X0 * (Y1 - Y0) / (X1 - X0)
        Y3 = Y0 + (self.size - X0) * (Y1 - Y0) / (X1 - X0)
        if X[0] > 0.0:
            X.insert(0, 0.0)
            Y.insert(0, Y2)
        if X[-1] < self.size:
            X.append(self.size)
            Y.append(Y3)
        # interpolate
        try:
            # interpolationCubSpline raises an exception if two points have identical x-coordinates
            self.spline = interpolationCubSpline(np.array(X), np.array(Y), clippingInterval=[-self.size, 0])
            # set the curve constant outside ]X0..X1[
            for P in self.spline:
                if P.x() < X0:
                    P.setY(Y0)
                elif P.x() > X1:
                    P.setY(Y1)
            # build path
            polygon = QPolygonF(self.spline)
            qpp = QPainterPath()
            qpp.addPolygon(polygon)
            # stroke path
            stroker = QPainterPathStroker()
            stroker.setWidth(self.strokeWidth)
            mboundingPath = stroker.createStroke(qpp)
            self.setPath(mboundingPath)
        except ValueError:
            pass

    def getLUTXY(self):
        """
        Returns the LUT.

        :return: LUT
        :rtype: ndarray
        """
        return self.LUTXY

    def getStackedLUTXY(self):
        """
        Returns the stacked LUT (A row for each channel).

        :return: LUT
        :rtype: ndarray, shape (3,n)
        """
        if self.channel == channelValues.RGB:
            return np.vstack((self.LUTXY, self.LUTXY, self.LUTXY))
        else:
            return np.vstack((self.scene().cubicR.LUTXY, self.scene().cubicG.LUTXY, self.scene().cubicB.LUTXY))

    def reset(self):
        for point in self.childItems():
            self.scene().removeItem(point)
        self.initFixedPoints()
        # calculate spline
        self.updatePath()
        self.LUTXY = np.arange(256)


class activeQuadricSpline(activeSpline):
    """
    Interactive quadratic spline. Control points can be :
        - added by a mouse click on the curve,
        - moved with the mouse (cf. activePoint.mouseMoveEvent())
        - removed by a mouse click on the point (cf. activePoint.mouseReleaseEvent())
    """
    halfTgLen = 50.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixedTangents = []
        self.hasVisibleTangents = False  # default: tangents not shown

    def setTangentsVisible(self, flag):
        """
        Toggles tangents visibility.

        :param flag:
        :type  flag: boolean
        """
        self.hasVisibleTangents = flag
        for t in self.fixedTangents:
            t.setVisible(flag)

    def updatePath(self, calculate=True):
        """
        Calculates (if calculate=true) and displays the spline.
        Called by activePoint and activeTangent mouse event.

        :param calculate:
        :type  calculate: bool
        """
        # calculate the array of slopes
        d = [item.controlPoint - item.contactPoint for item in self.fixedTangents]
        d1 = - np.array(list(map(lambda a: a.y(), d)))
        d2 = np.array(list(map(lambda a: a.x(), d)))
        d = d1 / d2
        # add boundary points if needed
        X = [item.x() for item in self.fixedPoints]
        Y = [item.y() for item in self.fixedPoints]
        X0, X1 = X[0], X[-1]
        Y0, Y1 = Y[0], Y[-1]
        t = (Y1 - Y0) / (X1 - X0)
        Y2 = Y0 - X0 * t
        Y3 = Y0 + (self.size - X0) * t
        d = d.tolist()
        if X[0] > 0.0:
            X.insert(0, 0.0)
            Y.insert(0, Y2)
            d.insert(0, t)
        if X[-1] < self.size:
            X.append(self.size)
            Y.append(Y3)
            d.append(t)
        d = np.array(d)
        try:
            if calculate:
                T = interpolationQuadSpline(np.array(X) / self.size, -np.array(Y) / self.size, d) * self.size
                self.spline = [QPointF(x, y) for x, y in zip(np.arange(256) * (self.size / 255.0), -T)]
            for P in self.spline:
                if P.x() < X0:
                    P.setY(Y0)
                elif P.x() > X1:
                    P.setY(Y1)
            polygon = QPolygonF(self.spline)
            qpp = QPainterPath()
            qpp.addPolygon(polygon)
            # stroke path
            stroker = QPainterPathStroker()
            stroker.setWidth(self.strokeWidth)
            mboundingPath = stroker.createStroke(qpp)
            self.setPath(mboundingPath)
        except Exception as e:
            print(str(e))

    def setCurve(self, a, b, d, T):
        """
        Initialises the spline and the LUT. See also setFixed().
        Parameters a, b, d, T  correspond to values
        returned by warpHistogram(), scaled by the size of curve axes.
        Class instance attributes self.a, self.b, self.d, self.T store
        the method parameters used in the last call,
        to enable auto curve reset. They should not be used for
        other purposes.
        a, b, d must have identical sizes.
        a, b, T values are in range 0.. self.size

        :param a: x-ccoordinates of control points
        :type  a: ndarray
        :param b: y-coordinates of control points
        :type  b: ndarray
        :param d: tangent slopes
        :type  d: ndarray
        :param T: spline array
        :type  T: ndarray, size 256
        """
        self.a, self.b, self.d, self.T = a, b, d, T
        rect = QRectF(0.0, -self.size, self.size, self.size)
        # half tangent length and orientation
        alpha = [self.halfTgLen] * len(d)
        # choose backward orientation for the last tangent
        alpha[-1] = - alpha[-1]
        for item in self.fixedPoints:
            self.scene().removeItem(item)
        for item in self.fixedTangents:
            self.scene().removeItem(item)
        self.fixedPoints = [activeSplinePoint(x, -y, rect=rect, parentItem=self) for x, y in zip(a, b)]
        # tangent normalization
        n = np.sqrt(1 + d * d)
        alpha = alpha / n  # converts the list to a ndarray
        self.fixedTangents = [activeTangent(controlPoint=QPointF(x + alpha[i], -y - alpha[i] * p),
                                            contactPoint=QPointF(x, -y), parentItem=self)
                              for i, (x, y, p) in enumerate(zip(a, b, d))]
        # link contact point to tangent
        for i1, i2 in zip(self.fixedPoints, self.fixedTangents):
            i1.tangent = i2
            i1.tangent.setVisible(self.hasVisibleTangents)
        self.spline = [QPointF(x, y) for x, y in zip(np.arange(256) * (self.size / 255), -T)]
        self.updatePath(calculate=False)  # don't recalculate the spline!
        self.updateLUTXY()

    def mouseReleaseEvent(self, e):
        """
        Adds a control point to the curve.

        :param e:
        """
        # click event
        if self.clicked:
            # add point
            p = e.pos()
            a = activeSplinePoint(p.x(), p.y(), parentItem=self)
            self.fixedPoints.append(a)
            self.fixedPoints.sort(key=lambda z: z.scenePos().x())
            t = activeTangent(controlPoint=p + QPointF(0.7, -0.7) * self.halfTgLen, contactPoint=p, parentItem=self)
            t.setVisible(self.hasVisibleTangents)
            a.tangent = t
            self.fixedTangents.insert(self.fixedPoints.index(a), t)
            self.updatePath()

    def getStackedLUTXY(self):
        """
        Returns the 3-channel LUT (A 1-line LUT for each channel).

        :return: LUT
        :rtype: ndarray, shape (3,n)
        """
        if self.channel == channelValues.RGB:
            return np.vstack((self.LUTXY, self.LUTXY, self.LUTXY))
        else:
            return np.vstack((self.scene().cubicR.LUTXY, self.scene().cubicG.LUTXY, self.scene().cubicB.LUTXY))

    def reset(self):
        self.setCurve(self.a, self.b, self.d, self.T)
        """
        #self.clicked = QPoint(0, 0)
        #self.selected = False
        for point in self.childItems():
            self.scene().removeItem(point)
        self.initFixedPoints()
        #calculate spline
        self.updatePath()
        LUT = range(256)
        self.LUTXY = np.array(LUT)  # buildLUT(LUT)
        """

    def setFixed(self, points, tangents):
        """
        Initialises the spline and the LUT.
        See also setCurve().

        :param points:
        :type  points: list of activeSplinePoint
        :param tangents:
        :type  tangents: list of activeTangent
        """
        for p in self.fixedPoints:
            sc = p.scene()
            if sc is not None:
                sc.removeItem(p)
        for p in points:
            p.setParentItem(self)
        self.fixedPoints = points
        for t in self.fixedTangents:
            sc = t.scene()
            if sc is not None:
                sc.removeItem(t)
        for t in tangents:
            t.setParentItem(self)
        for p, t in zip(points, tangents):
            p.tangent = t
        self.fixedTangents = tangents
        self.setTangentsVisible(self.hasVisibleTangents)  # TODO added 3/1/22 validate
        self.updatePath()
        self.updateLUTXY()
        # enable resetting to this curve (see method setCurve())
        self.a = np.array([p.x() for p in self.fixedPoints])
        self.b = np.array([- p.y() for p in self.fixedPoints])
        self.d = np.array([- (t.controlPoint.y() - t.contactPoint.y()) / (t.controlPoint.x() - t.contactPoint.x())
                           for t in self.fixedTangents])
        self.T = self.LUTXY / 256 * self.size

    def __getstate__(self):
        d = {}
        s = self.size
        d['fixedpoints'] = [(p.x() / s, p.y() / s) for p in self.fixedPoints]
        d['fixedtangentcontact'] = [(p.x() / s, p.y() / s) for p in [tg.contactPoint for tg in self.fixedTangents]]
        d['fixedtangentcontrol'] = [(p.x() / s, p.y() / s) for p in [tg.controlPoint for tg in self.fixedTangents]]
        return d

    def __setstate__(self, state):
        s = self.size
        fixedPoints = [activeSplinePoint(item[0] * s, item[1] * s) for item in state['fixedpoints']]
        fixedTangents = [activeTangent(contactPoint=QPointF(item[0][0] * s, item[0][1] * s),
                                       controlPoint=QPointF(item[1][0] * s, item[1][1] * s))
                         for item in zip(state['fixedtangentcontact'], state['fixedtangentcontrol'])]
        self.setFixed(fixedPoints, fixedTangents)


class graphicsSplineItem(QGraphicsPixmapItem):
    """
    graphic spline component
    """

    def __init__(self, size=100, parentItem=None):
        super().__init__()
        self.setParentItem(parentItem)
        self.targetObject = None
        self.axeSize = size
        # background
        pxmp = QPixmap(size, size)
        pxmp.fill(Qt.lightGray)
        self.setPixmap(pxmp)
        # curve
        cubic = activeCubicSpline(size)
        cubic.setVisible(True)
        cubic.setParentItem(self)
        cubic.setPos(0, size)
        self.defaultAxes = graphicsCurveForm.drawPlotGrid(size)
        cubic.axes = self.defaultAxes
        cubic.initFixedPoints()
        self.cubic = cubic
        # graphicsScene.cubicR = cubic
        # cubic.channel = channelValues.L
        # get histogram as a Qimage
        # cubic.histImg = graphicsScene.layer.inputImg().histogram(size=graphicsScene.axeSize,
        # bgColor=graphicsScene.bgColor, range=(0, 1),
        # chans=channelValues.L, mode='Lab')


class graphicsThrSplineItem(graphicsSplineItem):
    """
     graphic spline + range slider component
    """

    def __init__(self, size=100, border=20, parentItem=None):
        super().__init__(size=size, parentItem=parentItem)
        # brightness sliders
        self.brightnessSliderHeight = 20
        self.brightnessSliderWidth = size  # + 2 * border
        px = QPixmap(self.brightnessSliderWidth, self.brightnessSliderHeight)
        px.fill(Qt.gray)
        self.brightnessSlider = QGraphicsPixmapItem(px, parent=self)
        self.brightnessSlider.setPos(0, size + 20)
        # brightnessSlider handles
        self.brightnessThr0 = activeRsMarker.fromTriangle(parent=self.brightnessSlider, role='min')
        self.brightnessThr0.setMoveRange(
            QRectF(0.0, self.brightnessThr0.size, self.brightnessSlider.pixmap().width(), 0.0))
        self.brightnessThr0.setPos(0, self.brightnessSlider.pixmap().height() - self.brightnessThr0.size)
        self.brightnessThr0.val = 0.0
        self.brightnessThr1 = activeRsMarker.fromTriangle(parent=self.brightnessSlider, role='max')
        self.brightnessThr1.setMoveRange(
            QRectF(0.0, self.brightnessThr0.size, self.brightnessSlider.pixmap().width(), 0.0))
        self.brightnessThr1.setPos(self.brightnessSlider.pixmap().width(),
                                   self.brightnessSlider.pixmap().height() - self.brightnessThr0.size)
        self.brightnessThr1.val = 1.0

    def mousePressEvent(self, e):
        pass  # don't select on click!


class graphicsSplineForm(graphicsCurveForm):
    """
    Form for interactive cubic or quadratic spline.
    """

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, curveType='quadric'):
        newWindow = graphicsSplineForm(targetImage=targetImage, axeSize=axeSize, layer=layer,
                                       parent=parent, curveType=curveType)
        newWindow.setWindowTitle(layer.name)
        return newWindow

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, curveType='quadric'):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        graphicsScene = self.scene()
        # init the curve
        if curveType == 'quadric':
            curve = activeQuadricSpline(graphicsScene.axeSize)
        else:
            curve = activeCubicSpline(graphicsScene.axeSize)
        graphicsScene.addItem(curve)
        graphicsScene.quadricB = curve
        curve.channel = channelValues.Br
        curve.histImg = graphicsScene.layer.inputImg().histogram(size=graphicsScene.axeSize,
                                                                 bgColor=graphicsScene.bgColor,
                                                                 range=(0, 255),
                                                                 chans=channelValues.Br)  # , mode='Luminosity')
        curve.initFixedPoints()
        # set current curve
        graphicsScene.cubicItem = graphicsScene.quadricB
        graphicsScene.cubicItem.setVisible(True)
        self.setWhatsThis(
            """<b>Contrast Curve</b><br>
            Drag <b>control points</b> and <b>tangents</b> with the mouse.<br>
            <b>Add</b> a control point by clicking on the curve.<br>
            <b>Remove</b> a control point by clicking it.<br>
            <b>Zoom</b> with the mouse wheel.<br>
            """
        )

        def onResetCurve():
            """
            Reset the selected curve
            """
            self.scene().cubicItem.reset()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        # buttons
        pushButton1 = QbLUePushButton("Reset to Auto Curve")
        pushButton1.setGeometry(10, 20, 100, 30)  # x,y,w,h
        pushButton1.clicked.connect(onResetCurve)
        if curveType == 'quadric':
            optionList1, optionNames1 = ['Show Tangents'], ['Show Tangents']
            self.listWidget1 = optionsWidget(options=optionList1, optionNames=optionNames1, exclusive=False)

            def f():
                curve.setTangentsVisible(self.listWidget1.options['Show Tangents'])

            self.listWidget1.itemClicked.connect(f)

        # layout
        gl = QHBoxLayout()
        container = self.addCommandLayout(gl)
        if curveType == 'quadric':
            gl.addWidget(self.listWidget1)
        gl.addWidget(pushButton1)
        self.adjustSize()
        self.setViewportMargins(0, 0, 0, container.height() + 15)

        graphicsScene.addWidget(pushButton1)
        self.pushButton = pushButton1

    def setButtonText(self, text):
        self.pushButton.setText(text)

    def drawBackground(self, qp, qrF):
        graphicsScene = self.scene()
        s = graphicsScene.axeSize
        if graphicsScene.cubicItem.histImg is not None:
            qp.drawImage(QRect(0, -s, s, s), graphicsScene.cubicItem.histImg)
        qp.save()
        qp.setPen(Qt.red)
        if self.baseCurve is not None:
            qp.drawPolyline(self.baseCurve)
        qp.restore()

    def updateHist(self, curve):
        """
        Updates the channel histogram displayed under the curve.

        :param curve:
        :type  curve:

        """
        sc = self.scene()
        curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize, bgColor=sc.bgColor, chans=[], mode='Luminosity')

    def updateHists(self):
        """
        Updates all histograms
        """
        sc = self.scene()
        self.updateHist(sc.cubicItem)
        # Force to redraw histogram
        sc.invalidate(QRectF(0.0, -sc.axeSize, sc.axeSize, sc.axeSize),
                      sc.BackgroundLayer)

    def __getstate__(self):
        return self.scene().quadricB.__getstate__()

    def __setstate__(self, state):
        self.scene().quadricB.__setstate__(state)

    def writeToStream(self, outStream):
        graphicsScene = self.scene()
        layer = graphicsScene.layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        if layer.actionName in ['actionBrightness_Contrast', 'actionCurves_HSpB', 'actionCurves_Lab']:
            outStream.writeQString(self.listWidget1.selectedItems()[0].text())
            graphicsScene.quadricR.writeToStream(outStream)
            graphicsScene.quadricG.writeToStream(outStream)
            graphicsScene.quadricB.writeToStream(outStream)
        return outStream

    def readFromStream(self, inStream):
        actionName = inStream.readQString()
        name = inStream.readQString()
        sel = inStream.readQString()
        cubics = []
        # for i in range(3):
        # cubic = cubicItem.readFromStream(inStream)
        # cubics.append(cubic)
        # kwargs = dict(zip(['cubicR', 'cubicG', 'cubicB'], cubics))
        # self.setEntries(sel=sel, **kwargs)
        graphicsScene = self.scene()
        graphicsScene.quadricR.readFromStream(inStream)
        graphicsScene.quadricG.readFromStream(inStream)
        graphicsScene.quadricB.readFromStream(inStream)
        return inStream
