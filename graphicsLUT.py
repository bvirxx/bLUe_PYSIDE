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
from PySide2.QtGui import QPainterPathStroker, QBrush
from PySide2.QtCore import QRect, QPointF, QPoint
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QSizePolicy, QPushButton, QGraphicsPathItem
from PySide2.QtGui import QColor, QPen, QPainterPath, QPolygonF
from PySide2.QtCore import Qt, QRectF

from debug import tdec
from spline import interpolationCubSpline, interpolationQuadSpline
from utils import optionsWidget, channelValues, drawPlotGrid

##########################################################################################
# GUI for interactive construction of 1D-LUTs.
# 1D-LUT are calculated as interpolation splines for a set of control points.
# Control points and splines are implemented as QGraphicsPathItem instances in a QGraphicsScene.
# Mouse event handlers are reimplemented to provide full control (move range, removing,...)
# Classes : activePoint
#           activeTangent
#           activeSpline, activeCubicSpline, activeQuadricSpline
#           graphicsCurveForm, graphicsCubicForm, graphicsQuadricForm
###########################################################################################

class activePoint(QGraphicsPathItem):
    """
    Interactive (movable and removable) control point for a spline in a scene
    """
    def __init__(self, x,y, persistent=False, rect=None, parentItem=None):
        """
        Interactive control point for the scene current spline.
        Persistent activePoints cannot be removed
        by mouse click (default is non persistent). If rect is not None,
        the moves of the point are restricted to rect.
        @param x: initial x-coordinate
        @type x: float
        @param y: initial y-coordinate
        @type y: float
        @param persistent: persistent flag
        @type persistent: boolean
        @param parentItem:
        @type parentItem: object
        """
        super(activePoint, self).__init__()
        self.tangent = None  # used for qudratic spline
        self.setAcceptHoverEvents(True)
        self.setParentItem(parentItem)
        self.persistent = persistent
        self.rect = rect
        if self.rect is not None:
            self.xmin, self.xmax, self.ymin, self.ymax = rect.left(), rect.right(), rect.top(), rect.bottom()
            x = min(max(x, self.xmin), self.xmax)
            y = min(max(y, self.ymin), self.ymax)
        self.setPos(QPointF(x,y))
        self.clicked = False
        self.setPen(QPen(QColor(255, 255, 255), 2))
        qpp = QPainterPath()
        # coordinates are relative to activePoint
        qpp.addEllipse(-4,-4, 8, 8)
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
        if self.tangent is not None:
            controlPoint, contactPoint = self.tangent.controlPoint, self.tangent.contactPoint
            v = controlPoint - contactPoint
            contactPoint = QPointF(x,y)
            self.tangent.contactPoint = contactPoint
            self.tangent.controlPoint = contactPoint + v
            self.tangent.setPos(contactPoint)
        self.scene().cubicItem.fixedPoints.sort(key=lambda p: p.scenePos().x())
        self.scene().cubicItem.updatePath()

    def mouseReleaseEvent(self, e):
        # get scene current spline
        item = self.scene().cubicItem
        #item.fixedPoints.sort(key=lambda p : p.scenePos().x())
        x, y = e.scenePos().x(), e.scenePos().y()
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
            sc.removeItem(self)
            return
        self.scene().cubicItem.updatePath()
        self.scene().cubicItem.updateLUTXY()
        l = self.scene().layer
        l.applyToStack()  # call warpHistogram()
        l.parentImage.onImageChanged()

    def hoverEnterEvent(self, *args, **kwargs):
        self.setPen(QPen(QColor(0, 255, 0), 2))
        self.update()

    def hoverLeaveEvent(self, *args, **kwargs):
        self.setPen(QPen(QColor(255, 255, 255), 2))
        self.update()

class activeTangent(QGraphicsPathItem):
    """
    Interactive tangent
    """
    strokeWidth = 2
    penWidth = 2
    brushColor = Qt.darkGray

    def __init__(self, controlPoint=QPointF(), contactPoint=QPointF(), parentItem=None):
        super().__init__()
        self.setParentItem(parentItem)
        self.setAcceptHoverEvents(True)
        self.controlPoint = controlPoint
        self.contactPoint = contactPoint
        self.setPos(contactPoint)
        qpp = QPainterPath()
        # coordinates are relative to activeTangent object
        qpp.moveTo(0,0)
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
        if abs(self.contactPoint.x() - newPos.x())< 5:
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
        self.scene().cubicItem.updatePath()
        self.scene().cubicItem.updateLUTXY()
        l = self.scene().layer
        l.applyToStack()  # call warpHistogram()
        l.parentImage.onImageChanged()

    def hoverEnterEvent(self, *args, **kwargs):
        self.savedPen = self.pen()
        self.setPen(QPen(QColor(0, 255, 0), 2))
        self.update()

    def hoverLeaveEvent(self, *args, **kwargs):
        self.setPen(self.savedPen)
        self.update()

class activeSpline(QGraphicsPathItem):
    """
    Base class for interactive splines.
    """
    strokeWidth = 2
    penWidth = 2
    brushColor = Qt.darkGray

    def __init__(self, size, fixedPoints=[], parentItem=None):
        """
        Inits a cubicSpline with an empty set of control points and
        an empty curve
        @param size: initial path size
        @type size: int
        @param parentItem:
        @type parentItem: object
        """
        super().__init__()
        self.setParentItem(parentItem)
        qpp = QPainterPath()
        self.size = size
        # initial curve : diagonal
        qpp.lineTo(QPoint(size, -size))
        # stroke curve
        stroker = QPainterPathStroker()
        stroker.setWidth(self.strokeWidth)
        self.mboundingPath = stroker.createStroke(qpp)
        self.setPath(self.mboundingPath)
        self.clicked = False
        #self.selected = False
        self.setVisible(False)
        self.fixedPoints = fixedPoints
        # self.spline is the list of QPointF instances to plot (scene coordinates)
        self.spline = []
        # self.LUTXY is the 1D LUT : range 0..255 --> 0..255, type ndarray, dtype=int, size=256
        self.LUTXY = np.arange(256)
        self.channel = channelValues.RGB
        self.histImg = None
        # set item pen
        self.setPen(QPen(QBrush(self.brushColor), self.penWidth))

    def initFixedPoints(self):
        axeSize=self.size
        rect = QRectF(0.0, -axeSize, axeSize, axeSize)
        self.fixedPoints = [activePoint(0, 0, persistent=True, rect=rect, parentItem=self),
                            activePoint(axeSize / 2, -axeSize / 2, rect=rect, parentItem=self),
                            activePoint(axeSize, -axeSize, persistent=True, rect=rect, parentItem=self)]

    def updateLUTXY(self):
        """
        Sync the LUT with the spline
        """
        scale = 255.0 / self.size
        self.LUTXY = np.array([int((-p.y()) * scale) for p in self.spline])

    def writeToStream(self, outStream):
        outStream.writeInt32(self.size)
        outStream.writeInt32(len(self.fixedPoints))
        for point in self.fixedPoints:
            outStream << point.scenePos()
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
            self.fixedPoints.append(activePoint(point.x(), point.y(), parentItem=self))
        self.updatePath()
        self.updateLUTXY()
        return self

class activeCubicSpline(activeSpline) :
    """
    Interactive cubic spline. Control points can be :
        - added by a mouse click on the curve,
        - moved with the mouse (cf. activePoint.mouseMoveEvent())
        - removed by a mouse click on the point (cf. activePoint.mouseReleaseEvent())
    """
    def updatePath(self):
        """
        Updates and displays the spline. Should be called after
        each control point or tangent modification : see
        activePoint and activeTangent mouse event handlers
        """
        # add ending control points, if needed,
        # to get full range 0..self.size
        X = [item.x() for item in self.fixedPoints]
        Y = [item.y() for item in self.fixedPoints]
        X0, X1 = X[0], X[-1]
        Y0, Y1 = Y[0], Y[-1]
        Y2 = Y0 - X0 * (Y1-Y0)/(X1-X0)
        Y3 = Y0 + (self.size - X0) * (Y1-Y0)/(X1-X0)
        if X[0] > 0.0:
            X.insert(0, 0.0)
            Y.insert(0, Y2)
        if X[-1] < self.size:
            X.append(self.size)
            Y.append(Y3)
        # interpolate
        try:
            # interpolationCubSpline raises an exception if two points have identical x-coordinates
            self.spline = interpolationCubSpline(np.array(X), np.array(Y), clippingInterval= [-self.scene().axeSize, 0])
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
        except:
            pass

    def mousePressEvent(self, e):
        self.clicked = True
        #self.selected= True

    def mouseMoveEvent(self, e):
        self.clicked = False
        #self.updatePath()
        #updateScene(self.scene())

    def mouseReleaseEvent(self, e):
        """
        if click, add a control point to the curve
        """
        #self.selected = False
        # click event
        if self.clicked:
            #add point
            p=e.pos()
            a=activePoint(p.x(), p.y(), parentItem=self)
            self.fixedPoints.append(a)
            self.fixedPoints.sort(key=lambda z : z.scenePos().x())
            self.updatePath()

    def getStackedLUTXY(self):
        """
        Returns the  LUT (A row LUT for each channel)
        @return: LUT
        @rtype: ndarray, shape (3,n), dtype=
        """
        if self.channel == channelValues.RGB:
            return np.vstack((self.LUTXY, self.LUTXY, self.LUTXY))
        else:
            return np.vstack((self.scene().cubicR.LUTXY, self.scene().cubicG.LUTXY, self.scene().cubicB.LUTXY))

    def reset(self):
        for point in self.childItems():
            self.scene().removeItem(point)
        self.initFixedPoints()
        #calculate spline
        self.updatePath()
        self.LUTXY = np.arange(256)

class activeQuadricSpline(activeSpline) :
    """
    Interactive quadratic spline. Control points can be :
        - added by a mouse click on the curve,
        - moved with the mouse (cf. activePoint.mouseMoveEvent())
        - removed by a mouse click on the point (cf. activePoint.mouseReleaseEvent())
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixedTangents = []

    def updatePath(self, calculate=True):
        """
        Calculates (if calculate=true) and displays the spline.
        Called on activePoint and activeTangent mouse event
        @param calculate:
        @type calculate: bool
        """
        # add boundary points if needed
        X = [item.x() for item in self.fixedPoints]
        Y = [item.y() for item in self.fixedPoints]
        X0, X1 = X[0], X[-1]
        Y0, Y1 = Y[0], Y[-1]
        Y2 = Y0 - X0 * (Y1-Y0)/(X1-X0)
        Y3 = Y0 + (self.size - X0) * (Y1-Y0)/(X1-X0)
        if X[0] > 0.0:
            X.insert(0, 0.0)
            Y.insert(0, Y2)
        if X[-1] < self.size:
            X.append(self.size)
            Y.append(Y3)
        #d = np.array([1.0] * len(X))
        # calculate the array of slopes
        d = [item.controlPoint - item.contactPoint for item in self.fixedTangents]
        d1 = - np.array(list(map(lambda a : a.y(), d)))
        d2 = np.array(list(map(lambda a : a.x(), d)))
        d = d1 / d2
        try:
            if calculate:
                T = interpolationQuadSpline(np.array(X)/self.size, -np.array(Y)/self.size, d) * self.size
                self.spline = [QPointF(x, y) for x, y in zip(np.arange(256) * (self.size / 255.0), -T)]
            for P in self.spline:
                if P.x()<X0:
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
        Initialises the spline and the LUT.
        a, b, d must have identical sizes
        @param a: x-ccoordinates of control points
        @type a:
        @param b: y-coordinates of control points
        @type b:
        @param d: tangent slopes
        @type d:
        @param T: spline array
        @type T: ndarray
        """
        self.a, self.b, self.d, self.T = a, b, d, T
        rect = QRectF(0.0, -self.size, self.size, self.size)
        # half tangent length and orientation
        alpha = [50.0] * len(d)
        # choose backward orientation for the last tangent
        alpha[-1] = - alpha[-1]
        for item in self.fixedPoints:
            self.scene().removeItem(item)
        for item in self.fixedTangents:
            self.scene().removeItem(item)
        self.fixedPoints = [activePoint(x, -y, rect=rect, parentItem=self) for x,y in zip(a,b)]
        # tangent normalization
        n = np.sqrt(1 + d * d)
        alpha = alpha / n
        self.fixedTangents = [activeTangent(controlPoint=QPointF(x+alpha[i], -y-alpha[i]*p), contactPoint=QPointF(x,-y), parentItem=self)
                              for i,(x,y,p) in enumerate(zip(a,b,d))]
        # link contact point to tangent
        for i1, i2 in zip(self.fixedPoints, self.fixedTangents):
            i1.tangent = i2
        self.spline = [QPointF(x,y) for x, y in zip(np.arange(256)*(self.size/255), -T)]
        self.updatePath(calculate=False) # don't recalculate the spline!
        self.updateLUTXY()

    def mousePressEvent(self, e):
        self.clicked = True
        #self.selected= True

    def mouseMoveEvent(self, e):
        self.clicked = False
        #self.updatePath()
        #updateScene(self.scene())

    def mouseReleaseEvent(self, e):
        """
        Adds a control point to the curve
        @param e:
        """
        #self.selected = False
        # click event
        if self.clicked:
            #add point
            p=e.pos()
            a=activePoint(p.x(), p.y(), parentItem=self)
            self.fixedPoints.append(a)
            self.fixedPoints.sort(key=lambda z : z.scenePos().x())
            #self.scene().addItem(a)
            self.updatePath()

    def getStackedLUTXY(self):
        """
        Returns the 3-channel LUT (A 1-line LUT for each channel)
        @return: LUT
        @rtype: ndarray, shape (3,n)
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
class graphicsCurveForm(QGraphicsView):
    """
    Base class for interactive curve forms
    """
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize + 60, axeSize + 140)
        self.setAttribute(Qt.WA_DeleteOnClose)
        graphicsScene = QGraphicsScene()
        self.setScene(graphicsScene)
        graphicsScene.targetImage = targetImage
        graphicsScene.layer = layer
        graphicsScene.bgColor = QColor(200,200,200)
        self.mainForm = mainForm
        graphicsScene.axeSize = axeSize
        # add axes and grid
        item = drawPlotGrid(axeSize)
        graphicsScene.addItem(item)
        # default WhatsThis for interactive curves
        self.setWhatsThis(
"""
Drag control points with the mouse
Add a control point by clicking on the curve
Remove a control point by clicking it
Zoom with the mouse wheel
"""                      )

    def wheelEvent(self, e):
        """
        Overrides QGraphicsView wheelEvent
        Zoom the scene
        @param e:
        @type e:
        """
        pos = e.pos()
        # delta unit is 1/8 of degree
        # Most mice have a resolution of 15 degrees
        numSteps = 1 + e.delta() / 1200.0
        self.scale(numSteps, numSteps)

class graphicsQuadricForm(graphicsCurveForm) :
    """
    Form for interactive quadratic splines.
    Dynamic attributes are added to the scene in order
    to provide links to arbitrary graphics items:
        self.scene().cubicItem : current active curve
        self.scene().targetImage
        self.scene().layer
        self.scene().bgColor
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        newWindow = graphicsQuadricForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        newWindow.setWindowTitle(layer.name)
        return newWindow
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        graphicsScene = self.scene()
        # curve
        quadric = activeQuadricSpline(graphicsScene.axeSize)
        graphicsScene.addItem(quadric)
        graphicsScene.quadricB = quadric
        quadric.channel = channelValues.Br
        quadric.histImg = graphicsScene.layer.histogram(size=graphicsScene.axeSize, bgColor=graphicsScene.bgColor,
                                                       range=(0,255), chans=channelValues.Br, mode='Luminosity')
        quadric.initFixedPoints()
        # set current curve
        graphicsScene.cubicItem = graphicsScene.quadricB
        graphicsScene.cubicItem.setVisible(True)

        def onResetCurve():
            """
            Reset the selected curve
            """
            self.scene().cubicItem.reset()
            #self.scene().onUpdateLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        # buttons
        pushButton1 = QPushButton("Reset to Auto Curve")
        pushButton1.setGeometry(10, 20, 100, 30)  # x,y,w,h
        pushButton1.adjustSize()
        pushButton1.clicked.connect(onResetCurve)
        graphicsScene.addWidget(pushButton1)

    def drawBackground(self, qp, qrF):
        graphicsScene = self.scene()
        s = graphicsScene.axeSize
        if graphicsScene.cubicItem.histImg is not None:
            qp.drawImage(QRect(0, -s, s, s), graphicsScene.cubicItem.histImg)

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