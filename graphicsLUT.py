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
from PySide2.QtGui import QPainterPathStroker
from PySide2.QtCore import QRect, QPointF, QPoint
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QSizePolicy, QPushButton, QGraphicsPathItem
from PySide2.QtGui import QColor, QPen, QPainterPath, QPolygonF
from PySide2.QtCore import Qt, QRectF

from spline import interpolationCubSpline, interpolationQuadSpline
from utils import optionsWidget, channelValues, drawPlotGrid

def buildLUT(curve):  #unused
    """
    Build the LUT from a list of QPOINTF objects, representing
    a curve. The LUT values are interpolated between consecutive curve points.
    x-coordinates of points are assumed to be sorted in ascending order.
    y-coordinates of points are flipped to reflect y-axis orientation.
    @param curve: list of QPointF
    @return: list of 256 integer values, in range 0..255.
    """
    # add sentinels
    S1 = QPointF(-1, curve[0].y())
    S2 = QPointF(256, curve[-1].y())
    curve = [S1] + curve + [S2]

    LUTX = [p.x() for p in curve]
    LUTY = [p.y() for p in curve]

    #build LUTXY table
    LUTXY = -np.interp(range(256), LUTX, LUTY)
    LUTXY = np.around(LUTXY).astype(int)
    LUTXY = np.clip(LUTXY, 0, 255)
    return LUTXY

class activePoint(QGraphicsPathItem):
    """
    Interactive point
    """
    def __init__(self, x,y, persistent=False, rect=None, parentItem=None):
        """
        Interactive point. Persistent activePoints cannot be removed
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
        self.setParentItem(parentItem)
        self.persistent = persistent
        self.rect = rect
        if self.rect is not None:
            self.xmin, self.xmax, self.ymin, self.ymax = rect.left(), rect.right(), rect.top(), rect.bottom()
            x = min(max(x, self.xmin), self.xmax)
            y = min(max(y, self.ymin), self.ymax)
        self.setPos(QPointF(x,y))
        self.clicked = False
        self.moveStart=QPointF()
        self.setPen(QPen(QColor(255, 255, 255), 2))
        qpp = QPainterPath()
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
        self.scene().cubicItem.updatePath()

    def mouseReleaseEvent(self, e):
        cubicItem = self.scene().cubicItem
        cubicItem.fixedPoints.sort(key=lambda p : p.scenePos().x())
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
            cubicItem.fixedPoints.remove(self)
            sc.removeItem(self)
            return
        self.scene().cubicItem.updatePath()
        self.scene().cubicItem.updateLUTXY()
        # The curve change event handler is
        # defined in blue.py : Apply current LUT to stack and repaint window
        #self.scene().onUpdateLUT()
        l = self.scene().layer
        l.applyToStack()
        l.parentImage.onImageChanged()

class activeTangent(QGraphicsPathItem):
    """
    Interactive tangent
    """
    def __init__(self, controlPoint=QPointF(), contactPoint=QPointF(), parentItem=None):
        super().__init__()
        self.setParentItem(parentItem)
        self.controlPoint = controlPoint
        self.contactPoint = contactPoint
        self.updatePath()

    def updatePath(self):
        qpp = QPainterPath()
        #qpp.addEllipse(self.controlPoint, 5.0, 5.0)
        qpp.moveTo(self.controlPoint)
        qpp.lineTo(self.contactPoint)
        self.setPath(qpp)

    def mousePressEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        self.controlPoint = e.pos()
        # updateScene()

    def mouseReleaseEvent(self, e):
        if e.lastPos() == e.pos():
            print('tangent click')

class cubicItem(QGraphicsPathItem) :
    """
    Interactive cubic spline. Control points
    can be moved, added and removed with the mouse,
    while updating the corresponding 1D-LUT.
    """
    strokeWidth = 3

    def __init__(self, size, fixedPoints=[], parentItem=None):
        """
        Inits a cubicItem with an empty set of control points and
        an empty curve
        @param size: initial path size
        @type size: int
        @param parentItem:
        @type parentItem: object
        """
        super(cubicItem, self).__init__()
        self.setParentItem(parentItem)
        self.qpp = QPainterPath()
        self.size = size
        # initial curve : diagonal
        self.qpp.lineTo(QPoint(size, -size))
        # stroke curve
        stroker=QPainterPathStroker()
        stroker.setWidth(self.strokeWidth)
        self.mboundingPath = stroker.createStroke(self.qpp)
        self.setPath(self.mboundingPath)
        self.clicked=QPoint(0,0)
        self.selected = False
        self.setVisible(False)
        self.fixedPoints = fixedPoints
        # curve
        self.spline = []
        # 1D LUT : identity
        self.LUTXY = np.arange(256)
        self.channel = channelValues.RGB
        self.histImg = None

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
        LUT = []
        LUT.extend([int((-p.y()) * scale) for p in self.spline])
        self.LUTXY = np.array(LUT)

    def updatePath(self):
        """
        Calculates and displays the spline.
        """
        qpp = QPainterPath()
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
        # interpolationCubSpline raises an exception if two points have identical x-coordinates
        try:
            self.spline = interpolationCubSpline(np.array(X), np.array(Y), clippingInterval= [-self.scene().axeSize, 0])
            for P in self.spline:
                if P.x() < X0:
                    P.setY(Y0)
                elif P.x() > X1:
                    P.setY(Y1)
            polygon = QPolygonF(self.spline)
            qpp.addPolygon(polygon)
            # stroke path
            stroker = QPainterPathStroker()
            stroker.setWidth(3)
            mboundingPath = stroker.createStroke(qpp)
            self.setPath(mboundingPath)
        except:
            pass

    def mousePressEvent(self, e):
        self.beginMouseMove = e.pos()
        self.selected= True

    def mouseMoveEvent(self, e):
        pass
        #self.updatePath()
        #updateScene(self.scene())

    def mouseReleaseEvent(self, e):
        """
        Add a control point to the curve
        @param e:
        """
        self.selected = False
        # click event
        if self.beginMouseMove == e.pos():
            #add point
            p=e.pos()
            a=activePoint(p.x(), p.y(), parentItem=self)
            self.fixedPoints.append(a)
            self.fixedPoints.sort(key=lambda z : z.scenePos().x())
            self.updatePath()

    def getStackedLUTXY(self):
        """
        Returns the 3-channel LUT (A 1-row LUT for each channel)
        @return: LUT
        @rtype: ndarray, shape (3,n)
        """
        if self.channel == channelValues.RGB:
            return np.vstack((self.LUTXY, self.LUTXY, self.LUTXY))
        else:
            return np.vstack((self.scene().cubicR.LUTXY, self.scene().cubicG.LUTXY, self.scene().cubicB.LUTXY))

    def reset(self):
        self.clicked = QPoint(0, 0)
        self.selected = False
        for point in self.childItems():
            self.scene().removeItem(point)
        self.initFixedPoints()
        #calculate spline
        self.updatePath()
        LUT = range(256)
        self.LUTXY = np.array(LUT)  # buildLUT(LUT)

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
        #cubic = cubicItem(size, fixedPoints=fixedPoints)
        #self.fixedPoints = fixedPoints
        self.updatePath()
        self.updateLUTXY()
        return self

class graphicsCurveForm(QGraphicsView):
    """
    Base class for all interactive curve forms
    """
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize + 60, axeSize + 140)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.graphicsScene = QGraphicsScene()
        self.setScene(self.graphicsScene)
        self.scene().targetImage = targetImage
        self.scene().layer = layer
        self.scene().bgColor = QColor(200,200,200)
        self.mainForm = mainForm
        self.graphicsScene.axeSize = axeSize
        # add axes and grid to the scene
        item = drawPlotGrid(axeSize)
        self.graphicsScene.addItem(item)

class graphicsCubicForm(graphicsCurveForm) :
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        newWindow = graphicsCubicForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        newWindow.setWindowTitle(layer.name)
        return newWindow

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        # curve
        cubic = cubicItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicB = cubic
        cubic.channel = channelValues.Br
        cubic.histImg = self.scene().layer.histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, range=(0,1), chans=channelValues.Br, mode='Luminosity')
        cubic.initFixedPoints()
        # set current curve
        self.scene().cubicItem = self.graphicsScene.cubicB
        self.scene().cubicItem.setVisible(True)

        def onResetCurve():
            """
            Reset the selected curve
            """
            self.scene().cubicItem.reset()
            #self.scene().onUpdateLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        def onResetAllCurves():
            """
            Reset all curves
            """
            for cubicItem in [self.graphicsScene.cubicR, self.graphicsScene.cubicG, self.graphicsScene.cubicB]:
                cubicItem.reset()
            # call Curve change event handlerdefined in blue.menuLayer
            #self.scene().onUpdateLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        # buttons
        pushButton1 = QPushButton("Reset Curve")
        #pushButton1.setObjectName("btn_reset_channel")
        pushButton1.setMinimumSize(1, 1)
        pushButton1.setGeometry(80, 20, 100, 30)  # x,y,w,h
        pushButton1.adjustSize()
        pushButton1.clicked.connect(onResetCurve)
        self.graphicsScene.addWidget(pushButton1)
        pushButton2 = QPushButton("Reset All Curves")
        pushButton2.setMinimumSize(1, 1)
        pushButton2.setGeometry(80, 50, 100, 30)  # x,y,w,h
        pushButton2.adjustSize()
        pushButton2.clicked.connect(onResetAllCurves)
        self.graphicsScene.addWidget(pushButton2)

        # options
        options = ['H', 'S', 'B']
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        self.listWidget1.setGeometry(0, 10, self.listWidget1.sizeHintForColumn(0) + 5, self.listWidget1.sizeHintForRow(0) * len(options) + 5)
        self.graphicsScene.addWidget(self.listWidget1)

        # self.options is for convenience only
        self.options = {option: True for option in options}

        def onSelect1(item):
            self.scene().cubicItem.setVisible(False)
            if item.text() == 'H':
                self.scene().cubicItem = self.graphicsScene.cubicR
            elif item.text() == 'S':
                self.scene().cubicItem = self.graphicsScene.cubicG
            elif item.text() == 'B':
                self.scene().cubicItem = self.graphicsScene.cubicB

            self.scene().cubicItem.setVisible(True)

            # draw  histogram
            self.scene().invalidate(QRectF(0.0, -self.scene().axeSize, self.scene().axeSize, self.scene().axeSize), QGraphicsScene.BackgroundLayer)

        self.listWidget1.onSelect = onSelect1

        # set initial selection to Saturation
        item = self.listWidget1.items[options[1]]
        item.setCheckState(Qt.Checked)
        self.listWidget1.select(item)

    def drawBackground(self, qp, qrF):
        s = self.graphicsScene.axeSize
        if self.scene().cubicItem.histImg is not None:
            qp.drawImage(QRect(0, -s, s, s), self.scene().cubicItem.histImg)

    def writeToStream(self, outStream):
        layer = self.scene().layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        if layer.actionName in ['actionBrightness_Contrast', 'actionCurves_HSpB', 'actionCurves_Lab']:
            outStream.writeQString(self.listWidget1.selectedItems()[0].text())
            self.graphicsScene.cubicR.writeToStream(outStream)
            self.graphicsScene.cubicG.writeToStream(outStream)
            self.graphicsScene.cubicB.writeToStream(outStream)
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
        self.graphicsScene.cubicR.readFromStream(inStream)
        self.graphicsScene.cubicG.readFromStream(inStream)
        self.graphicsScene.cubicB.readFromStream(inStream)
        return inStream

class QuadricItem(QGraphicsPathItem) :
    """
    Interactive quadratic spline.
    """
    strokeWidth = 3

    def __init__(self, size, fixedPoints=[], parentItem=None):
        """
        Builds a spline with an empty set of fixed points
        @param size: initial path size
        @type size: int
        @param parentItem:
        @type parentItem: object
        """
        super().__init__()
        self.setParentItem(parentItem)
        self.qpp = QPainterPath()
        self.size = size
        # initial curve : diagonal
        self.qpp.lineTo(QPoint(size, -size))
        # stroke curve
        stroker=QPainterPathStroker()
        stroker.setWidth(self.strokeWidth)
        self.mboundingPath = stroker.createStroke(self.qpp)
        self.setPath(self.mboundingPath)
        #self.clicked=QPoint(0,0)
        #self.selected = False
        self.setVisible(False)
        self.fixedPoints = fixedPoints
        # curve
        self.spline = []
        # 1D-LUT : identity
        self.LUTXY = np.array(range(256))
        self.channel = channelValues.RGB
        self.histImg = None

    def initFixedPoints(self):
        axeSize = self.size
        rect = QRectF(0.0, -axeSize, axeSize, axeSize)
        self.fixedPoints = [activePoint(0, 0, persistent=True, rect=rect, parentItem=self),
                            activePoint(axeSize / 2, -axeSize / 2, rect=rect, parentItem=self),
                            activePoint(axeSize, -axeSize, persistent=True, rect=rect, parentItem=self)]
        #self.fixedTangents = [activeTangent(cp.pos(), -cp.pos(), parentItem=self) for cp in self.fixedPoints]

    def updateLUTXY(self):
        """
        Sync the LUT with the spline
        """
        scale = 255.0 / self.size
        LUT = []
        LUT.extend([int((-p.y()) * scale) for p in self.spline])
        self.LUTXY = np.array(LUT)

    def updatePath(self, calculate=True):
        """
        Calculates and displays the spline.
        """
        qpp = QPainterPath()
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
        d = np.array([1.0] * len(X))
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
            qpp.addPolygon(polygon)
            # stroke path
            stroker = QPainterPathStroker()
            stroker.setWidth(self.strokeWidth)
            mboundingPath = stroker.createStroke(qpp)
            self.setPath(mboundingPath)
        except Exception as e:
            print(str(e))

    def setCurve(self, a, b, d, T):
        self.a, self.b, self.d, self.T = a, b, d, T
        rect = QRectF(0.0, -self.size, self.size, self.size)
        alpha=10.0
        self.fixedPoints = [activePoint(x, -y, rect=rect, parentItem=self) for x,y in zip(a,b)]
        self.fixedtangents = [activeTangent(controlPoint=QPointF(x+alpha, -y-alpha*p), contactPoint=QPointF(x,-y), parentItem=self) for x,y,p in zip(a,b,d)]
        self.spline = [QPointF(x,y) for x, y in zip(np.arange(256)*(self.size/255), -T)]
        self.updatePath(calculate=False)
        self.updateLUTXY()

    def mousePressEvent(self, e):
        self.beginMouseMove = e.pos()
        #self.selected= True

    def mouseMoveEvent(self, e):
        pass
        #self.updatePath()
        #updateScene(self.scene())

    def mouseReleaseEvent(self, e):
        """
        Adds a control point to the curve
        @param e:
        """
        #self.selected = False
        # click event
        if self.beginMouseMove == e.pos():
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

class graphicsQuadricForm(graphicsCurveForm) :
    """
    Form for interactive quadratic splines
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        newWindow = graphicsQuadricForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        newWindow.setWindowTitle(layer.name)
        return newWindow

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        # curve
        quadric = QuadricItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(quadric)
        self.graphicsScene.quadricB = quadric
        quadric.channel = channelValues.Br
        quadric.histImg = self.scene().layer.histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, range=(0,1), chans=channelValues.Br, mode='Luminosity')
        quadric.initFixedPoints()
        # set current
        self.scene().cubicItem = self.graphicsScene.quadricB
        self.scene().cubicItem.setVisible(True)

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
        self.graphicsScene.addWidget(pushButton1)

    def drawBackground(self, qp, qrF):
        s = self.graphicsScene.axeSize
        if self.scene().cubicItem.histImg is not None:
            qp.drawImage(QRect(0, -s, s, s), self.scene().cubicItem.histImg)

    def writeToStream(self, outStream):
        layer = self.scene().layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        if layer.actionName in ['actionBrightness_Contrast', 'actionCurves_HSpB', 'actionCurves_Lab']:
            outStream.writeQString(self.listWidget1.selectedItems()[0].text())
            self.graphicsScene.quadricR.writeToStream(outStream)
            self.graphicsScene.quadricG.writeToStream(outStream)
            self.graphicsScene.quadricB.writeToStream(outStream)
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
        self.graphicsScene.quadricR.readFromStream(inStream)
        self.graphicsScene.quadricG.readFromStream(inStream)
        self.graphicsScene.quadricB.readFromStream(inStream)
        return inStream