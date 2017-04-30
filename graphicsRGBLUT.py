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

import sys

from PySide.QtCore import QRect
from PySide.QtGui import QApplication, QPainter, QWidget
from PySide.QtGui import QGraphicsView, QGraphicsScene, QGraphicsPathItem , QPainterPath, QPainterPathStroker, QPen, QBrush, QColor, QPixmap, QMainWindow, QLabel, QSizePolicy
from PySide.QtCore import Qt, QPoint, QPointF, QRectF
import numpy as np
from PySide.QtGui import QPolygonF
from PySide.QtGui import QPushButton

from colorModels import hueSatModel
from spline import cubicSplineCurve
from utils import optionsWidget, channelValues, drawPlotGrid

strokeWidth = 3
controlPoints =[]
computeControlPoints = True

def buildLUT(curve):  #unused
    """
    Build the LUT from a list of QPOINTF objects, representing
    a curve. The LUT values are interpolated between consecutive curve points.
    x-coordinates of points are assumed to be sorted in ascending order.
    y-coordinates of points are flipped to reflect y-axis orientation.
    @param curve: list of QPOINTF objects
    @return: list of 256 integer values, between 0 and 255.
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
        by mouse click (default is non persistent). 
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
        """
        LUT = []
        scale = 255.0 / self.scene().axeSize
        LUT.extend([int((-p.y()) * scale) for p in self.scene().cubicItem.spline])
        cubicItem.LUTXY = np.array(LUT)
        """

        self.scene().onUpdateLUT()

class cubicItem(QGraphicsPathItem) :
    """
    Interactive cubic spline.
    """

    def __init__(self, size, fixedPoints=[], parentItem=None):
        """
        Builds a spline with an empty set of fixed points
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
        stroker.setWidth(strokeWidth)
        self.mboundingPath = stroker.createStroke(self.qpp)
        self.setPath(self.mboundingPath)
        self.clicked=QPoint(0,0)
        self.selected = False
        self.setVisible(False)
        self.fixedPoints = fixedPoints
        self.spline = []
        self.LUTXY = np.array(range(256))
        self.channel = channelValues.RGB
        self.histImg = None

    def initFixedPoints(self):
        axeSize=self.size
        self.fixedPoints = [activePoint(0, 0, persistent=True, rect=QRectF(0.0, -axeSize, axeSize, axeSize), parentItem=self),
                            activePoint(axeSize / 2, -axeSize / 2, rect=QRectF(0.0, -axeSize, axeSize, axeSize), parentItem=self),
                            activePoint(axeSize, -axeSize, persistent=True, rect=QRectF(0.0, -axeSize, axeSize, axeSize), parentItem=self)]

    def updateLUTXY(self):
        scale = 255.0 / self.size
        LUT = []
        LUT.extend([int((-p.y()) * scale) for p in self.spline])
        self.LUTXY = np.array(LUT)

    def updatePath(self):
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
        # cubicSplineCurve raises an exception if two points have identical x-coordinates
        try:
            self.spline = cubicSplineCurve(np.array(X), np.array(Y), clippingInterval= [-self.scene().axeSize, 0])

            for P in self.spline:
                if P.x()<X0:
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
        self.selected = False
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
        if self.channel == channelValues.RGB:
            return np.vstack((self.LUTXY, self.LUTXY, self.LUTXY))
        else:
            return np.vstack((self.scene().cubicR.LUTXY, self.scene().cubicG.LUTXY, self.scene().cubicB.LUTXY))

    def reset(self):
        self.clicked = QPoint(0, 0)
        self.selected = False
        for point in self.childItems():
            self.scene().removeItem(point)
        """
        self.fixedPoints = [activePoint(0, 0, parentItem=self),
                             activePoint(self.scene().axeSize / 2,
                                         -self.scene().axeSize / 2, parentItem=self),
                             activePoint(self.scene().axeSize, -self.scene().axeSize,
                                         parentItem=self)]
        """
        self.initFixedPoints()
        #calculate spline
        self.updatePath()
        LUT = range(256)
        #scale = 255.0 / self.scene().axeSize
        #LUT.extend([int((-p.y()) * scale) for p in self.spline])
        self.LUTXY = np.array(LUT)  # buildLUT(LUT)

    def writeToStream(self, outStream):
        outStream.writeInt32(self.size)
        outStream.writeInt32(len(self.fixedPoints))
        for point in self.fixedPoints:
            outStream << point.scenePos()
        return outStream

    @classmethod
    def readFromStream(cls, inStream):
        size = inStream.readInt32()
        count = inStream.readInt32()
        fixedPoints = []
        for i in range(count):
            point = QPointF()
            inStream >> point
            fixedPoints.append(point)
        cubic = cubicItem(size, fixedPoints=fixedPoints)
        cubic.updatePath()
        cubic.updateLUTXY()
        return cubic


class graphicsForm(QGraphicsView) :

    @classmethod
    def getNewWindow(cls, cModel=None, targetImage=None, axeSize=500, layer=None, parent=None):
        newWindow = graphicsForm(cModel=cModel, targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        newWindow.setWindowTitle(layer.name)
        return newWindow

    def __init__(self, cModel=None, targetImage=None, axeSize=500, layer=None, parent=None):
        super(graphicsForm, self).__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize + 60, axeSize + 140)
        #self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        #self.setBackgroundBrush(QBrush(Qt.black, Qt.SolidPattern))
        #self.bgPixmap = QPixmap.fromImage(hueSatModel.colorWheel(size, size, cModel))
        self.graphicsScene = QGraphicsScene()
        self.setScene(self.graphicsScene)
        self.scene().targetImage = targetImage
        self.scene().layer = layer
        self.scene().bgColor = QColor(200,200,200)#self.palette().color(self.backgroundRole()) TODO parametrize

        self.graphicsScene.onUpdateScene = lambda : 0

        self.graphicsScene.axeSize = axeSize

        # axes and grid
        item = drawPlotGrid(axeSize)
        self.graphicsScene.addItem(item)

        self.graphicsScene.addItem(item)

        # curves
        cubic = cubicItem(axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicRGB = cubic
        cubic.channel = channelValues.RGB
        cubic.histImg = self.scene().layer.inputImgFull().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, chans=channelValues.RGB)
        cubic.initFixedPoints()
        cubic = cubicItem(axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicR = cubic
        cubic.channel = channelValues.Red
        cubic.histImg = self.scene().layer.inputImgFull().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, chans=channelValues.Red)
        cubic.initFixedPoints()
        cubic = cubicItem(axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicG = cubic
        cubic.channel = channelValues.Green
        cubic.histImg = self.scene().layer.inputImgFull().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, chans=channelValues.Green)
        cubic.initFixedPoints()
        cubic = cubicItem(axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicB = cubic
        cubic.channel = channelValues.Blue
        cubic.histImg = self.scene().layer.inputImgFull().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, chans=channelValues.Blue)
        cubic.initFixedPoints()
        # set current
        self.scene().cubicItem = self.graphicsScene.cubicRGB
        self.scene().cubicItem.setVisible(True)

        def onResetCurve():
            """
            Reset the selected curve
            """
            self.scene().cubicItem.reset()
            self.scene().onUpdateLUT()

        def updateStack():
            layer.applyToStack()
            targetImage.onImageChanged()

        # buttons
        pushButton1 = QPushButton("Reset Curve")
        pushButton1.setMinimumSize(1, 1)
        pushButton1.setGeometry(100, 20, 80, 30)  # x,y,w,h
        pushButton1.adjustSize()
        pushButton1.clicked.connect(onResetCurve)
        self.graphicsScene.addWidget(pushButton1)
        pushButton3 = QPushButton("Update Top Layers")
        pushButton3.setMinimumSize(1, 1)
        pushButton3.setGeometry(100, 80, 80, 30)  # x,y,w,h
        pushButton3.adjustSize()
        pushButton3.clicked.connect(updateStack)
        self.graphicsScene.addWidget(pushButton3)

        # options
        self.listWidget1 = optionsWidget(options=['RGB', 'Red', 'Green', 'Blue'], exclusive=True)
        self.listWidget1.select(self.listWidget1.items['RGB'])
        self.listWidget1.setGeometry(20, 20, 10, 100)
        self.graphicsScene.addWidget(self.listWidget1)
        #self.listWidget1.setStyleSheet("QListWidget{background: white;} QListWidget::item{color: black;}")

        def onSelect1(item):
            self.scene().cubicItem.setVisible(False)
            if item.isSelected():
                if item.text() == 'RGB' :
                    self.scene().cubicItem = self.graphicsScene.cubicRGB
                elif item.text() == 'Red':
                        self.scene().cubicItem = self.graphicsScene.cubicR
                elif item.text() == 'Green':
                        self.scene().cubicItem = self.graphicsScene.cubicG
                elif item.text() == 'Blue':
                        self.scene().cubicItem = self.graphicsScene.cubicB

                self.scene().cubicItem.setVisible(True)
                self.scene().onUpdateLUT()

                # draw  histogram
                self.scene().invalidate(QRectF(0.0, -self.scene().axeSize, self.scene().axeSize, self.scene().axeSize),
                                        QGraphicsScene.BackgroundLayer)

        self.listWidget1.onSelect = onSelect1

    def drawBackground(self, qp, qrF):
        s = self.graphicsScene.axeSize
        if self.scene().cubicItem.histImg is not None:
            qp.drawPixmap(QRect(0,-s, s, s), QPixmap.fromImage(self.scene().cubicItem.histImg))





