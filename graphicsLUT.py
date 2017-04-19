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
from utils import optionsWidget, Channel

strokeWidth = 3
controlPoints =[]
computeControlPoints = True

def buildLUT(curve):
    """
    Build the LUT from a list of QPOINTF objects, representing
    a curve. The LUT values are interpolated between consecutive curve points.
    x-coordinates of points are assumed to be sorted in ascending order.
    y-coordinates of points are flipped to reflect y-axis orientation.
    :param curve: list of QPOINTF objects
    :return: list of 256 integer values, between 0 and 255.
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
    def __init__(self, x,y, parentItem=None):
        super(activePoint, self).__init__()
        self.setParentItem(parentItem)
        self.setPos(QPointF(x,y))
        self.clicked = False
        self.moveStart=QPointF()
        self.setPen(QPen(QColor(255, 0, 0), 2))
        qpp = QPainterPath()
        qpp.addEllipse(0,0, 8, 8)
        self.setPath(qpp)

    def mousePressEvent(self, e):
        #self.moveStart = e.pos()
        self.clicked = True

    def mouseMoveEvent(self, e):
        self.clicked = False
        self.position_ = e.scenePos()
        self.setPos(e.scenePos())
        self.scene().cubicItem.updatePath()

    def mouseReleaseEvent(self, e):
        cubicItem = self.scene().cubicItem
        cubicItem.fixedPoints.sort(key=lambda p : p.scenePos().x())
        self.position_=e.scenePos()
        self.setPos(e.scenePos())
        sc = self.scene()
        # click event : remove point
        if self.clicked:
            cubicItem.fixedPoints.remove(self)
            sc.removeItem(self)
            return
        self.scene().cubicItem.updatePath()
        LUT = []
        scale = 255.0 / self.scene().axeSize
        LUT.extend([int((-p.y()) * scale) for p in self.scene().cubicItem.spline])
        cubicItem.LUTXY = np.array(LUT)
        self.scene().onUpdateLUT()
        img=self.scene().layer.inputImg()
        self.scene().cubicItem.histImg = img.histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, channel=cubicItem.channel)
        self.scene().invalidate(QRectF(0.0,-self.scene().axeSize, self.scene().axeSize, self.scene().axeSize), QGraphicsScene.BackgroundLayer)

class cubicItem(QGraphicsPathItem) :

    def __init__(self, size, parentItem=None):
        super(cubicItem, self).__init__()
        self.setParentItem(parentItem)
        self.qpp = QPainterPath()
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
        self.fixedPoints = []
        self.spline = []
        self.LUTXY = np.array(range(256))
        self.channel = Channel.RGB
        self.histImg = None

    def updatePath(self):
        qpp = QPainterPath()
        X = np.array([item.x() for item in self.fixedPoints])
        Y = np.array([item.y() for item in self.fixedPoints])
        # cubicSplineCurve raises an exception if two points have identical x-coordinates
        try:
            self.spline = cubicSplineCurve(X, Y, clippingInterval= [-self.scene().axeSize, 0])#self.scene().axeSize])

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
        if self.channel == Channel.RGB:
            return np.vstack((self.LUTXY, self.LUTXY, self.LUTXY))
        else:
            return np.vstack((self.scene().cubicR.LUTXY, self.scene().cubicG.LUTXY, self.scene().cubicB.LUTXY))

    def reset(self):
        self.clicked = QPoint(0, 0)
        self.selected = False
        for point in self.childItems():
            self.scene().removeItem(point)
        self.fixedPoints = [activePoint(0, 0, parentItem=self),
                             activePoint(self.scene().axeSize / 2,
                                         -self.scene().axeSize / 2, parentItem=self),
                             activePoint(self.scene().axeSize, -self.scene().axeSize,
                                         parentItem=self)]
        self.updatePath()
        LUT = []
        scale = 255.0 / self.scene().axeSize
        LUT.extend([int((-p.y()) * scale) for p in self.spline])
        self.LUTXY = np.array(LUT)  # buildLUT(LUT)

class graphicsForm(QGraphicsView) :

    @classmethod
    def getNewWindow(cls, cModel, targetImage=None, size=500, layer=None, parent=None):
        newWindow = graphicsForm(cModel, targetImage=targetImage, size=size, layer=layer, parent=parent)
        newWindow.setWindowTitle(layer.name)
        return newWindow

    def __init__(self, cModel, targetImage=None, size=500, layer=None, parent=None):
        super(graphicsForm, self).__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(size + 80, size + 200)
        #self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        #self.setBackgroundBrush(QBrush(Qt.black, Qt.SolidPattern))
        #self.bgPixmap = QPixmap.fromImage(hueSatModel.colorWheel(size, size, cModel))
        self.graphicsScene = QGraphicsScene()
        self.setScene(self.graphicsScene)
        self.scene().targetImage = targetImage
        self.scene().layer = layer
        self.scene().bgColor = QColor(200,200,200)#self.palette().color(self.backgroundRole()) TODO parametrize

        #self.LUTXY = LUTXY


        self.graphicsScene.onUpdateScene = lambda : 0

        self.graphicsScene.axeSize = size

        """
        self.graphicsScene.sampleSize = 400
        self.graphicsScene.tSample = [float(i) / self.graphicsScene.sampleSize for i in range(self.graphicsScene.sampleSize + 1)]
        self.graphicsScene.tSample1 = np.array([(1 - t) ** 2 for t in self.graphicsScene.tSample])
        self.graphicsScene.tSample2 = np.array([2 * t * (1 - t) for t in self.graphicsScene.tSample])
        self.graphicsScene.tSample3 = np.array([t ** 2 for t in self.graphicsScene.tSample])
        """
        # draw axes
        item=QGraphicsPathItem()
        item.setPen(QPen(QBrush(QColor(255, 0, 0)), 1, style=Qt.DashLine))
        qppath = QPainterPath()
        qppath.moveTo(QPoint(0, 0))
        qppath.lineTo(QPoint(self.graphicsScene.axeSize, 0))
        qppath.lineTo(QPoint(self.graphicsScene.axeSize, -self.graphicsScene.axeSize))
        qppath.lineTo(QPoint(0, -self.graphicsScene.axeSize))
        qppath.closeSubpath()
        qppath.lineTo(QPoint(self.graphicsScene.axeSize, -self.graphicsScene.axeSize))

        # axes
        item.setPath(qppath)
        self.graphicsScene.addItem(item)

        #self.graphicsScene.addPath(qppath, QPen(Qt.DashLine))  #create and add QGraphicsPathItem

        # curves
        cubic = cubicItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicRGB = cubic
        cubic.channel = Channel.RGB
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, channel=Channel.RGB)
        cubic.fixedPoints = [activePoint(0, 0, parentItem=cubic),
                                          activePoint(self.graphicsScene.axeSize / 2, -self.graphicsScene.axeSize / 2, parentItem=cubic),
                                          activePoint(self.graphicsScene.axeSize, -self.graphicsScene.axeSize, parentItem=cubic)]
        cubic = cubicItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicR = cubic
        cubic.channel = Channel.Red
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, channel=Channel.Red)
        cubic.fixedPoints = [activePoint(0, 0, parentItem=cubic),
                                             activePoint(self.graphicsScene.axeSize / 2,
                                                         -self.graphicsScene.axeSize / 2, parentItem=cubic),
                                             activePoint(self.graphicsScene.axeSize, -self.graphicsScene.axeSize,
                                                         parentItem=cubic)]
        cubic = cubicItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicG = cubic
        cubic.channel = Channel.Green
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, channel=Channel.Green)
        cubic.fixedPoints = [activePoint(0, 0, parentItem=cubic),
                                             activePoint(self.graphicsScene.axeSize / 2,
                                                         -self.graphicsScene.axeSize / 2, parentItem=cubic),
                                             activePoint(self.graphicsScene.axeSize, -self.graphicsScene.axeSize,
                                                         parentItem=cubic)]
        cubic = cubicItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicB = cubic
        cubic.channel = Channel.Blue
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, channel=Channel.Blue)
        cubic.fixedPoints = [activePoint(0, 0, parentItem=cubic),
                                             activePoint(self.graphicsScene.axeSize / 2,
                                                         -self.graphicsScene.axeSize / 2, parentItem=cubic),
                                             activePoint(self.graphicsScene.axeSize, -self.graphicsScene.axeSize,
                                                         parentItem=cubic)]
        # set current
        self.scene().cubicItem = self.graphicsScene.cubicRGB
        self.scene().cubicItem.setVisible(True)

        def onResetCurve():
            """
            Reset the selected curve
            """
            self.scene().cubicItem.reset()
            self.scene().onUpdateLUT()

        pushButton = QPushButton("Reset Curve")
        pushButton.setObjectName("btn_reset_channel")
        pushButton.setMinimumSize(1, 1)
        pushButton.setGeometry(140, 50, 80, 30)  # x,y,w,h
        pushButton.clicked.connect(onResetCurve)
        self.graphicsScene.addWidget(pushButton)

        # options
        self.listWidget1 = optionsWidget(options=['RGB', 'Red', 'Green', 'Blue'], exclusive=True)
        self.listWidget1.select(self.listWidget1.items['RGB'])
        self.listWidget1.setGeometry(50, 50, 10, 100)
        self.graphicsScene.addWidget(self.listWidget1)
        #self.listWidget1.setStyleSheet("QListWidget{background: white;} QListWidget::item{color: black;}")

        def onSelect1(item):
            self.scene().cubicItem.setVisible(False)
            if item.mySelectedAttr:
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



