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
from graphicsRGBLUT import cubicItem, activePoint
from spline import cubicSplineCurve
from utils import optionsWidget, Channel

strokeWidth = 3
controlPoints =[]
computeControlPoints = True


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
        cubic.histImg = self.scene().layer.inputImgFull().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, channel=Channel.RGB)
        cubic.fixedPoints = [activePoint(0, 0, parentItem=cubic),
                                          activePoint(self.graphicsScene.axeSize / 2, -self.graphicsScene.axeSize / 2, parentItem=cubic),
                                          activePoint(self.graphicsScene.axeSize, -self.graphicsScene.axeSize, parentItem=cubic)]
        cubic = cubicItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicR = cubic
        cubic.channel = Channel.Red
        cubic.histImg = self.scene().layer.inputImgFull().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, channel=Channel.Red)
        cubic.fixedPoints = [activePoint(0, 0, parentItem=cubic),
                                             activePoint(self.graphicsScene.axeSize / 2,
                                                         -self.graphicsScene.axeSize / 2, parentItem=cubic),
                                             activePoint(self.graphicsScene.axeSize, -self.graphicsScene.axeSize,
                                                         parentItem=cubic)]
        cubic = cubicItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicG = cubic
        cubic.channel = Channel.Green
        cubic.histImg = self.scene().layer.inputImgFull().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, channel=Channel.Green)
        cubic.fixedPoints = [activePoint(0, 0, parentItem=cubic),
                                             activePoint(self.graphicsScene.axeSize / 2,
                                                         -self.graphicsScene.axeSize / 2, parentItem=cubic),
                                             activePoint(self.graphicsScene.axeSize, -self.graphicsScene.axeSize,
                                                         parentItem=cubic)]
        cubic = cubicItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicB = cubic
        cubic.channel = Channel.Blue
        cubic.histImg = self.scene().layer.inputImgFull().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, channel=Channel.Blue)
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
        self.listWidget1 = optionsWidget(options=['Hue', 'Sat', 'Brightness'], exclusive=True)
        self.listWidget1.select(self.listWidget1.items['Brightness'])
        self.listWidget1.setGeometry(50, 50, 10, 100)
        self.graphicsScene.addWidget(self.listWidget1)
        #self.listWidget1.setStyleSheet("QListWidget{background: white;} QListWidget::item{color: black;}")

        def onSelect1(item):
            self.scene().cubicItem.setVisible(False)
            if item.mySelectedAttr:
                if item.text() == 'RGB' :
                    self.scene().cubicItem = self.graphicsScene.cubicRGB
                elif item.text() == 'Hue':
                        self.scene().cubicItem = self.graphicsScene.cubicR
                elif item.text() == 'Sat':
                        self.scene().cubicItem = self.graphicsScene.cubicG
                elif item.text() == 'Brightness':
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

