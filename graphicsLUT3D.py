import sys
from PyQt4.QtGui import QApplication, QPainter, QWidget
from PyQt4.QtGui import QGraphicsView, QGraphicsScene, QGraphicsPathItem , QGraphicsPixmapItem, QPainterPath, QPainterPathStroker, QPen, QBrush, QColor, QPixmap, QMainWindow, QLabel, QSizePolicy
from PyQt4.QtCore import Qt, QPoint, QPointF, QRect, QRectF
import numpy as np
from time import time
from LUT3D import LUT3D
from colorModels import hueSatModel, pbModel


#main class
class graphicsForm3DLUT(QGraphicsView) :

    @classmethod
    def getNewWindow(cls, *args, **kwargs):
        newwindow = graphicsForm3DLUT(*args, **kwargs)
        newwindow.setAttribute(Qt.WA_DeleteOnClose)
        newwindow.setWindowTitle('New Window')
        return newwindow

    def __init__(self, *args, **kwargs):
        super(graphicsForm3DLUT, self).__init__(*args, **kwargs)
        self.bgPixmap = QPixmap.fromImage(hueSatModel.colorPicker(500,500))
        self.graphicsScene = QGraphicsScene()
        self.setScene(self.graphicsScene)
        #self.LUTXY = LUTXY
        #self.graphicsScene.LUTXY=np.array(range(256))
        #label=self.graphicsScene.addWidget(QLabel())
        #label.setPos(50, 20)
        item1 = QGraphicsPixmapItem(self.bgPixmap);
        self.graphicsScene.addItem(item1)
        px = QPixmap.fromImage(pbModel.colorPicker(500,50, 180, 0.5))
        item2 = QGraphicsPixmapItem(px);
        item2.setPos(QPointF(0, 520))
        self.graphicsScene.addItem(item2)


        self.graphicsScene.onUpdateScene = lambda : 0
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

    #override GraphicsView.drawBackground
    def drawBackground(self, qp, qrF):
        #qp.drawPixmap(QRect(-self.width()/2, -self.height()/2, self.width(), self.height()), self.bgPixmap) #, QRect(0,0, self.bgPixmap.width(), self.bgPixmap.height())) #layer.qPixmap
        qp.drawPixmap(qrF, self.bgPixmap, QRectF(0,0, float(self.bgPixmap.width()), float(self.bgPixmap.height())))