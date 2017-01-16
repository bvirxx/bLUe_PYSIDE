import sys
from PyQt4.QtGui import QApplication, QPainter, QWidget, QPixmap
from PyQt4.QtGui import QGraphicsView, QGraphicsScene, QGraphicsPathItem , QGraphicsPixmapItem, QGraphicsTextItem, QPolygonF, QGraphicsPolygonItem , QPainterPath, QPainterPathStroker, QPen, QBrush, QColor, QPixmap, QMainWindow, QLabel, QSizePolicy
from PyQt4.QtCore import Qt, QPoint, QPointF, QRect, QRectF
import numpy as np
from time import time
from LUT3D import LUTSTEP, LUT3D, rgb2hsB, hsp2rgb
from colorModels import hueSatModel, pbModel


class activeNode(QGraphicsPathItem):

    def __init__(self, position, parent=None):
        super(activeNode, self).__init__()
        self.setPos(position)
        #~current scene
        scene = parent.scene()
        # color from model
        #c = QColor(SCENE['colorWheel'].QImg.pixel(int(position.x()), int(position.y())))
        c = QColor(scene.colorWheel.QImg.pixel(int(position.x()), int(position.y())))
        self.r, self.g, self.b = c.red(), c.green(), c.blue()
        self.rM, self.gM, self.bM = self.r, self.g, self.b
        self.hue, self.sat, self.pB= rgb2hsB(self.r, self.g, self.b, perceptual = True)
        self.LUTIndices = [(r / LUTSTEP, g / LUTSTEP, b / LUTSTEP) for (r, g, b) in [hsp2rgb(self.hue, self.sat, p / 100.0) for p in range(101)]]
        self.setParentItem(parent)
        qpp = QPainterPath()
        qpp.addEllipse(0,0, 5,5)
        self.setPath(qpp)

    def mousePressEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        self.setPos(e.scenePos())
        self.parentItem().drawGrid()

    def mouseReleaseEvent(self, e):
        # color from model
        c = QColor(self.scene().colorWheel.QImg.pixel(int(self.pos().x()), int(self.pos().y())))
        self.rM, self.gM, self.bM = c.red(), c.green(), c.blue()
        hue, sat,_ = rgb2hsB(self.rM, self.gM, self.bM, perceptual=True)
        #savedLUT = self.scene.LUT3D.copy()
        print self.LUTIndices
        for p, (i,j,k) in enumerate(self.LUTIndices):
            self.scene().LUT3D[k,j,i,::-1] = hsp2rgb(hue,sat, p/100.0)
        self.scene().onUpdateScene()
        #self.graphicsScene.LUT3D = savedLUT



class activeGrid(QGraphicsPathItem):

    def __init__(self, parent=None):
        super(activeGrid, self).__init__()
        self.setParentItem(parent)
        self.n = 17
        # grid step
        self.step = (parent.QImg.width() - 1) / float((self.n - 1))
        self.setPos(0,0)
        self.gridNodes = [[activeNode(QPointF(i*self.step,j*self.step), parent=self) for i in range(self.n) ] for j in range(self.n)]
        self.drawGrid()

    def drawGrid(self):
        qpp = QPainterPath()
        for i in range(self.n):
            qpp.moveTo(self.gridNodes[i][0].pos())
            for j in range(self.n):
                qpp.lineTo(self.gridNodes[i][j].pos())
        for j in range(self.n):
            qpp.moveTo(self.gridNodes[0][j].pos())
            for i in range(self.n):
                qpp.lineTo(self.gridNodes[i][j].pos())
        self.setPath(qpp)

class activeMarker(QGraphicsPolygonItem):

    @classmethod
    def fromTriangle(cls, parent=None):
        size = 10
        color = QColor(255, 255, 255)

        triangle = QPolygonF()
        triangle.append(QPointF(-size, size))
        triangle.append(QPointF(0, 0))
        triangle.append(QPointF(size, size))

        item = activeMarker(parent=parent)
        item.setPolygon(triangle)
        item.setPen(QPen(color))
        item.setBrush(QBrush(color))
        # set move range to parent bounding rect
        item.moveRange = item.parentItem().boundingRect().bottomRight()

        return item

    def __init__(self, *args, **kwargs):
        super(activeMarker, self).__init__(*args, **kwargs)
        self.onMouseMove, self.onMouseRelease  = lambda x,y: 0, lambda x,y: 0
        self.moveRange = QPointF(0.0, 0.0)

    def mousePressEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        pos = e.scenePos()
        x, y = pos.x(), pos.y()
        # limit move to (0,0) and moveRange
        xmax, ymax = self.moveRange.x(), self.moveRange.y()
        x, y = 0 if x < 0 else xmax if x > xmax else x, 0 if y < 0 else ymax if y > ymax else y
        self.setPos (x, y)
        self.onMouseMove(x,y)

    def mouseReleaseEvent(self, e):
        pos = e.scenePos()
        x, y = pos.x(), pos.y()
        self.onMouseRelease(x, y)

class colorPicker(QGraphicsPixmapItem):
    """
    implements a color picker : a mouse click event reads pixel color from image
    """
    def __init__(self, QImg):
        self.QImg = QImg
        super(colorPicker, self).__init__(QPixmap.fromImage(self.QImg))
        self.onMouseRelease = lambda x, y, z : 0

    def mousePressEvent(self, *args, **kwargs):
        pass

    def mouseMoveEvent(self, *args, **kwargs):
        pass

    def mouseReleaseEvent(self, e):
        point = e.pos().toPoint()
        i, j = point.x(), point.y()
        # get color from image
        c = QColor(self.QImg.pixel(i,j))
        r,g,b = c.red(), c.green(), c.blue()
        h, s, p = rgb2hsB(r, g, b, perceptual=True)
        self.onMouseRelease(h,s, p)

#main class
class graphicsForm3DLUT(QGraphicsView) :
    """
    Color wheel for 3D LUT
    """
    # markers for grid nodes
    qpp0 = QPainterPath()
    qpp0.addRect(0, 0, 10, 10)
    selectBrush = QBrush(QColor(255,255,255))

    qpp1 = QPainterPath()
    qpp1.addEllipse(0, 0, 5, 5)
    unselectBrush = QBrush()

    colorWheelPB = 0.45

    @classmethod
    def getNewWindow(cls, size=500):
        newWindow = graphicsForm3DLUT(size)
        newWindow.setAttribute(Qt.WA_DeleteOnClose)
        newWindow.setWindowTitle('3D LUT')
        return newWindow

    def __init__(self, size):
        super(graphicsForm3DLUT, self).__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(size+40, size+170)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setBackgroundBrush(QBrush(Qt.black, Qt.SolidPattern));
        self.currentHue, self.currentSat, self.currentPb = 0, 0, 0.45
        self.size = size
        # currently selected grid node
        self.selected = None
        #self.bgPixmap = QPixmap.fromImage(self.QImg)
        self.graphicsScene = QGraphicsScene()
        self.setScene(self.graphicsScene)
        self.graphicsScene.LUT3D = LUT3D

        # color wheel
        QImg = hueSatModel.colorWheel(size, size, perceptualBrightness=self.colorWheelPB)
        self.graphicsScene.colorWheel = colorPicker(QImg)

        self.graphicsScene.addItem(self.graphicsScene.colorWheel)

        # Brightness slider
        self.bSliderHeight = 30
        px = QPixmap.fromImage(pbModel.colorChart(size, self.bSliderHeight, self.currentHue, self.currentSat))
        self.graphicsScene.bSlider = QGraphicsPixmapItem(px)
        self.graphicsScene.bSlider.setPixmap(px)
        self.graphicsScene.bSlider.setPos(QPointF(0, self.graphicsScene.colorWheel.QImg.height()+20))
        self.graphicsScene.addItem(self.graphicsScene.bSlider)
        bSliderCursor = activeMarker.fromTriangle(parent=self.graphicsScene.bSlider)
        bSliderCursor.setPos(self.graphicsScene.bSlider.pixmap().width() / 2, self.graphicsScene.bSlider.pixmap().height())
        bSliderCursor.onMouseRelease = lambda p,q : self.graphicsScene.colorWheel.setPixmap(QPixmap.fromImage(hueSatModel.colorWheel(size, size, perceptualBrightness=p / float(size))))


        # status bar
        self.graphicsScene.statusBar = QGraphicsTextItem()
        self.graphicsScene.statusBar.setPos(0, size + 80)
        self.graphicsScene.statusBar.setDefaultTextColor(QColor(255,255,255))
        self.graphicsScene.statusBar.setPlainText('')
        self.graphicsScene.addItem(self.graphicsScene.statusBar)

        #self.graphicsScene.bSlider.setPixmap(QPixmap.fromImage(pbModel.colorChart(QImg.width(), QImg.width() / 10, self.currentHue, self.currentSat)))
        self.displayStatus()

        self.graphicsScene.onUpdateScene = lambda : 0
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self.grid = activeGrid(parent=self.graphicsScene.colorWheel)


    def selectGridNode(self, r, g, b):
        """
        select the nearest grid node corresponding to given
        values for hue and saturation.
        :param h: hue between 0 and 360.0
        :param s: saturation between 0 and 1.0
        """
        #reset previously selected marker
        if self.selected is not None:
            self.selected.setPath(self.qpp1)
            self.selected.setBrush(self.unselectBrush)
        # image coordinates of the pixel corresponding to hue=h and sat = s
        h, s, p = rgb2hsB(r, g, b, perceptual=True)
        self.currentHue, self.currentSat, self.currentPb = h, s, p
        x,y = self.graphicsScene.colorWheel.QImg.GetPoint(h, s)

        w=int(self.grid.step)
        x, y = int(round(x/w)), int(round(y/w))
        self.selected = self.grid.gridNodes[y][x]
        self.selected.setBrush(self.selectBrush)
        self.selected.setPath(self.qpp0)
        self.onSelectGridNode(h,s)

    def displayStatus(self):
        s1 = ('h : %d  ' % self.currentHue) + ('s : %d  ' % (self.currentSat * 100)) + ('p : %d  ' % (self.currentPb * 100))
        r, g, b = hsp2rgb(self.currentHue, self.currentSat, self.currentPb)
        h,s,v = rgb2hsB(r, g, b)
        s2 = ('r : %d  ' % r) + ('g : %d  ' % g) + ('b : %d  ' % b)
        s3 = ('h : %d  ' % h) + ('s : %d  ' % (s * 100)) + ('v : %d  ' % (v * 100))
        self.graphicsScene.statusBar.setPlainText(s1 + '\n\n' + s3 + '\n\n' + s2)

    def bSliderUpdate(self, h,s):
        # self.currentHue, self.currentSat = h, s
        px = QPixmap.fromImage(pbModel.colorChart(self.size, self.bSliderHeight, self.currentHue, self.currentSat))
        self.graphicsScene.bSlider.setPixmap(px)

    def onSelectGridNode(self, h, s):
        self.bSliderUpdate(h, s)
        self.displayStatus()