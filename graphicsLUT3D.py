import sys
from PyQt4.QtGui import QApplication, QPainter, QWidget
from PyQt4.QtGui import QGraphicsView, QGraphicsScene, QGraphicsPathItem , QGraphicsPixmapItem, QGraphicsTextItem, QPolygonF, QGraphicsPolygonItem , QPainterPath, QPainterPathStroker, QPen, QBrush, QColor, QPixmap, QMainWindow, QLabel, QSizePolicy
from PyQt4.QtCore import Qt, QPoint, QPointF, QRect, QRectF
import numpy as np
from time import time
from LUT3D import LUTSTEP, LUT3D, rgb2hsv, hsp2rgb
from colorModels import hueSatModel, pbModel


SCENE = {}
qpp0=QPainterPath()
qpp0.addRect(0,0,10,10)

qpp1=QPainterPath()
qpp0.addEllipse(0,0,5,5)
class activeNode(QGraphicsPathItem):

    def __init__(self, position, parent=None):
        super(activeNode, self).__init__()
        self.setPos(position)
        # color from model
        c = QColor(SCENE['item1'].QImg.pixel(int(position.x()), int(position.y())))
        self.r, self.g, self.b = c.red(), c.green(), c.blue()
        self.rM, self.gM, self.bM = self.r, self.g, self.b
        self.hue, self.sat, self.pB= rgb2hsv(self.r, self.g, self.b, perceptual = True)
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
        c = QColor(SCENE['item1'].QImg.pixel(int(self.pos().x()), int(self.pos().y())))
        self.rM, self.gM, self.bM = c.red(), c.green(), c.blue()
        hue, sat,_ = rgb2hsv(self.rM, self.gM, self.bM, perceptual=True)
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
        self.w = (parent.QImg.width() - 1) / float((self.n - 1))
        self.setPos(0,0)
        self.gridNodes = [[activeNode(QPointF(i*self.w,j*self.w), parent=self) for i in range(self.n) ] for j in range(self.n)]

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
    def FromPolygon(cls, parent=None):
        Triangle = QPolygonF()
        Triangle.append(QPointF(-10., 10))
        Triangle.append(QPointF(0., 0))
        Triangle.append(QPointF(10., 10))
        #Triangle.append(QPointF(-10., 10))

        item3 = activeMarker(parent=parent)
        item3.setPolygon(Triangle)
        item3.setPen(QPen(QColor(255, 255, 255)))
        item3.setBrush(QBrush(QColor(255, 255, 255)))
        item3.moveRange = item3.parentItem().boundingRect().bottomRight()
        item3.onMouseMove = lambda : 0
        return item3


    def mousePressEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        x,y=e.scenePos().x(), e.scenePos().y()
        xmax, ymax = self.moveRange.x(), self.moveRange.y()
        x= 0 if x < 0 else xmax if x > xmax else x
        y = 0 if y < 0 else ymax if y > ymax else y
        self.setPos (x, y)
        #self.onMouseMove()

    def mouseReleaseEvent(self, e):
        self.onMouseMove()

class myGraphicsPixmapItem(QGraphicsPixmapItem):

    def __init__(self, QImg):
        self.QImg = QImg
        super(myGraphicsPixmapItem, self).__init__(QPixmap.fromImage(self.QImg))

    def mousePressEvent(self, *args, **kwargs):
        pass

    def mouseMoveEvent(self, *args, **kwargs):
        pass

    def mouseReleaseEvent(self, e):
        i,j = int(e.pos().x()), int(e.pos().y())
        c = QColor(self.QImg.pixel(i,j))
        r,g,b = c.red(), c.green(), c.blue()
        hue,sat,p = rgb2hsv(r,g,b, perceptual=True)
        SCENE['item2'].setPixmap(QPixmap.fromImage(pbModel.colorPicker(500, 50, hue, sat)))
        SCENE['statusbar'].setPlainText(('h %d' % hue) + '\n\n' + ('s %d' % (sat * 100)) + '\n\n' + ('p %d' % (p * 100)))

#main class
class graphicsForm3DLUT(QGraphicsView) :

    @classmethod
    def getNewWindow(cls, *args, **kwargs):
        newwindow = graphicsForm3DLUT(*args, **kwargs)
        newwindow.setAttribute(Qt.WA_DeleteOnClose)
        newwindow.setWindowTitle('New Window')
        return newwindow

    def __init__(self, *args, **kwargs):
        super(graphicsForm3DLUT, self).__init__()
        self.setBackgroundBrush(QBrush(Qt.black, Qt.SolidPattern));
        self.QImg = hueSatModel.colorPicker(500,500)
        self.currentHue, self.currentSat, self.currentPb = 0,0,0.5
        self.selected = None
        #self.bgPixmap = QPixmap.fromImage(self.QImg)
        self.graphicsScene = QGraphicsScene()
        self.setScene(self.graphicsScene)
        self.graphicsScene.LUT3D = LUT3D
        #self.LUTXY = LUTXY
        #self.graphicsScene.LUTXY=np.array(range(256))
        #label=self.graphicsScene.addWidget(QLabel())
        #label.setPos(50, 20)
        item1 = myGraphicsPixmapItem(self.QImg);
        SCENE['item1']=item1
        self.graphicsScene.addItem(item1)
        px = QPixmap.fromImage(pbModel.colorPicker(500,50, 180, self.currentPb))
        item2 = QGraphicsPixmapItem(px);
        SCENE['item2'] = item2
        item2.setPixmap(px)
        item2.setPos(QPointF(0, 520))
        self.graphicsScene.addItem(item2)

        io = QGraphicsTextItem()
        SCENE['statusbar'] = io
        io.setPos(0, 580)
        io.setDefaultTextColor(QColor(255,255,255))
        io.setPlainText('')
        self.graphicsScene.addItem(io)

        item3=activeMarker.FromPolygon(parent=item2)
        #item3.setParentItem(item2)
        #self.graphicsScene.addItem(item3)
        #self.graphicsScene.addPolygon(Triangle, QPen(QColor(255,255,255)), QBrush(QColor(255,255,255))))
        item3.setPos(item2.pixmap().width()/2,item2.pixmap().height())
        #item3.setPos(item2.pos().x()+item2.pixmap().width()/2, item2.pos().y()+item2.pixmap().height())

        item4 = activeMarker.FromPolygon(parent=item1)
        SCENE['item4'] = item4
        #item4.setParentItem(item1)
        item4.setPos(0,0)
        item4.onMouseMove = lambda h=self.currentHue, s=self.currentSat, p=self.currentPb : self.updatePbModel(h,s,p)

        self.marker4 = item4

        self.graphicsScene.onUpdateScene = lambda : 0
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self.grid = activeGrid(parent=item1)

    def select(self, hue,sat,p):
        """

        :param hue:
        :param sat:
        :param p:
        :return:
        """
        self.currentHue, self.currentSat, currentPb = hue, sat, p
        i,j= self.QImg.colorPickerGetPoint(hue,sat)
        SCENE['item4'].setPos(i,j)

        self.LUTIndices = [(r/LUTSTEP, g/LUTSTEP, b/LUTSTEP) for (r,g,b) in [hsp2rgb(hue,sat, p/100.0) for p in range(101)] ]
        self.updatePbModel(hue, sat, p)
        #SCENE['item2'].setPixmap(QPixmap.fromImage(pbModel.colorPicker(500, 50, hue, sat)))
        #SCENE['statusbar'].setPlainText(
        #('h %d' % hue) + '\n\n' + ('s %d' % (sat * 100)) + '\n\n' + ('p %d' % (p * 100)))

    def selectGridNode(self, h,s):
        #reset
        if self.selected is not None:
            self.selected.setPath(qpp1)
            self.selected.setBrush(QBrush())
        x,y = SCENE['item1'].QImg.colorPickerGetPoint(h,s)
        w=int(self.grid.w)
        self.selected = self.grid.gridNodes[int(y)/w][int(x)/w]
        self.selected.setBrush(QColor(2,55,255,255))
        self.selected.setPath(qpp0)


    def updatePbModel(self, hue,sat, p):
        marker=SCENE['item4']
        self.currentHue, self.currentSat, self.currentPb = self.QImg.hsArray[int(marker.pos().y()), int(marker.pos().x())]
        SCENE['item2'].setPixmap(QPixmap.fromImage(pbModel.colorPicker(500, 50, self.currentHue, self.currentSat)))
        SCENE['statusbar'].setPlainText(
            ('h %d' % self.currentHue) + '\n\n' + ('s %d' % (self.currentSat * 100)) + '\n\n' + ('p %d' % (self.currentPb * 100)))
        #self.graphicsScene.LUT3D[:,:,:,:]=np.zeros(self.graphicsScene.LUT3D.shape)
        savedLUT = self.graphicsScene.LUT3D.copy()
        for p, (i,j,k) in enumerate(self.LUTIndices):
            self.graphicsScene.LUT3D[k,j,i,::-1]= hsp2rgb(self.currentHue, self.currentSat, p/100.0 )

        self.graphicsScene.onUpdateScene()
        self.graphicsScene.LUT3D = savedLUT

    """
    #override GraphicsView.drawBackground
    def drawBackground(self, qp, qrF):
        #qp.drawPixmap(QRect(-self.width()/2, -self.height()/2, self.width(), self.height()), self.bgPixmap) #, QRect(0,0, self.bgPixmap.width(), self.bgPixmap.height())) #layer.qPixmap
        qp.drawPixmap(qrF, self.bgPixmap, QRectF(0,0, float(self.bgPixmap.width()), float(self.bgPixmap.height())))
    """