import sys
from PyQt4.QtGui import QApplication, QPainter, QWidget, QPixmap
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
qpp1.addEllipse(0,0,5,5)
class activeNode(QGraphicsPathItem):

    def __init__(self, position, parent=None):
        super(activeNode, self).__init__()
        self.setPos(position)
        # color from model
        c = QColor(SCENE['colorWheel'].QImg.pixel(int(position.x()), int(position.y())))
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
        c = QColor(SCENE['colorWheel'].QImg.pixel(int(self.pos().x()), int(self.pos().y())))
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
        #self.w = (parent.QImg.width() - 1) / float((self.n - 1))
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
        self.onMouserelease = lambda x, y, z : 0

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
        self.onMouseRelease(r,g,b)
        """
        hue,sat,p = rgb2hsv(r,g,b, perceptual=True)
        size = self.QImg.width()
        SCENE['bSlider'].setPixmap(QPixmap.fromImage(pbModel.colorChart(500, 50, hue, sat)))
        SCENE['statusbar'].setPlainText(('h : %d ' % hue) + ('s : %d ' % (sat * 100)) + ('p : %d ' % (p * 100)))
        """

#main class
class graphicsForm3DLUT(QGraphicsView) :
    """
    Color wheel for 3D LUT
    """
    @classmethod
    def getNewWindow(cls, size=100):
        newWindow = graphicsForm3DLUT(size)
        newWindow.setAttribute(Qt.WA_DeleteOnClose)
        newWindow.setWindowTitle('3D LUT')
        return newWindow

    def __init__(self, size):
        super(graphicsForm3DLUT, self).__init__()
        self.setBackgroundBrush(QBrush(Qt.black, Qt.SolidPattern));
        self.currentHue, self.currentSat, self.currentPb = 0, 0, 0.45

        # currently selected grid node
        self.selected = None
        #self.bgPixmap = QPixmap.fromImage(self.QImg)
        self.graphicsScene = QGraphicsScene()
        self.setScene(self.graphicsScene)
        self.graphicsScene.LUT3D = LUT3D

        # color wheel
        QImg = hueSatModel.colorWheel(size, size, perceptualBrightness=self.currentPb)
        colorWheel = colorPicker(QImg)

        # update of bSlider
        def f(r, g, b):
            self.currentHue, self.currentSat, self.currentPb = rgb2hsv(r, g, b, perceptual=True)
            size = QImg.width()
            SCENE['bSlider'].setPixmap(QPixmap.fromImage(pbModel.colorChart(QImg.width(), QImg.width() / 10, self.currentHue, self.currentSat)))
            #SCENE['statusbar'].setPlainText(('h : %d ' % self.currentHue) + ('s : %d ' % (self.currentSat * 100)) + ('p : %d ' % (self.currentPb * 100)))
            self.displayStatus()

        colorWheel.onMouseRelease = f

        SCENE['colorWheel']=colorWheel
        self.graphicsScene.addItem(colorWheel)

        # Brightness slider
        px = QPixmap.fromImage(pbModel.colorChart(size, size/10, self.currentHue, self.currentSat))
        bSlider = QGraphicsPixmapItem(px)
        SCENE['bSlider'] = bSlider
        bSlider.setPixmap(px)
        bSlider.setPos(QPointF(0, colorWheel.QImg.height()+20))
        self.graphicsScene.addItem(bSlider)
        bSliderCursor = activeMarker.fromTriangle(parent=bSlider)
        bSliderCursor.setPos(bSlider.pixmap().width() / 2, bSlider.pixmap().height())
        bSliderCursor.onMouseRelease = lambda p,q : colorWheel.setPixmap(QPixmap.fromImage(hueSatModel.colorWheel(size, size, perceptualBrightness=p / float(size))))


        # status bar
        statusBar = QGraphicsTextItem()
        SCENE['statusbar'] = statusBar
        statusBar.setPos(0, size + 80)
        statusBar.setDefaultTextColor(QColor(255,255,255))
        statusBar.setPlainText('')
        self.graphicsScene.addItem(statusBar)

        SCENE['bSlider'].setPixmap(QPixmap.fromImage(pbModel.colorChart(QImg.width(), QImg.width() / 10, self.currentHue, self.currentSat)))
        #SCENE['statusbar'].setPlainText(('h : %d ' % self.currentHue) + ('s : %d ' % (self.currentSat * 100)) + ('p : %d ' % (self.currentPb * 100)))
        self.displayStatus()

        """
        # hue sat cursor
        item4 = activeMarker.fromTriangle(parent=colorWheel)
        SCENE['item4'] = item4
        #item4.setParentItem(item1)
        item4.setPos(0,0)
        #item4.setMouseMoveHook(lambda h=self.currentHue, s=self.currentSat, p=self.currentPb : self.updatePbModel(h,s,p))

        #self.marker4 = item4
        """
        self.graphicsScene.onUpdateScene = lambda : 0
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self.grid = activeGrid(parent=colorWheel)

    """
    def select(self, hue,sat,p):

        self.currentHue, self.currentSat, currentPb = hue, sat, p
        i,j= self.QImg.colorPickerGetPoint(hue,sat)
        SCENE['item4'].setPos(i,j)

        self.LUTIndices = [(r/LUTSTEP, g/LUTSTEP, b/LUTSTEP) for (r,g,b) in [hsp2rgb(hue,sat, p/100.0) for p in range(101)] ]
        self.updatePbModel(hue, sat, p)
        #SCENE['item2'].setPixmap(QPixmap.fromImage(pbModel.colorPicker(500, 50, hue, sat)))
        #SCENE['statusbar'].setPlainText(
        #('h %d' % hue) + '\n\n' + ('s %d' % (sat * 100)) + '\n\n' + ('p %d' % (p * 100)))
    """

    def selectGridNode(self, h,s):
        """
        select the grid node corresponding to the
        hue and sat parameters
        :param h: hue
        :param s: saturation
        """
        #reset marker
        if self.selected is not None:
            self.selected.setPath(qpp1)
            self.selected.setBrush(QBrush())
        # image coordinates of pixel corresponding to hue=h and sat = s
        x,y = SCENE['colorWheel'].QImg.colorPickerGetPoint(h,s)

        w=int(self.grid.step)
        self.selected = self.grid.gridNodes[int(y)/w][int(x)/w]
        self.selected.setBrush(QColor(255,255,255))
        self.selected.setPath(qpp0)

    def displayStatus(self):
        SCENE['statusbar'].setPlainText(('h : %d  ' % self.currentHue) + ('s : %d  ' % (self.currentSat * 100)) + ('p : %d  ' % (self.currentPb * 100)))

    """
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