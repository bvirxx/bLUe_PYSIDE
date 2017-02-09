"""
Copyright (C) 2017  Bernard Virot

PeLUT - Photo editing software using adjustment layers with 1D and 3D Look Up Tables.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
import sys
from PyQt4.QtGui import QApplication, QPainter, QWidget, QPixmap, QPushButton, QListWidget, QListWidgetItem
from PyQt4.QtGui import QGraphicsView, QGraphicsScene, QAbstractItemView, QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem , QGraphicsPixmapItem, QGraphicsTextItem, QPolygonF, QGraphicsPolygonItem , QPainterPath, QPainterPathStroker, QPen, QBrush, QColor, QPixmap, QMainWindow, QLabel, QSizePolicy
from PyQt4.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QString
import numpy as np
from time import time
from LUT3D import LUTSIZE, LUTSTEP, rgb2hsB, hsp2rgb, hsp2rgbVec, hsp2rgb_ClippingInd, LUT3DFromFactory
from colorModels import hueSatModel, pbModel
from utils import optionsWidget

class nodeGroup(QGraphicsItemGroup):

    def __init__(self, grid=None, position=QPointF(), parent=None):
        super(nodeGroup, self).__init__(parent=parent)
        self.grid = grid
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.a = QPointF(100,100)
        self.initialPosition = position

    def addToGroup(self, item):

        super(nodeGroup, self).addToGroup(item)
        children = self.childItems()
        # set item position
        item.setPos(item.initialPosition - self.initialPosition)

        return

        tmp = QPointF(0.0, 0.0)
        children = self.childItems()
        for i in children:
            tmp = tmp + i.scenePos()
        tmp = tmp / len(children)
        self.a = tmp

        for i in children:
            i.delta = i.scenePos() - self.a

    def mousePressEvent(self,e):
        print "group press"
        self.mouseIsPressed = True
        self.scene().update()
        return
        super(nodeGroup, self).mousePressEvent(e)
        #for i in self.childItems():
            #i.mousePressEvent(e)

    def mouseMoveEvent(self,e):
        print "group move"
        self.mouseIsMoved = True
        # move children
        """
        for i in self.childItems():
            i.setPos(e.pos() + i.delta)
        """
        self.setPos(e.scenePos())
        #update grid
        self.grid.drawGrid()



    def mouseReleaseEvent(self, e):
        print "group release"
        #click event
        if not self.mouseIsMoved:
            if self.isSelected():
                self.grid.selectedGroup = None
                self.setSelected(False)
            else:
                self.setSelected(True)
                if self.grid.selectedGroup is not None:
                    self.grid.selectedGroup.setSelected(False)
                self.grid.selectedGroup = self
                self.grid.selectionList = self.childItems()
            self.setScale(self.scale()-0.2)
            """
            for i in self.childItems():
                if i.sceneBoundingRect().contains(e.scenePos()):
                    print "removed"
                    self.removeFromGroup(i)
            """

        for i in self.childItems():
            i.setState(i.pos())

        self.prepareGeometryChange()
        #self.a = e.pos()
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        #update grid
        self.grid.drawGrid()
        self.scene().update()
        self.scene().onUpdateLUT(options=self.scene().options)
        return
        print "group selet", self.isSelected()
        super(nodeGroup, self).mouseReleaseEvent(e)
        print "group select", self.isSelected()
        return
        print "group handler"

        self.parentItem().drawGrid()
        for i in self.childItems():
            print i.scenePos(), i.pos()

    def paint(self, qpainter, options, widget):

        # painting in local coordinates
        self.a=QPointF(0,0)
        b = qpainter.brush()
        if self.isSelected():
            qpainter.setBrush(QBrush(QColor(255,255,255)))
        else:
            qpainter.setBrush(QBrush(QColor(0,0,0)))
        if self.mouseIsPressed:
            pen = qpainter.pen()
            #qpainter.pen().setStyle(Qt.DashDotLine)
            qpainter.setPen(QPen(Qt.white, 1, Qt.DotLine, Qt.RoundCap));
            for child in self.childItems():
                qpainter.drawLine(self.a, child.pos())
            qpainter.setPen(pen)
        qpainter.drawEllipse(self.a.x(), self.a.y(), 10.0/self.scale(),10.0/self.scale())
        qpainter.setBrush(b)

    def boundingRect(self):
        # local coordinates
        p=self.a
        return QRectF(p.x(), p.y(), 10.0/self.scale(), 10.0/self.scale())




class activeNode(QGraphicsPathItem):
    """
    Grid node class.
    Each node holds the r,g,b and h,s,p values  corresponding to its
    position on the color wheel.
    The attribute LUTIndices holds the list of LUT indices
    matching h and s.
    """
    # node drawing
    qpp = QPainterPath()
    qpp.addEllipse(0, 0, 7, 7)

    #liste=[]
    def __init__(self, position, parent=None, grid=None):
        """

        :param position: QpointF
        :param parent: parent item
        :param grid: underlying grid
        """
        super(activeNode, self).__init__()
        self.setPos(position)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        # current scene
        scene = parent.scene()
        # color from model
        #c = QColor(SCENE['colorWheel'].QImg.pixel(int(position.x()), int(position.y())))
        c = QColor(scene.colorWheel.QImg.pixel(int(position.x()), int(position.y())))
        self.r, self.g, self.b = c.red(), c.green(), c.blue()
        # modified colors
        self.rM, self.gM, self.bM = self.r, self.g, self.b
        self.hue, self.sat, self.pB= rgb2hsB(self.r, self.g, self.b, perceptual = True)
        # LUTIndices : list of LUT vertices bound to node
        # vectorize self.LUTIndices = [(r / LUTSTEP, g / LUTSTEP, b / LUTSTEP) for (r, g, b) in [hsp2rgb(self.hue, self.sat, p / 100.0) for p in range(101)]]
        self.LUTIndices = hsp2rgbVec(np.array([(self.hue, self.sat, p / 100.0) for p in range(101)])[:,None])[:,0] / LUTSTEP
        self.setParentItem(parent)
        #qpp = QPainterPath()
        #qpp.addEllipse(0,0, 7,7)
        self.setPath(self.qpp)
        self.grid = grid
        #self.g = None#QGraphicsItemGroup()
        self.delta=QPointF(0,0)
        self.initialPosition = position

    def setState(self, position):
        """
        update node position and synchronize LUT
        :param position: node position in grid coordinates
        """
        self.setPos(position)
        scene = self.scene()
        c = QColor(scene.colorWheel.QImg.pixel(int(self.gridPos().x()), int(self.gridPos().y())))
        self.rM, self.gM, self.bM = c.red(), c.green(), c.blue()
        hue, sat, _ = rgb2hsB(self.rM, self.gM, self.bM, perceptual=True)
        # update LUT vertices bound to node
        for p, (i, j, k) in enumerate(self.LUTIndices):
            scene.LUT3D[k, j, i, ::-1] = hsp2rgb(hue, sat, p / 100.0)

    def gridPos(self):
        return self.scenePos() - self.grid.scenePos()

    def mousePressEvent(self, e):
        # super Press set selected to True
        super(activeNode, self).mousePressEvent(e)
        if type(self.parentItem()) is nodeGroup:
            print "exploding"
            self.parentItem().setSelected(False)
            for i in self.parentItem().childItems():
                i.setSelected(False)
                i.grid.selectionList = []
            self.scene().destroyItemGroup(self.parentItem())

        print 'node mouse press'
        #self.g.setHandlesChildEvents(False)

    def mouseMoveEvent(self, e):
        #super(activeNode, self).mouseMoveEvent(e)
        #self.g.setPos(e.pos())
        #for l in self.liste:
             #l.setPos(e.scenePos())
        #self.p.drawGrid()
        #self.scene().grid.drawGrid()
        print 'node mouse move'
        #for i in self.parentItem().childItems():
            #i.setPos(e.pos() + i.delta)
            #i.setPos(e.scenePos())
        self.setPos(e.scenePos())
        self.grid.drawGrid()


    def mouseReleaseEvent(self, e):
        print 'node mouse release'
        self.grid.selectionList.append(self)

        self.setState(self.pos())

        scene = self.scene()

        """
        # read color from model
        c = QColor(scene.colorWheel.QImg.pixel(int(self.pos().x()), int(self.pos().y())))
        self.rM, self.gM, self.bM = c.red(), c.green(), c.blue()
        hue, sat,_ = rgb2hsB(self.rM, self.gM, self.bM, perceptual=True)
        #savedLUT = self.scene.LUT3D.copy()
        for p, (i,j,k) in enumerate(self.LUTIndices):
            scene.LUT3D[k,j,i,::-1] = hsp2rgb(hue,sat, p/100.0)
        """
        self.scene().onUpdateLUT(options=self.scene().options)

        #print self.scene().selectedItems()

        super(activeNode, self).mouseReleaseEvent(e)

        self.setSelected(False)
        return
        if self.grid.selectedGroup is None:
            self.grid.selectionList = [self]
            print "group added"
            self.grid.selectedGroup = nodeGroup(grid=self.grid)
            self.grid.selectedGroup.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.grid.selectedGroup.setFlag(QGraphicsItem.ItemIsMovable, True)

            self.grid.selectedGroup.setSelected(True)

        tmp = QPointF(0.0, 0.0)
        for i in self.grid.selectionList:
            tmp = tmp + i.scenePos()
        tmp = tmp / len(self.grid.selectionList)
        # self.g.setPos(a/len(self.liste))
        self.grid.selectedGroup.a = tmp

        if hasattr(self.parentItem(), 'drawGrid'):
            self.grid.selectedGroup.setParentItem(self.parentItem())


        for i in self.grid.selectionList:
            i.delta = i.scenePos() - self.grid.selectedGroup.a
            if i.parentItem() != self.grid.selectedGroup :
                self.grid.selectedGroup.addToGroup(i)
            i.setSelected(True)

        #self.g=self.scene().createItemGroup(self.liste)
        print self.grid.selectedGroup.childItems()

        #g.setPos(self.scenePos())
        scene.addItem(self.grid.selectedGroup)

        for i in self.grid.selectionList:
            i.setSelected(True)
        return
        self.grid.selectedGroup.setSelected(True)
        print scene.selectedItems()
        print 'group added'
        print self.isSelected()
        self.grid.selectedGroup.setHandlesChildEvents(True)



class activeGrid(QGraphicsPathItem):
    selectionList =[]

    def __init__(self, size, parent=None):
        super(activeGrid, self).__init__()
        self.setParentItem(parent)
        self.size = size
        # grid step
        self.step = (parent.QImg.width() - 1) / float((self.size - 1))
        self.setPos(0,0)
        self.gridNodes = [[activeNode(QPointF(i*self.step,j*self.step), parent=self, grid=self) for i in range(self.size)] for j in range(self.size)]
        self.drawGrid()
        self.selectedGroup = None

    def drawGrid(self):
        qpp = QPainterPath()
        for i in range(self.size):
            qpp.moveTo(self.gridNodes[i][0].gridPos())
            previous = self.gridNodes[i][0].isSelected()
            for j in range(self.size):
                if previous or self.gridNodes[i][j].isSelected():
                    qpp.lineTo(self.gridNodes[i][j].gridPos())
                else:
                    qpp.moveTo(self.gridNodes[i][j].gridPos())
                previous = self.gridNodes[i][j].isSelected()
        for j in range(self.size):
            qpp.moveTo(self.gridNodes[0][j].gridPos())
            previous = self.gridNodes[0][j].isSelected()
            for i in range(self.size):
                if previous or self.gridNodes[i][j].isSelected():
                    qpp.lineTo(self.gridNodes[i][j].gridPos())
                else:
                    qpp.moveTo(self.gridNodes[i][j].gridPos())
                previous = self.gridNodes[i][j].isSelected()
        self.setPath(qpp)

class activeMarker(QGraphicsPolygonItem):

    size = 10
    triangle = QPolygonF()
    triangle.append(QPointF(-size, size))
    triangle.append(QPointF(0, 0))
    triangle.append(QPointF(size, size))

    cross = QPolygonF()
    cross.append(QPointF(-size/2, -size/2))
    cross.append(QPointF(0, 0))
    cross.append(QPointF(size / 2, size / 2))
    cross.append(QPointF(0, 0))
    cross.append(QPointF(-size / 2, size / 2))
    cross.append(QPointF(0, 0))
    cross.append(QPointF(size / 2, -size / 2))
    cross.append(QPointF(0, 0))


    @classmethod
    def fromTriangle(cls, parent=None):
        size = 10
        color = QColor(255, 255, 255)

        item = activeMarker(parent=parent)
        item.setPolygon(cls.triangle)
        item.setPen(QPen(color))
        item.setBrush(QBrush(color))
        # set move range to parent bounding rect
        item.moveRange = item.parentItem().boundingRect().bottomRight()

        return item

    @classmethod
    def fromCross(cls, parent=None):
        size = 10
        color = QColor(0, 0, 0)

        item = activeMarker(parent=parent)
        item.setPolygon(cls.cross)
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
        print 'marker press'
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
    implements a color picker : mouse click events read pixel colors
    from the Qimage self.QImg
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
        #h, s, p = rgb2hsB(r, g, b, perceptual=True)
        self.onMouseRelease(i,j,r,g,b)

#main class
class graphicsForm3DLUT(QGraphicsView) :
    """
    Interactive color wheel for 3D LUT adjustment.
    Color model is hsp.
    """
    # markers for grid nodes
    qpp0 = QPainterPath()
    qpp0.addRect(0, 0, 10, 10)
    selectBrush = QBrush(QColor(255,255,255))

    qpp1 = QPainterPath()
    qpp1.addEllipse(0, 0, 5, 5)
    unselectBrush = QBrush()

    # default perceptual brightness
    colorWheelPB = 0.45

    @classmethod
    def getNewWindow(cls, size=500, LUTSize=LUTSIZE, title='', parent=None):
        """
        build a graphicsForm3DLUT object. The parameter size gives the size of
        the color wheel. The total size of the window is adjusted
        to fit the size of the color wheel.
        :param size: size of the color wheel
        :param parent: parent widget
        :return: graphicsForm3DLUT object
        """
        newWindow = graphicsForm3DLUT(size,LUTSize=LUTSize, parent=parent)
        newWindow.setWindowTitle(title)
        return newWindow

    def __init__(self, size, LUTSize = LUTSIZE, parent=None):
        super(graphicsForm3DLUT, self).__init__(parent=parent)
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
        # LUT
        self.LUTSize, self.LUTStep, self.graphicsScene.LUT3D = LUT3DFromFactory(size=LUTSize)
        # options
        self.graphicsScene.options = {'use Selection' : True}

        # color wheel
        QImg = hueSatModel.colorWheel(size, size, perceptualBrightness=self.colorWheelPB)
        self.graphicsScene.colorWheel = colorPicker(QImg)
        self.graphicsScene.selectMarker = activeMarker.fromCross(parent=self.graphicsScene.colorWheel)
        self.graphicsScene.selectMarker.setPos(size/2, size/2)
        def f(x,y,r,g,b):
            self.graphicsScene.selectMarker.setPos(x,y)
            h,s,p = rgb2hsB(r,g,b, perceptual = True)
            self.currentHue, self.currentSat, self.currentPb = h, s, p
            self.bSliderUpdate()
            self.displayStatus()
        self.graphicsScene.colorWheel.onMouseRelease = f

        self.graphicsScene.addItem(self.graphicsScene.colorWheel)

        # Brightness slider
        self.bSliderHeight = 30
        px = QPixmap.fromImage(pbModel.colorChart(size, self.bSliderHeight, self.currentHue, self.currentSat))
        self.graphicsScene.bSlider = QGraphicsPixmapItem(px)
        #self.graphicsScene.bSlider.setPixmap(px)
        self.graphicsScene.bSlider.setPos(QPointF(0, self.graphicsScene.colorWheel.QImg.height()+20))
        self.graphicsScene.addItem(self.graphicsScene.bSlider)
        bSliderCursor = activeMarker.fromTriangle(parent=self.graphicsScene.bSlider)
        bSliderCursor.setPos(self.graphicsScene.bSlider.pixmap().width() / 2, self.graphicsScene.bSlider.pixmap().height())
        #bSliderCursor.onMouseRelease = lambda p,q : self.graphicsScene.colorWheel.setPixmap(QPixmap.fromImage(hueSatModel.colorWheel(size, size, perceptualBrightness=p / float(size))))
        def f(p,q):
            self.currentPb = p / float(size)
            self.graphicsScene.colorWheel.QImg.setPb(self.currentPb)
            self.graphicsScene.colorWheel.setPixmap(QPixmap.fromImage(self.graphicsScene.colorWheel.QImg))
            self.displayStatus()

        #bSliderCursor.onMouseRelease = lambda p, q: self.graphicsScene.colorWheel.setPixmap(QPixmap.fromImage(self.graphicsScene.colorWheel.QImg.setPb(p / float(size))))
        bSliderCursor.onMouseRelease = f
        # status bar
        self.graphicsScene.statusBar = QGraphicsTextItem()
        self.graphicsScene.statusBar.setPos(0, size + 70)
        self.graphicsScene.statusBar.setDefaultTextColor(QColor(255,255,255))
        self.graphicsScene.statusBar.setPlainText('')
        self.graphicsScene.addItem(self.graphicsScene.statusBar)

        #self.graphicsScene.bSlider.setPixmap(QPixmap.fromImage(pbModel.colorChart(QImg.width(), QImg.width() / 10, self.currentHue, self.currentSat)))
        self.displayStatus()

        self.graphicsScene.onUpdateScene = lambda : 0  # never set
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self.grid = activeGrid(self.LUTSize, parent=self.graphicsScene.colorWheel)

        # reset button
        pushButton = QPushButton("reset grid")
        pushButton.setObjectName("btn_reset")
        pushButton.setGeometry(550, size+80, 100, 25)  # x,y,w,h
        pushButton.clicked.connect(self.onReset)
        self.graphicsScene.addWidget(pushButton)

        # list of options

        self.listWidget = optionsWidget(options=['use image', 'use selection'], exclusive=True)
        self.listWidget.select(self.listWidget.items['use selection'])
        def onSelect(item):
            self.graphicsScene.options['use_selection'] = item is self.listWidget.items['use selection']
        self.listWidget.onSelect = onSelect

        """
        self.listWidget = QListWidget()
        def select(it):
            for r in range(self.listWidget.count()):
                currentItem = self.listWidget.item(r)
                if currentItem is it:
                    currentItem.setCheckState(Qt.Checked)
                else:
                    currentItem.setCheckState(Qt.Unchecked)
            self.graphicsScene.options['use_selection'] = it is self.listWidget.item(1)

        self.listWidget.itemClicked.connect(select)
        listItem = QListWidgetItem("use image", self.listWidget)
        listItem.setCheckState(Qt.Unchecked)
        self.listWidget.addItem(listItem)
        listItem = QListWidgetItem("use selection", self.listWidget)
        listItem.setCheckState(Qt.Unchecked)
        self.listWidget.addItem(listItem)
        select(listItem)
        self.listWidget.setMinimumWidth(self.listWidget.sizeHintForColumn(0))
        """
        self.listWidget.setGeometry(700,size+80, 10,100)
        self.graphicsScene.addWidget(self.listWidget)
        self.listWidget.setStyleSheet("QListWidget{background: black;} QListWidget::item{color: white;}")



    def selectGridNode(self, r, g, b, rM,gM,bM, mode=''):
        """
        select the nearest grid node corresponding to r,g,b values.
        :param h: hue between 0 and 360.0
        :param s: saturation between 0 and 1.0
        """
        w = self.grid.size

        LUTNeighborhood = [(i,j,k) for i in [r/w, (r/w)+1] for j in [g/w, (g/w)+1] for k in [b/w, (b/w)+1]]


        #reset previous selected marker
        if self.selected is not None:
            self.selected.setPath(self.qpp1)
            self.selected.setBrush(self.unselectBrush)
        # x, y : color wheel (cartesian) coordinates of the pixel corresponding to hue=h and sat=s
        h, s, p = rgb2hsB(r, g, b, perceptual=True)
        hspNeighborhood=[rgb2hsB(i*w,j*w,k*w, perceptual=True) for (i,j,k) in LUTNeighborhood]

        self.currentHue, self.currentSat, self.currentPb = h, s, p
        x, y = self.graphicsScene.colorWheel.QImg.GetPoint(h, s)
        xyNeighborhood=[self.graphicsScene.colorWheel.QImg.GetPoint(h, s) for h,s,_ in hspNeighborhood]

        step = float(self.grid.step)


        #neighbors = [self.grid.gridNodes[j][i] for j in range(int(np.floor(y / step)) - 1, int(np.ceil( y / step)) +2) if j < w for i in range(int(np.floor(x / step))-1, int(np.ceil( x / step))+2) if i < w]

        neighbors = [self.grid.gridNodes[int(round(y/step))+c][int(round(x/step))+a] for (x,y) in xyNeighborhood
                                           for a in [-2,-1,0,1,2,3] if (int(round(x/step))+a >=0 and int(round(x/step))+a < w)
                                            for c in [-2,-1,0,1,2,3] if (int(round(y/step))+c >=0 and int(round(y/step))+c < w) ]
        neighbors.sort(key=lambda n : (n.gridPos().x() -x) * (n.gridPos().x() -x) + (n.gridPos().y() -y) * (n.gridPos().y() -y))

        boundIndices =[]
        for n in neighbors:
            boundIndices.extend([tuple(l) for l in n.LUTIndices.tolist()])

        print 'bound', set(LUTNeighborhood).isdisjoint(boundIndices)

        NNN = neighbors[0]

        print 'selectgridnode rgb', r,g,b ,rM,gM,bM,NNN.r, NNN.g, NNN.b, NNN.rM, NNN.gM, NNN.bM
        print 'selectgridNodehs' , h,s, NNN.hue, NNN.sat
        print 'selectGridNode', NNN.parentItem()

        # mark self.selected
        if self.selected is not None:
            self.selected.setSelected(False)
        self.selected = NNN
        self.selected.setSelected(True)
        self.selected.setBrush(self.selectBrush)
        self.selected.setPath(self.qpp0)
        #self.selected.setZValue(+1)

        if mode == '':
            self.selected.setSelected(True)
        elif mode == 'add':
            if self.grid.selectedGroup is None:
                self.grid.selectionList = [self]
                self.grid.selectedGroup = nodeGroup(grid=self.grid, position=NNN.pos(), parent=self.grid)
                self.grid.selectedGroup.setFlag(QGraphicsItem.ItemIsSelectable, True)
                self.grid.selectedGroup.setFlag(QGraphicsItem.ItemIsMovable, True)
                self.grid.selectedGroup.setSelected(True)
                #self.grid.selectedGroup.setPos(NNN.pos())
            for i in neighbors:
                if i.parentItem() is self.grid:
                    self.grid.selectedGroup.addToGroup(i)
        #update status
        self.onSelectGridNode(h,s)

    def displayStatus(self):
        s1 = ('h : %d  ' % self.currentHue) + ('s : %d  ' % (self.currentSat * 100)) + ('p : %d  ' % (self.currentPb * 100))
        r, g, b, clipped = hsp2rgb_ClippingInd(self.currentHue, self.currentSat, self.currentPb)
        h,s,v = rgb2hsB(r, g, b)
        s2 = ('r : %d  ' % r) + ('g : %d  ' % g) + ('b : %d  ' % b) + (' *' if clipped else '')
        s3 = ('h : %d  ' % h) + ('s : %d  ' % (s * 100)) + ('v : %d  ' % (v * 100))
        self.graphicsScene.statusBar.setPlainText(s1 + '\n\n' + s3 + '\n\n' + s2)

    def bSliderUpdate(self):
        # self.currentHue, self.currentSat = h, s
        px = QPixmap.fromImage(pbModel.colorChart(self.size, self.bSliderHeight, self.currentHue, self.currentSat))
        self.graphicsScene.bSlider.setPixmap(px)

    def onSelectGridNode(self, h, s):
        self.bSliderUpdate()
        self.displayStatus()

    def onReset(self):
        _, _ ,LUT3D = LUT3DFromFactory(size=self.LUTSize)
        self.graphicsScene.LUT3D = LUT3D
        self.graphicsScene.removeItem(self.grid)
        self.grid = activeGrid(self.LUTSize, parent=self.graphicsScene.colorWheel)
        self.selected = None
        self.scene().onUpdateScene()
