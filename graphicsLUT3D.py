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

from PyQt4.QtCore import QSize
from PyQt4.QtGui import QAction
from PyQt4.QtGui import QApplication, QPainter, QWidget, QPixmap, QPushButton, QListWidget, QListWidgetItem
from PyQt4.QtGui import QGraphicsView, QGraphicsScene, QAbstractItemView, QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem , QGraphicsPixmapItem, QGraphicsTextItem, QPolygonF, QGraphicsPolygonItem , QPainterPath, QPainterPathStroker, QPen, QBrush, QColor, QPixmap, QMainWindow, QLabel, QSizePolicy
from PyQt4.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QString
import numpy as np
from time import time

from PyQt4.QtGui import QMenu
from PyQt4.QtGui import QRubberBand

from LUT3D import LUTSIZE, LUTSTEP, rgb2hsB, hsp2rgb, hsp2rgbVec, hsp2rgb_ClippingInd, LUT3DFromFactory, LUT3D_SHADOW, LUT3D_ORI
from colorModels import hueSatModel, pbModel
from utils import optionsWidget

# node blocking factor
spread = 1


class index(object):
    """
    An index object represents a 4-uple (p,i,j, k).
    p is a perceived brightness
    and i, j, k are indices in the LUT3D
    table.
    A set of index objects contains unique (i, j, k) 3-uples
    """
    def __init__(self, p, i, j ,k):
        self.p = p
        self.ind = (i, j, k)
    def __repr__(self):
        return "index(%f, %s)" % (self.p, self.ind)
    def __eq__(self, other):
        if isinstance(other, index):
          return (self.ind == other.ind)
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return (not self.__eq__(other))
        return NotImplemented
    def __hash__(self):
          return hash("%d%d%d" % self.ind)

class nodeGroup(QGraphicsItemGroup):

    @classmethod
    def groupFromList(cls, items, grid=None, position=QPointF(), parent=None):
        if not items:
            return
        newGroup = nodeGroup(grid=grid, position=position, parent=parent)

        for item in items:
            if type(item) is activeNode:
                newGroup.addToGroup(item)
        newGroup.setSelected(True)
        return newGroup

    @classmethod
    def unGroup(cls, group):
        #items = group.childItems()
        for item in group.childItems():
            group.removeFromGroup(item)

    def __init__(self, grid=None, position=QPointF(), parent=None):
        super(nodeGroup, self).__init__(parent=parent)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.grid = grid
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.a = QPointF(100,100)
        self.initialPosition = position
        self.setPos(position)

    def addToGroup(self, item):
        item.setSelected(False)
        super(nodeGroup, self).addToGroup(item)
        # set item position
        item.setPos(item.initialPosition - self.initialPosition)

    def mousePressEvent(self,e):
        print "group press"
        super(nodeGroup, self).mousePressEvent(e)
        self.mouseIsPressed = True
        #self.scene().update()

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
        super(nodeGroup, self).mouseReleaseEvent(e)
        print "group release"
        #click event
        """
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
            self.setTransformOriginPoint(QPoint(400,400)-self.pos())
            self.setScale(self.scale()+0.2)

            for i in self.childItems():
                if i.sceneBoundingRect().contains(e.scenePos()):
                    print "removed"
                    self.removeFromGroup(i)
        """
        # move child nodes and synchronize LUT
        if self.mouseIsMoved:
            for i in self.childItems():
                i.setState(i.pos())
            self.grid.drawGrid()
            self.scene().onUpdateLUT(options=self.scene().options)
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        #update grid

        return

    def contextMenuEvent(self, event):
        menu = QMenu()
        # ungroup
        actionUnGroup = QAction('UnGroup', None)
        menu.addAction(actionUnGroup)
        def f1():
            nodeGroup.unGroup(self)
            self.grid.drawGrid()
            #self.scene().onUpdateLUT(options=self.scene().options)
        actionUnGroup.triggered.connect(f1)
        # scale up
        actionScaleUp = QAction('scale up', None)
        menu.addAction(actionScaleUp)
        def f2():
            self.setScale(self.scale() * 1.1)
            self.grid.drawGrid()
            for i in self.childItems():
                i.setState(i.pos())
            self.scene().onUpdateLUT(options=self.scene().options)
        actionScaleUp.triggered.connect(f2)
        # scale down
        actionScaleDown = QAction('scale down', None)
        menu.addAction(actionScaleDown)
        def f3():
            self.setScale(self.scale() / 1.1)
            self.grid.drawGrid()
            for i in self.childItems():
                i.setState(i.pos())
            self.scene().onUpdateLUT(options=self.scene().options)
        actionScaleDown.triggered.connect(f3)
        # rotate cw
        actionRotateCW = QAction('rotate CW', None)
        menu.addAction(actionRotateCW)
        def f4():
            self.setRotation(self.rotation() + 10)
            self.grid.drawGrid()
            for i in self.childItems():
                i.setState(i.pos())
            self.scene().onUpdateLUT(options=self.scene().options)
        actionRotateCW.triggered.connect(f4)
        # rotate ccw
        actionRotateCCW = QAction('rotate CCW', None)
        menu.addAction(actionRotateCCW)
        def f5():
            self.setRotation(self.rotation() - 10)
            self.grid.drawGrid()
            for i in self.childItems():
                i.setState(i.pos())
            self.scene().onUpdateLUT(options=self.scene().options)
        actionRotateCCW.triggered.connect(f5)

        menu.exec_(event.screenPos())


    def paint(self, qpainter, options, widget):
        # local coordinates
        b = qpainter.brush()
        if self.isSelected():
            qpainter.setBrush(QBrush(QColor(255,255,255)))
        else:
            qpainter.setBrush(QBrush(QColor(0,0,0)))
        if self.isSelected(): #self.mouseIsPressed:
            pen = qpainter.pen()
            #qpainter.pen().setStyle(Qt.DashDotLine)
            qpainter.setPen(QPen(Qt.white, 1, Qt.DotLine, Qt.RoundCap));
            for child in self.childItems():
                qpainter.drawLine(QPointF(), child.pos())
            qpainter.setPen(pen)
        qpainter.drawEllipse(0.0, 0.0, 10.0,10.0)
        qpainter.setBrush(b)
    """
    def boundingRectxxxx(self):
        # local coordinates
        #p = self.a
        p = QPointF(0, 0)
        return QRectF(p.x(), p.y(), 10.0, 10.0)
    """

class activeNode(QGraphicsPathItem):
    """
    Grid node class.
    Each node holds the r,g,b and h,s,p values  corresponding to its
    position on the color wheel.
    The attribute LUTIndices holds the list of LUT indices
    matching h and s.
    """
    # node drawing
    qppE = QPainterPath()
    qppE.addEllipse(0, 0, 7, 7)
    qppR = QPainterPath()
    qppR.addRect(0, 0, 10, 10)

    def __init__(self, position, gridRow=0, gridCol=0, parent=None, grid=None):
        """
        Grid Node class. Each node is bound to a fixed color, depending on its
        initial position on the color wheel. The node is also bound to a fixed list
        of LUT vertices, corresponding to its initial color.
        When a node is moved over the color wheel, calling the method setState synchronizes
        the values of the LUT vertices with the current node position.
        :param position: QpointF node position (relative to parent item)
        :param parent: parent item
        :param grid: owner grid
        """
        super(activeNode, self).__init__()
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.setPos(position)
        self.gridRow, self.gridCol = gridRow, gridCol
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        # current scene
        scene = parent.scene()

        # read color from color wheel.
        # Node parent is the grid and grandfather is the color wheel
        # grid is at pos (0,0) on the color wheel (colorPicker object)
        # Color wheel has a non null offset for border.
        p = position - parent.parentItem().offset()
        c = QColor(scene.colorWheel.QImg.pixel(p.toPoint()))
        self.r, self.g, self.b, _ = c.getRgb()
        self.hue, self.sat, self.pB = rgb2hsB(self.r, self.g, self.b, perceptual=True)
        # modified colors
        self.rM, self.gM, self.bM = self.r, self.g, self.b


        # list of LUT vertices bound to node
        # vectorization of self.LUTIndices = [(r / LUTSTEP, g / LUTSTEP, b / LUTSTEP) for (r, g, b) in [hsp2rgb(self.hue, self.sat, p / 100.0) for p in range(101)]]
        # to match hsp2rgbVec parameter type, an axis is added before computation and removed after
        tmp = hsp2rgbVec(np.array([(self.hue, self.sat, p / 100.0) for p in range(101)]) [:, None])
        #self.LUTIndices = tmp[:, 0] / LUTSTEP
        self.LUTIndices = np.round(tmp[:,0]/float(LUTSTEP)).astype(int)
        clipped = [ (i,j,k) for i,j,k in self.LUTIndices if  i < LUTSIZE - 2 and j < LUTSIZE - 2 and k < LUTSIZE - 2]
        clipped.extend( [tuple(self.LUTIndices[len(clipped)])] if len(clipped) < len(self.LUTIndices) else [] )
        #self.LUTIndices = set(clipped)
        #self.LUTIndices = clipped
        self.LUTIndices = set([index(p/100.0, i, j, k) for p, (i, j, k) in enumerate(clipped)])
        for x in self.LUTIndices:
            #LUT3D_SHADOW[i,j,k][3]=1
            (i, j, k) = x.ind
            LUT3D_SHADOW[max(i-spread,0):i+spread+1,max(j-spread,0):j+spread+1, max(k-spread,0):k+spread+1,3] = 1
        #np.where(LUT3D_SHADOW[:,:,:,3]==0)

        # mark central node
        self.setParentItem(parent)
        c = grid.size/2
        if self.gridRow == c and self.gridCol == c:
            self.setPath(self.qppR)
        else :
            self.setPath(self.qppE)
        self.grid = grid
        #self.g = None#QGraphicsItemGroup()
        self.delta=QPointF(0,0)
        self.initialPosition = position
        self.newPos = QPointF()

    def setState(self, position):
        """
        Synchronize LUT
        :param position: node position
        """
        # update position
        #self.setPos(position)
        img = self.scene().colorWheel.QImg
        w, h = img.width(), img.height()
        # clipping
        # As grid.pos() is (0,0), grid and color wheel coordinates are
        # identical.
        # self.grid.parentItem() : color wheel, p : img coordinates
        p = (self.gridPos() - self.grid.parentItem().offset()).toPoint()
        x , y = p.x(), p.y()
        if x < 0 or  y < 0 or x >= w or y >= h:
            x, y = min(w-1, max(0,x)), min(h-1, max(0,y))
        # read current color
        c = QColor(img.pixel(x, y))
        self.rM, self.gM, self.bM, _ = c.getRgb()
        hue, sat, _ = rgb2hsB(self.rM, self.gM, self.bM, perceptual=True)
        # update LUT vertices bound to node
        contrast = self.scene().LUTContrast
        #for p, (i, j, k) in enumerate(self.LUTIndices):
        for x in self.LUTIndices:
            (i,j,k) = x.ind
            p=x.p
            #self.scene().LUT3D[k, j, i, ::-1] = hsp2rgb(hue, sat, contrast(p / 100.0))
            #a= np.array(hsp2rgb(hue, sat, contrast(p / 100.0)))
            a = np.array(hsp2rgb(hue, sat, contrast(p)))
            b = self.scene().LUT3D[k, j , i,::-1]
            c = LUT3D_ORI[k, j, i, ::-1]

            c1 = LUT3D_ORI[max(k-spread,0):k+spread+1, max(j-spread,0):j+spread+1, max(i-spread,0):i+spread+1, ::-1]
            exp = c1 +(a-c)
            self.scene().LUT3D[max(k-spread,0):k+spread+1, max(j-spread,0):j+spread+1, max(i-spread,0):i+spread+1, ::-1] = np.clip(LUT3D_ORI[max(k-spread,0):k+spread+1, max(j-spread,0):j+spread+1, max(i-spread,0):i+spread+1, ::-1] + (np.array(hsp2rgb(hue, sat, contrast(p))) - LUT3D_ORI[k, j , i,::-1]),0,255)

    def gridPos(self):
        return self.scenePos() - self.grid.scenePos()

    def neighbors(self):
        nghb = []
        if self.gridRow >0 :
            nghb.append(self.grid.gridNodes[self.gridRow-1][self.gridCol])
        if self.gridCol >0 :
            nghb.append(self.grid.gridNodes[self.gridRow][self.gridCol-1])
        if self.gridRow < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow+1][self.gridCol])
        if self.gridCol < self.grid.size - 1 :
            nghb.append(self.grid.gridNodes[self.gridRow][self.gridCol+1])
        return nghb

    def computeForces(self):
        # sum up all forces pushing item away
        xvel, yvel = 0.0, 0.0
        for i in range(self.grid.size):
            if abs(i-self.gridRow) > 50 and abs(j-self.gridCol) > 50:
                continue
            for j in range(self.grid.size):
                item = self.grid.gridNodes[i][j]
                # Vec(item,self)
                vec = self.mapToItem(item, 0, 0)
                dx = vec.x()
                dy = vec.y()
                l = 2.0 * (dx * dx + dy * dy)
                if (l > 0) :
                    xvel += (dx * 1.0) / l
                    yvel += (dy * 1.0) / l
        # substract all forces pulling items together
        weight = 50.0
        for item in self.neighbors():
            vec = self.mapToItem(item, 0, 0);
            xvel -= vec.x() / weight;
            yvel -= vec.y() / weight;
        if abs(xvel) < 0.1 and abs(yvel) < 0.1 :
            xvel = yvel = 0
        self.newPos = self.pos() + QPointF(xvel, yvel)

    def mousePressEvent(self, e):
        # super Press select node
        print [(i,j,k) for i,j,k in zip(np.where(LUT3D_SHADOW[:, :, :, 3] == 0)[0], np.where(LUT3D_SHADOW[:, :, :, 3] == 0)[1], np.where(LUT3D_SHADOW[:, :, :, 3] == 0)[2] )if i >1 and j > 1 and k > 1 and i<31 and j < 31 and k < 31]
        self.mouseIsPressed = True
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
        self.mouseIsMoved = True
        #for i in self.parentItem().childItems():
            #i.setPos(e.pos() + i.delta)
            #i.setPos(e.scenePos())
        self.setPos(e.scenePos())
        self.grid.drawGrid()


    def mouseReleaseEvent(self, e):
        print 'node mouse release'
        #self.grid.selectionList.append(self)

        self.setState(self.pos())
        """
        for i in range(self.grid.size):
            for j in range(self.grid.size):
                self.grid.gridNodes[i][j].setState(self.grid.gridNodes[i][j].pos())
        """
        #scene = self.scene()

        """
        # read color from model
        c = QColor(scene.colorWheel.QImg.pixel(int(self.pos().x()), int(self.pos().y())))
        self.rM, self.gM, self.bM = c.red(), c.green(), c.blue()
        hue, sat,_ = rgb2hsB(self.rM, self.gM, self.bM, perceptual=True)
        #savedLUT = self.scene.LUT3D.copy()
        for p, (i,j,k) in enumerate(self.LUTIndices):
            scene.LUT3D[k,j,i,::-1] = hsp2rgb(hue,sat, p/100.0)
        """
        if self.mouseIsMoved:
            self.scene().onUpdateLUT(options=self.scene().options)

        self.mouseIsPressed = False
        self.mouseIsMoved = False

        #print self.scene().selectedItems()

        super(activeNode, self).mouseReleaseEvent(e)

        #self.setSelected(False)
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

    def contextMenuEvent(self, event):
        menu = QMenu()
        actionGroup = QAction('Group', None)
        menu.addAction(actionGroup)
        actionGroup.triggered.connect(lambda : nodeGroup.groupFromList(self.scene().selectedItems(), grid=self.grid, position=self.scenePos(), parent=self.parentItem()))
        menu.exec_(event.screenPos())
    """
    def paint(self, qpainter, options, widget):
        if self.isSelected():
            print 'active node selected', self.gridRow, self.gridCol
        self.setSelected(self.isSelected())
        super(activeNode, self).paint(qpainter, options, widget)
    """

class activeGrid(QGraphicsPathItem):
    selectionList =[]

    def __init__(self, size, parent=None):
        """

        :param size: number of nodes in each dim.
        :param parent:
        """
        super(activeGrid, self).__init__()
        self.setParentItem(parent)
        self.size = size
        # grid step
        #self.step = (parent.QImg.width() - 1) / float((self.size - 1))
        self.step = parent.size / float((self.size -1))
        self.setPos(0,0)
        self.gridNodes = [[activeNode(QPointF(i*self.step,j*self.step), gridRow=i, gridCol=j, parent=self, grid=self) for i in range(self.size)] for j in range(self.size)]
        self.drawGrid()
        self.selectedGroup = None
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def reset(self):
        """
        unselect and reset all nodes to their initial position
        """
        for i in range(self.size):
            for j in range(self.size):
                node = self.gridNodes[i][j]
                node.setPos(node.initialPosition)
                node.setSelected(False)

    def setElasticPos(self):
        for i in range(self.size) :
            for j in range(self.size):
                self.gridNodes[i][j].computeForces()
        for i in range(self.size) :
            for j in range(self.size):
                self.gridNodes[i][j].setPos(self.gridNodes[i][j].newPos)

    def drawGrid(self):
        #self.setElasticPos()
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
        pos = e.scenePos() -self.parentItem().scenePos()
        x, y = pos.x(), pos.y()
        # limit move to (0,0) and moveRange
        xmax, ymax = self.moveRange.x(), self.moveRange.y()
        x, y = 0 if x < 0 else xmax if x > xmax else x, 0 if y < 0 else ymax if y > ymax else y
        self.setPos (x, y)
        self.onMouseMove(x,y)

    def mouseReleaseEvent(self, e):
        pos = e.scenePos() - self.parentItem().scenePos()
        x, y = pos.x(), pos.y()
        # limit move to (0,0) and moveRange
        xmax, ymax = self.moveRange.x(), self.moveRange.y()
        x, y = 0 if x < 0 else xmax if x > xmax else x, 0 if y < 0 else ymax if y > ymax else y
        self.onMouseRelease(x, y)

class colorPicker(QGraphicsPixmapItem):
    """
    implements rubber band selection and color picking.
    Mouse click events read pixel colors from QImage self.QImg.
    """
    def __init__(self, QImg, size=0, border=0):
        self.QImg = QImg
        if size == 0:
            self.size = min(QImg.width(), QImg.heigth())
        else:
            self.size = size
        super(colorPicker, self).__init__(QPixmap.fromImage(self.QImg))
        self.setOffset(QPointF(-border, -border))
        self.border = border
        self.onMouseRelease = lambda x, y, z : 0
        self.rubberBand = None

    def mousePressEvent(self, e):
        #super(colorPicker, self).mousePressEvent(e)
        print 'color picker mouse press'
        if e.button() == Qt.RightButton:
            return
        self.origin = e.screenPos()
        if self.rubberBand is None:
            self.rubberBand = QRubberBand(QRubberBand.Rectangle, parent=None)
        self.rubberBand.setGeometry(QRect(self.origin, QSize()))
        self.rubberBand.show()

    def mouseMoveEvent(self, e):
        self.rubberBand.setGeometry(QRect(self.origin, e.screenPos()).normalized())

    def mouseReleaseEvent(self, e):
        # rubberBand selection
        print 'color picker mouse release'
        if e.button() == Qt.RightButton:
            return
        self.rubberBand.hide()
        grid = self.scene().grid
        screenOrigin = e.screenPos() - e.pos()
        rubberRect = QRect(self.origin, e.screenPos()).normalized()
        for i in range(grid.size):
            for j in range(grid.size):
                if rubberRect.contains((grid.gridNodes[i][j].pos() + screenOrigin).toPoint()):
                    grid.gridNodes[i][j].setSelected(True)
                else :
                    if type(grid.gridNodes[i][j].parentItem()) is nodeGroup:
                        grid.gridNodes[i][j].parentItem().setSelected(False)
                    grid.gridNodes[i][j].setSelected(False)

        print 'selected items', len(self.scene().selectedItems())
        # color picking
        p = (e.pos() - self.offset()).toPoint()
        c = QColor(self.QImg.pixel(p))
        r, g, b, _ = c.getRgb()
        self.onMouseRelease(i,j,r,g,b)

#main class
class graphicsForm3DLUT(QGraphicsView) :
    """
    Interactive grid for 3D LUT adjustment.
    Default color model is hsp.
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
        build a graphicsForm3DLUT object. The parameter size represents the size of
        the color wheel, border not included. The size of the window is adjusted.
        :param size: size of the color wheel
        :param LUTSize: size of the LUT
        :param parent: parent widget
        :return: graphicsForm3DLUT object
        """
        newWindow = graphicsForm3DLUT(size,LUTSize=LUTSize, parent=parent)
        newWindow.setWindowTitle(title)
        return newWindow

    def __init__(self, size, LUTSize = LUTSIZE, parent=None):
        super(graphicsForm3DLUT, self).__init__(parent=parent)
        border = 20
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(size+80, size+200)
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
        LUT3D = LUT3DFromFactory(size=LUTSize)
        self.LUTSize, self.LUTStep, self.graphicsScene.LUTContrast, self.graphicsScene.LUT3D = LUT3D.size, LUT3D.step, LUT3D.contrast, LUT3D.LUT3DArray
        # options
        self.graphicsScene.options = {'use selection' : True}

        # color wheel
        QImg = hueSatModel.colorWheel(size, size, perceptualBrightness=self.colorWheelPB, border=border)
        self.graphicsScene.colorWheel = colorPicker(QImg, size=size, border=border)
        self.graphicsScene.selectMarker = activeMarker.fromCross(parent=self.graphicsScene.colorWheel)
        self.graphicsScene.selectMarker.setPos(size/2, size/2)
        # color wheel event handler
        def f(x,y,r,g,b):
            self.graphicsScene.selectMarker.setPos(x,y)
            h,s,p = rgb2hsB(r,g,b, perceptual = True)
            self.currentHue, self.currentSat, self.currentPb = h, s, p
            self.bSliderUpdate()
            self.displayStatus()
        self.graphicsScene.colorWheel.onMouseRelease = f
        self.graphicsScene.addItem(self.graphicsScene.colorWheel)

        # Brightness slider
        self.bSliderHeight = 20
        self.bSliderWidth = self.graphicsScene.colorWheel.QImg.width()
        px = QPixmap.fromImage(pbModel.colorChart(self.bSliderWidth, self.bSliderHeight, self.currentHue, self.currentSat))
        self.graphicsScene.bSlider = QGraphicsPixmapItem(px, parent = self.graphicsScene.colorWheel)
        self.graphicsScene.bSlider.setPos(QPointF(-border, self.graphicsScene.colorWheel.QImg.height()-border))
        self.graphicsScene.addItem(self.graphicsScene.bSlider)
        bSliderCursor = activeMarker.fromTriangle(parent=self.graphicsScene.bSlider)
        bSliderCursor.setPos(self.graphicsScene.bSlider.pixmap().width() / 2, self.graphicsScene.bSlider.pixmap().height())
        # cursor event handler
        def f(p,q):
            self.currentPb = p / float(self.bSliderWidth)
            self.graphicsScene.colorWheel.QImg.setPb(self.currentPb)
            self.graphicsScene.colorWheel.setPixmap(QPixmap.fromImage(self.graphicsScene.colorWheel.QImg))
            self.displayStatus()
        bSliderCursor.onMouseRelease = f
        # status bar
        offset = 70
        self.graphicsScene.statusBar = QGraphicsTextItem()
        self.graphicsScene.statusBar.setPos(0, size+offset)
        self.graphicsScene.statusBar.setDefaultTextColor(QColor(255,255,255))
        self.graphicsScene.statusBar.setPlainText('')
        self.graphicsScene.addItem(self.graphicsScene.statusBar)

        #self.graphicsScene.bSlider.setPixmap(QPixmap.fromImage(pbModel.colorChart(QImg.width(), QImg.width() / 10, self.currentHue, self.currentSat)))
        self.displayStatus()

        self.graphicsScene.onUpdateScene = lambda : 0  # never set
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # grid
        self.grid = activeGrid(self.LUTSize, parent=self.graphicsScene.colorWheel)
        self.graphicsScene.grid = self.grid
        # reset button
        pushButton = QPushButton("reset grid")
        pushButton.setObjectName("btn_reset")
        pushButton.setMinimumSize(1,1)
        pushButton.setGeometry(550, size+offset, 80, 21)  # x,y,w,h
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
        self.listWidget.setGeometry(700,size+offset, 10,100)
        self.graphicsScene.addWidget(self.listWidget)
        self.listWidget.setStyleSheet("QListWidget{background: black;} QListWidget::item{color: white;}")



    def selectGridNode(self, r, g, b, rM,gM,bM, mode=''):
        """
        select the nearest grid nodes corresponding to r,g,b values.
        :param r,g,b : color
        :param rM, gM, bM : color for debugging purpose
        """
        w = self.grid.size

        LUTNeighborhood = [(i,j,k) for i in [r/w, (r/w)+1] for j in [g/w, (g/w)+1] for k in [b/w, (b/w)+1]]


        #reset previous selected marker
        if self.selected is not None:
            self.selected.setPath(self.qpp1)
            self.selected.setBrush(self.unselectBrush)


        h, s, p = rgb2hsB(r, g, b, perceptual=True)
        hspNeighborhood=[rgb2hsB(i*w,j*w,k*w, perceptual=True) for (i,j,k) in LUTNeighborhood if (i*w<=255 and j*w<=255 and k*w<=255)]
        # currently selected values in adjust layer
        self.currentHue, self.currentSat, self.currentPb = h, s, p
        # x, y : color wheel (cartesian origin top left corner) coordinates of the pixel corresponding to hue=h and sat=s
        xc, yc = self.graphicsScene.colorWheel.QImg.GetPoint(h, s)

        xyNeighborhood=[self.graphicsScene.colorWheel.QImg.GetPoint(h, s) for h,s,_ in hspNeighborhood]

        step = float(self.grid.step)
        border = self.graphicsScene.colorWheel.border
        #grid coordinates
        xcGrid, ycGrid = xc - border, yc -border

        #NNN = self.grid.gridNodes[int(round(ycGrid/step))][int(round(xcGrid/step))]
        NNN = self.grid.gridNodes[int(np.floor(ycGrid / step))][int(np.floor(xcGrid / step))]

        #neighbors = [self.grid.gridNodes[j][i] for j in range(int(np.floor(y / step)) - 1, int(np.ceil( y / step)) +2) if j < w for i in range(int(np.floor(x / step))-1, int(np.ceil( x / step))+2) if i < w]

        neighbors = [self.grid.gridNodes[int(round(y/step))+c][int(round(x/step))+a] for (x,y) in xyNeighborhood
                                           for a in [-2,-1,0,1,2,3] if (int(round(x/step))+a >=0 and int(round(x/step))+a < w)
                                            for c in [-2,-1,0,1,2,3] if (int(round(y/step))+c >=0 and int(round(y/step))+c < w) ]
        neighbors.sort(key=lambda n : (n.gridPos().x() -xc) * (n.gridPos().x() -xc) + (n.gridPos().y() -yc) * (n.gridPos().y() -yc))

        boundIndices =[]
        for n in neighbors:
            boundIndices.extend([tuple(l.ind) for l in n.LUTIndices])

        print 'bound', set(LUTNeighborhood).isdisjoint(boundIndices)

        #NNN = neighbors[0]

        print 'selectgridnode rgb', r,g,b ,rM,gM,bM,NNN.r, NNN.g, NNN.b, NNN.rM, NNN.gM, NNN.bM
        print 'selectgridNodehs' , h,s, NNN.hue, NNN.sat
        print 'selectGridNode', NNN.parentItem()

        # select and mark selected node
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
        px = QPixmap.fromImage(pbModel.colorChart(self.bSliderWidth, self.bSliderHeight, self.currentHue, self.currentSat))
        self.graphicsScene.bSlider.setPixmap(px)

    def onSelectGridNode(self, h, s):
        self.bSliderUpdate()
        self.displayStatus()

    def onReset(self):
        """

        """
        # get a fresh LUT
        #_, _ ,LUT3D = LUT3DFromFactory(size=self.LUTSize)
        self.graphicsScene.LUT3D = LUT3DFromFactory(size=self.LUTSize).LUT3DArray

        # explode all node groups
        groupList = [item for item in self.grid.childItems() if type(item) is nodeGroup]
        for item in groupList:
            item.prepareGeometryChange()
            self.scene().destroyItemGroup(item)

        # reset grid
        self.grid.reset()
        self.selected = None
        self.grid.drawGrid()
        self.scene().onUpdateScene()
