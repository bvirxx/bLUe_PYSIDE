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
import cv2

import numpy as np

from PySide2.QtCore import QSize
from PySide2.QtWidgets import QAction, QFileDialog, QToolTip, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QApplication
from PySide2.QtGui import QPainter, QPolygonF, QPainterPath, QPen, QBrush, QColor, QPixmap
from PySide2.QtWidgets import QGraphicsScene, QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem,\
    QGraphicsPixmapItem, QGraphicsTextItem,  QGraphicsPolygonItem, QSizePolicy
from PySide2.QtCore import Qt, QPointF, QRect, QRectF
from PySide2.QtGui import QImage
from PySide2.QtWidgets import QMenu, QRubberBand

from bLUeCore.bLUeLUT3D import LUT3D
from MarkedImg import QLayer
from bLUeCore.trilinear import interpTriLinear
from bLUeGui.graphicsForm import baseGraphicsForm
from bLUeGui.memory import weakProxy
from debug import tdec
from lutUtils import LUTSIZE, LUTSTEP, LUT3D_SHADOW, LUT3D_ORI, LUT3DIdentity
from versatileImg import vImage
from bLUeGui.colorPatterns import hueSatPattern, brightnessPattern
from bLUeGui.bLUeImage import QImageBuffer
from utils import optionsWidget, UDict, QbLUePushButton
from bLUeGui.dialog import dlgWarn, dlgInfo


#######################
# node neighborhood radius
spread = 1
######################


class index(object):
    """
    the class index is designed to handle sets of
    4-dim coordinates.
    An index object represents a 4-uple (p, i, j, k),
    i, j, k are indices in a 3D LUT and p is the corresponding brightness.
    A set of index objects contains unique (i, j, k) 3-uples
    """
    def __init__(self, p, i, j, k):
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
        """
        Inits a new group and adds all activeNode objects from list to the group.
        @param items: 
        @type items: list
        @param grid: 
        @type grid: activeGrid
        @param position: group position
        @type position: QPointF
        @param parent: 
        @return: group
        @rtype: nodeGroup
        """
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
        for item in group.childItems():
            group.removeFromGroup(item)

    def __init__(self, grid=None, position=QPointF(), parent=None):
        super().__init__(parent=parent)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.grid = grid
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.initialPosition = position
        self.setPos(position)

    def addToGroup(self, item):
        item.setSelected(False)
        super().addToGroup(item)

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        self.mouseIsPressed = True
        # draw links to neighbors
        self.grid.drawTrace = True

    def mouseMoveEvent(self, e):
        self.mouseIsMoved = True
        # move children
        self.setPos(e.scenePos())
        # update grid
        self.grid.drawGrid()

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        # click event
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
        self.grid.drawTrace = False
        if self.mouseIsMoved:
            for i in self.childItems():
                i.setState(i.pos())
                i.isControlPoint = True  # 28/10
            self.grid.drawGrid()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        self.mouseIsPressed = False
        self.mouseIsMoved = False

    def contextMenuEvent(self, event):
        menu = QMenu()
        # ungroup
        actionUnGroup = QAction('UnGroup', None)
        menu.addAction(actionUnGroup)

        def f1():
            nodeGroup.unGroup(self)
            self.grid.drawGrid()
            # self.scene().onUpdateLUT(options=self.scene().options)
        actionUnGroup.triggered.connect(f1)
        # scale up
        actionScaleUp = QAction('scale up', None)
        menu.addAction(actionScaleUp)

        def f2():
            self.setScale(self.scale() * 1.1)
            self.grid.drawGrid()
            for i in self.childItems():
                i.setState(i.pos())
            # self.scene().onUpdateLUT(options=self.scene().options)
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        actionScaleUp.triggered.connect(f2)
        # scale down
        actionScaleDown = QAction('scale down', None)
        menu.addAction(actionScaleDown)

        def f3():
            self.setScale(self.scale() / 1.1)
            self.grid.drawGrid()
            for i in self.childItems():
                i.setState(i.pos())
            # self.scene().onUpdateLUT(options=self.scene().options)
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        actionScaleDown.triggered.connect(f3)
        # rotate cw
        actionRotateCW = QAction('rotate CW', None)
        menu.addAction(actionRotateCW)

        def f4():
            self.setRotation(self.rotation() + 10)
            self.grid.drawGrid()
            for i in self.childItems():
                i.setState(i.pos())
            # self.scene().onUpdateLUT(options=self.scene().options)
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        actionRotateCW.triggered.connect(f4)
        # rotate ccw
        actionRotateCCW = QAction('rotate CCW', None)
        menu.addAction(actionRotateCCW)

        def f5():
            self.setRotation(self.rotation() - 10)
            self.grid.drawGrid()
            for i in self.childItems():
                i.setState(i.pos())
            # self.scene().onUpdateLUT(options=self.scene().options)
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        actionRotateCCW.triggered.connect(f5)

        menu.exec_(event.screenPos())

    def paint(self, qpainter, options, widget):
        """
        Overrides QGraphicsItemGroup paint
        @param qpainter: 
        @param options: 
        @param widget: 
        @return: 
        """
        # local coordinates
        qpainter.save()
        if self.isSelected():
            qpainter.setBrush(QBrush(QColor(255, 255, 255)))
        else:
            qpainter.setBrush(QBrush(QColor(0,0,0)))
        if self.isSelected():  # self.mouseIsPressed:
            qpainter.setPen(QPen(Qt.white, 1, Qt.DotLine, Qt.RoundCap))
            for child in self.childItems():
                qpainter.drawLine(QPointF(), child.pos())
        qpainter.restore()


class activeNode(QGraphicsPathItem):
    """
    Grid node class.
    Each node holds the r,g,b and h,s,p values  corresponding to its
    position on the color wheel.
    The attribute LUTIndices holds the list of LUT indices
    matching h and s.
        """
    # paths for node drawing
    qppE = QPainterPath()
    qppE.addEllipse(-3, -3, 6, 6)
    qppR = QPainterPath()
    qppR.addRect(-5, -5, 10, 10)

    @classmethod
    def resetNodes(cls, nodeList):
        if not nodeList:
            return
        for n in nodeList:
            if type(n) is activeNode:
                n.setPos(n.initialPos)
                n.setState(n.initialPos)
        grid = nodeList[0].grid
        grid.drawTrace = True
        grid.drawGrid()
        grid.drawTrace = False
        l = grid.scene().layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def __init__(self, position, cModel, gridRow=0, gridCol=0, parent=None, grid=None):
        """
        Grid Node class.
        Each node owns a fixed color, depending on its
        initial position on the color wheel. The node is bound to a fixed list
        of LUT vertices, corresponding to its initial color.
        When a node is moved over the color wheel, calling the method setState synchronizes
        the values of the LUT vertices with the current node position.
        @param position: node position (relative to parent item)
        @type position: QPointF
        @param cModel:
        @type cModel:
        @param gridRow:
        @type gridRow:
        @param gridCol:
        @type gridCol:
        @param parent: parent item
        @type parent:
        @param grid: owner grid
        @type grid:
        """
        super().__init__()
        self.setParentItem(parent)
        self.cModel = cModel
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.initialPos = position
        self.setPos(self.initialPos)
        self.__gridRow, self.__gridCol = gridRow, gridCol
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setVisible(False)
        # read color from the color wheel.
        # Node parent is the grid, and its grandfather is the color wheel
        # grid is at pos (0,0) on the color wheel (colorPicker object)
        # Color wheel has a non null offset for the border.
        p = position - parent.parentItem().offset()
        scene = parent.scene()
        c = QColor(scene.slider2D.QImg.pixel(p.toPoint()))
        # node color (LUT input)
        self.r, self.g, self.b, _ = c.getRgb()
        self.hue, self.sat, self.pB = self.cModel.rgb2cm(self.r, self.g, self.b)
        # modified colors (LUT output)
        self.rM, self.gM, self.bM = self.r, self.g, self.b
        # build the list of LUT vertices bound to the node:
        # we convert the list of HSV coord. (self.hue, self.sat, V) to RGB values
        # and search for the nearest neighbor vertices in the 3D LUT.
        # BrgbBuf axis 0 is brightness
        tmp = scene.slider2D.QImg.BrgbBuf[:, p.toPoint().y(), p.toPoint().x()].astype(np.float)[:, None]
        self.LUTIndices = np.round(tmp[:, 0] / float(LUTSTEP)).astype(int)
        clipped = [(i, j, k) for i, j, k in self.LUTIndices if i < LUTSIZE - 2 and j < LUTSIZE - 2 and k < LUTSIZE - 2]
        clipped.extend([tuple(self.LUTIndices[len(clipped)])] if len(clipped) < len(self.LUTIndices) else [])
        # remove duplicate vertices
        self.LUTIndices = set([index(p/100.0, i, j, k) for p, (i, j, k) in enumerate(clipped)])
        for x in self.LUTIndices:
            (i, j, k) = x.ind
            LUT3D_SHADOW[max(i-spread, 0):i + spread + 1,max(j - spread, 0):j + spread + 1, max(k - spread, 0):k + spread + 1,3] = 1
        # mark central node
        c = grid.size//2  # PYTHON 3 integer quotient
        if self.gridRow == c and self.gridCol == c:
            self.setPath(self.qppR)
        else:
            self.setPath(self.qppE)
        self.grid = grid
        # self.g = None#QGraphicsItemGroup()
        self.delta = QPointF(0, 0)
        self.initialPosition = position
        self.newPos = QPointF()
        self.isControlPoint = False  # 28/10

    @property  # read only
    def gridRow(self):
        return self.__gridRow

    @property  # read only
    def gridCol(self):
        return self.__gridCol

    def setState(self, position):
        """
        Synchronize LUT
        @param position: node position
        """
        img = self.scene().slider2D.QImg
        w, h = img.width(), img.height()

        # clipping
        p = (self.gridPos() - self.grid.parentItem().offset()).toPoint()
        x, y = p.x(), p.y()
        if x < 0 or y < 0 or x >= w or y >= h:
            x, y = min(w-1, max(0, x)), min(h - 1, max(0, y))
        # read current color
        c = QColor(img.pixel(x, y))
        self.rM, self.gM, self.bM, _ = c.getRgb()
        hue, sat, _ = self.cModel.rgb2cm(self.rM, self.gM, self.bM)  # , perceptual=True)
        # update LUT vertices bound to the node
        # A neighborhood of the vertex is built and the corresponding values
        # in the LUT are shifted by the same vector, defined by the position of the
        # node on the color wheel. The transformation keeps hue and saturation.
        lut = self.scene().lut.LUT3DArray
        for x in self.LUTIndices:
            i, j, k = x.ind
            p = x.p
            slc1 = slice(max(k - spread, 0), k + spread + 1)
            slc2 = slice(max(j - spread, 0), j + spread + 1)
            slc3 = slice(max(i - spread, 0), i + spread + 1)
            nbghd = LUT3D_ORI[slc1, slc2, slc3, ::-1]
            nbghd1 = lut[slc1, slc2, slc3, :]
            translat = np.array(self.cModel.cm2rgb(hue, sat, p)) - LUT3D_ORI[k, j, i, ::-1]
            trgNbghd = lut[slc1, slc2, slc3, :3][..., ::-1]
            trgNbghd[...] = nbghd + translat  # np.array(self.cModel.cm2rgb(hue, sat, p))
            # set alpha channel
            if np.all(translat == 0):
                nbghd1[..., 3] = 0
            else:
                nbghd1[..., 3] = 255

    def gridPos(self):
        """
        Current node position, relative to grid
        @return: position
        @rtype: QpointF
        """
        return self.scenePos() - self.grid.scenePos()

    def top(self):
        """
        Return the north neighbor
        @return:
        @rtype:
        """
        if self.gridRow > 0:
            return self.grid.gridNodes[self.gridRow-1][self.gridCol]  # TODO verify grid/col
        return None

    def left(self):
        """
        Return the east neighbor
        @return:
        @rtype:
        """
        if self.gridCol > 0:
            return self.grid.gridNodes[self.gridRow][self.gridCol-1]
        return None

    def neighbors(self):
        """
        Returns the list of grid neighbors
        @return: neighbors
        @rtype: list of activeNode objects
        """
        nghb = []
        if self.gridRow > 0 :
            nghb.append(self.grid.gridNodes[self.gridRow-1][self.gridCol])
        if self.gridCol > 0 :
            nghb.append(self.grid.gridNodes[self.gridRow][self.gridCol-1])
        if self.gridRow < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow+1][self.gridCol])
        if self.gridCol < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow][self.gridCol + 1])
        if self.gridRow > 0 and self.gridCol >0 :
            nghb.append(self.grid.gridNodes[self.gridRow - 1][self.gridCol-1])
        if self.gridRow > 0 and self.gridCol < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow-1][self.gridCol+1])
        if self.gridRow < self.grid.size - 1 and self.gridCol < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow + 1][self.gridCol + 1])
        if self.gridRow < self.grid.size - 1 and self.gridCol > 0:
            nghb.append(self.grid.gridNodes[self.gridRow + 1][self.gridCol - 1])
        return nghb

    def laplacian(self):
        """
        Return the laplacian (the mean) of the neighbor nodes.
        The laplacian coordinates are relative to the scene.
        @return:
        @rtype: QPointF
        """
        nullvec = QPointF(0.0, 0.0)
        laplacian = nullvec
        count = 0
        for item in self.neighbors():
            laplacian += item.scenePos()
            count += 1
        laplacian = laplacian / count
        return laplacian

    def mousePressEvent(self, e):
        self.mouseIsPressed = True
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        self.mouseIsMoved = True
        self.setPos(e.scenePos())
        self.grid.drawTrace = True
        self.grid.drawGrid()

    def mouseReleaseEvent(self, e):
        self.isControlPoint = True
        self.setState(self.pos())
        if self.mouseIsMoved:
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.grid.drawTrace = False
        super().mouseReleaseEvent(e)

    def contextMenuEvent(self, event):
        menu = QMenu()
        actionGroup = QAction('Group', None)
        menu.addAction(actionGroup)
        actionGroup.triggered.connect(lambda: nodeGroup.groupFromList(self.scene().selectedItems(), grid=self.grid, position=self.scenePos(), parent=self.parentItem()))
        actionGroup.setEnabled(len(self.scene().selectedItems()) > 1)
        actionReset = QAction('Reset', None)
        menu.addAction(actionReset)
        actionReset.triggered.connect(lambda: activeNode.resetNodes(self.scene().selectedItems()))
        menu.exec_(event.screenPos())


class activeGrid(QGraphicsPathItem):

    selectionList = []

    def __init__(self, size, cModel, parent=None):
        """

        @param size: node count in each dim.
        @param parent:
        """
        super().__init__()
        self.setParentItem(parent)
        self.size = size
        # parent should be the color wheel. step is the unitary coordinate increment
        # between consecutive nodes in each direction
        self.step = parent.size / float((self.size - 1))
        # set pos relative to parent
        self.setPos(0, 0)
        # np.fromiter does not handle dtype object, so we cannot use a generator
        self.gridNodes = [[activeNode(QPointF(i*self.step, j * self.step), cModel, gridRow=j, gridCol=i, parent=self, grid=self)
                           for i in range(self.size)]
                           for j in range(self.size)]
        self.drawTrace = False
        self.drawGrid()
        self.selectedGroup = None
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def getNodeAt(self, row, col):
        """
        Disambiguation of row/col
        @param row: grid row
        @type row: int
        @param col: grid col
        @type col: int
        @return: grid node
        @rtype: activeNode
        """
        return self.gridNodes[col][row]

    def paint(self, qpainter, options, widget):
        qpainter.save()
        qpainter.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine, Qt.RoundCap))
        qpainter.drawPath(self.path())
        qpainter.restore()

    def reset(self):
        """
        unselect and reset all nodes to their initial position
        """
        for i in range(self.size):
            for j in range(self.size):
                node = self.gridNodes[i][j]
                node.setPos(node.initialPosition)
                node.setSelected(False)
                node.setVisible(False)
                node.isControlPoint = False

    def smooth(self):
        """
        Try to smooth the grid by moving each non fixed point to the position of the laplacian
        of its neighbors.
        """
        """
        weight = 0.5
        # current positions of nodes
        f = lambda p: (p.scenePos().x(), p.scenePos().y())
        posArray = np.array([[[* f(self.gridNodes[i][j])] for j in range(self.size)] for i in range(self.size)])  # not transposed
        # get array of deltas : d[i,j,k,l] = coordinates of the vector gridNodes[i][j] --> gridNodes[k][l]
        d = posArray[None, None,:,:,:] - posArray[:,:,None, None,:]
        # square norms of deltas
        n2 = np.sum(np.square(d), axis=-1)
        # compute
        vel = (d / n2[...,None]) * weight
        vel = np.nan_to_num(vel)
        vel = - np.sum(vel, axis=(2,3))  # pulling nodes away
        """
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            for i in range(self.size):
                for j in range(self.size):
                    curnode = self.gridNodes[i][j]
                    curnode.newPos = curnode.laplacian()
            for i in range(self.size):
                for j in range(self.size):
                    curnode = self.gridNodes[i][j]
                    if abs(curnode.newPos.x()) < 0.1 and abs(curnode.newPos.y()) < 0.1:
                        continue
                    if curnode.gridRow == 0 or curnode.gridCol == 0 or curnode.gridRow == self.size - 1 or curnode.gridCol == self.size - 1 or curnode.isControlPoint:
                        continue
                    newPos = curnode.newPos
                    if curnode.scenePos() != newPos:
                        position = newPos - curnode.parentItem().scenePos()
                        curnode.setPos(position)  # parent coordinates
                        curnode.setState(position)
            for i in range(self.size):
                for j in range(self.size):
                    if self.gridNodes[i][j].isControlPoint:
                        self.gridNodes[i][j].setState(self.gridNodes[i][j].pos())
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()

    def drawGrid(self):
        step = 4
        qpp = QPainterPath()
        for i in range(self.size):
            for j in range(self.size):
                node = self.gridNodes[i][j]
                if i % step == 0 and j % step == 0:
                    if i > 0:
                        qpp.moveTo(node.gridPos())
                        qpp.lineTo(self.gridNodes[i-step][j].gridPos())
                    if j > 0:
                        qpp.moveTo(node.gridPos())
                        qpp.lineTo(self.gridNodes[i][j-step].gridPos())
                if not node.isSelected():
                    continue
                # mark initial position
                qpp.moveTo(node.gridPos())
                qpp.lineTo(node.initialPos)
                qpp.addEllipse(node.initialPos, 5, 5)
                # mark visible neighbors
                for n in node.neighbors():
                    if n.isVisible():
                        qpp.moveTo(n.gridPos())
                        qpp.lineTo(node.gridPos())
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
        color = QColor(0, 0, 0)
        item = activeMarker(parent=parent)
        item.setPolygon(cls.cross)
        item.setPen(QPen(color))
        item.setBrush(QBrush(color))
        # set move range to parent bounding rect
        item.moveRange = item.parentItem().boundingRect().bottomRight()
        return item

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.onMouseMove, self.onMouseRelease = lambda e, x, y: 0, lambda e, x, y: 0
        self.moveRange = QRectF(0.0, 0.0, 0.0, 0.0)

    def setMoveRange(self, rect):
        self.moveRange = rect

    def mousePressEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        # position relative to parent
        pos = e.scenePos() - self.parentItem().scenePos()
        x, y = pos.x(), pos.y()
        # limit move to moveRange
        xmin, ymin = self.moveRange.left(), self.moveRange.top()
        xmax, ymax = self.moveRange.right(), self.moveRange.bottom()
        x, y = xmin if x < xmin else xmax if x > xmax else x, ymin if y < ymin else ymax if y > ymax else y
        self.setPos(x, y)
        self.onMouseMove(e, x, y)

    def mouseReleaseEvent(self, e):
        # position relative to parent
        pos = e.scenePos() - self.parentItem().scenePos()
        x, y = pos.x(), pos.y()
        # limit move to (0,0) and moveRange
        xmin, ymin = self.moveRange.left(), self.moveRange.top()
        xmax, ymax = self.moveRange.right(), self.moveRange.bottom()
        x, y = xmin if x < xmin else xmax if x > xmax else x, ymin if y < ymin else ymax if y > ymax else y
        self.onMouseRelease(e, x, y)


class colorChooser(QGraphicsPixmapItem):
    """
    Color wheel wrapper : it is a 2D slider-like
    providing color selection from the wheel and
    rubber band selection from a grid
    of activeNode objects moving over the wheel.
    """
    def __init__(self, cModel, QImg, target=None, size=0, border=0):
        """
        @param cModel: color model
        @type cModel: cmConverter
        @param QImg: color wheel
        @type QImg: vImage
        @param target: image to sample
        @type target: QImage
        @param size: color wheel diameter
        @type size: integer
        @param border: border size
        @type border: int
        """
        self.QImg = QImg  # not a back link !!!
        self.border = border
        if size == 0:
            self.size = min(QImg.width(), QImg.heigth()) - 2 * border
        else:
            self.size = size
        self.origin = 0
        # calculate target histogram
        self.targetHist = None
        if target is not None:
            # convert to current color space
            hsxImg = cModel.rgb2cmVec(QImageBuffer(target)[:, :, :3][:, :, ::-1])
            # get polar coordinates relative to the color wheel
            xyarray = self.QImg.GetPointVec(hsxImg).astype(int)
            maxVal = self.QImg.width()
            STEP = 10
            # build 2D histogram for xyarray
            H, xedges, yedges = np.histogram2d(xyarray[:, :, 0].ravel(), xyarray[:,:,1].ravel(),
                                               bins=[np.arange(0, maxVal+STEP, STEP), np.arange(0, maxVal + STEP, STEP)],
                                               normed=True)
            w,h = QImg.width(), QImg.height()
            b = QImage(w, h, QImage.Format_ARGB32)
            b.fill(0)
            buf = QImageBuffer(b)
            # set the transparency of each pixel
            # proportionally to the height of its bin
            # get bin indices (u, v)
            u = xyarray[:,:,0] // STEP
            v = xyarray[:,:,1] // STEP
            # get heights of bins
            tmp = H[u,v]
            norma = np.amax(H)
            # color white
            buf[xyarray[:,:,1], xyarray[:,:,0],...] = 255
            # alpha channel
            buf[xyarray[:, :, 1], xyarray[:, :, 0], 3] = 90 + 128.0 * tmp / norma
            self.targetHist = b
            self.showTargetHist = False
        super().__init__(self.QImg.rPixmap)
        self.setPixmap(self.QImg.rPixmap)
        self.setOffset(QPointF(-border, -border))
        self.onMouseRelease = lambda p, x, y, z: 0
        self.rubberBand = None

    def setPixmap(self, pxmap):
        """
        Paints the histogram on a copy of pxmap
        and displays the copy.
        @param pxmap:
        @type pxmap: QPixmap
        """
        if self.targetHist is not None and self.showTargetHist:
            pxmap1 = QPixmap(pxmap)
            qp = QPainter(pxmap1)
            qp.drawImage(0, 0, self.targetHist)
            qp.end()
            pxmap = pxmap1
        super().setPixmap(pxmap)

    def updatePixmap(self):
        """
        Convenience method
        """
        self.setPixmap(self.QImg.rPixmap)

    def mousePressEvent(self, e):
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
        if e.button() == Qt.RightButton:
            return
        self.rubberBand.hide()
        grid = self.scene().grid
        screenOrigin = e.screenPos() - e.pos().toPoint()
        rubberRect = QRect(self.origin, e.screenPos()).normalized()
        for i in range(grid.size):
            for j in range(grid.size):
                if rubberRect.contains((grid.gridNodes[i][j].pos() + screenOrigin).toPoint()):
                    grid.gridNodes[i][j].setSelected(True)
                else:
                    if type(grid.gridNodes[i][j].parentItem()) is nodeGroup:
                        grid.gridNodes[i][j].parentItem().setSelected(False)
                    grid.gridNodes[i][j].setSelected(False)
        # pick color from self.QImg
        p = e.pos().toPoint()
        c = QColor(self.QImg.pixel(p - self.offset().toPoint()))
        r, g, b, _ = c.getRgb()
        self.onMouseRelease(p, r, g, b)


class graphicsForm3DLUT(baseGraphicsForm):
    """
    Form for 3D LUT editing.

    """
    # node markers
    qpp0 = activeNode.qppR
    qpp1 = activeNode.qppE
    selectBrush = QBrush(QColor(255, 255, 255))
    unselectBrush = QBrush()
    # default brightness
    defaultColorWheelBr = 0.60

    @classmethod
    def getNewWindow(cls, cModel, targetImage=None, axeSize=500, LUTSize=LUTSIZE, layer=None, parent=None, mainForm=None):
        """
        build a graphicsForm3DLUT object. The parameter size represents the size of
        the color wheel, border not included (the size of the window is adjusted).
        @param cModel
        @type cModel:
        @param targetImage
        @type targetImage:
        @param axeSize: size of the color wheel (default 500)
        @type axeSize:
        @param LUTSize: size of the LUT
        @type LUTSize:
        @param layer: layer of targetImage linked to graphics form
        @type layer:
        @param parent: parent widget
        @type parent:
        @param mainForm:
        @type mainForm:
        @return: graphicsForm3DLUT object
        @rtype:
        """
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            newWindow = graphicsForm3DLUT(cModel, targetImage=targetImage, axeSize=axeSize, LUTSize=LUTSize,
                                          layer=layer, parent=parent, mainForm=mainForm)
            newWindow.setWindowTitle(layer.name)
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
        return newWindow

    def __init__(self, cModel, targetImage=None, axeSize=500, LUTSize=LUTSIZE, layer=None, parent=None, mainForm=None):
        """
        @param cModel: color space used by colorPicker, slider2D and colorPicker
        @type cModel: cmConverter object
        @param axeSize: size of the color wheel
        @type axeSize: int
        @param targetImage:
        @type targetImage: imImage
        @param LUTSize:
        @type LUTSize: int
        @param layer: layer of targetImage linked to graphics form
        @type layer : QLayer
        @param parent:
        @type parent:
        """
        super().__init__(parent=parent)
        self.mainForm = mainForm  # used by saveLUT()
        # context help tag
        self.helpId = "LUT3DForm"
        self.cModel = cModel
        border = 20
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize + 90, axeSize + 200)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setBackgroundBrush(QBrush(Qt.black, Qt.SolidPattern))
        self.currentHue, self.currentSat, self.currentPb = 0, 0, self.defaultColorWheelBr
        self.currentR, self.currentG, self.currentB = 0, 0, 0
        self.size = axeSize
        # back links to image
        self.targetImage = weakProxy(targetImage)
        self.layer = weakProxy(layer)
        # currently selected grid node
        self.selected = None
        self.graphicsScene = QGraphicsScene()
        self.graphicsScene.options = None
        self.setScene(self.graphicsScene)
        # back to image layer
        self.graphicsScene.layer = weakProxy(layer)
        # init LUT
        freshLUT3D = LUT3D(None, size=LUTSize, alpha=True)
        self.graphicsScene.lut = freshLUT3D
        # init 2D slider
        QImg = hueSatPattern(axeSize, axeSize, cModel, bright=self.defaultColorWheelBr, border=border)
        self.graphicsScene.slider2D = colorChooser(cModel, QImg, target=self.targetImage, size=axeSize, border=border)
        self.graphicsScene.selectMarker = activeMarker.fromCross(parent=self.graphicsScene.slider2D)
        self.graphicsScene.selectMarker.setPos(axeSize / 2, axeSize / 2)
        # color wheel event handler
        def f1(p, r, g, b):
            h, s, br = self.cModel.rgb2cm(r, g, b)
            self.currentHue, self.currentSat, self.currentPb = h, s, br
            self.currentR, self.currentG, self.currentB = r, g, b
            self.bSliderUpdate()
            self.displayStatus()
        self.graphicsScene.slider2D.onMouseRelease = f1
        self.graphicsScene.addItem(self.graphicsScene.slider2D)

        # Brightness slider
        self.bSliderHeight = 20
        self.bSliderWidth = self.graphicsScene.slider2D.QImg.width()
        px = brightnessPattern(self.bSliderWidth, self.bSliderHeight, cModel, self.currentHue, self.currentSat).rPixmap
        self.graphicsScene.bSlider = QGraphicsPixmapItem(px, parent=self.graphicsScene.slider2D)
        self.graphicsScene.bSlider.setPos(QPointF(-border, self.graphicsScene.slider2D.QImg.height() - border))
        bSliderCursor = activeMarker.fromTriangle(parent=self.graphicsScene.bSlider)
        bSliderCursor.setMoveRange(QRectF(0.0, bSliderCursor.size, self.graphicsScene.bSlider.pixmap().width(), 0.0))
        bSliderCursor.setPos(self.graphicsScene.bSlider.pixmap().width() * self.defaultColorWheelBr, bSliderCursor.size)
        # cursor event handlers

        def f2(e, p, q):
            self.currentPb = p / float(self.bSliderWidth)
            QToolTip.showText(e.screenPos(), (str(int(self.currentPb * 100.0))), self)
        bSliderCursor.onMouseMove = f2

        def f3(e, p, q):
            self.currentPb = p / float(self.bSliderWidth)
            self.graphicsScene.slider2D.QImg.setPb(self.currentPb)
            self.graphicsScene.slider2D.setPixmap(self.graphicsScene.slider2D.QImg.rPixmap)
            self.displayStatus()
        bSliderCursor.onMouseRelease = f3
        # status bar
        offset = 60
        self.graphicsScene.statusBar = QGraphicsTextItem()
        self.graphicsScene.statusBar.setPos(-20, axeSize + offset)
        self.graphicsScene.statusBar.setDefaultTextColor(QColor(255, 255, 255))
        self.graphicsScene.statusBar.setPlainText('')
        self.graphicsScene.addItem(self.graphicsScene.statusBar)

        self.displayStatus()

        # self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # grid
        self.grid = activeGrid(self.graphicsScene.lut.size, self.cModel, parent=self.graphicsScene.slider2D)
        self.graphicsScene.grid = self.grid

        # buttons
        pushButton1 = QbLUePushButton("Reset Grid")
        pushButton1.clicked.connect(self.onReset)
        pushButton2 = QbLUePushButton("Save LUT")
        pushButton2.clicked.connect(self.saveLUT)
        pushButton3 = QbLUePushButton("Smooth Grid")
        pushButton3.clicked.connect(self.onSmoothGrid)
        pushButton4 = QbLUePushButton('Set Mask')
        pushButton4.clicked.connect(self.setMask)
        # pushButton4 needs enabling/disabling
        self.pushButton4 = pushButton4

        # options
        options1, optionNames1 = ['use image', 'use selection'], ['Use Image', 'Use Selection']
        self.listWidget1 = optionsWidget(options=options1, optionNames=optionNames1, exclusive=True)
        """
        def onSelect1(item):
            self.graphicsScene.options['use selection'] = item is self.listWidget1.items['use selection']
        self.listWidget1.onSelect = onSelect1
        """
        self.listWidget1.setFocusPolicy(Qt.NoFocus)
        # set initial selection to 'use image'
        self.listWidget1.checkOption(self.listWidget1.intNames[0])

        options2, optionNames2 = ['add node', 'remove node'], ['Add Node', 'Remove Node']
        self.listWidget2 = optionsWidget(options=options2, optionNames=optionNames2, exclusive=True)
        """
        def onSelect2(item):
            self.graphicsScene.options['add node'] = item is self.listWidget2.items['add node']
        self.listWidget2.onSelect = onSelect2
        """
        # set initial selection to add node'
        self.listWidget2.checkOption(self.listWidget2.intNames[0])

        options3 = ['select neighbors', 'reset removed nodes', 'show histogram', 'keep alpha']
        optionNames3 = ['Select Neighbors', 'Reset Removed', 'Show Histogram', 'Keep Alpha']
        self.listWidget3 = optionsWidget(options=options3, optionNames=optionNames3, exclusive=False)
        self.listWidget3.checkOption(self.listWidget3.intNames[0])
        self.listWidget3.checkOption(self.listWidget3.intNames[1])

        def onSelect3(item):
            option = item.internalName
            if option == 'show histogram':
                self.graphicsScene.slider2D.showTargetHist = self.graphicsScene.options[option]  # .isChecked()
                self.graphicsScene.slider2D.updatePixmap()
                return
            if option == 'keep alpha':
                self.enableButtons()
                self.layer.applyToStack()

        self.listWidget3.onSelect = onSelect3
        # set initial selection to 'select naighbors'
        item = self.listWidget3.items[options3[0]]
        item.setCheckState(Qt.Checked)
        self.graphicsScene.options = UDict((self.listWidget1.options, self.listWidget2.options,
                                            self.listWidget3.options))
        # layouts
        hlButtons = QHBoxLayout()
        hlButtons.addWidget(pushButton1)
        hlButtons.addWidget(pushButton3)
        hlButtons.addWidget(pushButton2)
        hl = QHBoxLayout()
        vl1 = QVBoxLayout()
        vl1.addWidget(self.listWidget1)
        vl1.addWidget(pushButton4)
        hl.addLayout(vl1)
        hl.addWidget(self.listWidget2)
        hl.addWidget(self.listWidget3)
        vl = QVBoxLayout()
        for l in [hlButtons, hl]:
            vl.addLayout(l)
        # Init QWidget container for adding buttons and options to graphicsScene
        container = QWidget()
        container.setObjectName("container")
        container.setLayout(vl)
        ss = """QWidget#container{background: black}\
                QListWidget{background-color: black; selection-background-color: black; border: none; font-size: 7pt}\
                QListWidget::item{color: white;}\
                QListWidget::item::selected{background: black; border: none}"""
        container.setStyleSheet(ss)
        for wdg in [self.listWidget1, self.listWidget2, self.listWidget3]:
            wdg.setMinimumWidth(wdg.sizeHintForColumn(0))
            wdg.setMinimumHeight(wdg.sizeHintForRow(0)*len(wdg.items))
        container.setGeometry(-offset//2, axeSize + offset - 20, axeSize + offset, 20)
        self.graphicsScene.addWidget(container)

        #container.setStyleSheet("QPushButton {color: white;}\
         #                       QPushButton:pressed, QPushButton:hover {background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0d5ca6, stop: 1 #2198c0);}")
        self.setWhatsThis(
""" <b>2.5D LUT Editor</b><br>
  HSpB layers are slower than HSV, but usually they give better results.<br>
  <b>Select nodes</b> with mouse clicks on the image. Selected nodes are shown
as small black circles on the color wheel.<br>
<b>Modify the color</b> of a node by dragging it on
the wheel. Several nodes can be moved simultaneously by grouping them.<br>
<b>Group nodes</b> :<br>
        &nbsp; 1 - select them with the mouse : while pressing the mouse left button, drag a rubber band around the nodes to select;<br>
        &nbsp; 2 - next, right click any one of the selected nodes and choose group from the context menu<br>
<b>unselect nodes</b> :<br>
        &nbsp; 1 - check the option Remove Node;<br>
        &nbsp; 2 -  ungroup;<br>
        &nbsp; 3 - on the image, click the pixels to unselect.<br>
<b>Caution</b> : Selecting nodes with the mouse is enabled only when
the Color Chooser is closed.<br>
Click the <b> Smooth Grid</b> button to smooth color transitions between neighbor nodes.<br>
Check the <br>Keep Alpha</b> option to forward the alpha channel without modifications.<br>
This option must be unchecked to build a mask from the 3D LUT.<br>
"""
                          )  # end of setWhatsThis

    def selectGridNode(self, r, g, b):
        """
        select the nearest grid nodes corresponding to r,g,b values.
        @param r: color
        @param g: color
        @param b: color
        """
        # surrounding cube
        # LUTNeighborhood = [(i,j,k) for i in [r/w, (r/w)+1] for j in [g/w, (g/w)+1] for k in [b/w, (b/w)+1]]

        # reset previous selected marker
        if self.selected is not None:
            self.selected.setPath(self.qpp1)
            self.selected.setBrush(self.unselectBrush)

        h, s, p = self.cModel.rgb2cm(r, g, b)  # , perceptual=True)
        # hspNeighborhood=[rgb2ccm(i * w, j * w, k * w, perceptual=True) for (i, j, k) in LUTNeighborhood if (i * w <= 255 and j * w <= 255 and k * w <= 255)]
        # currently selected values in adjust layer
        self.currentHue, self.currentSat, self.currentPb = h, s, p
        self.currentR, self.currentG, self.currentB = r, g, b
        # x, y : color wheel (cartesian origin top left corner) coordinates of the pixel corresponding to hue=h and sat=s
        xc, yc = self.graphicsScene.slider2D.QImg.GetPoint(h, s)

        # xyNeighborhood=[self.graphicsScene.slider2D.QImg.GetPoint(h, s) for h,s,_ in hspNeighborhood]

        # unitary coordinate increment between consecutive nodes
        step = float(self.grid.step)
        border = self.graphicsScene.slider2D.border
        # grid coordinates
        xcGrid, ycGrid = xc - border, yc - border
        # NNN = self.grid.gridNodes[int(round(ycGrid/step))][int(round(xcGrid/step))]
        NNN = self.grid.gridNodes[int(np.floor(ycGrid / step))][int(np.floor(xcGrid / step))]

        # select and mark selected node
        mode = self.graphicsScene.options['add node']
        if self.selected is not None:
            self.selected.setSelected(False)
        self.selected = NNN
        self.selected.setSelected(True)
        self.selected.setBrush(self.selectBrush)
        self.selected.setPath(self.qpp0)
        self.selected.setVisible(mode)
        nodesToSelect = NNN.neighbors() if self.graphicsScene.options['select neighbors'] else [NNN]
        for n in nodesToSelect:
            n.setVisible(mode)
            if self.graphicsScene.options['reset removed nodes'] and not mode:
                if isinstance(n.parentItem(), nodeGroup):
                    n.parentItem().removeFromGroup(n)
                n.setPos(n.initialPos)
                n.setState(n.initialPos)
            """
            for item in n.LUTIndices:
                #self.scene().lut.LUT3DArray[item.ind + (3,)] = 255  # TODO added 24/11/18
                i, j, k = item.ind
                self.scene().lut.LUT3DArray[k, j, i, 3] = 255  # TODO added 24/11/18
            """
        # update status
        self.onSelectGridNode(h, s)

    def displayStatus(self):
        s1 = ('h : %d  ' % self.currentHue) + ('s : %d  ' % (self.currentSat * 100))  # + ('p : %d  ' % (self.currentPb * 100))
        self.graphicsScene.statusBar.setPlainText(s1)  # + '\n' + s3 + '\n' + s2)

    def bSliderUpdate(self):
        px = brightnessPattern(self.bSliderWidth, self.bSliderHeight, self.cModel, self.currentHue, self.currentSat).rPixmap
        self.graphicsScene.bSlider.setPixmap(px)

    def onSelectGridNode(self, h, s):
        self.bSliderUpdate()
        self.displayStatus()

    def onSmoothGrid(self):
        """
        Button slot
        """
        self.grid.smooth()
        self.grid.drawGrid()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()
        # if getattr(self.layer.parentImage, 'window', None) is not None:
        # self.layer.parentImage.window.repaint()  # TODO modified 28/11/18 validate

    def onReset(self):
        """
        reset grid and LUT
        """
        # get a fresh LUT
        self.graphicsScene.lut = LUT3D(None, size=self.graphicsScene.lut.size, alpha=True)
        # explode all node groups
        groupList = [item for item in self.grid.childItems() if type(item) is nodeGroup]
        for item in groupList:
            item.prepareGeometryChange()
            self.scene().destroyItemGroup(item)
        # reset grid
        self.grid.reset()
        self.selected = None
        self.grid.drawGrid()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()
        # if getattr(self.layer.parentImage, 'window', None) is not None:
        # self.layer.parentImage.window.repaint()  # TODO modified 28/11/18 validate

    def enableButtons(self):
        self.pushButton4.setEnabled(not self.graphicsScene.options['keep alpha'])

    def saveLUT(self):
        """

        """
        mainForm = self.mainForm
        lut = self.scene().lut
        lastDir = str(mainForm.settings.value('paths/dlg3DLUTdir', '.'))
        dlg = QFileDialog(mainForm, "Save Color LUT", lastDir)
        dlg.setNameFilter('*.cube')
        dlg.setDefaultSuffix('cube')
        try:
            if dlg.exec_():
                filenames = dlg.selectedFiles()
                newDir = dlg.directory().absolutePath()
                mainForm.settings.setValue('paths/dlg3DLUTdir', newDir)
                lut.writeToTextFile(filenames[0])
                dlgInfo('3D LUT written')
        except IOError as e:
            dlgWarn(str(e))

    def setMask(self):
        """
        Build the layer mask from the image alpha channel.
        The Image alpha channel is recorded
        in the red channel of the mask.
        """
        layer = self.graphicsScene.layer
        currentImg = layer.getCurrentImage()
        imgBuf = QImageBuffer(currentImg)
        # resize the alpha channel
        imgmask = cv2.resize(imgBuf[:, :, 3], (layer.width(), layer.height()))
        mask = QImageBuffer(layer.mask)
        mask[:, :, 2] = imgmask
        layer.applyToStack()
        layer.parentImage.onImageChanged()


    def writeToStream(self, outStream):
        layer = self.layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        outStream.writeInt32(self.graphicsScene.LUTSize)
        outStream.writeInt32(self.graphicsScene.lut.step)
        byteData = self.graphicsScene.lut.LUT3DArray.tostring()
        outStream.writeInt32(len(byteData))
        outStream.writeRawData(byteData)
        return outStream

    def readFromStream(self, inStream):
        actionName = inStream.readQString()
        name = inStream.readQString()
        size = inStream.readInt32()
        self.graphicsScene.LUTsize = size
        self.graphicsScene.lut.step = inStream.readInt32()
        l = inStream.readInt32()
        byteData = inStream.readRawData(l)
        self.graphicsScene.lut.LUT3DArray = np.fromstring(byteData, dtype=int).reshape((size, size, size, 3))
        return inStream


if __name__ == '__main__':
    size = 4000
    # random ints in range 0 <= x < 256
    b = np.random.randint(0,256, size=size*size*3, dtype=np.uint8)
    testImg = np.reshape(b, (size, size, 3))
    interpImg = interpTriLinear(LUT3DIdentity.LUT3DArray, LUT3DIdentity.step, testImg)
    d = testImg - interpImg
    print ("max deviation : ", np.max(np.abs(d)))
