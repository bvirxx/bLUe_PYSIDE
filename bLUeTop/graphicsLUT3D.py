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
from collections import OrderedDict

import cv2

import numpy as np

from PySide2.QtCore import QSize
from PySide2.QtWidgets import QAction, QFileDialog, QToolTip, QHBoxLayout,\
    QApplication, QGridLayout, QComboBox, QLineEdit
from PySide2.QtGui import QPainter, QPolygonF, QPainterPath, QPen, QBrush, QColor, QPixmap
from PySide2.QtWidgets import QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem,\
    QGraphicsPixmapItem, QGraphicsPolygonItem, QSizePolicy
from PySide2.QtCore import Qt, QPointF, QRect, QRectF
from PySide2.QtGui import QImage
from PySide2.QtWidgets import QMenu, QRubberBand

from bLUeCore.bLUeLUT3D import LUT3D
from bLUeCore.trilinear import interpTriLinear
from bLUeGui.colorCube import rgb2hsB, cmyk2rgb
from bLUeGui.graphicsForm import baseGraphicsForm
from bLUeTop.lutUtils import LUTSIZE, LUTSTEP, LUT3D_ORI, LUT3DIdentity
from bLUeGui.colorPatterns import hueSatPattern, brightnessPattern
from bLUeGui.bLUeImage import QImageBuffer
from bLUeTop.utils import optionsWidget, UDict, QbLUePushButton
from bLUeGui.dialog import dlgWarn, dlgInfo


#######################
# node neighborhood radius
spread = 1
######################


class index(object):
    """
    the class index is designed to handle sets of
    4D coordinates.
    An index object represents a 4-uple (p, i, j, k),
    i, j, k are indices in a 3D LUT and p is a brightness.
    A set of index objects contains unique (i, j, k) 3-uples
    """
    def __init__(self, p, i, j, k):
        self.p = p
        self.ind = (i, j, k)

    def __repr__(self):
        return "index(%f, %s)" % (self.p, self.ind)

    def __eq__(self, other):
        if isinstance(other, index):
            return self.ind == other.ind
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash("%d%d%d" % self.ind)


class nodeGroup(QGraphicsItemGroup):

    @classmethod
    def groupFromList(cls, items, grid=None, position=QPointF(), parent=None):
        """
        Inits a new group and adds all activeNode objects from list to the group.
        @param items: list of nodes
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

    def __init__(self, grid=None, position=QPointF(), parent=None):
        super().__init__(parent=parent)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.grid = grid
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.initialPosition = position
        self.setPos(position)
        # record the coordinates (relative to group)
        # of the last pressed point.
        self.pressedPoint = QPointF()

    def addToGroup(self, item):
        item.setSelected(False)  # TODO removed 9/1/20 validate
        super().addToGroup(item)
        print(item.pos())
        item.setPen(QPen(Qt.yellow))

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        self.mouseIsPressed = True
        self.pressedPoint = e.pos()
        # draw links to neighbors
        self.grid.drawTrace = True
        # update grid
        self.grid.drawGrid()

    def mouseMoveEvent(self, e):
        self.mouseIsMoved = True
        rect = self.scene().sceneRect()
        newPos = e.scenePos() - self.pressedPoint
        rect1 = self.boundingRect().translated(newPos)
        if not rect.contains(rect1):
            return
        self.setPos(newPos)
        # update grid
        self.grid.drawGrid()

    def mouseReleaseEvent(self, e):
        # move child nodes and synchronize LUT
        if self.mouseIsMoved:
            for i in self.childItems():
                i.syncLUT()
                i.setControl(True)  # i.isControl = True  # 28/10
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.grid.drawTrace = False
        self.grid.drawGrid()
        super().mouseReleaseEvent(e)

    def contextMenuEvent(self, event):
        menu = QMenu()
        # ungroup
        actionUnGroup = QAction('UnGroup', None)
        menu.addAction(actionUnGroup)

        def f1():
            for item in self.childItems():
                item.setPen(QPen(Qt.red if item.isControl() else Qt.black))
            self.scene().destroyItemGroup(self)
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
                i.syncLUT()
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
                i.syncLUT()
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
                i.syncLUT()
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
                i.syncLUT()
            # self.scene().onUpdateLUT(options=self.scene().options)
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        actionRotateCCW.triggered.connect(f5)

        menu.exec_(event.screenPos())


class activeNode(QGraphicsPathItem):
    """
    Grid node class.
    Each node owns a fixed color, depending on its initial position on the color wheel.
    The node is bound to a fixed list of LUT vertices, corresponding to its initial hue and sat values.
    When a node is moved over the color wheel, calling the method setState synchronizes
    the values of the LUT vertices with the current node position.
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
                n.syncLUT()
        grid = nodeList[0].grid
        grid.drawTrace = True
        grid.drawGrid()
        grid.drawTrace = False
        l = grid.scene().layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def __init__(self, position, cModel, gridRow=0, gridCol=0, parent=None, grid=None):
        """
        Each node holds the r,g,b and h,s,p values corresponding to its
        initial position on the color wheel.
        The attribute LUTIndices holds the list of LUT vertices
        matching h and s.
        @param position: node position (relative to parent item)
        @type position: QPointF
        @param cModel: color model converter
        @type cModel: cmConverter
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
        # add node to the scene
        self.setParentItem(parent)
        self.cModel = cModel
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.initialPos = position
        self.setPos(self.initialPos)
        self.__gridRow, self.__gridCol = gridRow, gridCol
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        # recording of  node state
        self.control = False  # aspect and smoothing
        self.visible = False
        self.setVisible(self.visible)
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
        # To build the list of LUT vertices bound to the node,
        # we convert the list of HSV coord. (self.hue, self.sat, V) to RGB values
        # and search for the nearest neighbor vertices in the 3D LUT.
        # BrgbBuf axis 0 is brightness
        tmp = scene.slider2D.QImg.BrgbBuf[:, p.toPoint().y(), p.toPoint().x()].astype(np.float)[:, None]
        self.LUTIndices = np.round(tmp[:, 0] / float(LUTSTEP)).astype(int)
        clipped = [(i, j, k) for i, j, k in self.LUTIndices if i < LUTSIZE - 2 and j < LUTSIZE - 2 and k < LUTSIZE - 2]
        clipped.extend([tuple(self.LUTIndices[len(clipped)])] if len(clipped) < len(self.LUTIndices) else [])
        # remove duplicate vertices
        self.LUTIndices = set([index(p/100.0, i, j, k) for p, (i, j, k) in enumerate(clipped)])
        # sort indices by increasing brightnesses
        self.LUTindices = sorted(self.LUTIndices, key=lambda x: x.p)
        # set path
        c = grid.size // 2
        if self.gridRow == c and self.gridCol == c:
            self.setPath(self.qppR)
        else:
            self.setPath(self.qppE)
        self.grid = grid
        self.initialPosition = position

    @property  # read only
    def gridRow(self):
        return self.__gridRow

    @property  # read only
    def gridCol(self):
        return self.__gridCol

    def mapGrid2Parent(self, point):
        """
        Maps grid coordinates to node parent coordinates.
        @param point:
        @type point: QPointF
        @return:
        @rtype: QPointF
        """
        p = self.parentItem()
        if p is self.grid:
            return point
        else:
            return point + self.grid.scenePos() - p.scenePos()

    def setVisible(self, b):
        """
        Toggles and records node visibility.
        Call the super method to toggle visibility without
        changing the recorded state.
        Note that isVisible() always returns the actual visibility which
        may differ from the recorded one. This is useful to toggle all nodes visibility
        on/off at once. Use isVisibleState() to get the recorded state.
        @param b:
        @type b: boolean
        """
        super().setVisible(b)
        self.visible = b  # remember

    def isVisibleState(self):
        return self.visible

    def setControl(self, b):
        self.control = b
        if type(self.parentItem()) is not nodeGroup:  # priority to group settings
            self.setPen(QPen(Qt.red if self.isControl() else Qt.black))

    def isControl(self):
        return self.control

    def isTrackable(self):
        return self.isSelected()  # or self.isControl

    def syncLUT(self):
        """
        Synchronize the LUT
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
        # LUTIndices contains unique (i,j,k) 3-uples, but using a spread >= 1 can induce
        # conflicts between translations applied to a node. As LUTIndices are sorted by increasing
        # brightnesses, these conflicts are solved by giving priority to the translation corresponding
        # to the highest brightness.
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
            trgNbghd[...] = nbghd + translat
            # alpha is set to 0 for LUT vertices bound to  non moved nodes
            # and to 255 otherwise, allowing to build a mask based on color
            # selection.
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
            return self.grid.gridNodes[self.gridRow-1][self.gridCol]
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
        Returns the list of grid neighbors (self excluded)
        @return: neighbors
        @rtype: list of activeNode objects
        """
        nghb = []
        if self.gridRow > 0:
            nghb.append(self.grid.gridNodes[self.gridRow-1][self.gridCol])
        if self.gridCol > 0:
            nghb.append(self.grid.gridNodes[self.gridRow][self.gridCol-1])
        if self.gridRow < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow+1][self.gridCol])
        if self.gridCol < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow][self.gridCol + 1])
        if self.gridRow > 0 and self.gridCol > 0:
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
        Returns the laplacian (mean) of the positions
        of the neighbor nodes.
        Coordinates are relative to the scene.
        @return:
        @rtype: QPointF
        """
        laplacian = QPointF(0.0, 0.0)
        count = 0
        for item in self.neighbors():
            laplacian += item.scenePos()
            count += 1
        return laplacian / count

    def mousePressEvent(self, e):
        self.mouseIsPressed = True
        super().mousePressEvent(e)
        # TODO added 4 lines 7/1/19 validate
        # draw links to neighbors
        self.grid.drawTrace = True
        # update grid
        self.grid.drawGrid()

    def mouseMoveEvent(self, e):
        if e.modifiers() == Qt.ControlModifier | Qt.AltModifier:
            self.mouseIsMoved = True
            # item is not in a group, because
            # groups do not send mouse move events to child items.
            rect = self.scene().sceneRect()
            newPos = e.scenePos()
            if not rect.contains(newPos):
                return
            self.setPos(newPos)
            # moved nodes are visible and selected
            self.setVisible(True)
            self.setSelected(True)
            # make sure the move trace is shown.
            self.grid.drawTrace = True
            self.grid.drawGrid()

    def mouseReleaseEvent(self, e):
        self.syncLUT()
        if self.mouseIsMoved:
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        # Ctrl+click
        elif e.modifiers() == Qt.ControlModifier:
            # toggle control flag
            self.setControl(not self.control)
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.grid.drawTrace = False
        self.grid.drawGrid()
        super().mouseReleaseEvent(e)

    def contextMenuEvent(self, event):
        menu = QMenu()
        actionGroup = QAction('Group', None)
        menu.addAction(actionGroup)
        actionGroup.triggered.connect(lambda: nodeGroup.groupFromList(self.scene().selectedItems(),
                                                                      grid=self.grid,
                                                                      position=self.scenePos(),
                                                                      parent=self.parentItem()))
        actionGroup.setEnabled(len(self.scene().selectedItems()) > 1)
        actionReset = QAction('Reset', None)
        menu.addAction(actionReset)
        actionReset.triggered.connect(lambda: activeNode.resetNodes(self.scene().selectedItems()))
        menu.exec_(event.screenPos())


class activeGrid(QGraphicsPathItem):
    """
    Grid of active nodes.
    It includes a positioning grid and
    a trace item showing node moves.
    """

    def __init__(self, size, cModel, parent=None):
        """

        @param size: grid size
        @type size: int
        @param cModel: color model converter
        @type cModel: cmConverter
        @param parent:
        @type parent:
        """
        super().__init__()
        self.setParentItem(parent)
        self.size = size
        # parent should be the color wheel. step is the unitary coordinate increment
        # between consecutive nodes in each direction
        self.step = parent.size / float((self.size - 1))
        # set pos (relative to parent)
        self.setPos(0, 0)
        # add nodes to the scene
        # np.fromiter does not handle dtype object, so we cannot use a generator
        self.gridNodes = [[activeNode(QPointF(i*self.step, j * self.step), cModel, gridRow=j,
                                      gridCol=i, parent=self, grid=self)
                                    for i in range(self.size)] for j in range(self.size)]
        # add an item for proper visualization of node moves
        self.traceGraphicsPathItem = QGraphicsPathItem(parent=parent)
        # set pens
        self.setPen(QPen(QColor(128, 128, 128)))  # , 1, Qt.DotLine, Qt.RoundCap))
        self.traceGraphicsPathItem.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine, Qt.RoundCap))
        # default : don't show trace of moves
        self.drawTrace = False
        self.drawGrid()
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setDefaultControlNodes()
        self.showAllNodes = False

    def setDefaultControlNodes(self):
        """
        set border nodes as control nodes
        """
        for i in range(self.size):
            self.gridNodes[i][0].setControl(True)
            self.gridNodes[i][self.size - 1].setControl(True)
            self.gridNodes[0][i].setControl(True)
            self.gridNodes[self.size - 1][i].setControl(True)

    def toggleAllNodes(self):
        """
        Toggles full grid display on and off.
        Nodes having the visible flag set are shown in the two modes;
        this allows a nice toggling.
        """
        self.showAllNodes = not self.showAllNodes
        for i in range(self.size):
            for j in range(self.size):
                super(activeNode, self.gridNodes[i][j]).setVisible(self.showAllNodes or self.gridNodes[i][j].visible)

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

    def reset(self, unSmoothOnly=False):
        """
        unselect and reset nodes to their initial position.
        If unSmoothOnly is False all nodes are reset,
        otherwise control points are not moved.
        """
        if not unSmoothOnly:
            # explode all node groups  # TODO moved from onReset  7/1/20 validate
            groupList = [item for item in self.childItems() if type(item) is nodeGroup]
            for item in groupList:
                item.prepareGeometryChange()
                self.scene().destroyItemGroup(item)
        for i in range(self.size):
            for j in range(self.size):
                node = self.gridNodes[i][j]
                if unSmoothOnly:
                    if node.control:
                        continue
                node.setPos(node.mapGrid2Parent(node.initialPosition)) # node may still belong to a group
                node.setSelected(False)
                node.setControl(False)  # TODO added 7/1/20 validate
                node.setVisible(False)  # TODO added 7/1/20 validate
        self.setDefaultControlNodes()
        self.showAllNodes = False

    def smooth(self):
        """
        Try to smooth the grid by moving each non control point to the position of the laplacian
        of its neighbors.
        """
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            # apply laplacian kernel
            for i in range(self.size):
                for j in range(self.size):
                    curnode = self.gridNodes[i][j]
                    newPos = curnode.laplacian()  # scene coordinates
                    if curnode.control:
                        continue
                    # update position
                    if curnode.scenePos() != newPos:
                        position = newPos - curnode.parentItem().scenePos()
                        curnode.setPos(position)  # parent coordinates
                        curnode.syncLUT()
            # LUT vertices are possibly bound to several nodes. Hence,
            # output color of vertices bound to a control point may have been changed
            # by the syncLUT() above. Uncomment to restore them now :
            """
            for i in range(self.size):
                for j in range(self.size):
                    if self.gridNodes[i][j].isControl:
                        self.gridNodes[i][j].syncLUT()
            """
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()

    def drawGrid(self):
        """
        Builds and sets the path of the grid.
        """
        step = 4
        qpp = QPainterPath()
        qppTrace = QPainterPath()
        for i in range(self.size):
            for j in range(self.size):
                node = self.gridNodes[i][j]
                if i % step == 0 and j % step == 0:
                    if i > 0:
                        qpp.moveTo(node.initialPos)  # gridPos())
                        qpp.lineTo(self.gridNodes[i-step][j].initialPos)    # gridPos())  # TODO modified 7/1/20
                    if j > 0:
                        qpp.moveTo(node.initialPos)  # gridPos())
                        qpp.lineTo(self.gridNodes[i][j-step].initialPos)  # gridPos())  # TODO modified 7/1/20
                if not node.isTrackable():  # node.isSelected():  # TODO changed 7/1/19 validate
                    continue
                # Visualize displacement for trackable nodes
                qppTrace.moveTo(node.gridPos())
                qppTrace.lineTo(node.initialPos)
                qppTrace.addEllipse(node.initialPos, 3, 3)
                if self.drawTrace:
                    for n in node.neighbors():
                        if n.isVisible():
                            qppTrace.moveTo(n.gridPos())
                            qppTrace.lineTo(node.gridPos())
        self.setPath(qpp)
        self.traceGraphicsPathItem.setPath(qppTrace)


class activeMarker(QGraphicsPolygonItem):
    """
    Movable marker
    """

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
        item.moveRange = item.parentItem().boundingRect()
        return item

    @classmethod
    def fromCross(cls, parent=None):
        color = QColor(0, 0, 0)
        item = activeMarker(parent=parent)
        item.setPolygon(cls.cross)
        item.setPen(QPen(color))
        item.setBrush(QBrush(color))
        # set move range to parent bounding rect
        item.moveRange = item.parentItem().boundingRect()
        return item

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.onMouseMove, self.onMouseRelease = lambda e, x, y: 0, lambda e, x, y: 0
        self.moveRange = QRectF(0.0, 0.0, 0.0, 0.0)

    @property  # read only
    def currentColor(self):
        return self.scene().slider2D.QImg.pixelColor((self.pos() - self.parentItem().offset()).toPoint())

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
            H, xedges, yedges = np.histogram2d(xyarray[:, :, 0].ravel(), xyarray[:, :, 1].ravel(),
                                               bins=[np.arange(0, maxVal+STEP, STEP), np.arange(0, maxVal + STEP, STEP)],
                                               normed=True)
            w, h = QImg.width(), QImg.height()
            b = QImage(w, h, QImage.Format_ARGB32)
            b.fill(0)
            buf = QImageBuffer(b)
            # set the transparency of each pixel
            # proportionally to the height of its bin
            # get bin indices (u, v)
            u = xyarray[:, :, 0] // STEP
            v = xyarray[:, :, 1] // STEP
            # get heights of bins
            tmp = H[u, v]
            norma = np.amax(H)
            # color white
            buf[xyarray[:, :, 1], xyarray[:, :, 0], ...] = 255
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
        self.rubberBand.setGeometry(QRect(self.origin, e.screenPos()).normalized())  # topLeft, bottomRight

    def mouseReleaseEvent(self, e):
        # rubberBand selection
        if e.button() == Qt.RightButton:
            return
        self.rubberBand.hide()
        grid = self.scene().grid
        screenOrigin = e.screenPos() - e.pos().toPoint()
        rubberRect = QRect(self.origin, e.screenPos()).normalized()  # topLeft, bottomRight
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
    3D LUT perceptual editor
    """
    selectBrush = QBrush(QColor(255, 255, 255))
    unselectBrush = QBrush()
    # default brightness
    defaultColorWheelBr = 0.60

    @classmethod
    def getNewWindow(cls, cModel, targetImage=None, axeSize=500, LUTSize=LUTSIZE, layer=None, parent=None, mainForm=None):
        """
        build a graphicsForm3DLUT object. The parameter size represents the size of
        the color wheel, border not included (the size of the window is adjusted).
        @param cModel: color Model converter
        @type cModel: cmConverter
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
        super().__init__(targetImage=targetImage, layer=layer, parent=parent)
        self.mainForm = mainForm  # used by saveLUT()
        # context help tag
        self.helpId = "LUT3DForm"
        self.cModel = cModel
        border = 20
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize + 90, axeSize + 250)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setBackgroundBrush(QBrush(Qt.black, Qt.SolidPattern))
        self.currentHue, self.currentSat, self.currentPb = 0, 0, self.defaultColorWheelBr
        self.currentR, self.currentG, self.currentB = 0, 0, 0
        self.size = axeSize
        # currently selected grid node
        self.selected = None
        # init LUT
        freshLUT3D = LUT3D(None, size=LUTSize, alpha=True)
        self.graphicsScene.lut = freshLUT3D
        # init 2D slider (color wheel)
        swatchImg = hueSatPattern(axeSize, axeSize, cModel, bright=self.defaultColorWheelBr, border=border)
        slider2D = colorChooser(cModel, swatchImg, target=self.targetImage, size=axeSize, border=border)
        #########################################################################
        # CAUTION : sliedr2D has a non null offset
        # slider2D (and QImg) topleft corner is at scene pos -slider2D.offset()
        #########################################################################
        self.graphicsScene.addItem(slider2D)
        self.graphicsScene.slider2D = slider2D
        ##############################
        # neutral and working markers
        ##############################
        offset = slider2D.offset()
        neutralMarker = activeMarker.fromCross(parent=slider2D)
        neutralMarker.setPos(swatchImg.width() / 2 + offset.x(), swatchImg.height() / 2 + offset.y())
        self.workingMarker = activeMarker.fromCross(parent=slider2D)
        # default pos: average skin tone
        pt = QPointF(*swatchImg.GetPoint(*rgb2hsB(*cmyk2rgb(6, 25, 30, 0))[:2])) + offset
        self.workingMarker.setPos(pt.x(), pt.y())
        self.workingMarker.onMouseMove = lambda e, x, y: self.displayStatus()

        # swatch event handler
        def f1(p, r, g, b):
            h, s, br = self.cModel.rgb2cm(r, g, b)
            self.currentHue, self.currentSat, self.currentPb = h, s, br
            self.currentR, self.currentG, self.currentB = r, g, b
            self.bSliderUpdate()
            self.displayStatus()
        self.graphicsScene.slider2D.onMouseRelease = f1

        # Brightness slider
        self.bSliderHeight = 20
        self.bSliderWidth = self.graphicsScene.slider2D.QImg.width()
        px = brightnessPattern(self.bSliderWidth, self.bSliderHeight, cModel, self.currentHue, self.currentSat).rPixmap
        self.graphicsScene.bSlider = QGraphicsPixmapItem(px, parent=self.graphicsScene.slider2D)
        self.graphicsScene.bSlider.setPos(QPointF(-border, self.graphicsScene.slider2D.QImg.height() - border))
        bSliderCursor = activeMarker.fromTriangle(parent=self.graphicsScene.bSlider)
        bSliderCursor.setMoveRange(QRectF(0.0, bSliderCursor.size, self.graphicsScene.bSlider.pixmap().width(), 0.0))
        bSliderCursor.setPos(self.graphicsScene.bSlider.pixmap().width() * self.defaultColorWheelBr, bSliderCursor.size)

        # bSlider event handlers
        def f2(e, p, q):
            self.currentPb = p / float(self.bSliderWidth)
            self.graphicsScene.slider2D.QImg.setPb(self.currentPb)
            self.graphicsScene.slider2D.setPixmap(self.graphicsScene.slider2D.QImg.rPixmap)
            QToolTip.showText(e.screenPos(), (str(int(self.currentPb * 100.0))), self)
            
        bSliderCursor.onMouseMove = f2

        def f3(e, p, q):
            self.currentPb = p / float(self.bSliderWidth)
            # self.graphicsScene.slider2D.QImg.setPb(self.currentPb)
            # self.graphicsScene.slider2D.setPixmap(self.graphicsScene.slider2D.QImg.rPixmap)
            self.displayStatus()

        bSliderCursor.onMouseRelease = f3

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
        pushButton31 = QbLUePushButton("Unsmooth Grid")
        pushButton31.clicked.connect(self.onUnsmoothGrid)
        pushButton4 = QbLUePushButton('Set Mask')
        pushButton4.clicked.connect(self.setMaskFromAlpha)
        # pushButton4 needs enabling/disabling
        self.pushButton4 = pushButton4
        pushButton5 = QbLUePushButton("Show/Hide All")
        pushButton5.clicked.connect(self.toggleAllNodes)

        # options
        options1, optionNames1 = ['use image', 'use selection'], ['Use Image', 'Use Selection']
        self.listWidget1 = optionsWidget(options=options1, optionNames=optionNames1, exclusive=True)

        self.listWidget1.setFocusPolicy(Qt.NoFocus)
        # set initial selection to 'use image'
        self.listWidget1.checkOption(self.listWidget1.intNames[0])

        options2, optionNames2 = ['add node', 'remove node'], ['Add Node', 'Remove Node']
        self.listWidget2 = optionsWidget(options=options2, optionNames=optionNames2, exclusive=True)

        # set initial selection to 'add node'
        self.listWidget2.checkOption(self.listWidget2.intNames[0])

        options3 = ['select neighbors', 'reset removed nodes', 'show histogram', 'keep alpha']
        optionNames3 = ['Select Neighbors', 'Reset Removed', 'Show Histogram', 'Keep Alpha']
        self.listWidget3 = optionsWidget(options=options3, optionNames=optionNames3, exclusive=False)
        self.listWidget3.checkOption(self.listWidget3.intNames[0])
        self.listWidget3.checkOption(self.listWidget3.intNames[1])

        def onSelect3(item):
            option = item.internalName
            if option == 'show histogram':
                self.graphicsScene.slider2D.showTargetHist = self.graphicsScene.options[option]
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

        for wdg in [self.listWidget1, self.listWidget2, self.listWidget3]:
            wdg.setMinimumWidth(wdg.sizeHintForColumn(0))
            wdg.setMinimumHeight(wdg.sizeHintForRow(0)*len(wdg.items))

        # color format combo
        infoCombo = QComboBox()
        oDict = OrderedDict([('Marker RGB', 0), ('Marker CMYK', 1), ('Marker HSV', 2)])
        for key in oDict:
            infoCombo.addItem(key, oDict[key])

        def colorInfoFormatChanged(value):
            self.colorInfoFormat = infoCombo.itemData(value)
            self.displayStatus()

        infoCombo.currentIndexChanged.connect(colorInfoFormatChanged)

        # working marker position editor
        self.info = QLineEdit()
        self.info.setMaximumSize(120, 25)

        # returnPressed slot
        def infoDone():
            try:
                token = self.info.text().split(' ')
                if self.colorInfoFormat == 0:  # RGB
                    r, g, b = [int(x) for x in token if x.isdigit()]
                    pt = QPointF(*swatchImg.GetPoint(*rgb2hsB(r, g, b)[:2])) + offset
                elif self.colorInfoFormat == 1:  # CMYK
                    c, m, y, k = [int(x) for x in token if x.isdigit()]
                    pt = QPointF(*swatchImg.GetPoint(*rgb2hsB(*cmyk2rgb(c, m, y, k))[:2])) + offset
                elif self.colorInfoFormat == 2:  # HSV
                    h, s, _ = [int(x) for x in token if x.isdigit()]
                    if not 0 <= s <= 100:
                        raise ValueError
                    pt = QPointF(*swatchImg.GetPoint(h, s / 100.0)) + offset
                else:
                    raise ValueError
                self.workingMarker.setPos(pt.x(), pt.y())
            except ValueError:
                dlgWarn("Invalid color")

        self.info.returnPressed.connect(infoDone)

        # layout
        gl = QGridLayout()
        for i, button in enumerate([pushButton1, pushButton3, pushButton31, pushButton2]):
            gl.addWidget(button, 0, i)
        gl.addWidget(pushButton4, 1, 0)
        gl.addWidget(pushButton5, 1, 1)
        for i, widget in enumerate([self.listWidget1, self.listWidget2, self.listWidget3]):
            gl.addWidget(widget, 2 if i < 2 else 1, i, 1 if i < 2 else 2, 1)
        hl = QHBoxLayout()
        hl.addWidget(infoCombo)
        hl.addWidget(self.info)
        gl.addLayout(hl, 3, 0, -1, -1)
        self.addCommandLayout(gl)

        # set defaults
        self.colorInfoFormat = 0  # RGB
        colorInfoFormatChanged(self.colorInfoFormat)

        # whatsthis
        self.setWhatsThis(
                        """ <b>2.5D LUT Perceptual Editor</b><br>
                          <b>Select nodes</b> with mouse clicks on the image. Selected nodes are shown
                        as small black circles on the color wheel. Each node corresponds to a set of colors 
                        sharing the same hue and saturation.<br>
                        <b>Modify the color</b> of a node by dragging it on
                        the wheel. Several nodes can be moved simultaneously by grouping them.<br>
                        <b>Group nodes</b> :<br>
                                &nbsp; 1 - group nodes with the mouse : while pressing the mouse left button,
                                           drag a rubber band around the nodes to group;<br>
                                &nbsp; 2 - next, right click any one of the selected nodes and 
                                choose group from the context menu<br>
                        <b>Unselect nodes</b> :<br>
                                &nbsp; 1 - check the option Remove Node;<br>
                                &nbsp; 2 -  ungroup;<br>
                                &nbsp; 3 - on the image, click the pixels to unselect.<br>
                        <b>Warning</b> : Selecting/unselecting nodes with the mouse is enabled only when
                        the Color Chooser is closed.<br>
                        Press the <b> Smooth Grid</b> button to smooth color transitions between neighbor nodes.<br>
                        <b>Crosshair</b> markers indicate neutral gray tones and average 
                        skin tones. They can be moved with the mouse.
                        The position of the second marker is reflected in the editable 
                        bottom right box. Due to inherent properties
                        of the CMYK color model, CMYK input values may be modified silently.<br>
                        <b>The grid can be zoomed</b> with the mouse wheel.<br>
                        Check the <b>Keep Alpha</b> option to forward the alpha channel without modifications.<br>
                        This option must be unchecked to be able to build a mask from the selection.<br>
                        HSpB layer is slower than HSV, but usually gives better results.<br>    
                        """
                          )  # end of setWhatsThis

    def selectGridNode(self, r, g, b):
        """
        selects the nearest grid nodes corresponding to r,g,b values.
        @param r: color
        @param g: color
        @param b: color
        """
        h, s, p = self.cModel.rgb2cm(r, g, b)
        # currently selected values in adjust layer
        self.currentHue, self.currentSat, self.currentPb = h, s, p
        self.currentR, self.currentG, self.currentB = r, g, b
        xc, yc = self.graphicsScene.slider2D.QImg.GetPoint(h, s)

        # unitary coordinate increment between consecutive nodes
        step = float(self.grid.step)
        border = self.graphicsScene.slider2D.border
        # grid coordinates
        xcGrid, ycGrid = xc - border, yc - border
        NNN = self.grid.gridNodes[int(np.floor(ycGrid / step))][int(np.floor(xcGrid / step))]

        # select and mark selected node
        mode = self.graphicsScene.options['add node']
        # reset previous selected marker
        if self.selected is not None:
            self.selected.setSelected(False)
            # self.selected.setPath(activeNode.qppE)  #  TODO removed 7/1/20 validate
            self.selected.setBrush(self.unselectBrush)
        self.selected = NNN
        self.selected.setSelected(True)
        self.selected.setBrush(self.selectBrush)
        # self.selected.setPath(activeNode.qppR)  #  TODO removed 7/1/20 validate
        self.selected.setVisible(mode)
        nodesToSelect = NNN.neighbors() + [NNN] if self.graphicsScene.options['select neighbors'] else [NNN]  # TODO + [NNN] added 7/1/20
        for n in nodesToSelect:
            n.setVisible(mode)
            if self.graphicsScene.options['reset removed nodes'] and not mode:
                if isinstance(n.parentItem(), nodeGroup):
                    group = n.parentItem()
                    group.removeFromGroup(n)
                    if not group.childItems():
                        self.graphicsScene.destroyItemGroup(group)
                n.setPos(n.initialPos)
                n.syncLUT()
        self.grid.drawGrid()  # TODO added 7/1/20 validate
        # update status
        self.onSelectGridNode(h, s)

    def toggleAllNodes(self):
        self.grid.toggleAllNodes()

    def displayStatus(self):
        """
        Display an editable view of the working marker colors
        """
        color = self.workingMarker.currentColor
        if self.colorInfoFormat == 0:  # RGB
            r, g, b = color.red(), color.green(), color.blue()
            self.info.setText('%d  %d  %d' % (r, g, b))
        elif self.colorInfoFormat == 1:  # CMYK
            conv = 100.0 / 255.0
            c, m, y, k = color.cyan(), color.magenta(), color.yellow(), color.black()
            self.info.setText('%d  %d  %d  %d' % (int(c * conv), int(m * conv), int(y * conv), int(k * conv)))
        elif self.colorInfoFormat == 2:  # HSV
            h, s, v = color.hue(), color.hsvSaturation(), color.value()
            conv = 100.0 / 255.0
            self.info.setText('%d  %d  %d' % (h, int(s * conv), int(v * conv)))

    def bSliderUpdate(self):
        px = brightnessPattern(self.bSliderWidth, self.bSliderHeight, self.cModel, self.currentHue, self.currentSat).rPixmap
        self.graphicsScene.bSlider.setPixmap(px)

    def onSelectGridNode(self, h, s):
        self.bSliderUpdate()
        # self.displayStatus()

    def onSmoothGrid(self):
        """
        Button slot
        """
        self.grid.smooth()
        self.grid.drawGrid()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def onUnsmoothGrid(self):
        """
        Button slot
        """
        self.grid.reset(unSmoothOnly=True)
        self.grid.drawGrid()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def onReset(self):
        """
        reset grid and LUT
        """
        # get a fresh LUT
        self.graphicsScene.lut = LUT3D(None, size=self.graphicsScene.lut.size, alpha=True)
        """
        # explode all node groups  moved to reset()
        groupList = [item for item in self.grid.childItems() if type(item) is nodeGroup]
        for item in groupList:
            item.prepareGeometryChange()
            self.scene().destroyItemGroup(item)
        """
        # reset grid
        self.grid.reset()
        self.selected = None
        self.grid.drawGrid()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

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

    def setMaskFromAlpha(self):
        """
        Sets the layer mask from the image alpha channel:
        the image alpha channel is recorded
        in the red channel of the mask.
        """
        layer = self.graphicsScene.layer
        currentImg = layer.getCurrentImage()
        imgBuf = QImageBuffer(currentImg)
        # resize the alpha channel
        imgmask = cv2.resize(imgBuf[:, :, 3], (layer.width(), layer.height()))
        layer.historyListMask.addItem(layer.mask.copy())
        mask = QImageBuffer(layer.mask)
        mask[:, :, 2] = imgmask
        layer.applyToStack()
        layer.parentImage.onImageChanged()


if __name__ == '__main__':
    size = 4000
    # random ints in range 0 <= x < 256
    b = np.random.randint(0, 256, size=size*size*3, dtype=np.uint8)
    testImg = np.reshape(b, (size, size, 3))
    interpImg = interpTriLinear(LUT3DIdentity.LUT3DArray, LUT3DIdentity.step, testImg)
    d = testImg - interpImg
    print("max deviation : ", np.max(np.abs(d)))
