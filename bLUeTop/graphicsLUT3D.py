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
import threading
from collections import OrderedDict

import cv2

import numpy as np

from PySide2.QtCore import QSize, QObject
from PySide2.QtWidgets import QAction, QFileDialog, QToolTip, QHBoxLayout, \
    QApplication, QGridLayout, QComboBox, QLineEdit, QLabel, \
    QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem, \
    QGraphicsPixmapItem, QSizePolicy
from PySide2.QtGui import QPainter, QPolygonF, QPainterPath, QPen, QBrush, QColor, QPixmap, QTransform
from PySide2.QtCore import Qt, QPointF, QRect, QRectF
from PySide2.QtGui import QImage
from PySide2.QtWidgets import QMenu, QRubberBand

from bLUeCore.bLUeLUT3D import LUT3D
from bLUeCore.trilinear import interpTriLinear
from bLUeGui.colorCube import rgb2hsB, cmyk2rgb
from bLUeGui.graphicsForm import baseGraphicsForm
from bLUeGui.graphicsSpline import graphicsThrSplineItem, activeMarker
from bLUeTop.lutUtils import LUTSIZE, LUTSTEP, LUT3D_ORI, LUT3DIdentity
from bLUeGui.colorPatterns import hueSatPattern, brightnessPattern
from bLUeGui.bLUeImage import QImageBuffer
from bLUeTop.utils import optionsWidget, UDict, QbLUePushButton
from bLUeGui.dialog import dlgWarn, dlgInfo

#######################
# minimum radius of LUT vertex  neighborhood
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


class nodeGroup(QGraphicsItemGroup, QObject):  # QObject needed by disconnect()
    UID = 0

    @classmethod
    def groupFromList(cls, items, grid=None, position=QPointF(), parent=None):
        """
        Inits a new group and adds all activeNode objects from list to the group.

        :param items: list of nodes
        :type items: list
        :param grid:
        :type grid: activeGrid
        :param position: group position
        :type position: QPointF
        :param parent:
        :return: group
        :rtype: nodeGroup
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
        self.uid = nodeGroup.UID  # unique identifier
        nodeGroup.UID += 1  # not thread safe !
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.grid = grid
        self.mouseIsPressed = False
        self.mouseIsMoved = False
        self.initialPosition = position
        self.setPos(position)
        # record the coordinates (relative to group)
        # of the last pressed point.
        self.pressedPoint = QPointF()

        # brightness item
        graphicsForm = self.scene().views()[0]
        axeSize = graphicsForm.size
        border = graphicsForm.border
        offset = self.scene().slider2D.offset()
        self.brightnessItem = graphicsThrSplineItem(size=axeSize, border=border)
        # self.brightnessItem.setAcceptedMouseButtons(Qt.NoButton)  # does not work : mouse clicks are not ignored
        self.scene().addItem(self.brightnessItem)
        self.brightnessItem.setPos(offset.x(), axeSize + 50)
        self.brightnessItem.brightnessThr0.installSceneEventFilter(self.brightnessItem.brightnessThr1)
        self.brightnessItem.brightnessThr1.installSceneEventFilter(self.brightnessItem.brightnessThr0)
        # group brightness curve
        self.normalizedLUTXY = np.arange(256, dtype=np.float) / 255.0

        def f8():
            try:  # safety first : sometimes, disconnecting seems problematic
                self.brightnessItem.setVisible(self.isSelected())
            except RuntimeError:
                print('nodeGroup : calling method of a destroyed group')

        self.scene().selectionChanged.connect(f8)  # don't forget to disconnect before destroying group !

        def f5(e, x, y):
            self.brightnessItem.brightnessThr0.val = self.brightnessItem.brightnessThr0.pos().x() / self.brightnessItem.brightnessSliderWidth
            for i in self.childItems():
                i.syncLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        def f6(e, x, y):
            self.brightnessItem.brightnessThr1.val = self.brightnessItem.brightnessThr1.pos().x() / self.brightnessItem.brightnessSliderWidth
            for i in self.childItems():
                i.syncLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        def f7():
            self.normalizedLUTXY[...] = self.brightnessItem.cubic.getLUTXY() / 255.0
            for i in self.childItems():
                i.syncLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        self.brightnessItem.brightnessThr0.onMouseRelease, self.brightnessItem.brightnessThr1.onMouseRelease = f5, f6
        self.brightnessItem.cubic.curveChanged.sig.connect(f7)  # don't forget to disconnect before destroying group !

        self.brightnessItem.setVisible(False)

    def addToGroup(self, item):
        item.setSelected(False)
        super().addToGroup(item)
        item.setPen(QPen(Qt.yellow))

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        self.mouseIsPressed = True
        # initial mouse position is used in mouseMoveEvent()
        # for smoother moves
        self.__mouseEventOffset = e.pos()
        # draw links to neighbors
        self.grid.drawTrace = True
        # update grid
        self.grid.drawGrid()

    def mouseMoveEvent(self, e):
        self.mouseIsMoved = True
        rect = self.scene().sceneRect()
        # get mouse event position relative to (untransformed) grid
        newPos = - self.__mouseEventOffset + e.pos() + self.pos()
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

        # brightness settings
        actionBright = QAction('Brightness settings', None)
        menu.addAction(actionBright)

        def f6():
            self.brightnessItem.setVisible(True)

        actionBright.triggered.connect(f6)

        actionSync = QAction('Resync', None)
        menu.addAction(actionSync)

        def f7():
            for i in self.childItems():
                i.syncLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        actionSync.triggered.connect(f7)

        # ungroup
        actionUnGroup = QAction('UnGroup', None)
        menu.addAction(actionUnGroup)

        def f1():
            childs = self.childItems()
            # disconnect all signals received by node group
            try:
                self.scene().disconnect(self)
                self.brightnessItem.cubic.curveChanged.disconnect(self)
            except RuntimeError:
                pass
            self.scene().removeItem(self.brightnessItem)
            self.scene().destroyItemGroup(self)
            for item in childs:
                item.setPen(QPen(Qt.red if item.isControl() else Qt.black))
                item.setSelected(True)
            self.grid.drawGrid()

        actionUnGroup.triggered.connect(f1)

        # reset
        actionReset = QAction('Reset', None)
        menu.addAction(actionReset)

        def f0():
            self.setPos(self.initialPosition)
            for i in self.childItems():
                i.syncLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        actionReset.triggered.connect(f0)

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
    A LUT vertex may be bound to several (initially neighboring) grid nodes, thereby allowing to
    express antagonist constraints where nodes are moved away from each other. For the moment
    we solve conflictings moves on a "last sync first served" basis. activeNode and nodeGroup context menus
    provide Resync action to modify sync order.
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
        grid = None
        for n in nodeList:
            if type(n) is activeNode:
                n.setPos(n.initialPos)
                n.syncLUT()
                grid = n.grid
        if grid is None:
            # not any node selected
            return
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

        :param position: node position (relative to parent item)
        :type position: QPointF
        :param cModel: color model converter
        :type cModel: cmConverter
        :param gridRow:
        :type gridRow:
        :param gridCol:
        :type gridCol:
        :param parent: parent item
        :type parent:
        :param grid: owner grid
        :type grid:
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
        self.setAcceptHoverEvents(True)
        # recording of  node state
        self.control = False  # aspect and smoothing
        self.visible = False
        self.setVisible(self.visible)
        # read color from the color wheel.
        # Node parent is the grid, and its grandfather is the color wheel
        # grid is at pos (0,0) on the color wheel (colorPicker object)
        # Color wheel has a non null offset for the border.
        p = (position - parent.parentItem().offset()).toPoint()
        scene = parent.scene()
        c = QColor(scene.slider2D.QImg.pixel(p))
        # node color (LUT input)
        self.r, self.g, self.b, _ = c.getRgb()
        self.hue, self.sat, self.pB = self.cModel.rgb2cm(self.r, self.g, self.b)
        # modified colors (LUT output)
        self.rM, self.gM, self.bM = self.r, self.g, self.b
        # To build the list of LUT vertices bound to the node,
        # we use the table BrgbBuf to convert the list of HSV coord. (self.hue, self.sat, V)
        # with V in range(0, 101) into RGB values R, G, B.
        # For each brightness V, self.LUTIndices[V] is the the nearest neighbor vertex
        # of (R, G, B) in the 3D LUT.
        # (BrgbBuf axis 0 is V, axis 3 is color)
        tmp = scene.slider2D.QImg.BrgbBuf[:, p.y(), p.x()].astype(np.float)  # shape 101, 3
        tmp /= float(LUTSTEP)
        np.round(tmp, out=tmp)
        self.LUTIndices = tmp.astype(int)
        """
        clipped = [(i, j, k) for i, j, k in self.LUTIndices if i < LUTSIZE - 2 and j < LUTSIZE - 2 and k < LUTSIZE - 2]
        clipped.extend([tuple(self.LUTIndices[len(clipped)])] if len(clipped) < len(self.LUTIndices) else [])
        self.LUTIndices = set([index(p/100.0, i, j, k) for p, (i, j, k) in enumerate(clipped)])
        """
        # remove duplicate vertices
        self.LUTIndices = set([index(p / 100.0, *self.LUTIndices[p]) for p in range(0, 101)])
        # sort indices by increasing brightnesses
        self.LUTIndices = sorted(self.LUTIndices, key=lambda x: x.p)
        # set path
        c = grid.size // 2
        if self.gridRow == c and self.gridCol == c:
            self.setPath(self.qppR)
        else:
            self.setPath(self.qppE)
        self.grid = grid
        self.initialPosition = position
        self.vertexDict = None

    @property  # read only
    def gridRow(self):
        return self.__gridRow

    @property  # read only
    def gridCol(self):
        return self.__gridCol

    def mapGrid2Parent(self, point):
        """
        Maps grid coordinates to node parent coordinates.

        :param point:
        :type point: QPointF
        :return:
        :rtype: QPointF
        """
        p = self.parentItem()
        if p is self.grid:
            return point
        else:
            return point + self.grid.scenePos() - p.scenePos()

    def hasMoved(self):
        """
        Test for initial position, no matter if node belongs to a group.

        :return:
        :rtype: boolean
        """
        return self.pos() != self.mapGrid2Parent(self.initialPosition)

    def setVisible(self, b):
        """
        Toggles and records node visibility.
        Call the super method to toggle visibility without
        changing the recorded state.
        Note that isVisible() always returns the actual visibility which
        may differ from the recorded one. This is useful to toggle all nodes visibility
        on/off at once. Use isVisibleState() to get the recorded state.

        :param b:
        :type b: boolean
        """
        super().setVisible(b)
        self.visible = b  # remember

    def isVisibleState(self):
        return self.visible

    def setControl(self, b, forceEdges=False):
        # don't change state of grid edges
        border = [0, self.grid.size - 1]
        if not forceEdges:
            if self.gridRow in border or self.gridCol in border:
                return
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
        # get (transformation aware) position relative to grid parentItem (=colorChooser)
        p = (self.gridPos() - self.grid.parentItem().offset()).toPoint()
        x, y = p.x(), p.y()

        # update move history
        self.grid.updateHistoryListMove(self)

        # clipping
        x, y = min(w - 1, max(0, x)), min(h - 1, max(0, y))
        # read current color
        c = QColor(img.pixel(x, y))
        self.rM, self.gM, self.bM, _ = c.getRgb()
        hue, sat, _ = self.cModel.rgb2cm(self.rM, self.gM, self.bM)
        # update the LUT vertices bound to the node.
        # A neighborhood of the vertex is built and the corresponding values
        # in the LUT are shifted by the same vector, defined by the position of the
        # node on the color wheel. The transformation keeps hue and saturation.
        # LUTIndices contains unique (i,j,k) 3-uples, but using a spread >= 1 can induce
        # conflicts between translations applied to a node. As LUTIndices are sorted by increasing
        # brightnesses, these conflicts are solved by giving priority to the translation corresponding
        # to the highest brightness.
        lut = self.scene().lut.LUT3DArray
        prt = self.parentItem()
        isGrouped = False
        if type(prt) is nodeGroup:
            isGrouped = True
            # get group brightness curve : array shape (1, 256), dtype= float values [0..1]
            brLUT = prt.normalizedLUTXY
            thr0, thr1 = prt.brightnessItem.brightnessThr0.val, prt.brightnessItem.brightnessThr1.val
        else:
            thr0, thr1 = 0.0, 1.0
        for x in self.LUTIndices:
            p = x.p
            final_p = brLUT[int(p * 255.0)] if isGrouped else p
            i, j, k = x.ind
            slc1 = slice(max(k - spread, 0), k + spread + 1)
            slc2 = slice(max(j - spread, 0), j + spread + 1)
            slc3 = slice(max(i - spread, 0), i + spread + 1)
            nbghd = LUT3D_ORI[slc1, slc2, slc3, ::-1]
            nbghd1 = lut[slc1, slc2, slc3, :]
            translat = self.cModel.cm2rgb(hue, sat, final_p) - LUT3D_ORI[k, j, i, ::-1]
            trgNbghd = lut[slc1, slc2, slc3, :3][..., ::-1]
            trgNbghd[...] = nbghd
            if p >= thr0 and p <= thr1:
                trgNbghd += translat
            # alpha is set to 0 for LUT vertices bound to  non moved nodes
            # and to 255 otherwise, allowing to build a mask based on color
            # selection.
            if np.all(translat == 0):
                nbghd1[..., 3] = 0
            else:
                nbghd1[..., 3] = 255

    def gridPos(self):
        """
        Returns the current node position, relative to the grid.
        In contrast with self.pos(), it works for
        individual nodes and grouped nodes.
        The current grid transformation is applied.

        :return: position
        :rtype: QpointF
        """
        trans = self.grid.transform()
        if type(self.parentItem()) is not nodeGroup:
            if trans.isIdentity():
                return self.pos()
            else:
                return trans.map(self.pos())
        else:
            p = self.pos() + self.parentItem().pos()
            if trans.isIdentity():
                return p
            else:
                return trans.map(p)

    def gridPos_U(self):
        """
        Returns the current node position, relative to the grid.
        In contrast with self.pos(), it works for
        individual nodes and grouped nodes.
        The grid transformation is NOT applied.

        :return: position relative to grid axes.
        :rtype: QPointF
        """
        t = type(self.parentItem())
        return self.pos() + self.parentItem().pos() if t is nodeGroup else self.pos()

    def top(self):
        """
        Return the north neighbor.

        :return:
        :rtype:
        """
        if self.gridRow > 0:
            return self.grid.gridNodes[self.gridRow - 1][self.gridCol]
        return None

    def left(self):
        """
        Return the east neighbor.

        :return:
        :rtype:
        """
        if self.gridCol > 0:
            return self.grid.gridNodes[self.gridRow][self.gridCol - 1]
        return None

    def neighbors(self):
        """
        Returns the list of grid neighbors (self excluded).

        :return: neighbors
        :rtype: list of activeNode objects
        """
        nghb = []
        if self.gridRow > 0:
            nghb.append(self.grid.gridNodes[self.gridRow - 1][self.gridCol])
        if self.gridCol > 0:
            nghb.append(self.grid.gridNodes[self.gridRow][self.gridCol - 1])
        if self.gridRow < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow + 1][self.gridCol])
        if self.gridCol < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow][self.gridCol + 1])
        if self.gridRow > 0 and self.gridCol > 0:
            nghb.append(self.grid.gridNodes[self.gridRow - 1][self.gridCol - 1])
        if self.gridRow > 0 and self.gridCol < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow - 1][self.gridCol + 1])
        if self.gridRow < self.grid.size - 1 and self.gridCol < self.grid.size - 1:
            nghb.append(self.grid.gridNodes[self.gridRow + 1][self.gridCol + 1])
        if self.gridRow < self.grid.size - 1 and self.gridCol > 0:
            nghb.append(self.grid.gridNodes[self.gridRow + 1][self.gridCol - 1])
        return nghb

    def laplacian(self):
        """
        Returns the laplacian (mean) of the positions
        of the neighbor nodes.
        Coordinates are relative to the grid.

        :return:
        :rtype: QPointF
        """
        laplacian = QPointF(0.0, 0.0)
        count = 0
        for item in self.neighbors():
            laplacian += item.gridPos_U()  # scenePos()
            count += 1
        return laplacian / count

    def mousePressEvent(self, e):
        # initial mouse position is used in mouseMoveEvent()
        # instead of current position for smoother moves
        self.mouseIsPressed = True
        self.mouseIsMoved = False
        super().mousePressEvent(e)
        # draw links to neighbors
        # self.grid.drawTrace = True
        # update grid
        self.grid.drawGrid()
        # starting grid warping :
        # build dict of transformed grid vertex coordinates
        if e.modifiers() == Qt.ControlModifier | Qt.AltModifier:
            trans = self.grid.transform()
            size = self.grid.size
            self.vertexDict = {(r, c): trans.map(self.grid.gridNodes[r][c].pos()) for (c, r) in
                               [(0, 0), (size - 1, 0), (size - 1, size - 1), (0, size - 1)]}
        self.startingRect = self.scene().sceneRect()

    def mouseMoveEvent(self, e):
        # item is not in a group, because
        # groups do not send mouse move events to child items,
        # thus its parent is the grid
        # get mouse position relative to (untransformed) grid
        newPos = e.pos() + self.pos()
        if e.modifiers() == Qt.ControlModifier:
            if not self.startingRect.contains(newPos):
                return
            self.setPos(newPos)
            # moved nodes are visible and selected
            self.setVisible(True)
            self.setSelected(True)
            self.grid.drawGrid()
        # grid warping
        elif e.modifiers() == Qt.ControlModifier | Qt.AltModifier:
            row, col = self.gridRow, self.gridCol
            s = self.grid.size - 1
            # Warping is led by corner vertices only
            if row not in [0, s] or col not in [0, s] or (not self.startingRect.contains(newPos)):
                return
            # update vertexDict
            if self.vertexDict is None:
                return
            self.vertexDict[(row, col)] = self.grid.transform().map(newPos)  # = e.scenePos()
            # don't move to a degenerated quad
            thr = 0.2
            for r in (0, s):
                for c in (0, s):
                    V1, V2 = self.vertexDict[(r, c)] - self.vertexDict[(s - r, c)], self.vertexDict[(r, c)] - \
                             self.vertexDict[(r, s - c)]
                    a = np.array([[V1.x(), V1.y()], [V2.x(), V2.y()]])
                    if abs(np.linalg.det(a)) < thr * abs((V1.x() * V2.x() + V1.y() * V2.y())):
                        return
            # build target quad
            quad1 = QPolygonF()
            for (c, r) in [(0, 0), (s, 0), (s, s), (0, s)]:
                quad1.append(self.vertexDict[(r, c)])
            finalTrans = QTransform.quadToQuad(self.grid.initialQuad, quad1)
            if finalTrans is None:
                return
            self.grid.setTransform(finalTrans)
        self.mouseIsMoved = True

    def mouseReleaseEvent(self, e):
        # self.grid.drawTrace = False
        self.mouseIsPressed = False
        super().mouseReleaseEvent(e)
        if not self.mouseIsMoved:
            return
        self.grid.drawGrid()
        if e.modifiers() == Qt.ControlModifier:
            self.syncLUT()
        # grid warping: update all nodes
        if e.modifiers() == Qt.ControlModifier | Qt.AltModifier:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.grid.gridSyncLUT()
            finally:
                QApplication.restoreOverrideCursor()
        l = self.scene().layer
        l.applyToStack()
        l.parentImage.onImageChanged()
        self.mouseIsMoved = False
        self.vertexDict = None

    def hoverEnterEvent(self, e):
        self.setTransform(QTransform().scale(2.0, 2.0))

    def hoverLeaveEvent(self, e):
        self.setTransform(QTransform())

    def contextMenuEvent(self, event):
        menu = QMenu()
        actionControl = QAction('Control', None)
        actionControl.setCheckable(True)
        actionControl.setChecked(self.isControl())
        menu.addAction(actionControl)
        actionControl.toggled.connect(lambda b: self.setControl(b))
        actionGroup = QAction('Group', None)
        menu.addAction(actionGroup)
        actionGroup.triggered.connect(lambda: nodeGroup.groupFromList(self.scene().selectedItems(),
                                                                      grid=self.grid,
                                                                      position=self.pos(),
                                                                      # scenePos(),
                                                                      parent=self.parentItem()))
        actionGroup.setEnabled(len(self.scene().selectedItems()) > 1)
        actionSync = QAction('Resync', None)
        menu.addAction(actionSync)

        def f1():
            self.syncLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        actionSync.triggered.connect(f1)

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

       :param size: grid size
       :type size: int
       :param cModel: color model converter
       :type cModel: cmConverter
       :param parent:
       :type parent:
        """
        super().__init__()
        self.setParentItem(parent)
        self.size = size
        self.cModel = cModel
        self.historyListMove = []

        # parent should be the color wheel. step is the unitary coordinate increment
        # between consecutive nodes in each direction
        self.step = parent.size / float((self.size - 1))
        # set pos (relative to parent)
        self.setPos(0, 0)
        # add nodes to the scene
        # np.fromiter does not handle dtype object, so we cannot use a generator
        self.gridNodes = [[activeNode(QPointF(i * self.step, j * self.step), cModel, gridRow=j,
                                      gridCol=i, parent=self, grid=self)
                           for i in range(self.size)] for j in range(self.size)]
        # add an item for proper visualization of node moves
        self.traceGraphicsPathItem = QGraphicsPathItem()
        self.traceGraphicsPathItem.setParentItem(self)
        # set pens
        self.setPen(QPen(QColor(128, 128, 128)))  # , 1, Qt.DotLine, Qt.RoundCap))
        self.traceGraphicsPathItem.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine, Qt.RoundCap))
        # default : don't show trace of moves
        self.drawTrace = False
        self.drawGrid()
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setDefaultControlNodes()
        self.showAllNodes = False
        # build a quad from the 4 vertices of the grid in clockwise
        # order, starting at (0,0).
        w = self.gridNodes[size - 1][0].pos().y()
        self.initialQuad = QPolygonF()
        for p in [QPointF(0, 0), QPointF(w, 0), QPointF(w, w), QPointF(0, w)]:
            self.initialQuad.append(p)

    def setDefaultControlNodes(self):
        """
        set edge nodes as control nodes
        """
        for i in range(self.size):
            self.gridNodes[i][0].setControl(True, forceEdges=True)
            self.gridNodes[i][self.size - 1].setControl(True, forceEdges=True)
            self.gridNodes[0][i].setControl(True, forceEdges=True)
            self.gridNodes[self.size - 1][i].setControl(True, forceEdges=True)

    def updateHistoryListMove(self, node):
        """
        Update history of node moves over the grid:
        parameter node is added or moved to the end of history list.
        An unmoved node is removed from history.

        :param n:
        :type n:  activeNode
        """
        try:
            self.historyListMove.remove(node)
        except ValueError:
            pass
        if node.hasMoved():
            self.historyListMove.append(node)

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
        Disambiguation of row/col.

        :param row: grid row
        :type row: int
        :param col: grid col
        :type col: int
        :return: grid node
        :rtype: activeNode
        """
        return self.gridNodes[col][row]

    def reset(self, unSmoothOnly=False):
        """
        unselect and reset nodes to their initial position.
        If unSmoothOnly is False all nodes are reset,
        otherwise control points are not moved.
        """
        # reset current transformation to identity
        self.setTransform(QTransform())
        if not unSmoothOnly:
            # explode all node groups
            groupList = [item for item in self.childItems() if type(item) is nodeGroup]
            for item in groupList:
                item.prepareGeometryChange()
                try:
                    # disconnect all signals received by node group item
                    self.scene().disconnect(item)
                    item.brightnessItem.cubic.curveChanged.disconnect(item)
                    self.scene().removeItem(item.brightnessItem)
                except RuntimeError:
                    pass
                self.scene().destroyItemGroup(item)
        for i in range(self.size):
            for j in range(self.size):
                node = self.gridNodes[i][j]
                if unSmoothOnly:
                    if node.control:
                        continue
                node.setPos(node.mapGrid2Parent(node.initialPosition))  # node may still belong to a group
                node.setSelected(False)
                node.setControl(False)
                node.setVisible(False)
        self.setDefaultControlNodes()
        self.showAllNodes = False
        # empty list of moves
        self.historyListMove = []

    def smooth(self):
        """
        Smooths the grid by moving each non control node to the position
        of the laplacian mean of its neighbors.
        """
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            # apply laplacian kernel
            for i in range(self.size):
                for j in range(self.size):
                    curnode = self.gridNodes[i][j]
                    newPos = curnode.laplacian()  # grid (untransformed) coordinates
                    if curnode.isControl():
                        continue
                    # update position
                    if curnode.gridPos_U() != newPos:
                        # next line works for grouped and ungrouped nodes because the grid is at (0,0) in its parent (colorChooser)
                        position = newPos - curnode.parentItem().pos()  # position is in parent's coordinates
                        curnode.setPos(position)
                        curnode.syncLUT()
            # LUT vertices are possibly bound to several nodes. Hence,
            # output color of some vertices bound to a control point may be changed
            # by the syncLUT() above. Uncomment to restore them now :
            """
            for i in range(self.size):
                for j in range(self.size):
                    if self.gridNodes[i][j].isControl:
                        self.gridNodes[i][j].syncLUT()
            """
        finally:
            QApplication.restoreOverrideCursor()

    def drawGrid(self):
        """
        Builds and sets the path of the grid.
        """
        qpp = QPainterPath()
        qppTrace = QPainterPath()
        for i in range(self.size):
            for j in range(self.size):
                node = self.gridNodes[i][j]
                if i > 0:
                    qpp.moveTo(node.gridPos_U())
                    qpp.lineTo(self.gridNodes[i - 1][j].gridPos_U())
                if j > 0:
                    qpp.moveTo(node.gridPos_U())
                    qpp.lineTo(self.gridNodes[i][j - 1].gridPos_U())
                if not node.isTrackable():
                    continue
                # Visualize displacement for trackable nodes
                qppTrace.moveTo(node.gridPos_U())
                qppTrace.lineTo(node.initialPos)
                qppTrace.addEllipse(node.initialPos, 3, 3)
                if self.drawTrace:
                    for n in node.neighbors():
                        if n.isVisible():
                            qppTrace.moveTo(n.gridPos_U())
                            qppTrace.lineTo(node.gridPos_U())
        self.setPath(qpp)
        self.traceGraphicsPathItem.setPath(qppTrace)

    def gridSyncLUT(self):
        s = self.size
        for r in range(s):
            for c in range(s):
                self.gridNodes[r][c].syncLUT()


class colorChooser(QGraphicsPixmapItem):
    """
    Color wheel wrapper : it is a 2D-slider-like
    object providing color selection from the wheel and
    rubber band selection from a grid of activeNode objects
    moving over the wheel.
    """

    def __init__(self, cModel, QImg, target=None, size=0, border=0):
        """
       :param cModel: color model
       :type cModel: cmConverter
       :param QImg: color wheel
       :type QImg: bImage
       :param target: image to sample
       :type target: QImage
       :param size: color wheel diameter
       :type size: integer
       :param border: border size
       :type border: int
        """
        self.QImg = QImg  # not a back link !!!
        self.border = border
        self.cModel = cModel
        self.targetLayer = target
        if size == 0:
            self.size = min(QImg.width(), QImg.heigth()) - 2 * border
        else:
            self.size = size
        self.origin = 0
        # 2D-histogram of target image
        self.targetHist = None
        # if target is not None:
        # self.updateTargetHist()
        self.showTargetHist = False
        super().__init__(self.QImg.rPixmap)
        self.setPixmap(self.QImg.rPixmap)
        self.setOffset(QPointF(-border, -border))
        self.onMouseRelease = lambda p, x, y, z: 0
        self.rubberBand = None

    def updateTargetHist(self):
        """
        Updates the 2D hist of target image.
        The computation is done in background.
        """

        def bgTask():
            # convert to current color space
            img = self.targetLayer.inputImg()
            hsxImg = self.cModel.rgb2cmVec(QImageBuffer(img)[:, :, :3][:, :, ::-1])
            # get polar coordinates relative to the color wheel
            xyarray = self.QImg.GetPointVec(hsxImg).astype(int)
            maxVal = self.QImg.width()
            STEP = 10
            # build 2D histogram for xyarray
            H, xedges, yedges = np.histogram2d(xyarray[:, :, 0].ravel(), xyarray[:, :, 1].ravel(),
                                               bins=[np.arange(0, maxVal + STEP, STEP),
                                                     np.arange(0, maxVal + STEP, STEP)],
                                               normed=True)
            w, h = self.QImg.width(), self.QImg.height()
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
            self.updatePixmap()

        threading.Thread(target=bgTask).start()

    def setPixmap(self, pxmap):
        """
        Paints the histogram on a copy of pxmap
        and displays the copy.

        :param pxmap:
        :type pxmap: QPixmap
        """
        if self.targetHist is not None and self.showTargetHist:
            pxmap1 = QPixmap(pxmap)
            qp = QPainter(pxmap1)
            qp.drawImage(0, 0, self.targetHist)
            qp.end()
            pxmap = pxmap1
        super().setPixmap(pxmap)

    def updatePixmap(self):
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
        if self.rubberBand is not None:
            self.rubberBand.setGeometry(QRect(self.origin, e.screenPos()).normalized())  # topLeft, bottomRight

    def mouseReleaseEvent(self, e):
        # rubberBand selection
        if e.button() == Qt.RightButton:
            return
        if self.rubberBand is not None:
            self.rubberBand.hide()
        grid = self.scene().grid
        rubberRect = QRect(self.origin, e.screenPos()).normalized()  # topLeft, bottomRight
        # get QGraphicsView instance
        view = self.scene().views()[0]
        # select nodes in rubberband, unselect others
        for i in range(grid.size):
            for j in range(grid.size):
                viewCoords = view.mapFromScene(grid.gridNodes[i][j].scenePos().toPoint())
                screenCoords = view.mapToGlobal(viewCoords)
                if rubberRect.contains(screenCoords):
                    grid.gridNodes[i][j].setSelected(True)
                else:
                    if type(grid.gridNodes[i][j].parentItem()) is nodeGroup:
                        grid.gridNodes[i][j].parentItem().setSelected(False)
                    grid.gridNodes[i][j].setSelected(False)
        grid.drawGrid()
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
    def getNewWindow(cls, cModel, targetImage=None, axeSize=500, LUTSize=LUTSIZE, layer=None, parent=None,
                     mainForm=None):
        """
        build a graphicsForm3DLUT object. The parameter size represents the size of
        the color wheel, border not included (the size of the window is adjusted).

        :param cModel: color Model converter
        :type cModel: cmConverter
        :param targetImage
        :type targetImage:
        :param axeSize: size of the color wheel (default 500)
        :type axeSize:
        :param LUTSize: size of the LUT
        :type LUTSize:
        :param layer: layer of targetImage linked to graphics form
        :type layer:
        :param parent: parent widget
        :type parent:
        :param mainForm:
        :type mainForm:
        :return: graphicsForm3DLUT object
        :rtype:
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
       :param cModel: color space used by colorPicker, slider2D and colorPicker
       :type cModel: cmConverter object
       :param axeSize: size of the color wheel
       :type axeSize: int
       :param targetImage:
       :type targetImage: imImage
       :param LUTSize:
       :type LUTSize: int
       :param layer: layer of targetImage linked to graphics form
       :type layer : QLayer
       :param parent:
       :type parent:
        """
        super().__init__(targetImage=targetImage, layer=layer, parent=parent)
        self.mainForm = mainForm  # used by saveLUT()
        # context help tag
        self.helpId = "LUT3DForm"
        self.cModel = cModel
        self.border = 20
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize + 90, axeSize + 90)  # + 250)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setBackgroundBrush(QBrush(Qt.black, Qt.SolidPattern))
        self.currentHue, self.currentSat, self.currentPb = 0, 0, self.defaultColorWheelBr
        self.currentR, self.currentG, self.currentB = 0, 0, 0
        self.size = axeSize
        # init LUT
        freshLUT3D = LUT3D(None, size=LUTSize, alpha=True)
        self.graphicsScene.lut = freshLUT3D
        # init 2D slider (color wheel)
        swatchImg = hueSatPattern(axeSize, axeSize, cModel, bright=self.defaultColorWheelBr, border=self.border)
        slider2D = colorChooser(cModel, swatchImg, target=self.layer, size=axeSize, border=self.border)
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
        self.graphicsScene.bSlider.setPos(
            QPointF(-self.border, self.graphicsScene.slider2D.QImg.height() - self.border))
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
                opt = self.graphicsScene.options[option]
                self.graphicsScene.slider2D.showTargetHist = opt
                if opt:
                    self.graphicsScene.slider2D.updateTargetHist()  # bg task
                else:
                    self.graphicsScene.slider2D.updatePixmap()
                return
            if option == 'keep alpha':
                self.enableButtons()
                self.layer.applyToStack()

        self.listWidget3.onSelect = onSelect3
        # set initial selection to 'select naighbors'
        item = self.listWidget3.items[options3[0]]
        item.setCheckState(Qt.Checked)
        self.graphicsScene.options = UDict((self.listWidget2.options, self.listWidget3.options))

        for wdg in [self.listWidget2, self.listWidget3]:
            wdg.setMinimumWidth(wdg.sizeHintForColumn(0) + 20)  # prevent h. scrollbar
            wdg.setMinimumHeight(wdg.sizeHintForRow(0) * len(wdg.items) + 20)  # prevent v. scrollbar

        # grid size combo
        gridCombo = QComboBox()
        oDict = OrderedDict([('33', 33), ('17', 17), ('9', 9)])
        for key in oDict:
            gridCombo.addItem(key, oDict[key])

        def gridSizeChanged(value):
            global spread
            s = gridCombo.itemData(value)
            spread = 1 if s > 20 else 2 if s > 12 else 4
            self.onChangeGrid(s)
            self.onReset()

        gridCombo.currentIndexChanged.connect(gridSizeChanged)

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
        container = self.addCommandLayout(gl)  # add before populating
        for i, button in enumerate([pushButton1, pushButton3, pushButton31, pushButton2]):
            gl.addWidget(button, 0, i)
        gl.addWidget(pushButton4, 1, 0)
        gl.addWidget(pushButton5, 1, 1)
        for i, widget in enumerate([self.listWidget2, self.listWidget3]):
            gl.addWidget(widget, 2 if i == 0 else 1, i + 1, 1 if i == 0 else 2, 1)
        hl = QHBoxLayout()
        hl.addWidget(QLabel('Grid Size'))
        hl.addWidget(gridCombo)
        hl.addWidget(infoCombo)
        hl.addWidget(self.info)
        gl.addLayout(hl, 3, 0, -1, -1)
        container.adjustSize()
        self.setViewportMargins(0, 0, 0, container.height() + 35)

        # set defaults
        self.colorInfoFormat = 0  # RGB
        colorInfoFormatChanged(self.colorInfoFormat)

        # whatsthis
        self.setWhatsThis(
            """ <b>2.5D LUT Perceptual Editor</b><br>
            Nodes are displayed as small circles on the color wheel (push the button <i>Show/Hide All </i> if they are 
            not shown. Each node corresponds to a set of colors sharing the same hue and saturation.<br> The size of 
            the grid can be changed. Note that changing the grid size resets the whole grid.<br>
            Edition can be applied to a subset of the grid nodes or to all nodes simultaneously (gamut warping).<br> 
            <b>Image Driven Node Selection</b> Select nodes to edit by mouse clicks on the image or by Ctrl+clicks 
            on nodes.
            <b> When the Marquee Tool is active node Node Selection is disabled.</b>
            <b>To modify the colors</b> of a node <i>Ctrl+drag</i> it on
            the wheel. Several nodes can be moved simultaneously by grouping them : <br>
            <b>Grouping nodes</b> :<br>
                    &nbsp; 1 - Select the nodes to be grouped (selected nodes are white); a rubberband is available.<br>
                    &nbsp; 2 - next, right click any one of the selected nodes and 
                    choose group from the context menu wich opens.<br>
            <b>Image Driven Node Removal</b> :<br>
                    &nbsp; 1 - Check the option Remove Node;<br>
                    &nbsp; 2 - on the image, click the pixels to unselect.<br>
            <b>Gamut Warping</b> The whole grid of nodes can be twisted by using Ctrl+Alt+Drag on any of 
            the four corner vertices of the grid (Press <i>Show/Hide All </i> if they are not visible).<br>
            The two modes of edition can be freely mixed.<br>
            <b>Brightness Control</b> is found in the context menu of node groups.<br> 
            <b>Warning</b> : Selecting/unselecting nodes with the mouse is enabled only when
            the Color Chooser is closed.<br>
            Press the <b> Smooth Grid</b> button to smooth color transitions between neighbor nodes.
            Control nodes (red circles) are not moved. Use the context menu to switch control nodes on and off.
            <b>Crosshair</b> markers indicate neutral gray tones and average 
            skin tones. They can be moved with the mouse.
            The position of the second marker is reflected in the editable 
            bottom right box. Due to inherent properties
            of the CMYK color model, CMYK input values may be modified silently.<br>
            The editor window can be <b>zoomed</b> with the mouse wheel.<br>
            Check the <b>Keep Alpha</b> option to forward the alpha channel without modifications.<br>
            This option must be unchecked to be able to build a mask from the selection.<br>
            HSpB layer is slower than HSV, but usually gives better results.<br>    
            """
        )  # end of setWhatsThis

    def colorPickedSlot(self, x, y, modifiers):
        """
        Overriding method.

        :param x:
        :type x:
        :param y:
        :type y:
        :param modifiers:
        :type modifiers:
        :return:
        :rtype:
        """
        clr = self.layer.parentImage.getActivePixel(x, y, qcolor=True)
        red, green, blue = clr.red(), clr.green(), clr.blue()
        movedNodes = self.selectGridNode(red, green, blue)
        if movedNodes:
            self.layer.applyToStack()

    def selectGridNode(self, r, g, b):
        """
        selects the nearest grid nodes corresponding to r,g,b values.
        According to current options, selected nodes can be shown or
        hidden or reset. The function returns True if nodes are
        moved and False otherwise.

        :param r: color
        :param g: color
        :param b: color
        :return:
        :rtype: boolean
        """
        movedNodes = False
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
        nodesToSelect = NNN.neighbors() + [NNN] if self.graphicsScene.options['select neighbors'] else [NNN]
        for n in nodesToSelect:
            n.setVisible(mode)
            n.setSelected(mode)
            if self.graphicsScene.options['reset removed nodes'] and not mode:
                if isinstance(n.parentItem(), nodeGroup):
                    group = n.parentItem()
                    group.removeFromGroup(n)
                    if not group.childItems():
                        self.graphicsScene.destroyItemGroup(group)
                if n.pos() != n.initialPos:
                    n.setPos(n.initialPos)
                    movedNodes = True
                n.syncLUT()
        self.grid.drawGrid()
        # update status
        self.onSelectGridNode(h, s)
        return movedNodes

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
        px = brightnessPattern(self.bSliderWidth, self.bSliderHeight, self.cModel, self.currentHue,
                               self.currentSat).rPixmap
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
        self.grid.drawGrid()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def onChangeGrid(self, size):
        self.scene().removeItem(self.grid)
        self.grid = activeGrid(size, self.cModel, parent=self.graphicsScene.slider2D)
        self.graphicsScene.grid = self.grid

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

    def updateLayer(self):
        """
        Data changed slot
        """
        layer = self.layer
        layer.applyToStack()
        layer.parentImage.onImageChanged()

    def __getstate__(self):
        return {'history': [(p.gridRow,
                             p.gridCol,
                             p.gridPos_U().x(),
                             p.gridPos_U().y(),
                             p.parentItem().uid if type(p.parentItem()) is nodeGroup else -1,
                             # p.parentItem().x(), p.parentItem().y()
                             )
                            for p in self.grid.historyListMove]}

    def __setstate__(self, state):
        # prevent multiple updates
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        groupList = []
        for item in state['state']['history']:
            p = self.grid.getNodeAt(item[1], item[0])
            p.setPos(item[2], item[3])
            if item[4] != -1:
                readuidList = [g.readuid for g in groupList]
                if item[4] in readuidList:
                    gr = groupList[readuidList.index(item[4])]
                    gr.addToGroup(p)
                else:
                    gr = nodeGroup(grid=self.grid, position=p.pos(), parent=p.parentItem())
                    gr.readuid = item[4]
                    gr.addToGroup(p)
                    groupList.append(gr)
                    # gr.setPos(item[5], item[6])
            # self.grid.updateHistoryListMove(p) # called by syncLUT()
            p.setSelected(True)
            p.setVisible(True)
            p.syncLUT()
        self.grid.drawGrid()
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()


if __name__ == '__main__':
    size = 4000
    # random ints in range 0 <= x < 256
    b = np.random.randint(0, 256, size=size * size * 3, dtype=np.uint8)
    testImg = np.reshape(b, (size, size, 3))
    interpImg = interpTriLinear(LUT3DIdentity.LUT3DArray, LUT3DIdentity.step, testImg)
    d = testImg - interpImg
    print("max deviation : ", np.max(np.abs(d)))
