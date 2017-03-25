"""
Copyright (C) 2017  Bernard Virot

bLUe - Photo editing software.

With Blue you can enhance and correct the colors of your photos in a few clicks.
No need for complex tools such as lasso, magic wand or masks.
bLUe interactively constructs 3D LUTs (Look Up Tables), adjusting the exact set
of colors you want.

3D LUTs are widely used by professional film makers, but the lack of
interactive tools maked them poorly useful for photo enhancement, as the shooting conditions
can vary widely from an image to another. With bLUe, in a few clicks, you select the set of
colors to modify, the corresponding 3D LUT is automatically built and applied to the image.
You can then fine tune it as you want.

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
from PyQt4.QtCore import QRectF
from PyQt4.QtCore import QString
from PyQt4.QtCore import QVariant
from PyQt4.QtGui import QAction, QMenu, QSlider
from PyQt4.QtGui import QBrush
from PyQt4.QtGui import QTableView, QStandardItem, QStandardItemModel, QItemSelectionModel, QAbstractItemView, QPalette, QStyledItemDelegate, QColor, QImage, QPixmap, QIcon, QHeaderView
from PyQt4.QtCore import Qt
import resources_rc  # mandatory : DO NOT REMOVE !!!
import QtGui1

class layerModel(QStandardItemModel):

    def __init__(self):
        super(layerModel, self).__init__()

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled | Qt.ItemIsEditable


class itemDelegate(QStyledItemDelegate):
    """
    
    """

    def __init__(self, parent=None):
        QStyledItemDelegate.__init__(self, parent)

    def paint(self, painter, option, index):
        rect = QRectF(option.rect)
        if index.column() == 2:
            painter.drawText(rect, QString('A'))
        else:
            QStyledItemDelegate.paint(self, painter, option, index)


class QLayerView(QTableView) :
    """
    The class QLayerView inherits from QTableView. It is used
    in the main form to display lists
    of image layers.
    """
    def __init__(self, img):
        super(QLayerView, self).__init__()
        self.img = img
        # form to display
        self.currentWin = None
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.clicked.connect(self.viewClicked)
        self.customContextMenuRequested.connect(self.contextMenu)
        # behavior and style for selection
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setStyleSheet("QTableView { background-color: lightgray;\
                                          selection-background-color: gray;}"
                           )
        """
        self.verticalHeader().setMovable(True)
        self.verticalHeader().setDragEnabled(True)
        self.verticalHeader().setDragDropMode(QAbstractItemView.InternalMove)
        """
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

        self.setItemDelegate(itemDelegate())


    def addLayers(self, mImg):
        self.img=mImg
        mImg.layerView = self
        model = layerModel() #QStandardItemModel()
        # columns : visible | icon and name | current index in layersStack (hidden)
        model.setColumnCount(3)
        #self.setColumnHidden(2, True)
        self.setModel(model)
        l = len(mImg.layersStack)

        for r, lay in enumerate(reversed(mImg.layersStack)):
            items = []
            # col 0 : visibility icon
            if lay.visible :
                item_visible = QStandardItem(QIcon(":/images/resources/eye-icon.png"), "")

            else:
                item_visible = QStandardItem(QIcon(":/images/resources/eye-icon-strike.png"), "")
            items.append(item_visible)
            # col 1 : image icon and name
            item_name = QStandardItem(QIcon(lay.qPixmap), lay.name)
            items.append(item_name)
            # index in layersStack
            #item_rank = QStandardItem (l - r)
            #items.append(item_rank)
            if hasattr(lay, 'mask'):
                item_mask = QStandardItem('M')
            else:
                item_mask = QStandardItem('')
            items.append(item_mask)
            model.appendRow(items)
        model.setData(model.index(0, 2), QVariant(QBrush(Qt.red)), Qt.ForegroundRole | Qt.DecorationRole)

        self.setModel(model)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        header = self.horizontalHeader()
        header.setResizeMode(0, QHeaderView.ResizeToContents)
        header.setResizeMode(1, QHeaderView.ResizeToContents)
        header.setResizeMode(2, QHeaderView.ResizeToContents)
        # select active layer
        self.selectRow(len(mImg.layersStack) - 1 - mImg.activeLayerIndex)
        activeLayer = mImg.getActiveLayer()

        if hasattr(activeLayer, 'adjustView'):
            self.currentWin = activeLayer.adjustView
        if hasattr(activeLayer, 'segmentView'):
            self.currentWin = activeLayer.segmentView
        if self.currentWin is not None:
            self.currentWin.show()

    def update(self):
        activeLayer = self.img.getActiveLayer()

        if hasattr(activeLayer, 'adjustView'):
            self.currentWin = activeLayer.adjustView
        if hasattr(activeLayer, 'segmentView'):
            self.currentWin = activeLayer.segmentView
        if self.currentWin is not None:
            self.currentWin.show()

    def dropEvent(self, event):
        """
        drop event handler. Moving row
        :param event:
        """
        if event.source() == self:
            rows = set([mi.row() for mi in self.selectedIndexes()])
            targetRow = self.indexAt(event.pos()).row()
            rows.discard(targetRow)
            rows = sorted(rows)
            if not rows:
                return
            if targetRow == -1:
                targetRow = self.model().rowCount()
            # insert empty row
            for _ in range(len(rows)):
                self.model().insertRow(targetRow)
            # src row to target row mapping
            rowMapping = dict()
            for idx, row in enumerate(rows):
                if row < targetRow:
                    rowMapping[row] = targetRow + idx
                else:
                    rowMapping[row + len(rows)] = targetRow + idx
            colCount = self.model().columnCount()
            for srcRow, tgtRow in sorted(rowMapping.iteritems()):
                for col in range(0, colCount):
                    self.model().setItem(tgtRow, col, self.model().takeItem(srcRow, col))
            for row in reversed(sorted(rowMapping.iterkeys())):
                self.model().removeRow(row)

            rStack = self.img.layersStack[::-1]
            for _ in range(len(rows)):
                rStack.insert(targetRow, None)
            for srcRow, tgtRow in sorted(rowMapping.iteritems()):
                rStack[tgtRow] = rStack[srcRow]
            for row in reversed(sorted(rowMapping.iterkeys())):
                rStack.pop(row)

            self.img.layersStack = rStack[::-1]
            event.accept()


    def viewClicked(self, clickedIndex):
        """
        Mouse click event handler.
        :param clickedIndex:
        """
        row = clickedIndex.row()
        #model = clickedIndex.model()
        # toggle layer visibility
        if clickedIndex.column() == 0 :
            visible = not(self.img.layersStack[-1-row].visible)
            self.img.layersStack[-1-row].visible = visible
            # update visibility icon
            if visible:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon.png") ,Qt.DecorationRole)
            else:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon-strike.png"), Qt.DecorationRole)
        # hide/display adjustment form
        elif clickedIndex.column() == 1 :
            # make selected layer the active layer
            self.img.setActiveLayer(len(self.img.layersStack) - 1 - row, signaling=False)
            # update displayed window
            if self.currentWin is not None:
                self.currentWin.hide()
            if hasattr(self.img.layersStack[-1-row], "adjustView"):
                self.currentWin = self.img.layersStack[-1-row].adjustView
            else:
                self.currentWin = None
            if self.currentWin is not None:
                self.currentWin.show()
        QtGui1.window.label.repaint()

    def contextMenu(self, pos):
        """
        context menu event handler
        :param pos: event coordinates relative to widget
        """
        index = self.indexAt(pos)
        layer = self.img.layersStack[-1-index.row()]
        menu = QMenu()
        actionTransparency = QAction('Transparency', None)
        actionDup = QAction('Duplicate layer', None)
        menu.addAction(actionTransparency)
        menu.addAction(actionDup)
        self.wdgt = QSlider(Qt.Horizontal)
        self.wdgt.setMinimum(0)
        self.wdgt.setMaximum(100)
        #self.wdgt.valueChanged.connect
        def f():
            self.wdgt.show()
        def g(value):
            layer.setOpacity(value)
        def dup():
            self.img.dupLayer(index = len(self.img.layersStack) -1 - index.row())
            self.addLayers(self.img)
        actionTransparency.triggered.connect(f)
        actionDup.triggered.connect(dup)
        self.wdgt.valueChanged.connect(g)


        menu.exec_(self.mapToGlobal(pos))


