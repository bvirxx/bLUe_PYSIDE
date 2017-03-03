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
from PyQt4.QtGui import QAction
from PyQt4.QtGui import QMenu
from PyQt4.QtGui import QSlider
from PyQt4.QtGui import QTableView, QStandardItem, QStandardItemModel, QItemSelectionModel, QAbstractItemView, QPalette, QStyledItemDelegate, QColor, QImage, QPixmap, QIcon, QHeaderView
from PyQt4.QtCore import Qt
import resources_rc  # DO NOT REMOVE !!!
import QtGui1



"""
class ImageDelegate(QStyledItemDelegate):

    def __init__(self, parent):
        QStyledItemDelegate.__init__(self, parent)

    def paint(self, painter, option, index):

        painter.fillRect(option.rect, QColor(191,222,185))

        # path = "path\to\my\image.jpg"

        pixmap = QPixmap.fromImage(image)
        pixmap.scaled(50, 40, Qt.KeepAspectRatio)
        painter.drawPixmap(option.rect, pixmap)
"""
class QLayerView(QTableView) :
    """
    The class QLayerView inherits from QTableView. It is used
    in the main form built by Qt Designer to display lists
    of image layers.
    """

    def __init__(self, img):
        super(QTableView, self).__init__()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.clicked.connect(self.viewClicked)
        self.customContextMenuRequested.connect(self.contextMenu)
        # behavior and style for selection
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setStyleSheet("QTableView { background-color: lightgray;\
                                          selection-background-color: gray;}"
                           )

    def addLayers(self, mImg):
        self.img=mImg
        model = QStandardItemModel()
        # columns : visible | icon and name | unused
        model.setColumnCount(3)

        for lay in reversed(mImg.layersStack) :
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
            model.appendRow(items)
        self.setModel(model)
        # select top layer
        self.selectRow(0)
        self.img.activeLayer = self.img.layersStack[-1]

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        header = self.horizontalHeader()
        #header.setResizeMode(0, QHeaderView.Stretch)
        header.setResizeMode(0, QHeaderView.ResizeToContents)
        header.setResizeMode(1, QHeaderView.ResizeToContents)
        header.setResizeMode(2, QHeaderView.ResizeToContents)
        #self.setItemDelegateForColumn(1, ImageDelegate(None))

    def viewClicked(self, clickedIndex):
        """
        click event handler
        :param clickedIndex:
        """
        row = clickedIndex.row()
        model = clickedIndex.model()
        # toggle layer visibility
        if clickedIndex.column() == 0 :
            visible = not(self.img.layersStack[-1-row].visible)
            self.img.layersStack[-1-row].visible = visible
            if visible:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon.png") ,Qt.DecorationRole)
            else:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon-strike.png"), Qt.DecorationRole)
        # hide/display adjustment form
        elif clickedIndex.column() == 1 :
            # make selected layer the active layer
            self.img.activeLayer = self.img.layersStack[-1-row]
            # show/hide window for adjustment layer
            if hasattr(self.img.layersStack[-1-row], "adjustView"):
                win = self.img.layersStack[-1-row].adjustView
            else:
                win = None
            if win is not None:
                if win.widget().isVisible():
                    win.hide()
                else:
                    win.setFloating(True)
                    win.show()
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
        menu.addAction(actionTransparency)
        self.wdgt = QSlider(Qt.Horizontal)
        self.wdgt.setMinimum(0)
        self.wdgt.setMaximum(100)
        self.wdgt.valueChanged.connect
        def f():
            self.wdgt.show()
        def g(value):
            layer.setTransparency(value)
        actionTransparency.triggered.connect(f)
        self.wdgt.valueChanged.connect(g)


        menu.exec_(self.mapToGlobal(pos))


