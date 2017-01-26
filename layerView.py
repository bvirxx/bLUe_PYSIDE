from PyQt4.QtGui import QTableView, QStandardItem, QStandardItemModel, QItemSelectionModel, QAbstractItemView, QPalette, QStyledItemDelegate, QColor, QImage, QPixmap, QIcon, QHeaderView
from PyQt4.QtCore import Qt
import resources_rc  # DO NOT REMOVE !!!
import QtGui1

"""
The class QLayerView inherits from QTableView. It is used
in the main form built by Qt Designer to display lists
of image layers.
"""

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

    def __init__(self, img):
        super(QTableView, self).__init__()
        self.clicked.connect(self.viewClicked)
        # behavior and style for selection
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setStyleSheet("QTableView {selection-background-color: lightgray}")

    def addLayers(self, mImg):
        self.img=mImg
        model = QStandardItemModel()
        model.setColumnCount(3)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        for lay in reversed(mImg._layersStack) :
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
        # select background layer
        self.selectRow(0)

        #header = self.horizontalHeader()
        #header.setResizeMode(0, QHeaderView.Stretch)
        #header.setResizeMode(0, QHeaderView.ResizeToContents)
        #header.setResizeMode(1, QHeaderView.ResizeToContents)
        #header.setResizeMode(2, QHeaderView.ResizeToContents)

        #self.setItemDelegateForColumn(1, ImageDelegate(None))

    def viewClicked(self, clickedIndex):
        row = clickedIndex.row()
        model = clickedIndex.model()
        l = len(self.img._layersStack)
        if clickedIndex.column() == 0 :
            visible = not(self.img._layersStack[l-1-row].visible)
            self.img._layersStack[l-1-row].visible = visible
            if visible:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon.png") ,Qt.DecorationRole)
            else:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon-strike.png"), Qt.DecorationRole)
        elif clickedIndex.column() == 1 :
            # make sected layer the active layer
            self.img.activeLayer = self.img._layersStack[l-1-row]
            # show/hide window for adjustment layer
            if hasattr(self.img._layersStack[l-1-row], "adjustView"):
                win = self.img._layersStack[l-1-row].adjustView
            else:
                win = None
            if win is not None:
                if win.widget().isVisible():
                    win.hide()
                else:
                    win.show()
        QtGui1.window.label.repaint()

