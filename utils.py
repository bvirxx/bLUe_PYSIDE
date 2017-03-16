from PyQt4.QtGui import QApplication, QPainter, QWidget, QPixmap, QPushButton, QListWidget, QListWidgetItem
from PyQt4.QtGui import QGraphicsView, QGraphicsScene, QAbstractItemView, QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem , QGraphicsPixmapItem, QGraphicsTextItem, QPolygonF, QGraphicsPolygonItem , QPainterPath, QPainterPathStroker, QPen, QBrush, QColor, QPixmap, QMainWindow, QLabel, QSizePolicy
from PyQt4.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QString

class optionsWidget(QListWidget) :

    def __init__(self, options=[], exclusive=True):
        super(QListWidget, self).__init__()
        self.items = {}
        for option in options:
            listItem = QListWidgetItem(option, self)
            listItem.setCheckState(Qt.Unchecked)
            listItem.mySelectedAttr = False
            self.addItem(listItem)
            self.items[option] = listItem
        self.setMinimumWidth(self.sizeHintForColumn(0))
        self.exclusive = exclusive
        self.itemClicked.connect(self.select)
        self.onSelect = lambda x : 0

    def select(self, item):
        """
        Mouse click event handler
        :param item:
        """
        for r in range(self.count()):
            currentItem = self.item(r)
            if currentItem is item:
                if self.exclusive:
                    currentItem.setCheckState(Qt.Checked)
                    currentItem.mySelectedAttr = True
                else:
                    currentItem.setCheckState(Qt.Unchecked if currentItem.mySelectedAttr else Qt.Checked)
                    currentItem.mySelectedAttr = not currentItem.mySelectedAttr
            elif self.exclusive:
                currentItem.setCheckState(Qt.Unchecked)
                currentItem.mySelectedAttr = False
            self.onSelect(item)