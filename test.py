from PyQt4.QtCore import QSize
from PyQt4.QtGui import QAction
from PyQt4.QtGui import QApplication, QPainter, QWidget, QPixmap, QPushButton, QListWidget, QListWidgetItem
from PyQt4.QtGui import QGraphicsView, QGraphicsScene, QAbstractItemView, QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem , QGraphicsPixmapItem, QGraphicsTextItem, QPolygonF, QGraphicsPolygonItem , QPainterPath, QPainterPathStroker, QPen, QBrush, QColor, QPixmap, QMainWindow, QLabel, QSizePolicy
from PyQt4.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QString
import numpy as np
from time import time
import matplotlib as mpl
import matplotlib.pyplot as plt

from PyQt4.QtGui import QMenu
from PyQt4.QtGui import QRubberBand

from LUT3D import LUTSIZE, LUTSTEP, rgb2hsB, hsp2rgb, hsp2rgbVec, hsp2rgb_ClippingInd, LUT3DFromFactory, LUT3D_SHADOW, LUT3D_ORI
from colorModels import hueSatModel, pbModel
from utils import optionsWidget


hs = np.array([rgb2hsB(t[0], t[1], t[2], perceptual=True) for c in LUT3D_ORI for b in c for t in b if max(t)<256])

ind = np.lexsort((hs[:,2], hs[:,1], hs[:,0]))

print hs[ind][10000:10100]

exit()

x,xedges, yedges=np.histogram2d(hs[:,1], hs[:,2], bins=(20,100))

plt.hist(hsf, bins='auto')

#im = plt.imshow(x)
plt.show()