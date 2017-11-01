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
import numpy as np
from math import erf, factorial
from PySide2.QtGui import QColor, QPainterPath, QPen, QKeyEvent, QDesktopServices, QImage, QPainter
from PySide2.QtWidgets import QListWidget, QListWidgetItem, QGraphicsPathItem, QTableWidget
from PySide2.QtCore import Qt, QPoint, QEvent, QObject, QUrl, QRect
#from PySide2.QtWebEngine import QWebView
from imgconvert import QImageBuffer


class channelValues():
    RGB, Red, Green, Blue =[0,1,2], [0], [1], [2]
    HSB, Hue, Sat, Br = [0, 1, 2], [0], [1], [2]
    Lab, L, a, b = [0, 1, 2], [0], [1], [2]

class optionsWidget(QListWidget) :
    """
    Displays a list of options with checkboxes.
    The choices can be mutually exclusive (default) or not
    exclusive.
    """

    def __init__(self, options=[], exclusive=True, parent=None):
        """
        @param options: list of strings
        @param exclusive: boolean
        """
        super(optionsWidget, self).__init__(parent)
        self.items = {}
        for option in options:
            listItem = QListWidgetItem(option, self)
            listItem.setCheckState(Qt.Unchecked)
            self.addItem(listItem)
            self.items[option] = listItem
        #self.setSizeAdjustPolicy(QListWidget.AdjustToContents)
        self.setMinimumWidth(self.sizeHintForColumn(0))
        self.setMinimumHeight(self.sizeHintForRow(0)*len(options))
        self.exclusive = exclusive
        self.itemClicked.connect(self.select)
        # selection hook.
        self.onSelect = lambda x : 0

    def select(self, item):
        """
        Item clicked event handler
        @param item:
        @type item: QListWidgetItem
        """
        if self.exclusive:
            for r in range(self.count()):
                currentItem = self.item(r)
                if currentItem is not item:
                    currentItem.setCheckState(Qt.Unchecked)
                else:
                    currentItem.setCheckState(Qt.Checked)
        if item.checkState() == Qt.Checked:
            self.onSelect(item)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    This pure numpy implementation of the savitzky_golay filter is taken
    from http://stackoverflow.com/questions/22988882/how-to-smooth-a-curve-in-python
    Many thanks to elviuz.
    @param y: data (type numpy array)
    @param window_size:
    @param order:
    @param deriv:
    @param rate:
    @return: smoothed data array
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError :
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order+1)
    half_window = (window_size -1) // 2

    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def checkeredImage(w, h, format=QImage.Format_ARGB32):
    image = QImage(w, h, format)

    # init pattern
    base = QImage(20, 20, format)
    qp = QPainter(base)
    qp.setCompositionMode(QPainter.CompositionMode_Source)
    qp.fillRect(0, 0, 10, 10, Qt.gray)
    qp.fillRect(10, 0, 10, 10, Qt.white)
    qp.fillRect(0, 10, 10, 10, Qt.white)
    qp.fillRect(10, 10, 10, 10, Qt.gray)
    qp.end()

    qp=QPainter(image)
    qp.setCompositionMode(QPainter.CompositionMode_Source)

    # draw the pattern once at 0,0
    qp.drawImage(0, 0, base)

    imageW = image.width()
    imageH = image.height()
    baseW = base.width()
    baseH = base.height()
    while ((baseW < imageW) or (baseH < imageH) ):
        if (baseW < imageW) :
            # Copy and draw the existing pattern to the right
            qp.drawImage(QRect(baseW, 0, baseW, baseH), image, QRect(0, 0, baseW, baseH))
            baseW *= 2
        if (baseH < imageH) :
            # Copy and draw the existing pattern to the bottom
            qp.drawImage(QRect(0, baseH, baseW, baseH), image, QRect(0, 0, baseW, baseH))
            # Update height of our pattern
            baseH *= 2
    qp.end()
    return image

def clip(image, mask, inverted=False):
    """
    clip an image by applying a mask to its alpha channel
    @param image:
    @type image:
    @param mask:
    @type mask:
    @param inverted:
    @type inverted:
    @return:
    @rtype:
    """
    bufImg = QImageBuffer(image)
    bufMask = QImageBuffer(mask)
    if inverted:
        bufMask = bufMask.copy()
        bufMask[:,:,3] = 255 - bufMask[:,:,3]
    bufImg[:,:,3] = bufMask[:,:,3]

def drawPlotGrid(axeSize):
    item = QGraphicsPathItem()
    item.setPen(QPen(QColor(255, 0, 0), 1, Qt.DashLine))
    qppath = QPainterPath()
    qppath.moveTo(QPoint(0, 0))
    qppath.lineTo(QPoint(axeSize, 0))
    qppath.lineTo(QPoint(axeSize, -axeSize))
    qppath.lineTo(QPoint(0, -axeSize))
    qppath.closeSubpath()
    qppath.lineTo(QPoint(axeSize, -axeSize))
    for i in range(1, 5):
        a = (axeSize * i) / 4
        qppath.moveTo(a, -axeSize)
        qppath.lineTo(a, 0)
        qppath.moveTo(0, -a)
        qppath.lineTo(axeSize, -a)
    item.setPath(qppath)
    return item
    #self.graphicsScene.addItem(item)
"""
#pickle example
saved_data = dict(outputFile, 
                  saveFeature1 = feature1, 
                  saveFeature2 = feature2, 
                  saveLabel1 = label1, 
                  saveLabel2 = label2,
                  saveString = docString)

with open('test.dat', 'wb') as outfile:
    pickle.dump(saved_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
"""