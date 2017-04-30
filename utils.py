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
from math import erf
from PySide.QtGui import QListWidget, QListWidgetItem, QGraphicsPathItem, QColor, QPainterPath, QPen
from PySide.QtCore import Qt, QPoint


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
            #listItem.mySelectedAttr = False
            self.addItem(listItem)
            self.items[option] = listItem
        self.setMinimumWidth(self.sizeHintForColumn(0))
        #self.setGeometry(self.sizeHintForColumn(0), self.sizeHintForRow(0))
        self.exclusive = exclusive
        self.itemClicked.connect(self.select)
        # selection hook.
        self.onSelect = lambda x : 0

    def select(self, item):
        """
        Mouse click event handler
        @param item:
        """
        for r in range(self.count()):
            currentItem = self.item(r)
            if currentItem is item:
                if self.exclusive:
                    currentItem.setCheckState(Qt.Checked)
                    currentItem.setSelected(True)
                else:
                    currentItem.setCheckState(Qt.Unchecked if currentItem.isSelected() else Qt.Checked)
                    currentItem.setSelected(not currentItem.isSelected())
            elif self.exclusive:
                currentItem.setCheckState(Qt.Unchecked)
                currentItem.setSelected(False)
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

    import numpy as np
    from math import factorial

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