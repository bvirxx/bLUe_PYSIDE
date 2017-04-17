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
from PySide.QtGui import QListWidget, QListWidgetItem
from PySide.QtCore import Qt


class Channel():
    RGB, Red, Green, Blue = range(-1, 3)

class optionsWidget(QListWidget) :
    """
    Displays a list of options with checkboxes.
    The choices can be mutually exclusive (default) or not
    exclusive.
    """

    def __init__(self, options=[], exclusive=True):
        """
        :param options: list of strings
        :param exclusive: boolean
        """
        super(optionsWidget, self).__init__()
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
        # selection hook.
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

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    This pure numpy implementation of the savitzky_golay filter is taken
    from http://stackoverflow.com/questions/22988882/how-to-smooth-a-curve-in-python
    Many thanks to elviuz.
    :param y: data (type numpy array)
    :param window_size:
    :param order:
    :param deriv:
    :param rate:
    :return: smoothed data array
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

def phi(x, mu, sigma):
    """
    calculates the cumulative distribution function (CDF) phi of the
    normal distribution N(mu, sigma)
    :param x: float
    :param mu : mean value
    :param sigma : standard deviation
    :return: CDF value
    """
    return (1.0 + erf((x-mu)/(sigma*np.sqrt(2)))) / 2.0

def gaussianKernel(mu, w):
    """
    Calculates the 2D gaussian kernel of size w,
    for mean mu.
    The standard deviation sigma and w are bound by the relation w = 2.0 * int(4.0 * sigma + 0.5)
    :param mu:
    :param sigma:
    :param size:
    :return:
    """
    sigma = (w - 1.0) / 8.0
    interval = 4.0 * sigma
    points = np.linspace(-interval, interval, w + 1)

    points = map(lambda x : phi(x,0, sigma), points)
    kern1d = np.diff(points)
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

print gaussianKernel(0, 5)*256