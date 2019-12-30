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
from math import factorial

import numpy as np


class SavitzkyGolayFilter:
    """
    Savitzky-Golay Filter.
    This is a pure numpy implementation of the Savitzky_Golay filter. It is taken
    from U{http://stackoverflow.com/questions/22988882/how-to-smooth-a-curve-in-python}
    Many thanks to elviuz.
    """
    window_size = 11   # must be odd
    order = 3
    deriv = 0
    rate = 1
    kernel = None

    @classmethod
    def getKernel(cls):
        if cls.kernel is None:
            order_range = range(cls.order + 1)
            half_window = (cls.window_size - 1) // 2
            # compute the array m of filter coefficients
            b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
            cls.kernel = np.linalg.pinv(b).A[cls.deriv] * cls.rate ** cls.deriv * factorial(cls.deriv)  # pinv(b).A : conversion of matrix to array
        return cls.kernel

    @classmethod
    def filter(cls, y):
        """
        @param y: data
        @type y: 1D ndarray, dtype = float
        @return: the filtered data array
        """
        kernel = cls.getKernel()
        half_window = (cls.window_size - 1) // 2
        # pad the signal at the extremes with values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        # y = np.concatenate(([0]*half_window, y, [0]*half_window))
        # apply filter
        return np.convolve(kernel[::-1], y, mode='valid')
