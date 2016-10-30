from PyQt4 import QtCore, QtGui
from PyQt4 import QtGui

import numpy as np

def gray2qimage(gray):
    """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    if len(gray.shape) != 2:
        raise ValueError("gray2QImage can only convert 2D arrays")

    gray = np.require(gray, np.uint8, 'C')

    h, w = gray.shape

    result = QtGui.QImage(gray.data, w, h, w, QtGui.QImage.Format_Indexed8) # BV added 2nd w
    result.ndarray = gray
    for i in range(256):
        result.setColor(i, QtGui.QColor(i, i, i).rgb())
    return result

def rgb2qimage(rgb):
    """Convert the 3D numpy array `rgb` into a 32-bit QImage.  `rgb` must
    have three dimensions with the vertical, horizontal and RGB image axes.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    if len(rgb.shape) != 3:
        raise ValueError("rgb2QImage can only convert 3D arrays")
    if rgb.shape[2] not in (3, 4):
        raise ValueError("rgb2QImage can expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")

    h, w, channels = rgb.shape

    # Qt expects 32bit BGRA data for color images:
    bgra = np.empty((h, w, 4), np.uint8, 'C')
    bgra[...,0] = rgb[...,2]
    bgra[...,1] = rgb[...,1]
    bgra[...,2] = rgb[...,0]
    if rgb.shape[2] == 3:
        bgra[...,3].fill(255)
        fmt = QtGui.QImage.Format_RGB32
    else:
        bgra[...,3] = rgb[...,3]
        fmt = QtGui.QImage.Format_ARGB32

    result = QtGui.QImage(bgra.data, w, h, fmt)
    result.ndarray = bgra
    return result

def ndarrayToQimage(ndimg, format= QtGui.QImage.Format_ARGB32):
    """
    Convert a Numpy array to a QImage
    :param ndimg: The Numpy array to be converted
    :param format: The QImage pixel format (default ARGB32)
    :return: The converted image
    """
    if (len(ndimg.shape) == 2):   # Grayscale image
        ndimg = np.dstack((ndimg, ndimg, ndimg, ndimg) )   # convert to ARGB
        ndimg[:,:,0] = 255 # set alpha channel

    return QtGui.QImage(ndimg.tostring(), ndimg.shape[1], ndimg.shape[0], 4*ndimg.shape[1], format)

def Qimage2array(qimg):
    """
    Convert a QImage to a numpy array. Last dim of the array
    is in order RGBA.
    :param qimg: QImage in format ARGB
    :return: the converted array
    """
    w,h = qimg.width(), qimg.height()
    data = qimg.bits().asarray(w*h*4)

    return np.array(data, dtype=np.uint8).reshape(h, w, 4)

