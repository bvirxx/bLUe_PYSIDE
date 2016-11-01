import numpy as np
from PyQt4.QtGui import QImage

def ndarrayToQImage(ndimg, format=QImage.Format_ARGB32):
    """
    Convert a 3D numpy ndarray to a QImage. No sanity check is
    done concerning the compatibility between the ndarray shape and
    the QImage format.
    :param ndimg: The ndarray to be converted
    :param format: The QImage format (default ARGB32)
    :return: The converted image
    """
    if ndimg.ndim != 3 or ndimg.dtype != 'uint8':
        print "ndarray2QImage : array must be 3D with dpype=uint8"
        return None

    bytePerLine = ndimg.shape[1] * ndimg.shape[2]

    #return QImage(ndimg.tostring(), ndimg.shape[1], ndimg.shape[0], 4*ndimg.shape[1], format)
    return QImage(ndimg.tostring(), ndimg.shape[1], ndimg.shape[0], bytePerLine, format)

def QImageToNdarray(qimg):
    """
    Convert a QImage to a numpy ndarray.
    Format 1 bit per pixel is not supported
    :param qimg: QImage
    :return: The converted array, in BGRA order (little endian arch.) or ARGB (big endian arch.)
    """
    # get pixel depth
    bpp = qimg.depth()
    if bpp == 1:
        print "Qimage2array : unsupported image format 1 bit per pixel"
        return None
    Bpp = bpp / 8

    w,h = qimg.width(), qimg.height()

    # get memory buffer as a sip.array object
    data = qimg.bits().asarray(w*h*Bpp)

    return np.array(data, dtype=np.uint8).reshape(h, w, Bpp)

