import numpy as np
from PyQt4.QtGui import QImage
from PIL import Image

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
        raise ValueError("ndarray2QImage : array must be 3D with dtype=uint8, found ndim=%d, dtype=%s" %(ndimg.ndim, ndimg.dtype))

    bytePerLine = ndimg.shape[1] * ndimg.shape[2]
    if len(ndimg.data)!=ndimg.shape[0]*bytePerLine :
        raise ValueError("ndarrayToQImage : wrong conversion")

    qimg = QImage(ndimg.data, ndimg.shape[1], ndimg.shape[0], bytePerLine, format)
    if qimg.format() == QImage.Format_Invalid:
        raise ValueError("ndarrayToQImage : wrong conversion")

    return qimg

def QImageToNdarray(qimg):
    """
    Raw conversion of a QImage to a numpy ndarray. The size of the
    converted array 3rd axis depends on the image type. Pixel color is
    in BGRA order (little endian arch.) or ARGB (big  endian arch.)
    Format 1 bit per pixel is not supported
    :param qimg: QImage
    :return: The converted array
    """
    # get pixel depth
    bpp = qimg.depth()
    if bpp == 1:
        print "Qimage2array : unsupported image format 1 bit per pixel"
        return None
    Bpp = bpp / 8

    w,h = qimg.width(), qimg.height()

    # get memory buffer as a sip.array object of uint8, length unknown
    data = qimg.bits()

    data = data.asarray(w*h*Bpp)

    #convert sip array to ndarray and reshape
    return np.array(data, dtype=np.uint8).reshape(h, w, Bpp)


def PilImageToQImage(pilimg) :
    """
    Convert a PIL image to a QImage
    :param pilimg: The PIL image
    :return: QImage object
    """
    w, h = pilimg.width, pilimg.height
    mode = pilimg.mode

    # get data buffer (type str)
    data = pilimg.tobytes('raw', mode)

    qimFormat = QImage.Format_ARGB32

    if mode == 'RGB':
        qimFormat = QImage.Format_RGB888
        if len(data)!=w * h * 3 :
            raise ValueError("PilImageToQImage : wrong mode : %s" %mode)
    else :
        raise ValueError("PilImageToQImage : unrecognized mode : %s" %mode)

    #a=QImage(data, w, h, qimFormat )
    return QImage(data, w, h, qimFormat )

def QImageToPilImage(qimg) :
    """
    Convert a PIL image to a QImage
    :param pilimg: The PIL image
    :return: QImage object
    """
    a = QImageToNdarray(qimg)

    if (qimg.format() == QImage.Format_ARGB32) or (qimg.format() == QImage.Format_RGB32):
        # convert pixels from BGRA or BGRX to RGB
        a=a[:,:,:3][:,:,::-1]
        a = np.ascontiguousarray(a)

    w, h = qimg.width(), qimg.height()

    return Image.frombytes('RGB', (w,h), a.data)