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
import cv2
from sys import byteorder

import numpy as np
from PySide2.QtCore import QSize, Qt, QPointF
from PySide2.QtGui import QImage, QPixmap, QColor, QPainter, QPainterPath, QBrush, QPolygonF

from bLUeGui.colorCIE import sRGB2LabVec
from bLUeGui.colorCube import rgb2hspVec
from bLUeGui.const import channelValues


class trackImage(QImage):
    """
    Used to draw histograms with mouse tracking
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drawingScale, self.drawingWidth = (1.0,) * 2


class bImage(QImage):
    """
    Base class for all bLUe images.
    Inherits from QImage. Adds a mask, a pixmap
    and a bunch of caches encapsulated as properties.
    The pixmap is synchronized with the image by the
    method updatePixmap().
    """

    bigEndian = (byteorder == "big")
    defaultColorMaskOpacity = 128

    @staticmethod
    def fromImage(img, parentImage=None):
        bImg = bImage(img)
        bImg.parentImage = parentImage
        return bImg

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__filename = ''
        self.__rPixmap = None
        self.__hspbBuffer = None
        self.__LabBuffer = None
        self.__HSVBuffer = None
        self.maskedImageContainer = None
        self.maskedThumbContainer = None
        self._mask = None  # double underscore mangling conflicts with overriding
        self.maskIsEnabled = False
        self.maskIsSelected = False
        self.colorMaskOpacity = bImage.defaultColorMaskOpacity
        self.mergingFlag = False

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        self._mask = m

    @property
    def filename(self):
        return self.__filename

    @filename.setter
    def filename(self, s):
        self.__filename = s

    @property
    def rPixmap(self):
        return self.__rPixmap

    @rPixmap.setter
    def rPixmap(self, pixmap):
        self.__rPixmap = pixmap

    @property
    def hspbBuffer(self):
        return self.__hspbBuffer

    @hspbBuffer.setter
    def hspbBuffer(self, buffer):
        self.__hspbBuffer = buffer

    @property
    def LabBuffer(self):
        return self.__LabBuffer

    @LabBuffer.setter
    def LabBuffer(self, buffer):
        self.__LabBuffer = buffer

    @property
    def HSVBuffer(self):
        return self.__HSVBuffer

    @HSVBuffer.setter
    def HSVBuffer(self, buffer):
        self.__HSVBuffer = buffer

    #################################
    # convenience comparison operators,
    # implicitely used by 'in' op.
    #################################
    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def getHspbBuffer(self):
        """
        return the image buffer in color mode HSpB.
        Override to enable buffering
        @return: HSPB buffer
        @rtype: ndarray
        """
        self.hspbBuffer = rgb2hspVec(QImageBuffer(self)[:, :, :3][:, :, ::-1])
        return self.hspbBuffer

    def getLabBuffer(self):
        """
        return the image buffer in color mode Lab.
        Override to enable buffering
       """
        self.LabBuffer = sRGB2LabVec(QImageBuffer(self)[:, :, :3][:, :, ::-1])
        return self.LabBuffer

    def getHSVBuffer(self):
        """
        return the image buffer in color mode HSV.
        Override to enable buffering
        """
        self.HSVBuffer = cv2.cvtColor(QImageBuffer(self)[:, :, :3], cv2.COLOR_BGR2HSV)
        return self.HSVBuffer

    def cacheInvalidate(self):
        """
        Invalidate cache buffers.
        (called by applyToStack after layer.execute)
        """
        self.hspbBuffer = None
        self.LabBuffer = None
        self.HSVBuffer = None
        # maskedxxxContainer objects are meant to be
        # of type bImage, or a subclass of bImage : we
        # try to invalidate their cache buffers recursively.
        try:
            self.maskedImageContainer.cacheInvalidate()
        except AttributeError:
            pass
        try:
            self.maskedThumbContainer.cacheInvalidate()
        except AttributeError:
            pass

    def updatePixmap(self, maskOnly=False):
        """
        To respect the Substitutability Principle of Liskov
        for subtypes, we keep identical signatures for all
        overriding methods, so we define here an unused parameter
        maskOnly.
        @param maskOnly:
        @type maskOnly: boolean
        """
        self.rPixmap = QPixmap.fromImage(self)

    def waveFront(self):
        """
        Experimental waveFront. Unused yet
        @return:
        @rtype:
        """
        wfi = QImage(QSize(self.width(), 256), QImage.Format_ARGB32)
        wfi.fill(Qt.black)
        wfiBuf = QImageBuffer(wfi)[:, :, :3]
        frameWidth = 1
        buf = QImageBuffer(self)
        for x in range(0, self.width(), frameWidth):
            bufFrame = buf[:, x:x + frameWidth, 1]
            hist, bins = np.histogram(bufFrame, bins=128, range=(0, 255), density=True)
            wfiBuf[::2, x:x + frameWidth, :] = (hist * 256000)[..., np.newaxis, np.newaxis]

    def histogram(self, size=QSize(200, 200), bgColor=Qt.white, range=(0, 255),
                  chans=channelValues.RGB, chanColors=Qt.gray, mode='RGB', addMode='', clipping_threshold=0.02):
        """
        Plots the image histogram with the specified color mode and channels.
        Channel curves are scaled individually to fit the height of the plot.
        @param size: size of the histogram plot
        @type size: int or QSize
        @param bgColor: background color
        @type bgColor: QColor
        @param range: plotted data range
        @type range: 2-uple of int or float
        @param chans: channels to plot B=0, G=1, R=2
        @type chans: list of indices
        @param chanColors: color or 3-uple of colors
        @type chanColors: QColor or 3-uple of QColor
        @param mode: color mode ((one among 'RGB', 'HSpB', 'Lab', 'Luminosity')
        @type mode: str
        @param addMode:
        @type addMode:
        @param clipping_threshold: alert threshold for clipped areas
        @type clipping_threshold: float
        @return: histogram plot
        @rtype: QImage
        """
        binCount = 85  # 255 = 85 * 3
        if type(size) is int:
            size = QSize(size, size)
        # scaling factor for bin drawing
        spread = float(range[1] - range[0])
        scaleH = size.width() / spread
        upMargin = 10  # keep space for indicators on image top

        # per channel histogram function
        def drawChannelHistogram(painter, hist, bin_edges, color):
            # Draw histogram for a single channel.
            # param painter: QPainter
            # param hist: histogram to draw
            # To emphasize significant values we try to clip the first bin to max height of the others
            MA = max(hist)
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    MA_I = 1.0 / MA
                if not np.isfinite(MA_I):
                    raise ValueError
            except (ValueError, ArithmeticError, FloatingPointError, ZeroDivisionError):
                # if MA is too small we do not draw the histogram for this channel
                return
            M = max(hist[1:])
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    M_I = 1.0 / M
                if not np.isfinite(M_I):
                    raise ValueError
            except (ValueError, ArithmeticError, FloatingPointError, ZeroDivisionError):
                # if M is too small we do not clip the first bin
                M_I = MA_I
            imgH = size.height()
            scaleV = imgH * M_I
            # drawing  trapezia instead of rectangles to quickly "smooth" the histogram
            poly = QPolygonF()
            poly.append(QPointF(range[0], imgH))  # bottom left point
            for i, y in enumerate(hist):
                h = scaleV * y
                poly.append(QPointF((bin_edges[i] - range[0]) * scaleH, max(imgH - h, upMargin)))
                # clipping indicators
                if i == 0 or i == len(hist) - 1:
                    left = bin_edges[0 if i == 0 else -1] * scaleH
                    left = left - (10 if i > 0 else 0)  # shift the indicator at right
                    percent = hist[i] * (bin_edges[i + 1] - bin_edges[i])
                    if percent > clipping_threshold:
                        # set the color of the indicator according to percent value
                        nonlocal gPercent
                        gPercent = min(gPercent, np.clip((0.05 - percent) / 0.03, 0, 1))
                        painter.fillRect(left, 0, 10, 10, QColor(255, 255 * gPercent, 0))
            # complete last bin
            poly.append(QPointF((bin_edges[-1] - range[0]) * scaleH, max(imgH - h, upMargin)))
            # draw the filled polygon
            poly.append(QPointF(poly.constLast().x(), imgH))  # bottom right point
            path = QPainterPath()
            path.addPolygon(poly)
            path.closeSubpath()
            painter.setPen(Qt.NoPen)
            painter.fillPath(path, QBrush(color))

        # end of drawChannelHistogram

        # green percent for clipping indicators
        gPercent = 1.0
        buf = None
        if mode == 'RGB':
            buf = QImageBuffer(self)[:, :, :3][:, :, ::-1]
        elif mode == 'HSV':
            buf = self.getHSVBuffer()
        elif mode == 'HSpB':
            buf = self.getHspbBuffer()
        elif mode == 'Lab':
            buf = self.getLabBuffer()
        elif mode == 'Luminosity':
            chans = []
        # drawing the histogram onto img
        img = trackImage(size.width(), size.height(), QImage.Format_ARGB32)
        img.fill(bgColor)

        qp = QPainter(img)
        if type(chanColors) is QColor or type(chanColors) is Qt.GlobalColor:
            chanColors = [chanColors] * 3
        # compute histograms
        # bins='auto' sometimes causes a huge number of bins ( >= 10**9) and memory error
        # even for small data size (<=250000), so we don't use it.
        if mode == 'Luminosity' or addMode == 'Luminosity':
            bufL = cv2.cvtColor(QImageBuffer(self)[:, :, :3], cv2.COLOR_BGR2GRAY)[..., np.newaxis]
            hist, bin_edges = np.histogram(bufL, range=range, bins=binCount, density=True)
            drawChannelHistogram(qp, hist, bin_edges, Qt.gray)
        hist_L, bin_edges_L = [0] * len(chans), [0] * len(chans)
        for i, ch in enumerate(chans):
            buf0 = buf[:, :, ch]
            hist_L[i], bin_edges_L[i] = np.histogram(buf0, range=range, bins=binCount, density=True)
            drawChannelHistogram(qp, hist_L[i], bin_edges_L[i], chanColors[ch])
            # subsequent images are added using composition mode Plus
            # qp.setCompositionMode(QPainter.CompositionMode_Plus)  # uncomment for semi-transparent hists
        qp.end()

        img.drawingScale, img.drawingWidth = scaleH, size.width()
        return img


QImageFormats = {0: 'invalid',
                 1: 'mono',
                 2: 'monoLSB',
                 3: 'indexed8',
                 4: 'RGB32',
                 5: 'ARGB32',
                 6: 'ARGB32 Premultiplied',
                 7: 'RGB16',
                 8: 'ARGB8565 Premultiplied',
                 9: 'RGB666',
                 10:'ARGB6666 Premultiplied',
                 11:'RGB555',
                 12:'ARGB8555 Premultiplied',
                 13: 'RGB888',
                 14: 'RGB444',
                 15: 'ARGB4444 Premultiplied'}


def ndarrayToQImage(ndimg, format=QImage.Format_ARGB32):
    """
    Converts a 3D numpy ndarray to a QImage. No sanity check is
    done regarding the compatibility of the ndarray shape with
    the QImage format.
    @param ndimg: The ndarray to convert, ndimg.data order must be BGRA (little-endian arch.) or ARGB (big-endian)
    @type ndimg: ndarray, dtype np.uint8
    @param format: The QImage format (default ARGB32)
    @type format:
    @return: The converted image
    @rtype: QImage
    """
    if ndimg.ndim != 3 or ndimg.dtype != 'uint8':
        raise ValueError(
            "ndarray2QImage : array must be 3D with dtype=uint8, found ndim=%d, dtype=%s" % (ndimg.ndim, ndimg.dtype))
    bytePerLine = ndimg.shape[1] * ndimg.shape[2]
    if ndimg.size != ndimg.shape[0] * bytePerLine:
        raise ValueError("ndarrayToQImage : conversion error")
    # build QImage from buffer
    qimg = QImage(ndimg.data, ndimg.shape[1], ndimg.shape[0], bytePerLine, format)
    # keep a ref. to buffer to protect it from garbage collector
    qimg.buf_ = ndimg.data
    if qimg.format() == QImage.Format_Invalid:
        raise ValueError("ndarrayToQImage : wrong conversion")
    return qimg


def QImageBuffer(qimg):
    """
    Returns the buffer of a QImage as a numpy ndarray, dtype=uint8. The size of the
    3rd axis (channels) depends on the image type.
    Channels are always returned in BGRA order, regardless of architecture.
    Format 1 bit per pixel is not supported.
    Performance : 20 ms for a 15 Mpx image.
    @param qimg:
    @type qimg: QImage
    @return: The buffer array
    @rtype: numpy ndarray, shape = (h,w, bytes_per_pixel), dtype=uint8
    """
    # pixel depth
    bpp = qimg.depth()
    if bpp == 1:
        raise ValueError("QImageBuffer : unsupported image format 1 bit per pixel")
    # Bytes per pixel
    Bpp = bpp // 8
    # Get image buffer
    # Calling bits() performs a deep copy of the buffer,
    # suppressing all dependencies due to implicit data sharing.
    # To avoid deep copy use constBits() instead (Caution : it returns a read-only buffer).
    ptr = qimg.bits()  # type memoryview, items are bytes : ptr.itemsize = 1
    # convert buffer to ndarray and reshape
    h, w = qimg.height(), qimg.width()
    buf = np.asarray(ptr, dtype=np.uint8).reshape(h, w, Bpp)  # specifying dtype is mandatory to prevent copy of data
    if bImage.bigEndian:
        return buf[..., ::-1]
    else:
        return buf
