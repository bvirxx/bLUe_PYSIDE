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

import numpy as np
from PySide2.QtCore import QSize, QRect, QPoint, Qt
from PySide2.QtGui import QImage, QPixmap, QColor, QPainter

from bLUeCore.SavitskyGolay import SavitzkyGolayFilter
from bLUeGui.colorCIE import sRGB2LabVec
from bLUeGui.colorCube import rgb2hspVec
from bLUeGui.graphicsSpline import channelValues


class bImage(QImage):
    """
    Base class for all bLUe images.
    Inherits from QImage. Adds a mask, a pixmap
    and a bunch of caches encapsulated as properties.
    The pixmap is synchronized with the image by the
    method updatePixmap().
    """
    @classmethod
    def fromImage(cls, img, parentImage=None):
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
        self.mask = None
        self.maskIsEnabled = False
        self.maskIsSelected = False
        self.colorMaskOpacity = 255

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

    def histogram(self, size=QSize(200, 200), bgColor=Qt.white, range=(0, 255),
                  chans=channelValues.RGB, chanColors=Qt.gray, mode='RGB', addMode=''):
        """
        Plot the image histogram with the
        specified color mode and channels.
        Histograms are smoothed using a Savisky-Golay filter and curves are scaled individually
        to fit the height of the plot.
        @param size: size of the histogram plot
        @type size: int or QSize
        @param bgColor: background color
        @type bgColor: QColor
        @param range: plot data range
        @type range: 2-uple of int or float
        @param chans: channels to plot b=0, G=1, R=2
        @type chans: list of indices
        @param chanColors: color or 3-uple of colors
        @type chanColors: QColor or 3-uple of QColor
        @param mode: color mode ((one among 'RGB', 'HSpB', 'Lab', 'Luminosity')
        @type mode: str
        @param addMode:
        @type addMode:
        @return: histogram plot
        @rtype: QImage
        """
        # convert size to QSize
        if type(size) is int:
            size = QSize(size, size)
        # alert threshold for clipped areas
        clipping_threshold = 0.02
        # clipping threshold for black and white points
        # scaling factor for the bin edges
        spread = float(range[1] - range[0])
        scale = size.width() / spread

        # per channel histogram function
        def drawChannelHistogram(painter, hist, bin_edges, color):
            # Draw the (smoothed) histogram for a single channel.
            # param painter: QPainter
            # param hist: histogram to draw
            # smooth the histogram (first and last bins excepted) for a better visualization of clipping.
            hist = np.concatenate(([hist[0]], SavitzkyGolayFilter.filter(hist[1:-1]), [hist[-1]]))
            M = max(hist[1:-1])
            # draw histogram
            imgH = size.height()
            for i, y in enumerate(hist):
                try:
                    h = int(imgH * y / M)
                except (ValueError, ArithmeticError):
                    # don't draw the channel histogram if M is too small:
                    # It may happen when channel values are concentrated
                    # on the first and/or last bins.
                    return
                h = min(h, imgH - 1)  # height of rect must be < height of img, otherwise fillRect does nothing
                rect = QRect(int((bin_edges[i] - range[0]) * scale), max(img.height() - h, 0),
                             int((bin_edges[i + 1] - bin_edges[i]) * scale+1), h)
                painter.fillRect(rect, color)
                # clipping indicators
                if i == 0 or i == len(hist)-1:
                    left = bin_edges[0 if i == 0 else -1]
                    if range[0] < left < range[1]:
                        continue
                    left = left - (10 if i > 0 else 0)
                    percent = hist[i] * (bin_edges[i+1]-bin_edges[i])
                    if percent > clipping_threshold:
                        # calculate the color of the indicator according to percent value
                        nonlocal gPercent
                        gPercent = min(gPercent, np.clip((0.05 - percent) / 0.03, 0, 1))
                        painter.fillRect(left, 0, 10, 10, QColor(255, 255*gPercent, 0))
        # green percent for clipping indicators
        gPercent = 1.0
        bufL = cv2.cvtColor(QImageBuffer(self)[:, :, :3], cv2.COLOR_BGR2GRAY)[..., np.newaxis]  # returns Y (YCrCb) : Y = 0.299*R + 0.587*G + 0.114*B
        buf = None  # TODO added 5/11/18 validate
        if mode == 'RGB':
            buf = QImageBuffer(self)[:, :, :3][:, :, ::-1]  # RGB
        elif mode == 'HSV':
            buf = self.getHSVBuffer()
        elif mode == 'HSpB':
            buf = self.getHspbBuffer()
        elif mode == 'Lab':
            buf = self.getLabBuffer()
        elif mode == 'Luminosity':
            chans = []
        img = QImage(size.width(), size.height(), QImage.Format_ARGB32)
        img.fill(bgColor)
        qp = QPainter(img)
        try:
            if type(chanColors) is QColor or type(chanColors) is Qt.GlobalColor:
                chanColors = [chanColors]*3
            # compute histograms
            # bins='auto' sometimes causes a huge number of bins ( >= 10**9) and memory error
            # even for small data size (<=250000), so we don't use it.
            # This is a numpy bug : in the module function_base.py
            # a reasonable upper bound for bins should be chosen to prevent memory error.
            if mode == 'Luminosity' or addMode == 'Luminosity':
                hist, bin_edges = np.histogram(bufL, bins=100, density=True)
                drawChannelHistogram(qp, hist, bin_edges, Qt.gray)
            hist_L, bin_edges_L = [0]*len(chans), [0]*len(chans)
            for i, ch in enumerate(chans):
                buf0 = buf[:, :, ch]
                hist_L[i], bin_edges_L[i] = np.histogram(buf0, bins=100, density=True)
                # to prevent artifacts, the histogram bins must be drawn
                # using the composition mode source_over. So, we use
                # a fresh QImage for each channel.
                tmpimg = QImage(size, QImage.Format_ARGB32)
                tmpimg.fill(bgColor)
                tmpqp = QPainter(tmpimg)
                try:
                    drawChannelHistogram(tmpqp, hist_L[i], bin_edges_L[i], chanColors[ch])
                finally:
                    tmpqp.end()
                # add the channnel hist to img
                qp.drawImage(QPoint(0,0), tmpimg)
                # subsequent images are added using composition mode Plus
                qp.setCompositionMode(QPainter.CompositionMode_Plus)
        finally:
            qp.end()
        buf = QImageBuffer(img)
        # if len(chans) > 1, clip gray area to improve the aspect of the histogram
        if len(chans) > 1:
            buf[:, :, :3] = np.where(np.min(buf, axis=-1)[:, :, np.newaxis] >= 100,
                                     np.array((100, 100, 100))[np.newaxis, np.newaxis, :], buf[:, :, :3])
        return img


QImageFormats = {0: 'invalid', 1: 'mono', 2: 'monoLSB', 3: 'indexed8', 4: 'RGB32', 5: 'ARGB32',6: 'ARGB32 Premultiplied',
                 7: 'RGB16', 8: 'ARGB8565 Premultiplied', 9: 'RGB666',10: 'ARGB6666 Premultiplied', 11: 'RGB555',
                 12: 'ARGB8555 Premultiplied', 13: 'RGB888', 14: 'RGB444', 15: 'ARGB4444 Premultiplied'}


def ndarrayToQImage(ndimg, format=QImage.Format_ARGB32):
    """
    Converts a 3D numpy ndarray to a QImage. No sanity check is
    done regarding the compatibility of the ndarray shape with
    the QImage format.
    @param ndimg: The ndarray to be converted
    @type ndimg: ndarray
    @param format: The QImage format (default ARGB32)
    @type format:
    @return: The converted image
    @rtype: QImage
    """
    if ndimg.ndim != 3 or ndimg.dtype != 'uint8':
        raise ValueError("ndarray2QImage : array must be 3D with dtype=uint8, found ndim=%d, dtype=%s" % (ndimg.ndim, ndimg.dtype))
    bytePerLine = ndimg.shape[1] * ndimg.shape[2]
    if len(np.ravel(ndimg).data) != ndimg.shape[0]*bytePerLine:
        raise ValueError("ndarrayToQImage : conversion error")
    # build QImage from buffer
    qimg = QImage(ndimg.data, ndimg.shape[1], ndimg.shape[0], bytePerLine, format)
    if qimg.format() == QImage.Format_Invalid:
        raise ValueError("ndarrayToQImage : wrong conversion")
    return qimg


def QImageBuffer(qimg):
    """
    Returns the buffer of a QImage as a numpy ndarray, dtype=uint8. The size of the
    3rd axis (raw pixels) depends on the image type. Pixels are in
    BGRA order (little endian arch. (intel)) or ARGB (big  endian arch.)
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
    return np.asarray(ptr, dtype=np.uint8).reshape(h, w, Bpp)  # specifying dtype is mandatory to prevent copy of data
