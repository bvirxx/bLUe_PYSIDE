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
from PySide6 import QtCore
from PySide6.QtCore import Qt, QPointF, QLineF
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsPixmapItem

from bLUeTop.utils import QbLUeSlider
from .colorCube import hsv2rgbVec, hsp2rgb, rgb2hsp, rgb2hspVec, hsv2rgb, rgb2hsB, rgb2hsBVec, hsp2rgbVec
from PySide6.QtGui import QImage, QPixmap, QPainter, QPainterPath, QColor

from bLUeGui.bLUeImage import bImage, QImageBuffer


class cmConverter(object):
    """
    Gather conversion functions color space<-->RGB
    """

    def __init__(self):
        self.cm2rgb, self.cm2rgbVec, rgb2cm, rgb2cmVec = (None,) * 4


##########################################
# init color converters for HSpB and HSB
#########################################

cmHSP = cmConverter()
cmHSP.cm2rgb, cmHSP.cm2rgbVec, cmHSP.rgb2cm, cmHSP.rgb2cmVec = hsp2rgb, hsp2rgbVec, rgb2hsp, rgb2hspVec

cmHSB = cmConverter()
cmHSB.cm2rgb, cmHSB.cm2rgbVec, cmHSB.rgb2cm, cmHSB.rgb2cmVec = hsv2rgb, hsv2rgbVec, rgb2hsB, rgb2hsBVec


class hueSatPattern(bImage):
    """
    (hue, sat) color wheel image.
    """
    # wheel rotation
    rotation = 315
    # default brightness
    defaultBr = 0.45

    def __init__(self, w, h, converter, bright=defaultBr, border=0):
        """
        Builds a (hue, sat) color wheel image of size (w, h)
        For fast display, the correspondence with RGB values is tabulated
        for each brightness.

        :param w: image width
        :type  w: int
        :param h: image height
        :type  h: int
        :param converter: color space converter
        :type  converter: cmConverter
        :param bright: image brightness
        :type  bright: float
        :param border: image border
        :type  border: int
        """
        w += 2 * border
        h += 2 * border
        super().__init__(w, h, QImage.Format_ARGB32)
        self.pb = bright
        self.hsArray = None
        self.cModel = converter
        # uninitialized ARGB image
        self.border = border
        imgBuf = QImageBuffer(self)
        # set alpha channel
        imgBuf[:, :, 3] = 255
        # get RGB buffer
        imgBuf = imgBuf[:, :, :3][:, :, ::-1]

        # init array of grid (cartesian) coordinates
        coord = np.dstack(np.meshgrid(np.arange(w), - np.arange(h)))

        # wheel center  : i1 = i - cx, j1 = -j + cy
        self.center = QPointF(w, h) / 2
        cx, cy = self.center.toTuple()
        coord = coord + [-cx, cy]

        # init hue and sat arrays as polar coordinates.
        # arctan2 values are in range -pi, pi
        hue = np.arctan2(coord[:, :, 1], coord[:, :, 0]) * (180.0 / np.pi) + self.rotation
        # hue range 0..360, sat range 0..1
        hue = hue - np.floor(hue / 360.0) * 360.0
        sat = np.linalg.norm(coord, axis=2, ord=2) / (cx - border)
        np.minimum(sat, 1.0, out=sat)

        # init a stack of image buffers, one for each brightness in integer range 0..100
        hsBuf = np.dstack((hue, sat))[np.newaxis, :]  # shape (1, h, w, 2)
        hsBuf = np.tile(hsBuf, (101, 1, 1, 1))  # (101, h, w, 2)
        pArray = np.arange(101, dtype=float) / 100.0
        pBuf = np.tile(pArray[:, np.newaxis, np.newaxis], (1, h, w))  # 101, h, w
        hspBuf = np.stack((hsBuf[:, :, :, 0], hsBuf[:, :, :, 1], pBuf), axis=-1)  # 101, h, w, 3
        # convert the buffers to rgb
        self.BrgbBuf = converter.cm2rgbVec(hspBuf)  # shape 101, h, w, 3
        p = int(bright * 100.0)
        # select the right image buffer
        self.hsArray = hspBuf[p, ...]
        imgBuf[:, :, :] = self.BrgbBuf[p, ...]
        self.updatePixmap()

    def setPb(self, pb):
        """
        Set brightness and update image.

        :param pb: perceived brightness (range 0,..,1)
        """
        self.pb = pb
        self.hsArray[:, :, 2] = pb
        imgBuf = QImageBuffer(self)[:, :, :3][:, :, ::-1]
        imgBuf[:, :, :] = self.cModel.cm2rgbVec(self.hsArray)
        self.updatePixmap()

    def GetPoint(self, h, s):
        """
        convert (hue, sat) values to cartesian coordinates
        on the color wheel (origin top-left corner).

        :param h: hue in range 0..360
        :type h: float
        :param s: saturation in range 0..1
        :type s: float
        :return:cartesian coordinates
        :rtype: 2-uple of float
        """
        cx, cy = self.center.toTuple()
        x, y = (cx - self.border) * s * np.cos((h - self.rotation) * np.pi / 180.0), \
               (cy - self.border) * s * np.sin((h - self.rotation) * np.pi / 180.0)
        return x + cx, -y + cy

    def GetPointVec(self, hsarray):
        """
        convert (hue, sat) values to cartesian coordinates
        on the color wheel (origin top-left corner).
        Vectorized version of GetPoint.

        :param hsarray:
        :type hsarray: ndarray, shape=(w,h,2)
        :return: cartesian coordinates
        :rtype: ndarray, shape=(w,h,2), dtype=float
        """
        h, s = hsarray[:, :, 0], hsarray[:, :, 1]
        cx, cy = self.center.toTuple()
        x, y = (cx - self.border) * s * np.cos((h - self.rotation) * np.pi / 180.0), \
               (cy - self.border) * s * np.sin((h - self.rotation) * np.pi / 180.0)
        return np.dstack((x + cx, -y + cy))


class brightnessPattern(bImage):
    """
    linear gradient of brightnesses for fixed hue and sat.
    """

    def __init__(self, w, h, converter, hue, sat):
        """
        Build a linear gradient of size (w, h) with variable brightnesses
        and fixed hue and sat. The parameter converter defines the color space
        which is used (HSV, HSpB,...).
        :param w: image width
        :type  w: int
        :param h: image height
        :type  h: int
        :param converter: color space converter
        :type  converter: cmConverter
        :param hue: hue value, range [0..360[
        :type  hue: int or float
        :param sat: saturation value, range [0..1]
        :type  sat: float
        :return: the image of gradient
        :rtype: bImage
        """
        super().__init__(w, h, QImage.Format_ARGB32)
        self.cModel = converter
        imgBuf = QImageBuffer(self)
        # set alpha
        imgBuf[:, :, 3] = 255
        imgBuf1 = imgBuf[:, :, :3][:, :, ::-1]
        # build the array of (hue, sat, b), b in [0,1]
        a = np.zeros((w, 3), dtype=float)
        a[..., :2] += [hue, sat]
        a[..., 2] = np.arange(w) / (w - 1)
        # convert to rgb and broadcast to imgBuf
        imgBuf1[...] = converter.cm2rgbVec(a[np.newaxis, :, ...])
        self.updatePixmap()


class huePattern(QImage):
    """
    linear gradient of hues for fixed sat and brightness.
    """

    def __init__(self, w, h, converter, sat, br):
        """
        Build a linear gradient of size (w, h) with variable hue
        and fixed sat and brightness. The parameter converter defines the color space
        which is used (HSV, HSpB,...).
        :param w: image width
        :type  w: int
        :param h: image height
        :type  h: int
        :param converter: color space converter
        :type  converter: cmConverter
        :param sat: saturation value, range 0..1
        :type  sat: float
        :param br: brightness value, range 0..1
        :type  br: float
        """
        super().__init__(w, h, QImage.Format_ARGB32)
        self.cModel = converter
        imgBuf = QImageBuffer(self)
        # set alpha
        imgBuf[:, :, 3] = 255
        # build the array of (hue, sat, b), hue in range[0,360[
        a = np.zeros((w, 3), dtype=float)
        a[..., 1:] += [sat, br]
        a[..., 0] = np.arange(w) * 360 / w
        imgBuf1 = imgBuf[:, :, :3][:, :, ::-1]
        imgBuf1[...] = converter.cm2rgbVec(a)


class hueShiftPattern(QImage):
    """
    linear gradient of hues for fixed sat and brightness.
    """

    def __init__(self, w, h, converter, sat, br):
        """
        Build a linear gradient of size (w, h) with variable hue
        and fixed sat and brightness. The parameter converter defines the color space
        which is used (HSV, HSpB,...).
        :param w: image width
        :type  w: int
        :param h: image height
        :type  h: int
        :param converter: color space converter
        :type  converter: cmConverter
        :param sat: saturation value, range 0..1
        :type  sat: float
        :param br: brightness value, range 0..1
        :type  br: float
        """
        super().__init__(w, h, QImage.Format_ARGB32)
        self.cModel = converter
        imgBuf = QImageBuffer(self)
        # set alpha
        imgBuf[:, :, 3] = 255
        # build the array of (hue, sat, b), hue in range[0,360[
        a = np.zeros((h, w, 3), dtype=float)
        a[..., 1:] += [sat, br]
        a[..., 0] = (np.arange(w) * 360 / w)
        delta = (h // 2 - np.arange(h)) / 5
        a[..., 0] += delta[..., np.newaxis]
        a[..., 0] %= 360
        imgBuf1 = imgBuf[:, :, :3][:, :, ::-1]
        imgBuf1[...] = converter.cm2rgbVec(a)


class hueBrShiftPattern(QImage):
    """
    linear gradient of hues for fixed sat and shfted brightnesses.
    """

    def __init__(self, w, h, converter, sat, br):
        """
        Build a linear gradient of size (w, h) with variable hue,
        fixed sat and shifted brightnesses. The parameter converter defines the color space
        which is used (HSV, HSpB,...).
        :param w: image width
        :type  w: int
        :param h: image height
        :type  h: int
        :param converter: color space converter
        :type  converter: cmConverter
        :param sat: saturation value, range 0..1
        :type  sat: float
        :param br: brightness value, range 0..1
        :type  br: float
        """
        super().__init__(w, h, QImage.Format_ARGB32)
        self.cModel = converter
        imgBuf = QImageBuffer(self)
        # set alpha
        imgBuf[:, :, 3] = 255
        # build the array of (hue, sat, b), hue in range[0,360[
        a = np.zeros((h, w, 3), dtype=float)
        a[..., 1:] += [sat, 0.5]
        a[..., 0] = (np.arange(w) * 360 / w)
        delta = 1.0 - (np.arange(h) - h // 2) / h
        a[..., 2] *= delta[..., np.newaxis]
        imgBuf1 = imgBuf[:, :, :3][:, :, ::-1]
        imgBuf1[...] = converter.cm2rgbVec(a)


class graphicsHueShiftPattern(hueShiftPattern, QGraphicsPixmapItem):
    """
    QGraphicsItem huePattern
    """

    def __init__(self, w, h, converter, sat, br):
        """
        Build a linear gradient of size (w, h) with variable brightnesses
        and fixed hue and sat. The parameter converter defines the color space
        which is used (HSV, HSpB,...).
        :param w: image width
        :type  w: int
        :param h: image height
        :type  h: int
        :param converter: color space converter
        :type  converter: cmConverter
        :param sat: saturation value, range 0..1
        :type  sat: float
        :param br: brightness value, range 0..1
        :type  br: float
        """
        super().__init__(w, h, converter, sat, br)
        QGraphicsPixmapItem.__init__(self, QPixmap.fromImage(self))


class graphicsHueBrShiftPattern(hueBrShiftPattern, QGraphicsPixmapItem):
    """
    QGraphicsItem huePattern
    """

    def __init__(self, w, h, converter, sat, br):
        """
        Build a linear gradient of size (w, h) with variable brightnesses
        and fixed hue and sat. The parameter converter defines the color space
        which is used (HSV, HSpB,...).
        :param w: image width
        :type  w: int
        :param h: image height
        :type  h: int
        :param converter: color space converter
        :type  converter: cmConverter
        :param sat: saturation value, range 0..1
        :type  sat: float
        :param br: brightness value, range 0..1
        :type  br: float
        """
        super().__init__(w, h, converter, sat, br)
        QGraphicsPixmapItem.__init__(self, QPixmap.fromImage(self))


class colorWheelSampler(QLabel):
    """
    (Hue, Sat) color wheel picker
    """

    colorChanged = QtCore.Signal()
    samplerReleased = QtCore.Signal()

    def __init__(self, w, h):
        """
        :param w: image width
        :type w: int
        :param h: image height
        :type h: int
        """
        super().__init__()
        self.refImage = hueSatPattern(w, h, cmHSP, bright=1.0, border=5)
        self.bareWheel = QPixmap.fromImage(self.refImage)

        self.currentColor = QColor(255, 255, 255)

        self.center = QPointF(w, h) / 2
        cx, cy = self.center.toTuple()
        self.p = QPointF(cx, cy)  #  copy

        self.l1 = QLineF(cx - 5, cy, cx + 5, cy)
        self.l2 = QLineF(cx, cy - 5, cx, cy + 5)

        self.radius, self.theta = 0.0, 0.0  # used by mousePressEvent and mouseMoveEvent
        self.w, self.h = w, h  # used by paintEvent

        self.qp = QPainter()

        self.clPath = QPainterPath()
        self.clPath.addEllipse(0.0, 0.0, w, h)

        self.setPixmap(self.bareWheel)

    def paintEvent(self, e):
        self.qp.begin(self)
        self.qp.setClipPath(self.clPath)
        self.qp.drawPixmap(0, 0, self.bareWheel)
        self.qp.setPen(Qt.black)
        # central crosshair
        self.qp.drawLine(self.l1)
        self.qp.drawLine(self.l2)
        # current radius
        u = self.p - self.center
        u *= max(self.w, self.h) / max(u.manhattanLength(), 0.001)
        self.qp.drawLine(QLineF(self.center, self.p + u))
        self.qp.drawEllipse(self.p, 6.0, 6.0)
        self.qp.end()

    def setCurrentColor(self, h, v):
        """
        Set current color from hue ans saturation values.
        Brightness is set to the brightness of refImage.
        :param h: hue
        :type h: float, range 0..360
        :param v: saturation
        :type v: float, range 0..1
        """
        x, y = self.refImage.GetPoint(h, v)
        self.p.setX(x)
        self.p.setY(y)
        self.currentColor = self.refImage.pixelColor(self.p.toPoint())

    def mousePressEvent(self, e):
        p = e.position()
        x, y = p.toTuple()
        self.radius = np.sqrt((x - self.center.x()) ** 2 + (y - self.center.y()) ** 2)
        self.theta = np.arctan2(-y + self.center.y(), x - self.center.x())
        self.p.setX(x)
        self.p.setY(y)
        self.update()
        
    def mouseMoveEvent(self, e):
        p = e.position()
        x, y = p.toTuple()
        modifiers = e.modifiers()
        if modifiers == Qt.ControlModifier:
            # constant radius = self.radius
            self.theta = np.arctan2(y - self.center.y(), x - self.center.x())
            x, y = self.radius * np.cos(self.theta) + self.center.x(), self.radius * np.sin(self.theta) + self.center.y()
        else:
            self.theta = np.arctan2(y - self.center.y(), x - self.center.x())
            self.radius = np.sqrt((x - self.center.x()) ** 2 + (y - self.center.y()) ** 2)
        self.p.setX(x)
        self.p.setY(y)
        self.update()

    def mouseReleaseEvent(self, e):
        self.samplerReleased.emit()

    def update(self):
        self.currentColor = self.refImage.pixelColor(self.p.toPoint())
        self.colorChanged.emit()
        self.repaint()


class colorWheelChooser(QWidget):
    """
    (Hue, Sat) color wheel picker, displaying a a slider and a sample of the current color.
    """
    @staticmethod
    def getBr(v):
        return v / 200 + 0.75

    @staticmethod
    def getValue(br):
        return (br - 0.75) * 200

    def __init__(self, w, h, name=''):
        super().__init__()
        self.w, self.h = w, h
        self.sw, self.sh = int(w / 10), int(h / 10)
        self.sampler = colorWheelSampler(w, h)
        self.sample = QLabel()
        self.brSlider = QbLUeSlider(Qt.Horizontal)
        self.brSlider.setMinimum(0)
        self.brSlider.setMaximum(100)
        self.brSlider.setSliderPosition(50)
        pxmp = QPixmap(int(w / 10), int(h / 10))
        pxmp.fill(self.sampler.currentColor)
        self.sample.setPixmap(pxmp)

        vl = QVBoxLayout()
        vl.addWidget(self.sampler)
        vl.addWidget(self.brSlider)
        hl = QHBoxLayout()
        lb = QLabel()
        lb.setText(name)
        hl.addWidget(lb)
        hl.addWidget(self.sample)
        vl.addLayout(hl)
        self.setLayout(vl)

        self.sampler.colorChanged.connect(self.update)

    def update(self):
        pxmp = QPixmap(self.sw, self.sh)
        pxmp.fill(self.sampler.currentColor)
        self.sample.setPixmap(pxmp)

    def __getstate__(self):
        h, s, _ = rgb2hsB(*self.sampler.currentColor.getRgb())
        return {'brcoeff': self.getBr(self.brSlider.value()), 'H': h, 'S': s}

    def __setstate__(self, state):
        self.brSlider.setValue(self.getValue(state['brcoeff']))
        self.sampler.setCurrentColor(state['H'], state['S'])
        self.sampler.update()


