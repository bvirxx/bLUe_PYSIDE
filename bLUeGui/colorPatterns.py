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
from .colorCube import hsv2rgbVec, hsp2rgb, rgb2hsp, rgb2hspVec, hsv2rgb, rgb2hsB, rgb2hsBVec, hsp2rgbVec
from PySide6.QtGui import QImage

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
    For fast display, the correspondence with RGB values is tabulated for each brightness.
    """
    # hue rotation
    rotation = 315
    # default brightness
    defaultBr = 0.45

    def __init__(self, w, h, converter, bright=defaultBr, border=0.0):
        """
        Builds a (hue, sat) color wheel image of size (w, h)
        For fast display, the correspondence with RGB values is tabulated
        for each value of the brightness.
        @param w: image width
        @type w: int
        @param h: image height
        @type h: int
        @param converter: color space converter
        @type converter: cmConverter
        @param bright: image brightness
        @type bright: int
        @param border: image border
        @type border: int
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

        # center  : i1 = i - cx, j1 = -j + cy
        cx = w / 2
        cy = h / 2
        coord = coord + [-cx, cy]  # np.array([-cx, cy])

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
        pArray = np.arange(101, dtype=np.float) / 100.0
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
        Set brightness and update image
        @param pb: perceived brightness (range 0,..,1)
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
        @param h: hue in range 0..360
        @param s: saturation in range 0..1
        @return: cartesian coordinates
        """
        cx = self.width() / 2
        cy = self.height() / 2
        x, y = (cx - self.border) * s * np.cos((h - self.rotation) * np.pi / 180.0), \
               (cy - self.border) * s * np.sin((h - self.rotation) * np.pi / 180.0)
        x, y = x + cx, -y + cy
        return x, y

    def GetPointVec(self, hsarray):
        """
        convert (hue, sat) values to cartesian coordinates
        on the color wheel (origin top-left corner).
        Vectorized version of GetPoint
        @param hsarray
        @type: hsarray: ndarray, shape=(w,h,2)
        @return: cartesian coordinates
        @rtype: ndarray, shape=(w,h,2)
        """
        h, s = hsarray[:, :, 0], hsarray[:, :, 1]
        cx = self.width() / 2
        cy = self.height() / 2
        x, y = (cx - self.border) * s * np.cos((h - self.rotation) * np.pi / 180.0), \
               (cy - self.border) * s * np.sin((h - self.rotation) * np.pi / 180.0)
        x, y = x + cx, - y + cy
        return np.dstack((x, y))


class brightnessPattern(bImage):
    """
    linear gradient of brightnesses for fixed hue and sat.
    """

    def __init__(self, w, h, converter, hue, sat):
        """
        Build a linear gradient of size (w, h) with variable brightnesses
        and fixed hue and sat. The parameter converter defines the color space
        which is used (HSV, HSpB,...).
        @param w: image width
        @type w: int
        @param h: image height
        @type h: int
        @param converter: color space converter
        @type converter: cmConverter
        @param hue: hue value
        @type hue: int or float
        @param sat: saturation value
        @type sat: int or float
        @return: the image of gradient
        @rtype: bImage
        """
        super().__init__(w, h, QImage.Format_ARGB32)
        self.cModel = converter
        imgBuf = QImageBuffer(self)
        # set alpha
        imgBuf[:, :, 3] = 255
        imgBuf = imgBuf[:, :, :3][:, :, ::-1]
        # build the array of (hue, sat, b), b in [0,1], shape=(w,3)
        a = np.zeros((w, 2), dtype=np.float) + [hue, sat]
        hsArray = np.concatenate((a, (np.arange(w) / (w - 1))[..., np.newaxis]), axis=1)
        # convert to rgb and broadcast to imgBuf
        imgBuf[:, :, :] = converter.cm2rgbVec(hsArray[np.newaxis, ...])
        self.updatePixmap()
