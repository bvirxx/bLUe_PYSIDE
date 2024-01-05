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

from os.path import isfile

import numpy as np

from PySide6.QtCore import Qt, QSize

import cv2
from copy import copy

from PySide6.QtGui import QImageReader, QTransform, QBitmap
from PySide6.QtGui import QImage, QColor, QPainter
from PySide6.QtCore import QRect

from bLUeGui.bLUeImage import bImage, ndarrayToQImage
from bLUeTop.colorManagement import icc

from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.colorCube import rgb2hspVec
from bLUeGui.colorCIE import sRGB2LabVec, sRGB_lin2XYZInverse, sRGB_lin2XYZ

from bLUeTop.lutUtils import LUT3DIdentity


class ColorSpace:
    notSpecified = -1
    sRGB = 1


class metadataBag:
    """
    Container for vImage meta data
    """

    def __init__(self, name=''):
        self.name, self.colorSpace, self.rawMetadata, self.profile, self.orientation, self.rating = \
            name, ColorSpace.notSpecified, {}, '', None, 5


class vImage(bImage):
    """
    Versatile image class.
    Base class for multi-layered and interactive image
    classes, and for layer classes. It gathers all image information,
    including meta-data.
    A vImage object holds 4 images:
           - full size (self),
           - thumbnail (self.thumb),
           - hald (self.hald) for LUT3D conversion,
           - mask (self.mask)
    Note 1 : self.mask is instantiated only in the subclass QLayer.
    Note 2 : for the sake of performance self.thumb and self.hald are not synchronized with the image: they are initialized
    and handled independently of the full size image.
    """
    ################
    # max thumbnail size :
    # max(thumb.width(), thumb.height()) <= thumbsize
    thumbSize = 1500
    ################

    ###############
    # default base color : background color and transparent pixels
    defaultBgColor = QColor(128, 128, 128, 255)
    ###############

    ##############
    # Image Mask
    # QImage transformations sometimes give unexpected results
    # with fully transparent pixels. So, we use the red channel to record
    # mask opacity, instead of alpha channel.
    # A mask acts only while image viewing or saving (cf the method vImage.visualizeMask())
    # It has two possible effects, depending on its flags :
    #        - opacity mask : set the layer opacity to the value of the mask red channel;
    #        - color mask : blend mask with layer (mode source over)
    # NOTE 1: The mode "color mask" is meant for viewing only; it should not be used while saving an image !
    # NOTE 2 : Modifying the B, G, A channels of a mask has no effect in mode "opacity mask"
    ##############
    defaultColor_UnMasked = QColor(255, 0, 0, 255)
    defaultColor_Masked = QColor(0, 0, 0, 255)
    defaultColor_UnMasked_SM = QColor(255, 0, 0, 128)
    defaultColor_Masked_SM = QColor(0, 0, 0, 128)
    # segmentation uses the G channel to mark
    # mask pixel as modifiable/unmodifiable
    invalidG = 99  # modifiable if G==99 else unmodifiable
    defaultColor_Invalid = QColor(0, invalidG, 0, 128)
    defaultColor_UnMasked_Invalid = QColor(128, invalidG, 0, 255)

    @staticmethod
    def color2OpacityMask(mask):
        """
        Returns a copy of mask with the opacity channel set
        from the red channel (alpha = red). No thresholding
        is done.
        B, G, R channels are kept unchanged.

        :param mask: mask
        :type mask: QImage
        :return: opacity mask
        :rtype: QImage
        """
        mask = mask.copy()
        buf = QImageBuffer(mask)
        # set alpha channel from red channel
        buf[:, :, 3] = buf[:, :, 2]
        return mask

    @staticmethod
    def color2ViewMask(mask):
        """
        Returns a colored representation of mask,
        B, R, A channels are not modified. mask is kept unchanged.

        :param mask: mask
        :type mask: QImage
        :return: opacity mask
        :rtype: QImage
        """
        mask = mask.copy()
        buf = QImageBuffer(mask)
        # record mask alpha in the G channel
        buf[:, :, 1] = (255 - buf[:, :, 2])  # * 0.75
        return mask

    @staticmethod
    def colorMask2BinaryArray(mask, invert=False):
        """
        Returns a binary array with values 0/255.
        0 corresponds to masked pixels and 255 to unmasked ones.
        mask opacity is defined by the red channel.

        :param mask:
        :type mask: QImage
        :param invert:
        :type invert:
        :return:
        :rtype: ndarray dtype= uint8, shape (h, w)
        """
        buf = QImageBuffer(mask)
        if invert:
            return np.where(buf[:, :, 2] == 0, np.uint8(255), np.uint8(0))
        else:
            return np.where(buf[:, :, 2] == 0, np.uint8(0), np.uint8(255))

    @staticmethod
    def colorMask2QBitmap(mask, invert=False):
        """
        Returns a QBitmap object, with size identical to mask size.
        If invert is False (default) masked (resp. unmasked) pixels
        correspond to 0 (resp 1).
        mask opacity is defined by the red channel.

        :param mask: color mask
        :type mask: QImage
        :param invert:
        :type invert: boolean
        :return:
        :rtype: QBitmap
        """
        a = vImage.colorMask2BinaryArray(mask, invert=invert)
        return QBitmap.fromData(QSize(mask.size()), np.packbits(a))

    @classmethod
    def isAllMasked(cls, mask):
        buf = QImageBuffer(mask)
        if np.any(buf[:, :, 2] == cls.defaultColor_UnMasked.red()):
            return False
        return True

    @classmethod
    def isAllUnmasked(cls, mask):
        buf = QImageBuffer(mask)
        if np.any(buf[:, :, 2] == cls.defaultColor_Masked.red()):
            return False
        return True

    @classmethod
    def visualizeMask(cls, img, mask, color=True, inplace=False):
        """
        Blends img with mask. By default, img is copied before blending.
        If inplace is True no copy is made.
        If color is True (default), the mask is drawn over the image with opacity 0.5,
        using its own colors. If color is False the alpha channel of the mask is set from
        its red channel and, next, the mask is drawn over the image using the mode destinationIn :
        destination opacity is set to that of source.

        :param img:
        :type img: QImage
        :param mask:
        :type mask: QImage
        :param color:
        :type color: bool
        :param inplace:
        :type inplace: boolean
        :return:
        :rtype: QImage
        """
        # make a copy of img
        if not inplace:
            img = QImage(img)
        qp = QPainter(img)
        # color mask
        if color:
            # draw mask over image
            qp.setCompositionMode(QPainter.CompositionMode_SourceOver)
            qp.drawImage(QRect(0, 0, img.width(), img.height()), cls.color2ViewMask(mask))
        # opacity mask
        else:
            # mode DestinationIn (set image opacity to mask opacity)
            qp.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            omask = vImage.color2OpacityMask(mask)
            qp.drawImage(QRect(0, 0, img.width(), img.height()), omask)
        qp.end()
        return img

    @staticmethod
    def maskDilate(mask, ks=5, iterations=1):
        """
        Increases the masked region by applying
        a (ks, ks) min filter. Returns the dilated mask.
        The source mask is not modified.

        :param mask:
        :type mask: image ndarray
        :param ks: kernel size, should be odd
        :type ks: int
        :param iterations: filter iteration count
        :type iterations: int
        :return: the dilated mask
        :rtype: ndarray
        """
        if iterations <= 0:
            return mask
        kernel = np.ones((ks, ks), np.uint8)
        # CAUTION erode decreases values (min filter), so it extends the masked part of the image
        mask = cv2.erode(mask, kernel, iterations=iterations)
        return mask

    @staticmethod
    def maskErode(mask, ks=5, iterations=1):
        """
        Reduces the masked region by applying
        a (ks, ks) max filter. Returns the eroded mask.
        The source mask is not modified.

        :param mask:
        :type mask: image ndarray
        :param ks: kernel size, should be odd
        :type ks: int
        :param iterations: filter iteration count
        :type iterations: int
        :return: the eroded mask
        :rtype: ndarray
        """
        if iterations <= 0:
            return mask
        kernel = np.ones((ks, ks), np.uint8)
        # CAUTION dilate increases values (max filter), so it reduces the masked region of the image
        mask = cv2.dilate(mask, kernel, iterations=iterations)
        return mask

    @staticmethod
    def maskSmooth(mask, ks=11):
        """
        Smooths the mask by applying a mean kernel.
        The source mask is not modified.

        :type mask: image ndarray
        :param ks: kernel size, should be odd
        :type ks: int
        :return: the smoothed mask
        :rtype: ndarray
        """
        kernelMean = np.ones((ks, ks), float) / (ks * ks)
        return cv2.filter2D(mask, -1, kernelMean)  # -1 : keep depth unchanged

    def __init__(self, filename=None, cv2Img=None, QImg=None, format=QImage.Format_ARGB32,
                 name='', colorSpace=-1, orientation=None, rating=5, meta=None, rawMetadata=None, profile=b''):
        """
        With no parameter, builds a null image.
        image is assumed to be in the color space sRGB : colorSpace value is used only as meta data.

        :param filename: path to file
        :type filename: str
        :param cv2Img: data buffer
        :type cv2Img: ndarray
        :param QImg: image
        :type QImg: QImage
        :param format: QImage format (default QImage.Format_ARGB32)
        :type format: QImage.Format
        :param name: image name
        :type name: str
        :param colorSpace: color space (default : not specified)
        :type colorSpace: MarkedImg.colorSpace
        :param orientation: Qtransform object (default None)
        :type orientation: Qtransform
        :param meta: metadata instance (default None)
        :type meta: MarkedImg.metadataBag
        :param rawMetadata: dictionary
        :type rawMetadata: dictionary
        :param profile: embedded profile (default b'')
        :type profile: bytes
        """
        # formatted EXIF data (str)
        self.imageInfo = 'no EXIF data'  # default

        if rawMetadata is None:
            rawMetadata = {}
        self.isModified = False
        self.profileChanged = False
        self.__rect, self.marker = None, None  # selection rectangle, marker
        self.sRects = []  # selection rectangles
        self.isCropped = False
        self.cropTop, self.cropBottom, self.cropLeft, self.cropRight = (0,) * 4
        self.isRuled = False

        # mode flags
        self.useHald = False
        self.hald = None
        self.isHald = False
        self.useThumb = False

        # Caching flag
        self.cachesEnabled = True

        # preview image.
        # The layer stack can be seen as
        # the juxtaposition of two stacks:
        #  - a stack of full sized images
        #  - a stack of thumbnails
        # For the sake of performance, the two stacks are
        # NOT synchronized. Thus, after initialization, the
        # thumbnail should never be calculated from
        # the full size image.
        self.thumb = None
        self.onImageChanged = lambda: 0

        if meta is None:
            # init metadata container
            self.meta = metadataBag()
            self.meta.name, self.meta.colorSpace, self.meta.rawMetadata, self.meta.profile, self.meta.orientation, self.meta.rating = \
                name, colorSpace, rawMetadata, profile, orientation, rating
        else:
            self.meta = meta
        self.setProfile(icc.defaultWorkingProfile)  # default profile
        # self.colorSpace = self.meta.colorSpace
        # self.cmsProfile = icc.defaultWorkingProfile  # possibly does not match colorSpace : call setProfile()
        # self.RGB_lin2XYZ = sRGB_lin2XYZ
        # self.RGB_lin2XYZInverse = sRGB_lin2XYZInverse
        if filename is None and cv2Img is None and QImg is None:
            # create a null image
            super().__init__()
        if filename is not None:
            if not isfile(filename):
                raise ValueError('Cannot find file %s' % filename)
            # load image from file (should be a 8 bits/channel color image)
            if self.meta.orientation is not None:
                tmp = QImage(filename).transformed(self.meta.orientation)
            else:
                tmp = QImage(filename)
            tmp = tmp.convertToFormat(format)
            if tmp.isNull():
                raise ValueError(
                    'Cannot load %s\nSupported image formats\n%s' % (filename, QImageReader.supportedImageFormats()))
            # call to super is mandatory. Shallow copy : no harm !
            super().__init__(tmp)
        elif QImg is not None:
            # build image from QImage, shallow copy
            super().__init__(QImg)
        elif cv2Img is not None:
            # build image from buffer
            super().__init__(ndarrayToQImage(cv2Img, format=format))
        # check format
        if self.depth() != 32:
            raise ValueError('vImage : should be a 8 bits/channel color image')
        self.filename = filename if filename is not None else ''

    @property
    def rect(self):
        return self.__rect

    @rect.setter
    def rect(self, rect):
        self.__rect = rect

    def cropMargins(self):
        return (self.cropLeft, self.cropRight, self.cropTop, self.cropBottom)

    def setCropMargins(self, margins, croptool):
        """
        set image cropping margins to margins and set the positions of crop tool buttons
        accordingly.

        :param margins:
        :type margins: 4-uple of float
        :param croptool:
        :type croptool: cropTool
        """
        self.cropLeft, self.cropRight, self.cropTop, self.cropBottom = margins
        croptool.fit(self)

    def setProfile(self, profile):
        """
        Sets profile related attributes.

        :param profile:
        :type profile: CmsProfile instance
        """
        self.cmsProfile = profile
        if 'srgb' in profile.profile.profile_description.lower():
            self.colorSpace = 1
        else:
            self.colorSpace = 65535
        cr, cg, cb = profile.profile.red_colorant, profile.profile.green_colorant, profile.profile.blue_colorant
        if isinstance(cr, tuple) and isinstance(cg, tuple) and isinstance(cb, tuple):
            self.RGB_lin2XYZ = np.column_stack((cr[0], cg[0], cb[0]))
            self.RGB_lin2XYZInverse = np.linalg.inv(self.RGB_lin2XYZ)
        else:
            self.RGB_lin2XYZ = sRGB_lin2XYZ
            self.RGB_lin2XYZInverse = sRGB_lin2XYZInverse

    def setImage(self, qimg):
        """
        copies qimg to image. Does not update metadata.
        image and qimg must have identical dimensions and type.

        :param qimg: image
        :type qimg: QImage
        """
        # image layer
        if getattr(self, 'sourceImg', None) is not None:
            pass
        buf1, buf2 = QImageBuffer(self), QImageBuffer(qimg)
        if buf1.shape != buf2.shape:
            raise ValueError("QLayer.setImage : new image and layer must have identical shapes")
        buf1[...] = buf2
        self.thumb = None
        self.cacheInvalidate()
        self.updatePixmap()

    def cameraModel(self):
        tmp = [value for key, value in self.meta.rawMetadata.items() if 'model' in key.lower()]
        return tmp[0] if tmp else ''

    def initThumb(self):
        """
        Init the image thumbnail as a QImage. In contrast with
        maskedThumbContainer, thumb is never used as an input image, thus
        there is no need for a type featuring cache buffers.
        Layer thumbs own an attribute parentImage set by the overridden method QLayer.initThumb.
        For non adjustment layers, the thumbnail will never be updated. So, we
        perform a high quality scaling.
        """
        scImg = self.scaled(self.thumbSize, self.thumbSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # With the Qt.SmoothTransformation flag, the output image format is premultiplied
        self.thumb = scImg.convertToFormat(QImage.Format_ARGB32, Qt.DiffuseDither | Qt.DiffuseAlphaDither)

    def getThumb(self):
        """
        init image thumbnail if needed and return it.

        :return: thumbnail
        :rtype: QImage
        """
        if self.thumb is None:
            self.initThumb()
        return self.thumb

    def initHald(self):
        """
        UNUSED
        Builds a hald image (as a QImage) from identity 3D LUT.
        A hald can be viewed as a 3D LUT flattened and reshaped as a 2D array.
        """
        if not self.cachesEnabled:
            return
        s = int(LUT3DIdentity.size ** (3.0 / 2.0)) + 1
        buf0 = LUT3DIdentity.toHaldArray(s, s).haldBuffer
        buf1 = QImageBuffer(self.hald)
        buf1[:, :, :] = buf0
        buf1[:, :, 3] = 255  # added for coherence with the overriding function QLayer.initHald()

    def getHald(self):
        if not self.cachesEnabled:
            s = int(LUT3DIdentity.size ** (3.0 / 2.0)) + 1
            buf0 = LUT3DIdentity.toHaldArray(s, s).haldBuffer
            # self.hald = QLayer(QImg=QImage(QSize(190,190), QImage.Format_ARGB32))
            hald = QImage(QSize(s, s), QImage.Format_ARGB32)
            buf1 = QImageBuffer(hald)
            buf1[:, :, :3] = buf0
            buf1[:, :, 3] = 255
            hald.parentImage = self.parentImage
            return hald
        if self.hald is None:
            self.initHald()
        return self.hald

    def getCurrentImage(self):
        """
        Returns current (full, preview or hald) image,
        according to the values of the flag useThumb and useHald.
        The thumbnail and hald are computed if they are not initialized.
        Otherwise, they are not updated, unless self.thumb is None
        or purgeThumb is True.

        :return: image
        :rtype: QImage
        """
        if self.useHald:
            return self.getHald()
        if self.useThumb:
            return self.getThumb()
        else:
            return self

    def full2CurrentXY(self, x, y):
        """
        Maps x,y coordinates of pixel in the full size image to
        coordinates in current image.

        :param x:
        :type x: int or float
        :param y:
        :type y: int or float
        :return:
        :rtype: 2uple of int
        """
        if self.useThumb:
            currentImg = self.getThumb()
            x = (x * currentImg.width()) / self.width()
            y = (y * currentImg.height()) / self.height()
        return int(x), int(y)

    def current2FullXY(self, x, y):
        """
        Maps x,y coordinates of pixel in the current image to
        coordinates in full size image.

        :param x:
        :type x: int or float
        :param y:
        :type y: int or float
        :return:
        :rtype: 2uple of int
        """
        if self.useThumb:
            currentImg = self.getThumb()
            x = (x * self.width()) / currentImg.width()
            y = (y * self.height()) / currentImg.height()
        return int(x), int(y)

    def getHspbBuffer(self):
        """
        return the image buffer in color mode HSpB.
        The buffer is recalculated when needed.

        :return: HSPB buffer
        :rtype: ndarray
        """
        # inputImage = self.inputImgFull().getCurrentImage()
        if self.hspbBuffer is None or not self.cachesEnabled:
            currentImage = self.getCurrentImage()
            self.hspbBuffer = rgb2hspVec(QImageBuffer(currentImage)[:, :, :3][:, :, ::-1])
        return self.hspbBuffer

    def getLabBuffer(self):
        """
        return the image buffer in color mode Lab.
        The buffer is recalculated when needed.

        :return: Lab buffer, L range is 0..1, a, b ranges are -128..+128
        :rtype: numpy ndarray, dtype float
        """
        if self.LabBuffer is None or not self.cachesEnabled:
            currentImage = self.getCurrentImage()
            self.LabBuffer = sRGB2LabVec(QImageBuffer(currentImage)[:, :, :3][:, :, ::-1])
        return self.LabBuffer

    def getHSVBuffer(self):
        """
        return the image buffer in color mode HSV.
        The buffer is calculated if needed and cached.
        H,S,V ranges are 0..255 (opencv convention for 8 bits images).

        :return: HSV buffer
        :rtype: numpy ndarray, dtype float
        """
        if self.HSVBuffer is None or not self.cachesEnabled:
            currentImage = self.getCurrentImage()
            self.HSVBuffer = cv2.cvtColor(QImageBuffer(currentImage)[:, :, :3], cv2.COLOR_BGR2HSV)
        return self.HSVBuffer

    def setModified(self, b):
        """
        Sets the flag.

        :param b: flag
        :type b: boolean
        """
        self.isModified = b

    def updatePixmap(self, maskOnly=False):
        """
        For the sake of performance and memory usage,
        rPixmap is instantiated only in the subclass QLayer.

        :param maskOnly: not used
        :type maskOnly: boolean
        """
        pass

    def resetMask(self, maskAll=False, alpha=255):
        """
        Reinit the mask : all pixels are masked or all
        pixels are unmasked (default). The alpha channel of the
        mask is set to alpha (default 255). Mask alpha has no effect
        on layer masking; it is used only to display semi transparent color
        masks (e.g. for grabcut or cloning)

        :param maskAll:
        :type maskAll: boolean
        :param alpha
        :type alpha: int in range 0..255
        """
        color = vImage.defaultColor_Masked if maskAll else vImage.defaultColor_UnMasked
        color.setAlpha(alpha)
        self.mask.fill(color)
        self.updatePixmap(maskOnly=True)

    def invertMask(self):
        """
        Invert image mask
        """
        buf = QImageBuffer(self.mask)
        buf[:, :, 2] = 255 - buf[:, :, 2]

    def setMaskLuminosity(self, min=0, max=255):
        """
        luminosity mask. Masks pixels whose luminosity is between min and max.
        """
        buf = self.getHSVBuffer()
        buf = cv2.resize(buf, (self.mask.width(), self.mask.height()))
        LUT = np.arange(256)
        LUT[:min] = 0
        LUT[max + 1:] = 0
        LUT[min:max] = 255
        mbuf = QImageBuffer(self.mask)
        mbuf[..., 2] = LUT[buf[:, :, 2]]

    def resized(self, w, h, keepAspectRatio=True, interpolation=cv2.INTER_CUBIC):
        """
        Returns a resized vImage and the corresponding buffer.
         We use
        the opencv function cv2.resize() to perform the resizing operation, so we
        can choose among several interpolation methods (default cv2.INTER_CUBIC).
        The original image is not modified. A link to the ndarray buffer must be kept along
        with the resized image.

        :param w: new width
        :type w: int
        :param h: new height
        :type h: int
        :param keepAspectRatio:
        :type keepAspectRatio: boolean
        :param interpolation: interpolation method (default cv2.INTER_CUBIC)
        :type interpolation:
        :return: the resized vImage and the corresponding buffer
        :rtype: vImage, ndArray
        """
        if keepAspectRatio:
            pixels = w * h
            ratio = self.width() / float(self.height())
            w, h = int(np.sqrt(pixels * ratio)), int(np.sqrt(pixels / ratio))
        Buf = QImageBuffer(self)
        cv2Img = cv2.resize(Buf, (w, h), interpolation=interpolation)
        rszd = vImage(cv2Img=cv2Img, meta=copy(self.meta), format=self.format())
        # resize rect and mask
        if self.rect is not None:
            homX, homY = w / self.width(), h / self.height()
            rszd.rect = QRect(self.rect.left() * homX, self.rect.top() * homY, self.rect.width() * homX,
                              self.rect.height() * homY)
        if self.mask is not None:
            rszd.mask = self.mask.scaled(w, h)
        rszd.setModified(True)
        return rszd, cv2Img

    def bTransformed(self, transformation):
        """
        Applies transformation and returns a copy
        of the transformed image.

        :param transformation:
        :type transformation: QTransform
        :return:
        :rtype: vImage
        """
        img = vImage(QImg=self.transformed(transformation))
        img.meta = self.meta
        img.onImageChanged = self.onImageChanged
        img.useThumb = self.useThumb
        img.useHald = self.useHald
        return img


    def noisy(self, image):
        """
        Adds gaussian noise to image.

        :param image:
        :type image: ndArray
        :return:
        :rtype:
        """

        row, col, ch = image.shape
        mean = 0
        var = 0.5
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        np.clip(noisy, 0, 255, out=noisy)
        return noisy

