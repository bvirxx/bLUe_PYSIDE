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
from time import time
import numpy as np

from PySide2.QtCore import Qt, QRectF, QSize

import cv2
from copy import copy

from PySide2.QtGui import QImageReader, QTransform, QBitmap
from PySide2.QtWidgets import QApplication, QSplitter
from PySide2.QtGui import QImage, QColor, QPainter
from PySide2.QtCore import QRect

from bLUeGui.bLUeImage import bImage, ndarrayToQImage
from bLUeCore.multi import chosenInterp
from bLUeTop.QtGui1 import app
from bLUeTop.align import alignImages
from bLUeTop.cloning import alphaBlend
from bLUeTop.colorManagement import icc

from bLUeTop.graphicsBlendFilter import blendFilterIndex

from bLUeTop.graphicsFilter import filterIndex
from bLUeGui.histogramWarping import warpHistogram
from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.colorCube import rgb2hspVec, hsp2rgbVec, hsv2rgbVec
from bLUeGui.blend import blendLuminosity, blendLuminosityBuf
from bLUeGui.colorCIE import sRGB2LabVec, Lab2sRGBVec, rgb2rgbLinear, \
    rgbLinear2rgb, RGB2XYZ, sRGB_lin2XYZInverse, bbTemperature2RGB, sRGB_lin2XYZ
from bLUeGui.multiplier import temperatureAndTint2Multipliers
from bLUeGui.dialog import dlgWarn
from bLUeCore.kernel import getKernel
from bLUeTop.lutUtils import LUT3DIdentity
from bLUeTop.rawProcessing import rawPostProcess
from bLUeTop.utils import UDict
from bLUeCore.dwtDenoising import dwtDenoiseChan
from bLUeTop.mergeImages import expFusion


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
        @param mask: mask
        @type mask: QImage
        @return: opacity mask
        @rtype: QImage
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
        @param mask: mask
        @type mask: QImage
        @return: opacity mask
        @rtype: QImage
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
        @param mask:
        @type mask: QImage
        @param invert:
        @type invert:
        @return:
        @rtype: ndarray dtype= uint8, shape (h, w)
        """
        buf = QImageBuffer(mask)  # TODO 17/12/19 useless copy removed, validate
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
        @param mask: color mask
        @type mask: QImage
        @param invert:
        @type invert: boolean
        @return:
        @rtype: QBitmap
        """
        a = vImage.colorMask2BinaryArray(mask, invert=invert)
        return QBitmap.fromData(QSize(mask.size()), np.packbits(a))

    @ classmethod
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
        @param img:
        @type img: QImage
        @param mask:
        @type mask: QImage
        @param color:
        @type color: bool
        @param inplace:
        @type inplace: boolean
        @return:
        @rtype: QImage
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
        @param mask:
        @type mask: image ndarray
        @param ks: kernel size, should be odd
        @type ks: int
        @param iterations: filter iteration count
        @type iterations: int
        @return: the dilated mask
        @rtype: ndarray
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
        @param mask:
        @type mask: image ndarray
        @param ks: kernel size, should be odd
        @type ks: int
        @param iterations: filter iteration count
        @type iterations: int
        @return: the eroded mask
        @rtype: ndarray
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
        @type mask: image ndarray
        @param ks: kernel size, should be odd
        @type ks: int
        @return: the smoothed mask
        @rtype: ndarray
        """
        kernelMean = np.ones((ks, ks), np.float) / (ks * ks)
        return cv2.filter2D(mask, -1, kernelMean)  # -1 : keep depth unchanged

    def __init__(self, filename=None, cv2Img=None, QImg=None, format=QImage.Format_ARGB32,
                 name='', colorSpace=-1, orientation=None, rating=5, meta=None, rawMetadata=None, profile=''):
        """
        With no parameter, builds a null image.
        image is assumed to be in the color space sRGB : colorSpace value is used only as meta data.
        @param filename: path to file
        @type filename: str
        @param cv2Img: data buffer
        @type cv2Img: ndarray
        @param QImg: image
        @type QImg: QImage
        @param format: QImage format (default QImage.Format_ARGB32)
        @type format: QImage.Format
        @param name: image name
        @type name: str
        @param colorSpace: color space (default : not specified)
        @type colorSpace: MarkedImg.colorSpace
        @param orientation: Qtransform object (default None)
        @type orientation: Qtransform
        @param meta: metadata instance (default None)
        @type meta: MarkedImg.metadataBag
        @param rawMetadata: dictionary
        @type rawMetadata: dictionary
        @param profile: embedded profile (default '')
        @type profile: str
        """
        # formatted EXIF data (str)
        self.imageInfo = 'no EXIF data'  # default

        if rawMetadata is None:
            rawMetadata = {}
        self.isModified = False
        self.rect, self.marker = None, None  # selection rectangle, marker
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
        self.colorSpace = self.meta.colorSpace
        self.cmsProfile = icc.defaultWorkingProfile  # possibly does not match colorSpace : call setProfile()
        self.RGB_lin2XYZ = sRGB_lin2XYZ
        self.RGB_lin2XYZInverse = sRGB_lin2XYZInverse
        if filename is None and cv2Img is None and QImg is None:
            # create a null image
            super().__init__()
        if filename is not None:
            if not isfile(filename):
                raise ValueError('Cannot find file %s' % filename)
            # load image from file (should be a 8 bits/channel color image)
            if self.meta.orientation is not None:
                tmp = QImage(filename, format=format).transformed(self.meta.orientation)
            else:
                tmp = QImage(filename, format=format)
            # ensure format is format: JPEG are loaded with format RGB32 !!
            tmp = tmp.convertToFormat(format)
            if tmp.isNull():
                raise ValueError('Cannot load %s\nSupported image formats\n%s' % (filename, QImageReader.supportedImageFormats()))
            # call to super is mandatory. Shallow copy : no harm !
            super().__init__(tmp)
        elif QImg is not None:
            # build image from QImage, shallow copy
            super().__init__(QImg)
            # if hasattr(QImg, "meta"):
                # self.meta = copy(QImg.meta)  # TODO removed 30/01/20 validate
        elif cv2Img is not None:
            # build image from buffer
            super().__init__(ndarrayToQImage(cv2Img, format=format))
        # check format
        if self.depth() != 32:
            raise ValueError('vImage : should be a 8 bits/channel color image')
        self.filename = filename if filename is not None else ''

    def setProfile(self, profile):
        """
        Sets profile related attributes
        @param cmsProfile:
        @type cmsProfile: CmsProfile instance
        """
        self.cmsProfile = profile
        if 'srgb' in profile.profile.profile_description.lower():
            self.colorSpace = 1
        else:
            self.colorSpace = 65535
        self.RGB_lin2XYZ = np.column_stack((profile.profile.red_colorant[0], profile.profile.green_colorant[0], profile.profile.blue_colorant[0]))
        self.RGB_lin2XYZInverse = np.linalg.inv(self.RGB_lin2XYZ)

    def setImage(self, qimg):
        """
        copies qimg to image. Does not update metadata.
        image and qimg must have identical dimensions and type.
        @param qimg: image
        @type qimg: QImage
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
        @return: thumbnail
        @rtype: QImage
        """
        if self.thumb is None:
            self.initThumb()
        return self.thumb

    def initHald(self):
        """
        Builds a hald image (as a QImage) from identity 3D LUT.
        A hald can be viewed as a 3D LUT flattened and reshaped as a 2D array.
        """
        if not self.cachesEnabled:
            return
        s = int(LUT3DIdentity.size**(3.0/2.0)) + 1
        buf0 = LUT3DIdentity.toHaldArray(s, s).haldBuffer
        buf1 = QImageBuffer(self.Hald)
        buf1[:, :, :] = buf0
        buf1[:, :, 3] = 255  # added for coherence with the overriding function QLayer.initHald()

    def resize_coeff(self, widget):
        """
        Normalization of self.Zoom_coeff.
        Return the current resizing coefficient used by
        the widget paint event handler to display the image.
        The coefficient is chosen to initially (i.e. when self.Zoom_coeff = 1)
        fill the widget without cropping.
        For split views we use the size of the QSplitter parent
        container instead of the size of the widget.
        @param widget:
        @tyep widget: Qwidget
        @return: the (multiplicative) resizing coefficient
        @rtype: float
        """
        wp = widget.parent()
        if type(wp) == QSplitter:
            widget = wp
        w, h = self.width(), self.height()
        r_w, r_h = float(widget.width()) / w, float(widget.height()) / h
        if h > w:
            r = min(r_w, r_h)  # prevent cropping in diaporama for portrait mode
        else:
            r = max(r_w, r_h)
        return r * self.Zoom_coeff

    def getCurrentImage(self):
        """
        Returns current (full, preview or hald) image,
        according to the values of the flag useThumb and useHald.
        The thumbnail and hald are computed if they are not initialized.
        Otherwise, they are not updated, unless self.thumb is None
        or purgeThumb is True.
        @return: image
        @rtype: QImage
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
        @param x:
        @type x: int or float
        @param y:
        @type y: int or float
        @return:
        @rtype: 2uple of int
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
        @param x:
        @type x: int or float
        @param y:
        @type y: int or float
        @return:
        @rtype: 2uple of int
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
        @return: HSPB buffer
        @rtype: ndarray
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
        @return: Lab buffer, L range is 0..1, a, b ranges are -128..+128
        @rtype: numpy ndarray, dtype np.float
        """
        if self.LabBuffer is None or not self.cachesEnabled:
            currentImage = self.getCurrentImage()
            self.LabBuffer = sRGB2LabVec(QImageBuffer(currentImage)[:, :, :3][:, :, ::-1])
        return self.LabBuffer

    def getHSVBuffer(self):
        """
        return the image buffer in color mode HSV.
        The buffer is calculated if needed and cached.
        H,S,V ranges are 0..255 (opencv convention for 8 bits images)
        @return: HSV buffer
        @rtype: numpy ndarray, dtype np.float
        """
        if self.HSVBuffer is None or not self.cachesEnabled:
            currentImage = self.getCurrentImage()
            self.HSVBuffer = cv2.cvtColor(QImageBuffer(currentImage)[:, :, :3], cv2.COLOR_BGR2HSV)
        return self.HSVBuffer

    def setModified(self, b):
        """
        Sets the flag
        @param b: flag
        @type b: boolean
        """
        self.isModified = b

    def updatePixmap(self, maskOnly=False):
        """
        For the sake of performance and memory usage,
        rPixmap is instantiated only in the subclass QLayer.
        @param maskOnly: not used
        @type maskOnly: boolean
        """
        pass

    def resetMask(self, maskAll=False, alpha=255):
        """
        Reinit the mask : all pixels are masked or all
        pixels are unmasked (default). The alpha channel of the
        mask is set to alpha (default 255). Mask alpha has no effect
        on layer masking; it is used only to display semi transparent color
        masks (e.g. for grabcut or cloning)

        @param maskAll:
        @type maskAll: boolean
        @param alpha
        @type alpha: int in range 0..255
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
        Build the a luminosity mask
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
        @param w: new width
        @type w: int
        @param h: new height
        @type h: int
        @param keepAspectRatio:
        @type keepAspectRatio: boolean
        @param interpolation: interpolation method (default cv2.INTER_CUBIC)
        @type interpolation:
        @return: the resized vImage and the corresponding buffer
        @rtype: vImage, ndArray
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
            homX, homY = w / self.width() , h / self.height()
            rszd.rect = QRect(self.rect.left() * homX, self.rect.top() * homY, self.rect.width() * homX,
                              self.rect.height() * homY)
        if self.mask is not None:  # TODO for QLayer and subclasses this initializes the mask
            rszd.mask = self.mask.scaled(w, h)
        rszd.setModified(True)
        return rszd, cv2Img

    def bTransformed(self, transformation):
        """
        Applies transformation and returns a copy
        of the transformed image.
        @param transformation:
        @type transformation: QTransform
        @return:
        @rtype: vImage
        """
        img = vImage(QImg=self.transformed(transformation))
        img.meta = self.meta
        img.onImageChanged = self.onImageChanged
        img.useThumb = self.useThumb
        img.useHald = self.useHald
        return img

    def applyNone(self):
        """
        Pass through
        """
        imgIn = self.inputImg()
        bufIn = QImageBuffer(imgIn)
        bufOut = QImageBuffer(self.getCurrentImage())
        bufOut[:, :, :] = bufIn
        self.updatePixmap()

    def applyCloning(self, seamless=True, showTranslated=False, moving=False):  # TODO remove parameter showTranslated
        """
        Seamless cloning. In addition to the layer input image, (output) image
        and mask, the method uses a source pixmap. The pixmap can
        be interactively translated and zoomed and (next) cloned into the layer
        input image to produce the layer (output) image. The cloning
        source and output regions correspond to the unmasked areas.
        If seamless is True (default) actual cloning is done, otherwise
        the source is simply copied into the layer.
        If moving is True (default False) the input image is not updated.
        @param seamless:
        @type seamless: boolean
        @param showTranslated:  unused, set to True in all calls
        @type showTranslated:
        @param moving: flag indicating if the method is triggered by a mouse event
        @type moving: boolean
        """
        adjustForm = self.getGraphicsForm()
        options = adjustForm.options
        # No change is made to lower layers
        # while moving the virtual layer: then we set redo to False
        imgIn = self.inputImg(redo=not moving, drawTranslated=False)
        if not moving:
            self.updateSourcePixmap()
        self.updateCloningMask()
        sourcePixmap = adjustForm.sourcePixmap
        # sourcePixmapThumb = adjustForm.sourcePixmapThumb
        imgOut = self.getCurrentImage()
        ########################
        # hald pass through
        if self.parentImage.isHald:
            buf0 = QImageBuffer(imgIn)
            buf1 = QImageBuffer(imgOut)
            buf1[...] = buf0
            return
        ########################

        ##############################################
        # update the marker in the positioning window
        ##############################################
        # mask center coordinates relative to full size image
        r = self.monts['m00']
        if (not self.conts) or r == 0:
            # no mask found : reset
            # if moving:  # TODO 17/12/19 removed for testing
                #dlgWarn('No cloning destination found.', info='Use the Unmask/FG brush to select a cloning region')
            self.xAltOffset, self.yAltOffset = 0.0, 0.0
            self.AltZoom_coeff = 1.0
            seamless = False
        # coordinates of the center of cloning_mask (relative to full size image)
        if r > 0:
            xC, yC = self.monts['m10'] / r, self.monts['m01'] / r
        else:
            xC, yC = 0.0, 0.0
        """
        ratioX, ratioY = sourcePixmapThumb.width() / self.width(), sourcePixmapThumb.height() / self.height()
        ##############################
        # update source image pointer
        ##############################
        pxInScaled_Copy = sourcePixmapThumb.copy()
        qptemp = QPainter(pxInScaled_Copy)
        qptemp.setPen(Qt.white)
        qptemp.drawEllipse(QPoint((xC - self.xAltOffset) * ratioX, (yC- self.yAltOffset) * ratioY), 5, 5)
        qptemp.end()
        adjustForm.widgetImg.setPixmap(pxInScaled_Copy)
        """
        ##############################################################
        # erase previous transformed image : reset imgOut to ImgIn
        ##############################################################
        qp = QPainter(imgOut)
        qp.setCompositionMode(QPainter.CompositionMode_Source)
        qp.drawPixmap(QRect(0, 0, imgOut.width(), imgOut.height()), sourcePixmap, sourcePixmap.rect())
        # get translation relative to current Image
        currentAltX, currentAltY = self.full2CurrentXY(self.xAltOffset, self.yAltOffset)
        # get mask center coordinates relative to the translated current image
        xC_current, yC_current = self.full2CurrentXY(xC, yC)
        xC_current, yC_current = xC_current - currentAltX, yC_current - currentAltY
        ###################################################################################################
        # Draw the translated and zoomed source pixmap into imgOut (nothing is drawn outside of dest image).
        # The translation is adjusted to keep the point (xC_current, yC_current) invariant while zooming.
        ###################################################################################################
        qp.setCompositionMode(QPainter.CompositionMode_SourceOver)
        bRect = QRectF(currentAltX + (1 - self.AltZoom_coeff) * xC_current, currentAltY + (1 - self.AltZoom_coeff) * yC_current,
                       imgOut.width() * self.AltZoom_coeff, imgOut.height() * self.AltZoom_coeff)
        qp.drawPixmap(bRect, sourcePixmap, sourcePixmap.rect())
        qp.end()
        #####################
        # do seamless cloning
        #####################
        if seamless:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                app.processEvents()
                # temporary dest image
                imgInc = QImage(imgIn)
                ###########################
                # clone imgOut into imgInc
                ###########################
                self.seamlessMerge(imgInc, imgOut, self.mask, self.cloningMethod,
                                     version="blue" if options['blue'] else 'opencv', w=16)
                #########################################
                # copy imgInc into imgOut.
                # To ensure interactive mask
                # adjustments (painting brush effect)
                # we copy the cloned region only.
                # In this way, when adjusting the mask,
                # the unmasked region of imgOut stays always
                # a copy of the corresponding region of source.
                #########################################
                bufOut = QImageBuffer(imgOut)
                if self.parentImage.useThumb:
                    mask = self.mask.scaled(imgOut.size())
                else:
                    mask = self.mask
                bufOut[...] = alphaBlend(QImageBuffer(imgInc), bufOut, vImage.colorMask2BinaryArray(mask))
            finally:
                self.parentImage.setModified(True)
                QApplication.restoreOverrideCursor()
                QApplication.processEvents()
        # should we forward the alpha channel ?
        self.updatePixmap()
        # the presentation layer must be updated here because
        # applyCloning is called directly (mouse and Clone button events).
        self.parentImage.prLayer.update()  # = applyNone()
        self.parentImage.onImageChanged(hist=False)

    def applyGrabcut(self, nbIter=2, mode=cv2.GC_INIT_WITH_MASK):
        """
        Segmentation.
        The segmentation mask is built from the selection rectangle, if any, and from
        the user selection.
        @param nbIter:
        @type nbIter: int
        @param mode:
        @type mode:
        """
        invalid = vImage.defaultColor_Invalid.green()
        form = self.getGraphicsForm()
        # formOptions = form.listWidget1
        inputImg = self.inputImg()
        ##################################################################
        # pass through
        # grabcut is done only when clicking the Apply button of segmentForm.
        # No grabcut for hald image
        if self.noSegment or self.parentImage.isHald:
            inBuf = QImageBuffer(inputImg)
            outputImg = self.getCurrentImage()
            outBuf = QImageBuffer(outputImg)
            outBuf[:, :, :] = inBuf
            self.updatePixmap()
            return
        ##################################################################
        # selection rectangle
        rect = self.rect
        # resizing coeff fitting selection rectangle to the current image
        r = inputImg.width() / self.width()
        ############################
        # build the segmentation mask
        ############################
        if rect is not None:
            # inside rect: PR_FGD, outside BGD
            segMask = np.zeros((inputImg.height(), inputImg.width()), dtype=np.uint8) + cv2.GC_BGD
            segMask[int(rect.top() * r):int(rect.bottom() * r), int(rect.left() * r):int(rect.right() * r)] = cv2.GC_PR_FGD
        else:
            # everywhere : PR_BGD
            segMask = np.zeros((inputImg.height(), inputImg.width()), dtype=np.uint8) + cv2.GC_PR_BGD

        # add info from current self.mask
        # initially (i.e. before any painting with BG/FG tools and before first call to applygrabcut)
        # all mask pixels are marked as invalid. Painting a pixel marks it as valid, Ctrl+paint
        # switches it back to invalid.
        # Only valid pixels are added to segMask, fixing them as FG or BG
        if inputImg.size() != self.size():
            scaledMask = self.mask.scaled(inputImg.width(), inputImg.height())
        else:
            scaledMask = self.mask
        scaledMaskBuf = QImageBuffer(scaledMask)

        # copy valid pixels from scaledMaskBuf to the segmentation mask
        m = (scaledMaskBuf[:, :, 2] > 100) * (scaledMaskBuf[:, :, 1] != invalid)  # R>100 is unmasked, R=0 is masked
        segMask[m] = cv2.GC_FGD
        m = (scaledMaskBuf[:, :, 2] == 0) * (scaledMaskBuf[:, :, 1] != invalid)
        segMask[m] = cv2.GC_BGD
        # sanity check : at least one (FGD or PR_FGD)  pixel and one (BGD or PR_BGD) pixel
        if not ((np.any(segMask == cv2.GC_FGD) or np.any(segMask == cv2.GC_PR_FGD))
                and
                (np.any(segMask == cv2.GC_BGD) or np.any(segMask == cv2.GC_PR_BGD))):
            dlgWarn('You must select some background or foreground pixels', info='Use selection rectangle or Mask/Unmask tools')
            return
        #############
        # do segmentation
        #############
        bgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the background GMM model
        fgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the foreground GMM model
        t0 = time()
        inputBuf = QImageBuffer(inputImg)
        # get the fastest available grabcut function
        if getattr(cv2, 'grabCut_mt', None) is None:
            bGrabcut = cv2.grabCut
        else:
            bGrabcut = cv2.grabCut_mt
        bGrabcut(inputBuf[:, :, :3], segMask, None, bgdmodel, fgdmodel, nbIter, mode)
        print('%s : %.2f' % (bGrabcut.__name__, (time()-t0)))

        # back to mask
        unmasked = vImage.defaultColor_UnMasked.red()
        masked = vImage.defaultColor_Masked.red()
        buf = QImageBuffer(scaledMask)
        buf[:, :, 2] = np.where((segMask == cv2.GC_FGD) + (segMask == cv2.GC_PR_FGD), unmasked, masked)
        buf[:, :, 3] = 128  # 50% opacity

        # mark all mask pixels as valid, thus
        # further calls to applyGrabcut will not be
        # able to modify them. To enable further modifications
        # paint mask pixels white holding the Ctrl key.
        buf[:, :, 1] = 0

        invalidate_contour = form.contourMargin > 0
        # invalidate a stripe around the foreground/background boundary.
        # its width is 5 * form.contourMargin
        if invalidate_contour:
            # build the contour as a boolean mask
            maxIterations = form.contourMargin
            ebuf = vImage.maskErode(buf.copy(), iterations=maxIterations)
            dbuf = vImage.maskDilate(buf.copy(), iterations=maxIterations)
            m = ((buf[:, :, 2] == 0) & (ebuf[:, :, 2] == unmasked)) | ((buf[:, :, 2] == unmasked) & (dbuf[:, :, 2] == 0))
            # mark contour pixels as invalid and others as valid : the contour only can be modified
            buf[:, :, 1] = np.where(m, invalid, 0)

        """
        # dilate the mask to remove background dithering
        # iteration count should be <= maxIterations
        dbuf = vImage.maskDilate(buf.copy(), iterations=min(form.contourMargin, maxIterations))
        innerCbMask = ((buf[:, :, 2] == unmasked) & (dbuf[:, :, 2] == 0))
        buf[:, :, 2] = np.where(innerCbMask, 0, buf[:,:,2]) # [innerCbMask] = 0
        """

        if self.size() != scaledMask.size():
            self.mask = scaledMask.scaled(self.size())
        else:
            self.mask = scaledMask

        # forward the alpha channel
        # should we forward it?
        self.updatePixmap()

    def applyInvert(self):
        """
        Invert an  image. Depending of the graphics form options,
        the orange mask is estimated automaically or set from
        the graphic form parameters.
        """
        adjustForm = self.getGraphicsForm()
        options = adjustForm.options if adjustForm is not None else {'Auto': True}
        bufIn = QImageBuffer(self.inputImg())[:, :, :3]
        if options['Auto']:
            # get orange mask from negative brightest (unexposed) pixels
            temp = np.sum(bufIn, axis=2)
            ind = np.argmax(temp)
            ind = np.unravel_index(ind, (bufIn.shape[0], bufIn.shape[1]))
            Mask0, Mask1, Mask2 = bufIn[ind]
        else:
            Mask0, Mask1, Mask2 = adjustForm.Bmask, adjustForm.Gmask, adjustForm.Rmask
        currentImage = self.getCurrentImage()
        bufOut = QImageBuffer(currentImage)
        # eliminate mask
        tmp = bufIn[:, :, :3] / [Mask0, Mask1, Mask2]
        tmp *= 255  # TODO modified 23/01/20 validate
        np.clip(tmp, 0, 255, out=tmp)
        # invert
        bufOut[:, :, :3] = 255.0 - tmp
        self.updatePixmap()

    def applyHDRMerge(self, options):
        # form = self.getGraphicsForm()
        # search for layers to merge, below the merging layer
        stack = self.parentImage.layersStack
        i = self.getStackIndex()
        mergingLayers = []
        for layer in stack[:i]:
            if layer.visible and layer.mergingFlag:
                mergingLayers.append(layer)
        # pass through
        if not mergingLayers:
            inputImg = self.inputImg()
            inBuf = QImageBuffer(inputImg)
            outputImg = self.getCurrentImage()
            outBuf = QImageBuffer(outputImg)
            outBuf[:, :, :] = inBuf
            self.updatePixmap()
            return
        bufList = []
        pred = None
        for layer in mergingLayers:
            img = layer.getCurrentImage()
            buf = QImageBuffer(img)
            if pred is not None:
                buf[...] = alignImages(buf, pred)
            bufList.append(buf[:, :, :3])

        # buf = np.stack(bufList, axis=-1)
        # buf = np.median(buf, axis=-1)
        buf = expFusion(bufList)
        imgOut = self.getCurrentImage()
        bufOut = QImageBuffer(imgOut)
        bufOut[..., :3] = buf
        self.updatePixmap()

    def applyExposure(self, options):
        """
        Multiply the linearized RGB channels by
        c = 2**exposureCorrection.

        @param options:
        @type options:
        """
        form = self.getGraphicsForm()
        exposureCorrection = form.expCorrection
        # neutral point
        if abs(exposureCorrection) < 0.05:
            buf0 = QImageBuffer(self.getCurrentImage())
            buf1 = QImageBuffer(self.inputImg())
            buf0[:, :, :] = buf1
            self.updatePixmap()
            return
        bufIn = QImageBuffer(self.inputImg())
        buf = bufIn[:, :, :3][:, :, ::-1]
        # convert to linear
        buf = rgb2rgbLinear(buf)
        # apply correction
        buf[:, :, :] = buf * (2 ** exposureCorrection)
        np.clip(buf, 0.0, 1.0, out=buf)
        # convert back to RGB
        buf = rgbLinear2rgb(buf)
        np.clip(buf, 0.0, 255.0, out=buf)
        currentImage = self.getCurrentImage()
        ndImg1a = QImageBuffer(currentImage)
        ndImg1a[:, :, :3][:, :, ::-1] = buf
        # forward the alpha channel
        ndImg1a[:, :, 3] = bufIn[:, :, 3]
        self.updatePixmap()

    def applyMixer(self, options):
        form = self.getGraphicsForm()
        bufIn = QImageBuffer(self.inputImg())
        buf = bufIn[:, :, :3][:, :, ::-1]
        # convert to linear
        buf = rgb2rgbLinear(buf)
        # mix channels
        currentImage = self.getCurrentImage()
        bufOut = QImageBuffer(currentImage)
        buf = np.tensordot(buf, form.mixerMatrix, axes=(-1, -1))
        np.clip(buf, 0, 1.0, out=buf)
        # convert back to RGB
        buf = rgbLinear2rgb(buf)
        if form.options['Luminosity']:
            bufOut[:, :, :3][:, :, ::-1] = blendLuminosityBuf(bufIn[:, :, :3][:, :, ::-1], buf)
        else:
            bufOut[:, :, :3][:, :, ::-1] = buf
        # forward the alpha channel
        bufOut[:, :, 3] = bufIn[:, :, 3]
        self.updatePixmap()

    def applyTransForm(self, options):
        """
        Apply the geometric transformation defined by source and target quads
        @param options:
        @type options:
        """
        # if self.maskIsEnabled:
            # dlgWarn("A masked layer cannot be transformed.\nDisable the mask")
            # self.updatePixmap()
            # return
        inImg = self.inputImg()
        outImg = self.getCurrentImage()
        buf0 = QImageBuffer(outImg)
        w, h = inImg.width(), inImg.height()
        s = w / self.width()
        D = QTransform().scale(s, s)
        DInv = QTransform().scale(1/s, 1/s)
        q1Full, q2Full = self.tool.getSourceQuad(), self.tool.getTargetQuad()
        # map Quads to the current image coordinate system
        q1, q2 = D.map(q1Full), D.map(q2Full)
        # build transformation
        T = QTransform()
        res = QTransform.quadToQuad(q1, q2, T)
        if not res:
            print('applyTransform : no possible transformation')
            self.tool.restore()
            return
        # neutral point
        if T.isIdentity():
            buf1 = QImageBuffer(inImg)
            buf0[:, :, :] = buf1
            self.updatePixmap()
            return
        # get the bounding rect of the transformed image (in the full size image coordinate system)
        # (Avoid the conversion of QTransforms to QMatrix4x4 and matrix product)
        rectTrans = DInv.map(T.map(D.map(QImage.rect(self)))).boundingRect()
        # apply the transformation and re-translate the transformed image
        # so that the resulting transformation is T and NOT that given by QImage.trueMatrix()
        img = (inImg.transformed(T)).copy(QRect(-rectTrans.x()*s, -rectTrans.y()*s, w, h))
        # copy sets pixels beyond image to 0. To show these pixels
        # as black we set their alpha value to 255:
        if options['Transparent']:
            buf = QImageBuffer(img)
            buf[:, :, 3] = np.where(buf[:, :, 3] == 0, 255, buf[:, :, 3])
        if img.isNull():
            print('applyTransform : transformation fails')
            self.tool.restore()
            return
        buf0[:, :, :] = QImageBuffer(img)
        self.updatePixmap()

    def applyImage(self, options):
        self.applyTransForm(options)
        if options['Align']:
            inImg = self.inputImg()
            buf0 = QImageBuffer(inImg)
            outImg = self.getCurrentImage()
            buf1 = QImageBuffer(outImg)
            imalgn, h = alignImages(buf1, buf0)
            buf1[...] = imalgn
            self.updatePixmap()

    def applyNoiseReduction(self):
        """
        Wavelets, bilateral filtering, NLMeans
        """
        adjustForm = self.getGraphicsForm()
        noisecorr = adjustForm.noiseCorrection
        currentImage = self.getCurrentImage()
        inputImage = self.inputImg()
        buf0 = QImageBuffer(inputImage)
        buf1 = QImageBuffer(currentImage)
        ########################
        # hald pass through and neutral point
        if self.parentImage.isHald or noisecorr == 0:
            buf1[...] = buf0
            self.updatePixmap()
            return
        ########################
        w, h = self.width(), self.height()
        r = inputImage.width() / w
        if self.rect is not None:
            # slicing
            rect = self.rect
            imgRect = QRect(0, 0, w, h)
            rect = rect.intersected(imgRect)
            if rect.width() < 10 or rect.height() < 10:
                dlgWarn('Selection is too narrow')
                return
            slices = np.s_[int(rect.top() * r): int(rect.bottom() * r), int(rect.left() * r): int(rect.right() * r), :3]
            ROI0 = buf0[slices]
            # reset output image
            buf1[:, :, :] = buf0
            ROI1 = buf1[slices]
        else:
            ROI0 = buf0[:, :, :3]
            ROI1 = buf1[:, :, :3]
        buf01 = ROI0[:, :, ::-1]
        noisecorr *= currentImage.width() / self.width()
        if adjustForm.options['Wavelets']:
            noisecorr *= 100
            bufLab = cv2.cvtColor(buf01, cv2.COLOR_RGB2Lab)
            L = dwtDenoiseChan(bufLab, chan=0, thr=noisecorr, thrmode='wiener')  # level=8 if self.parentImage.useThumb else 11)
            A = dwtDenoiseChan(bufLab, chan=1, thr=noisecorr, thrmode='wiener')  # level=8 if self.parentImage.useThumb else 11)
            B = dwtDenoiseChan(bufLab, chan=2, thr=noisecorr, thrmode='wiener')  # level=8 if self.parentImage.useThumb else 11)
            np.clip(L, 0, 255, out=L)
            np.clip(A, 0, 255, out=A)
            np.clip(B, 0, 255, out=B)
            bufLab = np.dstack((L, A, B))
            # back to RGB
            ROI1[:, :, ::-1] = cv2.cvtColor(bufLab.astype(np.uint8), cv2.COLOR_Lab2RGB)
        elif adjustForm.options['Bilateral']:
            ROI1[:, :, ::-1] = cv2.bilateralFilter(buf01,
                                         9 if self.parentImage.useThumb else 15,    # 21:5.5s, 15:3.5s, diameter of
                                                                                    # (coordinate) pixel neighborhood,
                                                                                    # 5 is the recommended value for fast processing
                                         10 * adjustForm.noiseCorrection,           # std deviation sigma
                                                                                    # in color space,  100 middle value
                                         50 if self.parentImage.useThumb else 150,  # std deviation sigma
                                                                                    # in coordinate space,  100 middle value
                                         )
        elif adjustForm.options['NLMeans']:
            ROI1[:, :, ::-1] = cv2.fastNlMeansDenoisingColored(buf01, None, 1+noisecorr, 1+noisecorr, 7, 21)  # hluminance, hcolor,  last params window sizes 7, 21 are recommended values

        # forward the alpha channel
        buf1[:, :, 3] = buf0[:, :, 3]
        self.updatePixmap()

    def applyRawPostProcessing(self, pool=None):
        """
        Develop raw image.
        """
        if self.parentImage.isHald:
            raise ValueError('Cannot build a 3D LUT from raw stack')

        rawPostProcess(self, pool=pool)

    def applyContrast(self, version='HSV'):
        """
        Apply contrast, saturation and brightness corrections.
        If version is 'HSV' (default), the
        image is first converted to HSV and next a curve f(x)=x**alpha is applied to
        the S and V channels. Otherwise, the Lab color space is used :
        a curve f(x) = x**alpha is applied to the L channel and curves f(x) = x*slope
        are applied to the a and b channels.
        @param version:
        @type version: str
        """
        adjustForm = self.getGraphicsForm()
        options = adjustForm.options
        contrastCorrection = adjustForm.contrastCorrection
        satCorrection = adjustForm.satCorrection
        brightnessCorrection = adjustForm.brightnessCorrection
        inputImage = self.inputImg()
        tmpBuf = QImageBuffer(inputImage)
        currentImage = self.getCurrentImage()
        ndImg1a = QImageBuffer(currentImage)
        # neutral point : by pass
        if contrastCorrection == 0 and satCorrection == 0 and brightnessCorrection == 0:
            ndImg1a[:, :, :] = tmpBuf
            self.updatePixmap()
            return
        ##########################
        # Lab mode (slower than HSV)
        ##########################
        if version == 'Lab':
            # get l channel (range 0..1)
            LBuf = inputImage.getLabBuffer().copy()
            if brightnessCorrection != 0:
                alpha = (-adjustForm.brightnessCorrection + 1.0)
                # tabulate x**alpha
                LUT = np.power(np.arange(256) / 255.0, alpha)
                # convert L to L**alpha
                LBuf[:, :, 0] = LUT[(LBuf[:, :, 0] * 255.0).astype(np.uint8)]
            # contrast
            if contrastCorrection > 0:
                # CLAHE
                if options['CLAHE']:
                    if self.parentImage.isHald:
                        raise ValueError('cannot build 3D LUT from CLAHE ')
                    clahe = cv2.createCLAHE(clipLimit=contrastCorrection, tileGridSize=(8, 8))
                    clahe.setClipLimit(contrastCorrection)
                    res = clahe.apply((LBuf[:, :, 0] * 255.0).astype(np.uint8)) / 255.0
                # warping
                else:
                    if self.parentImage.isHald and not options['manualCurve']:
                        raise ValueError('A contrast curve was found.\nCheck the option Show Contrast Curve in Cont/Bright/Sat layer')
                    auto = self.autoSpline and not self.parentImage.isHald
                    res, a, b, d, T = warpHistogram(LBuf[:, :, 0], warp=contrastCorrection, preserveHigh=options['High'],
                                                spline=None if auto else self.getMmcSpline())
                    # show the spline viewer
                    if self.autoSpline and options['manualCurve']:
                        self.getGraphicsForm().setContrastSpline(a, b, d, T)
                        self.autoSpline = False
                LBuf[:, :, 0] = res
            # saturation
            if satCorrection != 0:
                slope = max(0.1, adjustForm.satCorrection / 25 + 1)
                # multiply a and b channels
                LBuf[:, :, 1:3] *= slope
                LBuf[:, :, 1:3] = np.clip(LBuf[:, :, 1:3], -127, 127)
            # back to RGB
            sRGBBuf = Lab2sRGBVec(LBuf)  # use cv2.cvtColor
        ###########
        # HSV mode
        ###########
        else:
            # get HSV buffer (H, S, V are in range 0..255)
            HSVBuf = inputImage.getHSVBuffer().copy()
            if brightnessCorrection != 0:
                alpha = 1.0 / (0.501 + adjustForm.brightnessCorrection) - 1.0  # approx. map -0.5...0.0...0.5 --> +inf...1.0...0.0
                # tabulate x**alpha
                LUT = np.power(np.arange(256) / 255, alpha)
                LUT *= 255.0  # TODO modified 23/01/20 validate
                # convert V to V**alpha
                HSVBuf[:, :, 2] = LUT[HSVBuf[:, :, 2]]  # faster than take
            if contrastCorrection > 0:
                # CLAHE
                if options['CLAHE']:
                    if self.parentImage.isHald:
                        raise ValueError('cannot build 3D LUT from CLAHE ')
                    clahe = cv2.createCLAHE(clipLimit=contrastCorrection, tileGridSize=(8, 8))
                    clahe.setClipLimit(contrastCorrection)
                    res = clahe.apply((HSVBuf[:, :, 2]))
                # warping
                else:
                    if self.parentImage.isHald and not options['manualCurve']:
                        raise ValueError('A contrast curve was found.\nCheck the option Show Contrast Curve in Cont/Bright/Sat layer')
                    buf32 = HSVBuf[:, :, 2].astype(np.float) / 255
                    auto = self.autoSpline and not self.parentImage.isHald  # flag for manual/auto spline
                    res, a, b, d, T = warpHistogram(buf32, warp=contrastCorrection, preserveHigh=options['High'],
                                                    spline=None if auto else self.getMmcSpline())
                    res = (res*255.0).astype(np.uint8)
                    # show the spline viewer
                    if self.autoSpline and options['manualCurve']:
                        self.getGraphicsForm().setContrastSpline(a, b, d, T)
                        self.autoSpline = False
                HSVBuf[:, :, 2] = res
            if satCorrection != 0:
                alpha = 1.0 / (0.501 + adjustForm.satCorrection) - 1.0  # approx. map -0.5...0.0...0.5 --> +inf...1.0...0.0
                # tabulate x**alpha
                LUT = np.power(np.arange(256) / 255, alpha)
                LUT *= 255  # TODO modified 23/01/20 validate
                # convert saturation s to s**alpha
                HSVBuf[:, :, 1] = LUT[HSVBuf[:, :, 1]]  # faster than take
            # back to RGB
            sRGBBuf = cv2.cvtColor(HSVBuf, cv2.COLOR_HSV2RGB)
        ndImg1a[:, :, :3][:, :, ::-1] = sRGBBuf
        # forward the alpha channel
        ndImg1a[:, :, 3] = tmpBuf[:, :, 3]
        self.updatePixmap()

    def apply1DLUT(self, stackedLUT):
        """
        Apply 1D LUTS to R, G, B channels (one for each channel)
        @param stackedLUT: array of color values (in range 0..255) : a row for each R, G, B channel
        @type stackedLUT : ndarray, shape=(3, 256), dtype=int
        """
        # neutral point: by pass
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast works
            buf1 = QImageBuffer(self.inputImg())
            buf2 = QImageBuffer(self.getCurrentImage())
            buf2[:, :, :] = buf1
            self.updatePixmap()
            return
        adjustForm = self.getGraphicsForm()
        options = adjustForm.graphicsScene.options
        inputImage = self.inputImg()
        currentImage = self.getCurrentImage()
        # get image buffers
        ndImg0a = QImageBuffer(inputImage)
        ndImg1a = QImageBuffer(currentImage)
        ndImg0 = ndImg0a[:, :, :3]
        ndImg1 = ndImg1a[:, :, :3]
        # apply LUTS to channels
        s = ndImg0[:, :, 0].shape
        if options['Luminosity']:
            buf = np.empty_like(ndImg1)
            for c in range(3):  # 0.36s for 15Mpx
                buf[:, :, c] = np.take(stackedLUT[2-c, :], ndImg0[:, :, c].reshape((-1,))).reshape(s)
            ndImg1[...] = blendLuminosityBuf(ndImg0, buf)
        else:
            for c in range(3):  # 0.36s for 15Mpx
                ndImg1[:, :, c] = np.take(stackedLUT[2-c, :], ndImg0[:, :, c].reshape((-1,))).reshape(s)

        # rList = np.array([2,1,0])  # B, G, R
        # ndImg1[:, :, :] = stackedLUT[rList, ndImg0]  # last dims of index arrays are equal : broadcast works. slower 0.66s for 15Mpx
        # forward alpha channel
        ndImg1a[:, :, 3] = ndImg0a[:, :, 3]
        self.updatePixmap()

    def applyLab1DLUT(self, stackedLUT, options=None):
        """
        Applies 1D LUTS (one row for each L,a,b channel)
        @param stackedLUT: array of color values (in range 0..255). Shape must be (3, 255) : a row for each channel
        @type stackedLUT: ndarray shape=(3,256) dtype=int or float
        @param options: not used yet
        """
        if options is None:
            options = UDict()
        # neutral point
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast is working
            buf1 = QImageBuffer(self.inputImg())
            buf2 = QImageBuffer(self.getCurrentImage())
            buf2[:, :, :] = buf1
            self.updatePixmap()
            return
        # convert LUT to float to speed up  buffer conversions
        stackedLUT = stackedLUT.astype(np.float)
        # get the Lab input buffer
        Img0 = self.inputImg()
        ndLabImg0 = Img0.getLabBuffer()  # copy()
        # conversion functions

        def scaleLabBuf(buf):
            buf = buf + [0.0, 128.0, 128.0]  # copy is mandatory here to avoid the corruption of the cached Lab buffer
            buf[:, :, 0] *= 255.0
            return buf

        def scaleBackLabBuf(buf):
            buf = buf - [0.0, 128.0, 128.0]  # no copy needed here, but seems faster than in place operation!
            buf[:, :, 0] /= 255.0
            return buf
        ndLImg0 = scaleLabBuf(ndLabImg0).astype(np.uint8)
        # apply LUTS to channels
        s = ndLImg0[:, :, 0].shape
        ndLabImg1 = np.zeros(ndLImg0.shape, dtype=np.uint8)
        for c in range(3):  # 0.43s for 15Mpx
            ndLabImg1[:, :, c] = np.take(stackedLUT[c, :], ndLImg0[:, :, c].reshape((-1,))).reshape(s)
        # ndLabImg1 = stackedLUT[rList, ndLImg0] # last dims are equal : broadcast works
        ndLabImg1 = scaleBackLabBuf(ndLabImg1)
        # back sRGB conversion
        ndsRGBImg1 = Lab2sRGBVec(ndLabImg1)
        # in place clipping
        np.clip(ndsRGBImg1, 0, 255, out=ndsRGBImg1)  # mandatory
        currentImage = self.getCurrentImage()
        ndImg1 = QImageBuffer(currentImage)
        ndImg1[:, :, :3][:, :, ::-1] = ndsRGBImg1
        # forward the alpha channel
        ndImg0 = QImageBuffer(Img0)
        ndImg1[:, :, 3] = ndImg0[:, :, 3]
        # update
        self.updatePixmap()

    def applyHSPB1DLUT(self, stackedLUT, options=None, pool=None):
        """
        Applies 1D LUTS to hue, sat and brightness channels.
        @param stackedLUT: array of color values (in range 0..255), a row for each channel
        @type stackedLUT : ndarray shape=(3,256) dtype=int or float
        @param options: not used yet
        @type options : dictionary
        @param pool: multiprocessing pool : unused
        @type pool: muliprocessing.Pool
        """
        if options is None:
            options = UDict()
        # neutral point
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast is working
            buf1 = QImageBuffer(self.inputImg())
            buf2 = QImageBuffer(self.getCurrentImage())
            buf2[:, :, :] = buf1
            self.updatePixmap()
            return
        Img0 = self.inputImg()
        ndHSPBImg0 = Img0.getHspbBuffer()   # time 2s with cache disabled for 15 Mpx
        # apply LUTS to normalized channels (range 0..255)
        ndLImg0 = (ndHSPBImg0 * [255.0/360.0, 255.0, 255.0]).astype(np.uint8)
        # rList = np.array([0,1,2]) # H,S,B
        ndHSBPImg1 = np.zeros(ndLImg0.shape, dtype=np.uint8)
        s = ndLImg0[:, :, 0].shape
        for c in range(3):  # 0.36s for 15Mpx
            ndHSBPImg1[:, :, c] = np.take(stackedLUT[c, :], ndLImg0[:, :, c].reshape((-1,))).reshape(s)
        # ndHSBPImg1 = stackedLUT[rList, ndLImg0] * [360.0/255.0, 1/255.0, 1/255.0]
        # back to sRGB
        ndRGBImg1 = hsp2rgbVec(ndHSBPImg1)  # time 4s for 15 Mpx
        # in place clipping
        np.clip(ndRGBImg1, 0, 255, out=ndRGBImg1)  # mandatory
        # set current image to modified image
        currentImage = self.getCurrentImage()
        ndImg1a = QImageBuffer(currentImage)
        ndImg1a[:, :, :3][:, :, ::-1] = ndRGBImg1
        # forward the alpha channel
        ndImg0 = QImageBuffer(Img0)
        ndImg1a[:, :, 3] = ndImg0[:, :, 3]
        # update
        self.updatePixmap()

    def applyHSV1DLUT(self, stackedLUT, options=None, pool=None):
        """
        Applies 1D LUTS to hue, sat and brightness channels.
        @param stackedLUT: array of color values (in range 0..255), a row for each channel
        @type stackedLUT : ndarray shape=(3,256) dtype=int or float
        @param options: not used yet
        @type options : Udict
        @param pool: multiprocessing pool : unused
        @type pool: muliprocessing.Pool
        """
        # neutral point
        if options is None:
            options = UDict()
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast is working
            buf1 = QImageBuffer(self.inputImg())
            buf2 = QImageBuffer(self.getCurrentImage())
            buf2[:, :, :] = buf1
            self.updatePixmap()
            return
        # convert LUT to float to speed up  buffer conversions
        stackedLUT = stackedLUT.astype(np.float)
        # get HSV buffer, range H: 0..180, S:0..255 V:0..255
        Img0 = self.inputImg()
        HSVImg0 = Img0.getHSVBuffer()
        HSVImg0 = HSVImg0.astype(np.uint8)
        # apply LUTS
        HSVImg1 = np.zeros(HSVImg0.shape, dtype=np.uint8)
        s = HSVImg0[:, :, 0].shape
        for c in range(3):  # 0.43s for 15Mpx
            HSVImg1[:, :, c] = np.take(stackedLUT[c, :], HSVImg0[:, :, c].reshape((-1,))).reshape(s)
        # back to sRGB
        RGBImg1 = hsv2rgbVec(HSVImg1, cvRange=True)
        # in place clipping
        np.clip(RGBImg1, 0, 255, out=RGBImg1)  # mandatory
        # set current image to modified image
        currentImage = self.getCurrentImage()
        ndImg1a = QImageBuffer(currentImage)
        ndImg1a[:, :, :3][:, :, ::-1] = RGBImg1
        # forward the alpha channel
        ndImg0 = QImageBuffer(Img0)
        ndImg1a[:, :, 3] = ndImg0[:, :, 3]
        # update
        self.updatePixmap()

    def applyHVLUT2D(self, LUT, options=None, pool=None):
        if options is None:
            options = UDict()
        # get buffers
        inputImage = self.inputImg()
        currentImage = self.getCurrentImage()
        bufOut = QImageBuffer(currentImage)
        if options is None:
            options = UDict()
        # get selection
        if self.rect is not None:
            w, wF = self.getCurrentImage().width(), self.width()
            h, hF = self.getCurrentImage().height(), self.height()
            wRatio, hRatio = float(w) / wF, float(h) / hF
            w1, w2, h1, h2 = int(self.rect.left() * wRatio), int(self.rect.right() * wRatio), int(self.rect.top() * hRatio), int(self.rect.bottom() * hRatio)
            w1, h1 = max(w1, 0), max(h1, 0)
            w2, h2 = min(w2, inputImage.width()), min(h2, inputImage.height())
            if w1 >= w2 or h1 >= h2:
                dlgWarn("Empty selection\nSelect a region with the marquee tool")
                return
            # reset layer image
            bufOut[:, :, :] = QImageBuffer(inputImage)
        else:
            w1, w2, h1, h2 = 0, self.inputImg().width(), 0, self.inputImg().height()
        # get HSV buffer, range H: 0..180, S:0..255 V:0..255  (opencv convention for 8 bits images)
        HSVImg0 = inputImage.getHSVBuffer()
        HSVImg0 = HSVImg0.astype(np.float)
        HSVImg0[:, :, 0] *= 2
        bufHSV_CV32 = HSVImg0[h1:h2 + 1, w1:w2 + 1, :]

        divs = LUT.divs
        steps = tuple([360 / divs[0], 255.0 / divs[1], 255.0 / divs[2]])
        interp = chosenInterp(pool, (w2 - w1) * (h2 - h1))
        coeffs = interp(LUT.data, steps, bufHSV_CV32, convert=False)
        bufHSV_CV32[:, :, 0] = np.mod(bufHSV_CV32[:, :, 0] + coeffs[:, :, 0], 360)
        bufHSV_CV32[:, :, 1:] = bufHSV_CV32[:, :, 1:] * coeffs[:, :, 1:]
        np.clip(bufHSV_CV32, (0, 0, 0), (360, 255, 255), out=bufHSV_CV32)
        bufHSV_CV32[:, :, 0] /= 2

        bufpostF32_1 = cv2.cvtColor(bufHSV_CV32.astype(np.uint8), cv2.COLOR_HSV2RGB)
        bufOut = QImageBuffer(currentImage)
        bufOut[h1:h2 + 1, w1:w2 + 1, :3] = bufpostF32_1[:, :, ::-1]
        self.updatePixmap()

    def apply3DLUT(self, lut3D, options=None, pool=None):
        """
        Apply a 3D LUT to the current view of the image (self or self.thumb).
        If pool is not None and the size of the current view is > 3000000,
        parallel interpolation on image slices is used.
        If options['keep alpha'] is False, alpha channel is interpolated too.
        LUT axes, LUT channels and image channels must be in BGR order.
        @param lut3D: LUT3D
        @type lut3D: LUT3D
        @param options:
        @type options: UDict
        @param pool:
        @type pool:
        """
        LUT = lut3D.LUT3DArray
        LUTSTEP = lut3D.step
        if options is None:
            options = UDict()
        # get buffers
        inputImage = self.inputImg()
        currentImage = self.getCurrentImage()
        # get selection
        w1, w2, h1, h2 = (0.0,) * 4
        useSelection = False
        if self.rect is not None:
            useSelection = self.rect.isValid()
        if useSelection:
            w, wF = self.getCurrentImage().width(), self.width()
            h, hF = self.getCurrentImage().height(), self.height()
            wRatio, hRatio = float(w) / wF, float(h) / hF
            if self.rect is not None:
                w1, w2, h1, h2 = int(self.rect.left() * wRatio), int(self.rect.right() * wRatio), int(self.rect.top() * hRatio), int(self.rect.bottom() * hRatio)
            w1, h1 = max(w1, 0), max(h1, 0)
            w2, h2 = min(w2, inputImage.width()), min(h2, inputImage.height())
            if w1 >= w2 or h1 >= h2:
                dlgWarn("Empty selection\nSelect a region with the marquee tool")
                return
        else:
            w1, w2, h1, h2 = 0, self.inputImg().width(), 0, self.inputImg().height()
        inputBuffer0 = QImageBuffer(inputImage)
        inputBuffer = inputBuffer0[h1:h2 + 1, w1:w2 + 1, :]
        imgBuffer = QImageBuffer(currentImage)[:, :, :]
        interpAlpha = not options['keep alpha']
        if interpAlpha:
            # interpolate alpha channel from LUT
            ndImg0 = inputBuffer
            ndImg1 = imgBuffer
        else:
            ndImg0 = inputBuffer[:, :, :3]
            ndImg1 = imgBuffer[:, :, :3]
            LUT = np.ascontiguousarray(LUT[..., :3])
        # choose the right interpolation method
        interp = chosenInterp(pool, (w2 - w1) * (h2 - h1))
        # apply LUT
        if useSelection:
            # need to reset the outside of the current selection
            ndImg1[:, :, :] = inputBuffer0
        ndImg1[h1:h2 + 1, w1:w2 + 1, :] = interp(LUT, LUTSTEP, ndImg0)
        if not interpAlpha:
            # forward the alpha channel
            imgBuffer[h1:h2 + 1, w1:w2 + 1, 3] = inputBuffer[:, :, 3]
        self.updatePixmap()

    def applyFilter2D(self, options=None):
        """
        Apply 2D kernel.
        """
        adjustForm = self.getGraphicsForm()
        inputImage = self.inputImg()
        currentImage = self.getCurrentImage()
        buf0 = QImageBuffer(inputImage)
        buf1 = QImageBuffer(currentImage)
        ########################
        # hald pass through
        if self.parentImage.isHald:
            buf1[...] = buf0
            return
        ########################
        w, h = self.width(), self.height()
        r = inputImage.width() / w
        if self.rect is not None:
            # slicing
            rect = self.rect
            imgRect = QRect(0, 0, w, h)
            rect = rect.intersected(imgRect)
            if rect.width() < 10 or rect.height() < 10:
                dlgWarn('Selection is too narrow')
                return
            slices = np.s_[int(rect.top() * r): int(rect.bottom() * r), int(rect.left() * r): int(rect.right() * r), :3]
            ROI0 = buf0[slices]
            # reset output image
            buf1[:, :, :] = buf0
            ROI1 = buf1[slices]
        else:
            ROI0 = buf0[:, :, :3]
            ROI1 = buf1[:, :, :3]
        # kernel based filtering
        if adjustForm.kernelCategory in [filterIndex.IDENTITY, filterIndex.UNSHARP,
                                         filterIndex.SHARPEN, filterIndex.BLUR1, filterIndex.BLUR2]:
            # correct radius for preview if needed
            radius = int(adjustForm.radius * r)
            kernel = getKernel(adjustForm.kernelCategory, radius, adjustForm.amount)
            ROI1[:, :, :] = cv2.filter2D(ROI0, -1, kernel)
        else:
            # bilateral filtering
            radius = int(adjustForm.radius * r)
            sigmaColor = 2 * adjustForm.tone
            sigmaSpace = sigmaColor
            ROI1[:, :, ::-1] = cv2.bilateralFilter(ROI0[:, :, ::-1], radius, sigmaColor, sigmaSpace)
        # forward the alpha channel
        buf1[:, :, 3] = buf0[:, :, 3]
        self.updatePixmap()

    def applyBlendFilter(self):
        """
        Apply a gradual neutral density filter
        """
        adjustForm = self.getGraphicsForm()
        inputImage = self.inputImg()
        currentImage = self.getCurrentImage()
        buf0 = QImageBuffer(inputImage)
        buf1 = QImageBuffer(currentImage)
        ########################
        # hald pass through
        if self.parentImage.isHald:
            buf1[...] = buf0
            return
        ########################
        r = inputImage.width() / self.width()
        ####################
        # We blend a neutral filter with density range 0.5*s...0.5 with the image b,
        # using blending mode overlay : f(a,b) = 2*a*b if b < 0.5 else f(a,b) = 1 - 2*(1-a)(1-b)
        ####################
        buf32Lab = cv2.cvtColor(((buf0.astype(np.float32)) / 256).astype(np.float32), cv2.COLOR_BGR2Lab)
        # get height of current image
        h = buf0.shape[0]
        """
        rect = getattr(self, 'rect', None)
        if rect is not None:
            rect = rect.intersected(QRect(0, 0, buf0.shape[1], buf0.shape[0]))
            adjustForm.filterStart = int((rect.top() / h) * 100.0)
            adjustForm.filterEnd = int((rect.bottom() / h) * 100.0)
            adjustForm.sliderFilterRange.setStart(adjustForm.filterStart)
            adjustForm.sliderFilterRange.setEnd(adjustForm.filterEnd)
        """
        if adjustForm.filterEnd > 4:
            # build the filter as a 1D array of size h
            s = 0  # strongest 0
            opacity = 1 - s
            if adjustForm.kernelCategory == blendFilterIndex.GRADUALNONE:
                start, end = 0, h - 1
            else:
                start = int(h * adjustForm.filterStart / 100.0)
                end = int(h * adjustForm.filterEnd / 100.0)
            test = np.arange(end - start) * opacity / (
                        2.0 * max(end - start - 1, 1)) + 0.5 * s  # range 0.5*s...0.5
            test = np.concatenate((np.zeros(start) + 0.5 * s, test, np.zeros(h - end) + 0.5))
            if adjustForm.kernelCategory == blendFilterIndex.GRADUALBT:
                # rotate filter 180°
                test = test[::-1]
            # blend the filter with the image
            Lchan = buf32Lab[:, :, 0]
            test1 = test[:, np.newaxis] + np.zeros(Lchan.shape)
            buf32Lab[:, :, 0] = np.where(Lchan < 50, Lchan * (test1 * 2.0),
                                         100.0 - 2.0 * (1.0 - test1) * (100.0 - Lchan))
            # luminosity correction
            # buf32Lab[:,:,0] = buf32Lab[:,:,0]*(1.0+0.1)
            bufRGB32 = cv2.cvtColor(buf32Lab, cv2.COLOR_Lab2RGB)
            buf1[:, :, :3][:, :, ::-1] = (bufRGB32 * 255.0).astype(np.uint8)
        # forward the alpha channel
        buf1[:, :, 3] = buf0[:, :, 3]
        self.updatePixmap()

    def applyTemperature(self):
        """
        Warming/cooling filter.
        The method implements two algorithms.
        - Photo/Color filter : Blending using mode multiply, plus correction of luminosity
            by blending the output image with the inputImage, using mode luminosity.
        - Chromatic adaptation : multipliers in linear sRGB.
        """
        adjustForm = self.getGraphicsForm()
        options = adjustForm.options
        temperature = adjustForm.tempCorrection
        tint = adjustForm.tintCorrection  # range -1..1
        inputImage = self.inputImg()
        buf1 = QImageBuffer(inputImage)
        currentImage = self.getCurrentImage()
        if not options['Color Filter']:
            # neutral point : forward input image and return
            if abs(temperature - 6500) < 200 and tint == 0:
                buf0 = QImageBuffer(currentImage)
                buf0[:, :, :] = buf1
                self.updatePixmap()
                return
        ################
        # photo filter
        ################
        if options['Photo Filter'] or options['Color Filter']:
            if options['Photo Filter']:
                # get black body color
                r, g, b = bbTemperature2RGB(temperature)
            else:
                # get current color from color chooser dialog
                # parent() is a dockWidget, parent().parent() is the main window
                r, g, b = adjustForm.filterColor.getRgb()[:3]
            filter = QImage(inputImage.size(), inputImage.format())
            filter.fill(QColor(r, g, b, 255))
            # draw image on filter using mode multiply
            qp = QPainter(filter)
            qp.setCompositionMode(QPainter.CompositionMode_Multiply)
            qp.drawImage(0, 0, inputImage)
            qp.end()
            # correct the luminosity of the resulting image,
            # by blending it with the inputImage, using mode luminosity.
            # Note that using perceptual brightness gives better results, unfortunately slower
            resImg = blendLuminosity(filter, inputImage)
            bufOutRGB = QImageBuffer(resImg)[:, :, :3][:, :, ::-1]
        #####################
        # Chromatic adaptation
        #####################
        elif options['Chromatic Adaptation']:
            """
            tint = 2**tint
            m1, m2, m3 = (1.0, tint, 1.0,) if tint >=1 else (1.0/tint, 1.0, 1.0/tint,)
            # get conversion matrix in XYZ color space
            M = conversionMatrix(temperature, sRGBWP)  # source is input image : sRGB, ref WP D65
            buf = QImageBuffer(inputImage)[:, :, :3]
            # convert to XYZ
            bufXYZ = sRGB2XYZVec(buf[:,:,::-1])             #np.tensordot(bufLinear, sRGB_lin2XYZ, axes=(-1, -1))
            # apply conversion matrix
            bufXYZ = np.tensordot(bufXYZ, M, axes=(-1, -1))
            """
            # get RGB multipliers
            m1, m2, m3, _ = temperatureAndTint2Multipliers(temperature, 2 ** tint, self.parentImage.RGB_lin2XYZInverse)  # TODO modified 24/02/20 validate
            buf = QImageBuffer(inputImage)[:, :, :3]
            bufXYZ = RGB2XYZ(buf[:, :, ::-1], RGB_lin2XYZ=self.parentImage.RGB_lin2XYZ)  # TODO modified 02/03/20 validate
            bufsRGBLinear = np.tensordot(bufXYZ, self.parentImage.RGB_lin2XYZInverse, axes=(-1, -1))  # TODO modified 24/02/20 validate
            # apply multipliers
            bufsRGBLinear *= [m1, m2, m3]
            # brightness correction
            M = np.max(bufsRGBLinear)
            bufsRGBLinear /= M
            bufOutRGB = rgbLinear2rgb(bufsRGBLinear)
            np.clip(bufOutRGB, 0, 255, out=bufOutRGB)
            bufOutRGB = bufOutRGB.astype(np.uint8)
        else:
            raise ValueError('applyTemperature : wrong option')
        # set output image
        bufOut0 = QImageBuffer(currentImage)
        bufOut = bufOut0[:, :, :3]
        bufOut[:, :, ::-1] = bufOutRGB
        # forward the alpha channel
        bufOut0[:, :, 3] = buf1[:, :, 3]
        self.updatePixmap()

