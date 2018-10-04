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
import gc

import itertools
import weakref

from os.path import isfile

from PySide2.QtCore import Qt, QDataStream, QFile, QIODevice, QSize, QPoint, QRectF, QMargins

import cv2
from copy import copy

from PySide2.QtGui import QImageReader, QTransform, QBrush
from PySide2.QtWidgets import QApplication, QSplitter
from PySide2.QtGui import QPixmap, QImage, QColor, QPainter
from PySide2.QtCore import QRect

import exiftool
from colorConv import sRGB2LabVec, Lab2sRGBVec, rgb2rgbLinearVec, \
    rgbLinear2rgbVec, sRGB2XYZVec, sRGB_lin2XYZInverse, temperatureAndTint2RGBMultipliers
from debug import tdec
from graphicsBlendFilter import blendFilterIndex

from graphicsFilter import filterIndex
from histogram import warpHistogram
from colorManagement import icc, convertQImage
from imgconvert import *
from colorCube import interpMulti, rgb2hspVec, hsp2rgbVec, LUT3DIdentity, LUT3D, interpVec_, hsv2rgbVec, \
    interpTetraVec_
from time import time

from settings import USE_TETRA
from utils import SavitzkyGolay, channelValues, checkeredImage, boundingRect, dlgWarn, baseSignal_bool, baseSignal_Int2, \
    qColorToRGB
from dwtdenoise import dwtDenoiseChan

class ColorSpace:
    notSpecified = -1; sRGB = 1

class metadataBag:
    """
    Container for vImage meta data
    """
    def __init__(self, name=''):
        self.name, self.colorSpace, self.rawMetadata, self.profile, self.orientation, self.rating = name, ColorSpace.notSpecified, [], '', None, 5

class vImage(QImage):
    """
    Versatile image class.
    This is the base class for multi-layered and interactive image
    classes, and for layer classes. It gathers all image information,
    including meta-data.
    A vImage object holds 4 images:
           - full (self),
           - thumbnail (self.thumb),
           - hald (self.hald) for LUT3D conversion,
           - mask (self.mask, disabled by default).
    Note : for performance self.thumb and self.hald are not synchronized with the image: they are initialized
    and handled independently of the full size image.
    """
    ################
    # max thumb size :
    # max(thimb.width(), thumb.height()) <= thumbsize
    ################
    thumbSize = 1500

    ###############
    # default base color, used to display semi transparent pixels
    ###############
    defaultBgColor = QColor(191, 191, 191,255)

    ##############
    # default mask colors
    # To be able to display masks as color masks, we use the red channel to code
    # the mask opacity, instead of its alpha channel.
    # When modifying these colors, it is mandatory to
    # modify the methods invertMask and color2OpacityMask accordingly.
    ##############
    defaultColor_UnMasked = QColor(128, 0, 0, 255)
    defaultColor_Masked = QColor(0, 0, 0, 255)
    defaultColor_Invalid = QColor(0, 99, 0, 255)
    defaultColor_UnMasked_Invalid = QColor(128, 99, 0, 255)

    @classmethod
    def color2OpacityMask(cls, mask):
        """
        Sets opacity channel from red channel (alpha = 0 if R==0 else 255),
        leaving other channels unchanged.
        @param mask: mask
        @type mask: QImage
        @return: opacity mask
        @rtype: QImage
        """
        mask = mask.copy()
        buf = QImageBuffer(mask)
        # set alpha channel from red channel
        buf[:, :, 3] = np.where(buf[:, :, 2] == 0, 0, 255)
        return mask
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
    def visualizeMask(cls, img, mask, color=True, clipping=False, copy=True):
        # copy image
        if copy:
            img = QImage(img)
        qp = QPainter(img)
        # qp.drawImage(QRect(0, 0, img.width(), img.height()), img) # TODO validate suppression 19/06/18
        if color:
            # draw mask as color mask with opacity 0.5
            qp.setCompositionMode(QPainter.CompositionMode_SourceOver)
            qp.setOpacity(0.5)
            qp.drawImage(QRect(0, 0, img.width(), img.height()), mask)
        else:
            # draw mask as opacity mask :
            # mode DestinationIn (set image opacity to mask opacity)
            qp.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            omask = vImage.color2OpacityMask(mask)
            qp.drawImage(QRect(0, 0, img.width(), img.height()), omask)
            if False:#clipping:  # TODO 6/11/17 may be we should draw checker for both selected and unselected mask
                # draw checker
                qp.setCompositionMode(QPainter.CompositionMode_DestinationOver)
                qp.setBrush(QBrush(checkeredImage()))
                qp.drawRect(QRect(0, 0, img.width(), img.height()))  # 0.26s for 15Mpx
        qp.end()
        return img
    @classmethod
    def maskDilate(cls, mask, iterations=1):
        """
        Copy the mask and increases the masked region by applying
        a 5x5 min filter.
        @param mask:
        @type mask: ndarray RGB channels
        @param iterations: kernal iteartion count
        @type iterations: int
        @return: the dialted mask
        @rtype:
        """
        mask = mask.copy()
        kernel = np.ones((5, 5), np.uint8)
        # CAUTION erode decreases values (min filter), so it extends the masked part of the image
        mask[:, :, 2] = cv2.erode(mask[:, :, 2], kernel, iterations=iterations)
        return mask
    @classmethod
    def maskErode(cls, mask,iterations=1):
        """
        Copy the mask and reduces the masked region by applying
        a 5x5 max filter.
        @param mask:
        @type mask: ndarray BGR channels
        @param iterations: kernel iteration count
        @type iterations: int
        @return: the eroded mask
        @rtype: ndarray
        """
        mask = mask.copy()
        kernel = np.ones((5, 5), np.uint8)
        # CAUTION dilate increases values (max filter), so it reduces the masked region of the image
        mask[:, :, 2] = cv2.dilate(mask[:, :, 2], kernel, iterations=iterations)
        return mask
    @ classmethod
    def seamlessMerge(cls, dest, source, mask, cloningMethod):
        """
        Seamless cloning of source into dest
        mask is a color mask.
        The dest image is modified.
        @param dest:
        @type dest: vImage
        @param source:
        @type source: vImage
        @param mask: color mask
        @type mask: QImage
        @param cloningMethod:
        @type cloningMethod:
        """
        # scale mask to source size and convert to opacity mask
        src_mask = vImage.color2OpacityMask(mask).scaled(source.size())
        buf = QImageBuffer(src_mask)
        tmp = np.where(buf[:, :, 3] == 0, 0, 255)
        # get src_mask bounding rect dilated by margin.
        # All rectangle coordinates are relative to src_mask
        margin = 100
        oRect = boundingRect(tmp, 255)
        bRect = oRect + QMargins(margin, margin, margin, margin)
        inRect = bRect & QImage.rect(src_mask)
        # look for masked pixels
        if bRect is None:
            # no white pixels
            dlgWarn("seamlessMerge : no masked pixel found")
            return
        # set white mask
        bt, bb, bl, br = inRect.top(), inRect.bottom(), inRect.left(), inRect.right()
        src_maskBuf = np.dstack((tmp, tmp, tmp)).astype(np.uint8)[bt:bb+1,bl:br+1,:]
        #center = (bl + bRect.width() // 2, bt + bRect.height() // 2)
        sourceBuf = QImageBuffer(source)[bt:bb+1,bl:br+1,:]
        destBuf = QImageBuffer(dest)[bt:bb+1,bl:br+1,:]
        # The cloning center is the center of oRect. We look for its coordinates
        # relative to inRect
        center = (oRect.width()//2 + oRect.left() - inRect.left() , oRect.height()//2 + oRect.top() - inRect.top())
        output = cv2.seamlessClone(sourceBuf[:, :, :3][:, :, ::-1],  # source
                                   destBuf[:, :, :3][:, :, ::-1],    # dest
                                   src_maskBuf,
                                   center, cloningMethod
                                   )
        # copy output into dest
        destBuf[:, :, :3][:, :, ::-1] = output  # assign src_ maskBuf for testing

    def __init__(self, filename=None, cv2Img=None, QImg=None, mask=None, format=QImage.Format_ARGB32,
                 name='', colorSpace=-1, orientation=None, rating=5, meta=None, rawMetadata=None, profile=''):
        """
        With no parameter, builds a null image.
        Mask is disabled by default.
        image is assumed to be in the color space sRGB : colorSpace value is used only as meta data.
        @param filename: path to file
        @type filename: str
        @param cv2Img: data buffer
        @type cv2Img: ndarray
        @param QImg: image
        @type QImg: QImage
        @param mask: Image mask. Should have format and dims identical to those of image
        @type mask: QImage
        @param format: QImage format (default QImage.Format_ARGB32)
        @type format: QImage.Format
        @param name: image name
        @type name: str
        @param colorSpace: color space (default : notSpecified)
        @type colorSpace: MarkedImg.colorSpace
        @param orientation: Qtransform object (default None)
        @type orientation: Qtransform 
        @param meta: metadata instance (default None)
        @type meta: MarkedImg.metadataBag
        @param rawMetadata: list of dictionaries (default [])
        @type rawMetadata: list of dictionaries
        @param profile: embedded profile (default '')
        @type profile: str
        """
        # color management : we assume the working profile is the image profile
        # self.colorTransformation = icc.workToMonTransform
        # current color managed image
        # self.cmImage = None
        if rawMetadata is None:
            rawMetadata = []
        self.isModified = False
        self.rect, self.mask, = None, mask
        self.filename = filename if filename is not None else ''
        self.isCropped = False
        self.cropTop, self.cropBottom, self.cropLeft, self.cropRight = (0,) * 4
        self.isRuled = False

        # mode flags
        self.useHald = False
        self.hald = None
        self.isHald = False
        self.useThumb = False

        # Cache buffers
        self.cachesEnabled = True
        self.qPixmap = None
        self.rPixmap = None
        self.hspbBuffer = None
        self.LabBuffer = None
        self.HSVBuffer = None

        # preview image.
        # Conceptually, the layer stack can be seen as
        # the juxtaposition of two stacks:
        #  - a stack of full sized images
        #  - a stack of thumbnails
        # For the sake of performance, the two stacks are
        # NOT synchronized : they are updated independently.
        # Thus, after initialization, the thumbnail should
        # NOT be calculated from the full size image.
        self.thumb = None
        self.onImageChanged = lambda: 0
        if meta is None:
            # init metadata container
            self.meta = metadataBag()
            self.meta.name, self.meta.colorSpace, self.meta.rawMetadata, self.meta.profile, self.meta.orientation, self.meta.rating = name, colorSpace, rawMetadata, profile, orientation, rating
        else:
            self.meta = meta
        if (filename is None and cv2Img is None and QImg is None):
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
            if hasattr(QImg, "meta"):
                self.meta = copy(QImg.meta)
        elif cv2Img is not None:
            # build image from buffer
            super().__init__(ndarrayToQImage(cv2Img, format=format))
        # check format
        if self.depth() != 32:
            raise ValueError('vImage : should be a 8 bits/channel color image')
        # mask
        self.maskIsEnabled = False
        self.maskIsSelected = False
        if self.mask is None:
            self.mask = QImage(self.width(), self.height(), format)
            # default : unmask all
            self.mask.fill(self.defaultColor_UnMasked)
        #self.updatePixmap()
        if type(self) in [QLayer]: # [QLayer, QPresentationLayer]:  TODO 07/06/18 validate modif
            #vImage.updatePixmap(self)
            self.updatePixmap()

    def setImage(self, qimg):
        """
        copies qimg to image. Does not update metadata.
        image and qimg must have identical dimensions and type.
        @param qimg: QImage object
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

    def initThumb(self):
        """
        Inits the image thumbnail as a QImage. In contrast to
        maskedThumbContainer, thumb is never used as an input image, thus
        there is no need for a type yielding color space buffers.
        Layer thumbs own an attribute parentImage set by the overridden method QLayer.initThumb.
        For non adjustment layers, the thumbnail will never be updated. So, we
        perform a high quality scaling.
        """
        scImg = self.scaled(self.thumbSize, self.thumbSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # With the Qt.SmoothTransformation flag, the output image format is premultiplied
        self.thumb = scImg.convertToFormat(QImage.Format_ARGB32, Qt.DiffuseDither | Qt.DiffuseAlphaDither)

    def getThumb(self):
        """
        inits image thumbnail if needed and returns it.
        @return: thumbnail
        @rtype: QImage
        """
        if self.thumb is None:
            self.initThumb()
        return self.thumb

    def initHald(self):
        """
        Builds a hald image (as a QImage) from identity 3D LUT.
    A hald can be viewed as a 3D LUT flattened and reshaped as an array representing an image.
        Pixel channels keep the same order, and

        """
        if not self.cachesEnabled:
            return
        s = int((LUT3DIdentity.size )**(3.0/2.0)) + 1
        buf0 = LUT3DIdentity.getHaldImage(s, s)
        buf1 = QImageBuffer(self.Hald)
        buf1[:,:,:]=buf0
        buf1[:, :, 3] = 255  # TODO added 15/11/17 for coherence with the overriding function QLayer.initHald()

    def resize_coeff(self, widget):
        """
        Normalization of self.Zoom_coeff.
        Returns the current resizing coefficient used by
        the widget paint event handler to display the image.
        The coefficient is chosen to initially (i.e. when self.Zoom_coeff = 1)
        fill the widget without cropping.
        For splitted views we use the size of the QSplitter parent
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
            r = min(r_w, r_h)  # TODO modified 17/09/18 validate to prevent cropping in diaporama for portrait mode
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
        The method is overridden in QLayer
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
        Maps x,y coordinates of pixel in the full image to
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

    def getHspbBuffer(self):
        """
        return the image buffer in color mode HSpB.
        The buffer is recalculated if needed and cached.
        @return: HSPB buffer 
        @rtype: ndarray
        """
        #inputImage = self.inputImgFull().getCurrentImage()
        if self.hspbBuffer is None  or not self.cachesEnabled:
            currentImage = self.getCurrentImage()
            self.hspbBuffer = rgb2hspVec(QImageBuffer(currentImage)[:,:,:3][:,:,::-1])
        return self.hspbBuffer

    def getLabBuffer(self):
        """
        returns the image buffer in color mode Lab.
        The buffer is recalculated if needed and cached.
        @return: Lab buffer, L range is 0..1, a, b ranges are -128..+128
        @rtype: numpy ndarray, dtype np.float
        """
        if self.LabBuffer is None  or not self.cachesEnabled:
            currentImage = self.getCurrentImage()
            self.LabBuffer = sRGB2LabVec(QImageBuffer(currentImage)[:, :, :3][:, :, ::-1])
        return self.LabBuffer

    def getHSVBuffer(self):
        """
        returns the image buffer in color mode HSV.
        The buffer is calculated if needed and cached.
        H,S,V ranges are 0..255 (opencv convention for 8 bits images)
        @return: HSV buffer
        @rtype: numpy ndarray, dtype np.float
        """
        if self.HSVBuffer is None  or not self.cachesEnabled:
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

    def cacheInvalidate(self):
        """
        Invalidate cache buffers. The method is
        called in applyToStack for each layer after layer.execute
        """
        self.hspbBuffer = None
        self.LabBuffer = None
        self.HSVBuffer = None
        if hasattr(self, 'maskedImageContainer'):
            if self.maskedImageContainer is not None:
                self.maskedImageContainer.cacheInvalidate()
        if hasattr(self, 'maskedThumbContainer'):
            if self.maskedThumbContainer is not None:
                self.maskedThumbContainer.cacheInvalidate()

    def updatePixmap(self, maskOnly=False):
        """
        Updates the qPixmap, rPixmap, thumb and cmImage caches.
        The image is that returned by getCurrentImage(), thus
        the caches are synchronized using the current image
        mode (full or preview).

        If maskOnly is True, cmImage is not updated.
        if maskIsEnabled is False, the mask is not shown.
        If maskIsEnabled is True, then
            - if maskIsSelected is True, the mask is drawn over
              the layer as a color and opacity mask, with its own
              pixel color and inverse opacity.
            - if maskIsSelected is False, the mask is drawn as an
              opacity mask, setting image opacity to that of mask
              (mode DestinationIn). Color mask is no used.
        NOTE : the fully masked part of the image corresponds to
        mask opacity = 0.
        @param maskOnly: default False
        @type maskOnly: boolean
        """
        currentImage = self.getCurrentImage()
        """
        if not maskOnly:
            # invalidate color managed cache
            self.cmImage = None
        # get current image and apply color management if self is the presentation layer
        if icc.COLOR_MANAGE and getattr(self, 'role', None) == 'presentation':
            if self.cmImage is None:
                # CAUTION : reset alpha channel
                img = convertQImage(currentImage, transformation=self.colorTransformation)
                # copy  alpha channel
                buf0 = QImageBuffer(img)
                buf1 = QImageBuffer(currentImage)
                buf0[:, :, 3] = buf1[:, :, 3]
            else:
                #img = QImage(self.cmImage)
                img = self.cmImage
        else:
            #img = QImage(currentImage)
            img = currentImage
        # refresh cache
        if maskOnly:
            #self.cmImage = QImage(img)
            self.cmImage = img
        self.cmImage = img
        qImg = img
        """
        rImg = currentImage
        if self.maskIsEnabled:
            #qImg = vImage.visualizeMask(qImg, self.mask, color=self.maskIsSelected, clipping=self.isClipping)
            rImg = vImage.visualizeMask(rImg, self.mask, color=self.maskIsSelected, clipping=self.isClipping)
        #self.qPixmap = QPixmap.fromImage(qImg)
        self.rPixmap = QPixmap.fromImage(rImg)

    def resetMask(self, maskAll=False):
        """
        Reinits the mask : all pixels are masked or all
        pixels are unmasked (default).
        @param maskAll:
        @type maskAll: boolean
        """
        self.mask.fill(vImage.defaultColor_Masked if maskAll else vImage.defaultColor_UnMasked)
        self.updatePixmap(maskOnly=True)

    def invertMask(self):
        """
        Inverts mask: masked/unmasked pixels
        are coded by red = 0/128
        """
        buf = QImageBuffer(self.mask)
        buf[:, :,2] = 128 - buf[:,:,2]  #np.where(buf[:,:,2]==128, 0, 128)

    def resize(self, pixels, interpolation=cv2.INTER_CUBIC):
        """
        Resizes an image while keeping its aspect ratio. We use
        the opencv function cv2.resize() to perform the resizing operation, so we
        can choose among several interpolation methods (default cv2.INTER_CUBIC).
        The original image is not modified.
        @param pixels: pixel count for the resized image
        @type pixels: int
        @param interpolation: interpolation method (default cv2.INTER_CUBIC)
        @type interpolation:
        @return : the resized vImage
        @rtype : vImage
        """
        ratio = self.width() / float(self.height())
        w, h = int(np.sqrt(pixels * ratio)), int(np.sqrt(pixels / ratio))
        hom = w / float(self.width())
        # resizing
        Buf = QImageBuffer(self)
        cv2Img = cv2.resize(Buf, (w, h), interpolation=interpolation)
        rszd = vImage(cv2Img=cv2Img, meta=copy(self.meta), format=self.format())
        # prevent buffer from garbage collector
        rszd.dummy = cv2Img
        #resize rect and mask
        if self.rect is not None:
            rszd.rect = QRect(self.rect.left() * hom, self.rect.top() * hom, self.rect.width() * hom, self.rect.height() * hom)
        if self.mask is not None:
            rszd.mask = self.mask.scaled(w, h)
        self.setModified(True)
        return rszd

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
        bufIn = QImageBuffer(self.inputImg())
        bufOut = QImageBuffer(self.getCurrentImage())
        bufOut[:,:,:] = bufIn
        self.updatePixmap()

    def applyCloning(self, seamless=True):
        """
        Seamless cloning
        @param seamless:
        @type seamless: boolean
        """
        # TODO 02/04/18 called multiple times by mouse events, cacheInvalidate should be called!!
        imgIn = self.inputImg()
        imgOut = self.getCurrentImage()
        pxIn = QPixmap.fromImage(imgIn)
        ########################
        # hald pass through
        if self.parentImage.isHald:
            buf0 = QImageBuffer(imgIn)
            buf1 = QImageBuffer(imgOut)
            buf1[...] = buf0
            return
        ########################
        # draw the translated and zoomed input image on the output image
        #if not self.cloned :
        # erase previous transformed image : reset imgOut to ImgIn
        qp = QPainter(imgOut)
        qp.setCompositionMode(QPainter.CompositionMode_Source)
        qp.drawPixmap(QRect(0, 0, imgOut.width(), imgOut.height()), pxIn, pxIn.rect())
        # get translation relative to current Image
        currentAltX, currentAltY = self.full2CurrentXY(self.xAltOffset, self.yAltOffset)
        # draw translated and zoomed input image (nothing is drawn outside of dest. image)
        qp.setCompositionMode(QPainter.CompositionMode_SourceOver)
        rect = QRectF(currentAltX, currentAltY, imgOut.width()*self.AltZoom_coeff, imgOut.height()*self.AltZoom_coeff)
        qp.drawPixmap(rect, pxIn, pxIn.rect())
        qp.end()
        # do seamless cloning
        if seamless:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                QApplication.processEvents()
                imgInc = imgIn.copy()
                src = imgOut
                vImage.seamlessMerge(imgInc, src, self.mask, self.cloningMethod)
                bufOut = QImageBuffer(imgOut)
                bufOut[:, :, :3] = QImageBuffer(imgInc)[:,:,:3]
            finally:
                self.parentImage.setModified(True)
                QApplication.restoreOverrideCursor()
                QApplication.processEvents()
        # forward the alpha channel
        # TODO 23/06/18 should forward ?
        self.updatePixmap()
        # the presentation layer must be updated here because
        # applyCloning is called directly (mouse and Clone button events).
        self.parentImage.prLayer.update()

    def applyKnitting(self):
        """
        Knitting
        """
        imgIn = self.inputImg()
        imgOut = self.getCurrentImage()
        ########################
        # hald pass through
        if self.parentImage.isHald:
            buf0 = QImageBuffer(imgIn)
            buf1 = QImageBuffer(imgOut)
            buf1[...] = buf0
            return
        ########################

        imgInc = imgIn.copy()
        src = self.parentImage.layersStack[self.sourceIndex].getCurrentImage()
        vImage.seamlessMerge(imgInc, src, self.mask, self.cloningMethod)
        bufOut = QImageBuffer(imgOut)
        bufOut[:, :, :3] = QImageBuffer(imgInc)[:,:,:3]
        self.updatePixmap()

    def applyGrabcut(self, nbIter=2, mode=cv2.GC_INIT_WITH_MASK):
        """
        Segmentation
        @param nbIter:
        @type nbIter: int
        @param mode:
        """
        form = self.view.widget()
        formOptions = form.listWidget1
        inputImg = self.inputImg()
        ##################################################################
        # pass through
        # grabcut is done only when clicking the Apply button of segmentForm.
        # No grabcut for hald image
        if self.noSegment or self.parentImage.isHald:
            inBuf = QImageBuffer(inputImg)
            outputImg = self.getCurrentImage()
            outBuf = QImageBuffer(outputImg)
            outBuf[:,:,:] = inBuf
            self.updatePixmap()
            return
        ##################################################################
        rect = self.rect
        # resizing coeff fitting selection rectangle with current image
        r = inputImg.width() / self.width()
        # mask
        if rect is not None:
            # inside : PR_FGD, outside BGD
            finalMask = np.zeros((inputImg.height(), inputImg.width()), dtype=np.uint8) + cv2.GC_BGD
            finalMask[int(rect.top() * r):int(rect.bottom() * r), int(rect.left() * r):int(rect.right() * r)] = cv2.GC_PR_FGD
        else:
            # all : PR_BGD
            finalMask = np.zeros((inputImg.height(), inputImg.width()), dtype=np.uint8) + cv2.GC_PR_BGD
        # add info from self.mask
        # initially (before any painting with BG/FG tools and before first call to applygrabcut)
        # all mask pixels are marked as invalid. Painting a pixel marks it as valid, Ctrl+paint
        # switches it to invalid. Only valid pixel info is added to finalMask, fixing them as FG or BG
        if inputImg.size() != self.size():
            scaledMask = self.mask.scaled(inputImg.width(), inputImg.height())
        else:
            scaledMask = self.mask
        scaledMaskBuf = QImageBuffer(scaledMask)
        # Only actually painted pixels of the mask should be considered
        finalMask[(scaledMaskBuf[:, :, 2] > 100 )* (scaledMaskBuf[:,:,1]!=99)] = cv2.GC_FGD  # 99=defaultColorInvalid R=128 unmasked R=0 masked
        finalMask[(scaledMaskBuf[:,:, 2]==0) *(scaledMaskBuf[:,:,1]!=99)] = cv2.GC_BGD
        # save a copy of the mask
        finalMask_save = finalMask.copy()
        # mandatory : at least one (FGD or PR_FGD)  pixel and one (BGD or PR_BGD) pixel
        if not ((np.any(finalMask == cv2.GC_FGD) or np.any(finalMask == cv2.GC_PR_FGD))
                and
                (np.any(finalMask == cv2.GC_BGD) or np.any(finalMask == cv2.GC_PR_BGD))):
            dlgWarn('You must select some background or foreground pixels', info='Use selection rectangle or draw mask')
            return
        bgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the background model
        fgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the foreground model
        t0 = time()
        inputBuf = QImageBuffer(inputImg)
        bgdmodel_test = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the background model
        fgdmodel_test = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the foreground model
        # get the fastest available method for segmentation
        if getattr(cv2, 'grabCut_mt', None) is None:
            bGrabcut = cv2.grabCut
        else:
            bGrabcut = cv2.grabCut_mt
        # apply grabcut
        bGrabcut(inputBuf[:, :, :3], finalMask, None,  # QRect2tuple(img0_r.rect),
                        bgdmodel, fgdmodel, nbIter, mode)
        print ('grabcut_mtd time : %.2f' % (time()-t0))
        """
        t1 = time()
        cv2.grabCut(inputBuf[:, :, :3], finalMask_test, None,  # QRect2tuple(img0_r.rect),
                        bgdmodel_test, fgdmodel_test, nbIter, mode)
        print('grabcut time : %.2f' % (time() - t1))
        """
        # keep unmodified initial FGD and BGD pixels : #TODO 22/06/18 alraedy done by the function grabcut
        finalMask = np.where((finalMask_save==cv2.GC_BGD) + (finalMask_save == cv2.GC_FGD), finalMask_save, finalMask)

        # set opacity (255=background, 0=foreground)
        # We want to keep the colors of mask pixels. Unfortunately,
        # while drawing or scaling, Qt replaces the colors of transparent pixels by 0.
        # So, we can't set now the mask alpha channel.
        finalOpacity = np.where((finalMask == cv2.GC_FGD) + (finalMask == cv2.GC_PR_FGD), 255, 0)
        buf = QImageBuffer(scaledMask)
        # set the red channel of the mask
        buf[:,:,2] = np.where(finalOpacity==255, 128, 0)
        invalidate_contour = True  # always True (for testing purpose)
        if invalidate_contour:
            # without manual corrections, only the contour region may be updated
            # by further calls to applyGrabcut()
            # build the contour as a boolean mask : cbMask is True iff the pixel belongs to the contour
            maxIterations = 8
            ebuf = vImage.maskErode(buf, iterations=maxIterations)
            dbuf = vImage.maskDilate(buf, iterations=maxIterations)
            cbMask = ((buf[:, :, 2] == 0) & (ebuf[:, :, 2] == 128)) | ((buf[:, :, 2] == 128) & (dbuf[:, :, 2] == 0))
            # mark contour pixels as invalid and others as valid : the contour only can be modified
            # Unlike in the case of integer index arrays, in the boolean case, the result is a 1-D array
            # containing all the elements in the indexed array corresponding to all the true elements in the boolean array.
            buf[:, :,1][cbMask] = 99
            buf[:,:,1][~cbMask] = 0
            # dilate the mask to remove background dithering
            # iterations should be <= maxIterations
            dbuf = vImage.maskDilate(buf, iterations=min(form.contourMargin, maxIterations))
            innerCbMask = ((buf[:, :, 2] == 128) & (dbuf[:, :, 2] == 0))
            buf[:, :, 2][innerCbMask] = 0
        else:
            # invalidate PR_FGD and PR_BGD pixels, validate others
            buf[:,:,1] = np.where((finalMask!=cv2.GC_FGD) * (finalMask!=cv2.GC_BGD), 99, 0)
        if self.size() != scaledMask.size():
            self.mask = scaledMask.scaled(self.size())
        else:
            self.mask = scaledMask
        # Set clipping mode for better viewing
        name = formOptions.intNames[0]
        form.start = False
        # check option clipping
        name = formOptions.intNames[0]
        item = formOptions.items[name]
        formOptions.checkOption(name, checked=self.isClipping, callOnSelect=False)
        #item.setCheckState(Qt.Checked if self.isClipping else Qt.Unchecked)
        #formOptions.options[name] = True if self.isClipping else False
        # forward the alpha channel
        # TODO 23/06/18 should forward ?
        self.updatePixmap()

    def applyInvert(self):
        """
        Inverts image
        """
        bufIn = QImageBuffer(self.inputImg())[:,:,:3]
        # get orange mask from negative brightest (unexposed) pixels
        temp = np.sum(bufIn, axis=2)
        ind = np.argmax(temp)
        ind = np.unravel_index(ind, (bufIn.shape[0], bufIn.shape[1],))
        Mask0, Mask1, Mask2 = bufIn[ind]
        if self.view is not None:
            form = self.view.widget()
            Mask0, Mask1, Mask2 =  form.Bmask, form.Gmask, form.Rmask
        currentImage = self.getCurrentImage()
        bufOut = QImageBuffer(currentImage)
        # eliminate mask
        tmp = (bufIn[:, :, :3] / [Mask0, Mask1, Mask2]) * 255
        np.clip(tmp, 0, 255, out=tmp)
        bufOut[:, :, :3] = 255.0 - tmp #(bufIn[:, :, :3] / [Mask0, Mask1, Mask2]) * 255
        self.updatePixmap()

    def applyExposure(self, exposureCorrection, options):
        """
        Applies exposure correction 2**exposureCorrection
        to the linearized RGB channels.

        @param exposureCorrection:
        @type exposureCorrection: float
        @param options:
        @type options:
        @return:
        @rtype:
        """
        # neutral point
        if abs(exposureCorrection) < 0.05:
            buf0 = QImageBuffer(self.getCurrentImage())
            buf1 = QImageBuffer(self.inputImg())
            buf0[:, :, :] = buf1
            self.updatePixmap()
            return
        bufIn = QImageBuffer(self.inputImg())
        buf = bufIn[:,:,:3][:,:,::-1]
        buf = rgb2rgbLinearVec(buf)

        buf[:,:,:] = buf * (2 ** exposureCorrection)

        buf = rgbLinear2rgbVec(buf)

        buf = np.clip(buf, 0.0, 255.0)
        currentImage = self.getCurrentImage()
        ndImg1a = QImageBuffer(currentImage)
        ndImg1a[:, :, :3][:, :, ::-1] = buf
        # forward the alpha channel
        ndImg1a[:, :, 3] = bufIn[:,:,3]
        self.updatePixmap()

    def applyTransForm(self, options):
        """
        Applies the geometric transformation defined by source and target quads
        @param options:
        @type options:
        """
        if self.maskIsEnabled:
            dlgWarn("A masked layer cannot be transformed.\nDisable the mask")
            return
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
        #dummy = inImg.transformed(T)
        img = (inImg.transformed(T)).copy(QRect(-rectTrans.x()*s, -rectTrans.y()*s, w, h))
        if img.isNull():
            print('applyTransform : transformation fails')
            self.tool.restore()
            return
        buf0[:,:,:] = QImageBuffer(img)
        self.updatePixmap()

    def applyImage(self, options):
        self.applyTransForm(options)

    def applyNoiseReduction(self):
        adjustForm = self.view.widget()
        noisecorr = adjustForm.noiseCorrection
        currentImage = self.getCurrentImage()
        inputImage = self.inputImg()
        ########################
        # hald pass through
        if self.parentImage.isHald:
            buf0 = QImageBuffer(inputImage)
            buf1 = QImageBuffer(currentImage)
            buf1[...] = buf0
            return
        ########################
        if noisecorr == 0:
            bufOut = QImageBuffer(currentImage)
            bufOut[:, :, :3] = QImageBuffer(currentImage)[:,:,:3]
            self.updatePixmap()
            return
        buf01 = QImageBuffer(inputImage)
        buf0 = buf01[:, :, :3][:, :, ::-1]
        noisecorr *= currentImage.width() / self.width()
        if adjustForm.options['Wavelets']:
            noisecorr *= 100
            bufLab = cv2.cvtColor(buf0, cv2.COLOR_RGB2Lab)
            L = dwtDenoiseChan(bufLab, chan=0, thr=noisecorr, thrmode='wiener', level=8 if self.parentImage.useThumb else 11)
            A = dwtDenoiseChan(bufLab, chan=1, thr=noisecorr, thrmode='wiener', level=8 if self.parentImage.useThumb else 11)
            B = dwtDenoiseChan(bufLab, chan=2, thr=noisecorr, thrmode='wiener', level=8 if self.parentImage.useThumb else 11)
            L = np.clip(L, 0, 255)
            A = np.clip(A, 0,255)
            B = np.clip(B, 0,255)
            bufLab = np.dstack((L, A, B))
            buf1 = cv2.cvtColor(bufLab.astype(np.uint8), cv2.COLOR_Lab2RGB)
        elif adjustForm.options['Bilateral']:
           buf1 = cv2.bilateralFilter(buf0,
                                         9 if self.parentImage.useThumb else 15,   # 21:5.5s, 15:3.5s, diameter of
                                                                                   # (coordinate) pixel neighborhood,
                                                                                   # 5 is the recommended value for fast processing
                                         10 * adjustForm.noiseCorrection,          # std deviation sigma
                                                                                   # in color space,  100 middle value
                                         50 if self.parentImage.useThumb else 150, # std deviation sigma
                                                                                   # in coordinate space,  100 middle value
                                         )
        elif adjustForm.options['NLMeans']:
            buf1 = cv2.fastNlMeansDenoisingColored(buf0, None, 1+noisecorr, 1+noisecorr, 7, 21) # hluminance, hcolor,  last params window sizes 7, 21 are recommended values

        bufOut = QImageBuffer(currentImage)
        bufOut[:, :, :3][:, :, ::-1] = buf1
        # forward the alpha channel
        bufOut[:,:,3] = buf01[:,:,3]
        self.updatePixmap()

    def applyRawPostProcessing(self):
        """
        Develop raw image.
        Processing order is the following:
             1 - process raw image
             2 - contrast correction
             3 - saturation correction
        All operations are applied to 16 bits per channel images and
        the final image is converted to 8 bits.
        An Exception AttributeError is raised if rawImage
        is not an attribute of self.parentImage.
        """
        if self.parentImage.isHald :
            raise ValueError('Cannot build a 3D LUT from raw stack')
        #################
        # a priori underexposition compensation for most DSLR
        #################
        baseExpShift = 3.0
        # get adjustment form and rawImage
        adjustForm = self.view.widget()
        options = adjustForm.options
        rawImage = getattr(self.parentImage, 'rawImage')
        currentImage = self.getCurrentImage()
        ######################################################################################################################
        # process raw image (16 bits mode)                        RGB ------diag(multipliers)-----------> RGB
        # post processing pipeline (from libraw):                   |                                      |
        # - black subtraction                                       | rawpyObj.rgb_xyz_matrix[:3,:]        | sRGB_lin2XYZ
        # - exposure correction                                     |                                      |
        # - white balance                                           |                                      |
        # - demosaic                                               XYZ                                    XYZ
        # - data scaling to use full range
        # - conversion to output color space
        # - gamma curve and brightness correction : gamma(imax) = 1, imax = 8*white/brightness
        # ouput is CV_16UC3
        ######################################################################################################################
        # postProcessCache is reset to None by graphicsRaw.updateLayer (graphicsRaw.dataChanged event handler)
        if self.postProcessCache is None:
            # build sample images for a set of multipliers
            if adjustForm.sampleMultipliers:
                bufpost16 = np.empty((self.height(), self.width(), 3), dtype=np.uint16)
                m = adjustForm.rawMultipliers
                co = np.array([0.85, 1.0, 1.2])
                mults = itertools.product(m[0]*co, [m[1]], m[2]*co)
                adjustForm.samples = []
                for i, mult in enumerate(mults):
                    adjustForm.samples.append(mult)
                    mult = (mult[0], mult[1], mult[2], mult[1])
                    print(mult, '   ', m)
                    bufpost_temp = rawImage.postprocess(
                                            exp_shift=baseExpShift+2.0**(2.0*adjustForm.expCorrection) if not options['Auto Brightness'] else 0,
                                            no_auto_bright= (not options['Auto Brightness']),
                                            use_auto_wb=options['Auto WB'],
                                            use_camera_wb=False,#options['Camera WB'],
                                            user_wb = mult,#adjustForm.rawMultipliers,
                                            gamma= (2.222, 4.5),  # default REC BT 709 exponent, slope
                                            #gamma=(2.4, 12.92), # sRGB exponent, slope cf. https://en.wikipedia.org/wiki/SRGB#The_sRGB_transfer_function_("gamma")
                                            exp_preserve_highlights = 1.0 if options['Preserve Highlights'] else 0.8,
                                            output_bps=16,
                                            bright=adjustForm.brCorrection,  # default 1
                                            hightlightmode=adjustForm.highCorrection if not options['Auto Brightness'] else 0,
                                            # no_auto_scale= (not options['Auto Scale']) don't use : green shift
                                            )
                    row = i // 3
                    col = i % 3
                    w, h = int(bufpost_temp.shape[1]/3), int(bufpost_temp.shape[0]/3)
                    bufpost_temp = cv2.resize(bufpost_temp, (w, h))
                    bufpost16[row*h:(row+1)*h, col*w:(col+1)*w,: ]=bufpost_temp
            # develop
            else:
                # highlight_mode : restauration of overexposed highlights. 0: clip, 1:unclip, 2:blend, 3...: rebuild
                bufpost16 = rawImage.postprocess(
                    exp_shift=baseExpShift+2.0**(2.0*adjustForm.expCorrection) if not options['Auto Brightness'] else 0,
                    no_auto_bright=(not options['Auto Brightness']),
                    use_auto_wb=options['Auto WB'],
                    use_camera_wb=options['Camera WB'],
                    user_wb=adjustForm.rawMultipliers,
                    gamma= (2.222, 4.5),  # default REC BT 709 exponent, slope
                    #gamma=(2.4, 12.92), # sRGB exponent, slope cf. https://en.wikipedia.org/wiki/SRGB#The_sRGB_transfer_function_("gamma")
                    exp_preserve_highlights=0.99 if options['Preserve Highlights'] else 0.6,
                    output_bps=16,
                    bright=adjustForm.brCorrection,  # > 0, default 1
                    highlight_mode=adjustForm.highCorrection if not options['Auto Brightness'] else 0,
                    #no_auto_scale=False# (not options['Auto Scale']) don't use : green shift
                    )
            bufHSV_CV32 = cv2.cvtColor(((bufpost16.astype(np.float32)) / 65536).astype(np.float32), cv2.COLOR_RGB2HSV)
        else:
            # get buffer from cache
            # bufpost16 = self.postProcessCache.copy()  TODO removed 20/07/18 unused validate
            bufHSV_CV32 = self.bufCache_HSV_CV32.copy()
        ###########
        # contrast and saturation correction (V channel).
        # We apply a (nearly) automatic histogram equalization
        # algorithm, well suited for multimodal histograms.
        ###########
        if adjustForm.contCorrection > 0:
            # warp should be in range 0..1.
            # warp = 0 means that no additional warping is done, but
            # the histogram is always stretched.
            warp = max(0, (adjustForm.contCorrection -1)) / 10
            bufHSV_CV32[:,:,2],a,b,d,T = warpHistogram(bufHSV_CV32[:, :, 2], valleyAperture=0.05, warp=warp, preserveHigh=options['Preserve Highlights'],
                                                       spline=None if self.autoSpline else self.getMmcSpline()) #preserveHigh=options['Preserve Highlights'])
            # show the spline
            if self.autoSpline and options['manualCurve']:
                self.getGraphicsForm().setContrastSpline(a, b, d, T)
                self.autoSpline = False  # mmcSpline = self.getGraphicsForm().scene().cubicItem # caution : misleading name for a quadratic s
        if adjustForm.satCorrection != 0:
            satCorr = adjustForm.satCorrection / 100   # range -0.5..0.5
            alpha = 1.0 / (0.501 + satCorr) - 1.0  # approx. map -0.5...0.0...0.5 --> +inf...1.0...0.0
            # tabulate x**alpha
            LUT = np.power(np.arange(256) / 255, alpha)
            # convert saturation s to s**alpha
            bufHSV_CV32[:, :, 1] = LUT[(bufHSV_CV32[:, :, 1] * 255).astype(int)]
        # back to RGB
        bufpost16 = (cv2.cvtColor(bufHSV_CV32, cv2.COLOR_HSV2RGB)*65535).astype(np.uint16)
        """
        ############
        # saturation (Lab)
        ############
        if False and adjustForm.satCorrection != 0:
            slope = max(0.1, adjustForm.satCorrection/25 +1) # range 0.1..3
            # multiply a and b channels
            buf32Lab[:,:,1:3] *= slope
            buf32Lab[:, :, 1:3] = np.clip(buf32Lab[:,:,1:3], -127, 127)
        """
        #############################
        # Conversion to 8 bits/channel
        #############################
        bufpost = (bufpost16.astype(np.float32)/256).astype(int).astype(np.uint8)

        if self.parentImage.useThumb:
            bufpost = cv2.resize(bufpost, (currentImage.width(), currentImage.height()))
        """
        ################################################################################
        # denoising
        # opencv fastNlMeansDenoisingColored is very slow (>30s for a 15 Mpx image).
        # Bilateral filtering is faster but not very accurate.
        ################################################################################
        noisecorr = adjustForm.noiseCorrection * currentImage.width() / self.width()
        if noisecorr > 0:
            noisecorr = noisecorr/100
            t = time()
            L = dwtDenoise_thr(buf32Lab[:, :, 0] / 100, thr=noisecorr, thrmode='wiener', level=8 if self.parentImage.useThumb else 11) * 100
            A = dwtDenoise_thr(buf32Lab[:, :, 1] / 127, thr=noisecorr*0.5, thrmode='wiener', level=8 if self.parentImage.useThumb else 11) * 127
            B = dwtDenoise_thr(buf32Lab[:, :, 2] / 127, thr=noisecorr*0.5, thrmode='wiener', level=8 if self.parentImage.useThumb else 11) * 127
            L = np.clip(L, 0, 100)
            # A = np.clip(A, -127, 127)
            # B = np.clip(B, -127, 127)
            print('denoise %.2f' % (time() - t))
            buf32Lab = np.dstack((L, A, B))
        """
        #bufRGB32 = cv2.cvtColor(buf32Lab, cv2.COLOR_Lab2RGB)
        #bufpost = (bufRGB32 * 255.0).astype(np.uint8)
        bufOut = QImageBuffer(currentImage)
        bufOut[:, :, :3][:, :, ::-1] = bufpost
        # base layer : no need to forward the alpha channel
        self.updatePixmap()

    def applyContrast(self, version='HSV'):
        """
        Applies contrast saturation and brightness corrections.
        If version is 'HSV' (default), the
        image is converted to HSV and the correction is applied to
        the S and V channels. Otherwise, the Lab color space is used.
        @param version:
        @type version:
        """
        adjustForm = self.view.widget()
        options = adjustForm.options
        contrastCorrection = adjustForm.contrastCorrection
        satCorrection = adjustForm.satCorrection
        brightnessCorrection = adjustForm.brightnessCorrection
        inputImage = self.inputImg()
        tmpBuf = QImageBuffer(inputImage)
        currentImage = self.getCurrentImage()
        ndImg1a = QImageBuffer(currentImage)
        # neutral point : forward changes
        if contrastCorrection == 0 and satCorrection == 0 and brightnessCorrection == 0:
            ndImg1a[:, :, :] = tmpBuf
            self.updatePixmap()
            return
        ##########################
        # Lab mode, slower than HSV
        ##########################
        if version=='Lab':
            # get l channel, range is 0..1
            LBuf = inputImage.getLabBuffer().copy()
            if brightnessCorrection != 0:
                alpha = (-adjustForm.brightnessCorrection + 1.0)
                # tabulate x**alpha
                LUT = np.power(np.arange(256) / 255, alpha)
                # convert L to L**alpha
                LBuf[:, :, 0] = LUT[LBuf[:, :, 0]]
            if contrastCorrection >0:
                # CLAHE
                if options['CLAHE']:
                    if self.parentImage.isHald:
                        raise ValueError('cannot build 3D LUT from CLAHE ')
                    clahe = cv2.createCLAHE(clipLimit=contrastCorrection, tileGridSize=(8, 8))
                    clahe.setClipLimit(contrastCorrection)
                    res = clahe.apply((LBuf[:,:,0] * 255.0).astype(np.uint8)) /255
                # warping
                else:
                    if self.parentImage.isHald and not options['manualCurve']:
                        raise ValueError('Check option Show Contrast Curve in Cont/Bright/Sat layer')
                    auto = self.autoSpline and not self.parentImage.isHald
                    res,a,b,d,T = warpHistogram(LBuf[:,:,0], warp=contrastCorrection, preserveHigh=options['High'],
                                                spline = None if auto else self.getMmcSpline())
                    # show the spline viewer
                    if self.autoSpline and options['manualCurve']:
                        self.getGraphicsForm().setContrastSpline(a, b, d, T)
                        self.autoSpline = False #mmcSpline = self.getGraphicsForm().scene().cubicItem # caution : misleading name for a quadratic spline !
                LBuf[:,:,0] = res
            if satCorrection != 0:
                slope = max(0.1, adjustForm.satCorrection / 25 + 1)  # ran# show the splinege 0.1..3
                # multiply a and b channels
                LBuf[:, :, 1:3] *= slope
                LBuf[:, :, 1:3] = np.clip(LBuf[:, :, 1:3], -127, 127)
            # back to RGB
            sRGBBuf = Lab2sRGBVec(LBuf)  # use opencv cvtColor
        ###########
        # HSV mode (default)
        ###########
        else:
            # get HSV buffer, H, S, V are in range 0..255
            HSVBuf = inputImage.getHSVBuffer().copy()
            if brightnessCorrection != 0:
                alpha = 1.0 / (0.501 + adjustForm.brightnessCorrection)  - 1.0  # approx. map -0.5...0.0...0.5 --> +inf...1.0...0.0
                # tabulate x**alpha
                LUT = np.power(np.arange(256) / 255, alpha) * 255
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
                        raise ValueError('Check option Show Contrast Curve in Cont/Bright/Sat layer')
                    buf32 = HSVBuf[:,:,2].astype(np.float)/255
                    auto = self.autoSpline and not self.parentImage.isHald
                    res,a,b,d,T = warpHistogram(buf32, warp=contrastCorrection, preserveHigh=options['High'],
                                                spline=None if auto else self.getMmcSpline())
                    res = (res*255.0).astype(np.uint8)
                    # show the spline viewer
                    if self.autoSpline and options['manualCurve']:
                        self.getGraphicsForm().setContrastSpline(a, b, d, T)
                        self.autoSpline = False # TODO added 18/07/18
                HSVBuf[:, :, 2] = res
            if satCorrection != 0:
                alpha = 1.0 / (0.501 + adjustForm.satCorrection) - 1.0  # approx. map -0.5...0.0...0.5 --> +inf...1.0...0.0
                # tabulate x**alpha
                LUT = np.power(np.arange(256) / 255, alpha) * 255
                # convert saturation s to s**alpha
                HSVBuf[:, :, 1] = LUT[HSVBuf[:, :, 1]]  # faster than take
            # back to RGB
            sRGBBuf = cv2.cvtColor(HSVBuf, cv2.COLOR_HSV2RGB)
        ndImg1a[:, :, :3][:,:,::-1] = sRGBBuf
        # forward the alpha channel
        ndImg1a[:, :,3] = tmpBuf[:,:,3]
        self.updatePixmap()

    def apply1DLUT(self, stackedLUT, options=None):
        """
        Applies 1D LUTS to RGB channels (one for each channel)
        @param stackedLUT: array of color values (in range 0..255) : a row for each RGB channel
        @type stackedLUT : ndarray, shape (3, 256) dtype int
        @param options: not used yet
        @type options : dictionary
        """
        # neutral point
        if options is None:
            options = {}
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast is working
            buf1 = QImageBuffer(self.inputImg())
            buf2 = QImageBuffer(self.getCurrentImage())
            buf2[:, :, :] = buf1
            self.updatePixmap()
            return
        inputImage = self.inputImg()
        currentImage = self.getCurrentImage()
        # get image buffers (BGR order on intel arch.)
        ndImg0a = QImageBuffer(inputImage)
        ndImg1a = QImageBuffer(currentImage)
        ndImg0 = ndImg0a[:,:,:3]
        ndImg1 = ndImg1a[:, :, :3]
        # apply LUTS to channels
        #rList = np.array([2,1,0]) #BGR
        s = ndImg0[:,:,0].shape
        for c in range(3): # 0.36s for 15Mpx
            ndImg1[:, :, c] = np.take(stackedLUT[2-c,:], ndImg0[:,:,c].reshape((-1,))).reshape(s)
        # ndImg1[:, :, :] = stackedLUT[rList, ndImg0]  # last dims of index arrays are equal : broadcast works. slower 0.66s for 15Mpx
        # forward the alpha channel
        ndImg1a[:,:,3] = ndImg0a[:,:,3]
        self.updatePixmap()

    def applyLab1DLUT(self, stackedLUT, options=None):
        """
        Applies 1D LUTS (one row for each L,a,b channel)
        @param stackedLUT: array of color values (in range 0..255). Shape must be (3, 255) : a row for each channel
        @type stackedLUT: ndarray shape=(3,256) dtype=int or float
        @param options: not used yet
        """
        # neutral point
        if options is None:
            options = {}
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast is working
            buf1 = QImageBuffer(self.inputImg())
            buf2 = QImageBuffer(self.getCurrentImage())
            buf2[:, :, :] = buf1
            self.updatePixmap()
            return
        from colorConv import Lab2sRGBVec
        # convert LUT to float to speed up  buffer conversions
        stackedLUT = stackedLUT.astype(np.float)
        # get the Lab input buffer
        Img0 = self.inputImg()
        ndLabImg0 = Img0.getLabBuffer() #.copy()
        # conversion functions
        def scaleLabBuf(buf):
            buf = buf + [0.0, 128.0, 128.0]  # copy is mandatory here to avoid the corruption of the cached Lab buffer
            buf[:,:,0] *= 255.0
            return buf
        def scaleBackLabBuf(buf):
            buf = buf - [0.0, 128.0, 128.0] # no copy needed here, but seems faster than in place operation!
            buf[:,:,0] /= 255.0
            return buf
        ndLImg0 = scaleLabBuf(ndLabImg0).astype(np.uint8)
        # apply LUTS to channels
        s = ndLImg0[:, :, 0].shape
        ndLabImg1 = np.zeros(ndLImg0.shape, dtype=np.uint8)
        for c in range(3):  # 0.43s for 15Mpx
            ndLabImg1[:, :, c] = np.take(stackedLUT[c, :], ndLImg0[:, :, c].reshape((-1,))).reshape(s)
        #ndLabImg1 = stackedLUT[rList, ndLImg0] # last dims are equal : broadcast works
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
        ndImg1[:,:,3] = ndImg0[:,:,3]
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
        # neutral point
        if options is None:
            options = {}
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast is working
            buf1 = QImageBuffer(self.inputImg())
            buf2=QImageBuffer(self.getCurrentImage())
            buf2[:,:,:] = buf1
            self.updatePixmap()
            return
        Img0 = self.inputImg()
        ndHSPBImg0 = Img0.getHspbBuffer()   # time 2s with cache disabled for 15 Mpx
        # apply LUTS to normalized channels (range 0..255)
        ndLImg0 = (ndHSPBImg0 * [255.0/360.0, 255.0, 255.0]).astype(np.uint8)
        #rList = np.array([0,1,2]) # H,S,B
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
        ndImg1a[:, :, :3][:,:,::-1] = ndRGBImg1
        # forward the alpha channel
        ndImg0 = QImageBuffer(Img0)
        ndImg1a[:,:,3] = ndImg0[:,:,3]
        # update
        self.updatePixmap()

    def applyHSV1DLUT(self, stackedLUT, options=None, pool=None):
        """
        Applies 1D LUTS to hue, sat and brightness channels.
        @param stackedLUT: array of color values (in range 0..255), a row for each channel
        @type stackedLUT : ndarray shape=(3,256) dtype=int or float
        @param options: not used yet
        @type options : dictionary
        @param pool: multiprocessing pool : unused
        @type pool: muliprocessing.Pool
        """
        # neutral point
        if options is None:
            options = {}
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast is working
            buf1 = QImageBuffer(self.inputImg())
            buf2=QImageBuffer(self.getCurrentImage())
            buf2[:,:,:] = buf1
            self.updatePixmap()
            return
        # convert LUT to float to speed up  buffer conversions
        stackedLUT = stackedLUT.astype(np.float)
        # get HSV buffer, range H: 0..180, S:0..255 V:0..255
        Img0 = self.inputImg()
        HSVImg0 = Img0.getHSVBuffer()
        #HSVImg0 = self.getMaskedCurrentContainer().getHSVBuffer()
        HSVImg0 = HSVImg0.astype(np.uint8)
        # apply LUTS
        HSVImg1 = np.zeros(HSVImg0.shape, dtype=np.uint8)
        #rList = np.array([0,1,2]) # H,S,V
        s = HSVImg0[:, :, 0].shape
        for c in range(3):  # 0.43s for 15Mpx
            HSVImg1[:, :, c] = np.take(stackedLUT[c, :], HSVImg0[:, :, c].reshape((-1,))).reshape(s)
        #HSVImg1 = stackedLUT[rList, HSVImg0]
        # back to sRGB
        RGBImg1 = hsv2rgbVec(HSVImg1, cvRange=True)
        # in place clipping
        np.clip(RGBImg1, 0, 255, out=RGBImg1)  # mandatory
        # set current image to modified image
        currentImage = self.getCurrentImage()
        ndImg1a = QImageBuffer(currentImage)
        ndImg1a[:, :, :3][:,:,::-1] = RGBImg1
        # forward the alpha channel
        ndImg0 = QImageBuffer(Img0)
        ndImg1a[:,:,3] = ndImg0[:,:,3]
        # update
        self.updatePixmap()

    def apply3DLUT(self, LUT, options=None, pool=None):
        """
        Applies a 3D LUT to the current view of the image (self or self.thumb or self.hald).
        If pool is not None and the size of the current view is > 3000000, the computation is
        done in parallel on image slices.
        The orders of LUT axes, LUT channels and image channels must be BGR
        @param LUT: LUT3D array (cf. colorCube.py)
        @type LUT: 3d ndarray, dtype = int
        @param options:
        @type options: dict of string:boolean pairs
        """
        # get buffers
        if options is None:
            options = {}
        inputImage = self.inputImg()
        currentImage = self.getCurrentImage()
        # get selection
        w1, w2, h1, h2 = (0.0,) * 4
        # use selection
        if options.get('use selection', False):
            w, wF = self.getCurrentImage().width(), self.width()
            h, hF = self.getCurrentImage().height(), self.height()
            wRatio, hRatio = float(w) / wF, float(h) / hF
            if self.rect is not None:
                w1, w2, h1, h2 = int(self.rect.left() * wRatio), int(self.rect.right() * wRatio), int(self.rect.top() * hRatio), int(self.rect.bottom() * hRatio)
            w1 , h1 = max(w1,0), max(h1, 0)
            w2, h2 = min(w2, inputImage.width()), min(h2, inputImage.height())
            if w1>=w2 or h1>=h2:
                dlgWarn("Empty selection\nSelect a region with the marquee tool")
                return
        # use image
        else:
            w1, w2, h1, h2 = 0, self.inputImg().width(), 0, self.inputImg().height()
        inputBuffer = QImageBuffer(inputImage)[h1:h2 + 1, w1:w2 + 1, :]
        imgBuffer = QImageBuffer(currentImage)[:, :, :]
        ndImg0 = inputBuffer[:, :, :3]
        ndImg1 = imgBuffer[:, :, :3]
        # choose the right interpolation method
        if (pool is not None) and (inputImage.width() * inputImage.height() > 3000000):
            interp = interpMulti
        else:
            interp = interpTetraVec_ if USE_TETRA else interpVec_
        # apply LUT
        ndImg1[h1:h2 + 1, w1:w2 + 1, :] = interp(LUT, ndImg0, pool=pool)
        # forward the alpha channel
        imgBuffer[h1:h2 + 1, w1:w2 + 1, 3] = inputBuffer[:,:,3]
        self.updatePixmap()

    """
    def applyHald(self, hald, pool=None):
        Convert a hald image to a 3DLUT object and applies
        the 3D LUT to the current view, using a pool of parallel processes if
        pool is not None.
        @param hald: hald image
        @type hald: QImage
        @param pool: pool of parallel processes, default None
        @type pool: multiprocessing.pool
        lut = LUT3D.HaldImage2LUT3D(hald)
        self.apply3DLUT(lut.LUT3DArray, options={'use selection' : False}, pool=pool)
    """

    def histogram(self, size=QSize(200, 200), bgColor=Qt.white, range =(0,255), chans=channelValues.RGB, chanColors=Qt.gray, mode='RGB', addMode=''):
        """
        Plots histogram with the
        specified color mode and channels.
        Luminosity is  Y = 0.299*R + 0.587*G + 0.114*B (YCrCb opencv color space).
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
        @param mode: color mode ((one of 'RGB', 'HSpB', 'Lab')
        @type mode: str
        @return: histogram plot
        @rtype: QImage
        """
        # convert size to QSize
        if type(size) is int:
            size = QSize(size, size)
        # scaling factor for the bin edges
        spread = float(range[1] - range[0])
        scale = size.width() / spread
        # per channel histogram function
        #def drawChannelHistogram(painter, channel, buf, color):
        def drawChannelHistogram(painter, hist, bin_edges, color):
            #Draw the (smoothed) histogram for a single channel.
            #param painter: QPainter
            #param hist: histogram to draw
            #param channel: channel index (BGRA (intel) or ARGB )
            # smooth the histogram (first and last bins excepted) for a better visualization of clipping.
            hist = np.concatenate(([hist[0]], SavitzkyGolay.filter(hist[1:-1]), [hist[-1]]))
            M = max(hist[1:-1])  # TODO added 04/10/18 + removed parameter M: validate
            # draw histogram
            imgH = size.height()
            lg = len(hist)
            for i, y in enumerate(hist):
                h = int(imgH * y / M)
                h = min(h, imgH - 1) # TODO added 04/10/18 height of rect must be < height of img, otherwise fillRect does nothing
                rect = QRect(int((bin_edges[i] - range[0]) * scale), max(img.height() - h, 0), int((bin_edges[i + 1] - bin_edges[i]) * scale+1), h)
                painter.fillRect(rect, color)
                # clipping indicators
                if i == 0 or i == len(hist)-1:
                    left = bin_edges[0 if i == 0 else -1]
                    if 0 < left < 255:
                        continue
                    left =  left - (10 if i > 0 else 0)
                    clipping_threshold = 0.02
                    percent = hist[i] * (bin_edges[i+1]-bin_edges[i])
                    if percent > clipping_threshold:
                        # calculate the color of the indicator according to percent value
                        nonlocal gPercent
                        gPercent = min(gPercent, np.clip((0.05 - percent) / 0.03, 0, 1))
                        painter.fillRect(left, 0, 10, 10, QColor(255, 255*gPercent, 0))
        # green percent for clipping indicators
        gPercent = 1.0
        bufL = cv2.cvtColor(QImageBuffer(self)[:, :, :3], cv2.COLOR_BGR2GRAY)[..., np.newaxis]  # returns Y (YCrCb) : Y = 0.299*R + 0.587*G + 0.114*B
        if mode == 'RGB':
            buf = QImageBuffer(self)[:,:,:3][:,:,::-1]  #RGB
        elif mode == 'HSpB':
            buf = self.getHspbBuffer()
        elif mode == 'Lab':
            buf = self.getLabBuffer()
        elif mode =='Luminosity':
            chans = []
        img = QImage(size.width(), size.height(), QImage.Format_ARGB32)
        img.fill(bgColor)
        qp = QPainter(img)
        if type(chanColors) is QColor or type(chanColors) is Qt.GlobalColor:
            chanColors = [chanColors]*3
        # compute histograms
        # bins='auto' sometimes causes a huge number of bins ( >= 10**9) and memory error
        # even for small data size (<=250000), so we don't use it.
        # This is a numpy bug : in the module function_base.py
        # a reasonable upper bound for bins should be chosen to prevent memory error.
        if mode=='Luminosity' or addMode=='Luminosity':
            hist, bin_edges = np.histogram(bufL, bins=100, density=True)
            drawChannelHistogram(qp, hist, bin_edges, Qt.gray)
        hist_L, bin_edges_L = [0]*len(chans), [0]*len(chans)
        for i,ch in enumerate(chans):
            buf0 = buf[:, :, ch]
            hist_L[i], bin_edges_L[i] = np.histogram(buf0, bins=100, density=True)
            # to prevent artifacts, the histogram bins must be drawn
            # using the composition mode source_over. So, we use
            # a fresh QImage for each channel.
            tmpimg = QImage(size, QImage.Format_ARGB32)
            tmpimg.fill(bgColor)
            tmpqp = QPainter(tmpimg)
            drawChannelHistogram(tmpqp, hist_L[i], bin_edges_L[i], chanColors[ch])
            tmpqp.end()
            # add the channnel hist to img
            qp.drawImage(QPoint(0,0), tmpimg)
            # subsequent images are added with composition mode Plus
            qp.setCompositionMode(QPainter.CompositionMode_Plus)
        qp.end()
        buf = QImageBuffer(img)
        # if len(chans) > 1, clip gray area to improve the aspect of the histogram
        if len(chans) > 1 :
            buf[:,:,:3] = np.where(np.min(buf, axis=-1)[:,:,np.newaxis]>=100, np.array((100,100,100))[np.newaxis, np.newaxis,:], buf[:,:,:3] )
        return img

    def applyFilter2D(self):
        """
        Apply 2D kernel.
        """
        adjustForm = self.view.widget()
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
        if self.rect is not None:
            rect = self.rect
            ROI0 = buf0[int(rect.top() * r): int(rect.bottom() * r), int(rect.left() * r): int(rect.right() * r),:3]
            # reset output image
            buf1[:,:,:] = buf0
            ROI1 = buf1[int(rect.top() * r): int(rect.bottom() * r), int(rect.left() * r): int(rect.right() * r),:3]
        else:
            ROI0 = buf0[:,:,:3]
            ROI1 = buf1[:,:,:3]
        if adjustForm.kernelCategory in [filterIndex.IDENTITY, filterIndex.UNSHARP, filterIndex.SHARPEN, filterIndex.BLUR1, filterIndex.BLUR2]:
            # kernel filter
            ROI1[:,:,:] = cv2.filter2D(ROI0, -1, adjustForm.kernel)
        else:
            # bilateral filter
            radius = adjustForm.radius
            if self.parentImage.useThumb:
                radius = int(radius * r)
            ROI1[:,:,::-1] = cv2.bilateralFilter( ROI0[:,:,::-1], radius, 10*adjustForm.tone, 50 if self.parentImage.useThumb else 200)
        # forward the alpha channel
        buf1[:,:,3] = buf0[:,:,3]
        self.updatePixmap()

    def applyBlendFilter(self):
        """
        Apply a gradual neutral density filter
        """
        adjustForm = self.view.widget()
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
            if adjustForm.kernelCategory ==  blendFilterIndex.GRADUALNONE:
                start, end = 0 , h -1
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
            buf1[:,:,:3][:,:,::-1] = (bufRGB32 * 255.0).astype(np.uint8)
        # forward the alpha channel
        buf1[:,:,3] = buf0[:,:,3]
        self.updatePixmap()

    def applyTemperature(self):
        """
        The method implements two algorithms for the correction of color temperature.
        - Chromatic adaptation : multipliers in linear sRGB.
        - Photo filter : Blending using mode multiply, plus correction of luminosity
        """
        adjustForm = self.view.widget()
        options = adjustForm.options
        temperature = adjustForm.tempCorrection
        tint = adjustForm.tintCorrection # range -1..1
        inputImage = self.inputImg()
        buf1 = QImageBuffer(inputImage)
        currentImage = self.getCurrentImage()
        # neutral point : forward input image and return
        if abs(temperature - 6500) < 200 and tint == 0:
            buf0 = QImageBuffer(currentImage)
            buf0[:, :, :] = buf1
            self.updatePixmap()
            return
        from blend import blendLuminosity
        from colorConv import bbTemperature2RGB, conversionMatrix, rgb2rgbLinearVec, rgbLinear2rgbVec
        ################
        # photo filter
        ################
        if options['Photo Filter']:
            # get black body color
            r, g, b = bbTemperature2RGB(temperature)
            filter = QImage(inputImage.size(), inputImage.format())
            filter.fill(QColor(r, g, b, 255))
            # draw image on filter using mode multiply
            qp = QPainter(filter)
            qp.setCompositionMode(QPainter.CompositionMode_Multiply)
            qp.drawImage(0, 0, inputImage)
            qp.end()
            # correct the luminosity of the resulting image,
            # by blending it with the inputImage, using mode luminosity.
            # We use a tuning coeff to control the amount of correction.
            # Note that using perceptual brightness gives better results, unfortunately slower
            resImg = blendLuminosity(filter, inputImage)
            bufOutRGB = QImageBuffer(resImg)[:,:,:3][:,:,::-1]
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
            m1, m2, m3, _ = temperatureAndTint2RGBMultipliers(temperature, 2**tint, sRGB_lin2XYZInverse)
            buf = QImageBuffer(inputImage)[:, :, :3]
            bufXYZ = sRGB2XYZVec(buf[:, :, ::-1])
            bufsRGBLinear = np.tensordot(bufXYZ, sRGB_lin2XYZInverse, axes=(-1, -1))
            # apply multipliers
            bufsRGBLinear *= [m1, m2, m3]
            # brightness correction
            M = np.max(bufsRGBLinear)
            bufsRGBLinear /= M
            bufOutRGB = rgbLinear2rgbVec(bufsRGBLinear)
            np.clip(bufOutRGB, 0, 255, out=bufOutRGB)
            bufOutRGB = bufOutRGB.astype(np.uint8)
        # set output image
        bufOut0 = QImageBuffer(currentImage)
        bufOut = bufOut0[:,:,:3]
        bufOut[:, :, ::-1] = bufOutRGB
        # forward the alpha channel
        bufOut0[:,:,3] = buf1[:,:,3]
        self.updatePixmap()

class mImage(vImage):
    """
    Multi-layer image : base class for editable images.
    A mImage holds a presentation layer
    for color management and a stack containing at least a
    background layer. All layers share the same metadata instance.
    To correctly render a mImage, widgets should override their
    paint event handler.
    """
    @classmethod
    def restoreMeta(cls, srcFile, destFile, defaultorientation=True, thumbfile=None):
        """
        # copy metadata from sidecar to image file. The sidecar is not removed.
        If defaultorientaion is True the orientaion of the destination file is
        set to "no change" (1). This way, saved images are shown as they were edited.
        @param srcFile: source image or sidecar (the extension is replaced by .mie).
        @type srcFile: str
        @param destFile: image file
        @type destFile: str
        @param defaultorientation
        @type defaultorientation: bool
        @param thumbnail: thumbnail data
        @type thumbnail: bytearray
        """
        with exiftool.ExifTool() as e:
            e.copySidecar(srcFile, destFile)
            if defaultorientation:
                e.writeOrientation(destFile, '1')
            if thumbfile is not None:
                    e.writeThumbnail(destFile, thumbfile)
    def __init__(self, *args, **kwargs):
        # as updatePixmap uses layersStack, it must be initialized
        # before the call to super(). __init__
        self.layersStack = []
        # link back to QLayerView window
        self.layerView = None
        super().__init__(*args, **kwargs)  # must be done before prLayer init.
        # add background layer
        bgLayer = QLayer.fromImage(self, parentImage=self)
        bgLayer.isClipping = True
        self.setModified(False)
        self.activeLayerIndex = None
        self.addLayer(bgLayer, name='Background')
        # color management : we assume that the image profile is the working profile
        self.colorTransformation = icc.workToMonTransform
        # presentation layer
        prLayer = QPresentationLayer(QImg=self, parentImage=self)
        prLayer.name = 'presentation'
        prLayer.role ='presentation'
        prLayer.execute = lambda l=prLayer, pool=None: prLayer.applyNone()
        prLayer.updatePixmap() # mandatory here as vImage init. can't do it
        self.prLayer = prLayer
        self.isModified = False
        # rawpy object
        self.rawImage = None

    def bTransformed(self, transformation):
        """
        Applies transformation to all layers in stack and returns
        the new mImage
        @param transformation:
        @type transformation: Qtransform
        @return:
        @rtype: mimage
        """
        img = mImage(QImg=self.transformed(transformation))
        img.meta = self.meta
        img.onImageChanged = self.onImageChanged
        img.useThumb = self.useThumb
        img.useHald = self.useHald
        stack = []
        for layer in self.layersStack:
            tLayer = layer.bTransformed(transformation, img)
            stack.append(tLayer)
        img.layersStack = stack
        gc.collect()
        return img

    def getActiveLayer(self):
        """
        Returns the currently active layer.
        @return: The active layer
        @rtype: QLayer
        """
        if self.activeLayerIndex is not None:
            return self.layersStack[self.activeLayerIndex]
        else:
            return None

    def setActiveLayer(self, stackIndex):
        """
        Assigns stackIndex value to  activeLayerIndex and
        updates the layer view and tools
        @param stackIndex: index in stack for the layer to select
        @type stackIndex: int
        @return: active layer
        @rtype: QLayer
        """
        lgStack = len(self.layersStack)
        if stackIndex < 0 or stackIndex >= lgStack:
            return
        # clean old active layer
        active = self.getActiveLayer()
        if active is not None:
            if active.tool is not None:
                active.tool.hideTool()
        # set new active layer
        self.activeLayerIndex = stackIndex
        if self.layerView is not None:
            self.layerView.selectRow(lgStack - 1 - stackIndex)
        active = self.getActiveLayer()
        if active.tool is not None and active.visible:
            active.tool.showTool()
        return active

    def getActivePixel(self,x, y, fromInputImg=True):
        """
        Reads RGB colors of the pixel at (x, y) from the active layer.
        If fromInputImg is True (default), pixel is taken from
        the input image, otherwise from the current image.
        Coordinates are relative to the full sized image.
        @param x, y: coordinates of pixel, relative to the full-sized image
        @return: pixel RGB colors
        @rtype: 3-uple of int
        """
        x, y = self.full2CurrentXY(x, y)
        activeLayer = self.getActiveLayer()
        # TODO modified 2/10/18 validate
        qclr = activeLayer.inputImg().pixelColor(x, y) if fromInputImg else activeLayer.getCurrentImage().pixelColor(x, y)
        return qColorToRGB(qclr)
        """
        if activeLayer.isAdjustLayer() :
            # layer is adjustment or segmentation : read from input image
            return activeLayer.inputImg().pixelColor(x, y)
        else:
            # read from current image
            return activeLayer.getCurrentImage().pixelColor(x, y)
        """

    def cacheInvalidate(self):
        """
        Invalidate cache buffers. The method is
        called by applyToStack for each layer after layer.execute
        """
        vImage.cacheInvalidate(self)
        for layer in self.layersStack:
            layer.cacheInvalidate()  # As Qlayer doesn't inherit from mImage, we call vImage.cacheInvalidate(layer)

    def setThumbMode(self, value):
        if value == self.useThumb:
            return
        self.useThumb = value
        # recalculate the whole stack
        self.layerStack[0].apply()

    def updatePixmap(self):
        """
        Updates all pixmaps
        Overrides vImage.updatePixmap()
        """
        vImage.updatePixmap(self)
        for layer in self.layersStack:
            vImage.updatePixmap(layer)
        self.prLayer.updatePixmap()

    def getStackIndex(self, layer):
        p = id(layer)
        i = -1
        for i,l in enumerate(self.layersStack):
            if id(l) == p:
                break
        return i

    def addLayer(self, layer, name='', index=None):
        """
        Adds a layer.

        @param layer: layer to add (fresh layer if None, type QLayer)
        @type layer: QLayer
        @param name:
        @type name: str
        @param index: index of insertion in layersStack (top of active layer if index=None)
        @type index: int
        @return: the layer added
        @rtype: QLayer
        """
        # build a unique name
        usedNames = [l.name for l in self.layersStack]
        a = 1
        trialname = name if len(name) > 0 else 'noname'
        while trialname in usedNames:
            trialname = name + '_'+ str(a)
            a = a+1
        if layer is None:
            layer = QLayer(QImg=self, parentImage=self) # TODO 11/09/18 added parentImage validate
            layer.fill(Qt.white)
        layer.name = trialname
        #layer.parentImage = self # TODO 07/06/18 validate suppression
        if index is None:
            if self.activeLayerIndex is not None:
                # add on top of active layer if any
                index = self.activeLayerIndex + 1 #TODO +1 added 03/05/18 validate
            else:
                # empty stack
                index = 0
        self.layersStack.insert(index, layer)
        self.setActiveLayer(index)
        layer.meta = self.meta
        layer.parentImage = self
        self.setModified(True)
        return layer

    def removeLayer(self, index=None):
        if index is None:
            return
        self.layersStack.pop(index)

    def addAdjustmentLayer(self, name='', role='',  index=None, sourceImg=None):
        """
        Adds an adjustment layer to the layer stack, at
        position index (default is top of active layer)
        @param name:
        @type name:
        @param role:
        @type role:
        @param index:
        @type index:
        @param sourceImg:
        @type sourceImg:
        @return:
        @rtype:
        """
        if index is None:
            # adding on top of active layer
            index = self.activeLayerIndex
        if sourceImg is None:
            # set image from active layer
            layer = QLayer.fromImage(self.layersStack[index], parentImage=self)
        else:
            # image layer :
            layer = QLayerImage.fromImage(self.layersStack[index], parentImage=self, sourceImg=sourceImg)
        layer.role = role
        self.addLayer(layer, name=name, index=index + 1)
        # add autoSpline attribute to contrast layer only
        if role in ['CONTRAST', 'RAW']:
            layer.autoSpline = True
        # init thumb
        if layer.parentImage.useThumb:
            layer.thumb = layer.inputImg().copy()
        group = self.layersStack[index].group
        if group:
            layer.group = group
            layer.mask = self.layersStack[index].mask
            layer.maskIsEnabled = True
        # sync caches
        layer.updatePixmap()
        return layer

    def addSegmentationLayer(self, name='', index=None):
        if index is None:
            index = self.activeLayerIndex
        layer = QLayer.fromImage(self.layersStack[index], parentImage=self)
        layer.role = 'SEGMENT'
        layer.inputImg = lambda: self.layersStack[layer.getLowerVisibleStackIndex()].getCurrentMaskedImage() # TODO 13/06/18 is it different from other layers?
        self.addLayer(layer, name=name, index=index + 1)
        layer.maskIsEnabled = True
        layer.maskIsSelected = True
        # mask pixels are not yet painted as FG or BG
        # so we mark them as invalid
        layer.mask.fill(vImage.defaultColor_Invalid)
        layer.paintedMask = layer.mask.copy()
        # add noSegment flag. It blocks/allows the execution of grabcut algorithm
        # in applyGrabcut : if True, further stack updates
        # do not redo the segmentation. The flag is toggled by the Apply Button
        # slot of segmentForm.
        layer.noSegment = False
        layer.updatePixmap()
        return layer

    def dupLayer(self, index=None):
        """
        inserts in layersStack at position index+1 a copy of the layer 
        at position index. If index is None (default value), the layer is inserted
        on top of the stack. Adjustment layers are not duplicated.
        @param index: 
        @type index: int
        @return: 
        """
        if index is None:
            index = len(self.layersStack) - 1
        layer0 = self.layersStack[index]
        if layer0.isAdjustLayer():
            return
        layer1 = QLayer.fromImage(layer0, parentImage=self)
        self.addLayer(layer1, name=layer0.name, index=index+1)

    def mergeVisibleLayers(self):
        """
        Merges the visible masked images and returns the
        resulting QImage, eventually scaled to fit the image size.
        @return: image
        @rtype: QImage
        """
        # init a new image
        img = QImage(self.width(), self.height(), self.format())
        # Image may contain transparent pixels, hence we
        # fill the image with a background color.
        img.fill(vImage.defaultBgColor)
        # draw layers with (eventually) masked areas.
        qp = QPainter(img)
        qp.drawImage(QRect(0, 0, self.width(), self.height()), self.layersStack[-1].getCurrentMaskedImage())
        qp.end()
        return img

    def save(self, filename, quality=-1, compression=-1, crops=None):
        """
        Builds image from visible layers
        and writes it to file. Raise IOError if
        saving fails. Overrides QImage.save().
        A thumbnail of standard size 160x120 or 120x160 is returned.
        @param filename:
        @type filename: str
        @param quality: integer value in range 0..100, or -1
        @type quality: int
        @return: thumbnail of the saved image
        @rtype: QImage
        """
        # don't save thumbnails
        if self.useThumb:
            return None
        # get the final image from the presentation layer.
        # It is important to note that this image is
        # NOT color managed (prLayer.qPixmap only
        # is color managed)
        img = self.prLayer.getCurrentImage() # self.mergeVisibleLayers()
        # imagewriter and QImage.save are unusable for tif files,
        # due to bugs in libtiff, hence
        # we use opencv imwrite.
        fileFormat = filename[-3:].upper()
        buf = QImageBuffer(img)
        if fileFormat == 'JPG':
            buf = buf[:, :, :3] # TODO added 19/06/18
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]  # quality range 0..100
        elif fileFormat == 'PNG':
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]  # compression range 0..9
        elif fileFormat == 'TIF':
            buf = buf[:, :, :3]  # TODO added 19/06/18
            params = []
        else:
            raise IOError("Invalid File Format\nValid formats are jpg, png, tif ")
        if self.isCropped:
            # make slices
            w, h = self.width(), self.height()
            w1, w2 = int(self.cropLeft), w - int(self.cropRight)
            h1, h2 = int(self.cropTop), h - int(self.cropBottom)
            buf = buf[h1:h2, w1:w2,:]
        # build thumbnail from (evenyually) cropped image
        # choose thumb size
        wf, hf = buf.shape[1], buf.shape[0]
        if wf > hf:
            wt, ht = 160, 120
        else:
            wt, ht = 120, 160
        thumb = ndarrayToQImage(np.ascontiguousarray(buf[:,:,:3][:,:,::-1]), format=QImage.Format_RGB888).scaled(wt,ht, Qt.KeepAspectRatio)
        #wr, hr = thumb.width(), thumb.height()
        #thumb = thumb.copy(QRect((wr-wt)//2,(hr-ht)//2, wt, ht))
        written = cv2.imwrite(filename, buf, params)  #BGR order
        if not written:
            raise IOError("Cannot write file %s " % filename)
        # self.setModified(False) # cannot be reset if the image is modified again
        return thumb

    def writeStackToStream(self, dataStream):
        dataStream.writeInt32(len(self.layersStack))
        for layer in self.layersStack:
            """
            dataStream.writeQString('menuLayer(None, "%s")' % layer.actionName)
            dataStream.writeQString('if "%s" != "actionNull":\n dataStream=window.label.img.layersStack[-1].readFromStream(dataStream)' % layer.actionName)
            """
            dataStream.writeQString(layer.actionName)
        for layer in self.layersStack:
            if getattr(layer, 'view', None) is not None:
                layer.view.widget().writeToStream(dataStream)

    def readStackFromStream(self, dataStream):
        # stack length
        count = dataStream.readInt32()
        script = []
        for i in range(count):
            actionName = dataStream.readQString()
            script.append('menuLayer(None, "%s")' % actionName)
            script.append('if "%s" != "actionNull":\n dataStream=window.label.img.layersStack[-1].readFromStream(dataStream)' % actionName)
        return script

    def saveStackToFile(self, filename):
        qf = QFile(filename)
        if not qf.open(QIODevice.WriteOnly):
            raise IOError('cannot open file %s' % filename)
        dataStream = QDataStream(qf)
        self.writeStackToStream(dataStream)
        qf.close()

    def loadStackFromFile(self, filename):
        qf = QFile(filename)
        if not qf.open(QIODevice.ReadOnly):
            raise IOError('cannot open file %s' % filename)
        dataStream = QDataStream(qf)
        script = self.readStackFromStream(dataStream)
        #qf.close()
        return script, qf, dataStream

class imImage(mImage) :
    """
    Zoomable and draggable multi-layer image :
    this is the base class for bLUe documents
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Zoom coeff :
        # Zoom_coeff = 1.0 displays an image fitting the
        # size of the current window ( NOT the actual pixels of the image).
        self.Zoom_coeff = 1.0
        self.xOffset, self.yOffset = 0, 0
        self.isMouseSelectable =True
        self.isModified = False

    def bTransformed(self, transformation):
        """
        Applies transformation to all layers in stack
        and returns the new imImage.
        @param transformation:
        @type transformation: QTransform
        @return:
        @rtype: imImage
        """
        img = imImage(QImg=self.transformed(transformation))
        img.meta = self.meta
        img.onImageChanged = self.onImageChanged
        img.useThumb = self.useThumb
        img.useHald = self.useHald
        stack = []
        # apply transformation to the stack. Note that
        # the presentation layer is automatically rebuilt
        for layer in self.layersStack:
            tLayer = layer.bTransformed(transformation, img)
            # keep ref to original layer (needed by execute)
            tLayer.parentLayer = layer.parentLayer
            # record new transformed layer in original layer (needed by execute)
            tLayer.parentLayer.tLayer = tLayer
            # link back grWindow to tLayer
            # using weak ref for back links
            if tLayer.view is not None:
                # for historical reasons, graphic forms inheriting
                # from QGraphicsView use form.scene().layer attribute,
                # others use form.layer
                if getattr(tLayer.view.widget(), 'scene', None) is None:
                    tLayer.view.widget().layer = weakref.proxy(tLayer)
                else:
                    tLayer.view.widget().scene().layer = tLayer.view.widget().layer = weakref.proxy(tLayer)
            stack.append(tLayer)
        img.layersStack = stack
        gc.collect()
        return img

    def resize(self, pixels, interpolation=cv2.INTER_CUBIC):
        """
        Resize image and layers
        @param pixels:
        @param interpolation:
        @return: resized imImage object
        @rtype: imImage
        """
        # resized vImage
        rszd0 = super().resize(pixels, interpolation=interpolation)
        # resized imImage
        rszd = imImage(QImg=rszd0,meta=copy(self.meta))
        rszd.rect = rszd0.rect
        for k, l  in enumerate(self.layersStack):
            if l.name != "background" and l.name != 'drawlayer':
                img = QLayer.fromImage(l.resize(pixels, interpolation=interpolation), parentImage=rszd)
                rszd.layersStack.append(img)
                #rszd._layers[l.name] = img
        self.isModified = True
        return rszd

    def view(self):
        return self.Zoom_coeff, self.xOffset, self.yOffset

    def setView(self, zoom=1.0, xOffset=0.0, yOffset=0.0):
        """
        Sets viewing conditions: zoom, offset
        @param zoom: zoom coefficient
        @type zoom: float
        @param xOffset: x-offset
        @type xOffset: int
        @param yOffset: y-offset
        @type yOffset: int
        @return: 
        """
        self.Zoom_coeff, self.xOffset, self.yOffset = zoom, xOffset, yOffset

    def fit_window(self, win):
        """
        reset Zoom_coeff and offset
        @param win: 
        @type win:
        """
        self.Zoom_coeff = 1.0
        self.xOffset, self.yOffset = 0.0, 0.0

class QLayer(vImage):
    """
    Base class for image layers
    """
    @classmethod
    def fromImage(cls, mImg, parentImage=None):
        """
        Returns a QLayer object initialized with mImg.
        @param mImg:
        @type mImg: QImage
        @param parentImage:
        @type parentImage: mImage
        @return:
        @rtype: Qlayer
        """
        layer = QLayer(QImg=mImg, parentImage=parentImage)
        layer.parentImage = parentImage
        return layer

    def __init__(self, *args, **kwargs):
        ############################################################
        # Signals
        # Making QLayer inherit from QObject leads to
        # a buggy behavior of hasattr and getattr.
        # So, we don't add signals as first level class attributes.
        # Instead, we use instances of ad hoc signal containers (cf. utils.py)

        self.visibilityChanged = baseSignal_bool()
        self.colorPicked = baseSignal_Int2()

        ###########################################################
        self.parentImage = kwargs.get('parentImage', None)  #TODO added 11/09/18 validate
        # when a geometric transformation is applied to the whole image
        # each layer must be replaced with a transformed layer, recorded in tLayer
        # and tLayer.parentLayer keeps a reference to the original layer.
        self.tLayer = self
        self.parentLayer = self
        self.modified = False
        self.name='noname'
        self.visible = True
        # add signal instance (layer visibility change,..)
        #self.signals = baseSignals()
        self.isClipping = False
        self.role = kwargs.pop('role', '')
        self.tool = None
        self.parentImage = kwargs.pop('parentImage', None)
        # layer opacity is used by QPainter operations.
        # Its value must be in the range 0.0...1.0
        self.opacity = 1.0
        self.compositionMode = QPainter.CompositionMode_SourceOver
        # The next two attributes are used by adjustment layers only.
        self.execute = lambda l=None, pool=None: l.updatePixmap() if l is not None else None
        self.options = {}
        # actionName is used by methods graphics***.writeToStream()
        self.actionName = 'actionNull'
        # associated form : view is set by bLUe.menuLayer()
        self.view = None
        # containers are initialized (only once) by
        # getCurrentMaskedImage. Their type is QLayer
        self.maskedImageContainer = None
        self.maskedThumbContainer = None
        # consecutive layers can be grouped.
        # A group is a list of QLayer objects
        self.group = []
        # layer offsets
        self.xOffset, self.yOffset = 0, 0
        self.Zoom_coeff = 1.0
        # clone dup layer shift and zoom  relative to current layer
        self.xAltOffset, self.yAltOffset = 0, 0
        self.AltZoom_coeff = 1.0
        super().__init__(*args, **kwargs)

    def getGraphicsForm(self):
        """
        Returns the graphics form associated to the layer
        @return:
        @rtype: QWidget
        """
        if self.view is not None:  #TODO added 2/10/18 validate
            return self.view.widget()
        return None

    def isActiveLayer(self):
        if self.parentImage.getActiveLayer() is self:
            return True
        return False

    def getMmcSpline(self):
        """
        Returns the spline used for multimode contrast
        correction if it is initialized and None otherwise.
        @return:
        @rtype: activeSpline
        """
        # get layer graphic form
        grf = self.getGraphicsForm()
        # manual curve form
        if grf.contrastForm is not None:
            return grf.contrastForm.scene().cubicItem
        return None

    def addTool(self, tool):
        """
        Adds tool to layer
        @param tool:
        @type tool: rotatingTool
        """
        self.tool = tool
        tool.modified = False
        tool.layer = self
        try:
            tool.layer.visibilityChanged.sig.disconnect()  # TODO signals removed 30/09/18 validate
        except:
            pass
        tool.layer.visibilityChanged.sig.connect(tool.setVisible) # TODO signals removed 30/09/18 validate
        tool.img = self.parentImage
        w, h = tool.img.width(), tool.img.height()
        for role, pos in zip(['topLeft', 'topRight', 'bottomRight', 'bottomLeft'],
                             [QPoint(0, 0), QPoint(w, 0), QPoint(w, h), QPoint(0, h)]):
            tool.btnDict[role].posRelImg = pos
            tool.btnDict[role].posRelImg_ori = pos
            tool.btnDict[role].posRelImg_frozen = pos
        tool.moveRotatingTool()

    def setVisible(self, value):
        """
        Sets self.visible to value and emit visibilityChanged.sig
        @param value:
        @type value: bool
        """
        self.visible = value
        self.visibilityChanged.sig.emit(value)  # TODO signals removed 30/09/18 validate

    def bTransformed(self, transformation, parentImage):
        """
        Applies transformation to a copy of layer and returns the copy.
        @param transformation:
        @type transformation: QTransform
        @param parentImage:
        @type parentImage: vImage
        @return: transformed layer
        @rtype: QLayer
        """
        # init a new layer from transformed image :
        # all static attributes (caches...) are reset to default, but thumb
        tLayer = QLayer.fromImage(self.transformed(transformation), parentImage=parentImage)
        # copy  dynamic attributes from old layer
        for a in self.__dict__.keys():
            if a not in tLayer.__dict__.keys():
                tLayer.__dict__[a] = self.__dict__[a]
        tLayer.name = self.name
        tLayer.actionName = self.actionName
        tLayer.view = self.view
        tLayer.visible = self.visible # TODO added 25/09/18 validate
        # cut link from old layer to graphic form
        # self.view = None                        # TODO 04/12/17 validate
        tLayer.execute = self.execute
        tLayer.mask = self.mask.transformed(transformation)
        tLayer.maskIsEnabled, tLayer.maskIsSelected = self.maskIsEnabled, self.maskIsSelected
        return tLayer

    def initThumb(self):
        """
        Override vImage.initThumb, to set the parentImage attribute
        """
        super().initThumb()
        self.thumb.parentImage = self.parentImage

    def initHald(self):
        """
        Build a hald image (as a QImage) from identity 3D LUT.
        """
        if not self.cachesEnabled:
            return
        s = int((LUT3DIdentity.size) ** (3.0 / 2.0)) + 1
        buf0 = LUT3DIdentity.getHaldImage(s, s)
        #self.hald = QLayer(QImg=QImage(QSize(190,190), QImage.Format_ARGB32))
        self.hald = QImage(QSize(s, s), QImage.Format_ARGB32)
        buf1 = QImageBuffer(self.hald)
        buf1[:, :, :3] = buf0
        buf1[:,:,3] = 255
        self.hald.parentImage = self.parentImage

    def getHald(self):
        if not self.cachesEnabled:
            s = int((LUT3DIdentity.size) ** (3.0 / 2.0)) + 1
            buf0 = LUT3DIdentity.getHaldImage(s, s)
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
        Returns current (full, preview or hald) image, according to
        the value of the flags useThumb and useHald. The thumbnail and hald
        are computed if they are not initialized.
        Otherwise, they are not updated unless self.thumb is
        None or purgeThumb is True.
        Overrides vImage method
        @return: current image
        @rtype: QLayer
        """
        if self.parentImage.useHald:
            return self.getHald()
        if self.parentImage.useThumb:
            return self.getThumb()
        else:
            return self

    def inputImg(self):
        """
        Updates and returns maskedImageContainer and maskedThumbContainer
        @return:
        @rtype: Qlayer
        """
        # TODO 29/05/18 optimize to avoid useless updates of containers (add a valid flag to each container and invalidate it in cacheInvalidate())
        return self.parentImage.layersStack[self.getLowerVisibleStackIndex()].getCurrentMaskedImage()

    def full2CurrentXY(self, x, y):
        """
        Maps x,y coordinates of pixel in the full image to
        coordinates in current image.
        @param x:
        @type x: int or float
        @param y:
        @type y: int or float
        @return:
        @rtype: 2uple of int
        """
        if self.parentImage.useThumb:
            currentImg = self.getThumb()
            x = (x * currentImg.width()) / self.width()
            y = (y * currentImg.height()) / self.height()
        return int(x), int(y)

    def getCurrentMaskedImage(self):
        """
        Reduces the layer stack up to self (included),
        taking into account the masks. if self.isClipping is True
        self.mask applies to all lower layers and to self only otherwise.
        The method uses the non color managed rPixmaps to build the masked image.
        For convenience, mainly to be able to use its color space buffers,
        the built image is of type QLayer. It is drawn on a container image,
        created only once.
        @return: masked image
        @rtype: QLayer
        """
        # init containers if needed
        if self.parentImage.useHald:
            return self.getHald()
        if self.maskedThumbContainer is None:
            self.maskedThumbContainer = QLayer.fromImage(self.getThumb(), parentImage=self.parentImage)
        if self.maskedImageContainer is None:
            self.maskedImageContainer = QLayer.fromImage(self, parentImage=self.parentImage)
        if self.parentImage.useThumb:
            img = self.maskedThumbContainer
        else:
            img = self.maskedImageContainer
        # no thumbnails for containers
        img.getThumb = lambda: img
        # draw lower stack
        qp = QPainter(img)
        top = self.parentImage.getStackIndex(self)
        bottom = 0
        for i, layer in enumerate(self.parentImage.layersStack[bottom:top+1]):
            if layer.visible:
                if i == 0:
                    qp.setCompositionMode(QPainter.CompositionMode_Source)
                else:
                    qp.setOpacity(layer.opacity)
                    qp.setCompositionMode(layer.compositionMode)
                if layer.rPixmap is not None:
                    qp.drawPixmap(QRect(0,0,img.width(), img.height()), layer.rPixmap)
                else:
                    qp.drawImage(QRect(0,0,img.width(), img.height()), layer.getCurrentImage())
                # clipping
                if layer.isClipping and layer.maskIsEnabled: #TODO modified 23/06/18
                    # draw mask as opacity mask
                    # mode DestinationIn (set dest opacity to source opacity)
                    qp.setCompositionMode(QPainter.CompositionMode_DestinationIn)
                    omask = vImage.color2OpacityMask(layer.mask)
                    qp.drawImage(QRect(0, 0, img.width(), img.height()), omask)
        qp.end()
        """
        if self.isClipping and self.maskIsEnabled: #TODO modified 23/06/18
            mask = self.mask.scaled(img.width(), img.height())  # vImage.color2OpacityMask(self.mask)
            img = vImage.visualizeMask(img, mask, color=False, clipping=False, copy=False)
        """
        # buf = QImageBuffer(img) #TODO 24/09/18 removed validate
        return img

    def applyToStack(self):
        """
        Applies transformation and propagates changes to upper layers.
        """
        # recursive function
        def applyToStack_(layer, pool=None):
            # apply Transformation (call vImage.apply*LUT...)
            if layer.visible:
                start = time()
                layer.execute(l=layer)#, pool=pool) use default pool
                layer.cacheInvalidate()
                print("%s %.2f" %(layer.name, time()-start))
            stack = layer.parentImage.layersStack
            ind = layer.getStackIndex() + 1
            # get next visible upper layer
            while ind < len(stack):
                if stack[ind].visible:
                    break
                ind += 1
            if ind < len(stack):
                layer1 = stack[ind]
                applyToStack_(layer1, pool=pool)
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            """
            if (not self.parentImage.useThumb or self.parentImage.useHald):
                pool = None
                # pool = multiprocessing.Pool(MULTIPROC_POOLSIZE)  # TODO time opt : pool is always created and used only by apply3DLUT; time 0.3s
            else:
                pool = None
            """
            applyToStack_(self, pool=None)
            """
            if pool is not None:
                pool.close()
            pool = None
            """
            # update the presentation layer
            self.parentImage.prLayer.execute(l=None, pool=None)
        finally:
            self.parentImage.setModified(True)
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
    """
    def applyToStackIter(self):
        #iterative version of applyToStack
        stack = self.parentImage.layersStack
        ind = self.getStackIndex() + 1
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            self.execute()
            for layer in stack[ind:]:
                if layer.visible:
                    layer.cacheInvalidate()
                    # for hald friendly layer compute output hald, otherwise compute output image
                    layer.execute()
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
    """
    def isAdjustLayer(self):
        return self.view is not None #hasattr(self, 'view')

    def isSegmentLayer(self):
        return 'SEGMENT' in self.role

    def isCloningLayer(self):
        return 'CLONING' in self.role

    def isGeomLayer(self):
        return 'GEOM' in self.role

    def is3DLUTLayer(self):
        return '3DLUT' in self.role

    def isRawLayer(self):
        return 'RAW' in self.role

    def updatePixmap(self, maskOnly = False):
        """
        Updates the cached rPixmap.
        if maskIsEnabled is False, the mask is not shown.
        If maskIsEnabled is True, then
            - if maskIsSelected is True, the mask is drawn over
              the layer as a color mask.
            - if maskIsSelected is False, the mask is drawn as an
              opacity mask, setting image opacity to that of mask
              (mode DestinationIn). Mask color is no used.
        @param maskOnly: not used yet
        @type maskOnly: boolean
        """
        rImg = self.getCurrentImage()
        # apply layer transformation. Missing pixels are set to QColor(0,0,0,0)
        if self.xOffset != 0 or self.yOffset != 0:
            x,y = self.full2CurrentXY(self.xOffset, self.yOffset)
            rImg = rImg.copy(QRect(-x, -y, rImg.width()*self.Zoom_coeff, rImg.height()*self.Zoom_coeff))
        if self.maskIsEnabled:
            rImg = vImage.visualizeMask(rImg, self.mask, color=self.maskIsSelected, clipping=self.isClipping)
        self.rPixmap = QPixmap.fromImage(rImg)
        self.setModified(True)

    def getStackIndex(self):
        """
        Returns layer index in the stack, len(stack) - 1 if
        the layer is not in the stack.
        @return:
        @rtype: int
        """
        for i, l in enumerate(self.parentImage.layersStack):
            if l is self:
                break
        return i

    def updateTableView(self, table):
        """
        refreshes the corresponding row in table
        @param table:
        @type table: QTableView
        """
        ind = self.getStackIndex()
        table.updateRow(len(self.parentImage.layersStack) - 1 - ind)

    def getTopVisibleStackIndex(self):
        """
        Returns the index of the top visible layer
        @return:
        @rtype:
        """
        stack = self.parentImage.layersStack
        lg = len(stack)
        for i in range(lg-1, -1, -1):
            if stack[i].visible:
                return i
        return -1
    def getLowerVisibleStackIndex(self):
        """
        Returns the index of the next lower visible layer,
        -1 if it does not exists
        @return:
        @rtype: int
        """
        ind = self.getStackIndex()
        stack = self.parentImage.layersStack
        for i in range(ind-1, -1, -1):
            if stack[i].visible:
                return i
        return -1

    def getUpperVisibleStackIndex(self):
        """
        Returns the index of the next upper visible layer,
        -1 if it does not exists
        @return:
        @rtype: int
        """
        ind = self.getStackIndex()
        stack = self.parentImage.layersStack
        lg = len(stack)
        for i in range(ind+1, lg, 1):
            if stack[i].visible:
                return i
        return -1

    def getLowerClippingStackIndex(self):
        """
         Returns the index of the next lower clipping layer,
        -1 if it does not exists

        @return:
        @rtype: int
        """
        ind = self.getStackIndex()
        for i in range(ind + 1, len(self.parentImage.layersStack), 1):
            if self.parentImage.layersStack[i].isClipping:
                return i
        return -1

    def linkMask2Lower(self):
        """
        share mask with next lower layer
        @return:
        @rtype:
        """
        ind = self.getStackIndex()
        if ind == 0:
            return
        lower = self.parentImage.layersStack[ind-1]
        # don't link two groups
        if self.group and lower.group:
            return
        if not self.group and not lower.group:
            self.group = [self, lower]
            lower.group = self.group
        elif not lower.group :
            if not any(o is lower for o in self.group):
                self.group.append(lower)
            lower.group  = self.group
        elif not self.group:
            if not any(item is self for item in lower.group):
                lower.group.append(self)
            self.group = lower.group
        self.mask = lower.mask

    def unlinkMask(self):
        self.mask =self.mask.copy()
        # remove self from group
        for i,item in enumerate(self.group):
            if item is self:
                self.group.pop(i)
                # don't keep  group with length 1
                if len(self.group) == 1:
                    self.group.pop(0)
                break
        self.group = []

    def merge_with_layer_immediately_below(self):
        """
        Merges a layer with the next lower visible layer. Does nothing
        if mode is preview or the target layer is an adjustment layer.
        """
        if not hasattr(self, 'inputImg'):
            return
        ind = self.getLowerVisibleStackIndex()
        if ind < 0:
            # no visible layer found
            return
        target = self.parentImage.layersStack[ind]
        if hasattr(target, 'inputImg') or self.parentImage.useThumb:
            info = "Uncheck Preview first" if self.parentImage.useThumb else "Target layer must be background or image"
            dlgWarn("Cannot Merge layers", info=info)
            return
        #update stack
        self.parentImage.layersStack[0].applyToStack()
        # merge
        #target.setImage(self)
        qp = QPainter(target)
        qp.setCompositionMode(self.compositionMode)
        qp.setOpacity(self.opacity)
        qp.drawImage(QRect(0,0,self.width(), self.height()), self)
        target.updatePixmap()
        self.parentImage.layerView.clear(delete=False)
        currentIndex = self.getStackIndex()
        self.parentImage.activeLayerIndex = ind
        self.parentImage.layersStack.pop(currentIndex)
        self.parentImage.layerView.setLayers(self.parentImage)

    def reset(self):
        """
        reset layer to inputImg
        @return:
        """
        self.setImage(self.inputImg())

    def setOpacity(self, value):
        """
        set the opacity attribute to value/100.0
        @param value:
        """
        self.opacity = value /100.0
        return

    def readFromStream(self, dataStream):

        if hasattr(self, 'view'):
            self.view.widget().readFromStream(dataStream)
        return dataStream

class QPresentationLayer(QLayer):
    """
    A presentation layer is used for color management. It is simply an
    adjustment layer that does nothing. It does not belong to the layer stack :
    conceptually, it is "above" the stack, so its output is the composition of
    all stacked layers. It is the sole color managed layer. It holds a cmImage
    attribute caching the current color managed image.
    """

    def __init__(self, *args, **kwargs):
        self.cmImage = None
        super().__init__(*args, **kwargs)

    def inputImg(self):
        return self.parentImage.layersStack[self.getTopVisibleStackIndex()].getCurrentMaskedImage()

    def updatePixmap(self, maskOnly = False):
        """
        Updates the caches qPixmap, rPixmap and cmImage.
        The input image is that returned by getCurrentImage(), thus
        the caches are synchronized using the current image
        mode (full or preview).
        If maskOnly is True, cmImage is not updated.
        if maskIsEnabled is False, the mask is not shown.
        If maskIsEnabled is True, then
            - if maskIsSelected is True, the mask is drawn over
              the layer as a color mask.
            - if maskIsSelected is False, the mask is drawn as an
              opacity mask, setting image opacity to that of mask
              (mode DestinationIn). Mask color is no used.
        @param maskOnly: default False
        @type maskOnly: boolean
        """
        currentImage = self.getCurrentImage()
        if not maskOnly:
            # invalidate color managed cache
            self.cmImage = None
        # get current image and apply color management if self is the presentation layer
        if icc.COLOR_MANAGE and self.parentImage is not None and getattr(self, 'role', None) == 'presentation':
            # layer color model is parent image color model
            if self.cmImage is None:
                # CAUTION : reset alpha channel
                img = convertQImage(currentImage, transformation=self.parentImage.colorTransformation)  # time 0.66 s for 15 Mpx.
                # preserve alpha channel
                img = img.convertToFormat(currentImage.format()) #TODO 14/06/18 convertToFormat is mandatory to get ARGB imagesinstead of RGB images
                buf0 = QImageBuffer(img)
                buf1 = QImageBuffer(currentImage)
                buf0[:,:,3] = buf1[:,:,3]
            else:
                img = self.cmImage
        else:
            img = currentImage
        self.cmImage = img
        qImg = img
        rImg = currentImage
        # apply layer transformation. Missing pixels are set to QColor(0,0,0,0)
        if self.xOffset != 0 or self.yOffset != 0:
            x,y = self.full2CurrentXY(self.xOffset, self.yOffset)
            qImg = qImg.copy(QRect(-x, -y, qImg.width()*self.Zoom_coeff, qImg.height()*self.Zoom_coeff))
            rImg = rImg.copy(QRect(-x, -y, rImg.width()*self.Zoom_coeff, rImg.height()*self.Zoom_coeff))
        if self.maskIsEnabled:
            #qImg = vImage.visualizeMask(qImg, self.mask, color=self.maskIsSelected, clipping=self.isClipping)
            rImg = vImage.visualizeMask(rImg, self.mask, color=self.maskIsSelected, clipping=self.isClipping)
        self.qPixmap = QPixmap.fromImage(qImg)
        self.rPixmap = QPixmap.fromImage(rImg)
        self.setModified(True)

    def update(self):
        self.applyNone()
        # self.updatePixmap() done by applyNone()

class QLayerImage(QLayer):
    """
    QLayer holding a source image
    """
    @classmethod
    def fromImage(cls, mImg, parentImage=None, sourceImg=None):
        layer = QLayerImage(QImg=mImg, parentImage=parentImage)
        layer.parentImage = parentImage
        layer.sourceImg = sourceImg
        return layer

    def __init__(self, *args, **kwargs):
        self.sourceImg = QImage()
        super().__init__(*args, **kwargs)

    def bTransformed(self, transformation, parentImage):
        """
        Applies transformation to a copy of layer and returns the copy.
        @param transformation:
        @type transformation: QTransform
        @param parentImage:
        @type parentImage: vImage
        @return: transformed layer
        @rtype: QLayerImage
        """
        # TODO modified 26/09/18 validate
        tLayer = super().bTransformed(self, transformation, parentImage)
        """ 
        tLayer = QLayerImage.fromImage(self.transformed(transformation), parentImage=parentImage, sourceImg=self.sourceImg.transformed(transformation))
        # add dynamic attributes
        for a in self.__dict__.keys():
            av = tLayer.__dict__.get(a, None)
            if av is None or av == '':
                tLayer.__dict__[a] = self.__dict__[a]
        # link back grWindow to tLayer
        if tLayer.view is not None:
            tLayer.view.widget().layer = tLayer
        if tLayer.tool is not None:
            tLayer.tool.layer = tLayer
            tLayer.tool.img = tLayer.parentImage
        tLayer.execute = self.execute
        tLayer.mask = self.mask.transformed(transformation)
        tLayer.maskIsEnabled, tLayer.maskIsSelected = self.maskIsEnabled, self.maskIsSelected
        # keep original layer
        tLayer.parentLayer = self.parentLayer
        # record new transformed layer
        self.parentLayer.tLayer = tLayer
        """
        if tLayer.tool is not None:
            tLayer.tool.layer = tLayer
            tLayer.tool.img = tLayer.parentImage
        return tLayer

    def inputImg(self):
        """
        Overrides QLayer.inputImg()
        @return:
        @rtype: QImage
        """
        return self.sourceImg.scaled(self.getCurrentImage().size())

def apply3DLUTSliceCls(LUT, inputBuffer, imgBuffer, s ):

    inputBuffer = inputBuffer[s[1], s[0], :]
    imgBuffer = imgBuffer[:, :, :]
    ndImg0 = inputBuffer[:, :, :3]
    ndImg1 = imgBuffer[:, :, :3]
    # apply LUT
    start = time()
    ndImg1[s[1], s[0], :] = interpVec_(LUT, ndImg0)
    end = time()
    #print 'Apply3DLUT time %.2f' % (end - start)

def applyHaldCls(item):
    """
    Transforms a hald image into a 3DLUT object and applies
    the 3D LUT to the current view of self.
    """
    # QImageBuffer(l.hald), QImageBuffer(l.inputImg()), QImageBuffer(l.getCurrentImage()), s
    lut = LUT3D.HaldBuffer2LUT3D(item[0])
    apply3DLUTSliceCls(lut.LUT3DArray, item[1], item[2], item[3])
    return item[2]
    #img.apply3DLUT(lut.LUT3DArray, options={'use selection' : True})




