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
import json
from trace import pickle

from PySide.QtCore import Qt

import cv2
from copy import copy

from PySide.QtGui import QImageWriter
from PySide.QtGui import QPixmap, QImage, QColor, QPainter, QMessageBox
from PySide.QtCore import QRect

from colorTemperature import sRGB2LabVec
from graphicsFilter import filterIndex
from icc import convertQImage
from imgconvert import *
from LUT3D import interpVec, rgb2hspVec, hsp2rgbVec
from time import time
import icc
from utils import savitzky_golay, channelValues


class ColorSpace:
    notSpecified = -1; sRGB = 1

class metadata:
    """
    Container for vImage meta data
    """
    def __init__(self, name=''):
        self.name, self.colorSpace, self.rawMetadata, self.profile, self.orientation = name, ColorSpace.notSpecified, [], '', None

class vImage(QImage):
    """
    versatile image class.
    This is the base class for all multi-layered and interactive image classes.
    It gathers all image information, including meta-data.
    Each image owns a mask (disabled by default). When enabled, masking operation is bitwise AND.
    Several data buffers are defined for thumbnails and color modes.
    they are cached; access functions are initThumb, getHspbBuffer, getLabBuffer,
    updatePixmap, cacheInvalidate.
    """
    thumbSize = 1000
    def __init__(self, filename=None, cv2Img=None, QImg=None, mask=None, format=QImage.Format_ARGB32,
                                            name='', colorSpace=-1, orientation=None, meta=None, rawMetadata=[], profile=''):
        """
        With no parameter, builds a null image.
        Mask is disabled by default.
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
        @param colorSpace: color space (default notSpecified)
        @type colorSpace: MarkedImg.colorSpace
        @param orientation: Qtransform object (default None)
        @type orientation: Qtransform 
        @param meta: meta data (default None)
        @type meta: MarkedImg.metadata
        @param rawMetadata: list of dictionaries (default [])
        @type rawMetadata: list of dictionaries
        @param profile: embedded profile (default '')
        @type profile: str
        """
        self.colorTransformation = None
        self.isModified = False
        self.onModify = lambda : 0
        self.rect, self.mask, = None, mask
        self.filename = filename

        # Caches
        self.useThumb = False
        self.thumb = None
        self.qPixmap = None
        self.hspbBuffer = None
        self.LabBuffer = None

        self.onImageChanged = lambda: 0

        if meta is None:
            # init container
            self.meta = metadata()
            self.meta.name, self.meta.colorSpace, self.meta.rawMetadata, self.meta.profile, self.meta.orientation = name, colorSpace, rawMetadata, profile, orientation
        else:
            self.meta = meta

        if (filename is None and cv2Img is None and QImg is None):
            # creates a null image
            super(vImage, self).__init__()

        if filename is not None:
            # loads image from file (should be a 8 bits/channel color image)
            if self.meta.orientation is not None:
                tmp =QImage(filename).transformed(self.meta.orientation)
            else:
                tmp = QImage(filename)
            # call to super is mandatory. Shallow copy : no harm !
            super(vImage, self).__init__(tmp)
        elif QImg is not None:
            # builds image from QImage, shallow copy
            super(vImage, self).__init__(QImg)
            if hasattr(QImg, "meta"):
                self.meta = copy(QImg.meta)
        elif cv2Img is not None:
            # builds image from buffer
            super(vImage, self).__init__(ndarrayToQImage(cv2Img, format=format))
        # Format check
        if self.depth() != 32:
            raise ValueError('vImage : should be a 8 bits/channel color image')
        # mask
        self.maskIsEnabled = False
        self.maskIsSelected = False
        if self.mask is None:
            self.mask = QImage(self.width(), self.height(), format)
            self.mask.fill(QColor(255, 0, 255))

        self.updatePixmap()

    def getCurrentImage(self):
        if self.useThumb:
            if self.thumb is None:
                self.initThumb()
            currentImage = self.thumb
        else:
            currentImage = self
        return currentImage

    def getHspbBuffer(self):
        """
        returns the image buffer in color mode HSpB.
        The buffer is calculated if needed and cached.
        @return: HSPB buffer 
        @rtype: ndarray
        """
        #inputImage = self.inputImgFull().getCurrentImage()
        if self.hspbBuffer is None:
            currentImage = self.getCurrentImage()
            self.hspbBuffer = rgb2hspVec(QImageBuffer(currentImage)[:,:,:3][:,:,::-1])
            """
            if self.useThumb:
                self.hspbBuffer = rgb2hspVec(QImageBuffer(self.thumb)[:,:,:3][:,:,::-1])
            else:
                self.hspbBuffer = rgb2hspVec(QImageBuffer(self)[:,:,:3][:,:,::-1])
            """
        return self.hspbBuffer

    def getLabBuffer(self):
        """
        returns the image buffer in color mode Lab.
        The buffer is calculated if needed and cached.
        @return: Lab buffer 
        @rtype: numpy ndarray, dtype numpy.float64
        """
        if self.LabBuffer is None:
            currentImage = self.getCurrentImage()
            self.LabBuffer = sRGB2LabVec(QImageBuffer(currentImage)[:, :, :3][:, :, ::-1])
            """
            if self.useThumb:
                self.LabBuffer = sRGB2LabVec(QImageBuffer(self.thumb)[:,:,:3][:,:,::-1])
            else:
                self.LabBuffer = sRGB2LabVec(QImageBuffer(self)[:,:,:3][:,:,::-1])
            """
        return self.LabBuffer

    def setModified(self, b):
        """
        Sets the flag isModified and calls handler.
        @param b: flag
        @type b: boolean
        """
        self.isModified = b
        self.onModify()

    def initThumb(self):
        """
        Inits thumbnail
        @return: 
        """
        self.thumb = self.scaled(self.thumbSize, self.thumbSize, Qt.KeepAspectRatio)  #QImage(QImg=self.scaled(500, 500, Qt.KeepAspectRatio))

    def cacheInvalidate(self):
        """
        Invalidates thumbnail and buffers.
        @return: 
        """
        self.thumb = None
        self.hspbBuffer = None
        self.LabBuffer = None

    def updatePixmap(self):
        """
        Updates image pixmap
        @return: 
        """
        if self.useThumb:
            currentImage = self.thumb
        else:
            currentImage = self
        if icc.COLOR_MANAGE:
            img = convertQImage(currentImage)
        else:
            img = QImage(currentImage)
        if self.maskIsEnabled:
            qp = QPainter(img)
            qp.setCompositionMode(QPainter.RasterOp_SourceAndDestination)
            qp.drawImage(0, 0, self.mask)
            qp.end()
        self.qPixmap = QPixmap.fromImage(img)

    def resetMask(self):
        self.mask.fill(QColor(255, 255, 255, 255))

    def resize(self, pixels, interpolation=cv2.INTER_CUBIC):
        """
        Resize an image while keeping its aspect ratio. We use
        the opencv function cv2.resize() to perform the resizing operation, so we
        can choose the resizing method (default cv2.INTER_CUBIC)
        The original image is not modified.
        @param pixels: pixel count for the resized image
        @param interpolation method (default cv2.INTER_CUBIC)
        @return : the resized vImage
        """
        ratio = self.width() / float(self.height())
        w, h = int(np.sqrt(pixels * ratio)), int(np.sqrt(pixels / ratio))
        hom = w / float(self.width())
        # resizing
        Buf = QImageBuffer(self)
        cv2Img = cv2.resize(Buf, (w, h), interpolation=interpolation)
        rszd = vImage(cv2Img=cv2Img, meta=copy(self.meta), format=self.format())
        # prevents buffer from garbage collector
        rszd.dummy = cv2Img
        #resize rect and mask
        if self.rect is not None:
            rszd.rect = QRect(self.rect.left() * hom, self.rect.top() * hom, self.rect.width() * hom, self.rect.height() * hom)
        if self.mask is not None:
            # tmp.mask=cv2.resize(self.mask, (w,h), interpolation=cv2.INTER_NEAREST )
            rszd.mask = self.mask.scaled(w, h)
        self.setModified(True)
        return rszd

    def apply1DLUT(self, stackedLUT, options={}):
        """
        Applies 1D LUTS (one for each channel)
        @param stackedLUT: array of color values (in range 0..255). Shape must be (3, 255) : a line for each channel
        @param options: not used yet
        """
        # get image buffers (BGR order on intel arch.)
        inputImage = self.inputImgFull().getCurrentImage()
        currentImage = self.getCurrentImage()
        ndImg0 = QImageBuffer(inputImage)[:, :, :3]
        ndImg1 = QImageBuffer(currentImage)[:, :, :3]
        # apply LUTS to channels
        rList = np.array([2,1,0]) #BGR
        ndImg1[:, :, :]= stackedLUT[rList[np.newaxis,:], ndImg0]
        # update
        self.updatePixmap()

    def applyLab1DLUT(self, stackedLUT, options={}):
        """
        Applies 1D LUTS (one for each L,a,b channel)
        @param stackedLUT: array of color values (in range 0..255). Shape must be (3, 255) : a line for each channel
        @param options: not used yet
        """
        from colorTemperature import sRGB2LabVec, Lab2sRGBVec, rgb2rgbLinearVec, rgbLinear2rgbVec
        # Lab mode
        ndLabImg0 = self.getLabBuffer()

        # apply LUTS to channels
        def scaleLabBuf(buf):
            buf = buf + [0.0, 128.0, 128.0]
            buf = buf * [255.0, 1.0, 1.0]# [255.0, 255.0/210.0, 255.0/210.0]
            return buf
        def scaleBackLabBuf(buf):
            #buf = np.dstack((buf[:, :, 0] / 255.0, buf[:, :, 1] * (210.0/255.0), buf[:, :, 2] * (210.0/ 255.0)))
            buf = np.dstack((buf[:, :, 0] / 255.0, buf[:, :, 1] , buf[:, :, 2] ))
            buf = buf - [0.0, 128.0, 128.0]
            return buf
        ndLImg0 = scaleLabBuf(ndLabImg0).astype(int)  #TODO problem here with astype(int) conversion
        #ndLImg0 = (ndLabImg0 * [1.0, 255.0, 255.0]).astype(int)
        rList = np.array([0, 1, 2])  # Lab
        ndLabImg1 = stackedLUT[rList[np.newaxis, :], ndLImg0]
        # LUT = stackedLUT[2,:]
        # ndLImg1 = stackedLUT[ndLImg0]
        ndLabImg1 = scaleBackLabBuf(ndLabImg1) #np.dstack((ndLabImg1[:, :, 0] / 255.0, ndLabImg1[:, :, 1] * (210.0/255.0), ndLabImg1[:, :, 2] * (210.0/ 255.0)))
        # back sRGB conversion
        ndsRGBImg1 = Lab2sRGBVec(ndLabImg1)
        # clipping is mandatory here : numpy bug ?
        ndsRGBImg1 = np.clip(ndsRGBImg1, 0, 255)
        currentImage = self.getCurrentImage()
        ndImg1 = QImageBuffer(currentImage)[:, :, :3]
        ndImg1[:, :, ::-1] = ndsRGBImg1
        # update
        self.updatePixmap()
        """
        # get image buffers (RGB order on intel arch.)
        ndImg0 = QImageBuffer(self.inputImgFull().getCurrentImage())[:, :, :3][:,:,::-1]
        # Lab conversion
        ndLabImg0 = sRGB2LabVec(ndImg0)

        # apply LUTS to channels
        ndLImg0 = (ndLabImg0[:,:,0]*255.0).astype(int)
        #rList = np.array([0,1,2]) # Lab
        #ndLabImg1 = stackedLUT[rList[np.newaxis,:], ndLabImg0]
        LUT = stackedLUT[0,:]
        ndLImg1 = LUT[ndLImg0] /255.0
        ndLabImg1 = np.dstack((ndLImg1, ndLabImg0[:,:,1], ndLabImg0[:,:,2]))
        # back sRGB conversion
        ndsRGBImg1 = Lab2sRGBVec(ndLabImg1)
        # clipping is mandatory here : numpy bug ?
        ndsRGBImg1 = np.clip(ndsRGBImg1, 0, 255)
        ndImg1 = QImageBuffer(self)[:, :, :3]
        ndImg1[:,:,::-1] = ndsRGBImg1

        # update
        self.updatePixmap()
        """

    def applyHSPB1DLUT(self, stackedLUT, options={}):
        """
        Applies 1D LUTS (one for each hue sat and brightness channels)
        @param stackedLUT: array of color values (in range 0..255). Shape must be (3, 255) : a line for each channel
        @param options: not used yet
        """
        # hspb mode
        ndHSPBImg0 = self.getHspbBuffer()

        # apply LUTS to channels
        #ndLImg0 = (ndHSPBImg0[:,:,1:] * 255).astype(int) # suppress hue channel
        ndLImg0 = (ndHSPBImg0 * [255.0/360.0, 255.0, 255.0]).astype(int)
        rList = np.array([0,1,2]) # HSB
        ndLabImg1 = stackedLUT[rList[np.newaxis,:], ndLImg0]
        #LUT = stackedLUT[2,:]
        #ndLImg1 = stackedLUT[ndLImg0]
        ndHSBPImg1 = np.dstack((ndLabImg1[:,:,0]*360.0/255.0, ndLabImg1[:,:,1]/255.0, ndLabImg1[:,:,2]/255.0))
        # back sRGB conversion
        ndRGBImg1 = hsp2rgbVec(ndHSBPImg1)
        # clipping is mandatory here : numpy bug ?
        ndRGBImg1 = np.clip(ndRGBImg1, 0, 255)
        currentImage = self.getCurrentImage()
        ndImg1 = QImageBuffer(currentImage)[:, :, :3]
        ndImg1[:,:,::-1] = ndRGBImg1
        # update
        self.updatePixmap()

    def apply3DLUT(self, LUT, options={}):
        """
        Applies 3D LUT to the image
        @param LUT: LUT3D array (see module LUT3D.py)
        @param options: dict of string:boolean records
        """
        # get selection
        w1, w2, h1, h2 = (0.0,) * 4
        if options['use selection']:
            if self.rect is not None:
                w1, w2, h1,h2= self.rect.left(), self.rect.right(), self.rect.top(), self.rect.bottom()
            if w1>=w2 or h1>=h2:
                msg = QMessageBox()
                msg.setText("Empty selection\nSelect a region with the marquee tool")
                msg.exec_()
                return
        else:
            w1, w2, h1, h2 = 0, self.inputImg().width(), 0, self.inputImg().height()

        ndImg0 = QImageBuffer(input)[h1+1:h2+1, w1+1:w2+1, :3]
        ndImg1 = QImageBuffer(self)[:, :, :3]

        # apply LUT
        start=time()
        ndImg1[h1+1:h2+1,w1+1:w2+1,:] = interpVec(LUT, ndImg0)
        end=time()

        print 'time %.2f' % (end-start)

        self.updatePixmap()

    def histogram(self, size=200, bgColor=Qt.white, range =(0,255), chans=channelValues.RGB, chanColors=Qt.gray, mode='RGB'):
        """
        Plots the histogram of the image for the
        specified color mode and channels.
        @param size: size of the histogram plot
        @type size: int
        @param bgColor: background color
        @type bgColor: QColor
        @param range: plot data range
        @type range: 2-uple of int or float
        @param chanColors: color or 3-uple of colors
        @type chanColors: QColor or 3-uple of QColor 
        @param mode: color mode ((one of 'RGB', 'HSpB', 'Lab')
        @type mode: str
        @return: histogram plot
        @rtype: QImage
        """
        # scaling factor for the bin edges
        spread = float(range[1] - range[0])
        scale = size / spread

        def drawChannelHistogram(qp, channel):
            """
            Computes and draws the (smoothed) histogram of the image for a single channel.
            @param channel: channel index (BGRA (intel) or ARGB )
            """
            buf0 = buf[:,:, channel]
            # bins='auto' sometimes causes huge number of bins ( >= 10**9) and memory error
            # even for small data size (<=250000), so we don't use it.
            # This is a numpy bug : in module function_base.py, to avoid memory error,
            # a reasonable upper bound for bins should be fixed at line 532.
            # hist, bin_edges = np.histogram(buf0, bins='auto', density=True)
            hist, bin_edges = np.histogram(buf0, bins=100, density=True)
            # smooth hist
            hist = savitzky_golay(hist, 11, 3)
            p = len(hist) - len(bin_edges)
            # P > 1 iff hist was padded by savitzky_golay filter
            # then, we pad bin_edges accordingly
            if p >1:
                p = p /2 +1
                bin_firstVals = (bin_edges[0] - np.arange(p))[::-1]
                bin_lastVals = bin_edges[-1] + np.arange(p-1)
                bin_edges = np.concatenate((bin_firstVals, bin_edges, bin_lastVals))
            # draw hist
            if type(chanColors) is QColor or type(chanColors) is Qt.GlobalColor:
                color = chanColors
            else:
                color = chanColors[channel] # Qt.red if channel == 0 else Qt.green if channel == 1 else Qt.blue
            qp.setPen(color)
            M = max(hist) #+ size / 4.0
            for i, y in enumerate(hist):
                #h = int(y*20*255)
                h = int(size * y / M)
                qp.drawRect(int((bin_edges[i]-range[0])*scale), max(img.height()-h,0), int((bin_edges[i+1]-bin_edges[i])*scale), h)

        if mode == 'RGB':
            buf = QImageBuffer(self)[:,:,:3][:,:,::-1]  #RGB
        elif mode == 'HSpB':
            buf = self.getHspbBuffer()
        elif mode == 'Lab':
            buf = self.getLabBuffer()
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(bgColor)
        qp = QPainter(img)
        for ch in chans:
            drawChannelHistogram(qp, ch)
        qp.end()
        buf = QImageBuffer(img)
        return img

    def applyFilter2D(self):
        """
        Applies 2D kernel. Available kernels are
        sharpen, unsharp, gaussian_blur
        @param radius: filter radius
        @type radius: float
        @param filter: filter type
        @type filter: filterIndex object (enum)
        @return: 
        """
        """
        unsharp_kernel = - np.array([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, -476, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]]) / 256.0

        sharpen_kernel = np.array([[0.0, -1.0, 0.0],
                                  [-1.0, 5.0, -1.0],
                                  [0.0,-1.0, 0.0]])


        gblur1_kernel = np.array([[1, 2, 1],
                                 [2, 4 ,2],
                                 [1, 2, 1]]) / 16.0

        gblur2_kernel = np.array([[1, 4,  6,  4,  1],
                                  [4, 16, 24, 16, 4],
                                  [6, 24, 36, 24, 6],
                                  [4, 16, 24, 16, 4],
                                  [1, 4,  6,  4,  1]]) / 256.0
        """

        inputImage = self.inputImgFull().getCurrentImage()
        currentImage = self.getCurrentImage()
        buf0 = QImageBuffer(inputImage)
        buf1 = QImageBuffer(currentImage)
        buf1[:, :, :] = cv2.filter2D(buf0, -1, self.kernel)
        self.updatePixmap()

    def applyTemperature(self, temperature, options, coeff=1.0):
        """

        @param qImg:
        @param temperature:
        @param coeff:
        @return:
        """
        from blend import blendLuminosity
        from colorTemperature import bbTemperature2RGB, conversionMatrix, rgb2rgbLinearVec, rgbLinear2rgbVec
        if options['use Chromatic Adaptation']:
            version = 2
        else:
            version = 0
        inputImage = self.inputImgFull().getCurrentImage()
        currentImage = self.getCurrentImage()
        if version == 0:
            # black body color
            r, g, b = bbTemperature2RGB(temperature)
            filter = QImage(inputImage)
            filter.fill(QColor(r, g, b, 255))
            qp = QPainter(filter)
            # qp.setOpacity(coeff)
            qp.setCompositionMode(QPainter.CompositionMode_Multiply)
            qp.drawImage(0, 0, inputImage)
            qp.end()
            resImg = blendLuminosity(filter, inputImage)
            res = QImageBuffer(resImg)[:,:,:3][:,:,::-1]
        else:
            M = conversionMatrix(temperature, 6500)
            buf = QImageBuffer(inputImage)[:, :, :3]
            bufLinear = rgb2rgbLinearVec(buf)
            resLinear = np.tensordot(bufLinear[:, :, ::-1], M, axes=(-1, -1))
            res = rgbLinear2rgbVec(resLinear)
            res = np.clip(res, 0, 255)
        bufOut = QImageBuffer(currentImage)[:,:,:3]
        bufOut[:, :, ::-1] = res
        self.updatePixmap()
       # return img

class mImage(vImage):
    """
    Multi-layer image. A mImage object holds at least a background
    layer. All layers share the same metadata object. To correctly render a
    mImage, widgets must override their paint event handler.
    """

    def __init__(self, *args, **kwargs):
        # as updatePixmap uses layersStack, must be before super __init__
        self._layers = {}
        self.layersStack = []
        super(mImage, self).__init__(*args, **kwargs)
        # add background layer
        bgLayer = QLayer.fromImage(self, parentImage=self)
        self.setModified(False)
        self.activeLayerIndex = None
        self.addLayer(bgLayer, name='background')
        self.isModified = False
        # will be set to QTableView referenec
        self.layerView = None

    def getActiveLayer(self):
        return self.layersStack[self.activeLayerIndex]

    def setActiveLayer(self, value):
        """
        Assigns value to  activeLayerIndex and
        updates the layer View Form
        @param value: 
        @return: 
        """
        self.activeLayerIndex = value
        if hasattr(self, 'layerView'):
            self.layerView.selectRow(len(self.layersStack) - 1 - value)

    def getActivePixel(self,x, y):
        """
        Reads pixel value from active layer. For
        adjustment or segmentation layer, we read pixel value
        from input image.
        @param x, y: coordinates of pixel
        @return: pixel value (type QRgb : unsigned int ARGB)
        """
        activeLayer = self.getActiveLayer()
        if  hasattr(activeLayer, "inputImg") and activeLayer.inputImg is not None:
            # layer is adjustment or segmentation : read from input image
            return activeLayer.inputImg().pixel(x, y)
        else:
            # read from image
            return activeLayer.pixel(x, y)

    def cacheInvalidate(self):
        vImage.cacheInvalidate(self)
        for layer in self.layersStack:
            layer.cacheInvalidate()

    def initThumb(self):
        vImage.initThumb(self)
        for layer in self.layersStack:
                vImage.initThumb(layer)

    def setThumbMode(self, value):
        if value == self.useThumb:
            return
        self.useThumb = value
        # recalculate the whole stack
        self.layerStack[0].apply()

    def updatePixmap(self):
        """
        Overrides vImage.updatePixmap()
        """
        vImage.updatePixmap(self)
        for layer in self.layersStack:
                vImage.updatePixmap(layer)


    def addLayer(self, layer, name='', index=None):
        """
        Adds layer.

        @param layer: layer to add (fresh layer if None, type QLayer)
        @param name:
        @param index: index of insertion in layersStack (top of active layer if index=None)
        @return: the layer added
        """
        # building a unique name
        usedNames = [l.name for l in self.layersStack]
        a = 1
        trialname = name if len(name) > 0 else 'noname'
        while trialname in usedNames:
            trialname = name + '_'+ str(a)
            a = a+1
        if layer is None:
            layer = QLayer(QImg=QImage(self))
            layer.fill(Qt.white)
        layer.name = trialname
        layer.parentImage = self
        if index==None:
            if self.activeLayerIndex is not None:
                # add on top of active layer if any
                index = self.activeLayerIndex
            else:
                # empty stack
                index = 0
        self.layersStack.insert(index, layer)
        self.setActiveLayer(index)
        layer.meta = self.meta
        layer.parentImage = self
        self.setModified(True)
        layer.initThumb()
        return layer

    def addAdjustmentLayer(self, name='', index=None):
        """
        Adds an adjustment layer to the layer stack, at
        position index (default is top of active layer)
        @param name: 
        @param index: 
        @return: 
        """
        if index == None:
            # add on top of active layer
            index = self.activeLayerIndex
        # adjust active layer only
        layer = QLayer.fromImage(self.layersStack[index], parentImage=self)
        # dynamic typing for adjustment layer
        layer.inputImg = lambda: self.layersStack[layer.getLowerVisibleStackIndex()].thumb if layer.parentImage.useThumb else self.layersStack[layer.getLowerVisibleStackIndex()]
        layer.inputImgFull = lambda: self.layersStack[layer.getLowerVisibleStackIndex()]
        self.addLayer(layer, name=name, index=index + 1)
        #layer.view = None
        #layer.parent = self
        #self.setModified(True)
        layer.initThumb()
        return layer

    def addSegmentationLayer(self, name='', index=None):
        if index == None:
            index = self.activeLayerIndex
        layer = QLayer.fromImage(self.layersStack[index], parentImage=self)
        layer.inputImg = lambda : self.layersStack[layer.getLowerVisibleStackIndex()]
        self.addLayer(layer, name=name, index=index + 1)
        layer.parent = self
        self.setModified(True)
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
        if index == None:
            index = len(self.layersStack) - 1
        layer0 = self.layersStack[index]
        if hasattr(layer0, 'inputImg'):
            return
        layer1 = QLayer.fromImage(layer0, parentImage=self)
        self.addLayer(layer1, name=layer0.name, index=index+1)

    def save(self, filename, quality=-1):
        """
        builds the resulting image from visible layers
        and writes it to file
        @param filename:
        @param quality: interger value in range 0..100
        @return: True if image is saved, False otherwise
        """

        img = QImage(self.width(), self.height(), self.format())
        img.fill(QColor(0,0,0,0))
        qp = QPainter(img)
        for layer in self.layersStack:
            if layer.visible:
                qp.setOpacity(layer.opacity)
                qp.setCompositionMode(layer.compositionMode)
                qp.drawImage(0,0, layer)
        qp.end()
        # save to file
        imgWriter = QImageWriter(filename)
        imgWriter.setQuality(quality)
        if imgWriter.write(img): #img.save(filename, quality=quality):
            self.setModified(False)
            return True
        else:
            msg = QMessageBox()
            msg.setWindowTitle('Warning')
            msg.setIcon(QMessageBox.Warning)
            msg.setText("cannot write file %s\n%s" % (filename, imgWriter.errorString()))
            msg.exec_()
            return False

class imImage(mImage) :
    """
    Interactive multi-layer image
    """
    def __init__(self, *args, **kwargs):
        super(imImage, self).__init__(*args, **kwargs)
        # Zoom_coeff = 1.0 displays an image fitting the
        # size of the current window ( NOT the actual pixels of the image).
        self.Zoom_coeff = 1.0
        self.xOffset, self.yOffset = 0, 0
        self.isMouseSelectable =True
        self.isModified = False

    def resize_coeff(self, widget):
        """
        return the current resizing coefficient. It is used by
        the paint event handler to display the image
        @param widget: Qwidget object
        @return: the (multiplicative) resizing coefficient
        """
        r_w, r_h = float(widget.width()) / self.width(), float(widget.height()) / self.height()
        r = max(r_w, r_h)
        return r * self.Zoom_coeff

    def resize(self, pixels, interpolation=cv2.INTER_CUBIC):
        """
        Resize image and layers
        @param pixels:
        @param interpolation:
        @return: resized imImage object
        """
        # resized vImage
        rszd0 = super(imImage, self).resize(pixels, interpolation=interpolation)
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
        self.Zoom_coeff, self.xOffset, self.yOffset = zoom, xOffset, yOffset

    def fit_window(self, win):
        self.Zoom_coeff = 1.0
        self.xOffset, self.yOffset = 0.0, 0.0

    def snapshot(self):
        """
        build a snapshot of image
        @return: imImage object
        """
        snap = imImage(QImg=self, meta=self.meta)
        #snap = QImage(self.width(), self.height(), QImage.Format_ARGB32)
        snap.fill(0)
        qp = QPainter(snap)
        for layer in self.layersStack:
            if layer.visible:
                # monitor profile is applied to all pixmaps, thus we should not build
                # images from pixmaps
                """
                #qp.setOpacity(layer.opacity)
                if layer.qPixmap is not None:
                    qp.drawPixmap(QRect(0, 0, self.width(), self.height()),  layer.transfer() )
                else:
                """
                qp.setOpacity(layer.opacity)
                qp.setCompositionMode(layer.compositionMode)
                qp.drawImage(QRect(0, 0, self.width(), self.height()),  layer)
        qp.end()
        # update background layer and call updatePixmap
        snap.layersStack[0].setImage(snap)
        return snap

class QLayer(vImage):
    @classmethod
    def fromImage(cls, mImg, parentImage=None):
        #mImg.visible = True
        #mImg.alpha = 255
        #mImg.window = None
        # for adjustment layer
        #mImg.transfer = lambda: mImg.qPixmap
        #mImg.inputImg = None
        layer = QLayer(QImg=mImg)
        layer.parentImage=parentImage
        layer.updatePixmap()
        return layer #QLayer(QImg=mImg) #mImg

    def __init__(self, *args, **kwargs):
        self.parentImage = None
        super(QLayer, self).__init__(*args, **kwargs)
        self.name='noname'
        self.visible = True
        # layer opacity is used by QPainter operations.
        # Its value must be in the range 0.0...1.0
        self.opacity = 1.0
        # default composition mode
        self.compositionMode = QPainter.CompositionMode_SourceOver
        self.transfer = lambda : self.qPixmap
        # Following attributes are used by adjustment layers only
        # wrapper for the right exec method
        #self.execute = lambda : 0
        self.execute = lambda : self.updatePixmap()
        self.temperature = 0
        self.options = {}
        # Following attributes (functions)  are reserved (dynamic typing for adjustment layers)
        # See addAdjustmentlayer above
            # self.inputImg : access to upper lower visible layer image or thumbnail, according to flag useThumb
            # self.inputImgFull : access to upper lower visible layer image
            # Accessing upper lower thumbnail must be done by calling inputImgFull().thumb. Using inputImg().thumb will fail if useThumb is True.

    def initThumb(self):
        """
        Overrides vImage.initThumb
        """
        if hasattr(self, 'inputImgFull'):
            self.thumb = QLayer(QImg=self.inputImgFull().scaled(self.thumbSize, self.thumbSize, Qt.KeepAspectRatio))
        else:
            self.thumb = QLayer(QImg=self.scaled(self.thumbSize, self.thumbSize, Qt.KeepAspectRatio))
        self.thumb.parentImage = self.parentImage

    def getCurrentImage(self):
        if self.parentImage.useThumb:
            if self.thumb is None:
                self.initThumb()
            currentImage = self.thumb
        else:
            currentImage = self
        return currentImage

    def getHspbBuffer(self):
        """
        returns the image buffer in color mode HSpB.
        The buffer is calculated if needed and cached.
        @return: HSPB buffer (type ndarray)
        """
        if self.hspbBuffer is None:
            if self.parentImage.useThumb:
                #self.hspbBuffer = rgb2hspVec(QImageBuffer(self.thumb)[:,:,:3][:,:,::-1])
                self.hspbBuffer = rgb2hspVec(QImageBuffer(self.inputImgFull().getCurrentImage())[:, :, :3][:, :, ::-1])
            else:
                if hasattr(self, 'inputImg'):
                    self.hspbBuffer = rgb2hspVec(QImageBuffer(self.inputImg())[:,:,:3][:,:,::-1])
                else:
                    self.hspbBuffer = rgb2hspVec(QImageBuffer(self)[:, :, :3][:, :, ::-1])
        return self.hspbBuffer

    def getLabBuffer(self):
        """
        returns the image buffer in color mode HSpB.
        The buffer is calculated if needed and cached.
        @return: HSPB buffer (type ndarray)
        """
        if self.LabBuffer is None:
            if self.parentImage.useThumb:
                self.LabBuffer = sRGB2LabVec(QImageBuffer(self.inputImgFull().getCurrentImage())[:, :, :3][:, :, ::-1])
            else:
                if hasattr(self, 'inputImg'):
                    self.LabBuffer = sRGB2LabVec(QImageBuffer(self.inputImg())[:,:,:3][:,:,::-1])
                else:
                    self.LabBuffer = sRGB2LabVec(QImageBuffer(self)[:, :, :3][:, :, ::-1])
        return self.LabBuffer

    def applyToStack(self):
        """
        Applies and propagates changes to upper layers.
        All event handlers changing parameters
        should call this method.
        """
        self.execute()
        ind = self.getStackIndex()
        if ind + 1 < len(self.parentImage.layersStack):
            img = self.parentImage.layersStack[ind + 1]
            img.cacheInvalidate()
            if img.visible:
                img.applyToStack()

        #self.apply = f #lambda : self.parentImage.layersStack[self.getStackIndex()+1].apply() if self.getStackIndex() +1 < len(self.parentImage.layersStack) else 0

    def updatePixmap(self):
        currentImage = self
        if self.parentImage is not None:
            if self.parentImage.useThumb:
                if self.thumb is not None:
                    currentImage = self.thumb
        if icc.COLOR_MANAGE and self.parentImage is not None:
            # layer color model is parent image colopr model
            img = convertQImage(currentImage, transformation=self.parentImage.colorTransformation)
        else:
            img = QImage(currentImage)
        if self.maskIsEnabled:
            qp=QPainter(img)
            qp.setOpacity(128)
            qp.drawImage(0,0,self.mask)
            qp.end()
        self.qPixmap = QPixmap.fromImage(img)

    def getStackIndex(self):
        for i, l in enumerate(self.parentImage.layersStack):
            if l is self:
                break
        return i

    def getLowerVisibleStackIndex(self):
        ind = self.getStackIndex()
        i = -1
        for i in range(ind-1, -1, -1):
            if self.parentImage.layersStack[i].visible:
                break
        return i

    def merge_with_layer_immediately_below(self):
        if not hasattr(self, 'inputImg'):
            return
        ind = self.getLowerVisibleStackIndex()
        if ind < 0:
            # no visible layer found
            return
        target = self.parentImage.layersStack[ind]
        if hasattr(target, 'inputImg') or self.parentImage.useThumb:
            # target is also an adjustment layer preview mode
            return

        #update stack
        self.parentImage.layersStack[0].applyToStack()
        target.setImage(self)
        #self.parentImage.activeLayerIndex = ind - 1
        self.parentImage.layerView.clear()
        currentIndex = self.getStackIndex()
        self.parentImage.activeLayerIndex = ind
        self.parentImage.layersStack.pop(currentIndex)
        self.parentImage.layerView.setLayers(self.parentImage)

    def setImage(self, qimg):
        """
        replace layer image with a copy of qimg buffer.
        The layer and qimg must have identical dimensions and type.
        @param qimg: QImage object
        """
        buf1, buf2 = QImageBuffer(self), QImageBuffer(qimg)
        if buf1.shape != buf2.shape:
            raise ValueError("QLayer.setImage : new image and layer must have identical shapes")
        buf1[...] = buf2
        self.updatePixmap()

    def reset(self):
        """
        reset layer by ressetting it to imputImg
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





