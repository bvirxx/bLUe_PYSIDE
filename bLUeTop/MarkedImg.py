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

import pickle
from io import BytesIO
from os import path

import numpy as np
import gc

from collections import OrderedDict

import tifffile

from PIL.ImageCms import ImageCmsProfile
from PySide6.QtCore import Qt, QSize, QPoint, QPointF, QFileInfo, QByteArray, QBuffer, QIODevice, QRectF, QTimer

import cv2
from copy import copy

from PySide6.QtGui import QTransform, QColor, QPixmap, QImage, QPainter
from PySide6.QtWidgets import QApplication, QSplitter
from PySide6.QtCore import QRect

import bLUeGui.blend
import bLUeTop
from bLUeCore.bLUeLUT3D import LUT3D
from bLUeCore.dwtDenoising import dwtDenoiseChan, dwtDenoise, chooseDWTLevel
from bLUeCore.kernel import filterIndex, getKernel
from bLUeCore.multi import chosenInterp
from bLUeGui.colorCIE import Lab2sRGBVec, bbTemperature2RGB, RGB2XYZ, rgbLinear2rgb, rgb2rgbLinear
from bLUeGui.colorCube import hsp2rgbVec, hsv2rgbVec
from bLUeGui.gradient import hsvGradientArray, setLUTfromGradient
from bLUeGui.histogramWarping import warpHistogram
from bLUeGui.multiplier import temperatureAndTint2Multipliers

from bLUeTop import exiftool
from bLUeGui.memory import weakProxy
from bLUeTop.align import alignImages
from bLUeTop.cloning import contours, moments, seamlessClone, alphaBlend

from bLUeTop.colorManagement import icc
from bLUeGui.bLUeImage import QImageBuffer, ndarrayToQImage, bImage
from bLUeGui.dialog import dlgWarn, dlgInfo, IMAGE_FILE_EXTENSIONS, RAW_FILE_EXTENSIONS, BLUE_FILE_EXTENSIONS
from time import time

from bLUeTop.graphicsBlendFilter import blendFilterIndex
from bLUeTop.lutUtils import LUT3DIdentity
from bLUeGui.baseSignal import baseSignal_bool, baseSignal_Int2, baseSignal_No
from bLUeTop.mergeImages import expFusion
from bLUeTop.rawProcessing import rawRead, rawPostProcess
from bLUeTop.settings import HAS_TORCH
from bLUeTop.utils import qColorToRGB, historyList, UDict

from bLUeTop.versatileImg import vImage

from version import BLUE_VERSION

if HAS_TORCH:
    from bLUeNN.classify import generateLUTfromQImage


class ColorSpace:
    notSpecified = -1
    sRGB = 1


class mImage(vImage):
    """
    Multi-layer image : base class for editable images.
    A mImage holds a presentation layer
    for color management and a stack containing at least a
    background layer. All layers share the same metadata instance.
    To correctly render a mImage, widgets should override their
    paint event handler.
    """

    @staticmethod
    def restoreMeta(srcFile, destFile, defaultorientation=True, thumbfile=None):
        """
        # copy metadata from sidecar into image file. The sidecar is not removed.
        If defaultorientation is True the orientation of the destination file is
        set to "no change" (1). In this way, saved images are displayed as they were edited.
        The thumbnail is updated if thumbfile is provided. Other metadata are kept unchanged.

        :param srcFile: source image or sidecar (the extension is replaced by .mie).
        :type srcFile: str
        :param destFile: image file
        :type destFile: str
        :param defaultorientation:
        :type defaultorientation: bool
        :param thumbfile: thumbnail file
        :type thumbfile: str
        """
        with exiftool.ExifTool() as e:
            e.copySidecar(srcFile, destFile)
            if defaultorientation:
                e.writeOrientation(destFile, '1')
            if thumbfile is not None:
                e.writeThumbnail(destFile, thumbfile)

    def __init__(self, *args, **kwargs):
        # as updatePixmap() uses layersStack, the latter must be initialized
        # before the call to super(). __init__()
        self.layersStack = []
        # link to QLayerView instance
        self.layerView = None
        super().__init__(*args, **kwargs)  # must be done before prLayer init.
        self.onActiveLayerChanged = lambda: 0
        # background layer
        bgLayer = QLayer.fromImage(self, parentImage=self)
        bgLayer.isClipping = True
        bgLayer.role = 'background'
        self.activeLayerIndex = None
        self.addLayer(bgLayer, name='Background')
        # presentation layer
        prLayer = QPresentationLayer(QImg=self, parentImage=self)
        prLayer.name = 'presentation'
        prLayer.role = 'presentation'
        prLayer.execute = lambda l=prLayer, pool=None: prLayer.applyNone()
        prLayer.updatePixmap()  # mandatory here as vImage init. can't do it
        self.prLayer = prLayer
        # link to rawpy instance
        self.rawImage = None
        self.sourceformat = ''  # needed to save as bLU file

    @property
    def colorTransformation(self):
        """
        Returns the current color transformation from working profile to monitor profile.

        :return:
        :rtype: Union[CmsTransform, QColorTransform]
        """
        return icc.workToMonTransform

    def copyStack(self, source):
        """
        Replaces layer stack, graphic forms and
        meta data by these from source.

        :param source:
        :type source: mImage
        """
        self.meta = source.meta
        self.onImageChanged = source.onImageChanged
        self.useThumb = source.useThumb
        self.useHald = source.useHald
        self.sRects = source.sRects.copy()
        for l in source.layersStack[1:]:
            lr = type(l).fromImage(l.scaled(self.size()),
                                   role=l.role,
                                   parentImage=self,
                                   sourceImg=l.sourceImg.scaled(self.size())
                                             if getattr(l, 'sourceImg', None) is not None else None
                                   )
            lr.execute = l.execute
            lr.actionName = l.actionName
            lr.name = l.name
            lr.view = l.view
            lr.view.widget().targetImage = self
            lr.view.widget().layer = lr

            if l._mask is not None:
                lr._mask = l._mask.scaled(self.size())

            lr.maskIsSelected = l.maskIsSelected
            lr.maskIsEnabled = l.maskIsEnabled

            sfi = getattr(l, 'sourceFromFile', None)
            if sfi is not None:
                # cloning layer only
                coeffX, coeffY = lr.width() / l.width(), lr.height() / l.height()
                lr.sourceFromFile = l.sourceFromFile
                lr.cloningMethod = l.cloningMethod
                lr.sourceX, lr.sourceY = l.sourceX * coeffX, l.sourceY * coeffY
                lr.xAltOffset, lr.yAltOffset = l.xAltOffset * coeffX, l.yAltOffset * coeffY
                lr.cloningState = l.cloningState
                lr.getGraphicsForm().updateSource()

            tool = getattr(l, 'tool', None)
            if tool:
                lr.addTool(tool)
                tool.showTool()

            self.layersStack.append(lr)

    def resized(self, w, h, keepAspectRatio=True, interpolation=cv2.INTER_CUBIC):
        """
        Resizes the image and the layer stack, while keeping the aspect ratio.

        :param interpolation:
        :return: resized imImage object
        :rtype: same type as self
        """
        rszd0, buf = super().resized(w, h, keepAspectRatio=keepAspectRatio, interpolation=interpolation)
        # get resized image (with a background layer)
        rszd = type(self)(QImg=rszd0, meta=copy(self.meta))
        rszd.__buf = buf  # protect buf from g.c.
        rszd.copyStack(self)
        return rszd

    def bTransformed(self, transformation):
        """
        Applies transformation to all layers in stack and returns
        the new mImage.

        :param transformation:
        :type transformation: Qtransform
        :return:
        :rtype: mimage
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

        :return: The active layer
        :rtype: QLayer
        """
        if self.activeLayerIndex is not None:
            return self.layersStack[self.activeLayerIndex]
        else:
            return None

    def setActiveLayer(self, stackIndex):
        """
        Assigns stackIndex value to  activeLayerIndex and
        updates the layer view and tools.

        :param stackIndex: index in stack for the layer to select
        :type stackIndex: int
        :return: active layer
        :rtype: QLayer
        """
        lgStack = len(self.layersStack)
        if stackIndex < 0 or stackIndex >= lgStack:
            return

        layer = self.layersStack[stackIndex]
        self.activeLayerIndex = stackIndex

        # update controls in tableView
        if self.layerView is not None:
            self.layerView.maskLabel.setEnabled(layer.maskIsSelected)
            self.layerView.maskSlider.setEnabled(layer.maskIsSelected)
            self.layerView.maskValue.setEnabled(layer.maskIsSelected)
            # update opacity and composition mode for current layer
            opacity = int(layer.opacity * 100)
            self.layerView.opacityValue.setText(str('%d ' % opacity))
            self.layerView.opacitySlider.setSliderPosition(opacity)
            compositionMode = layer.compositionMode
            ind = self.layerView.blendingModeCombo.findData(compositionMode)
            self.layerView.blendingModeCombo.setCurrentIndex(ind)
            self.layerView.selectRow(lgStack - 1 - stackIndex)

        # cleaning
        for lay in self.layersStack:
            currentWin = getattr(lay, 'view', None)
            if currentWin is not None:
                # hide sucontrols
                for dk in currentWin.widget().subControls:
                    dk.hide()
                if currentWin.isFloating():
                    currentWin.hide()
                if lay.tool is not None:
                    lay.tool.hideTool()
        # setting
        currentWin = getattr(layer, 'view', None)
        if currentWin is not None and layer.visible:
            currentWin.show()
            currentWin.raise_()
            # display subcontrols
            for dk in currentWin.widget().subControls:
                dk.setVisible(currentWin.widget().options[dk.widget().optionName])
            # make self.currentWin the active window
            currentWin.activateWindow()

        # if layer.tool is not None:
        # layer.tool.moveRotatingTool()  # TODO added 23/11/21 keep last

        # active = self.getActiveLayer()
        if layer.tool is not None and layer.visible:
            layer.tool.showTool()
        self.onActiveLayerChanged()
        return layer

    def getActivePixel(self, x, y, fromInputImg=True, qcolor=False):
        """
        Reads the RGB colors of the pixel at (x, y) from the active layer.
        If fromInputImg is True (default), the pixel is taken from
        the input image, otherwise from the current image.
        Coordinates are relative to the full sized image.
        If (x,y) is outside the image, (0, 0, 0) is returned.

        :param x: x-coordinates of pixel, relative to the full-sized image
        :type x: int
        :param y: y-coordinates of pixel, relative to the full-sized image
        :type y: int
        :param fromInputImg:
        :type fromInputImg:
        :param qcolor:
        :type qcolor:
        :return: color of pixel if qcolor else its R, G, B components
        :rtype: QColor if qcolor else 3-uple of int
        """
        x, y = self.full2CurrentXY(x, y)
        activeLayer = self.getActiveLayer()
        qClr = activeLayer.inputImg(redo=False).pixelColor(x, y) if fromInputImg \
            else activeLayer.getCurrentImage().pixelColor(x, y)
        # pixelColor returns an invalid color if (x,y) is out of range
        # we return black
        if not qClr.isValid():
            qClr = QColor(0, 0, 0)
        return qClr if qcolor else qColorToRGB(qClr)

    def getPrPixel(self, x, y):
        """
        Reads the RGB colors of the pixel at (x, y) from
        the presentation layer. They are the (non color managed)
        colors of the displayed pixel.
        Coordinates are relative to the full sized image.
        If (x,y) is outside the image, (0, 0, 0) is returned.

        :param x: x-coordinate of pixel, relative to the full-sized image
        :type x: int
        :param y: y-coordinate of pixel, relative to the full-sized image
        :type y: int
        :return: pixel RGB colors
        :rtype: 3-uple of int
        """
        x, y = self.full2CurrentXY(x, y)
        qClr = self.prLayer.getCurrentImage().pixelColor(x, y)
        if not qClr.isValid():
            qClr = QColor(0, 0, 0)
        return qColorToRGB(qClr)

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
        self.layersStack[0].apply()

    def updatePixmap(self):
        """
        Update the presentation layer only.
        Used when a change in color management occurs
        """
        self.prLayer.updatePixmap()

    def getStackIndex(self, layer):
        p = id(layer)
        i = -1
        for i, l in enumerate(self.layersStack):
            if id(l) == p:
                break
        return i

    def addLayer(self, layer, name='noname', index=None, activate=True):
        """
        Adds a layer to stack (If the parameter layer is None a fresh layer is added).
        The layer name may be modified to get a unique (case-insensitive) name.
        The layer is returned.

        :param layer: layer to add (if None, add a fresh layer)
        :type layer: QLayer
        :param name: layer proposed name
        :type name: str
        :param index: index of insertion in layersStack (top of active layer if index=None)
        :type index: int
        :param activate:
        :type activate: boolean
        :return: the layer added
        :rtype: QLayer
        """
        # build a unique name
        usedNames = [l.name.lower() for l in self.layersStack]
        a = 1
        trialname = name.lower() if len(name) > 0 else 'noname'
        while trialname.lower() in usedNames:
            trialname = name + '_' + str(a)
            a = a + 1
        if layer is None:
            layer = QLayer(QImg=self, parentImage=self)
            layer.fill(Qt.white)
        layer.name = trialname
        if index is None:
            if self.activeLayerIndex is not None:
                # add on top of active layer if any
                index = self.activeLayerIndex + 1
            else:
                # empty stack
                index = 0
        self.layersStack.insert(index, layer)
        if activate:
            self.setActiveLayer(index)
        layer.meta = self.meta
        layer.parentImage = weakProxy(self)
        self.setModified(True)
        return layer

    def removeLayer(self, index=None):
        if index is None:
            return
        self.layersStack.pop(index)

    def addAdjustmentLayer(self, layerType=None, name='', role='', index=None, sourceImg=None):
        """
        Adds an adjustment layer to the layer stack, at
        position index (default is top of active layer).
        The parameter layerType controls the class of the layer; it should
        be a subclass of QLayer, default is QLayer itself.
        If the parameter sourceImg is given, the layer is a
        QLayerImage object built from sourceImg, and layerType has no effect.

       :param layerType: layer class
       :type layerType: QLayer subclass
       :param name:
       :type name: str
       :param role:
       :type role: str
       :param index:
       :type index: int
       :param sourceImg: source image
       :type sourceImg: QImage
       :return: the new layer
       :rtype: subclass of QLayer
        """
        if index is None:
            # adding on top of active layer
            index = self.activeLayerIndex
        if sourceImg is None:
            if layerType is None:
                layerType = QLayer
            layer = layerType.fromImage(self.layersStack[index], parentImage=self)
        else:
            # set layer from image :
            if self.size() != sourceImg.size():
                sourceImg = sourceImg.scaled(self.size())
            if layerType is None:
                layerType = QLayerImage
            layer = layerType.fromImage(self.layersStack[index], parentImage=self, role=role, sourceImg=sourceImg)
        layer.role = role
        self.addLayer(layer, name=name, index=index + 1)
        # init thumb or QImage itself for correct display before first call to applyToStack()
        if layer.parentImage.useThumb:
            layer.thumb = layer.inputImg().copy()
        else:
            QImageBuffer(layer)[...] = QImageBuffer(layer.inputImg())
        """
        group = self.layersStack[index].group
        if group:
            layer.group = group
            layer.mask = self.layersStack[index].mask
            layer.maskIsEnabled = True
        """
        # sync caches
        layer.updatePixmap()
        return layer

    def addSegmentationLayer(self, name='', index=None):
        if index is None:
            index = self.activeLayerIndex
        layer = QLayer.fromImage(self.layersStack[index], parentImage=self)
        layer.role = 'SEGMENT'
        self.addLayer(layer, name=name, index=index + 1)
        layer.maskIsEnabled = True
        layer.maskIsSelected = True
        # mask pixels are not yet painted as FG or BG
        # so we mark them as invalid
        layer.mask.fill(vImage.defaultColor_Invalid)
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

        :param index:
        :type index: int
        :return:
        :rtype:
        """
        if index is None:
            index = len(self.layersStack) - 1
        layer0 = self.layersStack[index]
        if layer0.isAdjustLayer():
            return
        layer1 = QLayer.fromImage(layer0, parentImage=self)
        self.addLayer(layer1, name=layer0.name, index=index + 1)

    def mergeVisibleLayers(self):
        """
        Merges the visible masked images and returns the
        resulting QImage, eventually scaled to fit the image size.

        :return: image
        :rtype: QImage
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

    def snap(self, thumb_jpg):
        """
        Builds a snapshot of the document (imagej description dictionary, mask list, image list) for
        saving to bLU file. All changes to the document made between the call to
        snap() and the actual file writing will be ignored.

        :param thumb_jpg: image thumbnail (jpg)
        :type thumb_jpg: QByteArray
        :return:
        :rtype: 3uple (ordered dict, list of Qimage, list of QImage)
        """
        layernames = [(layer.name, pickle.dumps({'actionname': layer.actionName, 'state': layer.__getstate__()}))
                      for layer in self.layersStack] + \
                     [('sourceformat', self.sourceformat)] + \
                     [('version', BLUE_VERSION)] + \
                     [('cropmargins', pickle.dumps(self.cropMargins()))] + \
                     [('ThumbnailImage', pickle.dumps(thumb_jpg))]

        names = OrderedDict(layernames)  # values are not pickled str or pickled dict or tuple

        if len(names) != len(layernames):
            # search for duplicate names
            tmplist = [x[0] for x in layernames]
            duplicates = []
            for item in tmplist:
                if tmplist.count(item) > 1:
                    duplicates.append(item)
            if duplicates:
                dlgWarn('Duplicate name(s) %s' % duplicates, 'Please edit the names in layer stack')
            else:
                dlgWarn('Cannot build dictionary', 'Unknown error')
            raise IOError('Cannot save document')

        ####################################################################################################
        # tifffile calls imagej_description() internally with a wrong signature. Due to this bug
        # an exception is raised if dictionary keys contain the string 'rgb'. As a workaround
        # a layer named 'rgb' should be renamed by editing the corresponding layer stack item.
        if 'rgb' in names:
            dlgWarn('"rgb" is not allowed as a layer name', 'Please rename the layer')
            raise IOError('Cannot save document')
        ####################################################################################################

        mask_list = [layer._mask for layer in self.layersStack if layer._mask is not None]

        image_list = []
        for layer in self.layersStack:
            for aname in layer.innerImages:
                im = getattr(layer, aname, None)
                if im is not None:
                    image_list.append(im)

        return names, mask_list, image_list

    def save(self, filename, quality=-1, compression=-1):
        """
        Overrides QImage.save().
        If filename extension is a standard image format, the method writes
        the presentation layer to filename and returns a
        thumbnail with standard size (160x120 or 120x160). Parameters quality and compression
        are passed to the writer.
        If filename extension is 'bLU' the current state (stack, masks, images) of the document
        is saved to filename.
        Raises IOError if the saving fails.

        :param filename:
        :type filename: str
        :param quality: integer value in range 0..100, or -1
        :type quality: int
        :param compression: integer value in range 0..100, or -1
        :type compression: int
        :return: thumbnail of the saved image
        :rtype: QImage
        """

        def transparencyCheck(buf, fileformat):
            if fileFormat.upper() not in ['.JPG', '.TIF']:
                return
            if np.any(buf[:, :, 3] < 255):
                dlgWarn('Transparency will be lost. Use PNG format instead')

        fileFormat = filename[-4:].upper()

        # get the final image from the presentation layer.
        # This image is NOT color managed (only prLayer.qPixmap
        # is color managed)
        img = self.prLayer.getCurrentImage()

        # imagewriter and QImage.save are unusable for tif files,
        # due to bugs in libtiff, hence we use opencv imwrite.
        buf = QImageBuffer(img)
        if self.isCropped:
            # make slices
            w, h = self.width(), self.height()
            wp, hp = img.width(), img.height()
            wr, hr = wp / w, hp / h
            w1, w2 = int(self.cropLeft * wr), int((w - self.cropRight) * wr)
            h1, h2 = int(self.cropTop * hr), int((h - self.cropBottom) * hr)
            buf = buf[h1:h2, w1:w2, :]

        # build thumbnail from (eventually) cropped image
        # choose thumb size
        wf, hf = buf.shape[1], buf.shape[0]
        if wf > hf:
            wt, ht = 160, 120
        else:
            wt, ht = 120, 160
        thumb = ndarrayToQImage(np.ascontiguousarray(buf[:, :, :3][:, :, ::-1]),
                                format=QImage.Format_RGB888).scaled(wt, ht, Qt.KeepAspectRatio)

        # build jpg from thumb
        ba = QByteArray()
        buffer = QBuffer(ba)
        buffer.open(QIODevice.WriteOnly)
        thumb.save(buffer, 'JPG')

        transparencyCheck(buf, fileFormat)

        params = []
        if fileFormat == '.JPG':
            buf = buf[:, :, :3]
            if quality >= 0 and quality <= 100:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]  # quality range 0..100
        elif fileFormat == '.PNG':
            if compression >= 0 and compression <= 9:
                params = [cv2.IMWRITE_PNG_COMPRESSION, compression]  # compression range 0..9
        elif fileFormat in ['.TIF'] + list(BLUE_FILE_EXTENSIONS):
            buf = buf[:, :, :3]
        else:
            raise IOError("Invalid File Format\nValid formats are jpg, png, tif ")

        written = False

        if fileFormat in IMAGE_FILE_EXTENSIONS:  # dest format
            # save edited image -  mode preview is off
            written = cv2.imwrite(filename, buf, params)  # BGR order

        elif fileFormat in BLUE_FILE_EXTENSIONS:
            # records current state and save to bLU file
            names, mask_list, image_list = self.snap(ba)

            if self.sourceformat in RAW_FILE_EXTENSIONS:
                # copy raw file and layer stack to .bLU
                originFormat = self.filename[-4:]  # format of opened document
                if originFormat in BLUE_FILE_EXTENSIONS:  # format of source file
                    with tifffile.TiffFile(self.filename) as tfile:  # raw image will be copied from source file
                        sourcedata = tfile.series[0].pages[0].asarray()
                        buf_ori = sourcedata[0]  # [:, 0]
                elif originFormat in RAW_FILE_EXTENSIONS:
                    with open(self.filename, 'rb') as f:
                        bytes = f.read()
                    buf_ori = np.frombuffer(bytes, dtype=np.uint8)

                w, h = self.width(), self.height()
                images = np.empty((len(mask_list) + len(image_list) + 1, max(len(buf_ori), w * h * 4)), dtype=np.uint8)

                images[0, :len(buf_ori)] = buf_ori

                i = 1
                for m in mask_list + image_list:
                    if m is not None:
                        b = QImageBuffer(m)
                        images[i, :w * h * 4] = b.ravel()
                        i += 1

                names['mask_len'] = w * h * 4
                names['buf_ori_len'] = len(buf_ori)

                result = tifffile.imwrite(filename,
                                          data=images,
                                          # compression=6,
                                          compression='zlib',
                                          # compressionargs = {'level': 6},
                                          imagej=True,
                                          returnoffset=True,
                                          metadata=names
                                          )
                written = True  # with compression > 0 result is None

            elif self.sourceformat in IMAGE_FILE_EXTENSIONS or self.sourceformat == '':  # format == '' for new document
                # copy source image and layer stack to .BLU.
                img_ori = self

                w, h = self.width(), self.height()
                images = np.empty((len(mask_list) + len(image_list) + 1, h, w, 4), dtype=np.uint8)

                buf_ori = QImageBuffer(img_ori).copy()
                # BGRA to RGBA conversion needed : to reload image
                # the bLU file will be read as tiff file by QImageReader
                tmpview = buf_ori[..., :3]
                tmpview[...] = tmpview[..., ::-1]
                images[0, ...] = buf_ori

                i = 1
                for m in mask_list + image_list:
                    if m is not None:
                        images[i, ...] = QImageBuffer(m)
                        i += 1

                result = tifffile.imwrite(filename,
                                          data=images,
                                          compression='zlib',
                                          # compressionargs = {'level': 6},
                                          imagej=True,
                                          returnoffset=True,
                                          metadata=names
                                          )

                written = True  # with compression result is None
            """
            else:
                # invalid sourceformat
                written = False
            """
        """
        else:
            # invalid extension
            written = False
        """

        if not written:
            raise IOError("Cannot write file %s " % filename)

        return thumb


class imImage(mImage):
    """
    Zoomable and draggable multi-layer image :
    this is the base class for bLUe documents
    """

    @staticmethod
    def loadImageFromFile(f, rawiobuf=None, createsidecar=True, icc=icc, cmsConfigure=False, window=None):
        """
        load an imImage (image and base metadata) from file. Returns the loaded imImage :
        For a raw file, it is the image postprocessed with default parameters.

        :param f: path to file
        :type f: str
        :param rawiobuf: raw data
        :type rawiobuf:
        :param createsidecar:
        :type createsidecar: boolean
        :param icc:
        :type icc: class icc
        :param cmsConfigure:
        :type cmsConfigure: boolean
        :return: image
        :rtype: imImage
        """
        # extract metadata from sidecar (.mie) if it exists, otherwise from image file.
        # metadata is a dict.
        # The sidecar is created if it does not exist and createsidecar is True.
        try:
            with exiftool.ExifTool() as e:
                profile, metadata = e.get_metadata(f,
                                                   tags=("colorspace",
                                                         "profileDescription",
                                                         "orientation",
                                                         "model",
                                                         "rating"),
                                                   createsidecar=createsidecar)
                imageInfo = e.get_formatted_metadata(f)
        except ValueError:
            # Default metadata and profile
            metadata = {'SourceFile': f}
            profile = b''
            imageInfo = 'No data found'

        # try to find a valid color space : 1=sRGB 65535=uncalibrated
        tmp = [value for key, value in metadata.items() if 'colorspace' in key.lower()]
        colorSpace = tmp[0] if tmp else -1

        # try to find a valid embedded profile.
        # If everything fails, assign sRGB.
        cmsProfile = icc.defaultWorkingProfile
        try:
            # convert embedded profile to ImageCmsProfile object
            cmsProfile = ImageCmsProfile(BytesIO(profile))
        except (TypeError, OSError) as e:  # both raised by ImageCmsProfile()
            pass
        if not isinstance(cmsProfile, ImageCmsProfile):
            dlgInfo("Color profile is missing\nAssigning sRGB")
            # assign sRGB profile
            colorSpace = 1
            cmsProfile = icc.defaultWorkingProfile
        if cmsConfigure:
            # update the color management object with the image profile.
            icc.configure(colorSpace=colorSpace, workingProfile=cmsProfile)
        # orientation
        tmp = [value for key, value in metadata.items() if 'orientation' in key.lower()]
        orientation = tmp[0] if tmp else 0  # metadata.get("EXIF:Orientation", 0)
        transformation = exiftool.decodeExifOrientation(orientation)
        # rating
        tmp = [value for key, value in metadata.items() if 'rating' in key.lower()]
        rating = tmp[0] if tmp else 0  # metadata.get("XMP:Rating", 5)

        ############
        # load image
        ############
        name = path.basename(f)
        ext = name[-4:]
        if (ext in list(IMAGE_FILE_EXTENSIONS) + list(BLUE_FILE_EXTENSIONS)) and rawiobuf is None:
            # standard image file or .blu from image
            img = imImage(filename=f, colorSpace=colorSpace, orientation=transformation, rawMetadata=metadata,
                          profile=profile, name=name, rating=rating)
        elif ext in list(RAW_FILE_EXTENSIONS) + list(BLUE_FILE_EXTENSIONS):
            # load raw image file in a RawPy instance
            if rawiobuf is None:
                # standard raw file
                rawpyInst = rawRead(f)
            else:
                # .blu from raw
                rawpyInst = rawRead(rawiobuf)

            # postprocess raw image, applying default settings
            rawBuf = rawpyInst.postprocess(use_camera_wb=True)

            # build imImage : switch to BGR and add alpha channel
            rawBuf = np.dstack((rawBuf[:, :, ::-1], np.zeros(rawBuf.shape[:2], dtype=np.uint8) + 255))
            img = imImage(cv2Img=rawBuf, colorSpace=colorSpace, orientation=transformation,
                          rawMetadata=metadata, profile=profile, name=name, rating=rating)

            # keeping a reference to rawBuf along with img is
            # needed to protect the buffer from garbage collector
            img.rawBuf = rawBuf
            img.filename = f
            img.rawImage = rawpyInst
            """
            # get 16 bits Bayer bitmap
            img.demosaic = demosaic(rawpyInst.raw_image_visible, rawpyInst.raw_colors_visible,
                                    rawpyInst.black_level_per_channel)
            # correct orientation
            if orientation == 6:  # 90°
                img.demosaic = np.swapaxes(img.demosaic, 0, 1)
            elif orientation == 8:  # 270°
                img.demosaic = np.swapaxes(img.demosaic, 0, 1)
                img.demosaic = img.demosaic[:, ::-1, :]
            """
        else:
            raise ValueError("Cannot read file %s" % f)

        if img.isNull():
            raise ValueError("Cannot read file %s" % f)
        if img.format() in [QImage.Format_Invalid, QImage.Format_Mono, QImage.Format_MonoLSB, QImage.Format_Indexed8]:
            raise ValueError("Cannot edit indexed formats\nConvert image to a non indexed mode first")
        img.imageInfo = imageInfo
        window.settings.setValue('paths/dlgdir', QFileInfo(f).absoluteDir().path())
        img.initThumb()

        # update profile related attributes
        img.setProfile(cmsProfile)

        return img

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.savedBtnValues = {}  # saving of app button states for multi-docs edition
        # Zoom coeff :
        # Zoom_coeff = 1.0 displays an image fitting the
        # size of the current window ( NOT the actual pixels of the image).
        self.Zoom_coeff = 1.0
        self.xOffset, self.yOffset = 0, 0
        self.isMouseSelectable = True
        self.isModified = False

    def resize_coeff(self, widget):
        """
        Normalization of self.Zoom_coeff.
        Return the current resizing coefficient used by
        the widget paint event handler to display the image.
        The coefficient is chosen to initially (i.e. when self.Zoom_coeff = 1)
        fill the widget without cropping.
        For split views we use the size of the QSplitter parent
        container instead of the size of the widget.

       :param widget:
       :type widget: imageLabel
       :return: the (multiplicative) resizing coefficient
       :rtype: float
        """
        if widget.window.asButton.isChecked():
            # actual size
            return 1.0
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

    def bTransformed(self, transformation):
        """
        Applies transformation to all layers in stack
        and returns the new imImage.

        :param transformation:
        :type transformation: QTransform
        :return:
        :rtype: imImage
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
                grForm = tLayer.getGraphicsForm()
                # the grForm.layer property handles weak refs
                grForm.layer = tLayer
                if getattr(grForm, 'scene', None) is not None:
                    grForm.scene().layer = grForm.layer  # wtLayer
            stack.append(tLayer)
        img.layersStack = stack
        return img

    def view(self):
        return self.Zoom_coeff, self.xOffset, self.yOffset

    def setView(self, zoom=1.0, xOffset=0.0, yOffset=0.0):
        """
        Sets viewing conditions: zoom, offset.

        :param zoom: zoom coefficient
        :type zoom: float
        :param xOffset: x-offset
        :type xOffset: int
        :param yOffset: y-offset
        :type yOffset: int
        :return:
        """
        self.Zoom_coeff, self.xOffset, self.yOffset = zoom, xOffset, yOffset

    def fit_window(self):
        """
        reset Zoom_coeff and offset
        """
        self.Zoom_coeff = 1.0
        self.xOffset, self.yOffset = 0.0, 0.0


class QLayer(vImage):
    """
    Base class for image layers
    """

    @classmethod
    def fromImage(cls, mImg, role='', parentImage=None, **kwargs):
        """
        Return a QLayer object initialized with mImg.
        Derived classes get an instance of themselves
        without overriding.

        :param mImg:
        :type mImg: QImage
        :param parentImage:
        :type parentImage: mImage
        :return:
        :rtype: Qlayer
        """
        layer = cls(QImg=mImg, role=role, parentImage=parentImage)
        return layer

    def __init__(self, *args, **kwargs):
        ############################################################
        # Signals :
        # Making QLayer inherit from QObject leads to
        # a bugged behavior of hasattr and getattr.
        # So, we don't add signals as first level class attributes.
        # Instead, we use instances of ad hoc signal containers.
        ############################################################
        self.visibilityChanged = baseSignal_bool()
        self.colorPicked = baseSignal_Int2()
        self.selectionChanged = baseSignal_No()
        self.maskSettingsChanged = baseSignal_No()
        ###########################################################
        # when a geometric transformation is applied to the whole image
        # each layer must be replaced with a transformed layer, recorded in tLayer
        # and tLayer.parentLayer keeps a reference to the original layer.
        ###########################################################
        self.tLayer = self
        self.parentLayer = self
        self.modified = False
        self.name = 'noname'
        self.visible = True
        self.isClipping = False
        self.role = kwargs.pop('role', '')
        # add autoSpline attribute to contrast layer only
        if self.role in ['CONTRAST', 'RAW']:
            self.autoSpline = True
        self.tool = None
        parentImage = kwargs.pop('parentImage', None)
        super().__init__(*args, **kwargs)  # don't move
        # init back link to parent image : don't move
        self.parentImage = weakProxy(parentImage)

        # layer opacity, range 0.0...1.0
        self.opacity = 1.0
        # compositionMode type is QPainter.CompositionMode enum or int for modes added by bLUe
        self.compositionMode = QPainter.CompositionMode_SourceOver
        ###################################################################################
        # QLayer is not always subclassed to define multiple types of adjustment layers.
        # Instead, we may use the attribute execute as a wrapper to the right applyXXX method,
        # depending on the intended "type" of layer.
        # The method execute must call updatePixmap().
        ##################################################################################
        self.execute = lambda l=None, pool=None: l.updatePixmap() if l is not None else None
        self.options = {}
        self.actionName = 'actionNull'
        # view is the dock widget containing
        # the graphics form associated with the layer
        self.view = None
        # undo/redo mask history
        self.historyListMask = historyList(size=5)
        # list of auxiliary images to be saved/restored (drawing, cloning, merging,...)
        self.innerImages = ()
        # layer offsets
        self.xOffset, self.yOffset = 0, 0
        self.Zoom_coeff = 1.0
        # clone/dup virtual layer shift and zoom (relative to the full sized image)
        self.xAltOffset, self.yAltOffset = 0, 0
        self.sourceX, self.sourceY = 0, 0
        self.AltZoom_coeff = 1.0
        self.updatePixmap()

    @property
    def mask(self):
        if self._mask is None:
            if type(self) not in [QPresentationLayer]:
                self._mask = QImage(self.width(), self.height(), QImage.Format_ARGB32)
                # default : unmask all
                self._mask.fill(self.defaultColor_UnMasked)
        return self._mask

    @mask.setter
    def mask(self, m):  # the setter is NOT inherited from bImage
        self._mask = m

    def getGraphicsForm(self):
        """
        Return the graphics form associated with the layer.

        :return:
        :rtype: QWidget
        """
        if self.view is not None:
            return self.view.widget()
        return None

    def closeView(self, delete=False):
        """
        Closes all windows associated with layer.

        :param delete:
        :type delete: boolean
        """

        def closeDock(dock, delete=False):
            if dock is None:
                return
            if delete:
                form = dock.widget()
                # break back link
                if hasattr(form, 'layer'):
                    form.layer = None
                form.setAttribute(Qt.WA_DeleteOnClose)
                form.close()
                form.__dict__.clear()  # TODO awful - prepare for gc - probably useless test needed 30/11/21 validate
                dock.setAttribute(Qt.WA_DeleteOnClose)
                dock.setParent(None)
                dock.close()
                dock.__dict__.clear()  # TODO awful - prepare for gc - probably useless test needed 30/11/21 validate
                # self.view = None  # TODO removed 29/11/21 validate
            else:  # tabbed forms should not be closed
                temp = dock.tabbed
                dock.setFloating(True)
                dock.tabbed = temp  # remember last state to restore
                # window.removeDockWidget(dock)
                dock.hide()

        view = getattr(self, 'view', None)
        if view is None:
            return
        # close all subwindows
        form = self.view.widget()
        for dock in form.subControls:
            closeDock(dock, delete=delete)
        # close window
        closeDock(view, delete=delete)
        if delete:  # TODO modified 29/11/21 validate
            form.subControls = []
            self.execute = None  # TODO prepare for gc  probably useless test needed 30/11/21 validate
            self.view = None
            self.__dict__.clear()

    def isActiveLayer(self):
        if self.parentImage.getActiveLayer() is self:
            return True
        return False

    def getMmcSpline(self):
        """
        Returns the spline used for multimode contrast
        correction if it is initialized, and None otherwise.

        :return:
        :rtype: activeSpline
        """
        # get layer graphic form
        grf = self.getGraphicsForm()
        # manual curve form
        if grf.contrastForm is not None:
            return grf.contrastForm.scene().cubicItem
        return None

    def addTool(self, tool):
        """
        Adds tool to layer.

        :param tool:
        :type tool: rotatingTool
        """
        self.tool = tool
        tool.modified = False
        tool.layer = weakProxy(self)  # TODO added weakProxy 28/11/21 validate
        try:
            tool.layer.visibilityChanged.sig.disconnect()
        except RuntimeError:
            pass
        tool.layer.visibilityChanged.sig.connect(tool.setVisible)
        tool.img = weakProxy(self.parentImage)  # TODO added weakProxy 28/11/21 validate
        w, h = tool.img.width(), tool.img.height()
        for role, pos in zip(['topLeft', 'topRight', 'bottomRight', 'bottomLeft'],
                             [QPoint(0, 0), QPoint(w, 0), QPoint(w, h), QPoint(0, h)]):
            tool.btnDict[role].posRelImg = pos
            tool.btnDict[role].posRelImg_ori = pos
            tool.btnDict[role].posRelImg_frozen = pos
        tool.moveRotatingTool()

    def setVisible(self, value):
        """
        Sets self.visible to value and emit visibilityChanged.sig.

        :param value:
        :type value: bool
        """
        self.visible = value
        self.visibilityChanged.sig.emit(value)

    def bTransformed(self, transformation, parentImage):
        """
        Apply transformation to a copy of layer. Returns the transformed copy.

        :param transformation:
        :type transformation: QTransform
        :param parentImage:
        :type parentImage: mImage
        :return: transformed layer
        :rtype: QLayer
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
        tLayer.visible = self.visible
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
        s = int(LUT3DIdentity.size ** (3.0 / 2.0)) + 1
        buf0 = LUT3DIdentity.toHaldArray(s, s).haldBuffer
        # self.hald = QLayer(QImg=QImage(QSize(190,190), QImage.Format_ARGB32))
        self.hald = QImage(QSize(s, s), QImage.Format_ARGB32)
        buf1 = QImageBuffer(self.hald)
        buf1[:, :, :3] = buf0
        buf1[:, :, 3] = 255
        self.hald.parentImage = self.parentImage

    def getCurrentImage(self):
        """
        Returns current (full, preview or hald) image, according to
        the value of the flags useThumb and useHald. The thumbnail and hald
        are computed if they are not initialized.
        Otherwise, they are not updated unless self.thumb is
        None or purgeThumb is True.
        Overrides vImage method.

        :return: current image
        :rtype: QLayer
        """
        if self.parentImage.useHald:
            return self.getHald()
        if self.parentImage.useThumb:
            return self.getThumb()
        else:
            return self

    def inputImg(self, redo=True):
        """
        return maskedImageContainer/maskedThumbContainer.
        If redo is True(default), containers are updated and
        their rPixmap are invalidated to force refreshing.
        layer.applyToStack() always calls inputImg() with redo=True.
        So, to keep the containers up to date we only have to follow
        each layer modification by a call to layer.applyToStack().

        :param redo:
        :type redo: boolean
        :return:
        :rtype: bImage
        """
        lower = self.parentImage.layersStack[self.getLowerVisibleStackIndex()]
        container = lower.maskedThumbContainer if self.parentImage.useThumb else lower.maskedImageContainer
        if redo or container is None:
            # update container and invalidate rPixmap
            container = lower.getCurrentMaskedImage()
            container.rPixmap = None
        else:
            # don't update container and rebuild invalid rPixmap only
            if container.rPixmap is None:
                container.rPixmap = QPixmap.fromImage(container)
        return container

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
        if self.parentImage.useThumb:
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
        if self.parentImage.useThumb:
            currentImg = self.getThumb()
            x = (x * self.width()) / currentImg.width()
            y = (y * self.height()) / currentImg.height()
        return int(x), int(y)

    def getCurrentMaskedImage(self):
        """
        Blend the layer stack up to self (included),
        taking into account the masks. The method uses the
        non color managed rPixmap to build the masked image.
        For convenience, mainly to be able to use its color space buffers,
        the built image is of type bImage. It is drawn on a container image,
        instantiated only once.

        :return: masked image
        :rtype: bImage
        """
        # init containers if needed. They are instantiated only
        # once and updated by drawing.
        if self.parentImage.useHald:
            return self.getHald()
        if self.maskedThumbContainer is None:
            self.maskedThumbContainer = bImage.fromImage(self.getThumb(), parentImage=self.parentImage)
        if self.maskedImageContainer is None:
            self.maskedImageContainer = bImage.fromImage(self, parentImage=self.parentImage)
        if self.parentImage.useThumb:
            img = self.maskedThumbContainer
        else:
            img = self.maskedImageContainer
        # reset the container
        img.fill(QColor(0, 0, 0, 0))  # needed for (semi-)transparent background
        # blend lower stack
        qp = QPainter(img)
        top = self.parentImage.getStackIndex(self)
        bottom = 0
        for i, layer in enumerate(self.parentImage.layersStack[bottom:top + 1]):
            if layer.visible:
                if i == 0:
                    qp.setCompositionMode(QPainter.CompositionMode_Source)
                    qp.setOpacity(layer.opacity)  # enables semi transparent background layer
                else:
                    qp.setOpacity(layer.opacity)
                    if type(layer.compositionMode) is QPainter.CompositionMode:
                        qp.setCompositionMode(layer.compositionMode)
                if layer.rPixmap is None:
                    layer.rPixmap = QPixmap.fromImage(layer.getCurrentImage())
                # blend layer into img
                if type(layer.compositionMode) is QPainter.CompositionMode:
                    qp.drawPixmap(QRect(0, 0, img.width(), img.height()), layer.rPixmap)
                else:
                    buf = QImageBuffer(img)[..., :3][..., ::-1]
                    buf0 = QImageBuffer(layer.getCurrentImage())[..., :3][..., ::-1]
                    funcname = 'blend' + bLUeGui.blend.compositionModeDict_names[layer.compositionMode] + 'Buf'
                    func = getattr(bLUeGui.blend, funcname, None)
                    if func:
                        buf[...] = func(buf, buf0) * layer.opacity + buf * (1.0 - layer.opacity)
                    else:
                        # silently ignore unimplemented modes
                        pass
                # clipping
                if layer.isClipping and layer.maskIsEnabled:
                    # draw mask as opacity mask
                    # mode DestinationIn (set dest opacity to source opacity)
                    qp.setCompositionMode(QPainter.CompositionMode_DestinationIn)
                    omask = vImage.color2OpacityMask(layer.mask)
                    qp.drawImage(QRect(0, 0, img.width(), img.height()), omask)
        qp.end()
        return img

    def applyToStack(self):
        """
        Apply new layer parameters and propagate changes to upper layers.
        """

        # recursive function
        def applyToStack_(layer, pool=None):
            # apply transformation
            if layer.visible:
                start = time()
                layer.execute(l=layer)
                layer.cacheInvalidate()
                print("%s %.2f" % (layer.name, time() - start))
            stack = layer.parentImage.layersStack
            lg = len(stack)
            ind = layer.getStackIndex() + 1
            # update histograms displayed
            # on the layer form, if any
            if ind < lg:
                grForm = stack[ind].getGraphicsForm()
                if grForm is not None:
                    grForm.updateHists()
            # get next upper visible layer
            while ind < lg:
                if stack[ind].visible:
                    break
                ind += 1
            if ind < lg:
                layer1 = stack[ind]
                applyToStack_(layer1, pool=pool)

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            applyToStack_(self, pool=None)
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
    def isBackgroundLayer(self):
        return 'background' in self.role

    def isAdjustLayer(self):
        return self.view is not None  # hasattr(self, 'view')

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

    def isDrawLayer(self):
        return 'DRW' in self.role

    def isImageLayer(self):
        return 'Image' in self.role

    def isMergingLayer(self):
        return 'MERGING' in self.role

    def updateOnlyPixmap(self, bRect=None):
        """
        Sync rPixmap with current layer image. Unlike updatePixmap functions it does nothing else.
        If not None, bRect is a bounding rect of the modified  region of layer image. The coordinates
        of bRect are relative to the full size image.
        THe method returns the modified region of layer image. The coordinates are relative to current image.

        :param bRect: modified region of layer image (coord. relative to full size image)
        :type bRect: QRect
        :return: modified region of layer image (coord. relative to current image)
        :rtype: QRect
        """
        rImg = self.getCurrentImage()
        crect = rImg.rect()

        if self.maskIsEnabled:
            rImg = vImage.visualizeMask(rImg, self.mask, color=self.maskIsSelected)

        # convert bRect to current image coordinates
        if bRect:
            bRect = QRect(QPoint(*self.full2CurrentXY(bRect.topLeft().x(), bRect.topLeft().y())),
                          QPoint(*self.full2CurrentXY(bRect.bottomRight().x(), bRect.bottomRight().y()))
                          )
        else:
            bRect = crect

        # rImg may have transparencies (mask...), so we force alpha channel
        if self.rPixmap is None:
            self.rPixmap = QPixmap.fromImage(rImg, Qt.NoOpaqueDetection)
        elif self.rPixmap.size() != rImg.size():
            if self.rPixmap.convertFromImage(rImg, Qt.NoOpaqueDetection):
                pass
            else:
                raise ValueError('updatePixmap: conversion error')
        else:
            # alpha channel possibly does not exist yet
            if not self.rPixmap.hasAlphaChannel():
                self.rPixmap.convertFromImage(rImg, Qt.NoOpaqueDetection)
            qp = QPainter(self.rPixmap)
            qp.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            qp.drawImage(bRect.topLeft().x(), bRect.topLeft().y(),
                         rImg,
                         sx=bRect.topLeft().x(), sy=bRect.topLeft().y(),
                         sw=bRect.width(), sh=bRect.height()
                         )
            qp.end()
            crect = bRect
        return crect

    def updatePixmap(self, maskOnly=False, bRect=None):
        """
        Synchronize rPixmap with the layer image and mask.
        if maskIsEnabled is False, the mask is not used.
        If maskIsEnabled is True, then
            - if maskIsSelected is True, the mask is drawn over
              the layer as a color mask.
            - if maskIsSelected is False, the mask is drawn as an
              opacity mask, setting the image opacity to that of the mask
              (mode DestinationIn).

        :param maskOnly: not used : for consistency with overriding method signature
        :type maskOnly: boolean
        :param bRect: not used : for consistency with overriding method signature
        :type bRect: QRect
        """
        rImg = self.getCurrentImage()
        self.updateOnlyPixmap(bRect=bRect)
        self.setModified(True)

    def getStackIndex(self):
        """
        Returns layer index in the stack, len(stack) - 1 if
        the layer is not in the stack.

        :return:
        :rtype: int
        """
        i = -1
        for i, l in enumerate(self.parentImage.layersStack):
            if l is self:
                break
        return i

    def setMaskEnabled(self, color=False):
        self.maskIsEnabled = True
        self.maskIsSelected = color
        self.maskSettingsChanged.sig.emit()

    def setMaskDisabled(self):
        self.maskIsEnabled = False
        self.maskIsSelected = False
        self.maskSettingsChanged.sig.emit()

    def getTopVisibleStackIndex(self):
        """
        Returns the index of the top visible layer.

        :return:
        :rtype:
        """
        stack = self.parentImage.layersStack
        lg = len(stack)
        for i in range(lg - 1, -1, -1):
            if stack[i].visible:
                return i
        return -1

    def getLowerVisibleStackIndex(self, flag=''):
        """
        Returns the index of the next lower visible layer
        with flag set to True, -1 if it does not exists.

        :param flag:
        :type flag: bool
        :return:
        :rtype: int
        """
        ind = self.getStackIndex()
        stack = self.parentImage.layersStack
        for i in range(ind - 1, -1, -1):
            if stack[i].visible and getattr(stack[i], flag, True):
                return i
        return -1

    def getUpperVisibleStackIndex(self, stHeight=None):
        """
        Returns the index of the next upper visible layer,
        -1 if it does not exists. If stHeight is given the search is limited
        to a sub-stack of height stHeight.

        :param stHeight:
        :type stHeight: int
        :return:
        :rtype: int
        """
        ind = self.getStackIndex()
        stack = self.parentImage.layersStack
        lg = len(stack) if stHeight is None else min(stHeight, len(stack))
        for i in range(ind + 1, lg, 1):
            if stack[i].visible:
                return i
        return -1

    def getLowerClippingStackIndex(self):
        """
         Returns the index of the next lower clipping layer,
        -1 if it does not exists.

        :return:
        :rtype: int
        """
        ind = self.getStackIndex()
        for i in range(ind + 1, len(self.parentImage.layersStack), 1):
            if self.parentImage.layersStack[i].isClipping:
                return i
        return -1


    def compressToLower(self):
        """
        Compress all lower visible layers into target.
        Target is the nearest visible lower layer marked as 'C'.
        Does nothing if mode is preview or if the target layer is
        neither an image layer nor the background layer.
        """

        ind = self.getLowerVisibleStackIndex(flag='compressFlag')
        if ind < 0:
            # no target found
            return
        target = self.parentImage.layersStack[ind]
        if not (target.isImageLayer() or target.isBackgroundLayer()) or self.parentImage.useThumb:
            info = "Uncheck Preview first" if self.parentImage.useThumb else "Target layer must be background or image"
            dlgWarn("Cannot compress layers", info=info)
            return
        # update stack
        self.parentImage.layersStack[0].applyToStack()

        # merge
        img = target.sourceImg if target.isImageLayer() else target.getCurrentImage()
        if type(self.compositionMode) is QPainter.CompositionMode:
            qp = QPainter(img)
            qp.setCompositionMode(self.compositionMode)
            # qp.setOpacity(self.opacity)
            qp.drawImage(QRect(0, 0, self.width(), self.height()), self)
            qp.end()
        else:
            buf = QImageBuffer(img)[..., :3][..., ::-1]
            funcname = 'blend' + bLUeGui.blend.compositionModeDict_names[self.compositionMode] + 'Buf'
            func = getattr(bLUeGui.blend, funcname, None)
            buf[...] = func(buf, QImageBuffer(self.getCurrentImage()[..., :3][..., ::-1]))

        target.updatePixmap()

        self.parentImage.layerView.clear(delete=False)
        currentIndex = self.getStackIndex()  # source
        self.parentImage.activeLayerIndex = ind  # target
        skip = 0
        stack = self.parentImage.layersStack
        for i in range(ind+1, currentIndex+1):
            if not (stack[i].isRawLayer() or stack[i].isBackgroundLayer()):
                del stack[ind + 1 + skip]
            else:
                skip += 1
        self.parentImage.layerView.setLayers(self.parentImage)

        self.parentImage.layersStack[0].applyToStack()

    def reset(self):
        """
        reset layer to inputImg
        """
        self.setImage(self.inputImg())

    def setOpacity(self, value):
        """
        set layer opacity to value/100.0

        :param value:
        :type value: int in range 0..100
        """
        self.opacity = value / 100.0

    def setColorMaskOpacity(self, value):
        """
        Set mask alpha channel to value.

        :param value:
        :type value: int in range 0..255
        """
        self.colorMaskOpacity = value
        buf = QImageBuffer(self.mask)
        buf[:, :, 3] = np.uint8(value)

    def drag(self, x1, y1, x0, y0, widget):
        """
        Called by imageLabel.mouseMove() Ctrl+drag.
        Subclasses may override the method to provide
        a suitable response to this action.
        Coordinates (x1, y1) and (x0, y0) are respectively the ending and the beginning of the move event.
        They are relative to the calling imageLabel.

        :param x1:
        :type x1: int
        :param y1:
        :type y1: int
        :param x0:
        :type x0: int
        :param y0:
        :type y0: int
        :param widget: caller
        :type widget: imageLabel
        """
        window = widget.window
        if not window.label.img.isCropped:
            return
        window.cropTool.moveCrop(x1 - x0, y1 - y0, self.parentImage)

    def zoom(self, pos, numSteps, widget):
        """
        Called by imageLabel.wheelEvent() Ctrl+Wheel.
        Subclasses may override the method to provide
        a suitable response to this action.
        Default is a Crop Tool aware zooming.

        :param pos: mouse pos. (relative to widget)
        :type pos: QPoint
        :param numSteps: relative angle of rotation
        :type numSteps: float
        :param widget:
        :type widget: imageLabel
        """
        window = widget.window
        if not window.label.img.isCropped:
            return
        crpt = window.cropTool
        crpt.zoomCrop(pos, numSteps, self.parentImage)
        window.updateStatus()

    def __getstate__(self):

        def codeCompositionMode(cpm):
            if type(cpm) is QPainter.CompositionMode:
                return cpm.value
            return cpm

        d = {}
        grForm = self.getGraphicsForm()
        if grForm is not None:
            d = grForm.__getstate__()
        # pickling/unpickling enums may be unsafe !
        d['compositionMode'] = codeCompositionMode(self.compositionMode)
        d['opacity'] = self.opacity
        d['visible'] = self.visible
        d['maskIsEnabled'] = self.maskIsEnabled
        d['maskIsSelected'] = self.maskIsSelected
        d['mergingFlag'] = self.mergingFlag
        d['mask'] = 0 if self._mask is None else 1  # used by addAdjustmentLayers()
        aux = [0 if getattr(self, name, None) is None else 1 for name in self.innerImages]
        d['images'] = (*aux,)
        d['sRects'] = [(rect.left(), rect.right(), rect.top(), rect.bottom()) for rect in self.sRects]
        return d

    def __setstate__(self, state):

        def decodeCompositionMode(cpm):
            if type(cpm) is QPainter.CompositionMode:
                return cpm
            # type(cpm) is int:
            if cpm >= 0:
                return QPainter.CompositionMode(cpm)
            return cpm

        d = state['state']
        self.compositionMode = decodeCompositionMode(d['compositionMode'])
        self.opacity = d['opacity']
        self.visible = d['visible']
        self.maskIsEnabled = d['maskIsEnabled']
        self.maskIsSelected = d['maskIsSelected']
        self.sRects = [QRect(w1, h1, w2 - w1, h2 - h1) for w1, w2, h1, h2 in d.get('sRects', [])]
        if 'mergingFlag' in d:
            self.mergingFlag = d['mergingFlag']
        if self.maskIsSelected:
            self.setColorMaskOpacity(self.colorMaskOpacity)
        grForm = self.getGraphicsForm()
        if grForm is not None:
            grForm.__setstate__(state)

    def getCurrentSelCoords(self, emptyAllowed=False):
        """
        Returns the list of (left, right, top, bottom) coordinates of the selection rectangles.
        Coordinates are relative to the current image.
        If emptyAllowed is False and the selection is empty, the current image coordinates are returned.

        :param emptyAllowed:
        :type emptyAllowed:
        :return:
        :rtype: list of 4-uples of int
        """
        currentImage = self.getCurrentImage()
        w, h = currentImage.width(), currentImage.height()
        wF, hF = self.width(), self.height()
        wRatio, hRatio = float(w) / wF, float(h) / hF

        selList = []
        for rect in self.sRects:
            w1, w2, h1, h2 = int(rect.left() * wRatio), int(rect.right() * wRatio), int(
                rect.top() * hRatio), int(rect.bottom() * hRatio)
            w1, h1 = max(w1, 0), max(h1, 0)
            w2, h2 = min(w2, w), min(h2, h)
            if w1 <= w2 and h1 <= h2:
                selList.append((w1, w2, h1, h2))

        if not selList:
            selList.append((0, w, 0, h))

        return selList

    def applyNone(self, bRect=None):
        """
        Pass through
        """
        imgIn = self.inputImg()
        bufIn = QImageBuffer(imgIn)
        bufOut = QImageBuffer(self.getCurrentImage())
        bufOut[:, :, :] = bufIn
        self.updatePixmap(bRect=bRect)


    def applyGrabcut(self, nbIter=2, mode=cv2.GC_INIT_WITH_MASK):
        """
        Segmentation.
        The segmentation mask is built from the selection rectangle, if any, and from
        the user selection.

        :param nbIter:
        :type nbIter: int
        :param mode:
        :type mode:
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

        # resizing coeff
        r = inputImg.width() / self.width()

        ############################
        # init the segmentation mask
        ############################
        if not self.sRects:
            # no selection rectangle : PR_BGD everywhere
            segMask = np.zeros((inputImg.height(), inputImg.width()), dtype=np.uint8) + cv2.GC_PR_BGD
        else:
            # PR_FGD inside selection, BGD otherwise
            segMask = np.zeros((inputImg.height(), inputImg.width()), dtype=np.uint8) + cv2.GC_BGD
            for rect in self.sRects:
                segMask[int(rect.top() * r):int(rect.bottom() * r),
                int(rect.left() * r):int(rect.right() * r)] = cv2.GC_PR_FGD

        # add info from current self.innerSegMask to segMask
        # Only valid pixels are added to segMask, fixing them as FG or BG
        # initially (i.e. before any painting with BG/FG tools and before first call to applygrabcut)
        # all pixels are marked as invalid. Painting a pixel marks it as valid, Ctrl+paint
        # switches it back to invalid.

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
            dlgWarn('You must select some background or foreground pixels',
                    info='Use selection rectangle or Mask/Unmask tools')
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
        print('%s : %.2f' % (bGrabcut.__name__, (time() - t0)))

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
            m = ((buf[:, :, 2] == 0) & (ebuf[:, :, 2] == unmasked)) | (
                    (buf[:, :, 2] == unmasked) & (dbuf[:, :, 2] == 0))
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
        tmp *= 255
        np.clip(tmp, 0, 255, out=tmp)
        # invert
        bufOut[:, :, :3] = 255.0 - tmp
        self.updatePixmap()

    def applyHDRMerge(self, options):
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

        :param options:
        :type options:
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
        buf *= (2 ** exposureCorrection)
        np.clip(buf, 0.0, 1.0, out=buf)
        # convert back to RGB
        buf = rgbLinear2rgb(buf)
        np.clip(buf, 0.0, 255.0, out=buf)
        currentImage = self.getCurrentImage()
        ndImg1a = QImageBuffer(currentImage)

        ndImg1a[...] = bufIn
        for (w1, w2, h1, h2) in self.getCurrentSelCoords():
            ndImg1a[h1:h2 + 1, w1:w2 + 1, :3][:, :, ::-1] = buf[h1:h2 + 1, w1:w2 + 1]

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

        bufOut[...] = bufIn
        for (w1, w2, h1, h2) in self.getCurrentSelCoords():
            bufOut[h1:h2 + 1, w1:w2 + 1, :3][:, :, ::-1] = np.round(
                buf[h1:h2 + 1, w1:w2 + 1])  # truncation would be harmful here

        self.updatePixmap()

    def applyTransForm(self, options):
        """
        Apply the geometric transformation defined by source and target quads.

        :param options:
        :type options:
        """
        inImg = self.inputImg()
        outImg = self.getCurrentImage()
        buf0 = QImageBuffer(outImg)
        w, h = inImg.width(), inImg.height()
        s = w / self.width()
        D = QTransform().scale(s, s)
        DInv = QTransform().scale(1 / s, 1 / s)
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
        rectTrans = DInv.map(T.map(D.map(self.rect()))).boundingRect()
        # apply the transformation and re-translate the transformed image
        # so that the resulting transformation is T and NOT that given by QImage.trueMatrix()
        img = (inImg.transformed(T)).copy(QRect(-rectTrans.x() * s, -rectTrans.y() * s, w, h))
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

    def applyNoiseReduction(self):
        """
        Wavelets, bilateral filtering, NLMeans
        """
        adjustForm = self.getGraphicsForm()
        noisecorr = adjustForm.noiseCorrection  # range 0..10
        currentImage = self.getCurrentImage()
        inputImage = self.inputImg()
        buf0 = QImageBuffer(inputImage)
        buf1 = QImageBuffer(currentImage)
        ########################

        # hald pass through or no correction
        if self.parentImage.isHald or noisecorr == 0:
            buf1[...] = buf0
            self.updatePixmap()
            return
        ########################
        w, h = self.width(), self.height()
        w0, h0 = inputImage.width(), inputImage.height()

        # reset output image
        buf1[...] = buf0
        for w1, w2, h1, h2 in self.getCurrentSelCoords():
            # slicing
            rect = QRect(w1, h1, w2 - w1, h2 - h1)
            imgRect = QRect(0, 0, w0, h0)
            rect = rect.intersected(imgRect)
            if rect.width() < 10 or rect.height() < 10:
                dlgWarn('Selection is too narrow')
                return
            slices = np.s_[rect.top(): rect.bottom(), rect.left(): rect.right(), :3]
            ROI0 = buf0[slices]

            ROI1 = buf1[slices]
            buf01 = ROI0[:, :, ::-1]

            if adjustForm.options['Wavelets']:
                wavelet = 'haar'
                level = chooseDWTLevel(buf01, wavelet)

                if adjustForm.options['Luminosity']:
                    bufLab = cv2.cvtColor(buf01, cv2.COLOR_RGB2Lab)
                    bufLab[..., 0] = dwtDenoiseChan(bufLab, chan=0, thr=noisecorr, thrmode='wiener', level=level,
                                                    wavelet=wavelet)

                    bufLab[..., 0] = np.clip(bufLab[..., 0], 0, 255)

                    # back to RGB
                    ROI1[:, :, ::-1] = cv2.cvtColor(bufLab.astype(np.uint8), cv2.COLOR_Lab2RGB)

                elif adjustForm.options['RGB']:
                    dwtDenoise(ROI0, ROI1, thr=noisecorr, thrmode='wiener', wavelet=wavelet, level=level)

            elif adjustForm.options['Bilateral']:
                ROI1[:, :, ::-1] = cv2.bilateralFilter(buf01,
                                                       9 if self.parentImage.useThumb else 15,
                                                       # diameter of
                                                       # (coordinate) pixel neighborhood,
                                                       # 5 is the recommended value for
                                                       # fast processing (21:5.5s, 15:3.5s)
                                                       10 * adjustForm.noiseCorrection,
                                                       # std deviation sigma
                                                       # in color space,  100 middle value
                                                       50 if self.parentImage.useThumb else 150,
                                                       # std deviation sigma
                                                       # in coordinate space,
                                                       # 100 middle value
                                                       )

            elif adjustForm.options['NLMeans']:
                # hluminance, hcolor,  last params window sizes 7, 21 are recommended values
                ROI1[:, :, ::-1] = cv2.fastNlMeansDenoisingColored(buf01, None, 1 + noisecorr, 1 + noisecorr, 7,
                                                                   21)
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

        :param version:
        :type version: str
        """
        adjustForm = self.getGraphicsForm()
        options = adjustForm.options
        contrastCorrection = adjustForm.contrastCorrection
        satCorrection = adjustForm.satCorrection
        vibCorrection = adjustForm.vibCorrection
        brightnessCorrection = adjustForm.brightnessCorrection
        inputImage = self.inputImg()
        tmpBuf = QImageBuffer(inputImage)
        currentImage = self.getCurrentImage()
        ndImg1a = QImageBuffer(currentImage)

        # no correction : forward lower layer
        if contrastCorrection == 0 and satCorrection == 0 and brightnessCorrection == 0 and vibCorrection == 0:
            ndImg1a[:, :, :] = tmpBuf
            self.updatePixmap()
            return

        ##########################
        # Lab mode (slower than HSV)
        ##########################
        if version == 'Lab':
            # get l channel (range 0..1)
            LBuf = inputImage.getLabBuffer().copy()
            ndImg1a[:, :, :] = tmpBuf
            for w1, w2, h1, h2 in self.getCurrentSelCoords():
                LBuf = LBuf[h1:h2 + 1, w1:w2 + 1]
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
                            raise ValueError(
                                'A contrast curve was found.\nCheck the option Show Contrast Curve in Cont/Bright/Sat layer')
                        auto = self.autoSpline and not self.parentImage.isHald
                        res, a, b, d, T = warpHistogram(LBuf[:, :, 0], warp=contrastCorrection,
                                                        preserveHigh=options['High'],
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
            HSVBuf0 = inputImage.getHSVBuffer().copy()
            ndImg1a[:, :, :] = tmpBuf
            for w1, w2, h1, h2 in self.getCurrentSelCoords():
                HSVBuf = HSVBuf0[h1:h2 + 1, w1:w2 + 1]

                if brightnessCorrection != 0:
                    alpha = 1.0 / (
                            0.501 + adjustForm.brightnessCorrection) - 1.0  # approx. map -0.5...0.0...0.5 --> +inf...1.0...0.0
                    # tabulate x**alpha
                    LUT = np.power(np.arange(256) / 255, alpha)
                    LUT *= 255.0
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
                            raise ValueError(
                                'A contrast curve was found.\nCheck the option Show Contrast Curve in Cont/Bright/Sat layer')
                        buf32 = HSVBuf[:, :, 2].astype(float) / 255
                        auto = self.autoSpline and not self.parentImage.isHald  # flag for manual/auto spline
                        res, a, b, d, T = warpHistogram(buf32, warp=contrastCorrection, preserveHigh=options['High'],
                                                        spline=None if auto else self.getMmcSpline())
                        res = (res * 255.0).astype(np.uint8)
                        # show the spline viewer
                        if self.autoSpline and options['manualCurve']:
                            self.getGraphicsForm().setContrastSpline(a, b, d, T)
                            self.autoSpline = False
                    HSVBuf[:, :, 2] = res

                if satCorrection != 0:
                    # approx. map -0.5...0.0...0.5 --> +inf...1.0...0.0
                    alpha = 1.0 / (0.501 + satCorrection) - 1.0
                    # tabulate x**alpha
                    LUT = np.power(np.arange(256) / 255, alpha)
                    LUT *= 255
                    # convert saturation s to s**alpha
                    HSVBuf[:, :, 1] = LUT[HSVBuf[:, :, 1]]  # faster than take

                if vibCorrection != 0:
                    # threshold
                    thr = 0.5
                    # approx. map -0.5...0.0...0.5 --> +inf...1.0...0.0
                    alpha = 1.0 / (0.501 + vibCorrection) - 1.0
                    LUT = np.arange(256) / 255
                    LUT = np.where(LUT < thr, np.power(LUT / thr, alpha) * thr, LUT)
                    LUT *= 255
                    HSVBuf[:, :, 1] = LUT[HSVBuf[:, :, 1]]  # faster than take

            ndImg1a[:, :, :3][:, :, ::-1] = cv2.cvtColor(HSVBuf0, cv2.COLOR_HSV2RGB)  # sRGBBuf

        self.updatePixmap()

    def apply1DLUT(self, stackedLUT):
        """
        Apply 1D LUTS to R, G, B channels (one curve for each channel).

        :param stackedLUT: array of color values (in range 0..255) : a row for each R, G, B channel
        :type stackedLUT : ndarray, shape=(3, 256), dtype=int
        """
        # identity LUT: forward lower layer and return
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
        ndImg1a[...] = ndImg0a

        for w1, w2, h1, h2 in self.getCurrentSelCoords():
            for c in range(3):  # 0.36s for 15Mpx
                s = ndImg0[h1:h2 + 1, w1:w2 + 1, 0].shape
                ndImg1[h1:h2 + 1, w1:w2 + 1, c] = np.take(stackedLUT[2 - c, :],
                                                          ndImg0[h1:h2 + 1, w1:w2 + 1, c].reshape((-1,))
                                                         ).reshape(s)

        self.updatePixmap()

    def applyLab1DLUT(self, stackedLUT, options=None):
        """
        Applies 1D LUTS (one row for each L,a,b channel).

        :param stackedLUT: array of color values (in range 0..255). Shape must be (3, 255) : a row for each channel
        :type stackedLUT: ndarray shape=(3,256) dtype=int or float
        :param options: not used yet
        """
        if options is None:
            options = UDict()

        # identity LUT : forward lower layer and return
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast is working
            buf1 = QImageBuffer(self.inputImg())
            buf2 = QImageBuffer(self.getCurrentImage())
            buf2[:, :, :] = buf1
            self.updatePixmap()
            return

        # convert LUT to float to speed up  buffer conversions
        stackedLUT = stackedLUT.astype(float)
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
        ndLabImg1 = scaleBackLabBuf(ndLabImg1)
        # back sRGB conversion
        ndsRGBImg1 = Lab2sRGBVec(ndLabImg1)
        # in place clipping
        np.clip(ndsRGBImg1, 0, 255, out=ndsRGBImg1)  # mandatory

        currentImage = self.getCurrentImage()
        ndImg1 = QImageBuffer(currentImage)

        ndImg0 = QImageBuffer(Img0)
        ndImg1[...] = ndImg0
        for w1, w2, h1, h2 in self.getCurrentSelCoords():
            ndImg1[h1:h2 + 1, w1:w2 + 1, :3][:, :, ::-1] = ndsRGBImg1[h1:h2 + 1, w1:w2 + 1, :3]

        self.updatePixmap()

    def applyHSPB1DLUT(self, stackedLUT, options=None, pool=None):
        """
        Currently unused.

        Applies 1D LUTS to hue, sat and perceptual brightness channels.

        :param stackedLUT: array of color values (in range 0..255), a row for each channel
        :type stackedLUT : ndarray shape=(3,256) dtype=int or float
        :param options: not used yet
        :type options : dictionary
        :param pool: multiprocessing pool : unused
        :type pool: muliprocessing.Pool
        """
        if options is None:
            options = UDict()

        # identity LUT : forward lower layer and return
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast is working
            buf1 = QImageBuffer(self.inputImg())
            buf2 = QImageBuffer(self.getCurrentImage())
            buf2[:, :, :] = buf1
            self.updatePixmap()
            return
        Img0 = self.inputImg()
        ndHSPBImg0 = Img0.getHspbBuffer()  # time 2s with cache disabled for 15 Mpx
        # apply LUTS to normalized channels (range 0..255)
        ndLImg0 = (ndHSPBImg0 * [255.0 / 360.0, 255.0, 255.0]).astype(np.uint8)
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

        self.updatePixmap()

    def applyHSV1DLUT(self, stackedLUT, options=None, pool=None):
        """
        Applies 1D LUTS to hue, sat and brightness channels.

        :param stackedLUT: array of color values (in range 0..255), a row for each channel
        :type stackedLUT : ndarray shape=(3,256) dtype=int or float
        :param options: not used yet
        :type options : Udict
        :param pool: multiprocessing pool : unused
        :type pool: muliprocessing.Pool
        """
        if options is None:
            options = UDict()

        # identity LUT : forward lower layer and return
        if not np.any(stackedLUT - np.arange(256)):  # last dims are equal : broadcast is working
            buf1 = QImageBuffer(self.inputImg())
            buf2 = QImageBuffer(self.getCurrentImage())
            buf2[:, :, :] = buf1
            self.updatePixmap()
            return

        # convert LUT to float to speed up  buffer conversions
        stackedLUT = stackedLUT.astype(float)
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

        currentImage = self.getCurrentImage()
        ndImg1 = QImageBuffer(currentImage)

        ndImg0 = QImageBuffer(Img0)
        ndImg1[...] = ndImg0
        for w1, w2, h1, h2 in self.getCurrentSelCoords():
            ndImg1[h1:h2 + 1, w1:w2 + 1, :3][:, :, ::-1] = RGBImg1[h1:h2 + 1, w1:w2 + 1, :3]

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

        # get HSV buffer, range H: 0..180, S:0..255 V:0..255  (opencv convention for 8 bits images)
        HSVImg0 = inputImage.getHSVBuffer().astype(float)  # copy done
        HSVImg0[:, :, 0] *= 2  # 0..360

        # reset layer image
        bufOut[:, :, :] = QImageBuffer(inputImage)

        for w1, w2, h1, h2 in self.getCurrentSelCoords():

            rect = QRect(w1, h1, w2 - w1, h2 - h1)

            w, wF = self.getCurrentImage().width(), self.width()
            h, hF = self.getCurrentImage().height(), self.height()
            # wRatio, hRatio = float(w) / wF, float(h) / hF
            w0, h0 = inputImage.width(), inputImage.height()
            imgRect = QRect(0, 0, w0, h0)
            rect = rect.intersected(imgRect)
            if rect.width() < 10 or rect.height() < 10:
                dlgWarn('Selection is too narrow')
                return

            bufHSV_CV32 = HSVImg0[h1:h2 + 1, w1:w2 + 1, :]

            divs = LUT.divs
            steps = tuple([360 / divs[0], 255.0 / divs[1], 255.0 / divs[2]])
            interp = chosenInterp(pool, (w2 - w1) * (h2 - h1))
            coeffs = interp(LUT.data, steps, bufHSV_CV32, convert=False)
            bufHSV_CV32[:, :, 0] = np.mod(bufHSV_CV32[:, :, 0] + 2.0 * coeffs[:, :, 0], 360)  # scale Hue curve
            bufHSV_CV32[:, :, 1:] = bufHSV_CV32[:, :, 1:] * coeffs[:, :, 1:]
            np.clip(bufHSV_CV32, (0, 0, 0), (360, 255, 255), out=bufHSV_CV32)
            bufHSV_CV32[:, :, 0] /= 2  # 0..180

            bufpostF32_1 = cv2.cvtColor(bufHSV_CV32.astype(np.uint8), cv2.COLOR_HSV2RGB)

            bufOut = QImageBuffer(currentImage)
            bufOut[h1:h2 + 1, w1:w2 + 1, :3] = bufpostF32_1[:, :, ::-1]

        self.updatePixmap()

    def applyAuto3DLUT(self, options=None, pool=None):
        """
        The pretrained classifier of (C) Hui Zeng, Jianrui Cai, Lida Li, Zisheng Cao, and Lei Zhang
        is used to compute a 3D LUT for automatic image enhancement.
        CF. https://github.com/HuiZeng/Image-Adaptive-3DLUT.

        :param options:
        :type options:
        :param pool:
        :type pool:
        :return:
        :rtype:
        """
        adjustForm = self.getGraphicsForm()
        inputImage = self.inputImg()
        currentImage = self.getCurrentImage()
        inputBuffer = QImageBuffer(inputImage)[:, :, :3]
        imgBuffer = QImageBuffer(currentImage)[:, :, :3]
        ndImg0 = inputBuffer[:, :, :3]
        # get manual corrections
        coeffs = [adjustForm.slider1.value(), adjustForm.slider2.value(), adjustForm.slider3.value()]
        # get auto lut
        lutarray, pred = generateLUTfromQImage(inputImage, coeffs)

        adjustForm.predLabel1.setText('%f' % pred[0])
        adjustForm.predLabel2.setText('%f' % pred[1])
        adjustForm.predLabel3.setText('%f' % pred[2])

        lut3D = LUT3D(lutarray, dtype=np.float32)

        interp = chosenInterp(pool, inputImage.width() * inputImage.height())
        buf = interp(lut3D.LUT3DArray, lut3D.step, ndImg0.astype(np.float32), convert=False)
        np.clip(buf, 0, 255, out=buf)

        imgBuffer[..., :3] = inputBuffer
        for (w1, w2, h1, h2) in self.getCurrentSelCoords():
            imgBuffer[h1:h2 + 1, w1:w2 + 1, :3] = buf[h1:h2 + 1, w1:w2 + 1, ...]

        self.updatePixmap()

    def applyGrading(self, lut3D, options=None, pool=None):

        adjustForm = self.getGraphicsForm()
        inputImage = self.inputImg()
        currentImage = self.getCurrentImage()
        inputBuffer = QImageBuffer(inputImage)[:, :, :3]
        imgBuffer = QImageBuffer(currentImage)[:, :, :3]
        ndImg0 = inputBuffer[:, :, :3]

        # get HSV gradient
        hsvgrad = hsvGradientArray(adjustForm.grad)
        setLUTfromGradient(lut3D, hsvgrad, adjustForm.LUT3D_ori2hsv, adjustForm.brCoeffs)

        interp = chosenInterp(pool, inputImage.width() * inputImage.height())
        buf = interp(lut3D.LUT3DArray, lut3D.step, ndImg0.astype(np.float32), convert=False)
        np.clip(buf, 0, 255, out=buf)

        imgBuffer[..., :3] = inputBuffer
        for (w1, w2, h1, h2) in self.getCurrentSelCoords():
            imgBuffer[h1:h2 + 1, w1:w2 + 1, :3] = buf[h1:h2 + 1, w1:w2 + 1, :3]  # image buffers and luts are BGR

        self.updatePixmap()

    def apply3DLUT(self, lut3D, options=None, pool=None):
        """
        Apply a 3D LUT to the current view of the image (self or self.thumb).
        If pool is not None and the size of the current view is > 3000000,
        parallel interpolation on image slices is used.
        If options['keep alpha'] is False, alpha channel is interpolated too.
        LUT axes, LUT channels and image channels must be in BGR order.

        :param lut3D: LUT3D
        :type lut3D: LUT3D
        :param options:
        :type options: UDict
        :param pool:
        :type pool:
        """
        LUT = lut3D.LUT3DArray
        LUTSTEP = lut3D.step
        if options is None:
            options = UDict()

        # get buffers
        inputImage = self.inputImg()
        currentImage = self.getCurrentImage()
        inputBuffer0 = QImageBuffer(inputImage)
        imgBuffer = QImageBuffer(currentImage)
        interpAlpha = not options['keep alpha']

        # get selection
        w1, w2, h1, h2 = (0.0,) * 4
        useSelection = len(self.sRects) > 0

        if useSelection:
            # forward lower layer
            imgBuffer[...] = inputBuffer0

        for w1, w2, h1, h2 in self.getCurrentSelCoords():
            inputBuffer = inputBuffer0[h1:h2 + 1, w1:w2 + 1, :]
            if interpAlpha:
                # interpolate alpha channel from LUT
                ndImg0 = inputBuffer
                ndImg1 = imgBuffer
            else:
                ndImg0 = inputBuffer[:, :, :3]
                ndImg1 = imgBuffer[:, :, :3]
                LUT = np.ascontiguousarray(LUT[..., :3])
            interp = chosenInterp(pool, (w2 - w1) * (h2 - h1))
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
            self.updatePixmap()
            return
        ########################

        buf1[...] = buf0
        for w1, w2, h1, h2 in self.getCurrentSelCoords():
            # slicing
            rect = QRect(w1, h1, w2 - w1, h2 - h1)
            imgRect = QRect(0, 0, currentImage.width(), currentImage.height())
            rect = rect.intersected(imgRect)
            if rect.width() < 10 or rect.height() < 10:
                dlgWarn('Selection is too narrow')
                return
            slices = np.s_[rect.top():rect.bottom(), rect.left():rect.right(), :3]
            ROI0 = buf0[slices]
            ROI1 = buf1[slices]

            # kernel based filtering
            r = inputImage.width() / self.width()
            radius = int(adjustForm.radius * r)
            if adjustForm.kernelCategory in [filterIndex.IDENTITY, filterIndex.UNSHARP,
                                             filterIndex.SHARPEN, filterIndex.BLUR1, filterIndex.BLUR2]:
                # correct radius for preview if needed
                kernel = getKernel(adjustForm.kernelCategory, radius, adjustForm.amount)
                ROI1[:, :, :] = cv2.filter2D(ROI0, -1, kernel)
            else:
                # bilateral filtering
                sigmaColor = 2 * adjustForm.tone
                sigmaSpace = sigmaColor
                ROI1[:, :, ::-1] = cv2.bilateralFilter(ROI0[:, :, ::-1], radius, sigmaColor, sigmaSpace)

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
            self.updatePixmap()
            return
        ########################

        # We blend a neutral filter with density range 0.5*s...0.5 with the image b,
        # using blending mode overlay : f(a,b) = 2*a*b if b < 0.5 else f(a,b) = 1 - 2*(1-a)(1-b)

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
            # very small correction: forward lower layer and return
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
            resImg = bLUeGui.blend.blendLuminosity(filter, inputImage)
            bufOutRGB = QImageBuffer(resImg)[:, :, :3][:, :, ::-1]
        #####################
        # Chromatic adaptation
        #####################
        elif options['Chromatic Adaptation']:
            # get RGB multipliers
            m1, m2, m3, _ = temperatureAndTint2Multipliers(temperature, 2 ** tint,
                                                           self.parentImage.RGB_lin2XYZInverse)
            buf = QImageBuffer(inputImage)[:, :, :3]
            bufXYZ = RGB2XYZ(buf[:, :, ::-1],
                             RGB_lin2XYZ=self.parentImage.RGB_lin2XYZ)
            bufsRGBLinear = np.tensordot(bufXYZ, self.parentImage.RGB_lin2XYZInverse,
                                         axes=(-1, -1))
            # apply multipliers
            bufsRGBLinear *= [m1, m2, m3]
            # brightness correction
            M = np.max(bufsRGBLinear)
            bufsRGBLinear /= M
            bufOutRGB = rgbLinear2rgb(bufsRGBLinear)
            np.clip(bufOutRGB, 0, 255, out=bufOutRGB)
            bufOutRGB = np.round(bufOutRGB).astype(np.uint8)
        else:
            raise ValueError('applyTemperature : wrong option')

        # set output image
        bufOut0 = QImageBuffer(currentImage)
        bufOut = bufOut0[:, :, :3]
        bufOut0[...] = buf1
        for w1, w2, h1, h2 in self.getCurrentSelCoords():
            bufOut[h1:h2 + 1, w1:w2 + 1, ::-1] = bufOutRGB[h1:h2 + 1, w1:w2 + 1, ...]

        self.updatePixmap()


class QPresentationLayer(QLayer):
    """
    A presentation layer is used for color management. It is an
    adjustment layer whose output is equal to input. It does not belong to the layer stack :
    conceptually, it is "above" the stack, so it holds the composition of
    all stacked layers. It is the sole color managed layer, via its qPixmap
    attribute.
    """

    def __init__(self, *args, **kwargs):
        self.qPixmap = None
        self.cmImage = None
        super().__init__(*args, **kwargs)  # don't move up !

    def inputImg(self, redo=True):
        return self.parentImage.layersStack[self.getTopVisibleStackIndex()].getCurrentMaskedImage()

    def updatePixmap(self, maskOnly=False, bRect=None):
        """
        Synchronizes qPixmap and rPixmap with the image layer.
        THe Parameter maskOnly is provided for compatibility only and it is unused.
        """
        currentImage = self.getCurrentImage()

        crect = self.updateOnlyPixmap(bRect=bRect)

        # color management
        if icc.COLOR_MANAGE and self.parentImage is not None:
            if self.qPixmap is None:
                img = icc.convertQImage(currentImage, transformation=self.parentImage.colorTransformation)
                self.qPixmap = QPixmap.fromImage(img)
            elif self.qPixmap.size() != currentImage.size():
                img = icc.convertQImage(currentImage, transformation=self.parentImage.colorTransformation)
                self.qPixmap.convertFromImage(img)
            else:
                currentImage = currentImage.copy(crect)
                img = icc.convertQImage(currentImage, transformation=self.parentImage.colorTransformation)
                qp = QPainter(self.qPixmap)
                qp.drawImage(crect.topLeft().x(), crect.topLeft().y(),
                             img
                             )
                qp.end()
        else:
            self.qPixmap = self.rPixmap

        self.setModified(True)

    def applyNone(self, bRect=None):
        super().applyNone(bRect=bRect)
        self.parentImage.setModified(True)

    def update(self, bRect=None):
        self.applyNone(bRect=bRect)


class QCloningLayer(QLayer):
    """
    Cloning layer.
    To make mask retouching easier, the binary cloning mask
    is taken from the destination image
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = 'CLONING'
        # cloning source image
        self.sourceFromFile = False
        # virtual layer moved flag
        self.vlChanged = False
        self.cloningState = ''
        self.cloningMethod = None
        # bLU files must eventually save source Image
        self.innerImages = ('sourceImg',)
        # init self.cloning mask, self.monts, self.conts;
        # these attributes are relative to full sized images
        # and used in applyCloning() to speed up move display.
        self.updateCloningMask()

    #############################################
    # compatibility with QImageLayer attribute sourceImg
    @property
    def sourceImg(self):
        return self.getGraphicsForm().sourceImage

    @sourceImg.setter
    def sourceImg(self, img):
        self.getGraphicsForm().sourceImage = img

    ############################################

    def inputImg(self, redo=True, drawTranslated=False):  # True):
        """
        Overrides QLayer.inputImg().
        If drawTranslated is True (default False), draws the translated source image over
        the image.

        :param redo:
        :type redo:
        :param drawTranslated:
        :type drawTranslated:
        """
        img1 = super().inputImg(redo=redo)
        if not drawTranslated:
            return img1
        adjustForm = self.getGraphicsForm()
        if adjustForm is None:
            return img1
        qp = QPainter(img1)
        currentAltX, currentAltY = self.full2CurrentXY(self.xAltOffset, self.yAltOffset)
        if self.sourceFromFile:
            qp.drawPixmap(QPointF(currentAltX, currentAltY), adjustForm.sourcePixmap)
        else:
            qp.drawImage(QPointF(currentAltX, currentAltY), img1.copy())
        return img1

    def updateCloningMask(self):
        """
        Update the binary cloning mask (relative to the full sized image)
         and its moments
        """
        if not self.isCloningLayer():
            return
        self.cloning_mask = vImage.colorMask2BinaryArray(self.mask)
        self.monts = moments(self.cloning_mask)
        self.conts = contours(self.cloning_mask)

    def updateSourcePixmap(self):
        """
        If the cloning source is the (layer) input image
        the method refreshes the source pixmap, otherwise it does nothing.
        The method should be called every time the lower
        stack is modified.
        """
        if not self.sourceFromFile:
            img = self.inputImg(drawTranslated=False)
            if img.rPixmap is None:
                img.rPixmap = QPixmap.fromImage(img)
            self.getGraphicsForm().sourcePixmap = img.rPixmap

    def seamlessMerge(self, outImg, inImg, mask, cloningMethod, version='opencv', w=3):
        """
        Seamless cloning.  The cloning mask and contours are
        recomputed and scaled to image size.

        :param outImg: destination image
        :type outImg: vImage
        :param inImg: source image
        :type inImg: vImage
        :param mask: color mask
        :type mask: QImage
        :param cloningMethod:
        :type cloningMethod: opencv cloning method
        :param version:
        :type version: str
        :param w:
        :type w:
        """
        # build the working cloning mask.
        # scale mask to dest current size,  and convert to a binary mask
        src_mask = mask.scaled(outImg.size()).copy(QRect(QPoint(0, 0), inImg.size()))  # useless copy ?
        cloning_mask = vImage.colorMask2BinaryArray(src_mask)
        conts = contours(cloning_mask)
        if not conts:
            return
        # simplify contours and get bounding rect
        epsilon = 0.01 * cv2.arcLength(conts[0], True)
        bRect = QRect(*cv2.boundingRect(conts[0]))
        for cont in conts[1:]:
            acont = cv2.approxPolyDP(cont, epsilon, True)
            bRect |= QRect(*cv2.boundingRect(acont))  # union
        if not bRect.isValid():
            dlgWarn("seamlessMerge : no cloning region found")
            return
        inRect = bRect & inImg.rect()
        bt, bb, bl, br = inRect.top(), inRect.bottom(), inRect.left(), inRect.right()
        # cv2.seamlesClone uses a white mask, so we turn cloning_mask into
        # a 3-channel buffer.
        src_maskBuf = np.dstack((cloning_mask, cloning_mask, cloning_mask)).astype(np.uint8)[bt:bb + 1, bl:br + 1, :]
        sourceBuf = QImageBuffer(inImg)
        destBuf = QImageBuffer(outImg)
        # clone the unmasked region of source into dest.
        if version == 'opencv':
            sourceBuf = sourceBuf[bt:bb + 1, bl:br + 1, :]
            destBuf = destBuf[bt:bb + 1, bl:br + 1, :]
            output = cv2.seamlessClone(np.ascontiguousarray(sourceBuf[:, :, :3]),  # source
                                       np.ascontiguousarray(destBuf[:, :, :3]),  # dest
                                       src_maskBuf,
                                       ((br - bl) // 2, (bb - bt) // 2),  # The cloning center is the center of bRect.
                                       cloningMethod
                                       )
            destBuf[:, :, :3] = output
        else:
            output = seamlessClone(sourceBuf[:, :, :3],
                                   destBuf[:, :, :3],
                                   cloning_mask,
                                   conts,
                                   bRect,
                                   (0, 0),
                                   (0, 0),
                                   w=w,
                                   passes=2)
            destBuf[:, :, :3] = output

    def __getstate__(self):
        # tmp = self.innerImages
        # if self.sourceImg is None:
        # self.innerImages = []  # no image to save
        d = super().__getstate__()
        d['sourceX'] = self.sourceX
        d['sourceY'] = self.sourceY
        d['xAltOffset'] = self.xAltOffset
        d['yAltOffset'] = self.yAltOffset
        d['cloningState'] = self.cloningState
        # self.innerImages = tmp # restore
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        d = d['state']
        self.sourceX = d['sourceX']
        self.sourceY = d['sourceY']
        self.xAltOffset = d['xAltOffset']
        self.yAltOffset = d['yAltOffset']
        self.cloningState = d['cloningState']
        self.updateCloningMask()
        self.applyCloning()

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

        :param seamless:
        :type seamless: boolean
        :param showTranslated:  unused, set to True in all calls
        :type showTranslated:
        :param moving: flag indicating if the method is triggered by a mouse event
        :type moving: boolean
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
            self.updatePixmap()
            return
        ########################

        ##############################################
        # update the marker in the positioning window
        ##############################################
        # mask center coordinates relative to full size image
        r = self.monts['m00']
        if (not self.conts) or r == 0:
            self.xAltOffset, self.yAltOffset = 0.0, 0.0
            self.AltZoom_coeff = 1.0
            seamless = False
        # coordinates of the center of cloning_mask (relative to full size image)
        if r > 0:
            xC, yC = self.monts['m10'] / r, self.monts['m01'] / r
        else:
            xC, yC = 0.0, 0.0

        # erase previous transformed image : reset imgOut to ImgIn
        qp = QPainter(imgOut)
        qp.setCompositionMode(QPainter.CompositionMode_Source)
        qp.drawImage(QRect(QPoint(0, 0), imgOut.size()), imgIn, imgIn.rect())  # TODO modified 26/11/21 validate

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
        bRect = QRectF(currentAltX + (1 - self.AltZoom_coeff) * xC_current,
                       currentAltY + (1 - self.AltZoom_coeff) * yC_current,
                       imgOut.width() * self.AltZoom_coeff, imgOut.height() * self.AltZoom_coeff)
        qp.drawPixmap(bRect, sourcePixmap, sourcePixmap.rect())
        qp.end()

        #####################
        # do seamless cloning
        #####################
        if seamless:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                bLUeTop.Gui.app.processEvents()
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


class QLayerImage(QLayer):
    """
    QLayer containing a source image.
    The input image built from the stack is merged with the source image,
    using the blending mode and opacity of the layer. In this way, the layer
    contribution to the stack is reduced to applyNone().
    The source image is resized to fit the size of the current document.
    """

    @classmethod
    def fromImage(cls, mImg, parentImage=None, role='', sourceImg=None):
        layer = cls(QImg=mImg, role=role, parentImage=parentImage, sourceImg=sourceImg)
        return layer

    def __init__(self, *args, **kwargs):
        self.sourceImg = kwargs.pop('sourceImg', None)  # before super().__init__()
        super().__init__(*args, **kwargs)
        self.filename = ''  # path to sourceImg file
        # bLU files must eventually save/restore source image
        self.innerImages = ('sourceImg',)
        # undo/redo functionality
        self.history = historyList()

    def inputImg(self, redo=True):
        """
        Overrides QLayer.inputImg().
        The source image is drawn over the input image
        built from the stack, using the QPainter default
        composition mode and opacity.

        :return:
        :rtype: QImage
        """
        img1 = super().inputImg()
        # img1.fill(QColor(0, 0, 0, 0))
        # merging with sourceImg
        qp = QPainter(img1)
        qp.drawImage(QRect(0, 0, img1.width(), img1.height()), self.sourceImg)
        qp.end()
        return img1

    def bTransformed(self, transformation, parentImage):
        """
        Applies transformation to a copy of layer and returns the copy.

        :param transformation:
        :type transformation: QTransform
        :param parentImage:
        :type parentImage: mImage
        :return: transformed layer
        :rtype: QLayerImage
        """
        tLayer = super().bTransformed(transformation, parentImage)
        if tLayer.tool is not None:
            tLayer.tool.layer = tLayer
            tLayer.tool.img = tLayer.parentImage
        return tLayer

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


class QDrawingLayer(QLayerImage):
    """
    The drawing (sourceImg) is blended onto a transparent layer.
    This is the classical painting mode.
    Other Q*Layer classes implement in/out (or transformational)
    mode, suitable for image edition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # intermediate layer
        self.stroke = QImage(self.sourceImg.size(), self.sourceImg.format())
        # atomic stroke painting is needed to handle brush opacity:
        # We save layer.sourceImg as atomicStrokeImg at each stroke beginning.
        self.atomicStrokeImg = None
        # cache for current brush dict
        self.brushDict = None
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.last_refresh = 0  # refresh rate control
        self.uRect = QRect()  # modified region control

    def inputImg(self, redo=True):
        """
        Returns the drawing painted onto a transparent image.

        :param redo: unused
        :type redo: boolean
        :return:
        :rtype: QImage
        """
        img1 = super().inputImg(redo=False)
        img1.fill(QColor(0, 0, 0, 0))
        # merging sourceImg
        qp = QPainter(img1)
        qp.drawImage(QRect(0, 0, img1.width(), img1.height()), self.sourceImg)  # scale sourceImg
        qp.end()
        return img1

    def updatePixmap(self, maskOnly=False, bRect=None):
        """
        Transfers the layer translation to the drawing and
        resets the translation to 0 before updating pixmap.

        :param maskOnly:
        :type maskOnly: boolean
        """
        x, y = self.xOffset, self.yOffset
        if x != 0 or y != 0:
            self.sourceImg = self.sourceImg.copy(QRect(-x, -y, self.sourceImg.width(), self.sourceImg.height()))
            self.xOffset, self.yOffset = 0, 0
            img1 = self.inputImg()
            im = self.getCurrentImage()
            buf = QImageBuffer(im)
            buf[...] = QImageBuffer(img1)

        super().updatePixmap(maskOnly=maskOnly, bRect=bRect)

    def drag(self, x1, y1, x0, y0, widget):
        """
        Translates the drawing.

        :param x1:
        :type x1: int
        :param y1:
        :type y1: int
        :param x0:
        :type x0: int
        :param y0:
        :type y0: int
        :param widget:
        :type widget: imageLabel
        """
        self.xOffset += x1 - x0
        self.yOffset += y1 - y0
        self.updatePixmap()
        self.parentImage.prLayer.update()  # =applyNone()

class QRawLayer(QLayer):
    """
    Raw image development layer
    """

    @classmethod
    def fromImage(cls, mImg, parentImage=None):
        """
        Returns a QLayer object initialized with mImg.

        :param mImg:
        :type mImg: QImage
        :param parentImage:
        :type parentImage: mImage
        :return:
        :rtype: QRawLayer
        """
        layer = QRawLayer(QImg=mImg, parentImage=parentImage)
        layer.parentImage = parentImage
        return layer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postProcessCache = None
        self.bufCache_HSV_CV32 = None

    @property
    def postProcessCache(self):
        return self.__postProcessCache

    @postProcessCache.setter
    def postProcessCache(self, buf):
        self.__postProcessCache = buf
