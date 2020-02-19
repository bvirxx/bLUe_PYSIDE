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
from io import BytesIO
from os import path

import numpy as np
import gc

from PIL.ImageCms import ImageCmsProfile
from PySide2.QtCore import Qt, QSize, QPoint, QPointF, QFileInfo

import cv2
from copy import copy

from PySide2.QtGui import QTransform, QColor, QCursor
from PySide2.QtWidgets import QApplication
from PySide2.QtGui import QPixmap, QImage, QPainter
from PySide2.QtCore import QRect

from bLUeCore.demosaicing import demosaic
from bLUeTop import exiftool
from bLUeGui.memory import weakProxy
from bLUeTop.cloning import contours, moments, seamlessClone

from bLUeTop.colorManagement import icc, cmsConvertQImage
from bLUeGui.bLUeImage import QImageBuffer, ndarrayToQImage, bImage
from bLUeGui.dialog import dlgWarn, dlgInfo, IMAGE_FILE_EXTENSIONS, RAW_FILE_EXTENSIONS
from time import time

from bLUeTop.lutUtils import LUT3DIdentity
from bLUeGui.baseSignal import baseSignal_bool, baseSignal_Int2, baseSignal_No
from bLUeTop.rawProcessing import rawRead
from bLUeTop.utils import qColorToRGB, historyList

from bLUeTop.versatileImg import vImage


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
        @param srcFile: source image or sidecar (the extension is replaced by .mie).
        @type srcFile: str
        @param destFile: image file
        @type destFile: str
        @param defaultorientation
        @type defaultorientation: bool
        @param thumbfile: thumbnail file
        @type thumbfile: str
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
        self.activeLayerIndex = None
        self.addLayer(bgLayer, name='Background')
        # color management : we assume that the image profile is the working profile
        self.colorTransformation = icc.workToMonTransform
        # presentation layer
        prLayer = QPresentationLayer(QImg=self, parentImage=self)
        prLayer.name = 'presentation'
        prLayer.role = 'presentation'
        prLayer.execute = lambda l=prLayer, pool=None: prLayer.applyNone()
        prLayer.updatePixmap()  # mandatory here as vImage init. can't do it
        self.prLayer = prLayer
        # link to rawpy instance
        self.rawImage = None

    def copyStack(self, source):
       """
       Replaces layer stack, graphic forms and
       meta data by these from source.
       @param source:
       @type source: mImage
       """
       self.meta = source.meta
       self.onImageChanged = source.onImageChanged
       self.useThumb = source.useThumb
       self.useHald = source.useHald
       self.rect = source.rect
       for l in source.layersStack[1:]:
           lr = QLayer.fromImage(l.scaled(self.size()), parentImage=self)
           lr.execute = l.execute
           lr.name = l.name
           lr.role = l.role
           lr.view = l.view
           lr.view.widget().targetImage = self
           lr.view.widget().layer = lr
           self.layersStack.append(lr)

    def resize(self, pixels, interpolation=cv2.INTER_CUBIC):
        """
        Resizes the image and the layer stack, while keeping the aspect ratio.
        @param pixels:
        @param interpolation:
        @return: resized imImage object
        @rtype: same type as self
        """
        # resized vImage
        rszd0, buf = super().resize(pixels, interpolation=interpolation)
        # get resized image (with a background layer)
        rszd = type(self)(QImg=rszd0, meta=copy(self.meta))
        rszd.__buf = buf  # protect buf from g.c.
        rszd.copyStack(self)
        return rszd

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
        self.onActiveLayerChanged()
        return active

    def getActivePixel(self, x, y, fromInputImg=True, qcolor=False):
        """
        Reads the RGB colors of the pixel at (x, y) from the active layer.
        If fromInputImg is True (default), the pixel is taken from
        the input image, otherwise from the current image.
        Coordinates are relative to the full sized image.
        If (x,y) is outside the image, (0, 0, 0) is returned.
        @param x: x-coordinates of pixel, relative to the full-sized image
        @type x: int
        @param y: y-coordinates of pixel, relative to the full-sized image
        @type y: int
        @param fromInputImg:
        @type fromInputImg:
        @param qcolor:
        @type qcolor:
        @return: color of pixel if qcolor else its R, G, B components
        @rtype: QColor if qcolor else 3-uple of int
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
        @param x: x-coordinate of pixel, relative to the full-sized image
        @type x: int
        @param y: y-coordinate of pixel, relative to the full-sized image
        @type y: int
        @return: pixel RGB colors
        @rtype: 3-uple of int
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
        self.layerStack[0].apply()

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

    def addLayer(self, layer, name='', index=None):
        """
        Adds a layer.

        @param layer: layer to add (if None, add a fresh layer)
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
            trialname = name + '_' + str(a)
            a = a+1
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
        self.setActiveLayer(index)
        layer.meta = self.meta
        layer.parentImage = self
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

        @param layerType: layer class
        @type layerType: QLayer subclass
        @param name:
        @type name: str
        @param role:
        @type role: str
        @param index:
        @type index: int
        @param sourceImg: source image
        @type sourceImg: QImage
        @return: the new layer
        @rtype: subclass of QLayer
        """
        if index is None:
            # adding on top of active layer
            index = self.activeLayerIndex
        if sourceImg is None:
            # set layer from active layer
            if layerType is None:
                layer = QLayer.fromImage(self.layersStack[index], parentImage=self)
            else:
                layer = layerType.fromImage(self.layersStack[index], parentImage=self)
        else:
            # set layer from image :
            layer = QLayerImage.fromImage(self.layersStack[index], parentImage=self, sourceImg=sourceImg)
        layer.role = role
        self.addLayer(layer, name=name, index=index + 1)
        # add autoSpline attribute to contrast layer only
        if role in ['CONTRAST', 'RAW']:
            layer.autoSpline = True
        # init thumb
        if layer.parentImage.useThumb:
            layer.thumb = layer.inputImg().copy()
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
        # layer.paintedMask = layer.mask.copy()
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

    def save(self, filename, quality=-1, compression=-1):
        """
        Overrides QImage.save().
        Writes the presentation layer to a file and returns a
        thumbnail with standard size (160x120 or 120x160).
        Raises IOError if the saving fails.
        @param filename:
        @type filename: str
        @param quality: integer value in range 0..100, or -1
        @type quality: int
        @param compression: integer value in range 0..100, or -1
        @type compression: int
        @return: thumbnail of the saved image
        @rtype: QImage
        """
        def transparencyCheck(buf):
            if np.any(buf[:, :, 3] < 255):
                dlgWarn('Transparency will be lost. Use PNG format instead')
        # don't save thumbnails
        if self.useThumb:
            return None
        # get the final image from the presentation layer.
        # This image is NOT color managed (prLayer.qPixmap
        # only is color managed)
        img = self.prLayer.getCurrentImage()
        # imagewriter and QImage.save are unusable for tif files,
        # due to bugs in libtiff, hence we use opencv imwrite.
        fileFormat = filename[-3:].upper()
        buf = QImageBuffer(img)
        params = []
        if fileFormat == 'JPG':
            transparencyCheck(buf)
            buf = buf[:, :, :3]
            if quality >= 0 and quality <= 100:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]  # quality range 0..100
        elif fileFormat == 'PNG':
            if compression >= 0 and compression <= 9:
                params = [cv2.IMWRITE_PNG_COMPRESSION, compression]  # compression range 0..9
        elif fileFormat == 'TIF':
            transparencyCheck(buf)
            buf = buf[:, :, :3]
        else:
            raise IOError("Invalid File Format\nValid formats are jpg, png, tif ")
        if self.isCropped:
            # make slices
            w, h = self.width(), self.height()
            w1, w2 = int(self.cropLeft), w - int(self.cropRight)
            h1, h2 = int(self.cropTop), h - int(self.cropBottom)
            buf = buf[h1:h2, w1:w2, :]
        # build thumbnail from (evenyually) cropped image
        # choose thumb size
        wf, hf = buf.shape[1], buf.shape[0]
        if wf > hf:
            wt, ht = 160, 120
        else:
            wt, ht = 120, 160
        thumb = ndarrayToQImage(np.ascontiguousarray(buf[:, :, :3][:, :, ::-1]),
                                format=QImage.Format_RGB888).scaled(wt, ht, Qt.KeepAspectRatio)
        written = cv2.imwrite(filename, buf, params)  # BGR order
        if not written:
            raise IOError("Cannot write file %s " % filename)
        return thumb


class imImage(mImage):
    """
    Zoomable and draggable multi-layer image :
    this is the base class for bLUe documents
    """
    @staticmethod
    def loadImageFromFile(f, createsidecar=True, icc=icc, window=None):
        """
        load an imImage (image and metadata) from file. Returns the loaded imImage :
        For a raw file, it is the image postprocessed with default parameters.
        metadata is a list of dicts with len(metadata) >=1.
        metadata[0] contains at least 'SourceFile' : path.
        profile is a string containing the profile binary data,
        currently, we do not use these data : standard profiles
        are loaded from disk, non standard profiles are ignored.
        @param f: path to file
        @type f: str
        @param createsidecar:
        @type createsidecar: boolean
        @param icc:
        @type icc: class icc
        @return: image
        @rtype: imImage
        """
        ###########
        # read metadata
        ##########
        try:
            # read metadata from sidecar (.mie) if it exists, otherwise from image file.
            # The sidecar is created if it does not exist and createsidecar is True.
            with exiftool.ExifTool() as e:
                profile, metadata = e.get_metadata(f, tags=(
                "colorspace", "profileDescription", "orientation", "model", "rating"), createsidecar=createsidecar)
                imageInfo = e.get_formatted_metadata(f)
        except ValueError:
            # Default metadata and profile
            metadata = {'SourceFile': f}
            profile = ''
            imageInfo = 'No data found'
        # color space : 1=sRGB 65535=uncalibrated
        tmp = [value for key, value in metadata.items() if 'colorspace' in key.lower()]
        colorSpace = tmp[0] if tmp else -1
        # try again to find a valid color space tag and/or an imbedded profile.
        # If everything fails, assign sRGB.
        if colorSpace == -1 or colorSpace == 65535:
            tmp = [value for key, value in metadata.items() if 'profiledescription' in key.lower()]
            desc_colorSpace = tmp[0] if tmp else ''
            if isinstance(desc_colorSpace, str):
                if not ('sRGB' in desc_colorSpace) or hasattr(window, 'modeDiaporama'):
                    # setOverrideCursor does not work correctly for a MessageBox :
                    # may be a Qt Bug, cf. https://bugreports.qt.io/browse/QTBUG-42117
                    QApplication.changeOverrideCursor(QCursor(Qt.ArrowCursor))
                    QApplication.processEvents()
                    if len(desc_colorSpace) > 0:
                        # convert profile to ImageCmsProfile object
                        profile = ImageCmsProfile(BytesIO(profile))
                    else:
                        dlgInfo("Color profile is missing\nAssigning sRGB")  # modified 08/10/18 validate
                        # assign sRGB profile
                        colorSpace = 1
        # update the color management object with the image profile.
        icc.configure(colorSpace=colorSpace, workingProfile=profile)
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
        if ext in list(IMAGE_FILE_EXTENSIONS):
            img = imImage(filename=f, colorSpace=colorSpace, orientation=transformation, rawMetadata=metadata,
                          profile=profile, name=name, rating=rating)
        elif ext in list(RAW_FILE_EXTENSIONS):
            # load raw image file in a RawPy instance
            # rawpy.imread keeps f open. Calling raw.close() deletes the raw object.
            # As a workaround we use low-level file buffer and unpack().
            # Relevant RawPy attributes are black_level_per_channel, camera_white_balance, color_desc, color_matrix,
            # daylight_whitebalance, num_colors, raw_colors_visible, raw_image, raw_image_visible, raw_pattern,
            # raw_type, rgb_xyz_matrix, sizes, tone_curve.
            # raw_image and raw_image_visible are sensor data
            rawpyInst = rawRead(f)
            # postprocess raw image, applying default settings (cf. vImage.applyRawPostProcessing)
            rawBuf = rawpyInst.postprocess(use_camera_wb=True)
            # build Qimage : switch to BGR and add alpha channel
            rawBuf = np.dstack((rawBuf[:, :, ::-1], np.zeros(rawBuf.shape[:2], dtype=np.uint8) + 255))
            img = imImage(cv2Img=rawBuf, colorSpace=colorSpace, orientation=transformation,
                          rawMetadata=metadata, profile=profile, name=name, rating=rating)
            # keeping a reference to rawBuf along with img is
            # needed to protect the buffer from garbage collector
            img.rawBuf = rawBuf
            img.filename = f
            # keep references to rawPy instance. rawpyInst.raw_image is the (linearized) sensor image
            img.rawImage = rawpyInst
            #########################################################
            # Reconstructing the demosaic Bayer bitmap :
            # we need it to calculate the multipliers corresponding
            # to a user white point, and we cannot access the
            # native rawpy demosaic buffer from the RawPy instance !!!!
            #########################################################
            # get 16 bits Bayer bitmap
            img.demosaic = demosaic(rawpyInst.raw_image_visible, rawpyInst.raw_colors_visible,
                                    rawpyInst.black_level_per_channel)
            # correct orientation
            if orientation == 6:  # 90°
                img.demosaic = np.swapaxes(img.demosaic, 0, 1)
            elif orientation == 8:  # 270°
                img.demosaic = np.swapaxes(img.demosaic, 0, 1)
                img.demosaic = img.demosaic[:, ::-1, :]
        else:
            raise ValueError("Cannot read file %s" % f)
        if img.isNull():
            raise ValueError("Cannot read file %s" % f)
        if img.format() in [QImage.Format_Invalid, QImage.Format_Mono, QImage.Format_MonoLSB, QImage.Format_Indexed8]:
            raise ValueError("Cannot edit indexed formats\nConvert image to a non indexed mode first")
        img.imageInfo = imageInfo
        window.settings.setValue('paths/dlgdir', QFileInfo(f).absoluteDir().path())
        img.initThumb()
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
    def fromImage(cls, mImg, parentImage=None):
        """
        Return a QLayer object initialized with mImg.
        Derived classes get an instance of themselves
        without overriding.
        @param mImg:
        @type mImg: QImage
        @param parentImage:
        @type parentImage: mImage
        @return:
        @rtype: Qlayer
        """
        layer = cls(QImg=mImg, parentImage=parentImage)
        layer.parentImage = parentImage
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
        self.tool = None
        # back link to parent image
        parentImage = kwargs.pop('parentImage', None)
        self.parentImage = weakProxy(parentImage)
        super().__init__(*args, **kwargs)  # don't move backwards
        # layer opacity, range 0.0...1.0
        self.opacity = 1.0
        self.compositionMode = QPainter.CompositionMode_SourceOver
        ###################################################################################
        # For the sake of conciseness, QLayer is not subclassed to define multiple types of adjustment layers.
        # Instead, we use the attribute execute as a wrapper to the right applyXXX method, depending
        # on the intended "type" of layer.
        # Note : execute should always end by calling updatePixmap.
        ##################################################################################
        self.execute = lambda l=None, pool=None: l.updatePixmap() if l is not None else None
        self.options = {}
        # actionName is used by methods graphics***.writeToStream()
        self.actionName = 'actionNull'
        # view is the dock widget containing
        # the graphics form associated with the layer
        self.view = None
        # undo/redo mask history
        self.historyListMask = historyList(size=5)
        # layer offsets
        self.xOffset, self.yOffset = 0, 0
        self.Zoom_coeff = 1.0
        # clone/dup virtual layer shift and zoom (relative to the full sized image)
        self.xAltOffset, self.yAltOffset = 0, 0
        self.sourceX, self.sourceY = 0, 0
        self.AltZoom_coeff = 1.0
        self.updatePixmap()

    @property
    def mask(self):  # the setter is inherited from bImage
        if self._mask is None:
            if type(self) not in [QPresentationLayer]:
                self._mask = QImage(self.width(), self.height(), QImage.Format_ARGB32)
                # default : unmask all
                self._mask.fill(self.defaultColor_UnMasked)
        return self._mask

    @mask.setter
    def mask(self, m):
        self._mask = m

    def getGraphicsForm(self):
        """
        Return the graphics form associated with the layer
        @return:
        @rtype: QWidget
        """
        if self.view is not None:
            return self.view.widget()
        return None

    def isActiveLayer(self):
        if self.parentImage.getActiveLayer() is self:
            return True
        return False

    def getMmcSpline(self):
        """
        Returns the spline used for multimode contrast
        correction if it is initialized, and None otherwise.
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
            tool.layer.visibilityChanged.sig.disconnect()
        except RuntimeError:
            pass
        tool.layer.visibilityChanged.sig.connect(tool.setVisible)
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
        self.visibilityChanged.sig.emit(value)

    def bTransformed(self, transformation, parentImage):
        """
        Apply transformation to a copy of layer. Returns the transformed copy.
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

    def inputImg(self, redo=True):
        """
        return maskedImageContainer/maskedThumbContainer.
        If redo is True(default), containers are updated and
        their rPixmap are invalidated to force refreshing.
        layer.applyToStack() always calls inputImg() with redo=True.
        So, to keep the containers up to date we only have to follow
        each layer modification by a call to layer.applyToStack().
        @param redo:
        @type redo: boolean
        @return:
        @rtype: bImage
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
        @return: masked image
        @rtype: bImage
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
                if layer.rPixmap is None:
                    layer.rPixmap = QPixmap.fromImage(layer.getCurrentImage())
                qp.drawPixmap(QRect(0, 0, img.width(), img.height()), layer.rPixmap)
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
                print("%s %.2f" % (layer.name, time()-start))
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

    def updatePixmap(self, maskOnly=False):
        """
        Synchronize rPixmap with the layer image and mask.
        if maskIsEnabled is False, the mask is not used.
        If maskIsEnabled is True, then
            - if maskIsSelected is True, the mask is drawn over
              the layer as a color mask.
            - if maskIsSelected is False, the mask is drawn as an
              opacity mask, setting the image opacity to that of the mask
              (mode DestinationIn).
        @param maskOnly: not used : for consistency with overriding method signature
        @type maskOnly: boolean
        """
        rImg = self.getCurrentImage()
        # apply layer transformation. Missing pixels are set to QColor(0,0,0,0)
        if self.xOffset != 0 or self.yOffset != 0:
            x, y = self.full2CurrentXY(self.xOffset, self.yOffset)
            rImg = rImg.copy(QRect(-x, -y, rImg.width()*self.Zoom_coeff, rImg.height()*self.Zoom_coeff))
        if self.maskIsEnabled:
            rImg = vImage.visualizeMask(rImg, self.mask, color=self.maskIsSelected)
        self.rPixmap = QPixmap.fromImage(rImg)
        self.setModified(True)

    def getStackIndex(self):
        """
        Returns layer index in the stack, len(stack) - 1 if
        the layer is not in the stack.
        @return:
        @rtype: int
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

    def setMaskDisabled(self):  # TODO Added 11/12/19 validate
        self.maskIsEnabled = False
        self.maskIsSelected = False
        self.maskSettingsChanged.sig.emit()

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

    def getUpperVisibleStackIndex(self, stHeight=None):
        """
        Returns the index of the next upper visible layer,
        -1 if it does not exists. If stHeight is given the search is limited
        to a sub-stack of height stHeight.
        @param stHeight:
        @type stHeight: int
        @return:
        @rtype: int
        """
        ind = self.getStackIndex()
        stack = self.parentImage.layersStack
        lg = len(stack) if stHeight is None else min(stHeight, len(stack))
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

    """
    def linkMask2Lower(self):
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
        elif not lower.group:
            if not any(o is lower for o in self.group):
                self.group.append(lower)
            lower.group = self.group
        elif not self.group:
            if not any(item is self for item in lower.group):
                lower.group.append(self)
            self.group = lower.group
        self.mask = lower.mask

    def unlinkMask(self):
        self.mask = self.mask.copy()
        # remove self from group
        for i, item in enumerate(self.group):
            if item is self:
                self.group.pop(i)
                # don't keep  group with length 1
                if len(self.group) == 1:
                    self.group.pop(0)
                break
        self.group = []
    """
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
        # update stack
        self.parentImage.layersStack[0].applyToStack()
        # merge
        # target.setImage(self)
        qp = QPainter(target)
        qp.setCompositionMode(self.compositionMode)
        qp.setOpacity(self.opacity)
        qp.drawImage(QRect(0, 0, self.width(), self.height()), self)
        target.updatePixmap()
        self.parentImage.layerView.clear(delete=False)
        currentIndex = self.getStackIndex()
        self.parentImage.activeLayerIndex = ind
        self.parentImage.layersStack.pop(currentIndex)
        self.parentImage.layerView.setLayers(self.parentImage)

    def reset(self):
        """
        reset layer to inputImg
        """
        self.setImage(self.inputImg())

    def setOpacity(self, value):
        """
        set layer opacity to value/100.0
        @param value:
        @type value: int in range 0..100
        """
        self.opacity = value / 100.0

    def setColorMaskOpacity(self, value):
        """
        Set mask alpha channel to value
        @param value:
        @type value: int in range 0..255
        """
        self.colorMaskOpacity = value
        buf = QImageBuffer(self.mask)
        buf[:, :, 3] = np.uint8(value)

    def readFromStream(self, dataStream):
        grForm = self.getGraphicsForm()
        if grForm is not None:
            grForm.readFromStream(dataStream)
        return dataStream


class QPresentationLayer(QLayer):
    """
    A presentation layer is used for color management. It is an
    adjustment layer whose output is equal to input. It does not belong to the layer stack :
    conceptually, it is "above" the stack, so it holds the composition of
    all stacked layers. It is the sole color managed layer, via its qPixmap
    attribute.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qPixmap = None
        self.cmImage = None

    def inputImg(self, redo=True):
        return self.parentImage.layersStack[self.getTopVisibleStackIndex()].getCurrentMaskedImage()

    def updatePixmap(self, maskOnly=False):
        """
        Synchronize qPixmap and rPixmap with the image layer and mask.
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
        # color manage
        if icc.COLOR_MANAGE and self.parentImage is not None and getattr(self, 'role', None) == 'presentation':
            img = cmsConvertQImage(currentImage, cmsTransformation=self.parentImage.colorTransformation)
        else:
            img = currentImage
        qImg = img
        rImg = currentImage
        """
        Presentation Layer has not any mask !
        if self.maskIsEnabled:
            rImg = vImage.visualizeMask(rImg, self.mask, color=self.maskIsSelected, clipping=self.isClipping)
        """
        self.qPixmap = QPixmap.fromImage(qImg)
        self.rPixmap = QPixmap.fromImage(rImg)
        self.setModified(True)

    def applyNone(self):
        super().applyNone()
        self.parentImage.setModified(True)

    def update(self):
        self.applyNone()


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
        self.srcImg = None
        # virtual layer moved flag
        self.vlChanged = False
        self.cloningState = ''
        # init self.cloning mask, self.monts, self.conts;
        # these attributes are relative to full sized images
        # and used in applyCloning() to speed up move display.
        self.updateCloningMask()

    def equalityTest(self):
        imgIn = self.inputImg(drawTranslated=False)
        imgOut = self.getCurrentImage()
        return (QImageBuffer(imgIn) == QImageBuffer(imgOut)).all()

    def inputImg(self, redo=True, drawTranslated=False):  # True):
        """
        Overrides QLayer.inputImg().
        If drawTranslated is True (default False), draws the translated source image over
        the image.
        @param redo:
        @type redo:
        @param drawTranslated:
        @type drawTranslated:
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
        adjustForm = self.getGraphicsForm()
        if not self.sourceFromFile:
            img = self.inputImg(drawTranslated=False)  # TODO added drawtranslated 16/12/19
            if img.rPixmap is None:
                img.rPixmap = QPixmap.fromImage(img)
            adjustForm.sourcePixmap = img.rPixmap

    def seamlessMerge(self, outImg, inImg, mask, cloningMethod, version='opencv', w=3):
        """
        Seamless cloning.  The cloning mask and contours are
        recomputed and scaled to image size.
        @param outImg: destination image
        @type outImg: vImage
        @param inImg: source image
        @type inImg: vImage
        @param mask: color mask
        @type mask: QImage
        @param cloningMethod:
        @type cloningMethod: opencv cloning method
        @param version:
        @type version: str
        @param w:
        @type w:
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
        bRect = QRect(* cv2.boundingRect(conts[0]))
        for cont in conts[1:]:
            acont = cv2.approxPolyDP(cont, epsilon, True)
            bRect |= QRect(* cv2.boundingRect(acont))  # union
        if not bRect.isValid():
            dlgWarn("seamlessMerge : no cloning region found")
            return
        inRect = bRect & QImage.rect(inImg)  # QImage.rect(src_mask)
        bt, bb, bl, br = inRect.top(), inRect.bottom(), inRect.left(), inRect.right()
        # cv2.seamlesClone uses a white mask, so we turn cloning_mask into
        # a 3-channel buffer.
        src_maskBuf = np.dstack((cloning_mask, cloning_mask, cloning_mask)).astype(np.uint8)[bt:bb+1, bl:br+1, :]
        sourceBuf = QImageBuffer(inImg)
        destBuf = QImageBuffer(outImg)
        # clone the unmasked region of source into dest.
        if version == 'opencv':
            sourceBuf = sourceBuf[bt:bb + 1, bl:br + 1, :]
            destBuf = destBuf[bt:bb + 1, bl:br + 1, :]
            output = cv2.seamlessClone(np.ascontiguousarray(sourceBuf[:, :, :3]),  # source
                                       np.ascontiguousarray(destBuf[:, :, :3]),    # dest
                                       src_maskBuf,
                                       ((br-bl)//2, (bb-bt)//2),  # The cloning center is the center of bRect.
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


class QLayerImage(QLayer):
    """
    QLayer containing a source image.
    The input image built from the stack is merged with the source image,
    using the blending mode and opacity of the layer. In this way, the layer
    contribution to the stack is reduced to applyNone().
    The source image is resized to fit the size of the current document.
    """
    @staticmethod
    def fromImage(mImg, parentImage=None, sourceImg=None):
        layer = QLayerImage(QImg=mImg, parentImage=parentImage)
        layer.parentImage = parentImage
        layer.sourceImg = sourceImg
        # drawing buffers
        if sourceImg is not None:
            # intermediate layer
            layer.stroke = QImage(sourceImg.size(), sourceImg.format())
            # atomic stroke painting is needed to handle brush opacity:
            # We save layer.sourceImg in layer.strokeDest at each stroke beginning.
            layer.strokeDest = None
            # cache for current brush dict
            layer.brushDict = None
        # undo/redo functionality
        layer.history = historyList()
        return layer

    def __init__(self, *args, **kwargs):
        self.sourceImg = None
        super().__init__(*args, **kwargs)

    def inputImg(self, redo=True):
        """
        Overrides QLayer.inputImg().
        The input image built from the stack is merged with the source image,
        using the blending mode and opacity of the layer.
        @return:
        @rtype: QImage
        """
        img1 = super().inputImg()  # TODO maybe missing redo=redo 16/12/19
        # merging with sourceImg
        qp = QPainter(img1)
        qp.setOpacity(self.opacity)
        qp.setCompositionMode(self.compositionMode)
        qp.drawImage(QRect(0, 0, img1.width(), img1.height()), self.sourceImg)
        return img1
    """
    def undo(self):
        self.sourceImg = self.history.undo(saveitem=self.sourceImg)

    def redo(self):
        self.sourceImg = self.history.redo()
    """

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
        tLayer = super().bTransformed(transformation, parentImage)
        if tLayer.tool is not None:
            tLayer.tool.layer = tLayer
            tLayer.tool.img = tLayer.parentImage
        return tLayer


class QRawLayer(QLayer):
    """
    Raw image development layer
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
        @rtype: QRawLayer
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

    @property
    def bufCache_HSV_CV32(self):
        return self.__bufCache_HSV_CV32

    @bufCache_HSV_CV32.setter
    def bufCache_HSV_CV32(self, buffer):
        self.__bufCache_HSV_CV32 = buffer



