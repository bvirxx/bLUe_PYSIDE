"""
Copyright (C) 2017  Bernard Virot

PeLUT - Photo editing software using adjustment layers with 1D and 3D Look Up Tables.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""


import cv2
from imgconvert import *
from PyQt4.QtGui import QPixmap, QImage, QColor, QPainter, QMessageBox
from PyQt4.QtCore import QRect, QByteArray
from icc import convertQImage, MONITOR_PROFILE_PATH, COLOR_MANAGE
from LUT3D import interpVec
from time import time
import re
import icc
from copy import copy

class metadata:
    """
    Image meta data
    """
    def __init__(self):
        self.name, self.colorSpace, self.rawMetadata, self.profile, self.orientation = '', -1, [], '', None

class vImage(QImage):
    """
    versatile image class.
    This is the base class for all multi-layered and interactive image classes.
    When no data is passed as parameter, we build a null image.
    """
    def __init__(self, filename=None, cv2Img=None, QImg=None, cv2mask=None, format=QImage.Format_ARGB32,
                                            name='', colorSpace=-1, orientation=None, meta=None, rawMetadata=[], profile=''):
        """
        :param filename:
        :param cv2Img:
        :param QImg:
        :param cv2mask:
        :param format: QImage format (default QImage.Format_ARGB32)
        :param name:
        :param colorSpace:
        :param orientation: Qtransform object (default None)
        :param meta: metadata object (default None)
        :param rawMetadata: list of dictionaries (default [])
        :param profile: embedded profile (default '')
        """
        self.transformation = None
        self.isModified = False
        self.onModify = lambda : 0
        self.rect, self.mask, = None, cv2mask
        self.filename = filename

        if meta is None:
            self.meta = metadata()
            self.meta.name, self.meta.colorSpace, self.meta.rawMetadata, self.meta.profile, self.meta.orientation = name, colorSpace, rawMetadata, profile, orientation
        else:
            self.meta = meta

        # create a null image when no data given
        if (filename is None and cv2Img is None and QImg is None):
            super(vImage, self).__init__(format=format)

        # load image from file
        if filename is not None:
            if self.meta.orientation is not None:
                tmp =QImage(filename).transformed(self.meta.orientation)
            else:
                tmp = QImage(filename)
            # shallow copy : no harm !
            super(vImage, self).__init__(tmp)
            #if self.meta.orientation is not None:
                #self.transformed(self.meta.orientation)
        # build image from QImage, shallow copy
        elif QImg is not None:
            super(vImage, self).__init__(QImg)
            if hasattr(QImg, "meta"):
                self.meta = copy(QImg.meta)
        # build image from buffer
        elif cv2Img is not None:
            super(vImage, self).__init__(ndarrayToQImage(cv2Img, format=format))

        self.updatePixmap()

        if self.mask is None:
            self.mask = QImage(self.width(), self.height(), QImage.Format_ARGB32)
            self.mask.fill(0)

    def setModified(self, b):
        self.isModified = b
        self.onModify()

    def updatePixmap(self):
        if icc.COLOR_MANAGE:
            img = convertQImage(self)
        else:
            img = self
        img.setPixel(0,0, 0)
        self.qPixmap = QPixmap.fromImage(img)


    def resize(self, pixels, interpolation=cv2.INTER_CUBIC):
        """
        Resize an image while keeping its aspect ratio. We use
        the opencv finction cv2.resize() to perform the resizing operation, so we
        can choose the resizing method (default cv2.INTER_CUBIC)
        The original image is not modified.
        :param pixels: pixel count for the resized image
        :param interpolation method (default cv2.INTER_CUBIC)
        :return : the resized vImage
        """
        ratio = self.width() / float(self.height())
        w, h = int(np.sqrt(pixels * ratio)), int(np.sqrt(pixels / ratio))
        hom = w / float(self.width())
        # resizing
        cv2Img = cv2.resize(QImageBuffer(self), (w, h), interpolation=interpolation)
        # creating new vImage
        rszd = vImage(cv2Img=cv2Img, meta=copy(self.meta), format=self.format())

        #resize rect and mask
        if self.rect is not None:
            rszd.rect = QRect(self.rect.left() * hom, self.rect.top() * hom, self.rect.width() * hom, self.rect.height() * hom)
        if self.mask is not None:
            # tmp.mask=cv2.resize(self.mask, (w,h), interpolation=cv2.INTER_NEAREST )
            rszd.mask = self.mask.scaled(w, h)
        self.setModified(True)
        return rszd

    def applyLUT(self, LUT, widget=None, options={}):

        # get image buffer (BGR order on intel proc.)
        ndImg0 = QImageBuffer(self.inputImg)[:, :, :3] #[:, :, ::-1]
        ndImg1 = QImageBuffer(self)[:, :, :3]
        # apply LUT to each channel
        ndImg1[:,:,:]=LUT[ndImg0]
        #update
        self.updatePixmap()
        if widget is not None:
            widget.repaint()

    def apply3DLUT(self, LUT, widget=None, options={}):

        # get image buffer (type RGB)
        #w1, w2, h1, h2 = 0, self.inputImg.width(), 0, self.inputImg.height()
        w1, w2, h1, h2 = (0.0,) * 4
        #if self.parent.rect is not None:
        if options['use selection']:
            if self.rect is not None:
                w1, w2, h1,h2= self.rect.left(), self.rect.right(), self.rect.top(), self.rect.bottom()
            if w1>=w2 or h1>=h2:
                msg = QMessageBox()
                msg.setText("Empty selection\nSelect a region with marquee tool")
                msg.exec_()
                return
        else:
            w1, w2, h1, h2 = 0, self.inputImg.width(), 0, self.inputImg.height()

        ndImg0 = QImageBuffer(self.inputImg)[h1+1:h2+1, w1+1:w2+1, :3] #[:, :, ::-1]

        ndImg1 = QImageBuffer(self)[:, :, :3]

        # apply LUT
        start=time()
        ndImg1[h1+1:h2+1,w1+1:w2+1,:] = interpVec(LUT, ndImg0)
        end=time()
        print 'time %.2f' % (end-start)

        self.updatePixmap()

        if widget is not None:
            widget.repaint()

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
        bgLayer = QLayer.fromImage(self)
        self.addLayer(bgLayer, 'background')
        self.setModified(False)
        self.activeLayer = bgLayer

    def updatePixmap(self):
        """
        Override vImage.updatePixmap()
        """
        #super(mImage, self).updatePixmap()
        vImage.updatePixmap(self)
        for layer in self.layersStack:
                vImage.updatePixmap(layer)
        return
        qpainter = QPainter(self.qPixmap)
        for layer in self.layersStack:
            if layer.visible:
                qpainter.drawPixmap(0, 0, layer.qPixmap)
        qpainter.end()

    def addLayer(self, lay, name):
        # build a unique name
        usedNames = [l.name for l in self.layersStack]
        a = 1
        trialname = name
        while trialname in usedNames:
            trialname = name + '_'+ str(a)
            a = a+1
        lay.name = trialname
        self._layers[lay.name] = lay
        self.layersStack.append(lay)
        lay.meta = self.meta
        #if lay.name != 'drawlayer':
            #lay.updatePixmap()
        self.setModified(True)

    def addAdjustmentLayer(self, name='', window=None):
        lay = QLayer(QImg=self.layersStack[-1])
        lay.inputImg = QImage(self.layersStack[-1])
        self.addLayer(lay, name)
        #lay.window = window
        lay.parent = self
        self.setModified(True)
        return lay

    def cvtToGray(self):
        self.cv2Img = cv2.cvtColor(self.cv2Img, cv2.COLOR_BGR2GRAY)
        #self.qImg = gray2qimage(self.cv2Img)

    def save(self, filename, quality=-1):
        # build resulting image
        img = QImage(self.width(), self.height(), self.format())
        qpainter = QPainter(img)
        for layer in self.layersStack:
            if layer.visible:
                qpainter.drawImage(0,0, layer)
        qpainter.end()
        # save to file
        if not img.save(filename, quality=quality):
            msg = QMessageBox()
            msg.setText("unable to save file %s" % filename)
            msg.exec_()
        else:
            self.setModified(False)


class imImage(mImage) :
    """
    Interactive multi-layer image
    """
    def __init__(self, *args, **kwargs):
        super(imImage, self).__init__(*args, **kwargs)
        self.Zoom_coeff = 1.0
        self.xOffset, self.yOffset = 0, 0
        #
        self.isMouseSelectable =True
        #drawLayer = QLayer(QImage(self.width(), self.height(), QImage.Format_ARGB32))
        #drawLayer.fill(QColor(0,0,0,0))
        #self.addLayer(drawLayer, 'drawlayer')

    def resize_coeff(self, widget):
        """
        computes the resizing coefficient
        to be applied to mimg to display a non distorted view
        in the widget.

        :param mimg: mImage
        :param widget: Qwidget object
        :return: the (multiplicative) resizing coefficient
        """
        r_w, r_h = float(widget.width()) / self.width(), float(widget.height()) / self.height()
        r = max(r_w, r_h)
        return r * self.Zoom_coeff

    def resize(self, pixels, interpolation=cv2.INTER_CUBIC):
        """
        Resize image and layers
        :param pixels:
        :param interpolation:
        :return:
        """
        # resized vImage
        rszd0 = super(imImage, self).resize(pixels, interpolation=interpolation)
        # resized imImage
        rszd = imImage(QImg=rszd0,meta=copy(self.meta))
        rszd.rect = rszd0.rect
        for k, l  in enumerate(self.layersStack):
            if l.name != "background" and l.name != 'drawlayer':
                img = QLayer.fromImage(self._layers[l.name].resize(pixels, interpolation=interpolation))
                rszd.layersStack.append (img)
                rszd._layers[l.name] = img
        self.isModified = True
        return rszd

    def view(self):
        return self.Zoom_coeff, self.xOffset, self.yOffset

    def setView(self, zoom=1.0, xOffset=0.0, yOffset=0.0):
        self.Zoom_coeff, self.xOffset, self.yOffset = zoom, xOffset, yOffset

    def snapshot(self):
        snap = imImage(QImg=self, meta=self.meta)
        qp = QPainter(snap)
        for layer in self.layersStack:
            if layer.visible:
                """
                if layer.qPixmap is not None:
                    qp.drawPixmap(QRect(0, 0, self.width() , self.height()),
                                  # target rect
                                  layer.transfer()  # layer.qPixmap
                                  )
                else:
                """
                qp.drawImage(QRect(0, 0, self.width(), self.height()),
                                 # target rect
                                 layer
                                 )
        qp.end()
        snap.updatePixmap()
        return snap

class QLayer(vImage):
    @classmethod
    def fromImage(cls, mImg):
        mImg.visible = True
        mImg.alpha = 255
        mImg.window = None
        # for adjustment layer
        mImg.transfer = lambda: mImg.qPixmap
        mImg.inputImg = None
        return QLayer(QImg=mImg) #mImg

    def __init__(self, *args, **kwargs):
        super(QLayer, self).__init__(*args, **kwargs)
        self.name='anonymous'
        self.visible = True
        self.opacity= 1.0
        self.transfer = lambda : self.qPixmap
        # link to grid or curves view for adjustment layers
        self.adjustView = None
        self.parent = None

    def setImage(self, qimg):
        """
        replace layer image with qimg.
        The layer and qimg must have identical dimensions and type
        :param qimg: QImage
        """
        buf1, buf2 = QImageBuffer(self), QImageBuffer(qimg)
        if buf1.shape != buf2.shape:
            raise ValueError("QLayer.setImage : new image and layer must have identical shapes")
        buf1[...] = buf2
        self.updatePixmap()

    def reset(self):
        self.setImage(self.inputImg)

    def setTransparency(self, value):
        self.opacity = value /100.0
        return
        buf = QImageBuffer(self)
        buf[...,3] = value
        self.updatePixmap()



