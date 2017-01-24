import cv2
from imgconvert import *
from PyQt4.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt4.QtCore import QRect, QByteArray
from icc import convertQImage, MONITOR_PROFILE_PATH, COLOR_MANAGE
from LUT3D import LUT3D, interpVec
from multiprocessing import Pool
from time import time
import re
import icc
from copy import copy

class metadata:
    def __init__(self):
        self.name, self.colorSpace, self.rawMetadata, self.profile, self.orientation = '', -1, [], '', None

class vImage(QImage):
    """
    versatile image class.
    This is the base class for all multi-layered and interactive image classes
    """
    """
    def fromQImage(cls, qImg):
        qImg.rect, qImg.mask, qImg.name = None, None, ''
        qImg.qPixmap = QPixmap.fromImage(qImg)
        qImg.__class__ = vImage
        return qImg
    """
    def __init__(self, filename=None, cv2Img=None, QImg=None, cv2mask=None, copy=False, format=QImage.Format_ARGB32,
                 name='', colorSpace=-1, orientation=None, meta=None, rawMetadata=[], profile=''):

        # metadata
        if meta is None:
            self.meta = metadata()
            self.meta.name, self.meta.colorSpace, self.meta.rawMetadata, self.meta.profile, self.meta.orientation = name, colorSpace, rawMetadata, profile, orientation
        else:
            self.meta = meta

        self.rect, self.mask, = None, cv2mask

        self.cv2Cache = None # TODO : remove
        self.cv2CacheType=None
        if (filename is None and cv2Img is None and QImg is None):
            # create a null image
            super(vImage, self).__init__()
        if filename is not None:
            # load file
            if self.meta.orientation is not None:
                tmp =QImage(filename).transformed(self.meta.orientation)
            else:
                tmp = QImage(filename)

            super(vImage, self).__init__(tmp)
            if self.meta.orientation is not None:
                self.transformed(self.meta.orientation)
        elif QImg is not None:
            if copy:
                #d o a deep copy
                super(vImage, self).__init__(QImg.copy())
            else:
                # do a shallow copy (implicit data sharing : copy when writing to)
                super(vImage, self).__init__(QImg)
            if hasattr(QImg, "colorSpace"):
                self.colorSpace=QImg.colorSpace
        elif cv2Img is not None:
            if copy:
                cv2Img = cv2Img.copy()
            # shallow copy
            super(vImage, self).__init__(ndarrayToQImage(cv2Img, format=format))

        # prevent from garbage collector
        self.data=self.bits()

        self.updatePixmap()

        if self.mask is None:
            self.mask = QImage(self.width(), self.height(), QImage.Format_ARGB32)
            self.mask.fill(0)

    def updatePixmap(self):
        if icc.COLOR_MANAGE:
            # 1=sRGB
            if self.meta.colorSpace == 1 or len(self.meta.profile)> 0:
                print 'update convert cm', self.meta.colorSpace, self.format(), self.meta.name
                cvqim=convertQImage(self, toProfile=MONITOR_PROFILE_PATH)
            else:
                cvqim = self
            self.qPixmap = QPixmap.fromImage(cvqim)
        else:
            print 'no color space', self.meta.colorSpace, self.format(), self.meta.name
            self.qPixmap = QPixmap.fromImage(self)

    def cv2Img(self, cv2Type='BGRA'):
        if self.cv2Cache is not None and self.cv2CacheType==cv2Type:
            return self.cv2Cache
        else:
            self.cv2Cache = QImageBuffer(self)
            self.cv2CacheType = 'BGRA'
            if cv2Type == 'RGB':
                self.cv2Cache = self.cv2Cache[:,:,::-1]
                self.cv2Cache = self.cv2Cache[:,:,1:4]
                self.cv2Cache = np.ascontiguousarray(self.cv2Cache, dtype=np.uint8)
                self.cv2CacheType = 'RGB'
            return self.cv2Cache

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
        return rszd

    def applyLUT(self, LUT, widget=None):

        # get image buffer (BGR order on intel proc.)
        ndImg0 = QImageBuffer(self.inputImg)[:, :, :3] #[:, :, ::-1]
        ndImg1 = QImageBuffer(self)[:, :, :3]
        # apply LUT to each channel
        ndImg1[:,:,:]=LUT[ndImg0]
        #update
        self.updatePixmap()
        if widget is not None:
            widget.repaint()

    def apply3DLUT(self, LUT, widget=None):

        # get image buffer (type RGB)
        #ndImg = QImageBuffer(self)[:,:,:3][:,:,::-1]
        ndImg0 = QImageBuffer(self.inputImg)[:, :, :3] #[:, :, ::-1]
        ndImg1 = QImageBuffer(self)[:, :, :3]
        start=time()
        """
        f = lambda *args : [interp(LUT, *x) for x in args]

        ndImg[:] = map(f, * [ndImg[:,c] for c in range(ndImg.shape[1])])
        """
        ndImg1[:,:,:] = interpVec(LUT, ndImg0)

        """
        for r in range(ndImg.shape[0]):
            for c in range(ndImg.shape[1]):
                p=ndImg[r,c]
                ndImg[r,c]= LUT[(p[0]/2), (p[1]/2),(p[2]/2)]           #ndImg[interp(LUT, *ndImg[r,c])
        """
        """
        p = Pool(2)
        #f = lambda x : interp(LUT,x)
        ndImg[:]=p.map(fa, ndImg)
        """
        end=time()
        print 'time %.2f' % (end-start)
        self.updatePixmap()
        # apply LUT
        #convertedNdImg = LUT3DArray(ndImg[:,:], LUT)
        #convertedNdImg = convertedNdImg[:, :].astype(np.uint8)

        # update Pixmap ans repaint
        #self.qPixmap = QPixmap.fromImage(ndarrayToQImage(convertedNdImg, QImage.Format_RGB888))
        if widget is not None:
            widget.repaint()

class mImage(vImage):
    """
    Multilayer image
    """
    def __init__(self, *args, **kwargs):
        super(mImage, self).__init__(*args, **kwargs)
        self._layers = {}
        self._layersStack = []
        # add background layer
        bgLayer = QLayer.fromImage(self)
        bgLayer.name = 'background'
        self._layers[bgLayer.name] = bgLayer
        self._layersStack.append(bgLayer)
        self.activeLayer = bgLayer

    def addLayer(self, lay, name):
        # find an unused name
        usedNames = [l.name for l in self._layersStack]
        a = 1
        trialname = name
        while trialname in usedNames:
            trialname = name + '_'+ str(a)
            a = a+1
        lay.name = trialname
        #lay.window = None
        self._layers[lay.name] = lay
        self._layersStack.append(lay)
        lay.meta = self.meta
        if lay.name != 'drawlayer':
            lay.updatePixmap()

    def addAdjustmentLayer(self, name='', window=None):
        lay = QLayer(QImg=self)
        lay.inputImg = QImage(self.size(), self.format())
        self.addLayer(lay, name)
        # paint image from lower layers
        qp = QPainter(lay.inputImg)
        for l in self._layersStack:
            qp.drawImage(QRect(0, 0, l.width(), l.height()), l)
            """
            if l.qPixmap is not None:
                qp.drawPixmap(QRect(0, 0, l.width(), l.height()), l.qPixmap) # don't use drawPixmap , otherwise color management will be applied twice
            else:
                qp.drawImage(QRect(0,0, l.width(), l.height()), l)
            """
        lay.window = window
        return lay

    def updatePixmaps(self):
        for l in self._layersStack:
            l.updatePixmap()

    def refreshLayer(self, layer1, layer2):
        if layer1.name != layer2.name:
            print 'invalid layer refresh'
            return
        self._layers[layer1.name]=layer2
        i=self._layerStack.index(layer1)
        self._layerStack[i] = layer2


    def cvtToGray(self):
        self.cv2Img = cv2.cvtColor(self.cv2Img, cv2.COLOR_BGR2GRAY)
        #self.qImg = gray2qimage(self.cv2Img)

    def save(self, filename):
        img = QImage(self.width(), self.height(), self.format())
        qpainter = QPainter(img)
        for layer in self._layersStack:
            if layer.visible:
                if layer.qPixmap is not None:
                    qpainter.drawPixmap(0,0, layer.transfer()
                                  )
                else:
                    qpainter.drawImage(0,0, layer)

        img.save(filename)

class imImage(mImage) :
    """
    Interactive multilayer image
    """
    def __init__(self, *args, **kwargs):
        super(imImage, self).__init__(*args, **kwargs)
        self.Zoom_coeff = 1.0
        self.xOffset, self.yOffset = 0, 0

        drawLayer = QLayer(QImage(self.width(), self.height(), QImage.Format_ARGB32))
        drawLayer.fill(QColor(0,0,0,0))
        self.addLayer(drawLayer, 'drawlayer')

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
        for k, l  in enumerate(self._layersStack):
            if l.name != "background" and l.name != 'drawlayer':
                img = QLayer.fromImage(self._layers[l.name].resize(pixels, interpolation=interpolation))
                rszd._layersStack.append (img)
                rszd._layers[l.name] = img

        return rszd

class QLayer(vImage):
    @classmethod
    def fromImage(cls, mImg):
        mImg.visible = True
        mImg.alpha = 255
        mImg.window = None
        # for adjustment layer
        mImg.transfer = lambda: mImg.qPixmap
        mImg.inputImg = None
        return mImg #QLayer(QImg=mImg) # TOD0 retype mImg as QLayer

    def __init__(self, *args, **kwargs):
        super(QLayer, self).__init__(*args, **kwargs)
        self.name='anonymous'
        self.visible = True
        self.alpha=255
        self.transfer = lambda : self.qPixmap
        self.window = None



