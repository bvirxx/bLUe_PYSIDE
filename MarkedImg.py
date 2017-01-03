import cv2
from imgconvert import *
from PyQt4.QtGui import QPixmap, QImage, QColor
from PyQt4.QtCore import QRect, QByteArray
from icc import convert, convertQImage
from LUT3D import interp, LUT3D, interpVec
from multiprocessing import Pool
from time import time


def fa(*args):
    return [interp(LUT3D, *x) for x in args]

class vImage(QImage):
    """
    versatile image class.
    This is the base class for all multi-layered and interactive image classes
    """

    def fromQImage(cls, qImg):
        qImg.rect, qImg.mask = None, None
        qImg.qPixmap = QPixmap.fromImage(qImg)
        qImg.__class__ = vImage
        return qImg

    def __init__(self, filename=None, cv2Img=None, QImg=None, cv2mask=None, copy=False, format=QImage.Format_ARGB32,
                 colorSpace=-1, orientation=None, metadata=None, profile=''):
        self.rect, self.mask, self.colorSpace, self.metadata, self.profile = None, cv2mask, colorSpace, metadata, profile
        self.cv2Cache = None
        self.cv2CacheType=None
        if (filename is None and cv2Img is None and QImg is None):
            # create a null image
            super(vImage, self).__init__()
        if filename is not None:
            # load file
            if orientation is not None:
                tmp =QImage(filename).transformed(orientation)
            else:
                tmp = QImage(filename)

            super(vImage, self).__init__(tmp)
            if orientation is not None:
                self.transformed(orientation)
        elif QImg is not None:
            if copy:
                #d o a deep copy
                super(vImage, self).__init__(QImg.copy())
            else:
                # do a shallow copy
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
        """
        if self.colorSpace == 1 or len(self.profile)> 0:
            print self.colorSpace
            cvqim=convertQImage(self)
            self.qPixmap = QPixmap.fromImage(cvqim)
        else:
            self.qPixmap = QPixmap.fromImage(self)
        """
        if self.mask is None:
            self.mask = QImage(self.width(), self.height(), QImage.Format_ARGB32)
            self.mask.fill(0)

    def updatePixmap(self):
        if self.colorSpace == 1 or len(self.profile)> 0:
            print self.colorSpace
            cvqim=convertQImage(self)
            self.qPixmap = QPixmap.fromImage(cvqim)
        else:
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
        Resize an image while keeping its aspect ratio.
        The original image is not modified.
        :param pixels: pixel count for the resized image
        :return the resized vImage
        """
        ratio = self.width() / float(self.height())
        w, h = int(np.sqrt(pixels * ratio)), int(np.sqrt(pixels / ratio))
        hom = w / float(self.width())
        # resize
        cv2Img = cv2.resize(self.cv2Img(), (w, h), interpolation=interpolation)
        # create new vImage
        rszd = vImage(cv2Img=cv2Img)

        #resize rect and mask
        if self.rect is not None:
            rszd.rect = QRect(self.rect.left() * hom, self.rect.top() * hom, self.rect.width() * hom, self.rect.height() * hom)
        if self.mask is not None:
            # tmp.mask=cv2.resize(self.mask, (w,h), interpolation=cv2.INTER_NEAREST )
            rszd.mask = self.mask.scaled(w, h)
        return rszd

    def applyLUT(self, LUT, widget=None):

        # get image array
        ndImg = self.cv2Img(cv2Type='RGB')

        # apply LUT
        convertedNdImg = LUTArray(ndImg, LUT)
        convertedNdImg = convertedNdImg[:, :, :].astype(np.uint8)

        # update Pixmap ans repaint
        self.qPixmap = QPixmap.fromImage(ndarrayToQImage(convertedNdImg, QImage.Format_RGB888))
        if widget is not None:
            widget.repaint()

    def apply3DLUT(self, LUT, widget=None):

        # get image buffer (type RGB)
        ndImg = QImageBuffer(self)[:,:,:3][:,:,::-1]
        start=time()
        """
        f = lambda *args : [interp(LUT, *x) for x in args]

        ndImg[:] = map(f, * [ndImg[:,c] for c in range(ndImg.shape[1])])
        """
        ndImg[:,:,:] = interpVec(LUT, ndImg)
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
    Multi-layer image
    """
    def __init__(self, *args, **kwargs):
        super(mImage, self).__init__(*args, **kwargs)
        self._layers = {}
        self._layersStack = []
        # add background layer
        self.addLayer(QLayer.frommImage(self), 'background')

    def addLayer(self, layer, name):
        self._layers[name] = layer
        self._layersStack.append(layer)

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


class imImage(mImage) :
    """
    Interactive multi layer image
    """
    def __init__(self, *args, **kwargs):
        #__init__(self, filename=None, cv2Img=None, QImg=None, cv2mask=None, copy=False, format=QImage.Format_ARGB32, colorSpace=-1, orientation=None) :
        super(imImage, self).__init__(*args, **kwargs) #(filename=filename, cv2Img=cv2Img, QImg=QImg, cv2mask=cv2mask, copy=copy, format=format, colorSpace=colorSpace, orientation=orientation)
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
        rszd0 = super(imImage, self).resize(pixels)
        rszd = imImage(QImg=rszd0)
        rszd.rect = rszd0.rect
        for k in self._layers.keys():
            if k != "background":
                rszd._layers[k]=self._layers[k].resize(pixels, interpolation=cv2.INTER_NEAREST)
        """
        rszd.drawLayer = QImage(rszd.qImg.width(), rszd.qImg.height(), QImage.Format_ARGB32)
        rszd.drawLayer.fill(QColor(0, 0, 0, 0))
        rszd.layers.append(rszd.drawLayer)
        """
        return rszd

class LUTArray(object):
    """
    Array wrapper for LUT conversion
    """
    def __init__(self, a, LUT):
        self.buf=a
        self.LUT=LUT

    def __getitem__(self, item):
        return self.LUT[self.buf[item]]

class LUT3DArray(object):
    """
    Array wrapper for 3D LUT conversion
    """
    def __init__(self, a, LUT):
        self.buf=a
        self.LUT=LUT

    def __getitem__(self, item):
        if isinstance(item[0], slice):
            return np.array([[ interp(self.LUT, *self.buf[y,x])  for x in range(self.buf.shape[1])] for y in range(self.buf.shape[0])])
        else:
            return interp(self.LUT, self.buf[item])

class QLayer(vImage):
    @classmethod
    def frommImage(cls, mImg):
        mImg.visible = True
        mImg.alpha = 255
        mImg.transfer = lambda: mImg.qPixmap
        return mImg #QLayer(QImg=mImg)

    def __init__(self, *args, **kwargs):
        super(QLayer, self).__init__(*args, **kwargs)
        self.visible = True
        self.alpha=255
        self.transfer = lambda : self.qPixmap



