import numpy as np
from MarkedImg import imImage
from LUT3D import rgb2hsv, hsv2rgb,hsp2rgb, lutNN
from math import floor
from PyQt4.QtGui import QImage, QColor, QPainter, QBrush
from imgconvert import QImageBuffer

class hueSatModel (imImage):
    # hue rotation
    rotation=315
    # default perceptual brightness
    pb = 0.45

    @classmethod
    def colorWheel(cls, w, h, perceptualBrightness=pb):
        """
        Build a (hue, saturation) color chart imImage. All image pixels have the same
        (perceptual) brightness (default 0.45).
        :param w: image width
        :param h: image height
        :param perceptualBrightness: (perceptual) brightness of image pixels
        :return: imImage
        """

        img = hueSatModel(w, h, perceptualBrightness=perceptualBrightness)

        # image buffer (RGB type)
        imgBuf = QImageBuffer(img)

        # alpha
        imgBuf[:,:,3] = 255
        #RGB order
        imgBuf=imgBuf[:,:,:3][:,:,::-1]

        # image hue sat , pB data
        img.hsArray = np.empty(imgBuf.shape)

        #center
        cx = w / 2
        cy = h / 2

        clipping = np.zeros(imgBuf.shape, dtype=int)
        for i in range(w):
            for j in range(h):
                i1 = i - cx
                j1 = -j + cy
                m = max(abs(i1), abs(j1))
                hue = np.arctan2(j1, i1) * 180.0 / np.pi + cls.rotation
                hue = hue - floor(hue / 360.0) * 360.0
                assert hue >= 0 and hue <= 360
                sat = np.sqrt(i1 * i1 + j1 * j1) / cx
                #sat = min(1.0, sat)
                # r,g,b values
                #c = hsp2rgb(hue, sat, perceptualBrightness, trunc=False)
                img.hsArray[j,i,:]=(hue,sat, perceptualBrightness)
                if sat <= 1.0 :
                    # valid color
                    c = hsp2rgb(hue, sat, perceptualBrightness, trunc=False)
                    if c[0] <= 255 and c[1] <= 255 and c[2] <= 255:
                        imgBuf[j, i] = c
                    else:
                        imgBuf[j, i] = np.clip(c, 0, 255)
                        clipping[j,i] = True
                # sat>= 1
                else:
                    imgBuf[j, i] = 0 #np.clip(c, 0, 255)

        # mark center and clipped area
        qp = QPainter(img)
        qp.drawEllipse(cx, cy, 3,3)
        b=np.logical_xor(clipping , np.roll(clipping, 1,axis=0))
        imgBuf[b]=0

        img.updatePixmap()
        return img

    def __init__(self, w, h, picker = None, perceptualBrightness=pb):
        img = QImage(w,h, QImage.Format_ARGB32)
        super(hueSatModel, self).__init__(QImg=img)
        self.pb = perceptualBrightness

    def GetPoint(self, h, s):
        cx = self.width() / 2
        cy = self.height() / 2
        x,y = cx*s*np.cos((h-self.rotation)*np.pi/180.0), cy*s*np.sin((h-self.rotation)*np.pi/180.0)
        x,y = x + cx, - y + cy
        #return cx*s*np.cos((h-315)*np.pi/180.0), cx*s*np.sin((h-315)*np.pi/180.0)
        return x,y

    def colorPickerSetmark(self, r,g,b, LUT3D):
        h,s,v = rgb2hsv(r,g,b, perceptual=True)
        #r1,g1,b1=hsp2rgb(h,s,self.pb)
        #print r,r1,b,b1,g,g1
        #assert abs(r-r1)<20 and abs(g-g1)<20 and abs(b-b1)<20
        i,j= self.GetPoint(h, s)
        p =QPainter(self)
        #p.setBrush(QBrush(QColor(255,255,255)))
        p.drawEllipse(int(i),int(j),5,5)
        self.updatePixmap()
        #print '********************************** draw ellipse', int(i),int(j)
        tmp = lutNN(LUT3D, r, g, b)
        print 'NN', tmp, r,g,b, LUT3D[tmp[0], tmp[1], tmp[2]]
        l=[lutNN(LUT3D, *hsp2rgb(h, s, p / 100.0)) for p in range(100)]

        for t in l:
            LUT3D[t] = [0,0,0]

class pbModel (imImage):

    @classmethod
    def colorChart(cls, w, h, hue, sat):

        img = pbModel(w, h)
        # image buffer (RGB type)
        imgBuf = QImageBuffer(img)

        # alpha
        imgBuf[:,:,3] = 255
        #RGB order
        imgBuf=imgBuf[:,:,:3][:,:,::-1]


        clipping = np.zeros(imgBuf.shape, dtype=int)
        for i in range(w):
            for j in range(h):
                p= i / float(w)
                # r,g,b values
                c = hsp2rgb(hue, sat, p , trunc=False)

                if sat <=1.0 :
                    # valid color
                    if c[0] <= 255 and c[1] <= 255 and c[2] <= 255:
                        imgBuf[j, i] = c
                    else:
                        imgBuf[j, i] = np.clip(c, 0, 255)
                        clipping[j,i] = True
                # sat>= 1
                else:
                    imgBuf[j, i] = 0 #np.clip(c, 0, 255)

        # mark center and clipped area
        #qp = QPainter(img)
        #b=np.logical_xor(clipping , np.roll(clipping, 1,axis=0))
        #imgBuf[b]=0

        img.updatePixmap()
        return img

    def __init__(self, w, h):
        img = QImage(w,h, QImage.Format_ARGB32)
        super(pbModel, self).__init__(QImg=img)



"""
ptr = img.bits()
ptr.setsize(img.byteCount())

## copy the data out as a string
strData = ptr.asstring()

## get a read-only buffer to access the data
buf = buffer(ptr, 0, img.byteCount())

## view the data as a read-only numpy array
import numpy as np
arr = np.frombuffer(buf, dtype=np.ubyte).reshape(img.height(), img.width(), 4)

## view the data as a writable numpy array
arr = np.asarray(ptr).reshape(img.height(), img.width(), 4)
"""