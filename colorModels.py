import numpy as np
from MarkedImg import imImage
from LUT3D import rgb2hsv, hsv2rgb,hsp2rgb, lutNN
from math import floor
from PyQt4.QtGui import QImage, QColor, QPainter, QBrush
from imgconvert import QImageBuffer

class hueSatModel (imImage):
    rotation=315
    pb = 0.45
    def __init__(self, w, h, picker = None, perceptualBrightness=pb):
        img = QImage(w,h, QImage.Format_ARGB32)
        super(hueSatModel, self).__init__(QImg=img)
        self.pb = perceptualBrightness
        #self.picker = picker

    @classmethod
    def colorPicker(cls, w, h, perceptualBrightness=pb):
        """
        Build a (hue, saturation) color chart imImage. All image pixels have a fixed
        (perceptual) brightness (default 0.45)
        :param w: image width
        :param h: image height
        :param perceptualBrightness: (perceptual) brightness of image pixels
        :return: imImage
        """

        #img = QImage(w,h, QImage.Format_ARGB32)
        #img = imImage(QImg=img)
        img=hueSatModel(w,h, perceptualBrightness=perceptualBrightness)

        arr = QImageBuffer(img)
        #ptr = img.bits()
        #ptr.setsize(img.byteCount())
        #arr = np.asarray(ptr).reshape(img.height(), img.width(), 4)
        cx=w/2
        cy=h/2
        for i in range(w):
            for j in range(h):
                i1=i-cx
                j1=-j+cy
                m = max(abs(i1), abs(j1))
                hue = np.arctan2(j1,i1)*180.0/np.pi + cls.rotation
                hue = hue - floor(hue/360.0)*360.0
                assert hue>=0 and hue <=360
                sat = np.sqrt(i1*i1 + j1*j1)/cx
                sat = min(1.0, sat)
                #invert
                #print hue, sat, i1, j1, cx*sat*np.cos((hue-rot)*np.pi/180.0), cx*sat*np.sin((hue-rot)*np.pi/180.0)
                c =  hsp2rgb(hue,sat,perceptualBrightness, trunc=False)
                if sat<1.0 and c[0]<=255 and c[1]<= 255 and c[2]<=255:
                    arr[j,i,0], arr[j,i,1], arr[j,i,2], arr[j,i,3] = c[2], c[1], c[0], 255
                elif sat <1.0:
                    arr[j, i, 0], arr[j, i, 1], arr[j, i, 2], arr[j, i, 3] = 255, c[1], c[0], 255
                else:
                    arr[j, i, 0], arr[j, i, 1], arr[j, i, 2], arr[j, i, 3] = c[2], c[1], c[0], 128

        img.updatePixmap()
        return img

    def colorPickerGetPoint(self, h,s):
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
        i,j= self.colorPickerGetPoint(h,s)
        p =QPainter(self)
        #p.setBrush(QBrush(QColor(255,255,255)))
        p.drawEllipse(int(i),int(j),5,5)
        self.updatePixmap()
        #print '********************************** draw ellipse', int(i),int(j)
        tmp = lutNN(LUT3D, r, g, b)
        print 'NN', tmp, r,g,b, LUT3D[tmp[0], tmp[1], tmp[2]]
        l=[lutNN(LUT3D, *hsp2rgb(h, s, p / 100.0)) for p in range(100)]
        print l

        for t in l:
            LUT3D[t] = [0,0,0]

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