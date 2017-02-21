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

import numpy as np
from MarkedImg import imImage
from LUT3D import rgb2hsB, hsv2rgbVec,hsp2rgb,  hsp2rgbVec, lutNN
from math import floor
from PyQt4.QtGui import QImage, QColor, QPainter, QBrush
from imgconvert import QImageBuffer

class hueSatModel (imImage):
    """
    (hue, sat) color chart
    """
    # hue rotation
    rotation=315
    # default perceptual brightness
    pb = 0.45

    @classmethod
    def colorWheel(cls, w, h, perceptualBrightness=pb, border=0.0):
        """
        Build a hue, sat color chart imImage. All image pixels have the same
        (perceptual) brightness (default 0.45).
        :param w: image width
        :param h: image height
        :param perceptualBrightness: brightness of image pixels
        :return: imImage
        """
        w+= 2*border
        h+= 2*border
        # uninitialized ARGB image
        img = hueSatModel(w, h, perceptualBrightness=perceptualBrightness)
        img.border = border
        # image buffer (BGRA order) dtype=uint8
        imgBuf = QImageBuffer(img)

        # set alpha channel
        imgBuf[:,:,3] = 255
        # RGB buffer
        imgBuf=imgBuf[:,:,:3][:,:,::-1]

        clipping = np.zeros(imgBuf.shape, dtype=int)

        """
        for i in range(w):
            for j in range(h):
                i1 = i - cx
                j1 = -j + cy
                hue = np.arctan2(j1, i1) * radian2degree + cls.rotation
                hue = hue - floor(hue / 360.0) * 360.0
                #assert hue >= 0 and hue <= 360
                sat = np.sqrt(i1 * i1 + j1 * j1) / cx
                sat = min(1.0, sat)
                # r,g,b values
                #c = hsp2rgb(hue, sat, perceptualBrightness, trunc=False)
                img.hsArray[j,i,:]=(hue,sat, perceptualBrightness)
        """
        #coord = np.array([[[i, -j] for i in range(w)] for j in range(h)])
        coord = np.dstack(np.meshgrid(np.arange(w), - np.arange(h)))

        # center  : i1 = i - cx, j1 = -j + cy
        cx = w / 2
        cy = h / 2
        coord = coord + np.array([-cx, cy])

        # set hue, sat from polar coordinates
        # arctan2 values are in range -pi, pi
        hue = np.arctan2(coord[:,:,1], coord[:,:,0]) * (180.0 / np.pi) + cls.rotation
        # range 0..360
        hue = hue - np.floor(hue / 360.0) * 360.0
        sat = np.linalg.norm(coord, axis=2 ,ord=2) / (cx - border)
        sat = np.minimum(sat, 0.999)

        # fixed perceptual brightness
        pb = np.zeros(hue.shape) + perceptualBrightness

        # image buffer using HSP color model
        img.hsArray = np.dstack((hue, sat, pb))

        """
        if sat <= 1.0 :
            # valid color
            c = hsp2rgbNew(hue, sat, perceptualBrightness, trunc=False)
            if c[0] <= 255 and c[1] <= 255 and c[2] <= 255:
                imgBuf[j, i] = c
            else:
                imgBuf[j, i] = np.clip(c, 0, 255)
                clipping[j,i] = True
        # sat>= 1
        else:
            imgBuf[j, i] = 0 #np.clip(c, 0, 255)
        """
        imgBuf[:,:,:] = hsp2rgbVec(img.hsArray)
        #As node colors are read from img,
        # image should not be mangled
        """
        # mark center and clipped area
        qp = QPainter(img)
        qp.drawEllipse(cx, cy, 3,3)
        qp.end()
        b=np.logical_xor(clipping , np.roll(clipping, 1,axis=0))
        imgBuf[b]=0
        img.updatePixmap()
        """
        return img

    def __init__(self, w, h, picker = None, perceptualBrightness=pb):
        img = QImage(w,h, QImage.Format_ARGB32)
        super(hueSatModel, self).__init__(QImg=img)
        self.pb = perceptualBrightness
        self.hsArray = None

    def setPb(self,pb):
        self.pb = pb
        self.hsArray[:,:,2] = pb
        imgBuf = QImageBuffer(self)[:,:,:3][:,:,::-1]
        imgBuf[:,:,:] = hsp2rgbVec(self.hsArray)

    def GetPoint(self, h, s):
        """
        convert hue, sat values to cartesian coordinates
        on the color wheel (origin top-left corner).
        :param h:
        :param s:
        :return: cartesian coordinates
        """
        cx = self.width() / 2
        cy = self.height() / 2
        #x,y = cx*s*np.cos((h-self.rotation)*np.pi/180.0), cy*s*np.sin((h-self.rotation)*np.pi/180.0)
        x, y = (cx-self.border) * s * np.cos((h - self.rotation) * np.pi / 180.0), (cy-self.border) * s * np.sin((h - self.rotation) * np.pi / 180.0)
        x,y = x + cx, - y + cy
        return x,y

    """
    def colorPickerSetmark(self, r,g,b, LUT3D):
        h,s,v = rgb2hsB(r, g, b, perceptual=True)
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
    """

class pbModel (imImage):

    @classmethod
    def colorChart(cls, w, h, hue, sat):

        img = pbModel(w, h)
        # image buffer (BGRA type)
        imgBuf = QImageBuffer(img)
        # set alpha
        imgBuf[:,:,3] = 255
        #RGB order
        imgBuf=imgBuf[:,:,:3][:,:,::-1]
        wF = float(w)-1
        hsArray = np.array([[[hue, sat, i/wF] for i in range(w)] for j in range(h)])
        imgBuf[:,:,:] = hsp2rgbVec(hsArray)
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