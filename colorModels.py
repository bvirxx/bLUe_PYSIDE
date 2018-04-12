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

import numpy as np
from MarkedImg import imImage
from colorCube import hsv2rgbVec, hsp2rgb, rgb2hsp, rgb2hspVec, hsv2rgb, rgb2hsB, rgb2hsBVec, hsp2rgbVec
from PySide2.QtGui import QImage
from imgconvert import QImageBuffer

class cmConverter(object):
    """
    Gather conversion functions for color space<-->RGB
    """

    def __init__(self):
        self.cm2rgb, self.cm2rgbVec, rgb2cm, rgb2cmVec = (None,)*4

##########################################
# init color converters for HSpB and HSB
#########################################
cmHSP = cmConverter()
cmHSP.cm2rgb, cmHSP.cm2rgbVec, cmHSP.rgb2cm, cmHSP.rgb2cmVec = hsp2rgb, hsp2rgbVec, rgb2hsp, rgb2hspVec  #TODO 12/04/18 replace hsp2rgbVecSmall by hsp2rgbVec

cmHSB = cmConverter()
cmHSB.cm2rgb, cmHSB.cm2rgbVec, cmHSB.rgb2cm, cmHSB.rgb2cmVec = hsv2rgb, hsv2rgbVec, rgb2hsB, rgb2hsBVec

class hueSatModel (imImage):
    """
    (hue, sat) color chart
    """
    # hue rotation
    rotation=315
    # default perceptual brightness
    pb = 0.45

    @classmethod
    def colorWheel(cls, w, h, cModel, perceptualBrightness=pb, border=0.0):
        """
        Build a hue, sat color chart imImage. All image pixels have the same
        (perceptual) brightness (default 0.45).
        @param w: image width
        @param h: image height
        @param cModel: color model (cmConverter object)
        @param perceptualBrightness: brightness of image pixels
        @return: imImage
        """
        w+= 2*border
        h+= 2*border
        # uninitialized ARGB image
        img = hueSatModel(w, h, cModel, perceptualBrightness=perceptualBrightness)
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
        sat = np.minimum(sat, 1.0)

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
        #imgBuf[:,:,:] = ccm2rgbVec(img.hsArray)
        imgBuf[:, :, :] = cModel.cm2rgbVec(img.hsArray)

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
        # color manage the wheel
        img.meta.colorSpace = 1
        img.updatePixmap()
        return img

    def __init__(self, w, h, cModel, picker = None, perceptualBrightness=pb):
        img = QImage(w,h, QImage.Format_ARGB32)
        super(hueSatModel, self).__init__(QImg=img)  # modify tableview selected row !!!!!
        self.pb = perceptualBrightness
        self.hsArray = None
        self.cModel = cModel

    def setPb(self,pb):
        """
        Set brightness and update image
        @param pb: perceptive brightness (range 0,..,1)
        """
        self.pb = pb
        self.hsArray[:,:,2] = pb
        imgBuf = QImageBuffer(self)[:,:,:3][:,:,::-1]
        #imgBuf[:,:,:] = ccm2rgbVec(self.hsArray)
        imgBuf[:, :, :] = self.cModel.cm2rgbVec(self.hsArray)
        self.updatePixmap()

    def GetPoint(self, h, s):
        """
        convert hue, sat values to cartesian coordinates
        on the color wheel (origin top-left corner).
        @param h: hue in range 0..1
        @param s: saturation in range 0..1
        @return: cartesian coordinates
        """
        cx = self.width() / 2
        cy = self.height() / 2
        #x,y = cx*s*np.cos((h-self.rotation)*np.pi/180.0), cy*s*np.sin((h-self.rotation)*np.pi/180.0)
        x, y = (cx-self.border) * s * np.cos((h - self.rotation) * np.pi / 180.0), (cy-self.border) * s * np.sin((h - self.rotation) * np.pi / 180.0)
        x,y = x + cx, - y + cy
        return x,y

    def GetPointVec(self, hsarray):
        """
        convert hue, sat values to cartesian coordinates
        on the color wheel (origin top-left corner).
        @param hsarray
        @return: cartesian coordinates
        """
        h, s = hsarray[:,:,0], hsarray[:,:,1]
        cx = self.width() / 2
        cy = self.height() / 2
        #x,y = cx*s*np.cos((h-self.rotation)*np.pi/180.0), cy*s*np.sin((h-self.rotation)*np.pi/180.0)
        x, y = (cx-self.border) * s * np.cos((h - self.rotation) * np.pi / 180.0), (cy-self.border) * s * np.sin((h - self.rotation) * np.pi / 180.0)
        x,y = x + cx, - y + cy
        return np.dstack((x,y))

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
    def colorChart(cls, w, h, cModel, hue, sat):

        img = pbModel(w, h, cModel)
        # image buffer (BGRA type)
        imgBuf = QImageBuffer(img)
        # set alpha
        imgBuf[:,:,3] = 255
        #RGB order
        imgBuf=imgBuf[:,:,:3][:,:,::-1]
        wF = float(w)-1
        hsArray = np.array([[[hue, sat, i/wF] for i in range(w)] for j in range(h)])
        #imgBuf[:,:,:] = ccm2rgbVec(hsArray)
        imgBuf[:, :, :] = cModel.cm2rgbVec(hsArray)
        # color manage
        img.meta.colorSpace = 1
        img.updatePixmap()
        return img

    def __init__(self, w, h, cModel):
        img = QImage(w,h, QImage.Format_ARGB32)
        super(pbModel, self).__init__(QImg=img)
        self.cModel = cModel

