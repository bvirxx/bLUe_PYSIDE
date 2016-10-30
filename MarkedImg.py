import cv2
from imgconvert import *
import time
class MImage:
    def __init__(self, filename=None, cv2Img=None, copy=False):
        self.rect, self.mask = None, None
        self.Zoom_coeff = 1.0
        self.xOffset, self.yOffset,self.width, self.height = 0,0,0,0
        self.qImg, self.QFormat= None, None
        self.QFormat, self.ROI = None, None

        if filename is None and cv2Img is None:
            return
        elif filename is not None:
            self.cv2Img = cv2.imread(filename)
            self.cv2Img.astype(np.uint8)
            if self.cv2Img is None:
                print 'read error'
                raise Exception('cv2 read error %s' % filename)
            #else:
                #print 'read image format %s' % (self.cv2Img.shape, )
            if len(self.cv2Img.shape)<=2:               # gray image
                self.cv2Img = cv2.cvtColor(self.cv2Img, cv2.COLOR_GRAY2BGRA) # convert to BGRA
            elif self.cv2Img.shape[2]==3:               # BGR image
                self.cv2Img = cv2.cvtColor(self.cv2Img, cv2.COLOR_BGR2BGRA)  # convert to BGRA
            else:                                       # already BGRA image
                pass
                # self.cv2Img = cv2.cvtColor(self.cv2Img, cv2.COLOR_BGRA2RGBA) # convert to RGBA
        elif cv2Img is not None:
            if copy:
                self.cv2Img=cv2Img.copy()
            else:
                self.cv2Img=cv2Img
        self.set_cv2Img(self.cv2Img)

    def set_cv2Img (self, cv2Img):
        self.cv2Img = cv2Img
        self.width = self.cv2Img.shape[1]
        self.height = self.cv2Img.shape[0]
        if len(self.cv2Img.shape) >= 3:
            self.qImg = ndarrayToQimage(self.cv2Img)                # ARGB32 QImage
            self.QFormat = QtGui.QImage.Format_ARGB32
        else:
            self.qImg = gray2qimage(self.cv2Img)
            self.QFormat = QtGui.QImage.Format_Indexed8
        #self.mask = np.zeros(self.cv2Img.shape[:2],dtype = np.uint8)
        self.mask = QtGui.QImage(self.cv2Img.shape[1], self.cv2Img.shape[0], QtGui.QImage.Format_ARGB32)
        self.mask.fill(0)
        #self.rect = QtCore.QRect(100,100, self.width - 200,self.height -200)
        self.ROI= QtCore.QRect(0,0, self.width, self.height)

    def cvtToGray(self):
        self.cv2Img = cv2.cvtColor(self.cv2Img, cv2.COLOR_BGR2GRAY)
        self.qImg = gray2qimage(self.cv2Img)
        self.QFormat = QtGui.QImage.Format_Indexed8

    def resize(self, pixels):
        ratio=self.width/float(self.height)
        w,h=int(np.sqrt(pixels*ratio)), int(np.sqrt(pixels/ratio))
        hom = w/float(self.width)
        cv2Img=cv2.resize(self.cv2Img, (w,h), interpolation=cv2.INTER_CUBIC)
        tmp = MImage(cv2Img=cv2Img)
        if not (self.rect is None):
            tmp.rect = QtCore.QRect(self.rect.left()*hom, self.rect.top()*hom, self.rect.width()*hom, self.rect.height()*hom)
        if not (self.mask is None):
            #tmp.mask=cv2.resize(self.mask, (w,h), interpolation=cv2.INTER_NEAREST )
            tmp.mask = self.mask.scaled(w,h)
        return tmp
