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
import cv2
from time import time

from PySide.QtGui import QHBoxLayout
from PySide.QtGui import QImage
from PySide.QtGui import QPushButton
from PySide.QtGui import QWidget

from imgconvert import QImageBuffer


class segmentForm(QWidget):

    @classmethod
    def getNewWindow(cls, targetImage=None):
        wdgt = segmentForm(targetImage=targetImage)
        pushButton = QPushButton('apply', parent=wdgt)
        hLay = QHBoxLayout()
        wdgt.setLayout(hLay)
        hLay.addWidget(pushButton)
        pushButton.clicked.connect(lambda x : wdgt.execute())
        return wdgt

    def __init__(self, targetImage=None):
        super(segmentForm, self).__init__()
        self.targetImage=targetImage

    def execute(self):
        do_grabcut(self.targetImage.getActiveLayer(), preview=-1, nb_iter=1, mode=cv2.GC_INIT_WITH_MASK, again=False)


def do_grabcut(layer, preview=-1, nb_iter=1, mode=cv2.GC_INIT_WITH_MASK, again=False):
    """
    segment source MImage instance.

    :param inputImg: input image (type QLayer)
    :param nb_iter:
    :param mode
    :return:
    """
    global rect_or_mask
    inputImg = layer.inputImg

    mask = layer.mask #State['rawMask']

    img0_r=inputImg

    # set mask from selection rectangle
    rectMask = np.zeros((layer.height(), layer.width()), dtype=np.uint8)
    rectMask[layer.rect.top():layer.rect.bottom(), layer.rect.left():layer.rect.right()] = cv2.GC_PR_FGD

    if not again:
        #get painted values in BGRA order
        paintedMask = QImageBuffer(layer.mask)
        paintedMask[paintedMask==255]=cv2.GC_FGD
        paintedMask[paintedMask==0]=cv2.GC_BGD

        #np.copyto(rectMask, paintedMask[:,:,1], where=(paintedMask[:,:,3]>0)) # copy  painted (A > 0) pixels (G value only)

        #if mask is not None:
            #np.copyto(rectMask, mask, where=(np.logical_and((mask==0),(paintedMask[:,:,0]==0))))

        mask=rectMask
        rect_or_mask=0
    else:
        if mask is None:
            mask=rectMask
            print "None mask"
        else:
            print "reuse mask"


    bgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the background model
    fgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the foreground model

    t0 = time()
    #if preview >0:
        #img0_r=img0_r.resize(preview)
        #mask=cv2.resize(mask, (img0_r.width(), img0_r.height()), interpolation=cv2.INTER_NEAREST)
        #a=img0_r.cv2Img()
    #cv2.grabCut_mtd(img0_r.cv2Img()[:,:,:3],
    cv2.grabCut_mtd(QImageBuffer(inputImg)[:, :, :3],
                rectMask,
                None,#QRect2tuple(img0_r.rect),
                bgdmodel, fgdmodel,
                nb_iter,
                mode)
    print 'grabcut_mtd time :', time()-t0

    #img0_r = inputImg
    #if preview >0:
        #mask=cv2.resize(mask, (inputImg.width(), inputImg.height()), interpolation=cv2.INTER_NEAREST)

    #State['rawMask'] = mask
    # apply mask
    current_mask = rectMask

    current_mask = np.where((current_mask == cv2.GC_FGD) + (current_mask == cv2.GC_PR_FGD), 255, 0)
    #current_mask = np.where((current_mask == cv2.GC_FGD) + (current_mask == cv2.GC_PR_FGD), 1, 0.4)

    buf = QImageBuffer(layer.mask)

    buf[:,:,3] = current_mask

    return

    tmp = np.copy(img0_r.cv2Img())

    tmp[:, :, 3] = tmp[:, :, 3] * mask_s1 # cast float to uint8

    img1= imImage(cv2Img=tmp, cv2mask=current_mask)
    #display
    #window.label_2.repaint()

    b=np.zeros((img0_r.height(), img0_r.width()), dtype=np.uint8)
    c=np.zeros((img0_r.height(), img0_r.width()), dtype=np.uint8)
    b[:,:]=255
    alpha = ((1 - mask) * 255).astype('uint8')
    #cv2mask = cv2.resize(np.dstack((b, c, c, alpha)), (img0.qImg.width(), img0.qImg.height()), interpolation=cv2.INTER_NEAREST)
    cv2mask = np.dstack((c, c, b, alpha))
    inputImg._layers['masklayer']=QLayer(QImg=ndarrayToQImage(cv2mask))
    #img0.drawLayer=mImage(QImg=ndarrayToQImage(cv2mask))
    #img1=imImage(cv2Img=cv2.inpaint(img1.cv2Img[:,:,:3], mask_s, 20, cv2.INPAINT_NS), format=QImage.Format_RGB888)
    return img1


