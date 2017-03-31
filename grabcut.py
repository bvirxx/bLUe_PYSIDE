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
from PySide.QtGui import QMessageBox
from PySide.QtGui import QPushButton
from PySide.QtGui import QWidget

from imgconvert import QImageBuffer


class segmentForm(QWidget):
    """
    Form for applying segmentation
    """

    @classmethod
    def getNewWindow(cls, targetImage=None):
        wdgt = segmentForm(targetImage=targetImage)
        pushButton = QPushButton('apply', parent=wdgt)
        hLay = QHBoxLayout()
        wdgt.setLayout(hLay)
        hLay.addWidget(pushButton)
        pushButton.clicked.connect(lambda : wdgt.execute())
        return wdgt

    def __init__(self, targetImage=None):
        super(segmentForm, self).__init__()
        self.targetImage=targetImage

    def execute(self):
        do_grabcut(self.targetImage.getActiveLayer(), nb_iter=1, mode=cv2.GC_INIT_WITH_MASK, again=False)


def do_grabcut(layer, nb_iter=1, mode=cv2.GC_INIT_WITH_MASK, again=False):
    """

    :param layer: source image (type QLayer)
    :param nb_iter:
    :param mode:
    :param again:
    :return:
    """
    global rect_or_mask
    inputImg = layer.inputImg

    mask = layer.mask
    rect = layer.rect

    # set mask from selection rectangle, if any
    rectMask = np.zeros((layer.height(), layer.width()), dtype=np.uint8)
    if rect is not None:
        rectMask[rect.top():rect.bottom(), rect.left():rect.right()] = cv2.GC_PR_FGD
    else:
        rectMask = rectMask + cv2.GC_PR_FGD


    paintedMask = QImageBuffer(mask)
    #paintedMask[paintedMask==255]=cv2.GC_FGD
    #paintedMask[paintedMask==0]=cv2.GC_BGD

    #np.copyto(rectMask, paintedMask[:,:,3], where=(paintedMask[:,:,1]>0)) # copy  painted (G > 0) pixels (alpha value only)

    if not(np.any(rectMask==cv2.GC_FGD) and np.any(rectMask==cv2.GC_BGD )):
        reply = QMessageBox()
        reply.setText('You muest select some background or foreground pixels')
        reply.setInformativeText('Use selection rectangle or mask')
        reply.setStandardButtons(QMessageBox.Ok)
        ret = reply.exec_()
        return None

    bgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the background model
    fgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the foreground model

    t0 = time()

    cv2.grabCut_mtd(QImageBuffer(inputImg)[:, :, :3],
                rectMask,
                None,#QRect2tuple(img0_r.rect),
                bgdmodel, fgdmodel,
                nb_iter,
                mode)
    print 'grabcut_mtd time :', time()-t0

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


