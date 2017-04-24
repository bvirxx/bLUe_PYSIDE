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

    @param layer: source image (type QLayer)
    @param nb_iter:
    @param mode:
    @param again:
    """
    global rect_or_mask
    inputImg = layer.inputImg()

    mask = layer.mask
    rect = layer.rect

    # set mask from selection rectangle, if any
    rectMask = np.zeros((layer.height(), layer.width()), dtype=np.uint8)
    if rect is not None:
        rectMask[rect.top():rect.bottom(), rect.left():rect.right()] = cv2.GC_PR_FGD
    else:
        rectMask = rectMask + cv2.GC_PR_FGD


    paintedMask = QImageBuffer(mask)
    finalMask = rectMask
    #paintedMask[paintedMask==255]=cv2.GC_FGD
    #paintedMask[paintedMask==0]=cv2.GC_BGD

    #np.copyto(rectMask, paintedMask[:,:,3], where=(paintedMask[:,:,1]>0)) # copy  painted (G > 0) pixels (alpha value only)

    if not((np.any(rectMask==cv2.GC_FGD) or np.any(rectMask==cv2.GC_PR_FGD)) and (np.any(rectMask==cv2.GC_BGD) or np.any(rectMask==cv2.GC_PR_BGD))):
        reply = QMessageBox()
        reply.setText('You muest select some background or foreground pixels')
        reply.setInformativeText('Use selection rectangle or mask')
        reply.setStandardButtons(QMessageBox.Ok)
        ret = reply.exec_()
        return

    bgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the background model
    fgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the foreground model

    t0 = time()

    cv2.grabCut_mtd(QImageBuffer(inputImg)[:, :, :3],
                finalMask,
                None,#QRect2tuple(img0_r.rect),
                bgdmodel, fgdmodel,
                nb_iter,
                mode)
    print 'grabcut_mtd time :', time()-t0

    # set layer.mask to returned mask
    # foreground : white, background = black
    finalMask = np.where((finalMask==cv2.GC_FGD) + (finalMask==cv2.GC_PR_FGD), 255, 0)
    buf = QImageBuffer(layer.mask)
    buf[:,:,:3] = finalMask[...,None]
    pass



