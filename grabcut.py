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

from PySide2.QtCore import Qt
#from PySide2.QtGui import QHBoxLayout, QMessageBox, QPushButton, QWidget, QSizePolicy, QVBoxLayout, QColor, QPainter
from PySide2.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QWidget, QSizePolicy, QVBoxLayout
from imgconvert import QImageBuffer


class segmentForm(QWidget):
    """
    Segmentation layer form
    """

    @classmethod
    def getNewWindow(cls, targetImage=None, layer=None, mainForm=None):
        wdgt = segmentForm(targetImage=targetImage, layer=layer, mainForm=mainForm)
        return wdgt

    def __init__(self, targetImage=None, layer=None, mainForm=None):
        super(segmentForm, self).__init__()
        self.setWindowTitle('grabcut')
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(200, 200)
        self.setAttribute(Qt.WA_DeleteOnClose)
        hLay = QVBoxLayout()
        hLay.setAlignment(Qt.AlignTop)
        hLay.setContentsMargins(20, 8, 20, 25)  # left, top, right, bottom
        pushButton = QPushButton('apply')
        hLay.addWidget(pushButton)

        self.setLayout(hLay)
        pushButton.clicked.connect(lambda: self.execute())
        self.targetImage=targetImage
        self.mainForm = mainForm
        layer.maskIsEnabled = True
        layer.maskIsSelected = True
        buf = QImageBuffer(layer.mask)
        buf[:,:,:] = 255

    def execute(self):
        segmentLayer = self.targetImage.getActiveLayer()
        self.do_grabcut(segmentLayer, nb_iter=1, mode=cv2.GC_INIT_WITH_MASK, again=False)

    def do_grabcut(self, layer, nb_iter=1, mode=cv2.GC_INIT_WITH_MASK, again=False):
        """

        @param layer: source image (type QLayer)
        @param nb_iter:
        @param mode:
        @param again:
        """
        inputImg = layer.inputImg()

        rect = layer.rect
        # resizing coeff fitting selection rectangle with current image
        r = inputImg.width() / layer.width()

        # set mask from selection rectangle, if any
        rectMask = np.zeros((inputImg.height(), inputImg.width()), dtype=np.uint8)
        if rect is not None:
            rectMask[int(rect.top()*r):int(rect.bottom()*r), int(rect.left()*r):int(rect.right()*r)] = cv2.GC_PR_FGD
        else:
            rectMask = rectMask + cv2.GC_PR_FGD

        scaledMask = layer.mask.scaled(inputImg.width(), inputImg.height())
        paintedMask = QImageBuffer(scaledMask)
        #paintedMask = QImageBuffer(layer.mask)
        # CAUTION: mask is initialized to 255, thus discriminant is blue=0 for FG and green=0 for BG 'cf. Blue.mouseEvent())
        rectMask[(paintedMask[:, :, 0] == 0)*(paintedMask[:,:,1]==255)] = cv2.GC_FGD
        rectMask[(paintedMask[:, :, 0] == 0)*(paintedMask[:, :, 1] == 1)] = cv2.GC_PR_FGD

        #rectMask[paintedMask[:, :,1]==0 ] = cv2.GC_BGD
        rectMask[(paintedMask[:, :, 1] == 0) * (paintedMask[:,:,0]==255)] = cv2.GC_BGD
        rectMask[(paintedMask[:, :, 1] == 0) * (paintedMask[:, :, 0] == 1)] = cv2.GC_PR_BGD

        finalMask = rectMask

        if not((np.any(finalMask==cv2.GC_FGD) or np.any(finalMask==cv2.GC_PR_FGD)) and (np.any(finalMask==cv2.GC_BGD) or np.any(finalMask==cv2.GC_PR_BGD))):
            reply = QMessageBox()
            reply.setText('You must select some background or foreground pixels')
            reply.setInformativeText('Use selection rectangle or mask')
            reply.setStandardButtons(QMessageBox.Ok)
            ret = reply.exec_()
            return

        bgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the background model
        fgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the foreground model

        t0 = time()
        #tmp =inputImg.getHspbBuffer().astype(np.uint8)
        #cv2.grabCut_mtd(tmp[:,:,:3], #QImageBuffer(inputImg)[:, :, :3],
        #cv2.grabCut_mtd(QImageBuffer(inputImg)[:, :, :3],
        cv2.grabCut(QImageBuffer(inputImg)[:, :, :3],
                    finalMask,
                    None,#QRect2tuple(img0_r.rect),
                    bgdmodel, fgdmodel,
                    nb_iter,
                    mode)
        #print 'grabcut_mtd time :', time()-t0
        buf = QImageBuffer(scaledMask)
        # reset image mask to black
        buf[:,:,:3] = 0
        buf[:,:,3] = 255

        # set opacity (255=background, 0=foreground)
        buf[:, :, 3] = np.where((finalMask == cv2.GC_FGD) + (finalMask == cv2.GC_PR_FGD), 0, 255)
        # dilate background
        #kernel = np.ones((5, 5), np.uint8)
        #buf = cv2.dilate(buf, kernel, iterations=1)
        #buf = cv2.erode(buf, kernel, iterations=1)

        # R  G  B
        # *  0 255  background
        # *  1 255  probably background
        # * 255 0  foreground
        # * 255 1  probably foreground
        # *
        # set Green channel(255=foreground, 0=background, 1=PR_BGD)
        buf[:, :,1] = np.where(buf[:,:,3]==0, 255, 0)
        buf[:, :, 1][finalMask == cv2.GC_PR_BGD] = 1
        # set Blue channel(255=background, 0=foreground, 1=PR_FGD)
        buf[:,:,0] = np.where(buf[:,:,3]==255, 255, 0)
        buf[:, :, 0][finalMask == cv2.GC_PR_FGD] = 1
        # invert mask opacity
        buf[:,:,3] = 255 - buf[:,:,3]
        layer.mask = scaledMask.scaled(layer.width(), layer.height())
        # update
        layer.updatePixmap()


