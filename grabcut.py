"""
Copyright (C) 2017  Bernard Virot

bLUe - Photo editing software.

With Blue you can enhance and correct the colors of your photos in a few clicks.
No need for complex tools such as lasso, magic wand or masks.
bLUe interactively constructs 3D LUTs (Look Up Tables), adjusting the exact set
of colors you want.

3D LUTs are widely used by professional film makers, but the lack of
interactive tools maked them poorly useful for photo enhancement, as the shooting conditions
can vary widely from an image to another. With bLUe, in a few clicks, you select the set of
colors to modify, the corresponding 3D LUT is automatically built and applied to the image.
You can then fine tune it as you want.

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
import cv2
from time import time
from PyQt4.QtGui import QWidget


class segmentForm(QWidget):

    @classmethod
    def getNewWindow(cls):
        wdgt = segmentForm()
        pass

    def __init__(self):
        super(segmentForm, self)



def do_grabcut(img0, preview=-1, nb_iter=1, mode=cv2.GC_INIT_WITH_RECT, again=False):
    """
    segment source MImage instance.

    :param img0: source Mimage, unmodified.
    :param preview:
    :param nb_iter:
    :param mode
    :return:
    """
    #img0.rect = QRect(500, 400, Mimg.width() - 2000, Mimg.height() - 1000)

    print '********* do_grabCut call'
    mask_s = State['rawMask']
    global rect_or_mask

    #if preview>0:
        #img0_r=img0.resize(preview)
    #else:
    img0_r=img0

    # set rect mask
    rectMask = np.zeros((img0_r.height(), img0_r.width()), dtype=np.uint8)
    rectMask[img0_r.rect.top():img0_r.rect.bottom(), img0_r.rect.left():img0_r.rect.right()] = cv2.GC_PR_FGD

    if not again:
        #get painted values in BGRA order
        paintedMask = QImageBuffer(img0_r._layers['drawlayer'])

        paintedMask[paintedMask==255]=cv2.GC_FGD
        paintedMask[paintedMask==0]=cv2.GC_BGD

        np.copyto(rectMask, paintedMask[:,:,1], where=(paintedMask[:,:,3]>0)) # copy  painted (A > 0) pixels (G value only)

        if mask_s is not None:
            np.copyto(rectMask, mask_s, where=(np.logical_and((mask_s==0),(paintedMask[:,:,0]==0))))

        mask_s=rectMask
        rect_or_mask=0
    else:
        if mask_s is None:
            mask_s=rectMask
            print "None mask"
        else:
            print "reuse mask"

    bgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the background model
    fgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the foreground model

    t0 = time.time()
    if preview >0:
        img0_r=img0_r.resize(preview)
        mask_s=cv2.resize(mask_s, (img0_r.width(), img0_r.height()), interpolation=cv2.INTER_NEAREST)
        #a=img0_r.cv2Img()
    #cv2.grabCut_mtd(img0_r.cv2Img()[:,:,:3],
    cv2.grabCut_mtd(img0_r.cv2Img()[:, :, :3],
                mask_s,
                None,#QRect2tuple(img0_r.rect),
                bgdmodel, fgdmodel,
                nb_iter,
                mode)
    print 'grabcut_mtd time :', time.time()-t0

    img0_r = img0
    if preview >0:
        mask_s=cv2.resize(mask_s, (img0.width(), img0.height()), interpolation=cv2.INTER_NEAREST)

    State['rawMask'] = mask_s
    # apply mask
    current_mask = mask_s

    mask_s = np.where((current_mask == cv2.GC_FGD) + (current_mask == cv2.GC_PR_FGD), 1, 0)
    mask_s1 = np.where((current_mask == cv2.GC_FGD) + (current_mask == cv2.GC_PR_FGD), 1, 0.4)

    tmp = np.copy(img0_r.cv2Img())

    tmp[:, :, 3] = tmp[:, :, 3] * mask_s1 # cast float to uint8

    img1= imImage(cv2Img=tmp, cv2mask=current_mask)
    #display
    #window.label_2.repaint()

    b=np.zeros((img0_r.height(), img0_r.width()), dtype=np.uint8)
    c=np.zeros((img0_r.height(), img0_r.width()), dtype=np.uint8)
    b[:,:]=255
    alpha = ((1 - mask_s) * 255).astype('uint8')
    #cv2mask = cv2.resize(np.dstack((b, c, c, alpha)), (img0.qImg.width(), img0.qImg.height()), interpolation=cv2.INTER_NEAREST)
    cv2mask = np.dstack((c, c, b, alpha))
    img0._layers['masklayer']=QLayer(QImg=ndarrayToQImage(cv2mask))
    #img0.drawLayer=mImage(QImg=ndarrayToQImage(cv2mask))
    #img1=imImage(cv2Img=cv2.inpaint(img1.cv2Img[:,:,:3], mask_s, 20, cv2.INPAINT_NS), format=QImage.Format_RGB888)
    return img1


