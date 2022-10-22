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
from PySide6.QtCore import QRect

from bLUeTop.utils import array2DSlices


def show(bufList):
    i = 0
    for buf in bufList:
        cv2.namedWindow("output%d" % i, cv2.WINDOW_NORMAL)
        cv2.imshow("output%d" % i, buf)
        i += 1
    cv2.waitKey(0)


def createLineIterator(P1, P2, img):
    """
    Produces an array that consists of the coordinates and colors of
    pixels in the line joining the points P1 and P2.
    Taken from https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator
    :param P1:
    :type P1: 2-uple
    :param P2:
    :type P2: 2-uple
    :param img:
    :type img: ndarray shape (h, w, 3)
    :return: a numpy array that consists of the coordinates and intensities of each pixel in the radii
    :rtype: ndarray shape: [numPixels, 3], row = [x,y,intensity]
    """
    imageH, imageW = img.shape[:2]
    P1X, P1Y = P1
    P2X, P2Y = P2

    # difference and absolute difference between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]
    return itbuffer


def alphaBlend(imgBuf1, imgBuf2, mask):
    """
    Returns the blending of imgBuf2 and imgBuf1 defined by mask.
    The pixels in the unmasked (resp. masked) region are set to those of imgBuf1 (resp. imgBuf2).

    :param imgBuf1: image buffer
    :type imgBuf1: ndArray
    :param imgBuf2: image buffer
    :type imgBuf2: ndArray
    :param mask: values in 0..255
    :type mask: ndarray, shape(h, w)
    :return:
    :rtype: ndarray
    """
    # convert mask to 0/1
    mask = mask / 255.0
    mask = mask[..., np.newaxis]
    tmp = (imgBuf1 - imgBuf2) * mask
    tmp[...] += imgBuf2
    return tmp  # (imgBuf1 - imgBuf2) * mask + imgBuf2


def contours(maskBuf, thres=0):
    """
    Retrieve the contours of a 1-channel image.
    The image is first converted to a 0/255 binary image,
    using the threshold thres (default 0).
    Contours are returned as a 0/255 image.

    :param maskBuf: 1-channel image.
    :type maskBuf: ndarray shape=(h, w)
    :param thres: binary threshold
    :type thres: int
    :return: list of contours
    :rtype: list of vectors; each vector is a list of 2-uples of point coordinates.
    """
    _, binary = cv2.threshold(maskBuf, thres, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
        -2]  # old versions of findContours may return a 3-uple
    return contours


def moments(maskBuf):
    """
    Returns the moment dict for maskBuf.

    :param maskBuf:
    :type maskBuf: ndarray
    :return:
    :rtype: dict
    """
    return cv2.moments(maskBuf)


def membrane(inMBuf, maskBuf, maskContour, passes=1):
    """
    Calculates the harmonic function with boundary
    values imgBuf on the contour of
    maskBuf (Dirichlet conditions), using either the Jacobi Method or the Gauss Seidel Method:
    http://www.math.umbc.edu/~kogan/technical_papers/2007/Yang_Gobbert.pdf
    https://web.stanford.edu/class/cme324/saad.pdf
    The Laplacian kernel is applied to the interior
    of the unmasked region only. The exterior is
    returned unmodified.
    The input arrays imgBuf, maskBuf and maskContour are not modified.

    :param inMBuf: boundary values
    :type inMBuf: ndarray, shape (h, w, d)
    :param maskBuf: mask image
    :type maskBuf: ndarray, shape (h, w)
    :param maskContour:
    :type maskContour:
    :param passes: number of grid refinements
    :type passes: int
    :return: membrane buffer
    :rtype: ndarray, shape (h, w, d), dtype=np.float
    """
    steps = [33, 17, 9, 5]
    passes = min(max(1, passes), len(steps))
    # get the interior of the unmasked region (remove contour)
    innerRegion = (maskContour != 255) & (maskBuf == 255)
    ##############
    # stop criterion
    threshold = 1
    ##############
    dBuf = inMBuf.copy()
    # compute means per color channel over contour
    m = np.mean(dBuf[maskContour == 255], axis=0)
    if np.any(np.isnan(m)):
        return dBuf
    # init the interior area
    dBuf[innerRegion] = m
    # solve Laplace equation using a grid with unit cells of size step.
    # init the laplacian kernel
    r = 3
    lpKernel = np.zeros((r, r), dtype=np.float)
    lpKernel[0, r // 2], lpKernel[r // 2, 0], lpKernel[r - 1, r // 2], lpKernel[r // 2, r - 1] = (0.25,) * 4
    lpKernel[r // 2, r // 2] = 0.0
    for step in steps[:passes]:  # [33, 17]:
        bMask1 = innerRegion[::step, ::step]
        # init each grid vertex with the mean of dBuf values over a (step,step) neighborhood
        # and restore initial values for contour vertices.
        buf1 = cv2.blur(dBuf, (step, step))
        buf1[maskContour == 255] = inMBuf[maskContour == 255]
        buf1 = buf1[::step, ::step, :]
        ####################################
        # evaluation of SOR coeff
        # exact formula sorCoeff = 2 / (1 + sin(pi * h)) with h = 1 / ( N + 1),
        # where  N + 2 is the (square) grid size (= width or height)
        # We approximate it by sorCoeff = 2 * (1 - pi * h)
        ####################################
        # sorCoeff = 2.0 * (1.0 - np.pi / (np.amax(buf1.shape) - 1 ))
        # iterate the Laplacian kernel
        c = 0
        while True:
            c += 1
            outBuf1 = cv2.filter2D(buf1, -1, lpKernel)
            tmpa = (buf1 - outBuf1)[bMask1]
            # update the interior region
            buf1[bMask1] = outBuf1[bMask1]
            if (max(tmpa.max(initial=0), -tmpa.min(initial=0)) < threshold) or (c > 10 * 3):  # watchdog added
                break
        # interpolate the grid for next step
        buf1 = cv2.resize(buf1, (dBuf.shape[1], dBuf.shape[0]))
    # copy the cloned region into dBuf
    dBuf[innerRegion] = buf1[innerRegion]
    maskContour = cv2.blur(maskContour, (20, 20))
    dBuf[maskContour > 64] = buf1[maskContour > 64]
    return dBuf


def seamlessClone(srcBuf, destBuf, mask, conts, bRect, srcTr, destTr, w=3, passes=1):
    """
    The area in srcBuf delimited by the mask translated by srcTr is cloned
    into the area in destBuf delimited by the mask translated by destTr.

    :param srcBuf: source image
    :type srcBuf: ndarray, shape (h, w, d)
    :param destBuf: destination image
    :type destBuf: ndarray, shape (h, w, d)
    :param mask: binary mask (0/255)
    :type mask: ndarray shape (h, w)
    :param conts: contours
    :type conts: list of ndarrays
    :param bRect:  bounding rect of cloning area
    :type bRect: QRect
    :param srcTr: mask translation in source
    :type srcTr: 2-uple
    :param destTr: mask translation in destination
    :type destTr: 2-uple
    :param w: contour thickness
    :type w: int
    :param passes: number of grid refinements
    :type passes: int
    :return: cloned image
    :rtype: ndarray
    """
    srcTr = np.array(srcTr)
    destTr = np.array(destTr)
    # convert bRect into (x, y, w, h)
    bRect = (bRect.left(), bRect.top(), bRect.width(), bRect.height())
    # translate bRect for source and dest.
    rectSrc = (bRect[0] + srcTr[0], bRect[1] + srcTr[1], bRect[2], bRect[3])
    rectDest = (bRect[0] + destTr[0], bRect[1] + destTr[1], bRect[2], bRect[3])
    # finding source and dest regions in srcBuf and destBuf respectively
    srcBufT = srcBuf[array2DSlices(srcBuf, rectSrc)]
    destBufT = destBuf[array2DSlices(destBuf, rectDest)]
    # obtaining contour of mask
    maskContour = np.zeros(mask.shape, dtype=mask.dtype)  # dest of contours
    cv2.drawContours(maskContour, conts, -1, 255, w)  # -1: draw all contours; 0: draw contour 0
    # solving Laplace equation for delta = destBufT - srcBufT
    buf = membrane(destBufT.astype(np.float) - srcBufT.astype(np.float), mask[array2DSlices(mask, bRect)],
                   maskContour[array2DSlices(maskContour, bRect)], passes=passes)
    tmp = buf + srcBufT
    np.clip(tmp, 0, 255, tmp)
    result = destBuf.copy()
    result[array2DSlices(destBuf, rectDest)] = alphaBlend(tmp, destBufT, mask[array2DSlices(mask, bRect)])
    return result
