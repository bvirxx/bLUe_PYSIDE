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

Qt5 is licensed under the LGPL version 3,
Pyside2 is licensed under the LGPL version 2.1
exiftool Copyright © 2013-2016, Phil Harvey.exiftool is licensed under thePerl Artistic License
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh

Pillow is the friendly PIL fork. It is Copyright © 2010-2018 by Alex Clark and contributors

libRaw Copyright (C) 2008-2018 LibRaw LLC (http://www.libraw.org, info@libraw.org)
rawpy is licensed under the MIT license Copyright (c) 2014 Maik Riechert

The QtHelp module uses the CLucene indexing library
Copyright (C) 2003-2006 Ben van Klinken and the CLucene Team

Changes are Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

opencv copyright
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

pywin32 License

Unless stated in the specific source file, this work is
Copyright (c) 1996-2008, Greg Stein and Mark Hammond.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the distribution.

Neither names of Greg Stein, Mark Hammond nor the name of contributors may be used
to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
import numpy as np
import multiprocessing
import sys
import threading
from itertools import cycle
from os import path, walk
from io import BytesIO

from PIL.ImageCms import ImageCmsProfile
from os.path import basename, isfile
from types import MethodType
import rawpy

from bLUeCore.bLUeLUT3D import haldArray
from grabcut import segmentForm
from PySide2.QtCore import QRect, QEvent, QUrl, QSize, QFileInfo, QRectF, QObject
from PySide2.QtGui import QPixmap, QPainter, QCursor, QKeySequence, QBrush, QPen, QDesktopServices, QFont, \
    QPainterPath, QTransform, QContextMenuEvent, QColor, QImage
from PySide2.QtWidgets import QApplication, QAction, \
    QMainWindow, QDockWidget, QSizePolicy, QScrollArea, QSplashScreen, QWidget, \
    QStyle, QTabWidget
from QtGui1 import app, window, rootWidget
import exiftool
from graphicsBlendFilter import blendFilterForm
from graphicsInvert import invertForm
from graphicsMixer import mixerForm
from graphicsNoise import noiseForm
from graphicsRaw import rawForm
from graphicsTransform import transForm, imageForm
from bLUeGui.bLUeImage import QImageBuffer, QImageFormats
from versatileImg import vImage, metadataBag
from MarkedImg import imImage, QRawLayer
from graphicsRGBLUT import graphicsForm
from graphicsLUT3D import graphicsForm3DLUT
from lutUtils import LUTSIZE, LUT3D, LUT3DIdentity
from bLUeGui.colorPatterns import cmHSP, cmHSB
from colorManagement import icc
from graphicsCoBrSat import CoBrSatForm
from graphicsExp import ExpForm
from graphicsPatch import patchForm
from settings import USE_POOL, POOL_SIZE, THEME, MAX_ZOOM, TABBING
from utils import QbLUeColorDialog, colorInfoView
from bLUeGui.tool import cropTool, rotatingTool
from graphicsTemp import temperatureForm
from time import sleep
import gc
from graphicsFilter import filterForm
from graphicsHspbLUT import graphicsHspbForm
from graphicsLabLUT import graphicsLabForm
from splittedView import splittedWindow

from bLUeCore.demosaicing import demosaic
from bLUeGui.dialog import *
from viewer import playDiaporama, viewer

##################
#  Software Attributions
attributions = """
exiftool Copyright © 2013-2016, Phil Harvey
QRangeSlider Copyright (c) 2011-2012, Ryan Galloway
The Python Imaging Library (PIL) is
    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh
Pillow Copyright © 2010-2018 by Alex Clark and contributors
libraw Copyright (C) 2008-2018 
rawpy Copyright (c) 2014 Maik Riechert
seamlessClone and CLAHE are Opencv3 functions
grabCut is a parallel version of an Opencv3 function
This product includes DNG technology under license by Adobe Systems Incorporated
"""
#################

##############
#  Version number
VERSION = "v1.3.1"
##############

##############
# default adjustment form size
axeSize = 200
##############

##############
# multiprocessing pool
pool = None
##############

################################
# unbound generic event handlers.
# They should be bound  dynamically
# to specific widgets (e.g. QLabels)
# to provide interactions with images
################################

# init global QPainter for paint event (CAUTION non thread safe approach!)
qp = QPainter()
# stuff for Before/After tags
qp.font = QFont("Arial", 8)
qp.markPath = QPainterPath()
qp.markRect = QRect(0, 0, 50, 20)
qp.markPath.addRoundedRect(qp.markRect, 5, 5)


def paintEvent(widg, e):
    """
    Paint event handler.
    It displays the presentation layer of a vImage object in a Qlabel,
    with current offset and zooming coefficient.
    The widget must have a valid img attribute of type vImage.
    The handler should be used to override the paintEvent method of widg. This can be done
    by subclassing (not directly compatible with Qt Designer) or by assigning paintEvent
    to the method widg.paintEvent (cf. the function set_event_handler below).
    @param widg: widget
    @type widg: QLabel with a img attribute of type vImage
    @param e: paint event
    """
    mimg = getattr(widg, 'img', None)
    if mimg is None:
        return
    r = mimg.resize_coeff(widg)
    qp.begin(widg)
    # smooth painting
    qp.setRenderHint(QPainter.SmoothPixmapTransform)  # TODO may be useless
    # fill background
    qp.fillRect(QRect(0, 0, widg.width(), widg.height()), vImage.defaultBgColor)
    # draw the presentation layer.
    # As offsets can be float numbers, we use QRectF instead of QRect
    # r is relative to the full resolution image, so we use mimg width and height
    w, h = mimg.width() * r, mimg.height() * r
    rectF = QRectF(mimg.xOffset, mimg.yOffset, w, h)
    px = mimg.prLayer.qPixmap
    if px is not None:
        qp.drawPixmap(rectF, px, px.rect())
    else:
        currentImage = mimg.prLayer.getCurrentImage()
        qp.drawImage(rectF, currentImage, QImage.rect(currentImage))  # CAUTION : vImage.rect() is overwritten by attribute rect
    # draw the selection rectangle of the active layer, if any
    layer = mimg.getActiveLayer()
    rect = layer.rect
    if layer.visible and rect is not None:
        qp.setPen(QColor(0, 255, 0))
        qp.drawRect(rect.left()*r + mimg.xOffset, rect.top()*r + mimg.yOffset, rect.width()*r, rect.height()*r)
    # draw the cropping marks
    lm, rm, tm, bm = 0, 0, 0, 0
    if mimg.isCropped:
        c = QColor(128, 128, 128, 192)
        lm = window.cropTool.btnDict['left'].margin*r
        rm = window.cropTool.btnDict['right'].margin*r
        tm = window.cropTool.btnDict['top'].margin*r
        bm = window.cropTool.btnDict['bottom'].margin*r
        # left
        qp.fillRect(QRectF(mimg.xOffset, mimg.yOffset, lm, h), c)
        # top
        qp.fillRect(QRectF(mimg.xOffset+lm, mimg.yOffset, w - lm, tm), c)
        # right
        qp.fillRect(QRectF(mimg.xOffset+w-rm, mimg.yOffset+tm, rm, h-tm),  c)
        # bottom
        qp.fillRect(QRectF(mimg.xOffset+lm, mimg.yOffset+h-bm, w-lm-rm, bm), c)
    # draw rulers
    if mimg.isRuled:
        deltaX, deltaY = (w-lm-rm)//3, (h-tm-bm)//3
        qp.drawLine(lm+mimg.xOffset, deltaY+tm+mimg.yOffset, w-rm+mimg.xOffset, deltaY+tm+mimg.yOffset)
        qp.drawLine(lm+mimg.xOffset, 2*deltaY+tm+mimg.yOffset, w-rm+mimg.xOffset, 2*deltaY+tm+mimg.yOffset)
        qp.drawLine(deltaX+lm+mimg.xOffset, tm+mimg.yOffset, deltaX+lm+mimg.xOffset, h-bm+mimg.yOffset)
        qp.drawLine(2*deltaX+lm+mimg.xOffset, tm+mimg.yOffset, 2*deltaX+lm+mimg.xOffset, h-bm+mimg.yOffset)
    # tag before/after views
    name = widg.objectName()
    if name == "label_2" or name == "label_3":
        # draw filled rect
        qp.fillPath(qp.markPath, QBrush(Qt.gray))
        # draw text
        qp.setPen(Qt.white)
        qp.setFont(qp.font)
        qp.drawText(qp.markRect, Qt.AlignCenter | Qt.AlignVCenter, "Before" if name == "label_2" else "After")
    qp.end()


##############################################################
# global state variables used in mouseEvent.
pressed = False
clicked = True
# Recording of state and mouse coordinates (relative to widget)
State = {'ix': 0, 'iy': 0, 'ix_begin': 0, 'iy_begin': 0}
###############################################################

def mouseEvent(widget, event):  # TODO split into 3 handlers
    """
    Mouse event handler.
    The handler implements mouse actions on a vImage in a QLabel.
    It handles image positioning, zooming, and
    tool actions. It must be called by the mousePressed,
    mouseMoved and mouseReleased methods of the QLabel. This can be done by
    subclassing (not directly compatible with Qt Designer) or by
    dynamically assigning mouseEvent to the former three methods
    (cf. the function set_event_handler below).
    NOTE 1. Mouse hover generates mouse move events
    NOTE 2. Due to wheeelEvent, xOffset and yOffset are float numbers
    @param widget:
    @type widget: QLabel object with img attribute of type mImage
    @param event: mouse event
    @type event: QMouseEvent
    """
    global pressed, clicked
    if type(event) == QContextMenuEvent:
        return
    # get image and active layer
    img = widget.img
    layer = img.getActiveLayer()
    r = img.resize_coeff(widget)
    # x, y coordinates (relative to widget)
    x, y = event.x(), event.y()
    # keyboard modifiers
    modifiers = app.keyboardModifiers()
    eventType = event.type()
    ###################
    # mouse press event
    ###################
    if eventType == QEvent.MouseButtonPress:
        # Mouse hover generates mouse move events,
        # so, we set pressed to select only non hovering events
        pressed = True
        if event.button() == Qt.LeftButton:
            # no move yet
            clicked = True
        State['ix'], State['iy'] = x, y
        State['ix_begin'], State['iy_begin'] = x, y
        State['x_imagePrecPos'], State['y_imagePrecPos'] = (x - img.xOffset) // r, (y - img.yOffset) // r
        # add current mask to history
        if window.btnValues['drawFG'] or window.btnValues['drawBG']:
            if layer.maskIsEnabled:
                layer.historyListMask.addItem(layer.mask.copy())
        return  # no update needed
    ##################
    # mouse move event
    ##################
    elif eventType == QEvent.MouseMove:
        # hover event
        if not pressed:
            x_img, y_img = (x - img.xOffset) / r, (y - img.yOffset) / r
            # read input and current colors from active layer (coordinates are relative to the full-sized image)
            clr = img.getActivePixel(x_img, y_img, qcolor=True)
            clrC = img.getActivePixel(x_img, y_img, fromInputImg=False, qcolor=True)
            window.infoView.setText(clr, clrC)
            return
        clicked = False
        if img.isMouseSelectable:
            # don't draw on a non visible layer
            if window.btnValues['rectangle'] or window.btnValues['drawFG'] or window.btnValues['drawBG']:
                if not layer.visible:
                    dlgWarn('Select a visible layer for drawing or painting')
                    pressed = False
                    return
                elif not window.btnValues['rectangle'] and not layer.maskIsEnabled:
                    dlgWarn('Enable the mask before painting')
                    pressed = False
                    return
            # marquee tool
            if window.btnValues['rectangle']:
                # rectangle coordinates are relative to full image
                x_img = (min(State['ix_begin'], x) - img.xOffset) // r
                y_img = (min(State['iy_begin'], y) - img.yOffset) // r
                w = abs(State['ix_begin'] - x) // r
                h = abs(State['iy_begin'] - y) // r
                layer.rect = QRect(x_img, y_img, w, h)
            # drawing tools
            elif window.btnValues['drawFG'] or window.btnValues['drawBG']:
                if layer.maskIsEnabled:
                    if modifiers == Qt.NoModifier:
                        color = vImage.defaultColor_UnMasked if window.btnValues['drawFG'] else vImage.defaultColor_Masked
                    else:
                        color = vImage.defaultColor_UnMasked_Invalid if window.btnValues['drawFG'] else vImage.defaultColor_Invalid
                    qp.begin(layer.mask)
                    # get pen width
                    w_pen = window.verticalSlider1.value() // r
                    # mode source : result is source (=pen) pixel color and opacity
                    qp.setCompositionMode(qp.CompositionMode_Source)
                    tmp_x = (x - img.xOffset) // r
                    tmp_y = (y - img.yOffset) // r
                    qp.setPen(QPen(color, w_pen))
                    # draw line and final circle filling the region under the cursor
                    qp.drawLine(State['x_imagePrecPos'], State['y_imagePrecPos'], tmp_x, tmp_y)
                    qp.drawEllipse(tmp_x-w_pen//2, tmp_y-w_pen//2, w_pen, w_pen)
                    qp.end()
                    State['x_imagePrecPos'], State['y_imagePrecPos'] = tmp_x, tmp_y
                    ############################
                    # update upper stack
                    # should be layer.applyToStack() if any upper layer visible : too slow
                    layer.updatePixmap(maskOnly=True)  # TODO maskOnly not used: remove
                    #############################
                    img.prLayer.applyNone()
                    # img.prLayer.updatePixmap(maskOnly=True) # TODO called by applyNone 19/06/18
                    window.label.repaint()
            # dragBtn or arrow
            else:
                # drag image
                if modifiers == Qt.NoModifier:
                    img.xOffset += x - State['ix']
                    img.yOffset += y - State['iy']
                    if window.btnValues['Crop_Button']:
                        window.cropTool.drawCropTool(img)
                # drag active layer only
                elif modifiers == Qt.ControlModifier:
                    layer.xOffset += (x - State['ix'])
                    layer.yOffset += (y - State['iy'])
                    layer.updatePixmap()
                    img.prLayer.applyNone()
                # drag cloning virtual layer
                elif modifiers == Qt.ControlModifier | Qt.AltModifier:
                    if layer.isCloningLayer():
                        layer.xAltOffset += (x - State['ix'])
                        layer.yAltOffset += (y - State['iy'])
                        layer.autoclone = False
                        layer.maskIsEnabled = True
                        layer.maskIsSelected = False
                        layer.applyCloning(seamless=False)
        # not mouse selectable widget : probably before window alone !
        else:
            if modifiers == Qt.NoModifier:
                img.xOffset += (x - State['ix'])
                img.yOffset += (y - State['iy'])
            elif modifiers == Qt.ControlModifier:
                layer.xOffset += (x - State['ix'])
                layer.yOffset += (y - State['iy'])
                layer.updatePixmap()
        # update current coordinates
        State['ix'], State['iy'] = x, y
        if layer.isGeomLayer():
            layer.tool.moveRotatingTool()
    #####################
    # mouse release event
    # mouse click event
    ####################
    elif eventType == QEvent.MouseButtonRelease:
        pressed = False
        if event.button() == Qt.LeftButton:
            if layer.maskIsEnabled \
                    and layer.getUpperVisibleStackIndex() != -1\
                    and (window.btnValues['drawFG'] or window.btnValues['drawBG']):
                layer.applyToStack()
            if img.isMouseSelectable:
                # click event
                if clicked:
                    x_img, y_img = (x - img.xOffset) / r, (y - img.yOffset) / r
                    # read input and current colors from active layer (coordinates are relative to the full-sized image)
                    clr = img.getActivePixel(x_img, y_img, qcolor=True)
                    clrC = img.getActivePixel(x_img, y_img, fromInputImg=False, qcolor=True)
                    red, green, blue = clr.red(), clr.green(), clr.blue()
                    # read color from presentation layer
                    redP, greenP, blueP = img.getPrPixel(x_img, y_img)
                    # set color chooser value according to modifiers
                    if getattr(window, 'colorChooser', None) and window.colorChooser.isVisible():
                        if (modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier):
                            window.colorChooser.setCurrentColor(clr)
                        elif modifiers & Qt.ControlModifier:
                            window.colorChooser.setCurrentColor(clrC)
                        else:
                            window.colorChooser.setCurrentColor(QColor(redP, greenP, blueP))
                    else:
                        # emit colorPicked signal
                        layer.colorPicked.sig.emit(x_img, y_img, modifiers)
                        # select grid node for 3DLUT form
                        if layer.is3DLUTLayer():
                            layer.getGraphicsForm().selectGridNode(red, green, blue)
                        # rectangle selection
                        if window.btnValues['rectangle'] and (modifiers == Qt.ControlModifier):
                                layer.rect = None
                                layer.selectionChanged.sig.emit()
                        # for raw layer, set multipliers to get selected pixel as White Point
                        if layer.isRawLayer() and window.btnValues['colorPicker']:
                            # get demosaic buffer and sample raw pixels
                            bufRaw = layer.parentImage.demosaic
                            nb = QRect(x_img-2, y_img-2, 4, 4)
                            r = QImage.rect(layer.parentImage).intersected(nb)
                            if not r.isEmpty():
                                color = np.sum(bufRaw[r.top():r.bottom()+1, r.left():r.right()+1], axis=(0, 1))/(r.width()*r.height())
                            else:
                                color = bufRaw[y_img, x_img, :]  # TODO added 25/06/18 to avoid uninit. color validate
                            color = [color[i] - layer.parentImage.rawImage.black_level_per_channel[i] for i in range(3)]
                            form = layer.getGraphicsForm()
                            if form.sampleMultipliers:
                                row, col = 3*y_img//layer.height(), 3*x_img//layer.width()
                                if form.samples:
                                    form.setRawMultipliers(*form.samples[3*row + col], sampling=False)
                            else:
                                form.setRawMultipliers(1/color[0], 1/color[1], 1/color[2], sampling=True)
                else: # not clicked
                    if window.btnValues['rectangle']:
                        layer.selectionChanged.sig.emit()
    # updates
    widget.repaint()
    # sync split views
    linked = True
    if widget.objectName() == 'label_2':
        splittedWin.syncSplittedView(window.label_3, window.label_2, linked)
        window.label_3.repaint()
    elif widget.objectName() == 'label_3':
        splittedWin.syncSplittedView(window.label_2, window.label_3, linked)
        window.label_2.repaint()


def wheelEvent(widget, img, event):
    """
    Mouse wheel event handler : zooming
    for imImage objects.
    @param widget: widget displaying image
    @param img: imImage object to display
    @param event: mouse wheel event (type QWheelEvent)
    """
    pos = event.pos()
    # delta unit is 1/8 of degree
    # Most mice have a resolution of 15 degrees
    numSteps = event.delta() / 1200.0
    # keyboard modifiers
    modifiers = QApplication.keyboardModifiers()
    layer = img.getActiveLayer()
    if modifiers == Qt.NoModifier:
        img.Zoom_coeff *= (1.0 + numSteps)
        # max Zoom for previews
        if img.Zoom_coeff > MAX_ZOOM:
            img.Zoom_coeff /= (1.0 + numSteps)
            return
        # correct image offset to keep unchanged the image point
        # under the cursor : (pos - offset) / resize_coeff should be invariant
        img.xOffset = -pos.x() * numSteps + (1.0+numSteps)*img.xOffset
        img.yOffset = -pos.y() * numSteps + (1.0+numSteps)*img.yOffset
        if window.btnValues['Crop_Button']:
            window.cropTool.drawCropTool(img)
        if layer.isGeomLayer():
            # layer.view.widget().tool.moveRotatingTool()
            layer.tool.moveRotatingTool()
    elif modifiers == Qt.ControlModifier:
        layer.Zoom_coeff *= (1.0 + numSteps)
        layer.updatePixmap()
    elif modifiers == Qt.ControlModifier | Qt.AltModifier:
        if layer.isCloningLayer():
            layer.AltZoom_coeff *= (1.0 + numSteps)
            layer.autoclone = False
            layer.applyCloning(seamless=False)
            # layer.updatePixmap()
    widget.repaint()
    # sync split views
    linked = True
    if widget.objectName() == 'label_2':
        splittedWin.syncSplittedView(window.label_3, window.label_2, linked)
        window.label_3.repaint()
    elif widget.objectName() == 'label_3':
        splittedWin.syncSplittedView(window.label_2, window.label_3, linked)
        window.label_2.repaint()


def enterEvent(widget, img, event):
    """
    Mouse enter event handler
    @param widget:
    @param img:
    @param event:
    """
    if QApplication.overrideCursor():
        # don't stack multiple cursors
        return
    if window.btnValues['drawFG'] or window.btnValues['drawBG']:
        if not QApplication.overrideCursor():
            w = window.verticalSlider1.value()
            if w > 5:
                QApplication.setOverrideCursor(window.cursor_Circle_Pixmap.scaled(w*2.0, w*2.0))
            else:
                QApplication.setOverrideCursor(Qt.CrossCursor)
    elif window.btnValues['drag']:
        QApplication.setOverrideCursor(Qt.OpenHandCursor)
    elif window.btnValues['colorPicker']:
        layer = window.label.img.getActiveLayer()
        if layer.isAdjustLayer():
            if layer.view.isVisible():
                if not QApplication.overrideCursor():
                    QApplication.setOverrideCursor(window.cursor_EyeDropper)


def leaveEvent(widget, img, event):
    QApplication.restoreOverrideCursor()


def dragEnterEvent(widget, img, event):
    """
    Accept drop if mimeData contains text (e.g. file name)
    (convenient for main window only)
    @param widget:
    @type widget:
    @param img:
    @type img:
    @param event:
    @type event:
    """
    if event.mimeData().hasFormat("text/plain"):
        event.acceptProposedAction()


def dropEvent(widget, img, event):
    """
    get file name from event.mimeData and open it.
    @param widget:
    @type widget:
    @param img:
    @type img:
    @param event:
    @type event:

    """
    mimeData = event.mimeData()
    openFile(mimeData.text())


def set_event_handlers(widg, enterAndLeave=True):
    """
    Pythonic redefinition of event handlers, without
    subclassing or overriding. However, the PySide dynamic
    ui loader requires that we set the corresponding classes as customWidget
    (cf. file QtGui1.py and pyside_dynamicLoader.py).
    if enterAndLeave is False enter and leave event handlers are not set,
    otherwise all mouse event handlers are set.
    @param widg:
    @type widg : QObject
    @param enterAndLeave:
    @type enterAndLeave: boolean
    """
    widg.paintEvent = MethodType(lambda instance, e, wdg=widg: paintEvent(wdg, e), widg.__class__)
    widg.mousePressEvent = MethodType(lambda instance, e, wdg=widg: mouseEvent(wdg, e), widg.__class__)
    widg.mouseMoveEvent = MethodType(lambda instance, e, wdg=widg: mouseEvent(wdg, e), widg.__class__)
    widg.mouseReleaseEvent = MethodType(lambda instance, e, wdg=widg: mouseEvent(wdg, e), widg.__class__)
    widg.wheelEvent = MethodType(lambda instance, e, wdg=widg: wheelEvent(wdg, wdg.img, e), widg.__class__)
    if enterAndLeave:
        widg.enterEvent = MethodType(lambda instance, e, wdg=widg: enterEvent(wdg, wdg.img, e), widg.__class__)
        widg.leaveEvent = MethodType(lambda instance, e, wdg=widg: leaveEvent(wdg, wdg.img, e), widg.__class__)


def widgetChange(button):
    """
    called by all main form button and slider slots (cf. QtGui1.py onWidgetChange)

    @param button:
    @type button: QWidget
    """
    # wdgName = button.objectName()
    if button is window.fitButton:  # wdgName == "fitButton" :
        window.label.img.fit_window(window.label)
        # update crop button positions
        window.cropTool.drawCropTool(window.label.img)
        # window.label.repaint()
    elif button is window.cropButton:  # wdgName == "cropButton":
        if button.isChecked():
            window.cropTool.drawCropTool(window.label.img)
            for b in window.cropTool.btnDict.values():
                b.show()
        else:
            for b in window.cropTool.btnDict.values():
                b.hide()
        window.label.img.isCropped = button.isChecked()
        # window.label.repaint()
    elif button is window.rulerButton:  # wdgName == "rulerButton":
        window.label.img.isRuled = button.isChecked()
    elif button is window.eyeDropper:  # wdgName == 'eyeDropper':
        if button.isChecked():  # window.btnValues['colorPicker']:
            openColorChooser()
            dlg = window.colorChooser
            try:
                dlg.closeSignal.sig.disconnect()
            except RuntimeError:
                pass
            dlg.closeSignal.sig.connect(lambda: button.setChecked(False))
        else:
            if getattr(window, 'colorChooser', None) is not None:
                window.colorChooser.hide()
    window.label.repaint()


def contextMenu(pos, widget):
    """
    Context menu for image QLabel
    @param pos:
    @param widget:
    @return:
    """
    pass


def loadImageFromFile(f, createsidecar=True):
    """
    load an imImage (image and metadata) from file. Returns the loaded imImage :
    For a raw file, it is the image postprocessed with default parameters.
    metadata is a list of dicts with len(metadata) >=1.
    metadata[0] contains at least 'SourceFile' : path.
    profile is a string containing the profile binary data,
    currently, we do not use these data : standard profiles
    are loaded from disk, non standard profiles are ignored.
    @param f: path to file
    @type f: str
    @param createsidecar:
    @type createsidecar: boolean
    @return: image
    @rtype: imImage
    """
    ###########
    # read metadata
    ##########
    try:
        # read metadata from sidecar (.mie) if it exists, otherwise from image file.
        # The sidecar is created if it does not exist and createsidecar is True.
        with exiftool.ExifTool() as e:
            profile, metadata = e.get_metadata(f, tags=("colorspace", "profileDescription", "orientation", "model", "rating"), createsidecar=createsidecar)
            imageInfo = e.get_formatted_metadata(f)
    except ValueError:
        # Default metadata and profile
        metadata = [{'SourceFile': f}]
        profile = ''
        imageInfo = 'No data found'
    # color space : 1=sRGB 65535=uncalibrated
    tmp = [value for key, value in metadata.items() if 'colorspace' in key.lower()]
    colorSpace = tmp[0] if tmp else -1
    # try again to find a valid color space tag and/or an imbedded profile.
    # If everything fails, assign sRGB.
    if colorSpace == -1 or colorSpace == 65535:
        tmp = [value for key, value in metadata.items() if 'profiledescription' in key.lower()]
        desc_colorSpace = tmp[0] if tmp else ''
        #desc_colorSpace = metadata.get("ICC_Profile:ProfileDescription", '')
        if isinstance(desc_colorSpace, str):
            if not ('sRGB' in desc_colorSpace) or hasattr(window, 'modeDiaporama'):
                # setOverrideCursor does not work correctly for a MessageBox :
                # may be a Qt Bug, cf. https://bugreports.qt.io/browse/QTBUG-42117
                QApplication.changeOverrideCursor(QCursor(Qt.ArrowCursor))
                QApplication.processEvents()
                if len(desc_colorSpace) > 0:
                    # convert profile to ImageCmsProfile object
                    profile = ImageCmsProfile(BytesIO(profile))
                else:
                    dlgInfo("Color profile is missing\nAssigning sRGB")  # modified 08/10/18 validate
                    # assign sRGB profile
                    colorSpace = 1
    # update the color management object with the image profile.
    icc.configure(colorSpace=colorSpace, workingProfile=profile)
    # orientation
    tmp = [value for key, value in metadata.items() if 'orientation' in key.lower()]
    orientation = tmp[0] if tmp else 0  # metadata.get("EXIF:Orientation", 0)
    transformation = exiftool.decodeExifOrientation(orientation)
    # rating
    tmp = [value for key, value in metadata.items() if 'rating' in key.lower()]
    rating = tmp[0] if tmp else 0  # metadata.get("XMP:Rating", 5)
    ############
    # load image
    ############
    name = path.basename(f)
    ext = name[-4:]
    if ext in list(IMAGE_FILE_EXTENSIONS):
        img = imImage(filename=f, colorSpace=colorSpace, orientation=transformation, rawMetadata=metadata, profile=profile, name=name, rating=rating)
    elif ext in list(RAW_FILE_EXTENSIONS):
        # load raw image file in a RawPy instance
        # rawpy.imread keeps f open. Calling raw.close() deletes the raw object.
        # As a workaround we use a file buffer .
        # Relevant RawPy attributes are black_level_per_channel, camera_white_balance, color_desc, color_matrix,
        # daylight_whitebalance, num_colors, raw_colors_visible, raw_image, raw_image_visible, raw_pattern,
        # raw_type, rgb_xyz_matrix, sizes, tone_curve.
        # raw_image and raw_image_visble are sensor data
        rawpyInst = rawpy.RawPy()
        with open(f, "rb") as bufio:
            rawpyInst.open_buffer(bufio)
        ######################################################################################
        # unpack always applies the current tone curve (cf. https://www.libraw.org/node/2003)
        # read from file for Nikon, Sony and some other cameras.
        # Another curve (array, shape=65536) can be loaded here before unpacking.
        # NO EFFECT with files where the curve is calculated on unpack() phase (e.g.Nikon lossy NEF files).
        #####################################################################################
        rawpyInst.unpack()
        # postprocess raw image, applying default settings (cf. vImage.applyRawPostProcessing)
        rawBuf = rawpyInst.postprocess(use_camera_wb=True)
        # build Qimage
        rawBuf = np.dstack((rawBuf[:, :, ::-1], np.zeros(rawBuf.shape[:2], dtype=np.uint8)+255))
        img = imImage(cv2Img=rawBuf, colorSpace=colorSpace, orientation=transformation,
                      rawMetadata=metadata, profile=profile, name=name, rating=rating)
        img.filename = f
        # keep references to rawPy instance. rawpyInst.raw_image is the (linearized) sensor image
        img.rawImage = rawpyInst
        #########################################################
        # Reconstructing the demosaic Bayer bitmap :
        # we need it to calculate the multipliers corresponding
        # to a user white point, and we cannot access the
        # native rawpy demosaic buffer from the RawPy instance !!!!
        #########################################################
        # get 16 bits Bayer bitmap
        img.demosaic = demosaic(rawpyInst.raw_image_visible, rawpyInst.raw_colors_visible, rawpyInst.black_level_per_channel)
        # correct orientation
        if orientation == 6:  # 90°
            img.demosaic = np.swapaxes(img.demosaic, 0, 1)
        elif orientation == 8:  # 270°
            img.demosaic = np.swapaxes(img.demosaic, 0, 1)
            img.demosaic = img.demosaic[:, ::-1, :]
    else:
        raise ValueError("Cannot read file %s" % f)
    if img.isNull():
        raise ValueError("Cannot read file %s" % f)
    if img.format() in [QImage.Format_Invalid, QImage.Format_Mono, QImage.Format_MonoLSB, QImage.Format_Indexed8]:
        raise ValueError("Cannot edit indexed formats\nConvert image to a non indexed mode first")
    img.imageInfo = imageInfo
    window.settings.setValue('paths/dlgdir', QFileInfo(f).absoluteDir().path())
    img.initThumb()
    return img


def addBasicAdjustmentLayers(img):
    if img.rawImage is None:
        # menuLayer('actionColor_Temperature')
        # menuLayer('actionExposure_Correction')
        menuLayer('actionContrast_Correction')
    # select active layer : top row
    window.tableView.select(0, 1)


def addRawAdjustmentLayer():
    """
    Add a development layer to the layer stack
    """
    rlayer = window.label.img.addAdjustmentLayer(layerType=QRawLayer, name='Develop', role='RAW')
    grWindow = rawForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=rlayer, parent=window)
    # wrapper for the right apply method
    pool = getPool()
    rlayer.execute = lambda l=rlayer, pool=pool: l.tLayer.applyRawPostProcessing(pool=pool)
    # record action name for scripting
    rlayer.actionName = ''
    # dock the form
    dock = QDockWidget(window)
    dock.setWidget(grWindow)
    dock.setWindowFlags(grWindow.windowFlags())
    dock.setWindowTitle(grWindow.windowTitle())
    dock.move(900, 40)
    dock.setStyleSheet("QGraphicsView{margin: 10px; border-style: solid; border-width: 1px; border-radius: 1px;}")
    rlayer.view = dock
    # add to docking area
    window.addDockWidget(Qt.RightDockWidgetArea, dock)
    # update layer stack view
    window.tableView.setLayers(window.label.img)


def openFile(f):
    """
    Top level function for file opening, used by File Menu actions
    @param f: file name
    @type f: str
    """
    # close opened document, if any
    if not closeFile():
        return
    # load file
    try:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        img = loadImageFromFile(f)
        # init layers
        if img is not None:
            setDocumentImage(img)
            # switch to preview mode and process stack
            window.tableView.previewOptionBox.setChecked(True)
            window.tableView.previewOptionBox.stateChanged.emit(Qt.Checked)
            # add development layer for raw image, and develop
            if img.rawImage is not None:
                addRawAdjustmentLayer()
            # add default adjustment layers
            addBasicAdjustmentLayers(img)
            # updates
            img.layersStack[0].applyToStack()
            img.onImageChanged()
            updateStatus()
            # update list of recent files
            recentFiles = window.settings.value('paths/recent', [])
            # settings.values returns a str or a list of str,
            # depending on the count of items. May be a Pyside2 bug
            # in QVariant conversion.
            if type(recentFiles) is str:
                recentFiles = [recentFiles]
            recentFiles = list(filter(lambda a: a != f, recentFiles))
            recentFiles.insert(0, f)
            if len(recentFiles) > 10:
                recentFiles.pop()
            window.settings.setValue('paths/recent', recentFiles)
    except (ValueError, IOError, rawpy.LibRawFatalError) as e:
        QApplication.restoreOverrideCursor()
        QApplication.processEvents()
        dlgWarn(str(e))
    finally:
        QApplication.restoreOverrideCursor()
        QApplication.processEvents()


def closeFile():
    """
    Top Level function for file closing.
    Close the opened document and reset windows.
    return True if succeed, False otherwise.
    @return:
    @rtype: boolean
    """
    if not canClose():
        return False
    window.tableView.clear(delete=True)
    window.histView.targetImage = None
    window.label.img = defaultImImage
    window.label_2.img = defaultImImage
    window.label_3.img = defaultImImage
    gc.collect()
    window.label.update()
    window.label_2.update()
    window.label_3.update()
    return True


def setDocumentImage(img):
    """
    Inits GUI and displays the current document
    @param img: image
    @type img: imImage
    """
    window.cropButton.setChecked(False)
    window.rulerButton.setChecked(False)
    window.label.img = img
    # init histogram
    window.histView.targetImage = window.label.img
    # image changed event handler

    def f():
        # refresh windows (use repaint for faster update)
        window.label.repaint()
        window.label_3.repaint()
        # recompute and display histogram for the right image
        if window.histView.listWidget1.items['Original Image'].checkState() is Qt.Checked:
            histImg = vImage(QImg=window.label.img.getCurrentImage())  # must be vImage : histogram method needed
        else:
            histImg = window.label.img.layersStack[-1].getCurrentMaskedImage()
        if window.histView.listWidget2.items['Color Chans'].checkState() is Qt.Checked:
            window.histView.mode = 'RGB'
            window.histView.chanColors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(10, 10, 255)]
        else:
            window.histView.mode = 'Luminosity'
            window.histView.chanColors = [Qt.gray]
        histView = histImg.histogram(QSize(window.histView.width(), window.histView.height()),
                                     chans=list(range(3)), bgColor=Qt.black,
                                     chanColors=window.histView.chanColors, mode=window.histView.mode, addMode='')
        window.histView.Label_Hist.setPixmap(QPixmap.fromImage(histView))
        window.histView.Label_Hist.repaint()
    ###################################
    # init displayed images
    # label.img : working image
    # label_2.img  : before image (copy of the initial state of working image)
    # label_3.img : reference to working image
    ###################################
    window.label.img.onImageChanged = f
    # before image : the stack is not copied
    window.label_2.img = imImage(QImg=img, meta=img.meta)
    # after image : ref to the opened document
    window.label_3.img = img
    # no mouse drawing or painting
    window.label_2.img.isMouseSelectable = False
    # init layer view
    window.tableView.setLayers(window.label.img)
    window.label.update()
    window.label_2.update()
    window.label_3.update()
    # back links used by graphicsForm3DLUT.onReset
    window.label.img.window = window.label  # TODO 4/11/18 probably graphicsForm3DLUT.onReset should be modified to remove this dependency
    window.label_2.img.window = window.label_2
    window.label.img.setModified(True)


def updateMenuOpenRecent():
    """
    Update the list of recent files displayed
    in the QMenu menuOpen_recent, and init
    the corresponding actions
    """
    window.menuOpen_recent.clear()
    recentFiles = window.settings.value('paths/recent', [])
    # settings.values returns a str or a list of str,
    # depending on the count of items. May be a Pyside2 bug
    # in QVariant conversion.
    if type(recentFiles) is str:
        recentFiles = [recentFiles]
    for filename in recentFiles:
        window.menuOpen_recent.addAction(filename, lambda x=filename: openFile(x))


def updateEnabledActions():
    """
    Menu aboutToShow handler
    @return:
    @rtype:
    """
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    window.actionSave.setEnabled(window.label.img.isModified)
    window.actionSave_Hald_Cube.setEnabled(window.label.img.isHald)


def menuFile(name):
    """
    Menu handler
    @param name: action name
    @type name: str
    """
    # load image from file
    if name in ['actionOpen']:
        # get file name from dialog
        filename = openDlg(window)
        # open file
        if filename is not None:
            openFile(filename)
    # saving dialog
    elif name == 'actionSave':
        if window.label.img.useThumb:
            dlgWarn("Uncheck Preview mode before saving")
        else:
            try:
                filename = saveDlg(window.label.img, window)
                dlgInfo("%s written" % filename)
            except (ValueError, IOError) as e:
                dlgWarn(str(e))
    # closing dialog : close opened document
    elif name == 'actionClose':
        closeFile()
        global pool
        if pool is not None:
            pool.close()
            pool.join()
            pool = None
    updateStatus()

def menuView(name):
    """
    Menu handler
    @param name: action name
    @type name: str
    """
    ##################
    # before/after mode
    ##################
    if name == 'actionShow_hide_right_window_3':
        if window.splitter.isHidden():
            splittedWin.setSplittedView()
            window.viewState = 'Before/After'
        else:
            window.splitter.hide()
            window.label.show()
            window.splittedView = False
            window.viewState = 'After'
            if window.btnValues['Crop_Button']:
                window.cropTool.drawCropTool(window.label.img)
    ###########
    # slide show
    ###########
    elif name == 'actionDiaporama':
        if getattr(window, 'diaporamaGenerator', None) is not None:
            reply = QMessageBox()
            reply.setWindowTitle('Question')
            reply.setIcon(QMessageBox.Information)
            reply.setText("A diaporama was suspended. Resume ?")
            reply.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
            reply.setDefaultButton(QMessageBox.Yes)
            ret = reply.exec_()
            if ret == QMessageBox.No:
                window.diaporamaGenerator = None
        else:
            window.diaporamaGenerator = None
        if window.diaporamaGenerator is None:
            # start from parent dir of the last used directory
            lastDir = path.join(str(window.settings.value('paths/dlgdir', '.')), path.pardir)
            dlg = QFileDialog(window, "Select a folder to start the diaporama", lastDir)
            dlg.setNameFilters(IMAGE_FILE_NAME_FILTER)
            dlg.setFileMode(QFileDialog.Directory)
            diaporamaList = []
            # directory dialog
            if dlg.exec_():
                newDir = dlg.selectedFiles()[0]  # dlg.directory().absolutePath()
                window.settings.setValue('paths/dlgdir', newDir)
                for dirpath, dirnames, filenames in walk(newDir):
                    for filename in [f for f in filenames
                                     if f.endswith(IMAGE_FILE_EXTENSIONS)]:
                        diaporamaList.append(path.join(dirpath, filename))
            window.diaporamaGenerator = cycle(diaporamaList)
        playDiaporama(window.diaporamaGenerator, parent=window)
    #############
    # library viewer
    #############
    elif name == 'actionViewer':
        # start from parent dir of the last used directory
        lastDir = path.join(str(window.settings.value('paths/dlgdir', '.')), path.pardir)
        dlg = QFileDialog(window, "select", lastDir)
        dlg.setNameFilters(IMAGE_FILE_NAME_FILTER)
        dlg.setFileMode(QFileDialog.Directory)
        # open dialog
        if dlg.exec_():
            newDir = dlg.selectedFiles()[0]  # dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
            viewerInstance = viewer.getViewerInstance(mainWin=window)
            viewerInstance.playViewer(newDir)
    ###############
    # Color Chooser
    ###############
    elif name == 'actionColor_Chooser':
        openColorChooser()
    updateStatus()


def openColorChooser():
    if getattr(window, 'colorChooser', None) is None:
        window.colorChooser = QbLUeColorDialog(parent=window)
        window.colorChooser.setWhatsThis(
            """
            <b>ColorChooser</b><br>
            To <b>display the color of a pixel</b> click it in the image window:<br>
            &nbsp;&nbsp;<b>Click</b> : image pixel<br>
            &nbsp;&nbsp;<b>Ctrl + Click</b> : active layer pixel<br>
            &nbsp;&nbsp;<b>Ctrl+Shift + Click</b> : input pixel of active layer (adjustment layer only) <br>
            """
        )  # end of whatsthis
    window.colorChooser.show()


def menuImage(name):
    """
    Menu handler
    @param name: action name
    @type name: str
    """
    img = window.label.img
    # display image info
    if name == 'actionImage_info':
        # Format
        s = "Format : %s\n(cf. QImage formats in the doc for more info)" % QImageFormats.get(img.format(), 'unknown')
        # dimensions
        s = s + "\n\ndim : %d x %d" % (img.width(), img.height())
        # profile info
        if img.meta.profile is not None:  # len(img.meta.profile) > 0:
            s = s + "\n\nEmbedded profile found"  # length %d" % len(img.meta.profile)
        workingProfileInfo = icc.workingProfileInfo
        s = s + "\n\nWorking Profile : %s" % workingProfileInfo
        # rating
        s = s + "\n\nRating %s" % ''.join(['*']*img.meta.rating)
        # formatted meta data
        s = s + "\n\n" + img.imageInfo
        # display
        _, label = handleTextWindow(parent=window, title='Image info', wSize=QSize(700, 700))
        label.setWordWrap(True)
        label.setFont(QFont("Courier New"))
        label.setStyleSheet("background-color: white")
        label.setText(s)
    elif name == 'actionColor_manage':
        icc.COLOR_MANAGE = window.actionColor_manage.isChecked()
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            img.updatePixmap()
            window.label_2.img.updatePixmap()
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
        window.label.repaint()
        window.label_2.repaint()
        updateStatus()
    elif name == 'actionWorking_profile':
        w, label = handleTextWindow(parent=window, title='profile info')
        s = 'Working Profile : '
        if icc.workingProfile is not None:
            s = s + icc.workingProfileInfo
        s = s + '-------------\n' + 'Monitor Profile : '
        if icc.monitorProfile is not None:
            s = s + icc.monitorProfileInfo + '-------------\n'
        s = s + 'Note :\nThe working profile is the color profile assigned to the image.'
        s = s + 'The monitor profile should correspond to your monitor.'
        s = s + '\nBoth profiles are used in conjunction to display exact colors. '
        s = s + 'If one of them is missing, bLUe cannot color manage the image.'
        s = s + '\nIf the monitor profile listed above is not the right profile for your monitor, please check the system settings for color management'
        label.setWordWrap(True)
        label.setText(s)
    # rotations
    elif name in ['action90_CW', 'action90_CCW', 'action180']:
        try:
            angle = 90 if name == 'action90_CW' else -90 if name == 'action90_CCW' else 180
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            # get new imImage
            tImg = img.bTransformed(QTransform().rotate(angle))
            setDocumentImage(tImg)
            # attempting to free old imImage
            del img.prLayer
            del img
            gc.collect()
            tImg.layersStack[0].applyToStack()
            tImg.onImageChanged()
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
    # rating
    elif name in ['action0', 'action1', 'action2', 'action3', 'action4', 'action5']:
        img.meta.rating = int(name[-1:])
        updateStatus()
        with exiftool.ExifTool() as e:
            e.writeXMPTag(img.meta.filename, 'XMP:rating', img.meta.rating)


def getPool():
    global pool
    try:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        # init pool only once
        if USE_POOL and (pool is None):
            pool = multiprocessing.Pool(POOL_SIZE)
    finally:
        QApplication.restoreOverrideCursor()
        QApplication.processEvents()
    return pool


def menuLayer(name):
    """
    Menu Layer handler
    @param name: action name
    @type name: str
    """

    # curves
    if name in ['actionCurves_RGB', 'actionCurves_HSpB', 'actionCurves_Lab']:
        if name == 'actionCurves_RGB':
            layerName = 'RGB'
            form = graphicsForm
        elif name == 'actionCurves_HSpB':  # displayed as HSV in the layer menu !!
            layerName = 'HSV'
            form = graphicsHspbForm
        elif name == 'actionCurves_Lab':
            layerName = 'Lab'
            form = graphicsLabForm
        # add new layer on top of active layer
        layer = window.label.img.addAdjustmentLayer(name=layerName)
        grWindow = form.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer, parent=window)
        # wrapper for the right applyXXX method
        if name == 'actionCurves_RGB':
            layer.execute = lambda l=layer, pool=None: l.tLayer.apply1DLUT(grWindow.scene().cubicItem.getStackedLUTXY())
        elif name == 'actionCurves_HSpB':  # displayed as HSV in the layer menu !!
            layer.execute = lambda l=layer, pool=None: l.tLayer.applyHSV1DLUT(grWindow.scene().cubicItem.getStackedLUTXY(), pool=pool)
        elif name == 'actionCurves_Lab':
            layer.execute = lambda l=layer, pool=None: l.tLayer.applyLab1DLUT(grWindow.scene().cubicItem.getStackedLUTXY())
    # 3D LUT
    elif name in ['action3D_LUT', 'action3D_LUT_HSB']:
        # color model
        ccm = cmHSP if name == 'action3D_LUT' else cmHSB
        layerName = '2.5D LUT HSpB' if name == 'action3D_LUT' else '2.5D LUT HSV'
        layer = window.label.img.addAdjustmentLayer(name=layerName, role='3DLUT')
        grWindow = graphicsForm3DLUT.getNewWindow(ccm, axeSize=300, targetImage=window.label.img,
                                                  LUTSize=LUTSIZE, layer=layer, parent=window, mainForm=window)  # mainForm mandatory here
        # init pool only once
        pool = getPool()
        sc = grWindow.scene()
        layer.execute = lambda l=layer, pool=pool: l.tLayer.apply3DLUT(sc.lut.LUT3DArray, sc.lut.step,
                                                                       options=sc.options, pool=pool)
    # cloning
    elif name == 'actionNew_Cloning_Layer':
        lname = 'Cloning'
        layer = window.label.img.addAdjustmentLayer(name=lname, role='CLONING')
        grWindow = patchForm.getNewWindow(targetImage=window.label.img, layer=layer)
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyCloning(seamless=l.autoclone)
    # segmentation
    elif name == 'actionNew_segmentation_layer':
        lname = 'Segmentation'
        layer = window.label.img.addSegmentationLayer(name=lname)
        grWindow = segmentForm.getNewWindow(targetImage=window.label.img, layer=layer)
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyGrabcut(nbIter=grWindow.nbIter)
        # mask was modified
        # l.updatePixmap()
    # load an image from file
    elif name == 'actionLoad_Image_from_File':  # 'actionNew_Image_Layer':
        filename = openDlg(window, ask=False)
        if filename is None:
            return
        # load image from file, alpha channel is mandatory for applyTransform()
        imgNew = QImage(filename).convertToFormat(QImage.Format_ARGB32)  # QImage(filename, QImage.Format_ARGB32) does not work !
        if imgNew.isNull():
            dlgWarn("Cannot load %s: " % filename)
            return
        lname = path.basename(filename)
        layer = window.label.img.addAdjustmentLayer(name=lname, sourceImg=imgNew, role='GEOMETRY')
        grWindow = imageForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer, parent=window)
        # add transformation tool to parent widget
        tool = rotatingTool(parent=window.label)  # , layer=l, form=grWindow)
        layer.addTool(tool)
        tool.showTool()
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyImage(grWindow.options)
        layer.actioname = name
    # empty new image
    elif name == 'actionNew_Layer':
        processedImg = window.label.img
        w, h = processedImg.width(), processedImg.height()
        imgNew = QImage(w, h, QImage.Format_ARGB32)
        imgNew.fill(Qt.black)
        lname = 'Image'
        layer = window.label.img.addAdjustmentLayer(name=lname, sourceImg=imgNew, role='GEOMETRY')
        grWindow = imageForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer, parent=window)
        # add transformation tool to parent widget
        tool = rotatingTool(parent=window.label)  # , layer=l, form=grWindow)
        layer.addTool(tool)
        tool.showTool()
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyImage(grWindow.options)
        layer.actioname = name
    # Temperature
    elif name == 'actionColor_Temperature':
        lname = 'Color Temperature'
        layer = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = temperatureForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer, parent=window)
        # wrapper for the right apply method
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyTemperature()
    elif name == 'actionContrast_Correction':
        layer = window.label.img.addAdjustmentLayer(name=CoBrSatForm.layerTitle, role='CONTRAST')
        grWindow = CoBrSatForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer, parent=window)
        # clipLimit change event handler

        def h(lay, clipLimit):
            lay.clipLimit = clipLimit
            lay.applyToStack()
            window.label.img.onImageChanged()
        grWindow.onUpdateContrast = h
        # wrapper for the right apply method
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyContrast()
    elif name == 'actionExposure_Correction':
        lname = 'Exposure'
        layer = window.label.img.addAdjustmentLayer(name=lname)
        layer.clipLimit = ExpForm.defaultExpCorrection
        grWindow = ExpForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer, parent=window)
        """
        # clipLimit change event handler
        def h(lay, clipLimit):
            lay.clipLimit = clipLimit
            lay.applyToStack()
            window.label.img.onImageChanged()
        grWindow.onUpdateExposure = h
        """
        layer.execute = lambda l=layer,  pool=None: l.tLayer.applyExposure(grWindow.options)
    elif name == 'actionGeom_Transformation':
        lname = 'Transformation'
        layer = window.label.img.addAdjustmentLayer(name=lname, role='GEOMETRY')
        grWindow = transForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer, parent=window)
        # add transformation tool to parent widget
        tool = rotatingTool(parent=window.label)
        layer.addTool(tool)
        tool.showTool()
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyTransForm(grWindow.options)
    elif name == 'actionFilter':
        lname = 'Filter'
        layer = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = filterForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer)
        # wrapper for the right apply method
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyFilter2D()
    elif name == 'actionGradual_Filter':
        lname = 'Gradual Filter'
        layer = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = blendFilterForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img,
                                                layer=layer, parent=window)
        # wrapper for the right apply method
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyBlendFilter()
    elif name == 'actionNoise_Reduction':
        lname = 'Noise Reduction'
        layer = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = noiseForm.getNewWindow(axeSize=axeSize, layer=layer, parent=window)
        # wrapper for the right apply method
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyNoiseReduction()
    # invert image
    elif name == 'actionInvert':
        lname = 'Invert'
        layer = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = invertForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img,
                                           layer=layer, parent=window)
        layer.execute = lambda l=layer: l.tLayer.applyInvert()
        layer.applyToStack()
    elif name == 'actionChannel_Mixer':
        lname = 'Channel Mixer'
        layer = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = mixerForm.getNewWindow(axeSize=260, targetImage=window.label.img,
                                           layer=layer, parent=window)
        layer.execute = lambda l=layer: l.tLayer.applyMixer(grWindow.options)
    # load 3D LUT from .cube file
    elif name == 'actionLoad_3D_LUT':
        lastDir = window.settings.value('paths/dlg3DLUTdir', '.')
        dlg = QFileDialog(window, "select", lastDir)
        dlg.setNameFilter('*.cube')
        dlg.setDefaultSuffix('cube')
        if dlg.exec_():
            newDir = dlg.directory().absolutePath()
            window.settings.setValue('paths/dlg3DLUTdir', newDir)
            filenames = dlg.selectedFiles()
            name = filenames[0]
            try:
                lut = LUT3D.readFromTextFile(name)
            except (ValueError, IOError) as e:
                dlgWarn('Unable to load 3D LUT : ', info=str(e))
                return
            lname = path.basename(name)
            layer = window.label.img.addAdjustmentLayer(name=lname)
            pool = getPool()
            layer.execute = lambda l=layer, pool=pool: l.tLayer.apply3DLUT(lut.LUT3DArray,
                                                                           lut.step,
                                                                           {'use selection': False},
                                                                           pool=pool)
            window.tableView.setLayers(window.label.img)
            layer.applyToStack()
            # The resulting image is modified,
            # so we update the presentation layer before returning
            layer.parentImage.prLayer.update()
            layer.parentImage.onImageChanged()
        return
    elif name == 'actionSave_Layer_Stack_as_LUT_Cube':
        img = window.label.img
        # get current size
        s = (img.getCurrentImage()).size()
        # build input hald image from identity 3D LUT; channels are in BGR order
        buf = LUT3DIdentity.toHaldArray(s.width(), s.height()).haldBuffer
        # add hald to stack, on top of  background
        layer = img.addLayer(None, name='Hald', index=1)
        try:
            # set hald flag
            img.isHald = True
            QImageBuffer(layer.getCurrentImage())[:, :, :3] = buf
            # process hald
            layer.applyToStack()
            processedImg = img.prLayer.inputImg()
            buf = QImageBuffer(processedImg)
            # init haldArray from image
            hArray = haldArray(buf, LUT3DIdentity.size)
            # convert the hald array to a LUT3D object (BGR order)
            LUT = LUT3D.HaldBuffer2LUT3D(hArray)
            # write LUT to file
            lastDir = str(window.settings.value('paths/dlg3DLUTdir', '.'))
            dlg = QFileDialog(window, "select", lastDir)
            dlg.setNameFilter('*.cube')
            dlg.setDefaultSuffix('cube')
            if dlg.exec_():
                newDir = dlg.directory().absolutePath()
                window.settings.setValue('paths/dlg3DLUTdir', newDir)
                filenames = dlg.selectedFiles()
                newDir = dlg.directory().absolutePath()
                window.settings.setValue('paths/dlg3DLUTdir', newDir)
                filename = filenames[0]
                if isfile(filename):
                    reply = QMessageBox()
                    reply.setWindowTitle('Warning')
                    reply.setIcon(QMessageBox.Warning)
                    reply.setText("File %s already exists\n" % filename)
                    reply.setStandardButtons(QMessageBox.Cancel)
                    accButton = QPushButton("OverWrite")
                    reply.addButton(accButton, QMessageBox.AcceptRole)
                    reply.exec_()
                    retButton = reply.clickedButton()
                    if retButton is not accButton:
                        raise ValueError("Saving Operation Failure")
                LUT.writeToTextFile(filename)
                dlgInfo('3D LUT written')
        except (ValueError, IOError) as e:
            dlgWarn(str(e))
        finally:
            # restore stack
            img.removeLayer(1)
            # set hald flag
            img.isHald = False
            img.layersStack[0].applyToStack()
            img.prLayer.update()
            window.label.repaint()
            return
    # unknown action
    else:
        return
    # adding a new layer may modify the resulting image
    # (cf. actionNew_Image_Layer), so we update the presentation layer
    layer.parentImage.prLayer.update()
    layer.parentImage.onImageChanged()  # TODO added 06/09/18 validate
    # record action name for scripting
    layer.actionName = name
    # docking the form
    dock = QDockWidget(window)
    dock.setWidget(grWindow)
    dock.setWindowFlags(grWindow.windowFlags())
    dock.setWindowTitle(grWindow.windowTitle())
    if TABBING:
        # add form to docking area
        forms = [ item.view for item in layer.parentImage.layersStack if getattr(item, 'view', None) is not None]
        dockedForms = [item for item in forms if not item.isFloating()]
        if dockedForms:
            window.tabifyDockWidget(dockedForms[-1], dock)
        else:
            window.addDockWidget(Qt.RightDockWidgetArea, dock)
    else:
        window.addDockWidget(Qt.RightDockWidgetArea, dock)
    layer.view = dock
    # update the view of layer stack
    window.tableView.setLayers(window.label.img)


def menuHelp(name):
    """
    Menu handler
    Init help browser
    A single instance is created.
    Unused parameters are for the sake of symmetry
    with other menu function calls.
    @param name: action name
    @type name: str
    """
    if name == "actionBlue_help":
        w = app.focusWidget()
        link = QFileInfo('help.html').absoluteFilePath()
        # init url
        url = QUrl(link)
        # add fragment identifier
        if hasattr(w, 'helpId'):
            if w.helpId != '':
                # unfortunately, on Windows Qt does not pass the fragment to the browser,
                # so we do nothing.
                # cf. https://bugreports.qt.io/browse/QTBUG-14460
                pass
                # url.setFragment(w.helpId)
        QDesktopServices.openUrl(url)
    elif name == "actionAbout_bLUe":
        w, label = handleTextWindow(parent=window, title='About bLUe', center=False)
        label.setStyleSheet("background-image: url(logo.png); color: white;")
        label.setAlignment(Qt.AlignCenter)
        label.setText(VERSION + "\n" + attributions)
        # center window on screen
        w.setGeometry(QStyle.alignedRect(Qt.LeftToRight, Qt.AlignCenter, w.size(),
                                         rootWidget.availableGeometry()))
        w.show()


def handleNewWindow(imImg=None, parent=None, title='New window', show_maximized=False, event_handler=True, scroll=False):
    """
    Shows a floating window containing a QLabel object. It can be used
    to display text or image. If the parameter event_handler is True (default)
    the QLabel object redefines its handlers for paint and mouse events to display
    the image imImg
    @param imImg: Image to display
    @type imImg: imImage
    @param parent:
    @param title:
    @param show_maximized:
    @param event_handler:
    @type event_handler: boolean
    @param scroll:
    @type scroll:
    @return: new window, label
    @rtype: QMainWindow, QLabel
    """
    newwindow = QMainWindow(parent)
    newwindow.setAttribute(Qt.WA_DeleteOnClose)
    newwindow.setWindowTitle(parent.tr(title))
    newwindow.setStyleSheet("background-color: rgb(220, 220, 220); color: black")
    label = QLabel()
    if scroll:
        scarea = QScrollArea(parent=newwindow)
        scarea.setWidget(label)
        newwindow.setCentralWidget(scarea)
        scarea.setWidgetResizable(True)
    else:
        newwindow.setCentralWidget(label)
    # The attribute img is used by event handlers
    label.img = imImg
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    if event_handler:
        set_event_handlers(label)
    if show_maximized:
        newwindow.showMaximized()
    else:
        newwindow.show()
    return newwindow, label


def handleTextWindow(parent=None, title='', center=True, wSize=QSize(500, 500)):
    """
    Display a floating modal text window

    @param parent:
    @type parent:
    @param title:
    @type title:
    @param center:
    @type center:
    @param wSize:
    @type wSize:
    @return new window, label
    @rtype: QMainWindow, QLabel
    """
    w, label = handleNewWindow(parent=parent, title=title, event_handler=False, scroll=True)
    w.setFixedSize(wSize)
    label.setAlignment(Qt.AlignTop)
    w.hide()
    if center:
        # center at the parent current screen
        pw = w.parent()
        pw = w if pw is None else pw
        w.move(pw.windowHandle().screen().geometry().center() - w.rect().center())
    w.setWindowModality(Qt.WindowModal)
    w.show()
    return w, label


def canClose():
    """
    Saves the current image. Returns True if success.
    @return:
    @rtype: boolean
    """
    if window.label.img.isModified:
        try:
            # save/discard dialog
            ret = saveChangeDialog(window.label.img)
            if ret == QMessageBox.Save:
                # save dialog
                filename = saveDlg(window.label.img, window)
                # confirm saving
                dlgInfo("%s written" % filename)
                return True
            elif ret == QMessageBox.Cancel:
                return False
        except (ValueError, IOError) as e:
            dlgWarn(str(e))
            return False
    return True


def updateStatus():
    """
    Display current status

    """
    img = window.label.img
    # filename and rating
    s = '&nbsp;&nbsp;&nbsp;&nbsp;' + img.filename + '&nbsp;&nbsp;&nbsp;&nbsp;' + (' '.join(['*']*img.meta.rating))
    # color management
    s = s + '&nbsp;&nbsp;&nbsp;&nbsp;CM : ' + ('On' if icc.COLOR_MANAGE else 'Off')
    # Preview
    if img.useThumb:
        s = s + '<font color=red><b>&nbsp;&nbsp;&nbsp;&nbsp;Preview</b></font> '
    else:
        # mandatory to toggle html mode
        s = s + '<font color=black><b>&nbsp;&nbsp;&nbsp;&nbsp;</b></font> '
    # Before/After
    if window.viewState == 'Before/After':
        s += '&nbsp;&nbsp;&nbsp;&nbsp;Before/After : Ctrl+Space : cycle through views - Space : switch back to workspace'
    else:
        s += '&nbsp;&nbsp;&nbsp;&nbsp;Press Space Bar to toggle Before/After view'
    # cropping
    if window.label.img.isCropped:
        s = s + '&nbsp;&nbsp;&nbsp;&nbsp;Crop Tool : h/w ratio %.2f ' % window.cropTool.formFactor
    window.Label_status.setText(s)


def initCursors():
    """
    Inits app cursors
    """
    # EyeDropper cursor
    curImg = QImage(":/images/resources/Eyedropper-icon.png")
    pxmp = QPixmap.fromImage(curImg)
    w, h = pxmp.width(), pxmp.height()
    window.cursor_EyeDropper = QCursor(pxmp, hotX=0, hotY=h-1)
    # tool cursor, must be resizable
    curImg = QImage(":/images/resources/cursor_circle.png")
    # turn to white
    curImg.invertPixels()
    window.cursor_Circle_Pixmap = QPixmap.fromImage(curImg)


def initDefaultImage():
    img = QImage(200, 200, QImage.Format_ARGB32)
    img.fill(Qt.darkGray)
    return imImage(QImg=img, meta=metadataBag(name='noName'))


def screenUpdate(newScreenIndex):
    """
    screenChanged event handler. The image is updated in background
    """
    window.screenChanged.disconnect()
    # update the color management object with the monitor profile associated to the current monitor
    icc.configure(qscreen=rootWidget.screen(newScreenIndex).windowHandle().screen())
    window.actionColor_manage.setEnabled(icc.HAS_COLOR_MANAGE)
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    updateStatus()
    # launch a bg task for image update

    def bgTask():
        window.label.img.updatePixmap()
        window.label_2.img.updatePixmap()
        window.label.update()
        window.label_2.update()
    threading.Thread(target=bgTask)
    window.screenChanged.connect(screenUpdate)

def getColorInfoView():
    """
    Return a color info view
    @return:
    @rtype: QDockWidget
    """
    infoView = colorInfoView()
    infoView.label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
    infoView.label.setMaximumSize(400, 80)
    return infoView

def setRightPane():
    """
    Convenient modifications of the right pane
    loaded from blue.ui
    """
    # clean dock area
    window.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.East)
    window.removeDockWidget(window.dockWidget)
    # redo the layout of window.dockWidget
    widget = window.dockWidget.widget()
    vl = QVBoxLayout()
    vl.addWidget(window.tableView)
    # add sliders, blend modes ...
    vl.addLayout(window.tableView.propertyLayout)
    window.propertyWidget.setLayout(vl)
    # reinit the dockWidgetContents (created by blue.ui) layout to
    # nest it in a QHboxLayout containing a left stretch
    tmpV = QVBoxLayout()
    while window.dockWidgetContents.layout().count() != 0:
        w = widget.layout().itemAt(0).widget()
        # dock the histogram on top
        if w.objectName() == 'histView':
            w.setWindowTitle('Hist')
            histViewDock = QDockWidget()
            hl = QHBoxLayout()
            hl.addStretch(1)
            hl.addWidget(w)
            hl.addStretch(1)
            wdg = QWidget()
            wdg.setLayout(hl)
            histViewDock.setWidget(wdg)
            histViewDock.setWindowTitle(w.windowTitle())
            window.addDockWidget(Qt.RightDockWidgetArea, histViewDock)
            continue
        # add other widgets to layout
        tmpV.addWidget(w)
    tmpH = QHBoxLayout()
    tmpH.setAlignment(Qt.AlignCenter)
    # prevent tmpV horizontal stretching
    tmpH.addStretch(1)
    tmpH.addLayout(tmpV)
    tmpH.addStretch(1)
    tmpH.setContentsMargins(0, 0, 10, 0)
    tmpV.setContentsMargins(0, 0, 10, 0)
    # to remove the current layout we re-parent it to
    # an unreferenced widget.
    QWidget().setLayout(window.dockWidgetContents.layout())
    # set the new layout
    widget.setLayout(tmpH)
    widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
    window.addDockWidget(Qt.RightDockWidgetArea, window.dockWidget)
    window.dockWidget.show()
    # tabify colorInfoView with histView
    window.infoView = getColorInfoView()
    window.tabifyDockWidget(histViewDock, window.infoView)


if __name__ == '__main__':
    #################
    # multiprocessing
    # freeze_support() must be called at the start of __main__
    # to enable multiprocessing when the executable is frozen.
    # Otherwise, it does nothing.
    #################
    multiprocessing.freeze_support()
    # load UI
    window.init()
    # splash screen
    splash = QSplashScreen(QPixmap('logo.png'), Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage("Loading .", color=Qt.white, alignment=Qt.AlignCenter)
    app.processEvents()
    sleep(1)
    splash.showMessage(VERSION + "\n" + attributions, color=Qt.white, alignment=Qt.AlignCenter)
    app.processEvents()
    sleep(1)
    splash.finish(window)
    # app title
    window.setWindowTitle('bLUe')
    # app style sheet
    if THEME == "light":
        app.setStyleSheet("""QMainWindow, QGraphicsView, QListWidget, QMenu, QTableView {background-color: rgb(200, 200, 200)}\
                           QWidget, QTableView, QTableView * {font-size: 9pt} QPushButton {font-size: 6pt}"""
                         )
    else:
        app.setStyleSheet("""QMainWindow, QMainWindow *, QGraphicsView, QListWidget, QMenu,
                                        QTableView, QLabel, QGroupBox {background-color: rgb(40,40,40); 
                                                                       color: rgb(220,220,220)}
                           QListWidget::item{background-color: rgb(40, 40, 40); color: white}
                           QMenu, QTableView {selection-background-color: blue;
                                               selection-color: white;}
                           QWidget, QComboBox, QTableView, QTableView * {font-size: 9pt}
                           QWidget:disabled {color: rgb(96,96,96)}
                           QbLUeSlider::handle:horizontal {
                                                background: white; 
                                                width: 15px;
                                                border: 1px solid black; 
                                                border-radius: 4px; 
                                                margin: -3px;
                                                }
                           QbLUeSlider::handle:horizontal:hover {
                                                background: #DDDDFF;
                                                }
                           QbLUeSlider::groove:horizontal {
                                                margin: 3px;
                                               }
                           QbLUeSlider::groove:horizontal:enabled { 
                                                background-color: rgb(196,196,196);}
                           QbLUeSlider::groove:horizontal:disabled { 
                                                background-color: rgb(96,96,96)}
                           QPushButton {font-size: 8pt;}
                           QbLUePushButton {font-size: 7pt;
                                            background-color: rgb(100,100,100);
                                            color: white;
                                            border: 2px solid gray;
                                            border-radius: 5px;
                                            padding: 4px;}
                           QbLUePushButton:hover, QbLUePushButton:pressed {background-color: rgb(150,150,200);}
                           QbLUePushButton:disabled {color: rgb(50,50,50)}
                           QGraphicsView QPushButton:hover, baseForm QPushButton:hover {background-color: gray;
                                                                                        color: black}
                           QToolButton {background-color: #444455;
                                        color: rgb(200,200,200);
                                        border: 1px solid gray;
                                        border-radius: 6px}
                           QToolButton:hover {background-color: #555588;
                                              color: black}
                           QToolButton:checked {background-color: blue}
                           QGroupBox#groupbox_btn {border: 1px solid gray;}
                           QGroupBox#groupBox {border: 1px solid gray;}
                           QMessageBox QLabel, QDialog QLabel {background-color: white; 
                                                               color: black}
                           QColorDialog QLabel {background-color: gray; 
                                                color: white}
                           QStatusBar::item {border: none}
                           QTabBar::tab {background: #444455; 
                                         color: lightgray;
                                         min-width: 8ex; 
                                         border: 2px solid white; 
                                         border-color: gray;
                                         border-bottom-left-radius: 4px; 
                                         border-bottom-right-radius: 4px;
                                         margin: 3px;
                                         padding: 2px}
                           QTabBar::tab:hover {color: white}
                           QTabBar::tab:selected {border-top-color: white; 
                                                  color: white;}
                           QTabBar::tab:!selected {margin-bottom: 2px}
                           QDockWidget::title {background-color: #444455}
                           QDockWidget::title:hover{background-color: #555588}
                           QToolTip {border: 0px;
                                    background-color: lightyellow;
                                    color: black}"""  # border must be set, otherwise background-color has no effect : Qt bug?
                         )

    # Before/After view
    splittedWin = splittedWindow(window)

    # status bar
    window.Label_status = QLabel()
    # window.Label_status.setStyleSheet("border: 15px solid white;")
    window.statusBar().addWidget(window.Label_status)
    # permanent text to right
    window.statusBar().addPermanentWidget(QLabel('Shift+F1 for Context Help       '))
    window.updateStatus = updateStatus
    window.label.updateStatus = updateStatus

    # crop tool
    window.cropTool = cropTool(parent=window.label)

    # whatsThis
    window.cropButton.setWhatsThis("""To crop the image drag a gray curtain on either side using the 8 small square buttons around the image""")
    window.rulerButton.setWhatsThis("""Draw horizontal and vertical rulers over the image""")
    window.fitButton.setWhatsThis("""Reset the image size to the window size""")
    window.eyeDropper.setWhatsThis("""Color picker\n Click on the image to sample pixel colors""")
    window.dragBtn.setWhatsThis("""Drag\n left button : drag the whole image\n Ctrl+Left button : drag the active layer only""")
    window.rectangle.setWhatsThis(
"""<b>Marquee Tool/Selection Rectangle</b><br>
Draw a selection rectangle on the active layer.<br>
For a segmentation layer only, all pixels outside the rectangle are set to background.
"""
                                )
    window.drawFG.setWhatsThis(
"""
<b>Foreground/Unmask tool</b><br>
  Paint on the active layer to <b>unmask</b> a previously masked region or to <b>select foreground pixels</b> (segmentation layer only);
  the mask must be enabled as opacity or color mask in the layer panel.<br>
  With <b>Color Mask</b> enabled, masked pixels are grayed and unmasked pixels are reddish.<br>
  Use the <b>Brush Size slider</b> below to choose the size of the tool. 
"""                             )
    window.drawBG.setWhatsThis(
"""<b>Background/Mask tool</b><br>
  Paint on the active layer to mask a region or to select background pixels (segmentation layer only);
  (the mask must be enabled as opacity or color mask in the layer panel).<br>
  With <b>Color Mask</b> enabled, masked pixels are grayed and unmasked pixels are reddish.<br>
  Use the 'Brush Size' slider below to choose the size of the tool. 
"""                             )
    window.verticalSlider1.setWhatsThis("""Set the diameter of the painting brush""")

    # Before/After views flag
    window.splittedView = False

    window.histView.mode = 'Luminosity'
    window.histView.chanColors = Qt.gray  # [Qt.red, Qt.green,Qt.blue]

    # close event handler
    window.onCloseEvent = canClose

    # watch mouse hover events
    window.label.setMouseTracking(True)

    # connect menu event handlers
    window.menu_File.aboutToShow.connect(updateEnabledActions)
    window.menuLayer.aboutToShow.connect(updateEnabledActions)
    window.menuImage.aboutToShow.connect(updateEnabledActions)
    window.menuWindow.aboutToShow.connect(updateEnabledActions)
    window.menuHelp.aboutToShow.connect(updateEnabledActions)
    window.menuOpen_recent.aboutToShow.connect(updateMenuOpenRecent)
    window.menu_File.triggered.connect(lambda a: menuFile(a.objectName()))
    window.menuLayer.triggered.connect(lambda a: menuLayer(a.objectName()))
    window.menuImage.triggered.connect(lambda a: menuImage(a.objectName()))
    window.menuWindow.triggered.connect(lambda a: menuView(a.objectName()))
    window.menuHelp.triggered.connect(lambda a: menuHelp(a.objectName()))

    #  called by all main form button and slider slots (cf. QtGui1.py)
    window.onWidgetChange = widgetChange

    set_event_handlers(window.label)
    set_event_handlers(window.label_2, enterAndLeave=False)
    set_event_handlers(window.label_3, enterAndLeave=False)
    # drag and drop event handlers are specific for the main window
    window.label.dropEvent = MethodType(lambda instance, e, wdg=window.label: dropEvent(wdg, wdg.img, e),
                                        window.label.__class__)
    window.label.dragEnterEvent = MethodType(lambda instance, e, wdg=window.label: dragEnterEvent(wdg, wdg.img, e),
                                             window.label.__class__)
    window.label.setAcceptDrops(True)

    defaultImImage = initDefaultImage()
    window.label.img = defaultImImage
    window.label_2.img = defaultImImage
    window.label_3.img = defaultImImage

    window.showMaximized()
    splash.finish(window)

    initCursors()

    # init Before/after view and cycling action
    window.splitter.setOrientation(Qt.Horizontal)
    window.splitter.currentState = next(splittedWin.splittedViews)
    window.splitter.setSizes([2 ** 20, 2 ** 20])
    window.splitter.setHandleWidth(1)
    window.splitter.hide()
    window.viewState = 'After'
    action1 = QAction('cycle', None)
    action1.setShortcut(QKeySequence("Ctrl+ "))

    def f():
        window.viewState = 'Before/After'
        splittedWin.nextSplittedView()
        updateStatus()
    action1.triggered.connect(f)
    window.addAction(action1)

    #########################################
    # dynamic modifications of the main form loaded
    # from blue.ui
    ########################################
    setRightPane()
    ################################
    # color management configuration
    # must be done after showing window
    ################################
    window.screenChanged.connect(screenUpdate)
    # screen detection
    c = window.frameGeometry().center()
    scn = rootWidget.screenNumber(c)
    window.currentScreenIndex = scn
    # update the color management object with the current monitor profile
    icc.configure(qscreen=rootWidget.screen(scn).windowHandle().screen())
    icc.COLOR_MANAGE = icc.HAS_COLOR_MANAGE
    window.actionColor_manage.setEnabled(icc.HAS_COLOR_MANAGE)
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    updateStatus()
    window.label.setWhatsThis(
""" <b>Main Window<br>
Menu File > Open</b> to edit a photo.<br>
<b>Menu Layer > New Adjustment Layer</b> to add an adjustment layer.<br>
<b>Ctrl+L or Menu View > Library Viewer</b> to browse a folder.<br>
<b>Ctrl+C or Menu View > Color Chooser</b> to display the color chooser.<br>
"""
    )  # end of setWhatsThis
    window.label_3.setWhatsThis(
""" <b>Before/After View : After Window</b><br>
Shows the modified image.<br>
<b>Ctrl+Space</b> to cycle through views.<br>
<b>Space</b> to switch back to normal view.<br>
"""
    )  # end of setWhatsThis
    window.label_2.setWhatsThis(
""" <b>Before/After View : Before Window</b><br>
Shows the initial image.<br>
<b>Ctrl+Space</b> to cycle through views.<br>
<b>Space</b> to switch back to normal view.
"""
    )  # end of setWhatsThis
    ###############
    # launch app
    ###############
    sys.exit(app.exec_())

