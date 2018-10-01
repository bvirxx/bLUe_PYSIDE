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

Unless stated in the specfic source file, this work is
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
import json
import multiprocessing
import sys
import threading
from itertools import cycle
from multiprocessing import freeze_support
from os import path, walk

from os.path import basename
from types import MethodType
import rawpy
from grabcut import segmentForm
from PySide2.QtCore import Qt, QRect, QEvent, QUrl, QSize, QFileInfo, QRectF, QObject
from PySide2.QtGui import QPixmap, QPainter, QCursor, QKeySequence, QBrush, QPen, QDesktopServices, QFont, \
    QPainterPath, QTransform, QContextMenuEvent
from PySide2.QtWidgets import QApplication, QAction, QFileDialog, QMessageBox, \
    QMainWindow, QLabel, QDockWidget, QSizePolicy, QScrollArea, QSplashScreen, QWidget, \
    QStyle, QToolTip, QHBoxLayout, QVBoxLayout, QColorDialog
from QtGui1 import app, window, rootWidget
import exiftool
from graphicsBlendFilter import blendFilterForm
from graphicsNoise import noiseForm
from graphicsRaw import rawForm
from graphicsTransform import transForm, imageForm
from imgconvert import *
from MarkedImg import imImage, metadataBag, vImage, QLayerImage, QLayer
from graphicsRGBLUT import graphicsForm
from graphicsLUT3D import graphicsForm3DLUT
from colorCube import LUTSIZE, LUT3D, LUT3DIdentity
from colorModels import cmHSP, cmHSB
from colorManagement import icc
from graphicsCoBrSat import CoBrSatForm
from graphicsExp import ExpForm
from graphicsPatch import patchForm
from settings import USE_POOL, POOL_SIZE
from utils import saveChangeDialog, saveDlg, openDlg, cropTool, rotatingTool, IMAGE_FILE_NAME_FILTER, \
    IMAGE_FILE_EXTENSIONS, RAW_FILE_EXTENSIONS, demosaic, dlgWarn, dlgInfo
from graphicsTemp import temperatureForm
from time import sleep
import gc
from graphicsFilter import filterForm
from graphicsHspbLUT import graphicsHspbForm
from graphicsLabLUT import graphicsLabForm
from splittedView import splittedWindow


##################
#  Software Attributions
from viewer import playDiaporama, viewer

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
"""
#################

##############
#  Version number
VERSION = "v1.2.1.3"
##############

##############
# adjustment form size
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
# stuff for Before/After marks
qp.font = QFont("Arial", 8)
qp.markPath=QPainterPath()
qp.markRect = QRect(0, 0, 50, 20)
qp.markPath.addRoundedRect(qp.markRect, 5, 5)

def paintEvent(widg, e) :
    """
    Paint event handler.
    The handler displays a vImage object in a Qlabel, with
    current offset and zooming coefficient.
    The widget must have a valid img attribute of type vImage.
    The handler should be used override the paintEvent method of widg. This can be done
    by subclassing (not directly compatible with Qt Designer) or by assigning paintEvent
    to widg.paintEvent (cf. the function set_event_handler below).
    @param widg: widget
    @type widg: object with a img attribute of type vImage
    @param e: paint event
    """
    mimg = getattr(widg, 'img', None)
    if mimg is None:
        return
    r = mimg.resize_coeff(widg)
    qp.begin(widg)
    # smooth painting
    qp.setRenderHint(QPainter.SmoothPixmapTransform)  # TODO useless
    # fill background
    qp.fillRect(QRect(0, 0, widg.width() , widg.height() ), vImage.defaultBgColor)
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
    # draw selection rectangle for the active layer only
    layer = mimg.getActiveLayer()
    rect = layer.rect
    if layer.visible and (rect is not None ):
        qp.setPen(QColor(0, 255, 0))
        qp.drawRect(rect.left()*r + mimg.xOffset, rect.top()*r +mimg.yOffset, rect.width()*r, rect.height()*r)
    # cropping marks
    lm, rm, tm, bm = 0, 0, 0, 0
    if mimg.isCropped:
        c = QColor(128,128,128, 192)
        lm = window.cropTool.btnDict['left'].margin*r
        rm =  window.cropTool.btnDict['right'].margin*r
        tm =  window.cropTool.btnDict['top'].margin*r
        bm =  window.cropTool.btnDict['bottom'].margin*r
        #left
        qp.fillRect(QRectF(mimg.xOffset, mimg.yOffset, lm, h), c)
        #top
        qp.fillRect(QRectF(mimg.xOffset+lm, mimg.yOffset, w - lm, tm), c)
        #right
        qp.fillRect(QRectF(mimg.xOffset+w-rm, mimg.yOffset+tm, rm, h-tm),  c)
        #bottom
        qp.fillRect(QRectF(mimg.xOffset+lm, mimg.yOffset+h-bm, w-lm-rm, bm), c)
    # rulers
    if mimg.isRuled:
        deltaX, deltaY = (w-lm-rm)//3, (h-tm-bm)//3
        qp.drawLine(lm+mimg.xOffset, deltaY+tm+mimg.yOffset, w-rm+mimg.xOffset, deltaY+tm+mimg.yOffset)
        qp.drawLine(lm+mimg.xOffset, 2*deltaY+tm+mimg.yOffset, w-rm+mimg.xOffset, 2*deltaY+tm+mimg.yOffset)
        qp.drawLine(deltaX+lm+mimg.xOffset, tm+mimg.yOffset, deltaX+lm+mimg.xOffset, h-bm+mimg.yOffset)
        qp.drawLine(2*deltaX+lm+mimg.xOffset, tm+mimg.yOffset, 2*deltaX+lm+mimg.xOffset, h-bm+mimg.yOffset)
    # mark before/after views
    name = widg.objectName()
    if name == "label_2" or name == "label_3":
        # draw filled rect
        qp.fillPath(qp.markPath, QBrush(Qt.gray))
        # draw text
        qp.setPen(Qt.white)
        qp.setFont(qp.font)
        qp.drawText(qp.markRect, Qt.AlignCenter | Qt.AlignVCenter, "Before" if name == "label_2" else "After" )
    qp.end()

# global variables used in mouseEvent. CAUTION:  non thread safe approach
# Recording of state and mouse coordinates (relative to widget)
State = {'ix':0, 'iy':0, 'ix_begin':0, 'iy_begin':0} #{'drag':False, 'drawing':False , 'tool_rect':False, 'rect_over':False,
pressed=False
clicked = True
def mouseEvent(widget, event) :  # TODO split into 3 handlers
    """
    Mouse event handler.
    The handler implements mouse actions on a vImage
    displayed in a Qlabel.
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
    @type event:
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
    if eventType == QEvent.MouseButtonPress :
        # Mouse hover generates mouse move events,
        # so, we set pressed to only select non hovering events
        pressed=True
        if event.button() == Qt.LeftButton:
            # no move yet
            clicked=True
        State['ix'], State['iy'] = x, y
        State['ix_begin'], State['iy_begin'] = x, y
        State['x_imagePrecPos'], State['y_imagePrecPos'] = (x - img.xOffset) // r, (y - img.yOffset) // r
        return  # no update needed
    ##################
    # mouse move event
    ##################
    elif eventType == QEvent.MouseMove :
        # skip hover events
        if not pressed:
            return
        clicked=False
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
                    w = window.verticalSlider1.value() // (2 * r)
                    # mode source : result is source (=pen) pixel color and opacity
                    qp.setCompositionMode(qp.CompositionMode_Source)
                    tmp_x = (x - img.xOffset) // r
                    tmp_y = (y - img.yOffset) // r
                    qp.drawEllipse(tmp_x-w//2, tmp_y-w//2, w, w)
                    qp.setPen(QPen(color, 2*w))
                    qp.drawLine(State['x_imagePrecPos'], State['y_imagePrecPos'], tmp_x, tmp_y)
                    qp.end()
                    State['x_imagePrecPos'], State['y_imagePrecPos'] = tmp_x, tmp_y
                    # update upper stack
                    # should be layer.updateStack() if any upper layer visible : too slow
                    layer.updatePixmap(maskOnly=True)
                    img.prLayer.applyNone()
                    # img.prLayer.updatePixmap(maskOnly=True) # TODO called by applyNone 19/06/18
                    window.label.repaint()
            # dragBtn or arrow
            else:
                # drag image
                if modifiers == Qt.NoModifier:
                    img.xOffset+=(x-State['ix'])
                    img.yOffset+=(y-State['iy'])
                    if window.btnValues['Crop_Button']:
                        window.cropTool.drawCropTool(img)
                # drag active layer only
                elif modifiers == Qt.ControlModifier:
                    layer.xOffset += (x - State['ix'])
                    layer.yOffset += (y - State['iy'])
                    layer.updatePixmap()
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
        #update current coordinates
        State['ix'],State['iy']=x,y
        # Pick color from active layer. Coordinates are relative to the full-sized image
        if window.btnValues['colorPicker']:
            x_img, y_img = (x - img.xOffset) / r, (y - img.yOffset) / r
            x_img, y_img = min(int(x_img), img.width() - 1), min(int(y_img), img.height() - 1)
            r, g, b = img.getActivePixel(x_img, y_img)
            s = ('%s  %s  %s' % (r, g, b))
            QToolTip.showText(event.globalPos(), s, window, QRect(event.globalPos(), QSize(20,30)))
        if layer.isGeomLayer():
            #layer.view.widget().tool.moveRotatingTool()
            layer.tool.moveRotatingTool()
    #####################
    # mouse release event
    # mouse click event
    ####################
    elif eventType == QEvent.MouseButtonRelease :
        pressed=False
        if event.button() == Qt.LeftButton:
            if layer.maskIsEnabled \
                    and layer.getUpperVisibleStackIndex() != -1\
                    and (window.btnValues['drawFG'] or window.btnValues['drawBG']):
                layer.applyToStack()
            if img.isMouseSelectable:
                # click event
                if clicked:
                    x_img, y_img = (x - img.xOffset) / r, (y - img.yOffset) / r
                    x_img, y_img = min(int(x_img), img.width()-1), min(int(y_img), img.height()-1)
                    # Pick color from active layer. Coordinates are relative to the full-sized image
                    red, green, blue = img.getActivePixel(x_img, y_img)
                    if getattr(window, 'colorChooser', None) is not None:
                        if window.colorChooser.isVisible():
                            window.colorChooser.setCurrentColor(QColor(red,green,blue))
                    # emit colorPicked signal
                    layer.colorPicked.sig.emit(x_img, y_img, modifiers)
                    # select grid node for 3DLUT form
                    if layer.is3DLUTLayer():
                        layer.view.widget().selectGridNode(red, green, blue)
                    if window.btnValues['rectangle'] and (modifiers == Qt.ControlModifier):
                        layer.rect = None                               # TODO added 18/02/18 to clear selection
                    # for raw layer, set multipliers to get selected pixel as White Point
                    if layer.isRawLayer() and window.btnValues['colorPicker']:
                        bufRaw = layer.parentImage.demosaic
                        # demosaiced buffer
                        nb = QRect(x_img-2, y_img-2, 4, 4)
                        r = QImage.rect(layer.parentImage).intersected(nb)
                        if not r.isEmpty():
                            color = np.sum(bufRaw[r.top():r.bottom()+1, r.left():r.right()+1], axis=(0,1))/(r.width()*r.height())
                        else:
                            color = bufRaw[y_img, x_img, :]  # TODO added 25/06/18 to avoid uninit. color validate
                        #color = bufRaw[y_img, x_img, :]
                        color = [color[i] - layer.parentImage.rawImage.black_level_per_channel[i] for i in range(3)]
                        form =layer.view.widget()
                        if form.sampleMultipliers:
                            row, col = 3*y_img//layer.height(), 3*x_img//layer.width()
                            if form.samples:
                                form.setRawMultipliers(*form.samples[3*row + col], sampling=False)
                        else:
                            form.setRawMultipliers(1/color[0], 1/color[1], 1/color[2], sampling=True)
                """
                # cloning layer
                if layer.isCloningLayer():
                    if modifiers == Qt.ControlModifier | Qt.AltModifier:
                        if layer.keepCloned:
                            layer.maskIsEnabled = False
                            layer.maskisSelected = False
                            layer.applyCloning(seamless=False)
                        # update mask status in the table of layers
                        layer.updateTableView(window.tableView)
                """
    # updates
    widget.repaint()
    # sync splitted views
    linked = True
    if widget.objectName() == 'label_2' :
        splittedWin.syncSplittedView(window.label_3, window.label_2, linked)
        window.label_3.repaint()
    elif widget.objectName() == 'label_3':
        splittedWin.syncSplittedView(window.label_2, window.label_3, linked)
        window.label_2.repaint()

def wheelEvent(widget,img, event):
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
        if img.Zoom_coeff >2:
            img.Zoom_coeff /= (1.0 + numSteps)
            return
        # correct image offset to keep unchanged the image point
        # under the cursor : (pos - offset) / resize_coeff should be invariant
        img.xOffset = -pos.x() * numSteps + (1.0+numSteps)*img.xOffset
        img.yOffset = -pos.y() * numSteps + (1.0+numSteps)*img.yOffset
        if window.btnValues['Crop_Button']:
            window.cropTool.drawCropTool(img)
        if layer.isGeomLayer():
            #layer.view.widget().tool.moveRotatingTool()
            layer.tool.moveRotatingTool()
    elif modifiers == Qt.ControlModifier:
        layer.Zoom_coeff *= (1.0 + numSteps)
        layer.updatePixmap()
    elif modifiers == Qt.ControlModifier | Qt.AltModifier:
        if layer.isCloningLayer():
            layer.AltZoom_coeff *= (1.0 + numSteps)
            layer.autoclone = False
            layer.applyCloning(seamless=False)
            #layer.updatePixmap()
    widget.repaint()
    # sync splitted views
    linked = True
    if widget.objectName() == 'label_2':
        splittedWin.syncSplittedView(window.label_3, window.label_2, linked)
        window.label_3.repaint()
    elif widget.objectName() == 'label_3':
        splittedWin.syncSplittedView(window.label_2, window.label_3, linked)
        window.label_2.repaint()

def enterEvent(widget,img, event):
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
                QApplication.setOverrideCursor(window.cursor_Circle_Pixmap.scaled(w*1.5, w*1.5))
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

def leaveEvent(widget,img, event):
    QApplication.restoreOverrideCursor()

def dragEnterEvent(widget, img, event):
    """
    Accepts drop if mimeData contains text (e.g. file name)
    (convenient for main window only)
    @param widget:
    @type widget:
    @param img:
    @type img:
    @param event:
    @type event:
    """
    if (event.mimeData().hasFormat("text/plain")):
        event.acceptProposedAction()

def dropEvent(widget, img, event):
    """
    gets file name from event.mimeData and opens it.
    (Convenient for main window only)
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
    subclassing or overridding. However, the PySide dynamic
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
    widg.mousePressEvent = MethodType(lambda instance, e, wdg=widg : mouseEvent(wdg, e), widg.__class__)
    widg.mouseMoveEvent = MethodType(lambda instance, e, wdg=widg : mouseEvent(wdg, e), widg.__class__)
    widg.mouseReleaseEvent = MethodType(lambda instance, e, wdg=widg : mouseEvent(wdg, e), widg.__class__)
    widg.wheelEvent = MethodType(lambda instance, e, wdg=widg : wheelEvent(wdg, wdg.img, e), widg.__class__)
    if enterAndLeave:
        widg.enterEvent = MethodType(lambda instance, e, wdg=widg : enterEvent(wdg, wdg.img, e), widg.__class__)
        widg.leaveEvent = MethodType(lambda instance, e, wdg=widg : leaveEvent(wdg, wdg.img, e), widg.__class__)

def widgetChange(button):
    """
    called by all main form button and slider slots (cf. QtGui1.py onWidgetChange)

    @param button:
    @type button: QWidget
    """
    wdgName = button.objectName()
    if wdgName == "fitButton" :
        window.label.img.fit_window(window.label)
        # update crop button positions
        window.cropTool.drawCropTool(window.label.img)
        window.label.repaint()
    elif wdgName == "cropButton":
        if button.isChecked():
            window.cropTool.drawCropTool(window.label.img)
            for b in window.cropTool.btnDict.values():
                b.show()
        else:
            for b in window.cropTool.btnDict.values():
                b.hide()
        window.label.img.isCropped = button.isChecked()
        window.label.repaint()
    elif wdgName == "rulerButton":
        window.label.img.isRuled = button.isChecked()
        window.label.repaint()

def contextMenu(pos, widget):
    """
    Copntext menu for image QLabel
    @param pos:
    @param widget:
    @return:
    """
    pass

def loadImageFromFile(f, createsidecar=True):
    """
    loads an imImage (metadata and image) from file. Returns the loaded imImage :
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
            profile, metadata = e.get_metadata(f, createsidecar=createsidecar)
    except ValueError:
        # Default metadata and profile
        metadata = [{'SourceFile': f}]
        profile = ''
    # color space : 1=sRGB
    if "EXIF:ColorSpace" in metadata[0].keys():
        colorSpace = metadata[0].get("EXIF:ColorSpace")
    else:
        colorSpace = metadata[0].get("MakerNotes:ColorSpace", -1)
    if colorSpace < 0:
        desc_colorSpace = metadata[0].get("ICC_Profile:ProfileDescription", '')
        if isinstance(desc_colorSpace, str):# or isinstance(desc_colorSpace, unicode): python3
            if 'sRGB' in desc_colorSpace:
                # sRGBIEC61966-2.1
                colorSpace = 1
    # orientation
    orientation = metadata[0].get("EXIF:Orientation", 0)
    transformation = exiftool.decodeExifOrientation(orientation)
    # rating
    rating = metadata[0].get("XMP:Rating", 5)
    ############
    # load image
    ############
    name = path.basename(f)
    ext = name[-4:]
    if ext in list(IMAGE_FILE_EXTENSIONS):
        img = imImage(filename=f, colorSpace=colorSpace, orientation=transformation, rawMetadata=metadata, profile=profile, name=name, rating=rating)
    elif ext in list(RAW_FILE_EXTENSIONS):
        # load raw image file
        # rawpy.imread keeps f open. Calling raw.close() deletes the raw object.
        # As a workaround we use a file buffer .
        raw = rawpy.RawPy()
        with open(f, "rb") as bufio:
            raw.open_buffer(bufio)
        raw.unpack()
        # postprocess raw image with default parameters (cf. vImage.applyRawPostProcessing)
        rawBuf = raw.postprocess(use_camera_wb=True)
        # build Qimage
        rawBuf = np.dstack((rawBuf[:,:,::-1], np.zeros(rawBuf.shape[:2], dtype=np.uint8)+255))
        img = imImage(cv2Img=rawBuf, colorSpace=colorSpace, orientation=transformation, rawMetadata=metadata, profile=profile, name=name, rating=rating)
        # keep references to file and RawPy instance
        img.rawImage = raw
        img.filename = f
        #######################################
        # Reconstruct the demosaic Bayer bitmap :
        # it is needed to calculate the multipliers corresponding
        # to a user white point and we cannot access the
        # rawpy native demosaic buffer from RawPy instance
        #######################################
        # get 16 bits Bayer bitmap
        img.demosaic = demosaic(raw.raw_image_visible, raw.raw_colors_visible, raw.black_level_per_channel)
        # correct orientation
        if orientation == 6: # 90°
            img.demosaic = np.swapaxes(img.demosaic, 0, 1)
        elif orientation == 8: # 270°
            img.demosaic = np.swapaxes(img.demosaic, 0, 1)
            img.demosaic = img.demosaic[:,::-1,:]
    else:
        raise ValueError("Cannot read file %s" % f)
    if img.isNull():
        raise ValueError("Cannot read file %s" % f)
    window.settings.setValue('paths/dlgdir', QFileInfo(f).absoluteDir().path())
    img.initThumb()
    if img.format() in [QImage.Format_Invalid, QImage.Format_Mono, QImage.Format_MonoLSB, QImage.Format_Indexed8]:
        raise ValueError("Cannot edit indexed formats\nConvert image to a non indexed mode first")
    if colorSpace < 0 and not getattr(window, 'modeDiaporama', False) :
        # setOverrideCursor does not work correctly for a MessageBox :
        # may be a Qt Bug, cf. https://bugreports.qt.io/browse/QTBUG-42117
        QApplication.changeOverrideCursor(QCursor(Qt.ArrowCursor))
        QApplication.processEvents()
        dlgInfo("Color profile missing\nAssigning sRGB")
        img.meta.colorSpace = 1
        img.updatePixmap()
    return img

def addBasicAdjustmentLayers(img):
    if img.rawImage is None:
        #menuLayer('actionColor_Temperature')
        #menuLayer('actionExposure_Correction')
        menuLayer('actionContrast_Correction')
    # select active layer : top row
    window.tableView.select(0, 1)

def addRawAdjustmentLayer():
    """
    Adds a development layer to the layer stack

    """
    lname = 'Development'
    l = window.label.img.addAdjustmentLayer(name=lname, role='RAW')
    grWindow = rawForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window,
                                            mainForm=window)
    # wrapper for the right apply method
    l.execute = lambda l=l, pool=None: l.tLayer.applyRawPostProcessing()
    # record action name for scripting
    l.actionName = ''
    # dock the form
    dock = QDockWidget(window)
    dock.setWidget(grWindow)
    dock.setWindowFlags(grWindow.windowFlags())
    dock.setWindowTitle(grWindow.windowTitle())
    dock.move(900, 40)
    dock.setStyleSheet("QGraphicsView{margin: 10px; border-style: solid; border-width: 1px; border-radius: 1px;}")
    l.view = dock
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
    try :
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        img = loadImageFromFile(f)
        # display image
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
    Closes the opened document and clears windows.
    @return:
    @rtype: boolean
    """
    if not canClose():
        return False
    # watch memory leak : set weakref to image
    # r = weakref.ref(window.label.img)
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
    Inits GUI to display the current document
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
        # recompute and display histogram
        if window.histView.listWidget1.items['Original Image'].checkState() is Qt.Checked:
            histImg = vImage(QImg=window.label.img.getCurrentImage()) # must be vImage : histogram method needed
        else:
            histImg = window.label.img.layersStack[-1].getCurrentMaskedImage() # vImage(QImg=window.label.img.layersStack[-1].getCurrentMaskedImage())#mergeVisibleLayers())
        if window.histView.listWidget2.items['Color Chans'].checkState() is Qt.Checked:
            window.histView.mode = 'RGB'
            window.histView.chanColors = [QColor(255,0,0), QColor(0,255,0), QColor(10,10,255)]
        else:
            window.histView.mode = 'Luminosity'
            window.histView.chanColors = [Qt.gray]
        histView = histImg.histogram(QSize(window.histView.width(), window.histView.height()), chans=list(range(3)), bgColor=Qt.black,  # TODO 03/07/18 list added
                                     chanColors=window.histView.chanColors, mode=window.histView.mode, addMode='')
        window.histView.Label_Hist.setPixmap(QPixmap.fromImage(histView))
        window.histView.Label_Hist.repaint()
        # window.label.img.layersStack[-1].updatePixmap()  # TODO 17/06/18 useless?
        # window.label.repaint()                           # TODO 17/06/18 useless?

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
    window.label.img.window = window.label
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
    for filename in recentFiles :
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
    if name in ['actionOpen'] : #, 'actionHald_from_file'] :
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
        """
    elif name == 'actionHald_identity':
        img = QImage(LUTSIZE, LUTSIZE**2, QImage.Format_ARGB32)
        buf = QImageBuffer(img)
        buf[:,:,:3] = LUT3DIdentity.getHaldImage(LUTSIZE, LUTSIZE**2)
        img1 = imImage(QImg=img)
        img1.initThumb()
        setDocumentImage(img1)
        img1.isHald = True
    elif name == 'actionSave_Hald_Cube':
        # apply stack
        doc = window.label.img
        if not doc.isHald:
            return
        doc.layersStack[0].applyToStack()
        window.label.repaint()
        # get resulting image
        img = doc.mergeVisibleLayers()
        # convert image to LUT3D object
        LUT = LUT3D.HaldImage2LUT3D(img, size=33)
        # open file and save
        lastDir = str(window.settings.value('paths/dlgdir', '.'))
        dlg = QFileDialog(window, "select", lastDir)
        dlg.setNameFilter('*.cube')
        dlg.setDefaultSuffix('cube')
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            newDir = dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
            LUT.writeToTextFile(filenames[0])
        """
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
    if name == 'actionShow_hide_right_window_3' :
        if window.splitter.isHidden() :
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
                newDir = dlg.selectedFiles()[0] # dlg.directory().absolutePath()
                window.settings.setValue('paths/dlgdir', newDir)
                for dirpath, dirnames, filenames in walk(newDir):
                    for filename in [f for f in filenames if
                                f.endswith(IMAGE_FILE_EXTENSIONS)]:
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
            newDir = dlg.selectedFiles()[0] # dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
            viewerInstance = viewer.getViewerInstance(mainWin=window)
            viewerInstance.playViewer(newDir)
    ###############
    # Color Chooser
    ###############
    elif name == 'actionColor_Chooser':
        if getattr(window, 'colorChooser', None) is None:
            window.colorChooser = QColorDialog(parent=window)
        window.colorChooser.show()
    updateStatus()

def menuImage(name) :
    """
    Menu handler
    @param name: action name
    @type name: str
    """
    img = window.label.img
    # display image info
    if name == 'actionImage_info' :
        # Format
        s = "Format : %s\n(cf. QImage formats in the doc for more info)" % QImageFormats.get(img.format(), 'unknown')
        # dimensions
        s = s + "\n\ndim : %d x %d" % (img.width(), img.height())
        workingProfileInfo = icc.workingProfileInfo
        s = s + "\n\nWorking Profile : %s" % workingProfileInfo
        # embedded profile
        if len(img.meta.profile) > 0:
            s = s +"\n\nEmbedded profile found, length %d" % len(img.meta.profile)
        s = s + "\nRating %s" % ''.join(['*']*img.meta.rating)
        # get raw meta data dictionary
        l = img.meta.rawMetadata
        s = s + "\n\nMETADATA :\n"
        for d in l:
            s = s + '\n'.join('%s : %s' % (k,v) for k, v in d.items()) # python 3 iteritems -> items
        w, label = handleTextWindow(parent=window, title='Image info')
        label.setWordWrap(True)
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

def menuLayer(name):
    """
    Menu Layer handler
    @param name: action name
    @type name: str
    """
    global pool
    # curves
    if name in ['actionCurves_RGB', 'actionCurves_HSpB', 'actionCurves_Lab']:
        if name == 'actionCurves_RGB':
            layerName = 'RGB'
            form = graphicsForm
        elif name == 'actionCurves_HSpB':
            layerName = 'HSV'
            form = graphicsHspbForm
        elif name == 'actionCurves_Lab':
            layerName = 'Lab'
            form = graphicsLabForm
        # add new layer on top of active layer
        l = window.label.img.addAdjustmentLayer(name=layerName)
        grWindow=form.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # wrapper for the right applyXXX method
        if name == 'actionCurves_RGB':
            l.execute = lambda l=l, pool=None: l.tLayer.apply1DLUT(grWindow.scene().cubicItem.getStackedLUTXY())
        elif name == 'actionCurves_HSpB':
            l.execute = lambda l=l, pool=None: l.tLayer.applyHSV1DLUT(grWindow.scene().cubicItem.getStackedLUTXY(), pool=pool)
        elif name == 'actionCurves_Lab':
            l.execute = lambda l=l, pool=None: l.tLayer.applyLab1DLUT(grWindow.scene().cubicItem.getStackedLUTXY())
    # 3D LUT
    elif name in ['action3D_LUT', 'action3D_LUT_HSB']:
        # color model
        ccm = cmHSP if name == 'action3D_LUT' else cmHSB
        layerName = '3D LUT HSpB' if name == 'action3D_LUT' else '3D LUT HSV'
        l = window.label.img.addAdjustmentLayer(name=layerName, role='3DLUT')
        grWindow = graphicsForm3DLUT.getNewWindow(ccm, axeSize=300, targetImage=window.label.img, LUTSize=LUTSIZE, layer=l, parent=window, mainForm=window)
        # init pool only once
        if USE_POOL and (pool is None):
            print('launching process pool...', end='')
            pool = multiprocessing.Pool(POOL_SIZE)
            print('done')
        l.execute = lambda l=l, pool=pool: l.tLayer.apply3DLUT(grWindow.scene().LUT3DArray, options=grWindow.scene().options, pool=pool)
    # cloning
    elif name == 'actionNew_Cloning_Layer':
        lname = 'Cloning'
        l = window.label.img.addAdjustmentLayer(name=lname, role='CLONING')
        grWindow = patchForm.getNewWindow(targetImage=window.label.img, layer=l, mainForm=window)
        l.execute = lambda l=l, pool=None: l.tLayer.applyCloning(seamless=l.autoclone)
    # segmentation
    elif name == 'actionNew_segmentation_layer':
        lname = 'Segmentation'
        l = window.label.img.addSegmentationLayer(name=lname)
        grWindow = segmentForm.getNewWindow(targetImage=window.label.img, layer=l, mainForm=window)
        l.execute = lambda l=l, pool=None: l.tLayer.applyGrabcut(nbIter=grWindow.nbIter)
        # mask was modified
        #l.updatePixmap()
    # loads an image from file
    elif name == 'actionLoad_Image_from_File': #'actionNew_Image_Layer':
        filename = openDlg(window, ask=False)
        if filename is None:
            return
        # load image from file, alpha channel is mandatory for applyTransform()
        imgNew = QImage(filename).convertToFormat(QImage.Format_ARGB32)  # QImage(filename, QImage.Format_ARGB32) does not work !
        if imgNew.isNull():
            dlgWarn("Cannot load %s: " % filename)
            return
        lname = path.basename(filename)
        l = window.label.img.addAdjustmentLayer(name=lname, sourceImg=imgNew, role='GEOMETRY')
        grWindow = imageForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # add transformation tool to parent widget
        tool = rotatingTool(parent=window.label)#, layer=l, form=grWindow)
        l.addTool(tool)
        tool.showTool()
        l.execute = lambda l=l, pool=None: l.tLayer.applyImage(grWindow.options)
        l.actioname = name
    # empty new image
    elif name == 'actionNew_Layer':
        processedImg = window.label.img
        w, h = processedImg.width(), processedImg.height()
        imgNew = QImage(w, h, QImage.Format_ARGB32)
        imgNew.fill(Qt.black)
        lname = 'Image'
        l = window.label.img.addAdjustmentLayer(name=lname, sourceImg=imgNew, role='GEOMETRY')
        grWindow = imageForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window,
                                          mainForm=window)
        # add transformation tool to parent widget
        tool = rotatingTool(parent=window.label)  # , layer=l, form=grWindow)
        l.addTool(tool)
        tool.showTool()
        l.execute = lambda l=l, pool=None: l.tLayer.applyImage(grWindow.options)
        l.actioname = name
    # Temperature
    elif name == 'actionColor_Temperature':
        lname = 'Color Temperature'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = temperatureForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.tLayer.applyTemperature()
    elif name == 'actionContrast_Correction':
        l = window.label.img.addAdjustmentLayer(name=CoBrSatForm.layerTitle, role='CONTRAST')
        grWindow = CoBrSatForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # clipLimit change event handler
        def h(lay, clipLimit):
            lay.clipLimit = clipLimit
            lay.applyToStack()
            window.label.img.onImageChanged()
        grWindow.onUpdateContrast = h
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.tLayer.applyContrast()
    elif name == 'actionExposure_Correction':
        lname = 'Exposure'
        l = window.label.img.addAdjustmentLayer(name=lname)
        l.clipLimit = ExpForm.defaultExpCorrection
        grWindow = ExpForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # clipLimit change event handler
        def h(lay, clipLimit):
            lay.clipLimit = clipLimit
            lay.applyToStack()
            window.label.img.onImageChanged()
        grWindow.onUpdateExposure = h
        # wrapper for the right apply method
        l.execute = lambda l=l,  pool=None: l.tLayer.applyExposure(l.clipLimit, grWindow.options)
    elif name == 'actionGeom_Transformation':
        lname = 'Transformation'
        l = window.label.img.addAdjustmentLayer(name=lname, role='GEOMETRY')
        grWindow = transForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # add transformation tool to parent widget
        tool = rotatingTool(parent=window.label)#, layer=l, form=grWindow)
        l.addTool(tool)
        tool.showTool()
        l.execute = lambda l=l, pool=None: l.tLayer.applyTransForm(grWindow.options)
    elif name == 'actionFilter':
        lname = 'Filter'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = filterForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.tLayer.applyFilter2D()
    elif name == 'actionGradual_Filter':
        lname = 'Gradual Filter'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = blendFilterForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.tLayer.applyBlendFilter()
    elif name == 'actionNoise_Reduction':
        lname='Noise Reduction'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = noiseForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.tLayer.applyNoiseReduction()
        """
    elif name == 'actionSave_Layer_Stack':
        return # TODO 26/06/18 should be reviewed
        lastDir = str(window.settings.value('paths/dlg3DLUTdir', '.'))
        dlg = QFileDialog(window, "select", lastDir)
        dlg.setNameFilter('*.sba')
        dlg.setDefaultSuffix('sba')
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            newDir = dlg.directory().absolutePath()
            window.settings.setValue('paths/dlg3DLUTdir', newDir)
            window.label.img.saveStackToFile(filenames[0])
            return
    elif name == 'actionLoad_Layer_Stack':
        lastDir = str(window.settings.value('paths/dlgdir', '.'))
        dlg = QFileDialog(window, "select", lastDir)
        dlg.setNameFilter('*.sba')
        dlg.setDefaultSuffix('sba')
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            newDir = dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
            script, qf, dataStream = window.label.img.loadStackFromFile(filenames[0])
            script = '\n'.join(script)
            # secure env for exec
            safe_list = ['menuLayer', 'window']
            safe_dict = dict([(k, globals().get(k, None)) for k in safe_list])
            #exec script in safe_dict, locals() #globals(), locals()
            exec (script, safe_dict, locals)  #3.6
            qf.close()
            return
        """
    # invert image
    elif name == 'actionInvert':
        lname = 'Invert'
        l = window.label.img.addAdjustmentLayer(name=lname)
        l.execute = lambda l=l : l.tLayer.applyInvert()
        window.tableView.setLayers(window.label.img)
        l.applyToStack()
        l.parentImage.prLayer.update()
        l.parentImage.onImageChanged()
        return
    # load 3D LUT from .cube file
    elif name == 'actionLoad_3D_LUT' :
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
                LUT3DArray, size = LUT3D.readFromTextFile(name)
            except (ValueError, IOError) as e:
                dlgWarn('Unable to load 3D LUT : ', info=str(e))
                return
            lname = path.basename(name)
            l = window.label.img.addAdjustmentLayer(name=lname)
            # init pool only once
            if USE_POOL and (pool is None):
                print('launching process pool...', end='')
                pool = multiprocessing.Pool(POOL_SIZE)
                print('done')
            l.execute = lambda l=l, pool=pool: l.tLayer.apply3DLUT(LUT3DArray, {'use selection': False}, pool=pool)
            window.tableView.setLayers(window.label.img)
            l.applyToStack()
            # The resulting image is modified,
            # so we update the presentation layer before returning
            l.parentImage.prLayer.update()
            l.parentImage.onImageChanged()
        return
    elif name == 'actionSave_Layer_Stack_as_LUT_Cube':
        img = window.label.img
        # get current size
        s = (img.getCurrentImage()).size()
        # build input hald image from identity 3D LUT; channels are in BGR order
        buf = LUT3DIdentity.getHaldImage(s.width(), s.height())
        # add hald to stack, on top of  background
        layer = img.addLayer(None, name='Hald', index=1)
        try:
            # set hald flag
            img.isHald = True
            QImageBuffer(layer.getCurrentImage())[:,:,:3] = buf
            # process hald
            layer.applyToStack()
            processedImg = img.prLayer.inputImg()
            # convert the output hald to a LUT3D object in BGR order
            LUT = LUT3D.HaldImage2LUT3D(processedImg, size=LUT3DIdentity.size)
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
                LUT.writeToTextFile(filenames[0])
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
    l.parentImage.prLayer.update()
    l.parentImage.onImageChanged()  # TODO added 06/09/18 validate
    # record action name for scripting
    l.actionName = name
    # docking the form
    dock = QDockWidget(window)
    dock.setWidget(grWindow)
    dock.setWindowFlags(grWindow.windowFlags())
    dock.setWindowTitle(grWindow.windowTitle())
    #dock.setAttribute(Qt.WA_DeleteOnClose)
    dock.move(900, 40)
    dock.setStyleSheet("QGraphicsView{margin: 10px; border-style: solid; border-width: 1px; border-radius: 1px;}")
    l.view = dock
    # add to docking area
    window.addDockWidget(Qt.RightDockWidgetArea, dock)
    # update layer stack view
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
                #url.setFragment(w.helpId)
        QDesktopServices.openUrl(url)
    elif name == "actionAbout_bLUe":
        w, label = handleTextWindow(parent=window, title='About bLUe', center=False)
        label.setStyleSheet("background-image: url(logo.png); color: white;")
        label.setAlignment(Qt.AlignCenter)
        label.setText(VERSION + "\n" + attributions)
        # center window on screen
        w.setGeometry(QStyle.alignedRect(Qt.LeftToRight, Qt.AlignCenter, w.size(), rootWidget.availableGeometry())) # TODO changed app.desktop() to rootWidget 05/07/18
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

def handleTextWindow(parent=None, title='', center=True):
    """
    Displays a floating modal text window
    @param parent:
    @type parent:
    @param title:
    @type title:
    @param center:
    @type center:
    @return (new window, label)
    @rtype: QMainWindow, QLabel
    """
    w, label = handleNewWindow(parent=parent, title=title, event_handler=False, scroll=True)
    w.setFixedSize(500,500)
    label.setAlignment(Qt.AlignTop)
    w.hide()
    if center:
        # center to the parent current screen
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
    s = s + '&nbsp;&nbsp;&nbsp;&nbsp;Click an item and press Shift+F1 for context help'
    window.Label_status.setText(s)

def screenUpdate(newScreenIndex):
    """
    screenChanged event handler. The image is updated in background
    """
    window.screenChanged.disconnect()
    icc.configure(qscreen=rootWidget.screen(newScreenIndex).windowHandle().screen())  # TODO changed self.dktp to rootWidget 05/07/18 validate
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

###########
# app
###########
if __name__ =='__main__':
    #################
    # multiprocessing
    # freeze_support() must be called at the start of __main__
    # to enable multiprocessing when the executable is frozen.
    # Otherwise, it does nothing.
    #################
    freeze_support()

    # splash screen
    pixmap = QPixmap('logo.png')
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
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
    app.setStyleSheet("QMainWindow, QGraphicsView, QListWidget, QMenu, QTableView {background-color: rgb(200, 200, 200)}\
                       QMenu, QTableView { selection-background-color: blue; selection-color: white;}\
                        QWidget, QTableView, QTableView * {font-size: 9pt} QPushButton {font-size: 6pt}"
                     )

    # Before/After view
    splittedWin = splittedWindow(window)

    # status bar
    window.Label_status = QLabel()
    window.statusBar().addWidget(window.Label_status)
    window.updateStatus = updateStatus
    window.label.updateStatus = updateStatus

    # crop tool
    window.cropTool = cropTool(parent=window.label)

    #whatsThis
    window.cropButton.setWhatsThis("""To crop drag a gray curtain on either side using the 8 small square buttons around the image""")
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
<b>Foreground/Unmask</b><br
  Paint on the active layer to unmask a previously masked region or to select foreground pixels (segmentation layer only)<br>
  Use the 'Brush Size' slider below to choose the size of the tool. 
"""                             )
    window.drawBG.setWhatsThis(
"""<b>Background/Mask</b><br>
  Paint on the active layer to mask a region or to select background pixels (segmentation layer only)<br>
  Use the 'Brush Size' slider below to choose the size of the tool. 
"""                             )
    window.verticalSlider1.setWhatsThis("""Set the diameter of the painting brush""")

    # Before/After views flag
    window.splittedView = False

    window.histView.mode = 'Luminosity'
    window.histView.chanColors = Qt.gray #[Qt.red, Qt.green,Qt.blue]

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
    window.menu_File.triggered.connect(lambda a : menuFile(a.objectName()))
    window.menuLayer.triggered.connect(lambda a : menuLayer(a.objectName()))
    window.menuImage.triggered.connect(lambda a : menuImage(a.objectName()))
    window.menuWindow.triggered.connect(lambda a : menuView(a.objectName()))
    window.menuHelp.triggered.connect(lambda a : menuHelp(a.objectName()))

    #  called by all main form button and slider slots (cf. QtGui1.py)
    window.onWidgetChange = widgetChange

    # load current settings
    #window.readSettings() # TODO validate removing 05/07/18

    set_event_handlers(window.label)
    set_event_handlers(window.label_2, enterAndLeave=False)
    set_event_handlers(window.label_3, enterAndLeave=False)
    # drag and drop event handlers are specific for the main window
    window.label.dropEvent = MethodType(lambda instance, e, wdg=window.label: dropEvent(wdg, wdg.img, e),
                                        window.label.__class__)
    window.label.dragEnterEvent = MethodType(lambda instance, e, wdg=window.label: dragEnterEvent(wdg, wdg.img, e),
                                             window.label.__class__)
    window.label.setAcceptDrops(True)

    img=QImage(200, 200, QImage.Format_ARGB32)
    img.fill(Qt.darkGray)
    defaultImImage = imImage(QImg=img, meta=metadataBag(name='noName'))

    window.label.img = defaultImImage
    window.label_2.img = defaultImImage
    window.label_3.img = defaultImImage

    window.showMaximized()
    splash.finish(window)
    # init EyeDropper cursor
    window.cursor_EyeDropper = QCursor(QPixmap.fromImage(QImage(":/images/resources/Eyedropper-icon.png")))
    # init tool cursor, must be resizable
    curImg = QImage(":/images/resources/cursor_circle.png")
    # turn to white
    curImg.invertPixels()
    window.cursor_Circle_Pixmap = QPixmap.fromImage(curImg)

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

    # init property widget for tableView
    window.propertyWidget.setLayout(window.tableView.propertyLayout)

    # reinit the dockWidgetContents layout to
    # nest it in a QHboxLayout containing a (left) stretch
    tmpV = QVBoxLayout()
    while window.dockWidgetContents.layout().count() != 0:
        w = window.dockWidgetContents.layout().itemAt(0).widget()
        tmpV.addWidget(w)
        if w.objectName() == 'histView':
            tmpV.addSpacing(20)
    tmpH = QHBoxLayout()
    tmpH.addStretch(100)
    tmpH.addLayout(tmpV)
    tmpH.setContentsMargins(0,0,10,0)
    tmpV.setContentsMargins(0,0,10,0)
    # to remove the current layout we reparent it to
    # an unreferenced widget.
    QWidget().setLayout(window.dockWidgetContents.layout())
    # set the new layout
    window.dockWidgetContents.setLayout(tmpH)

    ################################
    # color magement configuration
    # must be done after showing window
    ################################
    window.screenChanged.connect(screenUpdate)
    # screen detection
    c = window.frameGeometry().center()
    scn = rootWidget.screenNumber(c) # TODO changed window.dktp to rootWidget 05/07/18 validate
    window.currentScreenIndex = scn
    icc.configure(qscreen=rootWidget.screen(scn).windowHandle().screen()) # TODO changed window.dktp to rootWidget 05/07/18 validate
    icc.COLOR_MANAGE = icc.HAS_COLOR_MANAGE
    window.actionColor_manage.setEnabled(icc.HAS_COLOR_MANAGE)
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    updateStatus()
    window.label.setWhatsThis(
""" <b>Main Window<br>
Menu File > Open</b> to edit a photo.<br>
<b>Menu Layer > New Adjustment Layer</b> to add an adjustment layer.<br>
<b>Menu View > Library Viewer</b> to browse a folder.<br>
"""
    ) # end of setWhatsThis
    window.label_3.setWhatsThis(
""" <b>Before/After View : After Window</b><br>
Shows the modified image.<br>
<b>Ctrl+Space</b> to cycle through views.<br>
<b>Space</b> to switch back to normal view.<br>
"""
    ) # end of setWhatsThis
    window.label_2.setWhatsThis(
""" <b>Before/After View : Before Window</b><br>
Shows the initial image.<br>
<b>Ctrl+Space</b> to cycle through views.<br>
<b>Space</b> to switch back to normal view.
"""
    ) # end of setWhatsThis
    ###############
    # launch app
    ###############
    sys.exit(app.exec_())