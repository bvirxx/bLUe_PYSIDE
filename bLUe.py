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
import sys
import threading
from itertools import cycle
from os import path, walk
from os.path import basename
from types import MethodType
from cv2 import NORMAL_CLONE
import rawpy
from grabcut import segmentForm
from PySide2.QtCore import Qt, QRect, QEvent, QUrl, QSize, QFileInfo, QRectF, QObject, QPoint, \
    QMimeData, QByteArray
from PySide2.QtGui import QPixmap, QPainter, QCursor, QKeySequence, QBrush, QPen, QDesktopServices, QFont, \
    QPainterPath, QTransform, QIcon, QContextMenuEvent
from PySide2.QtWidgets import QApplication, QMenu, QAction, QFileDialog, QMessageBox, \
    QMainWindow, QLabel, QDockWidget, QSizePolicy, QScrollArea, QSplashScreen, QWidget, \
    QListWidget, QListWidgetItem, QAbstractItemView, QStyle, QToolTip, QHBoxLayout, QVBoxLayout
from QtGui1 import app, window
import exiftool
from graphicsBlendFilter import blendFilterForm
from graphicsNoise import noiseForm
from graphicsRaw import rawForm
from graphicsTransform import transForm
from imgconvert import *
from MarkedImg import imImage, metadata, vImage, QLayer
from graphicsRGBLUT import graphicsForm
from graphicsLUT3D import graphicsForm3DLUT
from colorCube import LUTSIZE, LUT3D, LUT3DIdentity
from colorModels import cmHSP, cmHSB
from colorManagement import icc
from graphicsCoBrSat import CoBrSatForm
from graphicsExp import ExpForm
from graphicsPatch import patchForm, maskForm
from utils import saveChangeDialog, saveDlg, openDlg, cropTool, rotatingTool, IMAGE_FILE_NAME_FILTER, \
    IMAGE_FILE_EXTENSIONS, RAW_FILE_EXTENSIONS, demosaic, dlgWarn, dlgInfo
from graphicsTemp import temperatureForm
from time import sleep
from re import search
import gc
from graphicsFilter import filterForm
from graphicsHspbLUT import graphicsHspbForm
from graphicsLabLUT import graphicsLabForm
from splittedView import splittedWindow


##################
#  Software Attributions
attributions = """
exiftool Copyright © 2013-2016, Phil Harvey
QRangeSlider Copyright (c) 2011-2012, Ryan Galloway
Pillow Copyright © 2010-2018 by Alex Clark and contributors
libraw Copyright (C) 2008-2018 
rawpy Copyright (c) 2014 Maik Riechert
seamlessClone and CLAHE are Opencv3 functions
grabCut is a parallel version of an Opencv3 function
"""
#################

################
#  Version
VERSION = "v.0.1-alpha"
###############

###############
# adjustment form size
axeSize = 200
##############

#########
# init Before/After view
#########
splittedWin = splittedWindow(window)

###############
# Global QPainter for paint event
qp = QPainter()
# stuff for Before/After marks
qp.font = QFont("Arial", 8)
qp.markPath=QPainterPath()
qp.markRect = QRect(0, 0, 50, 20)
qp.markPath.addRoundedRect(qp.markRect, 5, 5)
##############

def paintEvent(widg, e) :
    """
    Paint event handler for widgets displaying a mImage object.
    The widget must have a valid img attribute of type QImage.
    The handler should override the paintEvent method of widg. This can be done
    by subclassing, or by dynamically assigning paintEvent
    to widg.paintEvent (cf. the function set_event_handler
    below).
    Layer qPixmaps (color managed) are painted in stack ascending order,
    each with its own opacity and composition mode.
    @param widg: widget
    @type widg: object with a img attribute of type mImage
    @param e: paint event
    """
    if not hasattr(widg, 'img'):
        raise ValueError("paintEvent: no image attribute")
    mimg = widg.img
    if mimg is None:
        return
    r = mimg.resize_coeff(widg)
    qp.begin(widg)
    qp.setRenderHint(QPainter.SmoothPixmapTransform)
    # fill  background
    qp.fillRect(QRectF(0, 0, widg.width() , widg.height() ), vImage.defaultBgColor)
    # draw layers.
    for layer in mimg.layersStack :
        if layer.visible:
            qp.setOpacity(layer.opacity)
            qp.setCompositionMode(layer.compositionMode)
            # As offsets can be float numbers, we use QRectF instead of QRect
            # r is relative to full resolution image, so we use mimg width and height
            rectF = QRectF(mimg.xOffset, mimg.yOffset, mimg.width()*r, mimg.height()*r)
            if layer.qPixmap is not None:
                px=layer.qPixmap
                qp.drawPixmap(rectF, px, px.rect())
            else:
                currentImage = layer.getCurrentImage()
                qp.drawImage(rectF, currentImage, currentImage.rect())
    # draw selection rectangle for active layer only
    layer = mimg.getActiveLayer()
    rect = layer.rect
    if (layer.visible) and (rect is not None ):
        qp.setPen(QColor(0, 255, 0))
        qp.drawRect(rect.left()*r + mimg.xOffset, rect.top()*r +mimg.yOffset, rect.width()*r, rect.height()*r)
    # draw cropping marks and rulers
    w, h = mimg.width() * r, mimg.height() * r
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

# mouse event handler for image widgets (i.e. with a dynamically set attribute widget.img of type imImage)
# global variables used in mouseEvent
# Recording of mouse coordinates, relative to widget
State = {'drag':False, 'drawing':False , 'tool_rect':False, 'rect_over':False, 'ix':0, 'iy':0, 'ix_begin':0, 'iy_begin':0}
# Recording of mouse state
pressed=False
clicked = True
def mouseEvent(widget, event) :
    """
    Mouse event handler for QLabel object.
    It handles image positioning, zooming, and
    tool actions. It must be called by mousePressed,
    mouseMoved and mouseReleased. This can be done by subclassing
    and overriding, or by dynamically assigning mouseEvent
    to the former three methods (cf. the function set_event_handler
    below).
    NOTE 1. Mouse hover generates mouse move events
    NOTE 2. Due to wheeelEvent, xOffset and yOffset are float
    @param widget:
    @type widget: QLabel object with img attribute
    @param event: mouse event
    @type event:
    """
    if type(event) == QContextMenuEvent:
        return

    global pressed, clicked
    # get image and active layer
    img= widget.img
    layer = img.getActiveLayer()
    r = img.resize_coeff(widget)
    # x, y coordinates (relative to widget)
    x, y = event.x(), event.y()
    # keyboard modifiers
    modifiers = QApplication.keyboardModifiers()
    # mouse press event
    if event.type() == QEvent.MouseButtonPress :
        # Mouse hover generates mouse move events,
        # so, we use a flag pressed to only select non hovering events
        pressed=True
        if event.button() == Qt.LeftButton:
            # no move yet
            clicked=True
        State['ix'], State['iy'] = x, y
        State['ix_begin'], State['iy_begin'] = x, y
        State['x_imagePrecPos'], State['y_imagePrecPos'] = (x - img.xOffset) // r, (y - img.yOffset) // r
    # mouse move event
    elif event.type() == QEvent.MouseMove :
        clicked=False
        if pressed :
            if img.isMouseSelectable:
                # don't draw on a non visible layer
                if window.btnValues['rectangle'] or window.btnValues['drawFG'] or window.btnValues['drawBG']:
                    if not layer.visible:
                        dlgWarn('Select a visible layer for drawing or painting')
                        pressed = False
                        return
                # marquee tool
                if window.btnValues['rectangle']:
                    # rectangle coordinates are relative to image
                    x_img = (min(State['ix_begin'], x) - img.xOffset) // r
                    y_img = (min(State['iy_begin'], y) - img.yOffset) // r
                    w = abs(State['ix_begin'] - x) // r
                    h = abs(State['iy_begin'] - y) // r
                    layer.rect = QRect(x_img, y_img, w, h)
                # drawing tools
                elif window.btnValues['drawFG'] or window.btnValues['drawBG']:
                    if layer.maskIsEnabled:
                        color = vImage.defaultColor_UnMasked if window.btnValues['drawFG'] else vImage.defaultColor_Masked
                        qp.begin(layer.mask)
                        # get pen width
                        w = window.verticalSlider1.value() // (2*r)
                        # mode source : result is source (=pen) pixel color and opacity
                        qp.setCompositionMode(qp.CompositionMode_Source)
                        tmp_x = (x - img.xOffset) // r
                        tmp_y = (y - img.yOffset) // r
                        qp.drawEllipse(tmp_x-w//2, tmp_y-w//2, w, w)
                        qp.setPen(QPen(color, 2*w))
                        qp.drawLine(State['x_imagePrecPos'], State['y_imagePrecPos'], tmp_x, tmp_y)
                        qp.end()
                        State['x_imagePrecPos'], State['y_imagePrecPos'] = tmp_x, tmp_y
                        # update mask stack
                        for l in img.layersStack :
                            l.updatePixmap(maskOnly=True)
                        window.label.repaint()
                # translations
                else:
                    # translate image
                    if modifiers == Qt.NoModifier:
                        img.xOffset+=(x-State['ix'])
                        img.yOffset+=(y-State['iy'])
                        if window.btnValues['Crop_Button']:
                            window.cropTool.drawCropTool(img)
                    # translate active layer only
                    elif modifiers == Qt.ControlModifier:
                        layer.xOffset += (x - State['ix'])
                        layer.yOffset += (y - State['iy'])
                        layer.updatePixmap()
                    # translate cloning virtual layer
                    elif modifiers == Qt.ControlModifier | Qt.AltModifier:
                        if layer.isCloningLayer():
                            layer.xAltOffset += (x - State['ix'])
                            layer.yAltOffset += (y - State['iy'])
                            layer.cloned = False
                            if layer.keepCloned:
                                layer.maskIsEnabled = True
                                layer.maskIsSelected = False
                            layer.applyCloning(seamless=False)
            # need to update before window
            else:
                if modifiers == Qt.NoModifier:
                    img.xOffset += (x - State['ix'])
                    img.yOffset += (y - State['iy'])
                elif modifiers == Qt.ControlModifier:
                    layer.xOffset += (x - State['ix'])
                    layer.yOffset += (y - State['iy'])
                    layer.updatePixmap()
                # cloning layer
                elif modifiers == Qt.ControlModifier | Qt.AltModifier:
                    if layer.isCloningLayer():
                        layer.xAltOffset += (x - State['ix'])
                        layer.yAltOffset += (y - State['iy'])
                        layer.cloned = False
                        if layer.keepCloned:
                            layer.applyCloning(seamless=False)
        #update current coordinates
        State['ix'],State['iy']=x,y
        # Pick color from active layer. Coordinates are relative to the full-sized image
        if window.btnValues['colorPicker']:
            x_img, y_img = (x - img.xOffset) / r, (y - img.yOffset) / r
            x_img, y_img = min(int(x_img), img.width() - 1), min(int(y_img), img.height() - 1)
            color = img.getActivePixel(x_img, y_img)                # TODO modified 12/03/18
            s = ('%s  %s  %s' % (color.red(), color.green(), color.blue()))
            QToolTip.showText(event.globalPos(), s, window, QRect(event.globalPos(), QSize(20,30)))
        if layer.isGeomLayer():
            layer.view.widget().tool.drawRotatingTool()
    #####################
    # mouse release event
    ####################
    elif event.type() == QEvent.MouseButtonRelease :
        pressed=False
        if event.button() == Qt.LeftButton:
            if img.isMouseSelectable:
                # click event
                if clicked:
                    x_img, y_img = (x - img.xOffset) / r, (y - img.yOffset) / r
                    x_img, y_img = min(int(x_img), img.width()-1), min(int(y_img), img.height()-1)
                    # Pick color from active layer. Coordinates are relative to the full-sized image
                    c = img.getActivePixel(x_img, y_img)                # TODO modified 12/03/18
                    red, green, blue = c.red(), c.green(), c.blue()
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
                        #color = bufRaw[y_img, x_img, :]
                        color = [color[i] - layer.parentImage.rawImage.black_level_per_channel[i] for i in range(3)]
                        form =layer.view.widget()
                        if form.sampleMultipliers:
                            row, col = 3*y_img//layer.height(), 3*x_img//layer.width()
                            if form.samples:
                                form.setRawMultipliers(*form.samples[3*row + col], sampling=False)
                        else:
                            form.setRawMultipliers(1/color[0], 1/color[1], 1/color[2], sampling=True)
                # for cloning layer do cloning
                if layer.isCloningLayer():
                    if modifiers == Qt.ControlModifier | Qt.AltModifier:
                        if layer.keepCloned:
                            layer.maskIsEnabled = False
                            layer.maskIsSelected = False
                            layer.applyCloning(seamless=True)
                        layer.updateTableView(window.tableView)
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
    of imImage objects.
    @param widget: widget displaying image
    @param img: imImage object to display
    @param event: mouse wheel event (type QWheelEvent)
    """
    pos = event.pos()
    # delta unit is 1/8 of degree
    # Most mice have a resolution of 15 degrees
    numDegrees = event.delta() / 8
    numSteps = numDegrees / 150.0
    # keyboard modifiers
    modifiers = QApplication.keyboardModifiers()
    layer = img.getActiveLayer()
    if modifiers == Qt.NoModifier:
        img.Zoom_coeff *= (1.0 + numSteps)
        # max Zoom for previews
        if img.Zoom_coeff >2:
            img.Zoom_coeff /= (1.0 + numSteps)
            return
        # correcting image offset to keep unchanged the image point
        # under the cursor : (pos - offset) / resize_coeff should be invariant
        img.xOffset = -pos.x() * numSteps + (1.0+numSteps)*img.xOffset
        img.yOffset = -pos.y() * numSteps + (1.0+numSteps)*img.yOffset
        if window.btnValues['Crop_Button']:
            window.cropTool.drawCropTool(img)
        if layer.isGeomLayer():
            layer.view.widget().tool.drawRotatingTool()
    elif modifiers == Qt.ControlModifier:
        layer.Zoom_coeff *= (1.0 + numSteps)
        layer.updatePixmap()
    elif modifiers == Qt.ControlModifier | Qt.AltModifier:
        if layer.isCloningLayer():
            layer.AltZoom_coeff *= (1.0 + numSteps)
            layer.cloned = False
            layer.applyCloning(seamless=layer.keepCloned)
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

# Event handler for top level buttons,
# assigned to onWidgetChange
def widgetChange(button):
    """
    event handler for top level buttons
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
    @return: image
    @rtype: imImage
    """
    ###########
    # read metadata
    ##########
    try:
        # read metadata from sidecar (.mie) if it exists, otherwise from image file.
        # Create sidecar if it does not exist.
        with exiftool.ExifTool() as e:
            profile, metadata = e.get_metadata(f, createsidecar=createsidecar)
    except ValueError as er:
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
    if colorSpace < 0:
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
    l.execute = lambda l=l, pool=None: l.applyRawPostProcessing()
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
            # window.label.img.onImageChanged() # TODO 23/04/18 validate removing mandatory because applyToStack may do nothing
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
        histView = histImg.histogram(QSize(window.histView.width(), window.histView.height()), chans=range(3), bgColor=Qt.black,
                                     chanColors=window.histView.chanColors, mode=window.histView.mode, addMode='')
        window.histView.Label_Hist.setPixmap(QPixmap.fromImage(histView))
        window.histView.Label_Hist.repaint()
        window.label.img.layersStack[-1].updatePixmap()
        window.label.repaint()

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
    if name in ['actionOpen', 'actionHald_from_file'] :
        filename = openDlg(window)
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
    updateStatus()

# global variable recording diaporama state
isSuspended= False

def playDiaporama(diaporamaGenerator, parent=None):
    """
    Open a new window and Play a diaporama.
    @param diaporamaGenerator: generator for file names
    @type  diaporamaGenerator: iterator object
    @param parent:
    @type parent:
    """
    global isSuspended
    isSuspended = False
    # init diaporama window
    newWin = QMainWindow(parent)
    newWin.setAttribute(Qt.WA_DeleteOnClose)
    newWin.setContextMenuPolicy(Qt.CustomContextMenu)
    newWin.setWindowTitle(parent.tr('Slide show'))
    label = QLabel()
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    label.img = None
    newWin.setCentralWidget(label)
    newWin.showMaximized()
    set_event_handlers(label)
    # Pause shortcut
    actionEsc = QAction('Pause', None)
    actionEsc.setShortcut(QKeySequence(Qt.Key_Escape))
    newWin.addAction(actionEsc)
    # context menu event handler
    def contextMenuHandler(action):
        global isSuspended
        if action.text() == 'Pause':
            isSuspended = True
        elif action.text() == 'Resume':
            newWin.close()
            isSuspended = False
            playDiaporama(diaporamaGenerator, parent=window)
        elif action.text() in ['0', '1','2','3','4','5']:
            with exiftool.ExifTool() as e:
                e.writeXMPTag(name, 'XMP:rating', int(action.text()))
    actionEsc.triggered.connect(lambda name=actionEsc: contextMenuHandler(name))
    # context menu
    def contextMenu(position):
        menu = QMenu()
        action1 = QAction('Pause', None)
        action1.setEnabled(not isSuspended)
        action3 = QAction('Resume', None)
        action3.setEnabled(isSuspended)
        for action in [action1, action3]:
            menu.addAction(action)
            action.triggered.connect(lambda name=action : contextMenuHandler(name))
        subMenuRating = menu.addMenu('Rating')
        for i in range(6):
            action = QAction(str(i), None)
            subMenuRating.addAction(action)
            action.triggered.connect(lambda name=action : contextMenuHandler(name))
        menu.exec_(position)
    newWin.customContextMenuRequested.connect(contextMenu)
    # play diaporama
    while True:
        if isSuspended:
            newWin.setWindowTitle(newWin.windowTitle() + ' Paused')
            break
        try:
            if not newWin.isVisible():
                break
            name = next(diaporamaGenerator)
            # search rating in metadata
            rating = 5 # default
            try:
                with exiftool.ExifTool() as e:
                    rt = e.readXMPTag(name, 'XMP:rating')
                    r = search("\d", rt)
                    if r is not None:
                        rating = int(r.group(0))
            except ValueError:
                rating = 5
            # don't display image with low rating
            if rating < 2:
                app.processEvents()
                continue
            imImg= loadImageFromFile(name, createsidecar=False)
            if label.img is not None:
                imImg.Zoom_coeff = label.img.Zoom_coeff
            coeff = imImg.resize_coeff(label)
            imImg.yOffset -= (imImg.height()*coeff - label.height()) / 2.0
            imImg.xOffset -= (imImg.width()*coeff - label.width()) / 2.0
            app.processEvents()
            if isSuspended:
                newWin.setWindowTitle(newWin.windowTitle() + ' Paused')
                break
            newWin.setWindowTitle(parent.tr('Slide show') + ' ' + name + ' ' + ' '.join(['*']*imImg.meta.rating))
            label.img = imImg
            label.repaint()
            app.processEvents()
            gc.collect()
            sleep(2)
            app.processEvents()
        except StopIteration:
            newWin.close()
            window.diaporamaGenerator = None
            break
        except ValueError:
            continue
        except RuntimeError:
            window.diaporamaGenerator = None
            break
        except:
            window.diaporamaGenerator = None
            raise
        app.processEvents()

class loader(threading.Thread):
    """
    Thread class for batch loading of images
    """
    def __init__(self, gen, wdg):
        super(loader, self).__init__()
        self.fileListGen = gen
        self.wdg = wdg
    def run(self):
        # next() raises a StopIteration exception when the generator ends.
        # If this exception is unhandled by run(), it causes thread termination.
        # If wdg internal C++ object was destroyed by main thread (form closing)
        # a RuntimeError exception is raised and causes thread termination too.
        # Thus, no further synchronization is needed.
        with exiftool.ExifTool() as e:
            while True:
                try:
                    filename = next(self.fileListGen)
                    # get orientation
                    try:
                        # read metadata from sidecar (.mie) if it exists, otherwise from image file.
                        #with exiftool.ExifTool() as e:
                        profile, metadata = e.get_metadata(filename, createsidecar=False)
                    except ValueError:
                        metadata = [{}]
                    orientation = metadata[0].get("EXIF:Orientation", 0)
                    date = metadata[0].get("EXIF:DateTimeOriginal", 'toto')
                    transformation = exiftool.decodeExifOrientation(orientation)
                    # As Qt cannot load imbedded thumbnails, we load and scale the image.
                    pxm = QPixmap.fromImage(QImage(filename).scaled(500,500, Qt.KeepAspectRatio))
                    if not transformation.isIdentity():
                        pxm = pxm.transformed(transformation)
                    item = QListWidgetItem(QIcon(pxm), basename(filename))
                    item.setToolTip(date)
                    item.setData(Qt.UserRole, (filename, transformation))
                    self.wdg.addItem(item)
                # for clean exiting we catch all exceptions and force break
                except:
                    break

def playViewer(fileListGen, dir, parent=None):
    """
    Open a form and display all images from a directory.
    The images are loaded asynchronously by a separate thread.
    @param fileListGen: file name generator
    @type fileListGen: generator object
    @param parent:
    @type parent:
    """
    # init form
    newWin = QMainWindow(parent)
    newWin.setAttribute(Qt.WA_DeleteOnClose)
    newWin.setContextMenuPolicy(Qt.CustomContextMenu)
    newWin.setWindowTitle(parent.tr('Image Viewer ' + dir))
    # init viewer
    wdg = QListWidget(parent=parent)
    wdg.setSelectionMode(QAbstractItemView.ExtendedSelection)
    wdg.setContextMenuPolicy(Qt.CustomContextMenu)
    wdg.label = None
    # handler for action copy_to_clipboard
    def hCopy():
        sel = wdg.selectedItems()
        l = []
        for item in sel:
            # get url from path
            l.append(QUrl.fromLocalFile(item.data(Qt.UserRole)))
        # init clipboard data
        q = QMimeData()
        # set some Windows magic values for copying files from system clipboard : Don't modify
        # 1 : copy; 2 : move
        q.setData("Preferred DropEffect", QByteArray("2"))
        q.setUrls(l)
        QApplication.clipboard().clear()
        QApplication.clipboard().setMimeData(q)
    def hZoom():
        sel = wdg.selectedItems()
        l = []
        # build list of file paths
        for item in sel:
            l.append(item.data(Qt.UserRole)[0])
        if wdg.label is None:
            wdg.label = QLabel(parent=wdg)
            wdg.label.setMaximumSize(500,500)
        # get selected item bounding rect (global coords)
        rect = wdg.visualItemRect(sel[0])
        # move label close to rect while keeping it visible
        point = QPoint(min(rect.left(), wdg.viewport().width() -500), min(rect.top(), wdg.viewport().height() -500))
        wdg.label.move(wdg.mapFromGlobal(point))
        # get correctly oriented image
        img = QImage(l[0]).transformed(item.data(Qt.UserRole)[1])
        wdg.label.setPixmap(QPixmap.fromImage(img.scaled(500,500, Qt.KeepAspectRatio)))
        wdg.label.show()
    def showContextMenu(pos):
        globalPos = wdg.mapToGlobal(pos)
        myMenu = QMenu()
        action_copy = myMenu.addAction("Copy to Clipboard", hCopy)
        action_zoom = myMenu.addAction("Zoom", hZoom)
        myMenu.exec_(globalPos)
    def hChange():
        if wdg.label is not None:
            wdg.label.hide()
    wdg.customContextMenuRequested.connect(showContextMenu)
    wdg.itemSelectionChanged.connect(hChange)
    wdg.setViewMode(QListWidget.IconMode)
    wdg.setIconSize(QSize(150, 150))
    newWin.setCentralWidget(wdg)
    newWin.showMaximized()
    # launch loader instance
    thr = loader(fileListGen, wdg)
    thr.start()

def menuView(name):
    """
    Menu handler
    @param name: action name
    @type name: str
    """
    # toggle before/after mode
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
    elif name == 'actionDiaporama':
        if hasattr(window, 'diaporamaGenerator'):
            if window.diaporamaGenerator is not None:
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
            dlg = QFileDialog(window, "select", lastDir)
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
            """
            def f():
                for filename in diaporamaList:
                    yield filename
            """
            # cycling diaporama. Use f() for one pass diaporama.
            window.diaporamaGenerator = cycle(diaporamaList) # f()
        playDiaporama(window.diaporamaGenerator, parent=window)
    elif name == 'actionViewer':
        # start from parent dir of the last used directory
        lastDir = path.join(str(window.settings.value('paths/dlgdir', '.')), path.pardir)
        dlg = QFileDialog(window, "select", lastDir)
        dlg.setNameFilters(IMAGE_FILE_NAME_FILTER)
        dlg.setFileMode(QFileDialog.Directory)
        viewList = []
        # directory dialog
        if dlg.exec_():
            newDir = dlg.selectedFiles()[0] # dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
            for dirpath, dirnames, filenames in walk(newDir):
                for filename in [f for f in filenames if
                            f.endswith(IMAGE_FILE_EXTENSIONS) or f.endswith(RAW_FILE_EXTENSIONS)]:
                    viewList.append(path.join(dirpath, filename))
        playViewer((filename for filename in viewList), newDir, parent=window)
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
    # snapshot
    elif name in ['action90_CW', 'action90_CCW', 'action180']:
        try:
            angle = 90 if name == 'action90_CW' else -90 if name == 'action90_CCW' else 180
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            tImg = img.bTransformed(QTransform().rotate(angle))
            setDocumentImage(tImg)
            tImg.layersStack[0].applyToStack()
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
    elif name in ['action0', 'action1', 'action2', 'action3', 'action4', 'action5']:
        img.meta.rating = int(name[-1:])
        updateStatus()
        with exiftool.ExifTool() as e:
            e.writeXMPTag(img.meta.filename, 'XMP:rating', img.meta.rating)

def menuLayer(name):
    """
    Menu handler
    @param name: action name
    @type action: str
    """
    # curves
    if name in ['actionBrightness_Contrast', 'actionCurves_HSpB', 'actionCurves_Lab']:
        if name == 'actionBrightness_Contrast':
            layerName = 'Curves R, G, B'
        elif name == 'actionCurves_HSpB':
            layerName = 'Curves H, S, pB'
        elif name == 'actionCurves_Lab':
            layerName = 'Curves L, a, b'
        # add new layer on top of active layer
        l = window.label.img.addAdjustmentLayer(name=layerName)
        if name == 'actionBrightness_Contrast':
            grWindow=graphicsForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        elif name == 'actionCurves_HSpB':
            grWindow = graphicsHspbForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        elif name == 'actionCurves_Lab':
            grWindow = graphicsLabForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # redimensionable window
        grWindow.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        # Curve change event handler
        # called by curve mouse events
        # Apply current LUT
        def f():
            l.applyToStack()
            window.label.img.onImageChanged()
        grWindow.graphicsScene.onUpdateLUT = f
        # wrapper for the right applyXXX method
        if name == 'actionBrightness_Contrast':
            l.execute = lambda l=l, pool=None: l.apply1DLUT(grWindow.graphicsScene.cubicItem.getStackedLUTXY())
        elif name == 'actionCurves_HSpB':
            l.execute = lambda l=l, pool=None: l.applyHSPB1DLUT(grWindow.graphicsScene.cubicItem.getStackedLUTXY(), pool=pool)
        elif name == 'actionCurves_Lab':
            l.execute = lambda l=l, pool=None: l.applyLab1DLUT(grWindow.graphicsScene.cubicItem.getStackedLUTXY())
    # 3D LUT
    elif name in ['action3D_LUT', 'action3D_LUT_HSB']:
        # color model
        ccm = cmHSP if name == 'action3D_LUT' else cmHSB
        layerName = '3D LUT HSpB' if name == 'action3D_LUT' else '3D LUT HSV'
        l = window.label.img.addAdjustmentLayer(name=layerName, role='3DLUT')
        grWindow = graphicsForm3DLUT.getNewWindow(ccm, axeSize=300, targetImage=window.label.img, LUTSize=LUTSIZE, layer=l, parent=window, mainForm=window)
        # LUT change event handler
        def g(options={}):
            #Apply current 3D LUT and repaint window
            #param options: dictionary of options
            l.applyToStack()
            window.label.img.onImageChanged()
        grWindow.graphicsScene.onUpdateLUT = g
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.apply3DLUT(grWindow.graphicsScene.LUT3DArray, options=grWindow.graphicsScene.options, pool=pool)
    # cloning
    elif name == 'actionNew_Cloning_Layer':
        lname = 'Cloning'
        l = window.label.img.addAdjustmentLayer(name=lname, role='CLONING')
        grWindow = patchForm.getNewWindow(targetImage=window.label.img, layer=l, mainForm=window)
        l.execute = lambda l=l, pool=None: l.applyCloning()
        l.maskIsEnabled = True
        l.maskIsSelected = True
        l.resetMask(maskAll=True)
        l.cloned = False
        l.cloningMethod = NORMAL_CLONE
        l.keepCloned = True
    # segmentation
    elif name == 'actionNew_segmentation_layer':
        lname = 'Segmentation'
        l = window.label.img.addSegmentationLayer(name=lname)
        l.maskIsEnabled = True
        l.maskIsSelected = True
        l.mask.fill(vImage.defaultColor_Invalid)
        #l.isClipping = True
        grWindow = segmentForm.getNewWindow(targetImage=window.label.img, layer=l, mainForm=window)
        l.execute = lambda l=l, pool=None: l.applyGrabcut(nbIter=grWindow.nbIter)
        # mask was modified
        l.updatePixmap()
    # loads an image
    elif name == 'actionNew_Image_Layer':
        filename = openDlg(window)
        img = window.label.img
        imgNew = QImage(filename)
        if imgNew.isNull():
            dlgWarn("Cannot load %s: " % filename)
            return
        size = img.size()
        sizeNew = imgNew.size()
        if sizeNew != size:
            imgNew = imgNew.scaled(size)
            dlgInfo("Image will be resized")
        l = QLayer(QImg=imgNew, parentImage=window.label.img)
        l.isClipping = False
        img.addLayer(l, name=path.basename(filename))
        l.updatePixmap()
        l.actioname = name
        # update layer stack view
        window.tableView.setLayers(window.label.img)
        return
    # Temperature
    elif name == 'actionColor_Temperature':
        lname = 'Color Temperature'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = temperatureForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.applyTemperature()
    elif name == 'actionContrast_Correction':
        lname = 'Cont. Sat. Br.'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = CoBrSatForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # clipLimit change event handler
        def h(lay, clipLimit):
            lay.clipLimit = clipLimit
            lay.applyToStack()
            window.label.img.onImageChanged()
        grWindow.onUpdateContrast = h
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.applyContrast()
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
        l.execute = lambda l=l,  pool=None: l.applyExposure(l.clipLimit, grWindow.options)
    elif name == 'actionGeom_Transformation':
        lname = 'Geometry'
        l = window.label.img.addAdjustmentLayer(name=lname, role='GEOMETRY')
        grWindow = transForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        l.geoTrans = QTransform()
        tool = rotatingTool(parent=window.label, layer=l, form=grWindow)
        l.execute = lambda l=l, pool=None: l.applyTransForm(l.geoTrans, grWindow.options)
    elif name == 'actionFilter':
        lname = 'Filter'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = filterForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.applyFilter2D()
        # l.execute = lambda: l.applyLab1DLUT(grWindow.graphicsScene.cubicItem.getStackedLUTXY())
    elif name == 'actionGradual_Filter':
        lname = 'Blend Filter'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = blendFilterForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.applyBlendFilter()
    elif name == 'actionNoise_Reduction':
        lname='Noise Reduction'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = noiseForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.applyNoiseReduction()
    elif name == 'actionSave_Layer_Stack':
        lastDir = str(window.settings.value('paths/dlgdir', '.'))
        dlg = QFileDialog(window, "select", lastDir)
        dlg.setNameFilter('*.sba')
        dlg.setDefaultSuffix('sba')
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            newDir = dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
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
    # adjustment layer from .cube file
    elif name == 'actionLoad_3D_LUT' :
        lastDir = window.settings.value('paths/dlgdir', '.')
        dlg = QFileDialog(window, "select", lastDir)
        dlg.setNameFilter('*.cube')
        dlg.setDefaultSuffix('cube')
        if dlg.exec_():
            newDir = dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
            filenames = dlg.selectedFiles()
            name = filenames[0]
            """
            if name[-4:] == '.png':
                LUT3DArray = LUT3D.readFromHaldFile(name)
            else :
            """
            try:
                LUT3DArray, size = LUT3D.readFromTextFile(name)
            except (ValueError, IOError) as e:
                dlgWarn('Unable to load 3D LUT', info=str(e))
                return
            lname = path.basename(name)
            l = window.label.img.addAdjustmentLayer(name=lname)
            l.execute = lambda l=l, pool=None: l.apply3DLUT(LUT3DArray, {'use selection': False})
            window.tableView.setLayers(window.label.img)
            l.applyToStack()
            #l.apply3DLUT(LUT3DArray, {'use selection':False})
        return
    elif name == 'actionSave_Layer_Stack_as_LUT_Cube':
        doc = window.label.img
        buf = LUT3DIdentity.getHaldImage(doc.width(), doc.height())
        try:
            layer = doc.addLayer(None, name='Hald', index=1)
            #layer = doc.layersStack[0]
            docBuf = QImageBuffer(layer)
            docBuf[:, :, :3] = buf
            layer.applyToStack()
            window.label.repaint()
            img = doc.mergeVisibleLayers()
            # convert image to LUT3D object
            LUT = LUT3D.HaldImage2LUT3D(img, size=33)
            # open file and save
            lastDir = str(window.settings.value('paths/dlgdir', '.'))
            dlg = QFileDialog(window, "select", lastDir)
            dlg.setNameFilter('*.cube')
            dlg.setDefaultSuffix('cube')
            if dlg.exec_():
                newDir = dlg.directory().absolutePath()
                window.settings.setValue('paths/dlgdir', newDir)
                filenames = dlg.selectedFiles()
                newDir = dlg.directory().absolutePath()
                window.settings.setValue('paths/dlgdir', newDir)
                LUT.writeToTextFile(filenames[0])
            return
        finally:
            # restore doc
            doc.removeLayer(1)
            doc.layersStack[0].applyToStack()
    else:
        return
    # record action name for scripting
    l.actionName = name
    # dock the form
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
        label.setText(VERSION + "\n"+attributions)
        # center window on screen
        w.setGeometry(QStyle.alignedRect(Qt.LeftToRight, Qt.AlignCenter, w.size(), app.desktop().availableGeometry()))
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
    window.Label_status.setText(s)

def screenUpdate(newScreenIndex):
    """
    screenChanged event handler. The image is updated in background
    """
    window.screenChanged.disconnect()
    icc.configure(qscreen=window.dktp.screen(newScreenIndex).windowHandle().screen())
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
    # add dynamic attributes dktp and currentScreenIndex, used for screen management
    window.dktp = app.desktop()
    window.currentScreenIndex = 0
    # splash screen
    pixmap = QPixmap('logo.png')
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    splash.showMessage("Loading .", color=Qt.white, alignment=Qt.AlignCenter)
    app.processEvents()
    sleep(1)
    splash.showMessage("Loading ...", color=Qt.white, alignment=Qt.AlignCenter)
    splash.finish(window)
    app.processEvents()
    # title
    window.setWindowTitle('bLUe')
    # style sheet
    app.setStyleSheet("QMainWindow, QGraphicsView, QListWidget, QMenu, QTableView {background-color: rgb(200, 200, 200)}\
                       QMenu, QTableView { selection-background-color: blue; selection-color: white;}\
                        QWidget, QTableView, QTableView * {font-size: 9pt}"
                     )
    # status bar
    window.Label_status = QLabel()
    window.statusBar().addWidget(window.Label_status)
    window.updateStatus = updateStatus
    window.label.updateStatus = updateStatus

    # crop buttons
    window.cropTool = cropTool(parent=window.label)

    # Before/After views flag
    window.splittedView = False

    window.histView.mode = 'Luminosity'
    window.histView.chanColors = Qt.gray #[Qt.red, Qt.green,Qt.blue]

    # close event handler
    window.onCloseEvent = canClose

    # mouse hover events
    window.label.setMouseTracking(True)

    # menu event handlers
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

    #  init convenience GUI hooks
    window.onWidgetChange = widgetChange

    # load current settings
    window.readSettings()
    # window._recentFiles = window.settings.value('paths/recent', []) # TODO validate removing 10/04/18

    set_event_handlers(window.label)
    set_event_handlers(window.label_2, enterAndLeave=False)
    set_event_handlers(window.label_3, enterAndLeave=False)

    img=QImage(200, 200, QImage.Format_ARGB32)
    img.fill(Qt.darkGray)
    defaultImImage = imImage(QImg=img, meta=metadata(name='noName'))

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
    id = window.dktp.screenNumber(c)
    window.currentScreenIndex = id
    icc.configure(qscreen=window.dktp.screen(id).windowHandle().screen())
    icc.COLOR_MANAGE = icc.HAS_COLOR_MANAGE
    window.actionColor_manage.setEnabled(icc.HAS_COLOR_MANAGE)
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    updateStatus()

    ###################
    # test numpy dll loading
    #################
    import numpy as np
    from time import time
    t = time()
    dummy = np.dot([[i] for i in range(1000)], [range(1000)])
    t = time() - t
    if t > 0.01:
        print('dot product time %.5f' % t)
    # launch app
    sys.exit(app.exec_())