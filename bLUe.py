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

The QtHelp module uses the CLucene indexing library
Copyright (C) 2003-2006 Ben van Klinken and the CLucene Team

Changes are Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
"""
import sys
from os import path, walk
from os.path import isfile
from types import MethodType
from grabcut import segmentForm
from PySide2.QtCore import Qt, QRect, QEvent, QDir, QUrl, QPoint, QSize, QFile, QFileInfo, QRectF
from PySide2.QtGui import QPixmap, QColor, QPainter, QCursor, QKeySequence, QBrush, QPen, QDesktopServices, QFont, \
    QPainterPath, QTransform
from PySide2.QtWidgets import QApplication, QMenu, QAction, QFileDialog, QMessageBox, \
    QMainWindow, QLabel, QDockWidget, QSizePolicy, QScrollArea, QSplashScreen, QVBoxLayout, QWidget, QPushButton
from QtGui1 import app, window
import exiftool
from imgconvert import *
from MarkedImg import imImage, metadata, vImage

from graphicsRGBLUT import graphicsForm
from graphicsLUT3D import graphicsForm3DLUT
from colorCube import LUTSIZE, LUT3D, LUT3DIdentity
from colorModels import cmHSP, cmHSB
import icc
from colorConv import sRGBWP
from graphicsCLAHE import CLAHEForm
from graphicsExp import ExpForm
from graphicsPatch import patchForm
from utils import savingDialog, saveChangeDialog, save, openDlg

from graphicsTemp import temperatureForm
from time import sleep

from re import search
import weakref

import gc

from graphicsFilter import filterForm
from graphicsHspbLUT import graphicsHspbForm
from graphicsLabLUT import graphicsLabForm

from splittedView import splittedWindow


##################
#  Refresh views after image processing
#################

"""
def updateDocView():
    window.label.repaint()
    window.label_3.repaint()
"""
#########
# init Before/After view
#########

splittedWin = splittedWindow(window)

###############
# paintEvent global QPainter
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
    # fill  background
    qp.fillRect(QRectF(0, 0, widg.width() , widg.height() ), vImage.defaultBgColor)
    # draw layers.
    for layer in mimg.layersStack :
        if layer.visible:
            qp.setOpacity(layer.opacity)
            qp.setCompositionMode(layer.compositionMode)
            # As offsets can be float numbers, we use QRectF instead of QRect
            rectF = QRectF(mimg.xOffset, mimg.yOffset, mimg.width()*r, mimg.height()*r)
            if layer.qPixmap is not None:
                qp.drawPixmap(rectF, layer.qPixmap, layer.qPixmap.rect())
            else:
                currentImage = layer.getCurrentImage()
                qp.drawImage(rectF, currentImage, currentImage.rect())
    # draw selection rectangle for active layer only
    layer = mimg.getActiveLayer()
    if (layer.visible) and (layer.rect is not None ):
        qp.setPen(QColor(0, 255, 0))
        rect = mimg.getActiveLayer().rect
        qp.drawRect(rect.left()*r + mimg.xOffset, rect.top()*r +mimg.yOffset, rect.width()*r, rect.height()*r)
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
                        msg = QMessageBox()
                        msg.setWindowTitle('Warning')
                        msg.setIcon(QMessageBox.Warning)
                        msg.setText('Select a visible layer for drawing or painting')
                        msg.exec_()
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
                        if layer.isSegmentLayer():
                            color = QColor(0, 255, 0, 128) if window.btnValues['drawFG'] else QColor(0, 0, 255, 128)
                        else:
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
                            layer.thumb.cloned = False
                            layer.cloned = False
                            layer.applyCloning(seamless=False)
            # needed to update before window
            else:
                if modifiers == Qt.NoModifier:
                    img.xOffset += (x - State['ix'])
                    img.yOffset += (y - State['iy'])
                elif modifiers == Qt.ControlModifier:
                    layer.xOffset += (x - State['ix'])
                    layer.yOffset += (y - State['iy'])
                    layer.updatePixmap()
                elif modifiers == Qt.ControlModifier | Qt.AltModifier:
                    if layer.isCloningLayer():
                        layer.xAltOffset += (x - State['ix'])
                        layer.yAltOffset += (y - State['iy'])
                        layer.cloned = False
                        layer.applyCloning()
                        layer.updatePixmap()
        #update current coordinates
        State['ix'],State['iy']=x,y
    # mouse release event
    elif event.type() == QEvent.MouseButtonRelease :
        pressed=False
        if event.button() == Qt.LeftButton:
            if img.isMouseSelectable:
                # click event
                if clicked:
                    # Picking color from active layer. Coordinates are relative to full-sized image
                    c = img.getActivePixel((State['ix']  -  img.xOffset) / r, (State['iy'] - img.yOffset) / r)
                    red, green, blue = c.red(), c.green(), c.blue()
                    # select grid node for 3DLUT form
                    if layer.is3DLUTLayer():
                            layer.view.widget().selectGridNode(red, green, blue)
                if window.btnValues['rectangle']:
                    """
                    layer.rect = QRect((min(State['ix_begin'], x) - img.xOffset) // r,
                                       (min(State['iy_begin'], y) - img.yOffset) // r,
                                       abs(State['ix_begin'] - x) // r, abs(State['iy_begin'] - y) // r)
                    """
                    # for cloning layer init mask from rectangle
                    if layer.isCloningLayer():
                        layer.mask.fill(vImage.defaultColor_Masked)
                        """
                        qptemp=QPainter(layer.mask)
                        qptemp.setBrush(QBrush(vImage.defaultColor_UnMasked))
                        qptemp.drawRect(layer.rect)
                        qptemp.end()
                        """
                        layer.updatePixmap(maskOnly=True)
                """
                # do shifts
                else:
                    if modifiers == Qt.NoModifier:
                        img.xOffset += (x - State['ix'])
                        img.yOffset += (y - State['iy'])
                    elif modifiers == Qt.ControlModifier:
                        layer.xOffset += (x - State['ix'])
                        layer.yOffset += (y - State['iy'])
                    elif modifiers == Qt.ControlModifier | Qt.AltModifier:
                        if layer.isCloningLayer():
                            layer.xAltOffset += (x - State['ix'])
                            layer.yAltOffset += (y - State['iy'])
                            layer.cloned = False
                            layer.applyCloning()
                            layer.updatePixmap()
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
        # correcting image offset to keep unchanged the image point
        # under the cursor : (pos - offset) / resize_coeff should be invariant
        img.xOffset = -pos.x() * numSteps + (1.0+numSteps)*img.xOffset
        img.yOffset = -pos.y() * numSteps + (1.0+numSteps)*img.yOffset
    elif modifiers == Qt.ControlModifier:
        layer.Zoom_coeff *= (1.0 + numSteps)
        layer.updatePixmap()
    elif modifiers == Qt.ControlModifier | Qt.AltModifier:
        layer.AltZoom_coeff *= (1.0 + numSteps)
        layer.thumb.cloned = False
        layer.cloned = False
        layer.applyCloning(seamless=False)
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
    Pythonic way for redefining event handlers, without
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

# button change event handler
def widgetChange(button):
    """
    event handler for top level buttons
    @param button:
    @type button: QWidget
    """
    wdgName = button.accessibleName()
    if wdgName == "Fit_Screen" :
        window.label.img.fit_window(window.label)
        window.label.repaint()

def contextMenu(pos, widget):
    """
    Copntext menu for image QLabel
    @param pos:
    @param widget:
    @return:
    """
    pass
    """
    qmenu = QMenu("Context menu")
    if not hasattr(widget, 'img'):
        return

    action1 = QAction('test', qmenu, checkable=True)
    action1.setShortcut(QKeySequence("Ctrl+ "))
    action1.setShortcutContext(Qt.ApplicationShortcut)
    qmenu.addAction(action1)
    action1.triggered[bool].connect(lambda b, widget=widget : print('toto'))
    action1.setChecked(True)

    if widget is window.label_2:
        print('2')
    elif widget is window.label_3:
        print('3')
    qmenu.exec_(QCursor.pos())
    """

def loadImageFromFile(f):
    """
    loads metadata and image from file.
    metadata is a list of dicts with len(metadata) >=1.
    metadata[0] contains at least 'SourceFile' : path.
    profile is a string containing the profile binary data.
    Currently, we do not use these data : standard profiles
    are loaded from disk, non standard profiles are ignored.
    @param f: path to file
    @type f: str
    @return: image
    @rtype: imImage
    """
    # get metadata
    try:
        with exiftool.ExifTool() as e:
            profile, metadata = e.get_metadata(f)
    except ValueError as er:
        # Default metadata and profile
        metadata = [{'SourceFile': f}]
        profile = ''
    # trying to get color space info : 1 = sRGB
    colorSpace = metadata[0].get("EXIF:ColorSpace", -1)
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
    # load image file
    name = path.basename(f)
    img = imImage(filename=f, colorSpace=colorSpace, orientation=transformation, rawMetadata=metadata, profile=profile, name=name, rating=rating)
    window.settings.setValue('paths/dlgdir', QFileInfo(f).absoluteDir().path())
    img.initThumb()
    if img.format() < 4:
        msg = QMessageBox()
        msg.setText("Cannot edit indexed formats\nConvert image to a non indexed mode first")
        msg.exec_()
        return None
    if colorSpace < 0:
        msg = QMessageBox()
        msg.setText("Color profile missing\nAssigning sRGB profile")
        msg.exec_()
        img.meta.colorSpace = 1
        img.updatePixmap()
    return img

def addBasicAdjustmentLayers():
    menuLayer('actionColor_Temperature')
    menuLayer('actionExposure_Correction')
    menuLayer('actionContrast_Correction')
    window.tableView.select(2, 1)

def openFile(f):
    """
    Top level function for file opening, called by File Menu actions
    @param f: file name
    @type f: str
    """
    closeFile()
    # load file
    try :
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        img = loadImageFromFile(f)
        # display image
        if img is not None:
            setDocumentImage(img)
            # add starting adjusgtment layers
            addBasicAdjustmentLayers()
            # switch to preview mode and process stack
            window.tableView.previewOptionBox.setChecked(True)
            window.tableView.previewOptionBox.stateChanged.emit(Qt.Checked)
            # updates
            updateStatus()
            window.label.img.onImageChanged()
    except ValueError as e:
        msg = QMessageBox()
        msg.setWindowTitle('Warning')
        msg.setIcon(QMessageBox.Warning)
        msg.setText(str(e))
        msg.exec_()
    finally:
        QApplication.restoreOverrideCursor()
        QApplication.processEvents()

def closeFile():
    if window.label.img.isModified:
        ret = saveChangeDialog(window.label.img)
        if ret == QMessageBox.Yes:
            save(window.label.img, window)
        elif ret == QMessageBox.Cancel:
            return
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

def setDocumentImage(img):
    """
    Inits GUI to display the image
    @param img: image
    @type img: imImage
    """
    window.label.img = img
    # init histogram
    window.histView.targetImage = window.label.img
    # image changed event handler
    def f():
        # refresh windows (use repaint for faster update)
        window.label.repaint()
        window.label_3.repaint()
        # refresh histogram window
        if window.histView.listWidget1.items['Original Image'].checkState() is Qt.Checked:
            histImg = vImage(QImg=window.label.img.getCurrentImage()) # must be vImage : histogram method needed
        else:
            histImg = window.label.img.layersStack[-1].getCurrentMaskedImage() # vImage(QImg=window.label.img.layersStack[-1].getCurrentMaskedImage())#mergeVisibleLayers())
        if window.histView.listWidget2.items['Color Chans'].checkState() is Qt.Checked:
            window.histView.mode = 'RGB'
            window.histView.chanColors = [Qt.red, Qt.green, Qt.blue]
        else:
            window.histView.mode = 'Luminosity'
            window.histView.chanColors = [Qt.gray]
        histView = histImg.histogram(QSize(window.histView.width(), window.histView.height()), chans=range(3), bgColor=Qt.black,
                                     chanColors=window.histView.chanColors, mode=window.histView.mode, addMode='Luminosity')
        window.histView.Label_Hist.setPixmap(QPixmap.fromImage(histView))
        window.histView.Label_Hist.repaint()

    window.label.img.onImageChanged = f
    # before image. Stack is not copied
    window.label_2.img = imImage(QImg=img, meta=img.meta)
    # after image
    window.label_3.img = img
    # no mouse drawing or painting
    window.label_2.img.isMouseSelectable = False
    # init layer view
    window.tableView.setLayers(window.label.img)
    window.label.update()
    window.label_2.update()
    window.label_3.update()
    # used by graphicsForm3DLUT.onReset
    window.label.img.window = window.label
    window.label_2.img.window = window.label_2
    window.label.img.setModified(True)

def updateMenuOpenRecent():
    window.menuOpen_recent.clear()
    for f in window._recentFiles :
        window.menuOpen_recent.addAction(f, lambda x=f: openFile(x))

def updateEnabledActions():
    window.actionSave.setEnabled(window.label.img.isModified)
    window.actionSave_Hald_Cube.setEnabled(window.label.img.isHald)

def menuFile(name):
    """
    Menu handler
    @param name: action name
    @type name: str
    """
    window._recentFiles = window.settings.value('paths/recent', [])
    # update menu and actions
    #updateMenuOpenRecent()
    # load image from file
    if name in ['actionOpen', 'actionHald_from_file'] :
        filename = openDlg(window)
        if filename is not None:
            openFile(filename)
    # saving dialog
    elif name == 'actionSave':
        if window.label.img.useThumb:
            msg = QMessageBox()
            msg.setWindowTitle('Warning')
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Uncheck Preview mode before saving")
            msg.exec_()
        else:
            try:
                filename = save(window.label.img, window)
                msg = QMessageBox()
                msg.setWindowTitle('Information')
                msg.setIcon(QMessageBox.Information)
                msg.setText("%s written" % filename)
                msg.exec_()
            except ValueError as e:
                msg = QMessageBox()
                msg.setWindowTitle('Warning')
                msg.setIcon(QMessageBox.Warning)
                msg.setText(str(e))
                msg.exec_()
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
        lastDir = window.settings.value('paths/dlgdir', '.')
        dlg = QFileDialog(window, "select", lastDir)
        dlg.setNameFilter('*.cube')
        dlg.setDefaultSuffix('cube')
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            newDir = dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
            LUT.writeToTextFile(filenames[0])
    updateStatus()

isSuspended= False
def playDiaporama(diaporamaGenerator, parent=None):
    """
    Plays diaporama
    @param diaporamaGenerator: generator of file names
    @type  diaporamaList: list of str
    @param parent: 
    @return: 
    """
    newwindow = QMainWindow(parent)
    newwindow.setAttribute(Qt.WA_DeleteOnClose)
    newwindow.setContextMenuPolicy(Qt.CustomContextMenu)
    newwindow.setWindowTitle(parent.tr('Slide show'))
    global isSuspended
    isSuspended = False
    # Pause shortcut
    actionEsc = QAction('Pause', None)
    actionEsc.setShortcut(QKeySequence(Qt.Key_Escape))
    newwindow.addAction(actionEsc)
    # context menu event handler
    def h(action):
        global isSuspended
        if action.text() == 'Pause':
            isSuspended = True
        elif action.text() == 'Resume':
            newwindow.close()
            isSuspended = False
            playDiaporama(diaporamaGenerator, parent=window)
        elif action.text() in ['0', '1','2','3','4','5']:
            with exiftool.ExifTool() as e:
                e.writeXMPTag(name, 'XMP:rating', int(action.text()))
    actionEsc.triggered.connect(lambda name=actionEsc: h(name))
    # context menu
    def contextMenu(position):
        menu = QMenu()
        action1 = QAction('Pause', None)
        action1.setEnabled(not isSuspended)
        action3 = QAction('Resume', None)
        action3.setEnabled(isSuspended)
        for action in [action1, action3]:
            menu.addAction(action)
            action.triggered.connect(lambda name=action : h(name))
        subMenuRating = menu.addMenu('Rating')
        for i in range(6):
            action = QAction(str(i), None)
            subMenuRating.addAction(action)
            action.triggered.connect(lambda name=action : h(name))
        menu.exec_(position)

    newwindow.customContextMenuRequested.connect(contextMenu)

    label = QLabel()
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    label.img = None
    newwindow.setCentralWidget(label)
    newwindow.showMaximized()
    set_event_handlers(label)
    while True:
        if isSuspended:
            newwindow.setWindowTitle(newwindow.windowTitle() + ' Paused')
            break
        try:
            # test if window was closed :
            # close = hide();delete_later()
            if not newwindow.isVisible():
                break
            name = diaporamaGenerator.next()
            try:
                with exiftool.ExifTool() as e:
                    rating = e.readXMPTag(name, 'XMP:rating')
                    r = search("\d", rating)
                    if r:
                        rating = int(r.group(0))
            except ValueError as er:
                # No metadata found
                rating = 5
            if rating < 2:
                app.processEvents()
                #print 'skip'
                continue
            imImg= loadImageFromFile(name)
            if label.img is not None:
                imImg.Zoom_coeff = label.img.Zoom_coeff
            coeff = imImg.resize_coeff(label)
            imImg.yOffset -= (imImg.height()*coeff - label.height()) / 2.0
            imImg.xOffset -= (imImg.width()*coeff - label.width()) / 2.0
            app.processEvents()
            if isSuspended:
                newwindow.setWindowTitle(newwindow.windowTitle() + ' Paused')
                break
            newwindow.setWindowTitle(parent.tr('Slide show') + ' ' + name + ' ' + ' '.join(['*']*imImg.meta.rating))
            label.img = imImg
            label.repaint()
            app.processEvents()
            gc.collect()
            sleep(2)
            app.processEvents()
        except StopIteration:
            newwindow.close()
            window.diaporamaGenerator = None
            break
        except ValueError as ev:
            continue
        except RuntimeError as er:
            #newwindow.close()
            window.diaporamaGenerator = None
            break
        except:
            #print "Unexpected error:", sys.exc_info()[0]
            #newwindow.close()
            window.diaporamaGenerator = None
            raise
        app.processEvents()

def menuView(name):
    """
    Menu handler
    @param name: action name
    @type name: str
    """
    # togle before/after mode
    if name == 'actionShow_hide_right_window_3' :
        if window.splitter.isHidden() :
            splittedWin.setSplittedView()
            window.viewState = 'Before/After'
        else:
            window.splitter.hide()
            window.label.show()
            window.splittedView = False
            window.viewState = 'After'
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
            lastDir = window.settings.value('paths/dlgdir', '.')
            dlg = QFileDialog(window, "select", lastDir)
            dlg.setNameFilters( ['Image Files (*.jpg *.png)'])
            dlg.setFileMode(QFileDialog.Directory)
            diaporamaList = []
            if dlg.exec_():
                #filenames = dlg.selectedFiles()
                newDir = dlg.directory().absolutePath()
                window.settings.setValue('paths/dlgdir', newDir)
                for dirpath, dirnames, filenames in walk(newDir):
                    for filename in [f for f in filenames if (f.endswith(".jpg") or f.endswith(".png"))]:
                        diaporamaList.append(path.join(dirpath, filename))
            def f():
                for filename in diaporamaList:
                    yield filename
            window.diaporamaGenerator = f()
        playDiaporama(window.diaporamaGenerator, parent=window)
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
        # working profile
        if img.colorTransformation is not None:
            workingProfileInfo = img.colorTransformation.fromProfile.info
        else:
            workingProfileInfo = 'None'
        s = s + "\n\nWorking Profile : %s" % workingProfileInfo
        # embedded profile
        if len(img.meta.profile) > 0:
            s = s +"\n\nEmbedded profile found, length %d" % len(img.meta.profile)
        s = s + "\nRating %s" % ''.join(['*']*img.meta.rating)
        # raw meta data
        l = img.meta.rawMetadata
        s = s + "\n\nMETADATA :\n"
        for d in l:
            s = s + '\n'.join('%s : %s' % (k,v) for k, v in d.items()) # python 3 iteritems -> items
        w, label = handleTextWindow(parent=window, title='Image info')
        label.setWordWrap(True)
        label.setText(s)
    elif name == 'actionColor_manage':
        icc.COLOR_MANAGE = window.actionColor_manage.isChecked()
        img.updatePixmap()
        window.label_2.img.updatePixmap()
    elif name == 'actionWorking_profile':
        w, label = handleTextWindow(parent=window, title='profile info')
        s = 'Working profile\n' + icc.workingProfile.info
        s = s + '-------------\n' + 'Monitor profile\n'+ icc.monitorProfile.info
        s = s + '\nNote : The working profile is the color profile currently associated\n with the opened image.'
        s = s + '\nThe monitor profile is associated with your monitor'
        s = s + '\nBoth profles are used in conjunction to display exact colors'
        s = s + '\n\nIf the monitor profile listed above is not the right profile\nfor your monitor, check the system settings for color management'
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
    axeSize = 200
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

        # wrapper for the right apply method
        if name == 'actionBrightness_Contrast':
            l.execute = lambda pool=None: l.apply1DLUT(grWindow.graphicsScene.cubicItem.getStackedLUTXY())
        elif name == 'actionCurves_HSpB':
            l.execute = lambda  pool=None: l.applyHSPB1DLUT(grWindow.graphicsScene.cubicItem.getStackedLUTXY(), pool=pool)
        elif name == 'actionCurves_Lab':
            l.execute = lambda l=l, pool=None: l.applyLab1DLUT(grWindow.graphicsScene.cubicItem.getStackedLUTXY())
    # 3D LUT
    elif name in ['action3D_LUT', 'action3D_LUT_HSB']:
        # color model
        ccm = cmHSP if name == 'action3D_LUT' else cmHSB
        layerName = '3D LUT HSpB' if name == 'action3D_LUT' else '3D LUT HSB'
        # add new layer on top of active layer
        l = window.label.img.addAdjustmentLayer(name=layerName)
        #window.tableView.setLayers(window.label.img)
        grWindow = graphicsForm3DLUT.getNewWindow(ccm, size=300, targetImage=window.label.img, LUTSize=LUTSIZE, layer=l, parent=window, mainForm=window)
        # LUT change event handler
        def g(options={}):
            """
            Apply current 3D LUT and repaint window
            @param options: dictionary of options
            """
            #l.options = options
            l.applyToStack()
            window.label.img.onImageChanged()
            #updateDocView()
            #l.apply3DLUT(grWindow.graphicsScene.LUT3D, options=options)
            #window.label.repaint()
        grWindow.graphicsScene.onUpdateLUT = g
        # wrapper for the right apply method
        #l.execute = lambda : l.apply3DLUT(grWindow.graphicsScene.LUT3DArray, options=l.options)
        l.execute = lambda l=l, pool=None: l.apply3DLUT(grWindow.graphicsScene.LUT3DArray, options=grWindow.graphicsScene.options, pool=pool)
    # cloning
    elif name == 'actionNew_Cloning_Layer':
        lname = 'Cloning'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = patchForm.getNewWindow(targetImage=window.label.img, layer=l, mainForm=window)
        l.execute = lambda l=l, pool=None: l.applyCloning()
        l.maskIsEnabled = True
        l.maskIsSelected = True
        l.resetMask(maskAll=True)
        l.cloned = False
        l.thumb.cloned = False
    # segmentation grabcut
    elif name == 'actionNew_segmentation_layer':
        lname = 'Segmentation'
        l = window.label.img.addSegmentationLayer(name=lname)
        l.isClipping = True
        grWindow = segmentForm.getNewWindow(targetImage=window.label.img, layer=l, mainForm=window)
        l.execute = lambda l=l, pool=None: l.applyGrabcut(nbIter=grWindow.nbIter)
    # Temperature
    elif name == 'actionColor_Temperature':
        lname = 'Color Temperature'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = temperatureForm.getNewWindow(size=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # default temperature
        l.temperature = sRGBWP
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.applyTemperature(l.temperature, grWindow.options)
    elif name == 'actionContrast_Correction':
        lname = 'Contrast'
        l = window.label.img.addAdjustmentLayer(name=lname)
        l.clipLimit = CLAHEForm.defaultClipLimit
        grWindow = CLAHEForm.getNewWindow(size=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # clipLimit change event handler
        def h(lay, clipLimit):
            lay.clipLimit = clipLimit
            lay.applyToStack()
            window.label.img.onImageChanged()
        grWindow.onUpdateContrast = h
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.applyCLAHE(l.clipLimit, grWindow.options)
    elif name == 'actionExposure_Correction':
        lname = 'Exposure'
        l = window.label.img.addAdjustmentLayer(name=lname)
        l.clipLimit = ExpForm.defaultExpCorrection
        grWindow = ExpForm.getNewWindow(size=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # clipLimit change event handler
        def h(lay, clipLimit):
            lay.clipLimit = clipLimit
            lay.applyToStack()
            #updateDocView()
            window.label.img.onImageChanged()
            #window.label.repaint()
        grWindow.onUpdateExposure = h
        # wrapper for the right apply method
        l.execute = lambda l=l,  pool=None: l.applyExposure(l.clipLimit, grWindow.options)
    elif name == 'actionFilter':
        lname = 'Filter'
        l = window.label.img.addAdjustmentLayer(name=lname)
        grWindow = filterForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=l, parent=window, mainForm=window)
        # temperature change event handler
        def h(category, radius, amount):
            l.kernelCategory = category
            l.radius = radius
            l.amount = amount
            l.applyToStack()
            #updateDocView()
            window.label.img.onImageChanged()
            #window.label.repaint()
        grWindow.onUpdateFilter = h
        # wrapper for the right apply method
        l.execute = lambda l=l, pool=None: l.applyFilter2D()
        # l.execute = lambda: l.applyLab1DLUT(grWindow.graphicsScene.cubicItem.getStackedLUTXY())
    elif name == 'actionSave_Layer_Stack':
        lastDir = window.settings.value('paths/dlgdir', '.')
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
        lastDir = window.settings.value('paths/dlgdir', '.')
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
                reply = QMessageBox()
                reply.setText('Unable to load 3D LUT')
                reply.setInformativeText(str(e))
                reply.setStandardButtons(QMessageBox.Ok)
                ret = reply.exec_()
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
            lastDir = window.settings.value('paths/dlgdir', '.')
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
    # set modal : docked windows
    # are not modal, so docking
    # must be disabled
    #dock.setAllowedAreas(Qt.NoDockWidgetArea)
    #dock.setWindowModality(Qt.ApplicationModal)
    # set ref. to the associated form
    #dock.setStyleSheet("border: 1px solid red")
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
        w, label = handleTextWindow(parent=window, title='About bLUe')
        label.setStyleSheet("background-image: url(logo.png); color: white;")
        label.setAlignment(Qt.AlignCenter)
        label.setText("Version 1.0")
        w.show()

def handleNewWindow(imImg=None, parent=None, title='New window', show_maximized=False, event_handler=True, scroll=False):
    """
    Show a floating window containing a QLabel object. It can be used
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

def handleTextWindow(parent=None, title=''):
    """
    Display a floating modal text window
    @param parent:
    @param title:
    @return (new window, label)
    @rtype: QMainWindow, QLabel
    """
    w, label = handleNewWindow(parent=parent, title=title, event_handler=False, scroll = True)

    w.setFixedSize(500,500)
    label.setAlignment(Qt.AlignTop)
    w.hide()
    w.setWindowModality(Qt.WindowModal)
    w.show()
    return w, label

def initMenuAssignProfile():
    window.menuAssign_profile.clear()
    for f in PROFILES_LIST:
        window.menuAssign_profile.addAction(f[0]+" "+f[1], lambda x=f : window.execAssignProfile(x))

def close(e):
    """
    app close event handler
    @param e: close event
    """
    if window.label.img.isModified:
        ret = saveChangeDialog(window.label.img)
        if ret == QMessageBox.Save:
            save(window.label.img, window)
            return True
        elif ret == QMessageBox.Cancel:
            return False
    return True

def updateStatus():
    img = window.label.img
    # filename and rating
    s = img.filename + ' ' + (' '.join(['*']*img.meta.rating))
    if img.useThumb:
        s = s + '      ' + '<font color=red><b>Preview</b></font> '
    else:
        # mandatory to toggle html mode
        s = s + '      ' + '<font color=black><b> </b></font> '
    if window.viewState == 'Before/After':
        s += '&nbsp;&nbsp;Before/After : Ctrl+Space : cycle through views - Space : switch back to workspace'
    else:
        s += '&nbsp;&nbsp;press Space Bar to toggle Before/After view'
    window.Label_status.setText(s)

###########
# app init
##########
if __name__ =='__main__':

    # style sheet
    app.setStyleSheet("QMainWindow, QGraphicsView, QListWidget, QMenu, QTableView {background-color: rgb(200, 200, 200)}\
                       QMenu, QTableView { selection-background-color: blue; selection-color: white;}\
                        QWidget, QTableView, QTableView * {font-size: 9pt}"
                     )
    # status bar
    window.Label_status = QLabel()
    window.statusBar().addWidget(window.Label_status)
    window.updateStatus = updateStatus

    # Before/After views flag
    window.splittedView = False

    window.histView.mode = 'Luminosity'
    window.histView.chanColors = Qt.gray #[Qt.red, Qt.green,Qt.blue]

    # splash screen
    pixmap = QPixmap('logo.png')
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    splash.showMessage("Loading .", color=Qt.white, alignment=Qt.AlignCenter)
    app.processEvents()
    sleep(1)
    splash.showMessage("Loading ...", color=Qt.white, alignment=Qt.AlignCenter)
    app.processEvents()

    # close event handler
    window.onCloseEvent = close

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
    #window.onShowContextMenu = contextMenu

    # load current settings
    window.readSettings()
    window._recentFiles = window.settings.value('paths/recent', [])

    set_event_handlers(window.label)
    set_event_handlers(window.label_2, enterAndLeave=False)
    set_event_handlers(window.label_3, enterAndLeave=False)

    img=QImage(200, 200, QImage.Format_ARGB32)
    img.fill(Qt.darkGray)
    defaultImImage = imImage(QImg=img, meta=metadata(name='noName'))

    PROFILES_LIST = icc.getProfiles()
    initMenuAssignProfile()
    #updateMenuOpenRecent()

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
    action1.triggered.connect(lambda: splittedWin.nextSplittedView())
    window.addAction(action1)

    # init property widget for tableView
    window.propertyWidget.setLayout(window.tableView.propertyLayout)

    # launch app
    sys.exit(app.exec_())