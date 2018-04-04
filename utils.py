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
import cv2
from PySide2 import QtCore

import numpy as np
from math import factorial
from PySide2.QtGui import QColor, QPainterPath, QPen, QImage, QPainter, QTransform, QPolygonF
from PySide2.QtWidgets import QListWidget, QListWidgetItem, QGraphicsPathItem, QDialog, QVBoxLayout, \
    QFileDialog, QSlider, QWidget, QHBoxLayout, QLabel, QMessageBox, QPushButton, QToolButton
from PySide2.QtCore import Qt, QPoint, QObject, QRect, QDir
from os.path import isfile
from itertools import product

from numpy.lib.stride_tricks import as_strided

import exiftool
from imgconvert import QImageBuffer

##################
# file extension constants
IMAGE_FILE_EXTENSIONS = (".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF", "*.bmp", "*.BMP")
RAW_FILE_EXTENSIONS = (".nef", ".NEF", ".dng", ".DNG", ".cr2", ".CR2")
IMAGE_FILE_NAME_FILTER = ['Image Files (*.jpg *.png *.tif *.JPG *.PNG *.TIF)']
#################

def rolling_window(a, winsize):
    """
    Add a last axis to an array, filled with the values of a
    1-dimensional sliding window

    @param a: array
    @type a: ndarray, dtype= any numeric type
    @param winsize: size of the moving window
    @type winsize: int
    @return: array with last axis added
    @rtype: ndarray
    """
    shape = a.shape[:-1] + (a.shape[-1] - winsize + 1, winsize)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def strides_2d(a, r, linear=True):
    """
    Compute the 2D moving windows of an array. The size of the windows
    is h=2*r[0]+1, w=2*r[1]+1, they are centered at the given array item
    and completed by reflection at borders if needed. If linear is True, the
    windows are reshaped as 1D arrays, otherwise they are left unchanged.
    The output array keeps the shape and dtype of a.
    The original idea is taken from

    U{https://gist.github.com/thengineer/10024511}

    @param a: 2D array
    @type a: ndarray, ndims=2
    @param r: window sizes
    @type r: 2-uple of int
    @param linear:
    @type linear: boolean
    @return: array of moving windows
    @rtype: ndarray, shape=a.shape, dtype=a.dtype
    """
    ax = np.zeros(shape=(a.shape[0] + 2 * r[0], a.shape[1] + 2 * r[1]), dtype=a.dtype)
    ax[r[0]:ax.shape[0] - r[0], r[1]:ax.shape[1] - r[1]] = a
    # reflection mode for rows:  ...2,1,0,1,2...
    for i in range(r[0]):
        imod = (i+1) % a.shape[0] - 1 # cycle through rows if r[0] >= a.shape[0]
        ax[r[0]-i-1, r[1]:-r[1]]= a[imod+1,:]  # copy rows a[1,:]... to  ax[r[0]-1,:]...
        ax[i-r[0], r[1]:-r[1]] = a[-imod-2,:]  # copy rows a[-2,:]... to  ax[-r[0],:]...
    #ax[:r[0],r[1]:-r[1]] = a[::-1,:][-r[0]-1:-1,:]
    #ax[-r[0]:,r[1]:-r[1]] = a[::-1,:][1:r[0]+1,:]
    # reflection mode for cols: cf rows
    ax[:,:r[1]] = ax[:,::-1][:,-2*r[1]-1:-r[1]-1]
    ax[:,-r[1]:] = ax[:,::-1][:,r[1]+1:2*r[1]+1]
    # add two new axes and strides for the windows
    shape = a.shape + (1 + 2 * r[0], 1 + 2 * r[1]) # concatenate t-uples
    strides = ax.strides + ax.strides # concatenate t-uples
    s = as_strided(ax, shape=shape, strides=strides)
    # reshape
    return s.reshape(a.shape + (shape[2] * shape[3],)) if linear else s

def movingAverage(a, winsize, version='kernel'):
    """
    Compute the moving averages of a 1D or 2D array.
    For 1D arrays, the borders are not handled : the dimension of
    the returned array is a.shape[0] - winsize//2.
    For 2D arrays, the window is square (winsize*winsize), the
    borders are handled by reflection and the returned array
    keeps the shape of a. For 2D arrays, if version='kernel' (default)
    we use the opencv function filter2D to compute the moving average. It is
    fast but suffers from a lack of precision. If version = 'strides',
    we perform a direct and more precise computation,
    using 64 bits floating numbers.
    @param a: array
    @type a: ndarray ndims = 1 or 2
    @param winsize: size of moving window
    @type winsize: int
    @param version: 'kernel' or 'strides'
    @type version: str
    @return: array of moving averages
    @rtype: ndarray, dtype = np.float32 if a.ndims==2 and version=='kernel', otherwise
            a.dtype (int types are cast to np.float64)
    """
    n = a.ndim
    if n == 1:
        return np.mean(rolling_window(a, winsize), axis=-1)
    elif n == 2:
        if version == 'kernel':
            kernel = np.ones((winsize, winsize), dtype=np.float32) / (winsize * winsize)
            return cv2.filter2D(a.astype(np.float32), -1, kernel.astype(np.float32))
        else:
            r = int((winsize - 1) / 2)
            b = strides_2d(a, (r, r), linear=False)
            m = np.mean(b, axis=(-2,-1))
            return m
    else:
        raise ValueError('array ndims must be 1 or 2')

def movingVariance(a, winsize, version='kernel'):
    """
    Compute the moving variance of a 1D or 2D array.
    For 1D arrays, the borders are not handled : the dimension of
    the returned array is a.shape[0] - winsize//2.
    For 2D arrays, the window is square (winsize*winsize), the
    borders are handled by reflection and the returned array
    keeps the shape of a.
    @param a: array
    @type a: ndarray ndims = 1 or 2
    @param winsize: size of moving window
    @type winsize: int
    @return: array of moving variances
    @rtype: ndarray, dtype = np.float32 or np.float64
    """
    if a.ndim > 2:
        raise ValueError('array ndims must be 1 or 2')
    #a = a - np.mean(a)
    if version == 'kernel':
        a = a.astype(np.float32)
        f1 = movingAverage(a, winsize, version=version)
        f2 = movingAverage(a * a, winsize, version=version)
        return f2 - f1 * f1
    else:
        a = a.astype(np.float64)
        """
        r = int((winsize - 1) / 2)
        b = strides_2d(a, (r, r))
        return np.var(b, axis=-1)
        """
        # faster than np.var !!!
        f1 = movingAverage(a, winsize, version=version)
        f2 = movingAverage(a*a, winsize, version=version)
        return f2-f1*f1

class channelValues():
    RGB, Red, Green, Blue =[0,1,2], [0], [1], [2]
    HSB, Hue, Sat, Br = [0, 1, 2], [0], [1], [2]
    Lab, L, a, b = [0, 1, 2], [0], [1], [2]

def demosaic(raw_image_visible, raw_colors_visible, black_level_per_channel):
    """
    demosaic a sensor bitmap. The input array raw_image_visble has the same dimensions as the image,
    BUT NO CHANNEL. The array raw_colors_visible (identical shape) gives the color channel (0=R, 1+G, 2=B)
    corresponding to each point.
    @param raw_image_visible:
    @type raw_image_visible: nd_array, dtype uint16, shape(img_h, img_w)
    @param raw_colors_visible:
    @type raw_colors_visible: nd_array, dtype u1, shape(img_h, img_w)
    @param black_level_per_channel:
    @type black_level_per_channel: list or array, dtype= int
    @return: demosaic array
    @rtype: ndarray, dtype uint16, shape (img_width, img_height, 3)
    """
    black_level_per_channel = np.array(black_level_per_channel, dtype=np.uint16)
    # Bayer bitmap (16 bits), subtract black level for each channel
    if np.any(black_level_per_channel!=0):
        bayerBuf = raw_image_visible - black_level_per_channel[raw_colors_visible]
    else:
        bayerBuf = raw_image_visible
    # encode Bayer pattern to opencv constant
    tmpdict = {0:'R', 1:'G', 2:'B'}
    pattern = 'cv2.COLOR_BAYER_' + tmpdict[raw_colors_visible[1,1]] + tmpdict[raw_colors_visible[1,2]] + '2RGB'
    # demosaic
    demosaic = cv2.cvtColor(bayerBuf, eval(pattern))
    return demosaic

def multiply(matr_a, matr_b):
    """Return product of an MxP matrix A with an PxN matrix B."""
    cols, rows = len(matr_b[0]), len(matr_b)
    resRows = range(len(matr_a))
    rMatrix = [[0] * cols for _ in resRows]
    for idx in resRows:
        for j, k in product(range(cols), range(rows)):
            rMatrix[idx][j] += matr_a[idx][k] * matr_b[k][j]
    return rMatrix

def inversion(m):
    """
    @param m:
    @type m:
    @return:
    @rtype:
    """
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = m.ravel()
    inv = np.array([[m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
                    [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
                    [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4]])
    return inv / multiply(inv[0], m[:, 0])

def saveChangeDialog(img):
    reply = QMessageBox()
    reply.setText("%s has been modified" % img.meta.name if len(img.meta.name) > 0 else 'unnamed image')
    reply.setInformativeText("Save your changes ?")
    reply.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
    reply.setDefaultButton(QMessageBox.Save)
    ret = reply.exec_()
    return ret

def save(img, mainWidget):
    """
    Image saving dialogs. The actual saving is
    done by calling mImage.save(). Metadata is copied from sidecar
    to image file. The function returns the image file name.
    Exception ValueError or IOError are raised if the saving fails.
    @param img:
    @type img: QImage
    @param mainWidget:
    @type mainWidget: QWidget
    @return: filename
    @rtype: str
    """
    # get last accessed dir
    lastDir = mainWidget.settings.value("paths/dlgdir", QDir.currentPath())
    # file dialogs
    dlg = savingDialog(mainWidget, "Save", lastDir)
    # default saving format JPG
    dlg.selectFile(img.filename[:-3] + 'JPG')
    dlg.dlg.currentChanged.connect(lambda f: print(f))
    if dlg.exec_():
        newDir = dlg.directory().absolutePath()
        mainWidget.settings.setValue('paths/dlgdir', newDir)
        filenames = dlg.selectedFiles()
        if filenames:
            filename = filenames[0]
        else:
            raise ValueError("You must select a file")
        if isfile(filename):
            reply = QMessageBox()
            reply.setWindowTitle('Warning')
            reply.setIcon(QMessageBox.Warning)
            reply.setText("File %s already exists\n" % filename)
            reply.setStandardButtons(QMessageBox.Cancel)
            accButton = QPushButton("Save as New Copy")
            rejButton = QPushButton("OverWrite")
            reply.addButton(accButton, QMessageBox.AcceptRole)
            reply.addButton(rejButton, QMessageBox.RejectRole)
            reply.setDefaultButton(accButton)
            reply.exec_()
            retButton = reply.clickedButton()
            # build a unique name
            if retButton is accButton:
                i = 0
                base = filename
                if '_copy' in base:
                    flag = '_'
                else:
                    flag = '_copy'
                while isfile(filename):
                    filename = base[:-4] + flag + str(i) + base[-4:]
                    i = i+1
            # overwrite
            elif retButton is rejButton:
                pass
            else:
                raise ValueError("Saving Operation Failure")
        # get parameters
        quality = dlg.sliderQual.value()
        compression = dlg.sliderComp.value()
        # write image file : throw ValueError or IOError
        img.save(filename, quality=quality, compression=compression)  #mImage.save()
        # copy metadata to image file. The sidecar is not removed
        with exiftool.ExifTool() as e:
            e.restoreMetadata(img.filename, filename)
        return filename
    else:
        raise ValueError("Saving Operation Failure")

def openDlg(mainWidget):
    if mainWidget.label.img.isModified:
        ret = saveChangeDialog(mainWidget.label.img)
        if ret == QMessageBox.Yes:
            save(mainWidget.label.img, mainWidget)
        elif ret == QMessageBox.Cancel:
            return
    lastDir = mainWidget.settings.value('paths/dlgdir', '.')
    dlg = QFileDialog(mainWidget, "select", lastDir, " *".join(IMAGE_FILE_EXTENSIONS) + " *".join(RAW_FILE_EXTENSIONS))
    if dlg.exec_():
        filenames = dlg.selectedFiles()
        newDir = dlg.directory().absolutePath()
        mainWidget.settings.setValue('paths/dlgdir', newDir)
        return filenames[0]
    else:
        return None

class UDict(object):
    """
    Union of dictionaries
    """
    def __init__(self, d1, d2):
       self.d1, self.d2 = d1, d2
    def __getitem__(self, item):
       if item in self.d1:
           return self.d1[item]
       return self.d2[item]

class QbLUeSlider(QSlider):

    bLueSliderDefaultColorStylesheet = """QSlider::groove:horizontal:enabled {margin: 3px; 
                                              background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 blue, stop:1 red);}
                                           QSlider::groove:horizontal:disabled {margin: 3px; 
                                              background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8888FF, stop:1 #FF8888);}"""
    bLueSliderDefaultBWStylesheet = """QSlider::groove:horizontal:enabled {margin: 3px; 
                                              background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 black, stop:1 white);}
                                           QSlider::groove:horizontal:disabled {margin: 3px; background: #888888;}"""
    def __init__(self, parent=None):
        super(QbLUeSlider, self).__init__(parent)
        self.setTickPosition(QSlider.NoTicks)
        self.setMaximumSize(16777215, 10)
        self.setStyleSheet("""QSlider::handle:horizontal {background: white; width: 10px; border: 1px solid black; border-radius: 4px; margin: -3px;}
                              QSlider::handle:horizontal:hover {background: #DDDDFF;}""")

    def setStyleSheet(self, sheet):
        QSlider.setStyleSheet(self, self.styleSheet() + sheet)

class optionsWidget(QListWidget) :
    """
    Displays a list of options with checkboxes.
    The choices can be mutually exclusive (default) or not
    exclusive. Actions can be done on item selection by assigning
    a function to onSelect. It is called after the selection of the new item.
    if changed is not None, the signal is emitted when an item is clicked.
    """

    def __init__(self, options=[], optionNames=None, exclusive=True, changed=None, parent=None):
        """
        @param options: list of options
        @type options: list of str
        @param optionNames : list of displayed names corresponding to options
        @type optionNames: lit of str
        @param exclusive :
        @type exclusive: boolean
        @param changed: signal
        @type changed: QSignal
        @param oarent:
        @type parent: QObject
        """
        super(optionsWidget, self).__init__(parent)
        if optionNames is None:
            self.extNames = options
        else:
            self.extNames = optionNames
        self.intNames = options
        # dict of items with keys option internal name
        self.items = {}
        # dict of item states with option internal name as key
        self.options = {}
        for option, name in zip(self.intNames, self.extNames):
            listItem = QListWidgetItem(name, self)
            listItem.setCheckState(Qt.Unchecked)
            self.addItem(listItem)
            self.items[option] = listItem
            self.options[option] = (listItem.checkState() == Qt.Checked)
        #self.setSizeAdjustPolicy(QListWidget.AdjustToContents)
        self.setMinimumWidth(self.sizeHintForColumn(0))
        self.setMinimumHeight(self.sizeHintForRow(0)*len(options))
        self.exclusive = exclusive
        self.itemClicked.connect(self.select)
        if changed is not None:
            self.itemClicked.connect(changed)
        # selection hook.
        self.onSelect = lambda x : 0

    def select(self, item):
        """
        Item clicked event handler
        @param item:
        @type item: QListWidgetItem
        """
        if self.exclusive:
            for r in range(self.count()):
                currentItem = self.item(r)
                if currentItem is not item:
                    currentItem.setCheckState(Qt.Unchecked)
                else:
                    currentItem.setCheckState(Qt.Checked)
        for option in self.options.keys():
            self.options[option] = (self.items[option].checkState() == Qt.Checked)
        self.onSelect(item)

    def checkOption(self, name):
        """

        @param name: internal name of option
        @type name: str
        """
        item = self.items[name]
        item.setCheckState(Qt.Checked)
        self.select(item)

    def unCheckAll(self):
        if self.exclusive:
            return
        for r in range(self.count()):
            self.item(r).setCheckState(Qt.Unchecked)

class croppingHandle(QToolButton):
    """
    Active button, draggable with the mouse

    """
    def __init__(self, role='', tool=None, parent=None):
        """
        parent is the widget showing the image.
        role is 'left, 'right', 'top', 'bottom',
        'topRight', 'topLeft', 'bottomRight', 'bottomLeft'
        @param role:
        @type role: str
        @param parent:
        @type parent: QWidget
        """
        super().__init__(parent=parent)
        self.role = role
        self.margin = 0.0
        self.group = {}
        self.tool = tool
        self.setVisible(False)
        self.setGeometry(0,0,10,10)
        self.setAutoFillBackground(True)
        self.setAutoRaise(True)
        self.setStyleSheet("QToolButton:hover {background-color:#00FF00} QToolButton {background-color:#555555}")

    def setPosition(self, p):
        """
        Updates button margins in response to a mouse move event
        @param p: mouse cursor position (relative to parent widget)
        @type p: QPoint

        """
        widg = self.parent()
        img = widg.img
        r = img.resize_coeff(widg)
        # middle buttons
        if self.role == 'left':
            margin = (p.x() - img.xOffset + self.width()) / r
            if margin < 0 or margin >= img.width() - self.group['right'].margin:
                return
            self.margin = margin
        elif self.role == 'right':
            margin = img.width() - (p.x() - img.xOffset) / r
            if margin < 0 or margin >= img.width() - self.group['left'].margin:
                return
            self.margin = margin
        elif self.role == 'top':
            margin = (p.y() - img.yOffset + self.height()) / r
            if margin < 0 or margin >= img.height() - self.group['bottom'].margin:
                return
            self.margin = margin
        elif self.role == 'bottom':
            margin = img.height() - (p.y() - img.yOffset) / r
            if margin < 0 or margin >= img.height() - self.group['top'].margin:
                return
            self.margin = margin
        # angle buttons
        elif self.role == 'topRight':
            rMargin = img.width() - (p.x() - img.xOffset) / r
            lMargin = self.group['left'].margin
            bMargin = self.group['bottom'].margin
            w = img.width() - rMargin - lMargin
            h = w * self.tool.formFactor
            tMargin = img.height() - h - bMargin
            if rMargin < 0 or rMargin >= img.width() - lMargin or tMargin < 0 or tMargin >= img.height() - bMargin:
                return
            self.group['right'].margin = rMargin
            self.group['top'].margin = tMargin
        elif self.role == 'topLeft':
            lBtn = self.group['left']
            lMargin = (p.x() - img.xOffset + lBtn.width()) / r
            rMargin = self.group['right'].margin
            bMargin = self.group['bottom'].margin
            w = img.width() - lMargin - rMargin
            h = w * self.tool.formFactor
            tMargin = img.height() - h - bMargin
            if lMargin < 0 or lMargin >= img.width() - rMargin or tMargin < 0 or tMargin >= img.height() - bMargin:
                return
            self.group['top'].margin = tMargin
            self.group['left'].margin = lMargin
        elif self.role == 'bottomLeft':
            lBtn = self.group['left']
            lMargin = (p.x() - img.xOffset + lBtn.width()) / r
            rMargin = self.group['right'].margin
            tMargin = self.group['top'].margin
            w = img.width() - lMargin - rMargin
            h = w * self.tool.formFactor
            bMargin = img.height() - h - tMargin
            if lMargin < 0 or lMargin >= img.width() - rMargin or tMargin < 0 or tMargin >= img.height() - bMargin:
                return
            self.group['bottom'].margin = bMargin
            self.group['left'].margin = lMargin
        elif self.role == 'bottomRight':
            btn = self.group['right']
            rMargin = img.width() - (p.x() - img.xOffset) / r
            lMargin = self.group['left'].margin
            tMargin = self.group['top'].margin
            w = img.width() - lMargin - rMargin
            h = w * self.tool.formFactor
            bMargin = img.height() - h - tMargin
            if rMargin < 0 or rMargin >= img.width() - lMargin or bMargin < 0 or bMargin >= img.height() - tMargin:
                return
            self.group['right'].margin = rMargin
            self.group['bottom'].margin = bMargin
        img.cropLeft, img.cropRight, img.cropTop, img.cropBottom = lMargin, rMargin, tMargin, bMargin

    def mousePressEvent(self, event):
        img = self.parent().img
        self.tool.formFactor = (img.height() - self.group['top'].margin - self.group['bottom'].margin) / (img.width() - self.group['left'].margin - self.group['right'].margin)

    def mouseMoveEvent(self, event):
        img = self.parent().img
        pos = self.mapToParent(event.pos())
        oldPos = self.pos()
        if self.role in ['left', 'right']:
            self.setPosition(self.pos() + QPoint((pos-oldPos).x(),0))
        elif self.role in ['top', 'bottom']:
            self.setPosition(self.pos() + QPoint(0, (pos - oldPos).y()))
        # angle buttons
        else:
            self.setPosition(pos)
        self.tool.drawCropTool(self.parent().img)
        self.tool.formFactor = (img.height() - self.group['top'].margin - self.group['bottom'].margin) / (
                                            img.width() - self.group['left'].margin - self.group['right'].margin)
        self.parent().updateStatus()
        self.parent().repaint()

class cropTool(QObject):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        cropButtonLeft = croppingHandle(role='left', tool=self, parent=parent)
        cropButtonRight = croppingHandle(role='right', tool=self, parent=parent)
        cropButtonTop = croppingHandle(role='top', tool=self, parent=parent)
        cropButtonBottom = croppingHandle(role='bottom', tool=self, parent=parent)
        cropButtonTopLeft = croppingHandle(role='topLeft', tool=self, parent=parent)
        cropButtonTopRight = croppingHandle(role='topRight', tool=self, parent=parent)
        cropButtonBottomLeft = croppingHandle(role='bottomLeft', tool=self, parent=parent)
        cropButtonBottomRight = croppingHandle(role='bottomRight', tool=self, parent=parent)
        btnList = [cropButtonLeft, cropButtonRight, cropButtonTop, cropButtonBottom,
                   cropButtonTopLeft, cropButtonTopRight, cropButtonBottomLeft, cropButtonBottomRight]
        self.btnDict = {btn.role: btn for btn in btnList}
        for btn in btnList:
            btn.group = self.btnDict

        self.cRect = QRect(0, 0, 0, 0)

        self.formFactor = 1.0

    def drawCropTool(self, img):
        """
        Draws the 8 crop handles around the displayed image,
        with their current margins.
        @param img:
        @type img: QImage
        @param r: current resizing coefficient (normalized zoom coeff.)
        @type r: float
        """
        r = self.parent().img.resize_coeff(self.parent())
        left = self.btnDict['left']
        top = self.btnDict['top']
        bottom = self.btnDict['bottom']
        right = self.btnDict['right']
        self.cRect = QRect(round(left.margin), round(top.margin), img.width()-round(right.margin+left.margin), img.height()-round(bottom.margin+top.margin))
        p = self.cRect.topLeft()*r + QPoint(img.xOffset, img.yOffset)
        x, y = p.x(), p.y()
        w, h = self.cRect.width()*r, self.cRect.height()*r
        left.move(x - left.width(), y + h//2)
        right.move(x + w, y + h//2)
        top.move(x + w//2, y - top.height())
        bottom.move(x + w//2, y + h)
        topLeft = self.btnDict['topLeft']
        topLeft.move(x - topLeft.width(), y - topLeft.height())
        topRight = self.btnDict['topRight']
        topRight.move(x + w, y - topRight.height())
        bottomLeft = self.btnDict['bottomLeft']
        bottomLeft.move(x - bottomLeft.width(), y + h)
        bottomRight = self.btnDict['bottomRight']
        bottomRight.move(x + w, y + h)

class rotatingHandle(QToolButton):
    def __init__(self, role=None, tool=None, pos=QPoint(0,0), parent=None):
        super().__init__(parent=parent)
        self.tool = tool
        self.role=role
        self.posRelImg = pos
        self.setVisible(False)
        self.setGeometry(0, 0, 20, 20)
        self.setAutoFillBackground(True)
        self.setAutoRaise(True)
        self.setStyleSheet("QToolButton:hover {background-color:#00FF00} QToolButton {background-color:#AA0000}")

    def mousePressEvent(self, event):
        self.resizingCoeff = self.tool.layer.parentImage.resize_coeff(self.tool.parent())

    def mouseMoveEvent(self, event):
        pos = self.mapToParent(event.pos())
        img = self.tool.layer.parentImage
        r = self.resizingCoeff
        # get coordinates relative to full resolution image
        p = pos - QPoint(img.xOffset, img.yOffset)
        self.posRelImg = QPoint(p.x()/r, p.y()/r)

        self.tool.drawRotatingTool()
        poly = self.tool.getQuad()
        T2 = QTransform()
        b = QTransform().quadToQuad(self.tool.oriQuad, poly, T2)
        if self.tool.form.options['Rotation']:
            T2 = QTransform().rotate(self.tool.form.sliderRot.value())
        self.tool.layer.geoTrans = T2
        self.tool.layer.rectTrans = poly.boundingRect()
        self.tool.layer.applyToStack()
        self.parent().repaint()

class rotatingTool(QObject):

    @classmethod
    def getNewRotatingTool(cls, parent=None, layer=None, form=None):
        tool = rotatingTool(parent=parent, layer=layer, form=form)
        form.tool = tool
        return tool

    def __init__(self, parent=None, layer=None, form=None):
        self.layer = layer
        self.form = form
        self.img=layer.parentImage
        w,h = self.img.width(), self.img.height()
        super().__init__(parent=parent)
        rotatingButtonLeft = rotatingHandle(role='topLeft', tool=self, pos=QPoint(0,0), parent=parent)
        rotatingButtonRight = rotatingHandle(role='topRight', tool=self, pos=QPoint(w,0), parent=parent)
        rotatingButtonTop = rotatingHandle(role='bottomLeft', tool=self, pos=QPoint(0,h), parent=parent)
        rotatingButtonBottom = rotatingHandle(role='bottomRight', tool=self, pos=QPoint(w,h), parent=parent)
        btnList = [rotatingButtonLeft, rotatingButtonRight, rotatingButtonTop, rotatingButtonBottom]
        self.btnDict = {btn.role: btn for btn in btnList}
        for btn in btnList:
            btn.group = self.btnDict
        self.cRect = QRect(0, 0, 0, 0)
        self.oriQuad = self.getQuad()
        self.drawRotatingTool()
        self.showTool()
        # rotation angle changed handler
        def g():
            self.drawRotatingTool()
            poly = self.getQuad()
            T2 = QTransform().rotate(self.form.sliderRot.value())
            self.layer.geoTrans = T2
            self.layer.rectTrans = poly.boundingRect()
            self.layer.applyToStack()
            self.parent().repaint()
        self.form.sliderRot.valueChanged.connect(g)

    def showTool(self):
        for btn in self.btnDict.values():
            btn.show()

    def hideTool(self):
        for btn in self.btnDict.values():
            btn.hide()

    def setVisible(self, value):
        for btn in self.btnDict.values():
            btn.setVisible(value)

    def setTransform(self, transformation):
        rect0 = QRect(0, 0, self.img.width(), self.img.height())
        rect1 = transformation.mapRect(rect0)
        for role,pos in zip(['topLeft', 'topRight', 'bottomLeft', 'bottomRight'], [rect1.topLeft(), rect1.topRight(), rect1.bottomLeft(), rect1.bottomRight()]):
            self.btnDict[role].posRelImg = pos
        self.layer.geoTrans = transformation
        self.drawRotatingTool()

    def getQuad(self):
        poly = QPolygonF()
        s = self.img.getCurrentImage().width() / self.img.width()
        for role in ['topLeft', 'topRight', 'bottomRight', 'bottomLeft']:
            poly.append(self.btnDict[role].posRelImg * s)
        return poly

    def drawRotatingTool(self):
        """
        Draws the 4 handles around the displayed image,
        at their current position
        @param img:
        @type img: QImage
        @param r: current resizing coefficient (normalized zoom coeff.)
        @type r: float
        """
        r = self.parent().img.resize_coeff(self.parent())
        topLeft = self.btnDict['topLeft']
        topRight = self.btnDict['topRight']
        bottomLeft = self.btnDict['bottomLeft']
        bottomRight = self.btnDict['bottomRight']
        self.cRect = QRect(0, 0, self.img.width(), self.img.height())
        # get coordinates of image topLeft point (relative to widget)
        p = self.cRect.topLeft() + QPoint(self.img.xOffset, self.img.yOffset)
        x, y = p.x(), p.y()
        w, h = self.cRect.width()*r, self.cRect.height()*r
        bottomLeft.move(x + bottomLeft.posRelImg.x()*r, y - bottomLeft.height() + bottomLeft.posRelImg.y()*r)
        bottomRight.move(x - bottomRight.width() + bottomRight.posRelImg.x()*r, y - bottomRight.height() + bottomRight.posRelImg.y()*r)
        topLeft.move(x + topLeft.posRelImg.x()*r, y + topLeft.posRelImg.y()*r)
        topRight.move(x - topRight.width() + topRight.posRelImg.x()*r, y + topRight.posRelImg.y()*r)

class savingDialog(QDialog):
    """
    File dialog with quality and compression sliders.
    We use a standard QFileDialog as a child widget and we
    forward its methods to the top level.
    """
    def __init__(self, parent, text, lastDir):
        """

        @param parent:
        @type parent: QObject
        @param text:
        @type text: str
        @param lastDir:
        @type lastDir:str
        """
        # QDialog __init__
        super().__init__()
        self.setWindowTitle(text)
        # File Dialog
        self.dlg = QFileDialog(caption=text, directory=lastDir)
        # sliders
        self.sliderComp = QSlider(Qt.Horizontal)
        self.sliderComp.setTickPosition(QSlider.TicksBelow)
        self.sliderComp.setRange(0, 9)
        self.sliderComp.setSingleStep(1)
        self.sliderComp.setValue(5)
        self.sliderQual = QSlider(Qt.Horizontal)
        self.sliderQual.setTickPosition(QSlider.TicksBelow)
        self.sliderQual.setRange(0, 100)
        self.sliderQual.setSingleStep(10)
        self.sliderQual.setValue(90)
        self.dlg.setVisible(True)
        l = QVBoxLayout()
        h = QHBoxLayout()
        l.addWidget(self.dlg)
        h.addWidget(QLabel("Quality"))
        h.addWidget(self.sliderQual)
        h.addWidget(QLabel("Compression"))
        h.addWidget(self.sliderComp)
        l.addLayout(h)
        self.setLayout(l)
        # file dialog close event handler
        def f():
            self.close()
        self.dlg.finished.connect(f)

    def exec_(self):
        # QDialog exec_
        super().exec_()
        # forward file dialog result
        return self.dlg.result()

    def selectFile(self, fileName):
        self.dlg.selectFile(fileName)

    def selectedFiles(self):
        return self.dlg.selectedFiles()

    def directory(self):
        return self.dlg.directory()

class SavitzkyGolay:
    """
    Savitzky-Golay Filter.
    This is a pure numpy implementation of the Savitzky_Golay filter. It is taken
    from U{http://stackoverflow.com/questions/22988882/how-to-smooth-a-curve-in-python}
    Many thanks to elviuz.
    """
    window_size = 11   # must be odd
    order = 3
    deriv = 0
    rate = 1
    kernel = None
    @classmethod
    def getKernel(cls):
        if cls.kernel is None:
            order_range = range(cls.order + 1)
            half_window = (cls.window_size - 1) // 2
            # compute the array m of filter coefficients
            b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
            cls.kernel = np.linalg.pinv(b).A[cls.deriv] * cls.rate ** cls.deriv * factorial(cls.deriv)
        return cls.kernel
    @classmethod
    def filter(cls, y):
        """
        @param y: data
        @type y: 1D ndarray, dtype = float
        @return: the filtered data array
        """
        kernel = cls.getKernel()
        half_window = (cls.window_size -1) // 2
        # pad the signal at the extremes with values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        #y = np.concatenate(([0]*half_window, y, [0]*half_window))
        # apply filter
        return np.convolve( kernel[::-1], y, mode='valid')

def checkeredImage(w, h, format=QImage.Format_ARGB32):
    image = QImage(w, h, format)

    # init pattern
    base = QImage(20, 20, format)
    qp = QPainter(base)
    qp.setCompositionMode(QPainter.CompositionMode_Source)
    qp.fillRect(0, 0, 10, 10, Qt.gray)
    qp.fillRect(10, 0, 10, 10, Qt.white)
    qp.fillRect(0, 10, 10, 10, Qt.white)
    qp.fillRect(10, 10, 10, 10, Qt.gray)
    qp.end()

    qp=QPainter(image)
    qp.setCompositionMode(QPainter.CompositionMode_Source)

    # draw the pattern once at 0,0
    qp.drawImage(0, 0, base)

    imageW = image.width()
    imageH = image.height()
    baseW = base.width()
    baseH = base.height()
    while ((baseW < imageW) or (baseH < imageH) ):
        if (baseW < imageW) :
            # Copy and draw the existing pattern to the right
            qp.drawImage(QRect(baseW, 0, baseW, baseH), image, QRect(0, 0, baseW, baseH))
            baseW *= 2
        if (baseH < imageH) :
            # Copy and draw the existing pattern to the bottom
            qp.drawImage(QRect(0, baseH, baseW, baseH), image, QRect(0, 0, baseW, baseH))
            # Update height of our pattern
            baseH *= 2
    qp.end()
    return image

def clip(image, mask, inverted=False):
    """
    clip an image by applying a mask to its alpha channel
    @param image:
    @type image:
    @param mask:
    @type mask:
    @param inverted:
    @type inverted:
    @return:
    @rtype:
    """
    bufImg = QImageBuffer(image)
    bufMask = QImageBuffer(mask)
    if inverted:
        bufMask = bufMask.copy()
        bufMask[:,:,3] = 255 - bufMask[:,:,3]
    bufImg[:,:,3] = bufMask[:,:,3]

def drawPlotGrid(axeSize):
    item = QGraphicsPathItem()
    item.setPen(QPen(QColor(255, 0, 0), 1, Qt.DashLine))
    qppath = QPainterPath()
    qppath.moveTo(QPoint(0, 0))
    qppath.lineTo(QPoint(axeSize, 0))
    qppath.lineTo(QPoint(axeSize, -axeSize))
    qppath.lineTo(QPoint(0, -axeSize))
    qppath.closeSubpath()
    qppath.lineTo(QPoint(axeSize, -axeSize))
    for i in range(1, 5):
        a = (axeSize * i) / 4
        qppath.moveTo(a, -axeSize)
        qppath.lineTo(a, 0)
        qppath.moveTo(0, -a)
        qppath.lineTo(axeSize, -a)
    item.setPath(qppath)
    return item
    #self.graphicsScene.addItem(item)

def boundingRect(img, pattern):
    """
    Given an image img, the function builds the bounding rectangle
    of the region defined by img == pattern. If the region is empty, the function
    returns an invalid rectangle.
    @param img:
    @type img: 2D array
    @param pattern:
    @type pattern: a.dtype
    @return:
    @rtype: QRect or None
    """
    def leftPattern(b):
        """
        For a 1-channel image, returns the leftmost
        x-coordinate of max value.
        @param b: image
        @type b: 2D array, dtype=int or float
        @return: leftmost x-coordinate of max value
        @rtype: int
        """
        # we build the array of first occurrences of row max
        XMin = np.argmax(b, axis=1)
        # To exclude the rows with a max different of the global max,
        # we assign to them a value greater than all possible indices.
        XMin = np.where(np.diagonal(b[:, XMin])==np.max(b), XMin, np.sum(b.shape)+1)
        return np.min(XMin)

    # indicator function of the region
    img = np.where(img==pattern, 1, 0)
    # empty region
    if np.max(img) == 0:
        return None
    # building enclosing rectangle
    left = leftPattern(img)
    right = img.shape[1] - 1 - leftPattern(img[::-1, ::-1])
    top = leftPattern(img.T)
    bottom = img.shape[0] - 1 - leftPattern(img.T[::-1, ::-1])
    return QRect(left, top, right - left, bottom - top)

if __name__ == '__main__':
    a= np.array((1,)*100, dtype=int).reshape(10,10)
    #b=strides_2d(a, (11,11))
    m = movingVariance(a,7)
    print(m)
"""
#pickle example
saved_data = dict(outputFile, 
                  saveFeature1 = feature1, 
                  saveFeature2 = feature2, 
                  saveLabel1 = label1, 
                  saveLabel2 = label2,
                  saveString = docString)

with open('test.dat', 'wb') as outfile:
    pickle.dump(saved_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
"""