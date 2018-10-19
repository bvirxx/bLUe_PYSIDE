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
import ctypes
import os
import threading
from tempfile import mktemp

import cv2
import numpy as np
from math import factorial

from PySide2 import QtCore
from PySide2.QtGui import QColor, QImage, QPainter, QPixmap, QIcon
from PySide2.QtWidgets import QListWidget, QListWidgetItem, QDialog, QVBoxLayout, \
    QFileDialog, QSlider, QWidget, QHBoxLayout, QLabel, QMessageBox, QPushButton, QDockWidget, QStyle, QColorDialog
from PySide2.QtCore import Qt, QObject, QRect, QDir, QPointF
from os.path import isfile, basename
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

##################
# Base classes for signals
# They are mainly containers.
# As multiple inheritance leads to
# bugs with QObject, they can be used
# as a workaround to define
# custom signals (cf. QLayer).

class baseSignal_No(QObject):
    sig = QtCore.Signal()

class baseSignal_bool(QObject):
    sig = QtCore.Signal(bool)

class baseSignal_Int2(QObject):
    sig = QtCore.Signal(int, int, QtCore.Qt.KeyboardModifiers)
################

def qColorToRGB(color):
    """
    Converts a QColor to R,G,B components (range 0..255)
    @param color:
    @type color: QColor
    @return:
    @rtype: 3-uple of int
    """
    return color.red(), color.green(), color.blue()

def hideConsole():
    """
    Hides the console window
    """
    whnd = ctypes.windll.kernel32.GetConsoleWindow()
    if whnd != 0:
        ctypes.windll.user32.ShowWindow(whnd, 0)
        ctypes.windll.kernel32.CloseHandle(whnd)

def showConsole():
    """
    Shows the console window
    """
    whnd = ctypes.windll.kernel32.GetConsoleWindow()
    if whnd != 0:
        ctypes.windll.user32.ShowWindow(whnd, 1)
        ctypes.windll.kernel32.CloseHandle(whnd)

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
    # reflection mode for cols: cf rows above
    ax[:,:r[1]] = ax[:,::-1][:,-2*r[1]-1:-r[1]-1]
    ax[:,-r[1]:] = ax[:,::-1][:,r[1]+1:2*r[1]+1]
    # add two axes and strides for the windows
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
        # faster than np.var !!!
        f1 = movingAverage(a, winsize, version=version)
        f2 = movingAverage(a*a, winsize, version=version)
        return f2-f1*f1

def demosaic(raw_image_visible, raw_colors_visible, black_level_per_channel):
    """
    demosaic a sensor bitmap. The input array raw_image_visble has the same dimensions as the image,
    BUT NO channel. The array raw_colors_visible (identical shape) gives the color channel (0=R, 1=G, 2=B)
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

def dlgInfo(text, info=''):
    """
    Shows a simple information dialog.
    @param text:
    @type text: str
    @param info:
    @type info: str
    """
    msg = QMessageBox()
    msg.setWindowTitle('Information')
    msg.setIcon(QMessageBox.Information)
    msg.setText(text)
    msg.setInformativeText(info)
    msg.exec_()

def dlgWarn(text, info=''):
    """
    Shows a simple warning dialog.
    @param text:
    @type text: str
    @param info:
    @type info: str
    """
    msg = QMessageBox()
    msg.setWindowTitle('Warning')
    msg.setIcon(QMessageBox.Warning)
    msg.setText(text)
    msg.setInformativeText(info)
    msg.exec_()

def saveChangeDialog(img):
    """
    Save/discard dialog. Returns the chosen button.
    @param img: image to save
    @type img: vImage
    @return:
    @rtype: QMessageBox.StandardButton
    """
    reply = QMessageBox()
    reply.setText("%s was modified" % img.meta.name if len(img.meta.name) > 0 else 'unnamed image')
    reply.setInformativeText("Save your changes ?")
    reply.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
    reply.setDefaultButton(QMessageBox.Save)
    ret = reply.exec_()
    return ret

def saveDlg(img, mainWidget):
    """
    Image saving dialogs. The actual saving is
    done by a call to mImage.save(). Metadata is copied from sidecar
    to image file. The function returns the image file name.
    Exception ValueError or IOError are raised if the saving fails.
    @param img:
    @type img: vImage
    @param mainWidget:
    @type mainWidget: QWidget
    @return: filename
    @rtype: str
    """
    # get last accessed dir
    lastDir = str(mainWidget.settings.value("paths/dlgdir", QDir.currentPath()))
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
        # call mImage.save to write image to file : throw ValueError or IOError
        thumb = img.save(filename, quality=quality, compression=compression)
        tempFilename = mktemp('.jpg')
        # save jpg to temp file
        thumb.save(tempFilename)
        # copy temp file to image file
        img.restoreMeta(img.filename, filename, thumbfile=tempFilename)
        os.remove(tempFilename)
        return filename
    else:
        raise ValueError("Saving Operation Failure")

def openDlg(mainWidget, ask=True):
    """
    Returns a file name or None.
    @param mainWidget:
    @type mainWidget:
    @param ask:
    @type ask:
    @return:
    @rtype:
    """
    if ask and mainWidget.label.img.isModified:
        ret = saveChangeDialog(mainWidget.label.img)
        if ret == QMessageBox.Yes:
            try:
                saveDlg(mainWidget.label.img, mainWidget)
            except (ValueError, IOError) as e:
                dlgWarn(str(e))
                return
        elif ret == QMessageBox.Cancel:
            return
    # don't ask again for saving
    mainWidget.label.img.isModified = False
    lastDir = str(mainWidget.settings.value('paths/dlgdir', '.'))
    dlg = QFileDialog(mainWidget, "select", lastDir, " *".join(IMAGE_FILE_EXTENSIONS) + " *".join(RAW_FILE_EXTENSIONS))
    if dlg.exec_():
        filenames = dlg.selectedFiles()
        newDir = dlg.directory().absolutePath()
        mainWidget.settings.setValue('paths/dlgdir', newDir)
        return filenames[0]

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

class QbLUeColorDialog(QColorDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.closeSignal = baseSignal_No()
    def closeEvent(self, e):
        self.closeSignal.sig.emit()

class QbLUeSlider(QSlider):
    """
    Enhanced QSlider.
    Overrides mousepressevent to update the slider
    value with a single jump when clicking.
    """
    bLueSliderDefaultColorStylesheet = """QSlider::groove:horizontal:enabled {margin: 3px; 
                                              background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 blue, stop:1 red);}
                                           QSlider::groove:horizontal:disabled {margin: 3px; 
                                              background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8888FF, stop:1 #FF8888);}"""
    bLueSliderDefaultMGColorStylesheet = """QSlider::groove:horizontal:enabled {margin: 3px; 
                                                  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 magenta, stop:1 green);}
                                               QSlider::groove:horizontal:disabled {margin: 3px; 
                                                  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8888FF, stop:1 #FF8888);}"""
    bLueSliderDefaultIColorStylesheet = """QSlider::groove:horizontal:enabled {margin: 3px; 
                                                  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 red, stop:1 blue);}
                                               QSlider::groove:horizontal:disabled {margin: 3px; 
                                                  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8888FF, stop:1 #FF8888);}"""
    bLueSliderDefaultBWStylesheet = """QSlider::groove:horizontal:enabled {margin: 3px; 
                                              background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #333333, stop:1 white);}
                                           QSlider::groove:horizontal:disabled {margin: 3px; background: #888888;}"""
    bLueSliderDefaultIBWStylesheet = """QSlider::groove:horizontal:enabled {margin: 3px; 
                                                  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 gray, stop:1 #333333);}
                                               QSlider::groove:horizontal:disabled {margin: 3px; background: #888888;}"""
    def __init__(self, parent=None):
        super(QbLUeSlider, self).__init__(parent)
        self.setTickPosition(QSlider.NoTicks)
        self.setMaximumSize(16777215, 10)
        self.setStyleSheet("""QSlider::handle:horizontal {background: white; width: 15px; border: 1px solid black; border-radius: 4px; margin: -3px;}
                              QSlider::handle:horizontal:hover {background: #DDDDFF;}""")

    def mousePressEvent(self, event):
        """
        Updates the slider value with a single jump when clicking.

        @param event:
        @type event:
        """
        pressVal = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width(), 0)  # 0 is for horizontal slider only
        if abs(pressVal - self.value()) > 7:  # 7 to prevent jumps around max position
            self.setValue(pressVal)
            return
        super().mousePressEvent(event)

    def setStyleSheet(self, sheet):
        super().setStyleSheet(self.styleSheet() + sheet)

class QbLUeLabel(QLabel):
    """
    Emits a signal when double clicked
    """
    doubleClicked = QtCore.Signal()

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit()

class optionsWidgetItem(QListWidgetItem):
    def __init__(self, *args, intName='',**kwargs, ):
        super().__init__(*args, **kwargs)
        self._internalName = intName

    @property
    def internalName(self):
        return self._internalName

    def setInternalName(self, name):
        self._internalName = name

class optionsWidget(QListWidget) :
    """
    Displays a list of options with checkboxes.
    The choices can be mutually exclusive (default) or not
    exclusive. Actions can be done on item selection by assigning
    a function to onSelect. It is called after the selection of the new item.
    if changed is not None, it is called when an item is clicked.
    """

    def __init__(self, options=None, optionNames=None, exclusive=True, changed=None, parent=None):
        """
        @param options: list of options
        @type options: list of str
        @param optionNames: list of displayed names corresponding to options
        @type optionNames: list of str
        @param exclusive:
        @type exclusive: bool
        @param changed: SLOT for itemclicked signal
        @type changed: function
        @param parent:
        @type parent: QObject
        """
        super().__init__(parent)
        if options is None:
            options = []
        if optionNames is None:
            self.extNames = options
        else:
            self.extNames = optionNames
        self.intNames = options
        # dict of items with option internal name as keys
        self.items = {}
        # dict of item states (True, False) with option internal name as key
        self.options = {}
        for intName, name in zip(self.intNames, self.extNames):
            listItem = optionsWidgetItem(name, self, intName=intName)
            listItem.setCheckState(Qt.Unchecked)
            self.addItem(listItem)
            self.items[intName] = listItem
            self.options[intName] = (listItem.checkState() == Qt.Checked)
        #self.setMinimumWidth(self.sizeHintForColumn(0)) # TODO 18/04/18 validate suppression to improve graphicsLUT3D
        self.setMinimumHeight(self.sizeHintForRow(0)*len(options))
        self.setMaximumHeight(self.sizeHintForRow(0) * len(options) + 10) # TODO added 14/09/18 to improve the aspect of all graphic forms. Validate
        self.exclusive = exclusive
        self.itemClicked.connect(self.select)
        if changed is not None:
            self.itemClicked.connect(changed)
        # selection hook.
        self.onSelect = lambda x : 0

    def select(self, item, callOnSelect=True):
        """
        Item clicked event handler. It updates the states of the items and
        the dict of options. Next, if callOnSelect is True, onSelect is called.
        @param item:
        @type item: QListWidgetItem
        @param callOnSelect:
        @type callOnSelect: bool
        """
        # Update item states:
        # if exclusive, clicking on an item should turn it
        # into (or keep it) checked. Otherwise, there is nothing to do
        # since select is called after the item state has changed.
        if self.exclusive:
            for r in range(self.count()):
                currentItem = self.item(r)
                if currentItem is not item:
                    currentItem.setCheckState(Qt.Unchecked)
                else:
                    currentItem.setCheckState(Qt.Checked)
        # update options dict
        for option in self.options.keys():
            self.options[option] = (self.items[option].checkState() == Qt.Checked)
        if callOnSelect:
            self.onSelect(item)

    def checkOption(self, name, checked=True, callOnSelect=True):
        """
        Check or (for non exclusive options only) uncheck an item.
        Next, if callOnSelect is True, onSelect is called.
        A ValueError exception is raised  if an attempt is done to
        uncheck an item in a list of mutually exclusive options.
        @param name: internal name of option
        @type name: str
        @param checked: check/uncheck flag
        @type checked: bool
        @param callOnSelect:
        @type callOnSelect: bool
        """
        item = self.items[name]
        if not checked and self.exclusive:
            raise ValueError('For mutually exclusive options, unchecking is not possible. Please, check another item')
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.select(item, callOnSelect=callOnSelect)

    def unCheckAll(self):
        if self.exclusive:
            return
        for r in range(self.count()):
            self.item(r).setCheckState(Qt.Unchecked)

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

def checkeredImage(format=QImage.Format_ARGB32):
    """
    Returns a 20x20 checker
    @param format:
    @type format:
    @return: checker
    @rtype: QImage
    """
    base = QImage(20, 20, format)
    qp = QPainter(base)
    qp.setCompositionMode(QPainter.CompositionMode_Source)
    qp.fillRect(0, 0, 10, 10, Qt.gray)
    qp.fillRect(10, 0, 10, 10, Qt.white)
    qp.fillRect(0, 10, 10, 10, Qt.white)
    qp.fillRect(10, 10, 10, 10, Qt.gray)
    qp.end()
    return base
    qp=QPainter(image)
    qp.setCompositionMode(QPainter.CompositionMode_Source)
    # draw the pattern once at 0,0
    qp.drawImage(0, 0, base)
    imageW, imageH = image.width(), image.height()
    baseW, baseH = base.width(), base.height()
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

class stateAwareQDockWidget(QDockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._closed = False
    def closeEvent(self, event):
        self._closed = True
        super().closeEvent(event)
    @property
    def isClosed(self):
        return self._closed

class loader(threading.Thread):
    """
    Thread class for batch loading of images in a
    QListWidget object
    """
    def __init__(self, gen, wdg):
        """

        @param gen: generator of image file names
        @type gen: generator
        @param wdg:
        @type wdg: QListWidget
        """
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
                        profile, metadata = e.get_metadata(filename, createsidecar=False)
                    except ValueError:
                        metadata = [{}]
                    # get image info
                    orientation = metadata[0].get("EXIF:Orientation", 1)
                    # EXIF:DateTimeOriginal seems to be missing in many files
                    date = metadata[0].get("EXIF:ModifyDate", '')
                    rating = metadata[0].get("XMP:Rating", 5)
                    rating = ''.join(['*']*int(rating))
                    transformation = exiftool.decodeExifOrientation(orientation)
                    # get thumbnail
                    img = e.get_thumbNail(filename, thumbname='thumbnailimage')
                    # no thumbnail found : try preview
                    if img.isNull():
                        img = e.get_thumbNail(filename, thumbname='PreviewImage')  # the order is important : for jpeg PreviewImage is full sized !
                    # all failed : open image
                    if img.isNull():
                        img = QImage(filename)
                    # remove possible black borders, except for .NEF
                    if filename[-3:] not in ['nef', 'NEF']:
                        bBorder = 7
                        img = img.copy(QRect(0,bBorder, img.width(), img.height()-2*bBorder))
                    pxm = QPixmap.fromImage(img)
                    if not transformation.isIdentity():
                        pxm = pxm.transformed(transformation)
                    # set item caption and tooltip
                    item = QListWidgetItem(QIcon(pxm), basename(filename)) # + '\n' + rating)
                    item.setToolTip(basename(filename) + ' ' + date + ' ' + rating)
                    # set item mimeData to get filename=item.data(Qt.UserRole)[0] transformation=item.data(Qt.UserRole)[1]
                    item.setData(Qt.UserRole, (filename, transformation))
                    self.wdg.addItem(item)
                # for clean exiting we catch all exceptions and force break
                except OSError:
                    continue
                except:
                    break

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

def boundingRect(img, pattern):
    """
    Given an image img, the function builds the bounding rectangle
    of the region defined by (img == pattern). If the region is empty, the function
    returns an invalid rectangle.
    @param img:
    @type img: 2D array
    @param pattern:
    @type pattern: img.dtype
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
    # build the enclosing rectangle
    left = leftPattern(img)
    right = img.shape[1] - 1 - leftPattern(img[::-1, ::-1])
    top = leftPattern(img.T)
    bottom = img.shape[0] - 1 - leftPattern(img.T[::-1, ::-1])
    return QRect(left, top, right - left, bottom - top)

if __name__ == '__main__':
    a= np.ones(dtype=int).reshape(10,10)
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