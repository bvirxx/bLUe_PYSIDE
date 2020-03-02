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
from os import path, walk, remove
from os.path import basename, isfile
from tempfile import mktemp

from bLUeTop import resources_rc  # mandatory

import numpy as np
import multiprocessing
import sys
import threading
from itertools import cycle
from time import sleep
import gc

from types import MethodType
import rawpy

from bLUeCore.bLUeLUT3D import HaldArray
from bLUeTop.drawing import initBrushes
from bLUeTop.graphicsDraw import drawForm
from bLUeTop.graphicsHDRMerge import HDRMergeForm
from bLUeTop.graphicsSegment import segmentForm
from PySide2.QtCore import QUrl, QFileInfo
from PySide2.QtGui import QPixmap, QCursor, QKeySequence, QDesktopServices, QFont, \
    QTransform, QColor, QImage
from PySide2.QtWidgets import QApplication, QAction, \
    QDockWidget, QSizePolicy, QSplashScreen, QWidget, \
    QTabWidget, QToolBar, QComboBox, QTabBar
from bLUeTop.QtGui1 import app, window, splitWin
from bLUeTop import exiftool
from bLUeTop.graphicsBlendFilter import blendFilterForm
from bLUeTop.graphicsHVLUT2D import HVLUT2DForm
from bLUeTop.graphicsInvert import invertForm
from bLUeTop.graphicsMixer import mixerForm
from bLUeTop.graphicsNoise import noiseForm
from bLUeTop.graphicsRaw import rawForm
from bLUeTop.graphicsTransform import transForm, imageForm
from bLUeGui.bLUeImage import QImageBuffer, QImageFormats
from bLUeTop.rawProcessing import rawRead
from bLUeTop.versatileImg import vImage, metadataBag
from bLUeTop.MarkedImg import imImage, QRawLayer, QCloningLayer
from bLUeTop.graphicsRGBLUT import graphicsForm
from bLUeTop.graphicsLUT3D import graphicsForm3DLUT
from bLUeTop.lutUtils import LUTSIZE, LUT3D, LUT3DIdentity
from bLUeGui.colorPatterns import cmHSP, cmHSB
from bLUeTop.colorManagement import icc
from bLUeTop.graphicsCoBrSat import CoBrSatForm
from bLUeTop.graphicsExp import ExpForm
from bLUeTop.graphicsPatch import patchForm
from bLUeTop.settings import USE_POOL, POOL_SIZE, THEME, TABBING
from bLUeTop.utils import UDict, stateAwareQDockWidget
from bLUeGui.tool import cropTool, rotatingTool
from bLUeTop.graphicsTemp import temperatureForm
from bLUeTop.graphicsFilter import filterForm
from bLUeTop.graphicsHspbLUT import graphicsHspbForm
from bLUeTop.graphicsLabLUT import graphicsLabForm

from bLUeGui.dialog import *
from bLUeTop.viewer import playDiaporama, viewer

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
seamlessClone and CLAHE are Opencv functions
mergeMertens is an Opencv class
grabCut is a parallel version of an Opencv function
This product includes DNG technology under license by Adobe Systems Incorporated
credit https://icones8.fr/
"""
#################

##############
#  Version number
VERSION = "v2.1.3"
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


def widgetChange(button, window=window):
    """
    called by all main form button and slider slots (cf. QtGui1.py onWidgetChange)

    @param button:
    @type button: QWidget
    @param window:
    @type window: QWidget
    """
    # wdgName = button.objectName()
    if button is window.fitButton:  # wdgName == "fitButton" :
        window.label.img.fit_window()
        # update crop button positions
        window.cropTool.drawCropTool(window.label.img)
    elif button is window.cropButton:
        if button.isChecked():
            window.cropTool.drawCropTool(window.label.img)
            for b in window.cropTool.btnDict.values():
                b.show()
        else:
            for b in window.cropTool.btnDict.values():
                b.hide()
        window.label.img.isCropped = button.isChecked()
    elif button is window.rulerButton:  # wdgName == "rulerButton":
        window.label.img.isRuled = button.isChecked()
    elif button is window.eyeDropper:  # wdgName == 'eyeDropper':
        if button.isChecked():  # window.btnValues['colorPicker']:
            dlg = window.colorChooser
            dlg.show()
            try:
                dlg.closeSignal.sig.disconnect()
            except RuntimeError:
                pass
            dlg.closeSignal.sig.connect(lambda: button.setChecked(False))
        else:
            window.colorChooser.close()
    updateStatus()
    window.label.repaint()


def addBasicAdjustmentLayers(img, window=window):
    if img.rawImage is None:
        # menuLayer('actionColor_Temperature')
        # menuLayer('actionExposure_Correction')
        menuLayer('actionContrast_Correction')
    # select active layer : top row
    window.tableView.select(0, 1)


def addRawAdjustmentLayer(window=window):
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
    dock = stateAwareQDockWidget(window)
    dock.tabbed = TABBING
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


def loadImage(img, withBasic=True, window=window):
    """
    load a vImage into bLUe
    @param img:
    @type img: vImage
    @param withBasic:
    @type withBasic: boolean
    @param window:
    @type window: QWidget
    """
    tabBar = window.tabBar
    ind = tabBar.addTab(basename(img.filename))
    tabBar.setCurrentIndex(ind)
    tabBar.setTabData(ind, img)
    setDocumentImage(img)
    # switch to preview mode and process stack
    window.tableView.previewOptionBox.setChecked(True)
    # window.tableView.previewOptionBox.stateChanged.emit(Qt.Checked)  # TODO removed 22/02/20 stack is processed below validate
    # add development layer for raw image, and develop
    if img.rawImage is not None:
        addRawAdjustmentLayer()
    # add default adjustment layers
    if withBasic:
        addBasicAdjustmentLayers(img)
    # updates
    img.layersStack[0].applyToStack()
    img.onImageChanged()


def openFile(f, window=window):
    """
    Top level function for file opening, used by File Menu actions
    @param f: file name
    @type f: str
    @param window:
    @type window: QWidget
    """
    try:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        # load imImage from file
        img = imImage.loadImageFromFile(f, cmsConfigure=True, window=window)
        # init layers
        if img is not None:
            loadImage(img)
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

def saveFile(filename, img, quality=-1, compression=-1, writeMeta=True):
    """
    Save image and meta data to file
    @param filename:
    @type filename:
    @param img:
    @type img:
    @param quality:
    @type quality:
    @param compression:
    @type compression:
    @param writeMeta:
    @type writeMeta:
    """
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
                i = i + 1
        # overwrite
        elif retButton is rejButton:
            pass
        else:
            raise ValueError("Saving Operation Failure")
    # call mImage.save to write image to file and return a thumbnail
    # throw ValueError or IOError
    thumb = img.save(filename, quality=quality, compression=compression)
    # write metadata
    if writeMeta:
        tempFilename = mktemp('.jpg')
        # save thumb jpg to temp file
        thumb.save(tempFilename)
        # copy temp file to image file, img.filename not updated yet
        img.restoreMeta(img.filename, filename, thumbfile=tempFilename)
        remove(tempFilename)
    return filename


def closeTabs(index=None, window=window):
    """
    Tries to save and close the opened document in tab index, or all opened documents if index is None .
    If it succeeds to close all opened documents, the method resets the GUI to default.
    """
    if not canClose(index=index) or window.tabBar.count() > 0:
        return
    window.tableView.clear(delete=True)
    window.histView.targetImage = None
    defaultImImage = initDefaultImage()
    window.label.img = defaultImImage
    window.label_2.img = defaultImImage
    window.label_3.img = defaultImImage
    gc.collect()
    window.label.update()
    window.label_2.update()
    window.label_3.update()


def showHistogram(window=window):
    """
    Update and display the histogram of the
    currently opened document
    """
    if window.histView.listWidget1.items['Original Image'].checkState() is Qt.Checked:
        histImg = vImage(QImg=window.label.img.getCurrentImage())  # must be vImage : histogram method needed
    else:
        histImg = window.label.img.layersStack[-1].getCurrentMaskedImage()
    if window.histView.options['R'] or window.histView.options['G'] or window.histView.options['B']:
        window.histView.mode = 'RGB'
        window.histView.chanColors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(10, 10, 255)]
        window.histView.chans = [ ['R', 'G', 'B'].index(ch) for ch in ['R', 'G', 'B'] if window.histView.options[ch]]
    else:
        window.histView.mode = 'Luminosity'
        window.histView.chanColors = [Qt.gray]
        window.histView.chans = []
    histView = histImg.histogram(QSize(window.histView.width(), window.histView.height()),
                                 chans=window.histView.chans, bgColor=Qt.black,
                                 chanColors=window.histView.chanColors, mode=window.histView.mode, addMode='Luminosity' if window.histView.options['L'] else '')
    window.histView.cache = QPixmap.fromImage(histView)
    window.histView.Label_Hist.setPixmap(window.histView.cache.scaled(window.histView.width() - 20, window.histView.height()-50))
    window.histView.Label_Hist.repaint()


def adjustHistogramSize(window=window):
    pxm = getattr(window.histView, 'cache', None)
    if pxm is not None:
        window.histView.Label_Hist.setPixmap(pxm.scaled(window.histView.width() - 20, window.histView.height()-50))
        window.histView.Label_Hist.repaint()

def restoreBrush(d):
    """
    Sync brush tools with values in d
    @param d:
    @type d: dict
    """
    if d is None:
        return
    window.verticalSlider1.setValue(d['size'])
    window.verticalSlider2.setValue(int(d['opacity'] * 100.0))
    window.verticalSlider3.setValue(int(d['hardness'] * 100.0))
    window.verticalSlider4.setValue(int(d['flow'] * 100.0))
    ind = window.brushCombo.findText(d['name'])
    if ind != -1:
        window.brushCombo.setCurrentIndex(ind)
    window.colorChooser.setCurrentColor(d['color'])
    window.label.State['brush'] = d.copy()


def setDocumentImage(img, window=window):
    """
    Inits GUI and displays the current document
    @param img: image
    @type img: imImage
    @param window:
    @type window: QWidget
    """
    if img is None:
        return
    # restore color management
    icc.configure(colorSpace=img.colorSpace, workingProfile=img.cmsProfile)
    # restore GUI
    window.label.img.savedBtnValues = window.btnValues.copy()
    d = img.savedBtnValues
    if d:  # a saved dict exists
        window.btnValues = d.copy()
    else: # reset to default
        for k in window.btnValues:
            window.btnValues[k] = False
        window.btnValues['pointer'] = True  # default checked autoexclusive button (needed)
    window.label.img = img
    ind = window.tabBar.currentIndex()
    if window.tabBar.tabData(ind) is not img:
        window.tabBar.setTabText(ind, basename(img.filename))
        window.tabBar.setTabData(ind, img)
    window.cropTool.fit(img)
    window.cropTool.drawCropTool(img)
    # set button states
    for btn in window.btns.values():
        s = btn.autoExclusive()
        btn.setAutoExclusive(False)
        btn.setChecked(window.btnValues[btn.accessibleName()])
        btn.setAutoExclusive(s)
    # init histogram
    window.histView.targetImage = window.label.img

    # image changed event handler
    def f(hist=True):
        # refresh windows (use repaint for faster update)
        window.label.repaint()
        window.label_3.repaint()
        if not hist:
            return
        # recompute and display histogram for the selected image
        showHistogram()

    # active layer changed event handler
    def g():
        layer = window.label.img.getActiveLayer()
        if layer.isDrawLayer():
            restoreBrush(layer.brushDict)

    window.label.img.onImageChanged = f
    window.label.img.onActiveLayerChanged = g

    # init = first change
    f()
    g()
    ###################################
    # init displayed images
    # label.img : working image
    # label_2.img  : before image (copy of the initial state of working image)
    # label_3.img : reference to working image
    ###################################
    # before image : the stack is not copied
    window.label_2.img = imImage(QImg=img, meta=img.meta)
    # restore normal mode
    if window.viewState != 'After':
        window.viewState = 'After'
        window.splitter.hide()
        window.label.show()
    # after image : ref to the opened document
    window.label_3.img = img
    # no mouse drawing or painting
    window.label_2.img.isMouseSelectable = False
    # init layer view
    window.tableView.setLayers(window.label.img)
    tool = window.label.img.getActiveLayer().tool
    if tool is not None:
        tool.showTool()
    window.label.update()
    window.label_2.update()
    window.label_3.update()
    updateStatus()
    gc.collect()  # tested (very) efficient here
    # back links used by graphicsForm3DLUT.onReset  # TODO 3/1/20 unused removed validate
    # window.label.img.window = window.label
    # window.label_2.img.window = window.label_2
    # window.label.img.setModified(True)  # TODO removed 27/01/20 validate


def updateMenuOpenRecent(window=window):
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


def updateEnabledActions(window=window):
    """
    Menu aboutToShow handler
    @return:
    @rtype:
    """
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    window.actionSave.setEnabled(window.label.img.isModified)
    window.actionSave_As.setEnabled(window.label.img.isModified)
    window.actionSave_Hald_Cube.setEnabled(window.label.img.isHald)


def menuFile(name, window=window):
    """
    Menu handler
    @param name: action name
    @type name: str
    @param window:
    @type window: QWidget
    """
    # new image
    if name == 'actionNew_2':
        img = None
        cb = QApplication.clipboard()
        Qimg = cb.image()
        if not Qimg.isNull():
            img = imImage(QImg=Qimg)
            cb.clear()
        else:
            dims = {'w': 200, 'h': 200}
            dlg = dimsInputDialog(dims)
            if dlg.exec_():
                imgNew = QImage(dims['w'], dims['h'], QImage.Format_ARGB32)
                imgNew.fill(Qt.white)
                img = imImage(QImg=imgNew)
        if img is None:
            return
        img.filename = 'unnamed'
        loadImage(img, withBasic=False)
    # load image from file
    elif name in ['actionOpen']:
        # get file name from dialog
        filename = openDlg(window, ask=False)
        # open file
        if filename is not None:
            openFile(filename)
    # saving dialog
    elif name == 'actionSave' or name == 'actionSave_As':
        saveAs = (name=='actionSave_As')
        if window.label.img.useThumb:
            dlgWarn("Uncheck Preview mode before saving")
        else:
            img = window.label.img
            try:
                if saveAs:
                    filename, quality, compression, writeMeta = saveDlg(img, window, selected=not saveAs)
                    filename = saveFile(filename, img, quality=quality, compression=compression, writeMeta=writeMeta)
                else:
                    filename = saveFile(img.filename, img, writeMeta=True)
                img.filename = filename  # don't move up !
                window.tabBar.setTabText(window.tabBar.currentIndex(), basename(filename))
                dlgInfo("%s written" % filename)
                window.label.img.setModified(False)
            except (ValueError, IOError) as e:
                dlgWarn(str(e))
    # closing dialog : close opened document
    elif name == 'actionClose':
        closeTabs()
        # global pool
        # if pool is not None:
            # pool.close()
            # pool.join()
            # pool = None  # TODO removed 28/01/20 for multi doc
    updateStatus()


def menuView(name, window=window):
    """
    Menu handler
    @param name: action name
    @type name: str
    @param window:
    @type window: QWidget
    """
    ##################
    # before/after mode
    ##################
    if name == 'actionShow_hide_right_window_3':
        if window.splitter.isHidden():
            splitWin.setSplitView()
            window.viewState = 'Before/After'
        else:
            window.splitter.hide()
            window.label.show()
            window.splitView = False
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
        dlg.setOptions(QFileDialog.DontUseNativeDialog)  # Native Dialog is too slow
        # open dialog
        if dlg.exec_():
            newDir = dlg.selectedFiles()[0]  # dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
            viewerInstance = viewer.getViewerInstance(mainWin=window)
            viewerInstance.playViewer(newDir)  # asynchronous
    ###############
    # Color Chooser
    ###############
    elif name == 'actionColor_Chooser':
        window.colorChooser.show()
    elif name == 'actionHistogram':
        window.histViewDock.show()
    updateStatus()


def menuImage(name, window=window):
    """
    Menu handler
    @param name: action name
    @type name: str
    @param window:
    @type window: QWidget
    """
    img = window.label.img
    # display image info
    if name == 'actionImage_info':
        # Format
        s = "Format : %s\n(cf. QImage formats in the doc for more info)" % QImageFormats.get(img.format(), 'unknown')
        # dimensions
        s = s + "\n\ndim : %d x %d" % (img.width(), img.height())
        # profile info
        if img.meta.profile is not None:
                s = s + "\n\nEmbedded profile found"  # length %d" % len(img.meta.profile)
        workingProfileInfo = icc.workingProfileInfo
        s = s + "\n\nWorking Profile : %s" % workingProfileInfo
        # rating
        s = s + "\n\nRating %s" % ''.join(['*']*img.meta.rating)
        # formatted meta data
        s = s + "\n\n" + img.imageInfo
        # display
        w = labelDlg(parent=window, title='Image info', wSize=QSize(700, 700), search=True)
        w.label.setWordWrap(True)
        w.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        font = QFont("Courier New")
        w.label.setFont(font)
        w.label.setText(w.wrapped(s))
        w.show()
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
        w = labelDlg(parent=window, title='profile info')
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
        w.label.setWordWrap(True)
        w.label.setText(s)
        w.show()
    # rotations
    elif name in ['action90_CW', 'action90_CCW', 'action180']:
        try:
            angle = 90 if name == 'action90_CW' else -90 if name == 'action90_CCW' else 180
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            # get new imImage
            tImg = img.bTransformed(QTransform().rotate(angle))
            setDocumentImage(tImg)
            tImg.layersStack[0].applyToStack()
            tImg.onImageChanged()
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
    # resize
    elif name == 'actionImage_Resizing':
        w, h = img.width(), img.height()
        dims = {'w': w, 'h': h}
        dlg = dimsInputDialog(dims, keepBox=True)
        if dlg.exec_():
            img = window.label.img.resize(dims['w'] * dims['h'])
            img.filename = 'unnamed'
            setDocumentImage(img)
            img.layersStack[0].applyToStack()
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


def menuLayer(name, window=window):
    """
    Menu Layer handler
    @param name: action name
    @type name: str
    @param window:
    @type window: QWidget
    """
    # postlude
    def post(layer):
        # adding a new layer may modify the resulting image
        # (cf. actionNew_Image_Layer), so we update the presentation layer
        layer.parentImage.prLayer.update()
        layer.parentImage.onImageChanged()
        # record action name for scripting
        layer.actionName = name
        # docking the form
        dock = stateAwareQDockWidget(window)
        dock.tabbed = TABBING
        dock.setWidget(grWindow)
        dock.setWindowTitle(grWindow.windowTitle())
        layer.view = dock
        # update the view of layer stack
        window.tableView.setLayers(window.label.img)

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
        layer.execute = lambda l=layer, pool=pool: l.tLayer.apply3DLUT(sc.lut,
                                                                       options=sc.options,
                                                                       pool=pool)
    elif name == 'action2D_LUT_HV':
        layerName = '3D LUT HV Shift'
        layer = window.label.img.addAdjustmentLayer(name=layerName, role='2DLUT')
        grWindow = HVLUT2DForm.getNewWindow(axeSize=300, targetImage=window.label.img,
                                                  layer=layer, parent=window)
        # init pool only once
        pool = getPool()
        sc = grWindow.scene()
        layer.execute = lambda l=layer, pool=pool: l.tLayer.applyHVLUT2D(grWindow.LUT, options=sc.options, pool=pool)
    # cloning
    elif name == 'actionNew_Cloning_Layer':
        lname = 'Cloning'
        layer = window.label.img.addAdjustmentLayer(layerType=QCloningLayer, name=lname, role='CLONING')
        grWindow = patchForm.getNewWindow(targetImage=window.label.img, layer=layer, parent=window)
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyCloning(seamless=l.autoclone)
    # segmentation
    elif name == 'actionNew_segmentation_layer':
        lname = 'Segmentation'
        layer = window.label.img.addSegmentationLayer(name=lname)
        grWindow = segmentForm.getNewWindow(targetImage=window.label.img, layer=layer)
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyGrabcut(nbIter=grWindow.nbIter)
    # load an image from file
    elif name == 'actionLoad_Image_from_File':
        filenames = openDlg(window, ask=False, multiple=True)
        if not filenames:
            return
        for filename in filenames:
            # load image from file, alpha channel is mandatory for applyTransform()
            ext = filename[-4:]
            if ext in list(IMAGE_FILE_EXTENSIONS):
                imgNew = QImage(filename).convertToFormat(QImage.Format_ARGB32)  # QImage(filename, QImage.Format_ARGB32) does not work !
            elif ext in list(RAW_FILE_EXTENSIONS):
                rawpyInst = rawRead(filename)
                # postprocess raw image, applying default settings (cf. vImage.applyRawPostProcessing)
                rawBuf = rawpyInst.postprocess(use_camera_wb=True)
                # build Qimage : swittch to BGR and add alpha channel
                rawBuf = np.dstack((rawBuf[:, :, ::-1], np.zeros(rawBuf.shape[:2], dtype=np.uint8) + 255))
                imgNew = vImage(cv2Img=rawBuf)
                # keeping a reference to rawBuf along with img is
                # needed to protect the buffer from garbage collector
                imgNew.rawBuf = rawBuf
            else:
                return
            if imgNew.isNull():
                dlgWarn("Cannot load %s: " % filename)
                return
            lname = path.basename(filename)
            layer = window.label.img.addAdjustmentLayer(name=lname, sourceImg=imgNew, role='Image')
            grWindow = imageForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer, parent=window)
            # add transformation tool to parent widget
            tool = rotatingTool(parent=window.label)  # , layer=l, form=grWindow)
            layer.addTool(tool)
            tool.showTool()
            layer.execute = lambda l=layer, pool=None: l.tLayer.applyImage(grWindow.options)
            layer.actioname = name
            layer.filename = filename
            post(layer)
        return
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
    elif name == 'actionNew_Drawing_Layer':
        processedImg = window.label.img
        w, h = processedImg.width(), processedImg.height()
        imgNew = QImage(w, h, QImage.Format_ARGB32)
        #imgNew.fill(Qt.white)
        imgNew.fill(QColor(0, 0, 0, 0))
        lname = 'Drawing'
        layer = window.label.img.addAdjustmentLayer(name=lname, sourceImg=imgNew, role='DRW')
        grWindow = drawForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer, parent=window)
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyNone()
        layer.actioname = name
    # Color filter
    elif name == 'actionColor_Temperature':
        lname = 'Color Filter'
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
        layer.execute = lambda l=layer,  pool=None: l.tLayer.applyExposure(grWindow.options)
    elif name == 'actionHDR_Merge':
        lname = 'Merge'
        layer = window.label.img.addAdjustmentLayer(name=lname, role='MERGING')
        layer.clipLimit = ExpForm.defaultExpCorrection
        grWindow = HDRMergeForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=layer, parent=window)
        layer.execute = lambda l=layer,  pool=None: l.tLayer.applyHDRMerge(grWindow.options)
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
            layer.execute = lambda l=layer, pool=pool: l.tLayer.apply3DLUT(lut,
                                                                           UDict(({'use selection': False, 'keep alpha': True},)),
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
            hArray = HaldArray(buf, LUT3DIdentity.size)
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
    post(layer)


def menuHelp(name, window=window):
    """
    Menu handler
    Init help browser
    A single instance is created.
    Unused parameters are for the sake of symmetry
    with other menu function calls.
    @param name: action name
    @type name: str
    @param window:
    @type window: QWidget
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
        w = labelDlg(parent=window, title='About bLUe', wSize=QSize(520, 520))  # 500 + layout margins
        w.label.setStyleSheet("background-image: url(logo.png); color: white;")
        w.label.setAlignment(Qt.AlignCenter)
        w.label.setText(VERSION + "\n" + attributions + "\n" + "http://bernard.virot.free.fr")
        w.show()


def canClose(index=None, window=window):
    """
    If index is None, tries to save and close all opened documents, otherwise only
    the document in index tab is considered.
    Returns True if all requested tabs could be closed, False otherwise.
    Called by the application closeEvent slot, by closeTabs() and by closeTab().
    @param index: a valid tab index or None
    @type index: int or None
    @param window:
    @type window:
    @return:
    @rtype: boolean
    """
    if window.tabBar.count() == 0:
        return True
    closeAllRequested = (index is None)

    def canCloseTab(ind):
        img = window.tabBar.tabData(ind)
        if img.isModified:
            if ind != window.tabBar.currentIndex():
                dlgWarn('Image was modified', info='Save it first' )
                return False
            try:
                # save/discard dialog
                ret = saveChangeDialog(img)
                if ret == QMessageBox.Save:
                    if img.useThumb:
                        dlgWarn("Uncheck Preview Mode before saving")
                        return False
                    # save dialog
                    filename, quality, compression, writeMeta = saveDlg(img, window, selected=False)
                    # actual saving
                    filename = saveFile(filename, img, quality=quality, compression=compression,
                                            writeMeta=writeMeta)
                    # confirm saving
                    dlgInfo("%s written" % filename)
                    if ind > 0:
                        switchDoc(ind - 1)
                    window.tabBar.removeTab(ind)
                    return True # window.tabBar.count() == 0
                elif ret == QMessageBox.Cancel:
                    return False
            except (ValueError, IOError) as e:
                dlgWarn(str(e))
                return False
        # discard changes or img not modified : remove tab
        if ind > 0:
            switchDoc(ind - 1)
        window.tabBar.removeTab(ind)
        return True

    if closeAllRequested:
        indList = list(range(window.tabBar.count()))
        while indList:
            ind = window.tabBar.currentIndex()
            if not canCloseTab(ind):
                break
            try:
                indList.remove(ind)
            except ValueError:
                pass
    else:
        return canCloseTab(index)
    return window.tabBar.count() == 0


def updateStatus(window=window):
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
        s += '&nbsp;&nbsp;&nbsp;&nbsp;Space : toggle Before/After view'
    # cropping
    if window.label.img.isCropped:
        w, h = window.cropTool.crWidth, window.cropTool.crHeight
        s = s + '&nbsp;&nbsp;&nbsp;&nbsp;Cropped : %dx%d h/w=%.2f ' % (w, h, h / w)
    window.Label_status.setText(s)


def initCursors(window=window):
    """
    Init app cursors
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


def screenUpdate(newScreenIndex, window=window):
    """
    screenChanged event handler.
    The image is updated in background
    @param newScreenIndex:
    @type newScreenIndex: QScreen
    @param window:
    @type window: QWidget
    """
    window.screenChanged.disconnect()
    # update the color management object using the profile associated with the current monitor
    icc.configure(qscreen=newScreenIndex)
    window.actionColor_manage.setEnabled(icc.HAS_COLOR_MANAGE)
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    updateStatus()

    # launch a bg task for updating of presentation layers
    def bgTask():
        window.label.img.updatePixmap()
        window.label_2.img.updatePixmap()
        window.label.update()
        window.label_2.update()
    threading.Thread(target=bgTask).start()
    window.screenChanged.connect(screenUpdate)


class HistQDockWidget(QDockWidget):
    def resizeEvent(self, e):
        adjustHistogramSize()


def setRightPane(window=window):
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
            histViewDock = HistQDockWidget()
            hl = QHBoxLayout()
            hl.setAlignment(Qt.AlignLeft)
            hl.addWidget(w)
            w.setMaximumSize(140000, 140000)
            wdg = QWidget()
            wdg.setMaximumSize(140000, 140000)
            wdg.setLayout(hl)
            histViewDock.setWidget(wdg)
            # short title
            histViewDock.setWindowTitle(w.windowTitle())
            window.addDockWidget(Qt.RightDockWidgetArea, histViewDock)
            window.histViewDock = histViewDock
        # add other widgets to layout
        else:
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
    window.tabifyDockWidget(histViewDock, window.infoView)


def setColorManagement(window=window):
    """
    color management configuration
    must be done after showing window
    """
    window.screenChanged.connect(screenUpdate)
    # screen detection
    # get current QScreen instance
    window.currentScreenIndex = window.windowHandle().screen()
    # update the color management object with the current monitor profile
    icc.configure(qscreen=window.currentScreenIndex)
    icc.COLOR_MANAGE = icc.HAS_COLOR_MANAGE
    window.actionColor_manage.setEnabled(icc.HAS_COLOR_MANAGE)
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    updateStatus()


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


def setupGUI(window=window):
    """
    Display splash screen and set app style sheet
    @param window:
    @type window:
    """
    # splash screen
    splash = QSplashScreen(QPixmap('logo.png'), Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage("Loading .", color=Qt.white, alignment=Qt.AlignCenter)
    app.processEvents()
    sleep(1)
    splash.showMessage(VERSION + "\n" + attributions + "\n" + "http://bernard.virot.free.fr", color=Qt.white,
                       alignment=Qt.AlignCenter)
    app.processEvents()
    sleep(1)
    splash.finish(window)
    # app title
    window.setWindowTitle('bLUe')

    #######################
    # docking areas options
    #######################
    # The right docking area extends to window bottom
    window.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

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
                               QListWidget::item:disabled{color: gray}
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
                               QTabBar::tab:selected {border-top-color: red; 
                                                      color: white;}
                               QTabBar::tab:!selected {margin-bottom: 2px}
                               QDockWidget::title {background-color: #444455}
                               QDockWidget::title:hover{background-color: #555588}
                               QToolTip {border: 0px;
                                        background-color: lightyellow;
                                        color: black}"""
                          # border must be set, otherwise background-color has no effect : Qt bug?
                          )

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

    # init button tool bars
    toolBar = QToolBar()
    window.verticalSlider1, window.verticalSlider2, window.verticalSlider3, window.verticalSlider4\
        = QSlider(Qt.Horizontal), QSlider(Qt.Horizontal), QSlider(Qt.Horizontal), QSlider(Qt.Horizontal)
    window.verticalSlider1.setAccessibleName('verticalSlider1')
    window.verticalSlider1.setRange(2, 100)
    window.verticalSlider1.setSliderPosition(20)
    window.verticalSlider1.setToolTip('Size')
    window.verticalSlider2.setAccessibleName('verticalSlider2')
    window.verticalSlider2.setRange(0, 100)
    window.verticalSlider2.setSliderPosition(100)
    window.verticalSlider2.setToolTip('Opacity')
    window.verticalSlider3.setAccessibleName('verticalSlider3')
    window.verticalSlider3.setRange(0, 100)
    window.verticalSlider3.setSliderPosition(100)
    window.verticalSlider3.setToolTip('Hardness')
    window.verticalSlider4.setAccessibleName('verticalSlider4')
    window.verticalSlider4.setRange(0, 100)
    window.verticalSlider4.setSliderPosition(100)
    window.verticalSlider4.setToolTip('Flow')
    # get brush and eraser families
    window.brushes = initBrushes()
    window.brushCombo = QComboBox()
    for b in window.brushes[:-1]:  # don't add eraser to combo
        window.brushCombo.addItem(b.name, b)
    window.verticalSlider1.sliderReleased.connect(window.label.brushUpdate)
    window.verticalSlider2.sliderReleased.connect(window.label.brushUpdate)
    window.verticalSlider3.sliderReleased.connect(window.label.brushUpdate)
    window.verticalSlider4.sliderReleased.connect(window.label.brushUpdate)
    window.brushCombo.currentIndexChanged.connect(window.label.brushUpdate)
    window.colorChooser.colorSelected.connect(window.label.brushUpdate)
    # init tool bar
    toolBar.addWidget(QLabel(' Brush  '))
    for slider in [window.verticalSlider1, window.verticalSlider2, window.verticalSlider3, window.verticalSlider4]:
        toolBar.addWidget(QLabel(slider.toolTip() + '  '))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setMaximumSize(100, 15)
        toolBar.addWidget(slider)
        empty = QWidget()
        empty.setFixedHeight(30)
        empty.setFixedWidth(50)
        empty.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolBar.addWidget(empty)
    toolBar.addWidget(window.brushCombo)
    # link tooLBar to the group of tool buttons
    for button in window.drawFG.group().buttons():
        button.toolBar = toolBar
    window.addToolBar(toolBar)

    # whatsThis
    window.cropButton.setWhatsThis(
        """To crop the image drag a gray curtain on either side using the 8 small square buttons around the image""")
    window.rulerButton.setWhatsThis("""Draw horizontal and vertical rulers over the image""")
    window.fitButton.setWhatsThis("""Reset the image size to the window size""")
    window.eyeDropper.setWhatsThis("""Color picker\n Click on the image to sample pixel colors""")
    window.dragBtn.setWhatsThis(
        """Drag\n left button : drag the whole image\n Ctrl+Left button : drag the active layer only""")
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
        """)
    window.drawBG.setWhatsThis(
        """<b>Background/Mask tool</b><br>
          Paint on the active layer to mask a region or to select background pixels (segmentation layer only);
          (the mask must be enabled as opacity or color mask in the layer panel).<br>
          With <b>Color Mask</b> enabled, masked pixels are grayed and unmasked pixels are reddish.<br>
          Use the 'Brush Size' slider below to choose the size of the tool. 
        """)
    window.verticalSlider1.setWhatsThis("""Set the diameter of the painting brush""")
    window.verticalSlider2.setWhatsThis("""Set the opacity of the painting brush""")

    # Before/After views flag
    window.splitView = False

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

    #  onWidgetChange is called by all main form button and slider slots (cf. QtGui1.py)
    window.onWidgetChange = widgetChange

    # init imageLabel objects
    window.label.window = window
    window.label.splitWin = splitWin
    window.label_2.window = window
    window.label_2.splitWin = splitWin
    window.label_3.window = window
    window.label_3.splitWin = splitWin
    window.label.enterAndLeave = True
    window.label_2.enterAndLeave = False
    window.label_3.enterAndleave = False
    # drag and drop event handlers are specific for the main window
    window.label.dropEvent = MethodType(lambda instance, e, wdg=window.label: dropEvent(wdg, wdg.img, e),
                                        window.label.__class__)
    window.label.dragEnterEvent = MethodType(lambda instance, e, wdg=window.label: dragEnterEvent(wdg, wdg.img, e),
                                             window.label.__class__)
    window.label.setAcceptDrops(True)

    # recording of the (initial) current brush in label.State
    window.label.brushUpdate()

    defaultImImage = initDefaultImage()
    window.label.img = defaultImImage
    window.label_2.img = defaultImImage
    window.label_3.img = defaultImImage

    window.showMaximized()

    initCursors()

    # init Before/after view and cycling action
    window.splitter.setOrientation(Qt.Horizontal)
    window.splitter.currentState = next(splitWin.splitViews)
    window.splitter.setSizes([2 ** 20, 2 ** 20])
    window.splitter.setHandleWidth(1)
    window.splitter.hide()
    window.viewState = 'After'
    actionCycle = QAction('cycle', window)
    actionCycle.setShortcut(QKeySequence(Qt.CTRL+Qt.Key_Space))

    def f():
        window.viewState = 'Before/After'
        splitWin.nextSplitView()
        updateStatus()

    actionCycle.triggered.connect(f)
    window.addAction(actionCycle)
    actionCycle.setShortcutContext(Qt.ApplicationShortcut)

    #########################################
    # dynamic modifications of the main form loaded
    # from blue.ui
    ########################################
    setRightPane()

    ##################
    # Color Management
    ##################
    setColorManagement()

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
        Shows the edited image.<br>
        <b>Ctrl+Space</b> cycles through views.<br>
        <b>Space</b> switches back to normal view.<br>
        """
    )  # end of setWhatsThis
    window.label_2.setWhatsThis(
        """ <b>Before/After View : Before Window</b><br>
        Shows the initial (background) image.<br>
        <b>Ctrl+Space</b> cycles through views.<br>
        <b>Space</b> switches back to normal view.
        """
    )  # end of setWhatsThis

def switchDoc(index):
    """
    tabBarClicked slot
    @param index: tab index
    @type index: int
    """
    # clean up
    layer = window.label.img.getActiveLayer()
    if layer.tool is not None:
        layer.tool.hideTool()
    img = window.tabBar.tabData(index)
    setDocumentImage(img)


def closeTab(index, window=window):
    """
    tabCloseRequested Slot.
    Tries to save and close a single document.
    @param index: valid tab index
    @type index: int
    @param window:
    @type window:
    """
    closeTabs(index=index)


def setupTabBar(window=window):
    tabBar = QTabBar()
    tabBar.currentChanged.connect(switchDoc)
    tabBar.tabCloseRequested.connect(closeTab)
    tabBar.setMaximumHeight(25)
    # tabBar.setAutoHide(True)
    tabBar.setStyleSheet("QTabBar::tab {height: 15px; width: 100px;}")
    tabBar.setTabsClosable(True)
    vlay = QVBoxLayout()
    hlay2 = QHBoxLayout()
    hlay2.addWidget(tabBar)
    hlay2.addStretch(100)
    vlay.addLayout(hlay2)
    hlay1 = QHBoxLayout()
    hlay1.addWidget(window.label)
    hlay1.addWidget(window.splitter)
    vlay.addLayout(hlay1)
    hlay = window.horizontalLayout_2
    hlay.addLayout(vlay)
    window.tabBar = tabBar


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
    # display splash screen and set app style sheet
    setupGUI(window)
    setupTabBar()

    ###############
    # launch app
    ###############
    sys.exit(app.exec_())



