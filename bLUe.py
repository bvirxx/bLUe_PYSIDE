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
PySide2 is licensed under the LGPL version 2.1
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

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General
Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option)
any later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
for more details.

You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the
Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

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
import io
import os
from os import path, walk, remove
from os.path import isfile
from tempfile import mktemp
from pathlib import Path

import numpy as np
import multiprocessing
import sys
import threading
from itertools import cycle
from time import sleep
import gc
import tifffile
from ast import literal_eval
from types import MethodType
import pickle
import rawpy
from PIL import ImageCms

from PySide2.QtCore import QUrl, QFileInfo
from PySide2.QtGui import QPixmap, QCursor, QKeySequence, QDesktopServices, QFont, \
    QTransform, QColor, QImage, QIcon
from PySide2.QtWidgets import QApplication, \
    QDockWidget, QSizePolicy, QSplashScreen, QWidget, \
    QTabWidget, QToolBar, QComboBox, QTabBar, QAction

import bLUeTop.QtGui1

from bLUeGui.dialog import *
from bLUeGui.colorPatterns import cmHSP, cmHSB
from bLUeGui.graphicsForm import baseGraphicsForm
from bLUeGui.tool import cropTool, rotatingTool
from bLUeCore.bLUeLUT3D import HaldArray
from bLUeTop import exiftool, Gui
from bLUeTop.drawing import initBrushes, loadPresets
from bLUeTop.graphicsDraw import drawForm
from bLUeTop.graphicsHDRMerge import HDRMergeForm
from bLUeTop.graphicsSegment import segmentForm
from bLUeTop.graphicsBlendFilter import blendFilterForm
from bLUeTop.graphicsHVLUT2D import HVLUT2DForm
from bLUeTop.graphicsInvert import invertForm
from bLUeTop.graphicsMixer import mixerForm
from bLUeTop.graphicsNoise import noiseForm
from bLUeTop.graphicsRaw import rawForm
from bLUeTop.graphicsTransform import transForm, imageForm
from bLUeGui.bLUeImage import QImageBuffer, QImageFormats, ndarrayToQImage
from bLUeTop.presetReader import aParser
from bLUeTop.rawProcessing import rawRead
from bLUeTop.versatileImg import vImage, metadataBag
from bLUeTop.MarkedImg import imImage, QRawLayer, QCloningLayer, QLayerImage, QDrawingLayer
from bLUeTop.graphicsRGBLUT import graphicsForm
from bLUeTop.graphicsLUT3D import graphicsForm3DLUT
from bLUeTop.graphicsAutoLUT3D import graphicsFormAuto3DLUT
from bLUeTop.lutUtils import LUTSIZE, LUT3D, LUT3DIdentity
from bLUeTop.colorManagement import icc
from bLUeTop.graphicsCoBrSat import CoBrSatForm
from bLUeTop.graphicsExp import ExpForm
from bLUeTop.graphicsPatch import patchForm
from bLUeTop.settings import USE_POOL, POOL_SIZE, THEME, TABBING, BRUSHES_PATH, COLOR_MANAGE_OPT, HAS_TORCH
from bLUeTop.utils import UDict, stateAwareQDockWidget, QImageFromFile, imagej_description_metadata, compat
from bLUeTop.graphicsTemp import temperatureForm
from bLUeTop.graphicsFilter import filterForm
from bLUeTop.graphicsHspbLUT import graphicsHspbForm
from bLUeTop.graphicsLabLUT import graphicsLabForm
from bLUeTop.viewer import playDiaporama, viewer

from version import BLUE_VERSION

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
rawpy Copyright (C) 2014 Maik Riechert
seamlessClone and CLAHE are Opencv functions
mergeMertens is an Opencv class
grabCut is a parallel version of an Opencv function
Pretrained classifier (C) Hui Zeng, Jianrui Cai, Lida Li, Zisheng Cao, and Lei Zhang
This product includes DNG technology under license by Adobe Systems Incorporated
credit https://icones8.fr/
"""

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


def widgetChange(button, window=bLUeTop.Gui.window):
    """
    called by all main form button and slider slots (cf. QtGui1.py onWidgetChange)

    :param button:
    :type  button: QWidget
    :param window:
    :type  window: QWidget
    """
    # wdgName = button.objectName()
    if button is window.fitButton:  # wdgName == "fitButton" :
        window.label.img.fit_window()
        # update crop button positions
        window.cropTool.setCropTool(window.label.img)
    elif button is window.cropButton:
        if button.isChecked():
            window.cropTool.setCropTool(window.label.img)
            for b in window.cropTool.btnDict.values():
                b.show()
        else:
            for b in window.cropTool.btnDict.values():
                b.hide()
        window.label.img.isCropped = button.isChecked()
    elif button is window.asButton:
        # update crop tool
        window.cropTool.setCropTool(window.label.img)
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


def addAdjustmentLayers(layers, images):
    """
    Adds a list of layers to the current document and restore their states.
    Entries not corresponding to menu layers actions are skipped.

    :param layers:
    :type  layers: list of (key, dict)
    :param images
    :type  images TiffPageSeries
    """
    if layers is None:
        return
    count = 1

    waitImages = []
    for ind, item in enumerate(layers):
        d = item[1]['state']
        # add layer to stack
        layer = menuLayer(item[1]['actionname'], sname=item[0], script=True)
        if d['mask'] == 1:
            if layer is not None:
                buf = images.asarray()[count]
                buf = buf.reshape(layer.height(), layer.width(), 4)
                layer.mask = ndarrayToQImage(buf, QImage.Format_ARGB32)
                layer.__setstate__(item[1])  # keep after mask init
            count += 1
        else:
            if layer is not None:
                layer.__setstate__(item[1])  # keep after mask init
        if 'images' in d:  # for retro compatibility with previous bLU file formats
            n = d['images']
            if type(n) is tuple:
                n = n.count(1)
            if n > 0:
                waitImages.append((layer, n))  # image offset not known yet

    for layer, n in waitImages:
        t = type(layer)
        # all layers managing images should be subclasses of QLayer.
        # to handle multiple images n should be a tuple of binary values
        if t in [QLayerImage, QDrawingLayer, QCloningLayer]:
            buf = images.asarray()[count]
            buf = buf.reshape(layer.height(), layer.width(), 4)
            layer.sourceImg = ndarrayToQImage(buf, QImage.Format_ARGB32)
            if t is QCloningLayer:
                layer.getGraphicsForm().updateSource()
            layer.applyToStack()  # needed because images are loaded after all calls to __setstate__()
            # parentImage = layer.parentImage  # same for all layers
        count += n


def addBasicAdjustmentLayers(img, window=bLUeTop.Gui.window):
    """
    Adds any default adjustment layers to layer stack.

    :param img:
    :type img: mImage
    :param window:
    :type window:
    """
    if img.rawImage is None:
        pass
        # menuLayer('actionColor_Temperature')
        # menuLayer('actionExposure_Correction')
        # menuLayer('actionContrast_Correction')
    # select active layer : top row
    window.tableView.select(0, 1)


def addRawAdjustmentLayer(window=bLUeTop.Gui.window):
    """
    Add a development layer to the layer stack
    """
    rlayer = window.label.img.addAdjustmentLayer(layerType=QRawLayer, name='Develop', role='RAW')
    grWindow = rawForm.getNewWindow(axeSize=axeSize, targetImage=window.label.img, layer=rlayer, parent=window)
    # wrapper for the right apply method
    pool = getPool()
    rlayer.execute = lambda l=rlayer, pool=pool: l.tLayer.applyRawPostProcessing(pool=pool)
    # record action name for scripting
    rlayer.actionName = 'Develop'  # not a menu action !
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
    return rlayer


def loadImage(img, tfile=None, version='unknown', withBasic=True, window=bLUeTop.Gui.window):
    """
    load a vImage into bLUe and build layer stack.
    if tfile is an opened TiffFile instance, import layer stack from file

    :param img:
    :type  img: vImage
    :param tfile:
    :type  tfile: TiffFile instance
    :param version: version of bLU file writer
    :type  version: str
    :param withBasic:
    :type  withBasic: boolean
    :param window:
    :type  window: QWidget
    """

    tabBar = window.tabBar
    ind = tabBar.addTab(basename(img.filename))
    tabBar.setCurrentIndex(ind)
    tabBar.setTabData(ind, img)
    setDocumentImage(img)

    window.tableView.previewOptionBox.setChecked(True)

    # add development layer for raw image, and develop
    rlayer = None
    if img.rawImage is not None:
        rlayer = addRawAdjustmentLayer()

    fromBlue = tfile is not None and img.filename[-4:].upper() in BLUE_FILE_EXTENSIONS
    # import layer stack from .BLU file
    if fromBlue:
        # get ordered dict of layers.
        # Tifffile relies on Python 3 formatting for bytes --> str decoding.
        # It implicitly calls __str__() to get "standard" string representation of bytes
        # (cf. the function imagej_description() in tifffile.py)
        # This last representation is then encoded/decoded to and from the tiff file.
        # str representations of tag values may contain arbitrary '=' char, so we need
        # our modified version of imagej_description_metadata() to get the dict.
        # Next, str representation of tag values are decoded to bytes by applying literal_eval
        # as inverse of __str__(), and finally tag values are unpickled.

        # The Tifffile/ImageJ format has no clear specification. Data can be retrieved from the imagej_description
        # or description attributes, depending on the tifffile version. The attribute imagej_description
        # does not exist in older versions so we use the second.

        meta_dict = imagej_description_metadata(tfile.pages[0].description)

        try:
            if rlayer is not None:
                d = pickle.loads(literal_eval(compat(meta_dict['develop'], version)))  # keys are turned to lower !
                rlayer.__setstate__(d)
            # import layer stack
            # Every Qlayer state dict is pickled. Thus, to import layer stack, unpickled entries may be skipped safely.
            # This enables the cancellation of spurious entries added by tifffile/ImageJ protocol.
            withBasic = False  # the imported layer stack only
            layers = []
            for key in meta_dict:
                # unpickled values raise exceptions that are all skipped. The corresponding keys
                # can be 'version' (here renamed 'imageJ' by tifffile), 'sourceformat' and some keys set
                # by tifffile: 'hyperstack' and 'images' (don't confuse with the key
                # 'images' in QLayer's dict).
                try:
                    v = meta_dict[key]
                    if type(v) is str:
                        # possibly pickled string. Try conversion to the
                        # right bLUe version and unpickle.
                        v = compat(v, version)
                    d = pickle.loads(literal_eval(v))
                    if key == 'cropmargins' and d != (0.0, 0.0, 0.0, 0.0):
                        img.setCropMargins(d, window.cropTool)  # type(d) is tuple
                        window.cropButton.setChecked(Qt.Checked)
                    elif type(d) is dict:
                        layers.append((key, d))
                except (SyntaxError, ValueError, pickle.UnpicklingError):
                    continue
        except (SyntaxError, ValueError, ModuleNotFoundError, pickle.UnpicklingError) as e:
            # exceptions raised while unpickling meta_dict['develop'] cannot be
            # skipped.
            # dlgWarn('loadImage: Invalid format %s' % img.filename, str(e))
            raise
        # load layers
        addAdjustmentLayers(layers, tfile.series[0])
        img.setActiveLayer(len(img.layersStack) - 1)
        # img.onImageChanged()
    else:
        # add default adjustment layers
        if withBasic:
            addBasicAdjustmentLayers(img)
        # updates
        # for bLU file updates are already done by __setstate__() methods of graphic forms
        img.layersStack[0].applyToStack()

    img.onImageChanged()


def openFile(f, window=bLUeTop.Gui.window):
    """
    Top level function for file opening, used by File Menu actions

    :param f: file name
    :type  f: str
    :param window:
    :type  window: QWidget
    """
    iobuf = None
    sourceformat = path.basename(f)[-4:].upper()
    tfile = None
    version = 'unknown'
    try:
        window.status_loadingFile = True
        window.tabBar.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        updateStatus()
        QApplication.processEvents()

        if sourceformat in BLUE_FILE_EXTENSIONS:
            tfile = tifffile.TiffFile(f)
            meta_dict = tfile.imagej_metadata  # no unpickling needed here, so we use imagej_metadata
            version = meta_dict.get('version', 'unknown')
            # get recorded source format
            sourceformat = meta_dict.get('sourceformat')  # read actual source format
            if sourceformat in RAW_FILE_EXTENSIONS:
                # is .blu file from raw
                buf_ori_len = meta_dict['buf_ori_len']
                rawbuf = tfile.series[0].pages[0].asarray()[0, :buf_ori_len]
                iobuf = io.BytesIO(rawbuf.tobytes())

        ##############################################################
        # load imImage instance from file. If rawiobuf is None the
        # file is read using QImageReader, bLU file included (tif file).
        ##############################################################
        img = imImage.loadImageFromFile(f, rawiobuf=iobuf, cmsConfigure=True, window=window)
        QApplication.processEvents()
        img.sourceformat = sourceformat
        ###########################
        # init or load layer stack
        ##########################
        if img is not None:
            window.status_loadingLayers = True
            loadImage(img, tfile=tfile, version=version)
            updateStatus()
            QApplication.processEvents()
            # update the list of recent files
            recentFiles = window.settings.value('paths/recent', [])
            # settings.values returns a str or a list of str,
            # depending on the count of items. May be a PySide2 bug
            # in QVariant conversion.
            if type(recentFiles) is str:
                recentFiles = [recentFiles]
            recentFiles = list(filter(lambda a: a != f, recentFiles))
            recentFiles.insert(0, f)
            if len(recentFiles) > 10:
                recentFiles.pop()
            window.settings.setValue('paths/recent', recentFiles)
    except (ValueError, KeyError, IOError, rawpy.LibRawFatalError, SyntaxError,
            ModuleNotFoundError, pickle.UnpicklingError) as e:
        dlgWarn(repr(e))
    finally:
        if tfile is not None:
            tfile.close()
        window.tabBar.setEnabled(True)
        window.status_loadingFile = False
        window.status_loadingLayers = False
        QApplication.restoreOverrideCursor()
        QApplication.processEvents()


def saveFile(filename, img, quality=-1, compression=-1, writeMeta=True):
    """
    Save image and meta data to file

    :param filename:
    :type  filename:
    :param img:
    :type  img:
    :param quality:
    :type  quality:
    :param compression:
    :type  compression:
    :param writeMeta:
    :type  writeMeta:
    """
    if isfile(filename):
        reply = QMessageBox()
        reply.setWindowTitle('Warning')
        reply.setIcon(QMessageBox.Warning)
        reply.setText("File %s already exists\n" % filename)
        reply.setStandardButtons(QMessageBox.Cancel)
        accButton = QPushButton("Save as New Copy")
        rejButton = QPushButton("OverWrite")
        if img.profileChanged:
            rejButton.setEnabled(False)
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
    try:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        thumb = img.save(filename, quality=quality, compression=compression)
        # write metadata
        if writeMeta:
            if thumb is None:
                img.restoreMeta(img.filename, filename)
            else:
                tempFilename = mktemp('.jpg')
                # save thumb jpg to temp file
                thumb.save(tempFilename)
                # copy temp file to image file, img.filename not updated yet
                img.restoreMeta(img.filename, filename, thumbfile=tempFilename)
                remove(tempFilename)
    finally:
        QApplication.restoreOverrideCursor()
        QApplication.processEvents()
    return filename


def closeTabs(index=None, window=bLUeTop.Gui.window):
    """
    Tries to save and close the document open in tab index,
    or all open documents if index is None . If the method  succeeds in
    closing all open documents, it resets the GUI to default.

    :param index:
    :type index: int
    :param window:
    :type window:
    """

    if not canClose(index=index) or window.tabBar.count() > 0:
        gc.collect()
        return
    # window.tableView.clear(delete=True)
    window.histView.targetImage = None
    defaultImImage = initDefaultImage()
    window.label.img = defaultImImage
    window.label_2.img = defaultImImage
    window.label_3.img = defaultImImage
    window.tableView.clear(delete=True)  # 30/11/21
    gc.collect()
    window.label.update()
    window.label_2.update()
    window.label_3.update()


def showHistogram(window=bLUeTop.Gui.window):
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
        window.histView.chans = [['R', 'G', 'B'].index(ch) for ch in ['R', 'G', 'B'] if window.histView.options[ch]]
    else:
        window.histView.mode = 'Luminosity'
        window.histView.chanColors = [Qt.gray]
        window.histView.chans = []
    histView = histImg.histogram(QSize(window.histView.width(), window.histView.height()),
                                 chans=window.histView.chans, bgColor=Qt.black,
                                 chanColors=window.histView.chanColors, mode=window.histView.mode,
                                 addMode='Luminosity' if window.histView.options['L'] else '')
    window.histView.cache = QPixmap.fromImage(histView)
    window.histView.Label_Hist.setPixmap(window.histView.cache)
    window.histView.Label_Hist.drawingWidth = histView.drawingWidth
    window.histView.Label_Hist.drawingScale = histView.drawingScale


def restoreBrush(layer, window=bLUeTop.Gui.window):
    """
    Sync brush tools with brushDict

    :param layer:
    :type  layer:
    """
    d = layer.brushDict
    if d is None:
        return
    window.verticalSlider1.setValue(d['size'])
    window.verticalSlider2.setValue(int(d['opacity'] * 100.0))
    window.verticalSlider3.setValue(int(d['hardness'] * 100.0))
    window.verticalSlider4.setValue(int(d['flow'] * 100.0))
    window.colorChooser.setCurrentColor(d['color'])
    graphicsForm = layer.getGraphicsForm()
    # graphicsForm may be None if the layer is being created
    if graphicsForm is not None:
        graphicsForm.spacingSlider.setValue(int(d['spacing']) * 10)
        graphicsForm.jitterSlider.setValue(int(d['jitter']) * 10)
        graphicsForm.orientationSlider.setValue(int(d['orientation']) + 180)
    ind = window.brushCombo.findText(d['name'])
    if ind != -1:
        window.brushCombo.setCurrentIndex(ind)  # trigger brushUpdate() - keep last
    window.label.State['brush'] = d.copy()


def setDocumentImage(img, window=bLUeTop.Gui.window):
    """
    Inits GUI and displays the current document

    :param img: image
    :type  img: imImage
    :param window:
    :type  window: QWidget
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
    else:  # reset to default
        for k in window.btnValues:
            window.btnValues[k] = False
        window.btnValues['pointer'] = True  # default checked autoexclusive button (needed)
    window.label.img = img
    # update img in current tab, if it was recreated (rotation, resizing,...)
    ind = window.tabBar.currentIndex()
    if window.tabBar.tabData(ind) is not img:
        window.tabBar.setTabText(ind, basename(img.filename))
        window.tabBar.setTabData(ind, img)
    window.cropTool.fit(img)
    window.cropTool.setCropTool(img)
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
            if layer.brushDict is None:  # no brush set yet
                window.label.brushUpdate()
                layer.brushDict = window.label.State['brush']
            restoreBrush(layer)

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
    # color management settings may have changed : update presentation layers
    updateCurrentViews()
    window.label_3.update()
    updateStatus()
    gc.collect()  # tested : (very) efficient here


def updateMenuOpenRecent(window=bLUeTop.Gui.window):
    """
    Update the list of recent files displayed
    in the QMenu menuOpen_recent, and init
    the corresponding actions
    """
    window.menuOpen_recent.clear()
    recentFiles = window.settings.value('paths/recent', [])
    # settings.values returns a str or a list of str,
    # depending on the count of items. May be a PySide2 bug
    # in QVariant conversion.
    if type(recentFiles) is str:
        recentFiles = [recentFiles]
    for filename in recentFiles:
        window.menuOpen_recent.addAction(filename, lambda x=filename: openFile(x))


def updateMenuLoadPreset(window=bLUeTop.Gui.window):
    """
    Menu aboutToShow handler
    """

    def f(filename):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            brushes, patterns = loadPresets(filename, first=window.brushCombo.count() + 1)
            window.brushes.extend(brushes)
            for b in brushes:
                if b.preset is None:
                    window.brushCombo.addItem(b.name, b)
                else:
                    window.brushCombo.addItem(QIcon(b.preset), b.name, b)
            for p in patterns:
                window.patternCombo.addItem(QIcon(p.pxmp), p.name, p)
        finally:
            QApplication.restoreOverrideCursor()

    window.menuLoad_Preset.clear()
    for entry in os.scandir(BRUSHES_PATH):
        if entry.is_file():
            ext = entry.name[-4:].lower()
            if ext in ['.png', '.jpg', '.abr']:
                filename = os.getcwd() + '\\' + BRUSHES_PATH + '\\' + entry.name
                # filter .abr versions
                if ext == '.abr':
                    v = aParser.getVersion(filename)
                    if v != '6.2':
                        continue
                # add file to sub menu
                window.menuLoad_Preset.addAction(entry.name, lambda x=filename: f(x))


def updateEnabledActions(window=bLUeTop.Gui.window):
    """
    Menu aboutToShow handler
    """
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    window.actionSave.setEnabled(window.label.img.isModified)
    window.actionSave_As.setEnabled(window.label.img.isModified)
    window.actionSave_As_bLU_Doc.setEnabled(window.label.img.isModified)
    window.actionSave_Hald_Cube.setEnabled(window.label.img.isHald)
    window.actionAuto_3D_LUT.setEnabled(HAS_TORCH)


def menuFile(name, window=bLUeTop.Gui.window):
    """
    Menu handler

    :param name: action name
    :type  name: str
    :param window:
    :type  window: QWidget
    """
    # new document
    if name == 'actionNew_2':
        # get image from clipboard, if any
        cb = QApplication.clipboard()
        cbImg = cb.image()
        w, h = (200,) * 2
        if not cbImg.isNull():
            w, h = cbImg.width(), cbImg.height()
            cb.clear()
        dlg = dimsInputDialog(w, h)

        def f():
            if cbImg.isNull():
                # new image
                imgNew = QImage(dlg.dims['w'], dlg.dims['h'], QImage.Format_ARGB32)
                # set background color for the new document
                imgNew.fill(Qt.white)
                img = imImage(QImg=imgNew)
            else:
                # paste
                if w == dlg.dims['w'] and h == dlg.dims['h']:
                    img = imImage(QImg=cbImg)
                else:
                    img = imImage(QImg=cbImg.scaled(dlg.dims['w'], dlg.dims['h']))
            img.filename = 'unnamed'
            loadImage(img, withBasic=False)  # don't add any adjustment layer

        dlg.onAccept = f
        dlg.exec_()

    # load image from file
    elif name in ['actionOpen']:
        # get file name from dialog
        filename = openDlg(window, ask=False)
        # open file
        if filename is not None:
            openFile(filename)

    # saving dialog
    elif name in ['actionSave', 'actionSave_As', 'actionSave_As_bLU_Doc']:
        saveAsBLU = (name == 'actionSave_As_bLU_Doc')
        saveAs = (name == 'actionSave_As') or saveAsBLU
        if window.label.img.useThumb and not saveAsBLU:
            dlgWarn("Uncheck Preview mode before saving")
        else:
            if saveAsBLU and not window.label.img.useThumb:
                dlgWarn("for better performances switch to Preview mode before saving a bLU document")
            img = window.label.img
            try:
                if saveAs:
                    ext = 'blu' if saveAsBLU else 'jpg'
                    filename, quality, compression, writeMeta = saveDlg(img,
                                                                        window,
                                                                        ext=ext,
                                                                        selected=True
                                                                        )
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
    updateStatus()


def menuView(name, window=bLUeTop.Gui.window):
    """
    Menu handler

    :param name: action name
    :type  name: str
    :param window:
    :type  window: QWidget
    """
    ##################
    # before/after mode
    ##################
    if name == 'actionShow_hide_right_window_3':
        if window.splitter.isHidden():
            bLUeTop.Gui.splitWin.setSplitView()
            window.viewState = 'Before/After'
        else:
            window.splitter.hide()
            window.label.show()
            window.splitView = False
            window.viewState = 'After'
            if window.btnValues['Crop_Button']:
                window.cropTool.setCropTool(window.label.img)
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
            # dlg.setNameFilters(IMAGE_FILE_NAME_FILTER)
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
    ############
    # Histogram
    ############
    elif name == 'actionHistogram':
        window.histViewDock.show()
    updateStatus()


def menuImage(name, window=bLUeTop.Gui.window):
    """
    Menu handler

    :param name: action name
    :type  name: str
    :param window:
    :type  window: QWidget
    """
    img = window.label.img

    # display image info
    if name == 'actionImage_info':
        # Format
        s = "Format : %s\n(cf. QImage formats in the doc for more info)" % QImageFormats.get(img.format(), 'unknown')
        # dimensions
        s = s + "\n\ndim : %d x %d" % (img.width(), img.height())
        # profile info
        if img.meta.profile is not None and len(img.meta.profile) > 0:
            s = s + "\n\nEmbedded profile found"  # length %d" % len(img.meta.profile)
        workingProfileInfo = icc.workingProfileInfo
        s = s + "\n\nWorking Profile : %s" % workingProfileInfo
        # rating
        s = s + "\n\nRating %s" % ''.join(['*'] * img.meta.rating)
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

    elif name == 'actionSoft_proofing':
        proofingOn = window.actionSoft_proofing.isChecked()
        if proofingOn:
            lastDir = str(window.settings.value('paths/profdlgdir', '.'))
            filter = "Profiles ( *" + " *".join(['.icc', '.icm']) + ")"
            dlg = QFileDialog(window, "Select", lastDir, filter)
            try:
                filenames = []
                if dlg.exec_():
                    filenames = dlg.selectedFiles()
                    newDir = dlg.directory().absolutePath()
                    window.settings.setValue('paths/profdlgdir', newDir)
                    icc.configure(qscreen=window.currentScreenIndex,
                                  workingProfile=icc.workingProfile,
                                  softproofingwp=ImageCms.getOpenProfile(filenames[0])
                                  )
                else:
                    raise ImageCms.PyCMSError
            except ImageCms.PyCMSError:
                dlgWarn('Invalid Profile', filenames[0] if filenames else 'No file selected')
                window.actionSoft_proofing.setChecked(False)
        else:
            icc.configure(qscreen=window.currentScreenIndex, workingProfile=icc.workingProfile, softproofingwp=None)
        updateCurrentViews()
        updateStatus()

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

    # force current display profile re-detection
    elif name == 'actionUpdate_display_profile':
        icc.configure(qscreen=window.currentScreenIndex,
                      workingProfile=icc.workingProfile,
                      softproofingwp=icc.softProofingProfile
                      )
        updateCurrentViews()

    # show info for display and working profiles
    elif name == 'actionWorking_profile':
        w = labelDlg(parent=window, title='profile info')
        s = 'Working Profile : '
        if icc.workingProfile is not None:
            s = s + icc.workingProfileInfo
        s = s + '-------------\n' + 'Monitor Profile : '
        if icc.monitorProfile is None:
            if not COLOR_MANAGE_OPT:
                s = s + 'Color Management is disabled : check your config.json file\n\n'
            else:
                s = s + 'Automatic detection failed and no default profile was found\n\n'
            s = s + 'Define SYSTEM_PROFILE_DIR and DEFAULT_MONITOR_PROFILE_NAME in ' \
                    'the configuration file config.json ' \
                    'to match the path to your current display profile.\n' \
                    'Usual Profile dirs are on Linux ~/.local/share/icc\n' \
                    'and on Windows C:\Windows\System32\spool\drivers\color\n'
        else:
            s = s + icc.monitorProfileInfo + '-------------\n\n'

        s = s + 'Note :\nThe working profile is the color profile assigned to the image.' \
                'The monitor profile should correspond to your monitor. ' \
                'Both profiles are used in conjunction to display exact colors. ' \
                'If one of them is missing, bLUe cannot color manage the image. ' \
                'If the monitor profile listed above is not the right profile for your monitor, ' \
                'please check the system settings for color management.'
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
            # copy info strings
            tImg.filename = img.filename
            tImg.imageInfo = img.imageInfo
            setDocumentImage(tImg)
            tImg.layersStack[0].applyToStack()
            tImg.onImageChanged()
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()

    # resize
    elif name == 'actionImage_Resizing':
        w, h = img.width(), img.height()
        dlg = dimsInputDialog(w, h, keepBox=True)
        dlg.open()

        def f():
            img = window.label.img.resized(dlg.dims['w'], dlg.dims['h'], keepAspectRatio=dlg.dims['kr'])
            img.filename = 'unnamed'
            setDocumentImage(img)
            img.layersStack[0].applyToStack()

        dlg.onAccept = f

    # rating
    elif name in ['action0', 'action1', 'action2', 'action3', 'action4', 'action5']:
        img.meta.rating = int(name[-1:])
        updateStatus()
        with exiftool.ExifTool() as e:
            e.writeXMPTag(img.meta.filename, 'XMP:rating', img.meta.rating)

    elif name in ['actionImage_Profile', 'actionConvert_To_Profile']:
        lastDir = str(window.settings.value('paths/profdlgdir', '.'))
        filter = "Profiles ( *" + " *".join(['.icc', '.icm']) + ")"
        dlg = QFileDialog(window, "Select", lastDir, filter)
        try:
            filenames = []
            if dlg.exec_():
                filenames = dlg.selectedFiles()
                newDir = dlg.directory().absolutePath()
                window.settings.setValue('paths/profdlgdir', newDir)
                profile = ImageCms.getOpenProfile(filenames[0])
                oldprofile = img.cmsProfile
                img.setProfile(profile)

                icc.configure(qscreen=window.currentScreenIndex, workingProfile=profile)

                # update sidecar
                with exiftool.ExifTool() as e:
                    tmp = Path(img.filename).with_suffix('.mie')
                    e.writeProfile(str(tmp), filenames[0])

                img.profileChanged = True  # prevent overwriting image file

                if name == 'actionConvert_To_Profile':
                    transform = ImageCms.buildTransformFromOpenProfiles(oldprofile,
                                                                        profile,
                                                                        "RGB",
                                                                        "RGB",
                                                                        renderingIntent=ImageCms.Intent.PERCEPTUAL
                                                                        )
                    icc.convertQImage(img, transform, inPlace=True)
                    # update background layer
                    QImageBuffer(img.layersStack[0])[...] = QImageBuffer(img)
                    img.layersStack[0].thumb = None
                    img.layersStack[0].applyToStack()

                updateCurrentViews()
            else:
                raise ValueError

        except (ImageCms.PyCMSError, ValueError):
            dlgWarn('Invalid Profile', filenames[0] if filenames else 'No file selected')


def updateCurrentViews(window=bLUeTop.Gui.window):
    """
    Sync the presentation layers of current document views (before/after)
    with the state of color management.
    :param window:
    :type window:
    """
    window.label.img.updatePixmap()
    window.label.update()
    window.label_2.img.updatePixmap()
    window.label_2.update()


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


def menuLayer(name, window=bLUeTop.Gui.window, sname=None, script=False):
    """
    Menu Layer handler and scripting.
    Creates a layer and its associated graphic form.
    Returns the newly created layer or None.

    :param window:
    :type window:
    :param name: action name
    :type  name: str
    :param sname: layer name from script
    :type  sname: str
    :param script:
    :type  script: boolean
    :return: layer
    :rtype: Union[QLayer, None]
    """

    # adhoc dict for getNewWindow() calls
    def envdict():
        return {'targetImage': window.label.img,
                'layer': layer,
                'parent': window,
                'mainForm': window}

    # get layer instance name with priority to script if any
    def gn(name):
        return name if sname is None else sname

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
        # enhance graphic scene display
        if isinstance(grWindow, baseGraphicsForm):
            grWindow.fitInView(grWindow.scene().sceneRect(), Qt.KeepAspectRatio)
        if isinstance(grWindow, graphicsFormAuto3DLUT):
            # apply auto 3D LUT immediately when the layer is added
            grWindow.dataChanged.emit()

    # check open document
    if window.tabBar.count() == 0:
        dlgWarn('Cannot add layer : no document found', 'Open an existing image or create a new one')
        return

    if name in ['actionCurves_RGB', 'actionCurves_HSpB', 'actionCurves_Lab']:
        if name == 'actionCurves_RGB':
            layerName = 'Curves RGB'
            form = graphicsForm
        elif name == 'actionCurves_HSpB':
            layerName = 'Curves HSV'
            form = graphicsHspbForm
        elif name == 'actionCurves_Lab':
            layerName = 'Curves Lab'
            form = graphicsLabForm
        layer = window.label.img.addAdjustmentLayer(name=gn(layerName))
        grWindow = form.getNewWindow(axeSize=axeSize, **envdict())
        # wrapper for the right applyXXX method
        if name == 'actionCurves_RGB':
            layer.execute = lambda l=layer, pool=None: l.tLayer.apply1DLUT(grWindow.scene().cubicItem.getStackedLUTXY())
        elif name == 'actionCurves_HSpB':  # displayed as HSV in the layer menu !!
            layer.execute = lambda l=layer, pool=None: l.tLayer.applyHSV1DLUT(
                grWindow.scene().cubicItem.getStackedLUTXY(), pool=pool)
        elif name == 'actionCurves_Lab':
            layer.execute = lambda l=layer, pool=None: l.tLayer.applyLab1DLUT(
                grWindow.scene().cubicItem.getStackedLUTXY())

    elif name == 'actionAuto_3D_LUT' and HAS_TORCH:
        layerName = 'Auto 3D LUT'
        layer = window.label.img.addAdjustmentLayer(name=gn(layerName),
                                                    role='AutoLUT')  # do not use a role containing '3DLUT'
        grWindow = graphicsFormAuto3DLUT.getNewWindow(axeSize=300, LUTSize=LUTSIZE, **envdict())
        pool = getPool()
        layer.execute = lambda l=layer, pool=pool: l.tLayer.applyAuto3DLUT(pool=pool)

    # 3D LUT
    elif name in ['action3D_LUT', 'action3D_LUT_HSB']:
        # color model
        ccm = cmHSP if name == 'action3D_LUT' else cmHSB
        layerName = '2.5D LUT HSpB' if name == 'action3D_LUT' else '2.5D LUT HSV'
        layer = window.label.img.addAdjustmentLayer(name=gn(layerName), role='3DLUT')
        grWindow = graphicsForm3DLUT.getNewWindow(ccm, axeSize=300, LUTSize=LUTSIZE, **envdict())
        # init pool only once
        pool = getPool()
        sc = grWindow.scene()
        layer.execute = lambda l=layer, pool=pool: l.tLayer.apply3DLUT(sc.lut,
                                                                       options=sc.options,
                                                                       pool=pool
                                                                       )
    elif name == 'action2D_LUT_HV':
        layerName = '3D LUT HV Shift'
        if sname is not None:
            layerName = sname
        layer = window.label.img.addAdjustmentLayer(name=gn(layerName), role='2DLUT')
        grWindow = HVLUT2DForm.getNewWindow(axeSize=300, **envdict())
        # init pool only once
        pool = getPool()
        sc = grWindow.scene()
        layer.execute = lambda l=layer, pool=pool: l.tLayer.applyHVLUT2D(grWindow.LUT, options=sc.options, pool=pool)

    # cloning
    elif name == 'actionNew_Cloning_Layer':
        lname = 'Cloning'
        layer = window.label.img.addAdjustmentLayer(layerType=QCloningLayer, name=gn(lname), role='CLONING')
        grWindow = patchForm.getNewWindow(**envdict())
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyCloning(seamless=True)

    # segmentation
    elif name == 'actionNew_segmentation_layer':
        lname = 'Segmentation'
        layer = window.label.img.addSegmentationLayer(name=gn(lname))
        grWindow = segmentForm.getNewWindow(**envdict())
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyGrabcut(nbIter=grWindow.nbIter)

    # load images from file
    elif name == 'actionLoad_Image_from_File':
        if script:
            filenames = ['nofiles']
        else:
            filenames = openDlg(window, ask=False, multiple=True, key='paths/dlgimdir')
        if not filenames:
            return
        for filename in filenames:
            # load image from file, alpha channel is mandatory for applyTransform()
            ext = filename[-4:]
            if ext in list(IMAGE_FILE_EXTENSIONS) + list(SVG_FILE_EXTENSIONS):
                imgNew = QImageFromFile(filename)
            elif ext in list(RAW_FILE_EXTENSIONS):
                # get a RawPy instance from raw file
                rawpyInst = rawRead(filename)
                # postprocess raw image, applying default settings (cf. vImage.applyRawPostProcessing)
                rawBuf = rawpyInst.postprocess(use_camera_wb=True)
                # build Qimage : switch to BGR and add alpha channel
                rawBuf = np.dstack((rawBuf[:, :, ::-1], np.zeros(rawBuf.shape[:2], dtype=np.uint8) + 255))
                imgNew = vImage(cv2Img=rawBuf)
                # keeping a reference to rawBuf along with img is
                # needed to protect the buffer from garbage collector
                imgNew.rawBuf = rawBuf
            else:
                imgNew = QImage(1000, 1000, QImage.Format_ARGB32)  # will be resized later
                imgNew.fill(QColor(0, 0, 0, 0))
            if imgNew.isNull():
                dlgWarn("Cannot load %s: " % filename)
                return
            lname = path.basename(filename)
            layer = window.label.img.addAdjustmentLayer(name=gn(lname), sourceImg=imgNew, role='Image+GEOM')
            grWindow = imageForm.getNewWindow(axeSize=axeSize, **envdict())
            # add transformation tool to parent widget
            tool = rotatingTool(parent=window.label)  # , layer=l, form=grWindow)
            layer.addTool(tool)
            tool.showTool()
            layer.execute = lambda l=layer, pool=None: l.tLayer.applyImage(grWindow.options)
            layer.actioname = name
            layer.filename = filename
            post(layer)
        return layer if script else None  # exactly one layer added if script

    # empty new image
    elif name == 'actionNew_Layer':
        processedImg = window.label.img
        w, h = processedImg.width(), processedImg.height()
        imgNew = QImage(w, h, QImage.Format_ARGB32)
        imgNew.fill(Qt.black)
        lname = 'Image'
        layer = window.label.img.addAdjustmentLayer(name=gn(lname), sourceImg=imgNew, role='GEOMETRY')
        grWindow = imageForm.getNewWindow(axeSize=axeSize, **envdict())
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
        # imgNew.fill(Qt.white)
        imgNew.fill(QColor(0, 0, 0, 0))
        lname = 'Drawing'
        layer = window.label.img.addAdjustmentLayer(name=gn(lname), layerType=QDrawingLayer, sourceImg=imgNew,
                                                    role='DRW')
        grWindow = drawForm.getNewWindow(axeSize=axeSize, **envdict())
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyNone()
        layer.actioname = name

    # Color filter
    elif name == 'actionColor_Temperature':
        lname = 'Color Filter'
        layer = window.label.img.addAdjustmentLayer(name=gn(lname))
        grWindow = temperatureForm.getNewWindow(axeSize=axeSize, **envdict())
        # wrapper for the right apply method
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyTemperature()

    elif name == 'actionContrast_Correction':
        layer = window.label.img.addAdjustmentLayer(name=gn(CoBrSatForm.layerTitle), role='CONTRAST')
        grWindow = CoBrSatForm.getNewWindow(axeSize=axeSize, **envdict())

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
        layer = window.label.img.addAdjustmentLayer(name=gn(lname))
        layer.clipLimit = ExpForm.defaultExpCorrection
        grWindow = ExpForm.getNewWindow(axeSize=axeSize, **envdict())
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyExposure(grWindow.options)

    elif name == 'actionHDR_Merge':
        lname = 'Merge'
        layer = window.label.img.addAdjustmentLayer(name=gn(lname), role='MERGING')
        layer.clipLimit = ExpForm.defaultExpCorrection
        grWindow = HDRMergeForm.getNewWindow(axeSize=axeSize, **envdict())
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyHDRMerge(grWindow.options)

    elif name == 'actionGeom_Transformation':
        lname = 'Transformation'
        layer = window.label.img.addAdjustmentLayer(name=gn(lname), role='GEOMETRY')
        grWindow = transForm.getNewWindow(axeSize=axeSize, **envdict())
        # add transformation tool to parent widget
        tool = rotatingTool(parent=window.label)
        layer.addTool(tool)
        tool.showTool()
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyTransForm(grWindow.options)

    elif name == 'actionFilter':
        lname = 'Filter'
        layer = window.label.img.addAdjustmentLayer(name=gn(lname))
        grWindow = filterForm.getNewWindow(axeSize=axeSize, **envdict())
        # wrapper for the right apply method
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyFilter2D()

    elif name == 'actionGradual_Filter':
        lname = 'Gradual Filter'
        layer = window.label.img.addAdjustmentLayer(name=gn(lname))
        grWindow = blendFilterForm.getNewWindow(axeSize=axeSize, **envdict())
        # wrapper for the right apply method
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyBlendFilter()

    elif name == 'actionNoise_Reduction':
        lname = 'Noise Reduction'
        layer = window.label.img.addAdjustmentLayer(name=gn(lname))
        grWindow = noiseForm.getNewWindow(axeSize=axeSize, **envdict())
        # wrapper for the right apply method
        layer.execute = lambda l=layer, pool=None: l.tLayer.applyNoiseReduction()

    # invert image
    elif name == 'actionInvert':
        lname = 'Invert'
        layer = window.label.img.addAdjustmentLayer(name=gn(lname))
        grWindow = invertForm.getNewWindow(axeSize=axeSize, **envdict())
        layer.execute = lambda l=layer: l.tLayer.applyInvert()
        layer.applyToStack()

    elif name == 'actionChannel_Mixer':
        lname = 'Channel Mixer'
        layer = window.label.img.addAdjustmentLayer(name=gn(lname))
        grWindow = mixerForm.getNewWindow(axeSize=260, **envdict())
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
            layer = window.label.img.addAdjustmentLayer(name=gn(lname))
            pool = getPool()
            layer.execute = lambda l=layer, pool=pool: l.tLayer.apply3DLUT(lut,
                                                                           UDict(({'use selection': False,
                                                                                   'keep alpha': True},)),
                                                                           pool=pool
                                                                           )
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
            dlgWarn("Cannot build the 3D LUT:", str(e))
        finally:
            # restore stack
            img.removeLayer(1)
            # set hald flag
            img.isHald = False
            img.layersStack[0].applyToStack()
            img.prLayer.update()
            window.label.repaint()
            return

    # unknown or null action
    else:
        return

    post(layer)

    return layer if script else None


def menuHelp(name, window=bLUeTop.Gui.window):
    """
    Menu handler
    Init help browser
    A single instance is created.
    Unused parameters are for the sake of symmetry
    with other menu function calls.

    :param name: action name
    :type  name: str
    :param window:
    :type  window: QWidget
    """
    if name == "actionBlue_help":
        w = bLUeTop.Gui.app.focusWidget()
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
        w.label.setText(BLUE_VERSION + "\n" + attributions + "\n" + "http://bernard.virot.free.fr")
        w.show()


def canClose(index=None, window=bLUeTop.Gui.window):
    """
    If index is None, tries to save and close all opened documents, otherwise only
    the document in index tab is considered.
    Returns True if all requested tabs could be closed, False otherwise.
    Called by the application closeEvent slot, by closeTabs() and by closeTab().

    :param index: a valid tab index or None
    :type  index: int or None
    :param window:
    :type  window:
    :return:
    :rtype: boolean
    """
    if window.tabBar.count() == 0:
        return True
    closeAllRequested = (index is None)

    def canCloseTab(ind):
        img = window.tabBar.tabData(ind)
        if img.isModified:
            if ind != window.tabBar.currentIndex():
                window.tabBar.setCurrentIndex(ind)
                # dlgWarn('Image was modified', info='Save it first')  # TODO removed 19/12/21 validate
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
                    window.tabBar.removeTab(ind)
                    return True
                elif ret == QMessageBox.Cancel:
                    return False
            except (ValueError, IOError) as e:
                dlgWarn(str(e))
                return False
        # discard changes or img not modified : remove tab
        img = window.tabBar.tabData(ind)
        window.tabBar.removeTab(ind)  # keep before closeView
        stack = img.layersStack
        for layer in stack:  # little improvement for gc
            layer.closeView(delete=True)
        return True

    if closeAllRequested:
        while window.tabBar.count() > 0:
            ind = window.tabBar.currentIndex()
            if not canCloseTab(ind):
                break
    else:
        return canCloseTab(index)
    return window.tabBar.count() == 0


def updateStatus(window=bLUeTop.Gui.window):
    """
    Display current status
    """
    img = window.label.img
    # filename and rating
    s = '&nbsp;&nbsp;&nbsp;&nbsp;' + img.filename + '&nbsp;&nbsp;&nbsp;&nbsp;' + (' '.join(['*'] * img.meta.rating))
    # color management
    s += '&nbsp;&nbsp;&nbsp;&nbsp;CM : ' + ('On' if icc.COLOR_MANAGE else 'Off')
    if window.actionSoft_proofing.isChecked():
        s += '<font color=red<b>&nbsp;&nbsp;Soft Proofing :%s</b></font>' % basename(icc.softProofingProfile.filename)
    # Preview
    if img.useThumb:
        s += '<font color=red><b>&nbsp;&nbsp;&nbsp;&nbsp;Preview</b></font> '
    else:
        # mandatory to toggle html mode
        s += '<font color=black><b>&nbsp;&nbsp;&nbsp;&nbsp;</b></font> '
    # Before/After
    if window.viewState == 'Before/After':
        s += '&nbsp;&nbsp;&nbsp;&nbsp;Before/After : Ctrl+Space : cycle through views - Space : switch back to workspace'
    else:
        s += '&nbsp;&nbsp;&nbsp;&nbsp;Space : toggle Before/After view'
    # cropping
    if window.label.img.isCropped:
        w, h = window.cropTool.crWidth, window.cropTool.crHeight
        s += '&nbsp;&nbsp;&nbsp;&nbsp;Cropped : %dx%d (h/w=%.2f) ' % (w, h, h / w)
        s += '&nbsp;&nbsp;Use Ctrl+Drag to move and Ctrl+Wheel to zoom'
    # loading document
    if window.status_loadingFile:
        s += '&nbsp;&nbsp;&nbsp;&nbsp;Loading Image...'
    if window.status_loadingLayers:
        s += 'Layers...'
    window.Label_status.setText(s)


def initCursors(window=bLUeTop.Gui.window):
    """
    Init app cursors
    """
    # EyeDropper cursor
    curImg = QImage(":/images/resources/Eyedropper-icon.png")
    pxmp = QPixmap.fromImage(curImg)
    w, h = pxmp.width(), pxmp.height()
    window.cursor_EyeDropper = QCursor(pxmp, hotX=0, hotY=h - 1)
    # paint bucket cursor
    curImg = QImage(":/images/resources/icons8-windows-metro-26.png")
    pxmp = QPixmap.fromImage(curImg)
    w, h = pxmp.width(), pxmp.height()
    window.cursor_Bucket = QCursor(pxmp, hotX=w - 1, hotY=0)
    # tool cursor, must be resizable
    curImg = QImage(":/images/resources/cursor_circle.png")
    # turn to white
    curImg.invertPixels()
    window.cursor_Circle_Pixmap = QPixmap.fromImage(curImg)


def initDefaultImage():
    img = QImage(200, 200, QImage.Format_ARGB32)
    img.fill(Qt.darkGray)
    return imImage(QImg=img, meta=metadataBag(name='noName'))


def screenUpdate(newScreenIndex, window=bLUeTop.Gui.window):
    """
    screenChanged event handler.
    The image is updated in background

    :param newScreenIndex:
    :type  newScreenIndex: QScreen
    :param window:
    :type  window: QWidget
    """
    window.screenChanged.disconnect()
    # update the color management object using the profile associated with the current monitor
    icc.configure(qscreen=newScreenIndex)
    window.actionColor_manage.setEnabled(icc.HAS_COLOR_MANAGE)
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    updateStatus()

    # launch a bg task for updating of presentation layers
    def bgTask():
        updateCurrentViews()

    threading.Thread(target=bgTask).start()
    window.screenChanged.connect(screenUpdate)


class HistQDockWidget(QDockWidget):
    pass


def setRightPane(window=bLUeTop.Gui.window):
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

    # put all stuff into propertyWidget and
    # fix its size : scrollbars are more acceptable for the
    # display of layer stack than in graphic forms.
    window.propertyWidget.setLayout(vl)
    window.propertyWidget.setFixedSize(250, 200)

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


def setColorManagement(window=bLUeTop.Gui.window):
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
    window.actionUpdate_display_profile.setEnabled(COLOR_MANAGE_OPT)
    window.actionColor_manage.setEnabled(icc.HAS_COLOR_MANAGE)
    window.actionColor_manage.setChecked(icc.COLOR_MANAGE)
    updateStatus()


def dragEnterEvent(widget, img, event):
    """
    Accept drop if mimeData contains text (e.g. file name)
    (convenient for main window only)

    :param widget:
    :type  widget:
    :param img:
    :type  img:
    :param event:
    :type  event:
    """
    if event.mimeData().hasFormat("text/plain"):
        event.acceptProposedAction()


def dropEvent(widget, img, event):
    """
    get file name from event.mimeData and open it.

    :param widget:
    :type  widget:
    :param img:
    :type  img:
    :param event:
    :type  event:

    """
    mimeData = event.mimeData()
    openFile(mimeData.text())


def setupGUI(window=bLUeTop.Gui.window):
    """
    Display splash screen, set app style sheet

    :param window:
    :type  window:
    """
    # splash screen
    splash = QSplashScreen(QPixmap('logo.png'), Qt.WindowStaysOnTopHint)
    font = splash.font()
    font.setPixelSize(12)
    splash.setFont(font)
    splash.show()
    splash.showMessage("Loading .", color=Qt.white, alignment=Qt.AlignCenter)
    bLUeTop.Gui.app.processEvents()
    sleep(1)
    splash.showMessage(BLUE_VERSION + "\n" + attributions + "\n" + "http://bernard.virot.free.fr", color=Qt.white,
                       alignment=Qt.AlignCenter)
    bLUeTop.Gui.app.processEvents()
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
        bLUeTop.Gui.app.setStyleSheet("""QMainWindow, QGraphicsView, QListWidget, QMenu, QTableView {background-color: rgb(200, 200, 200)}\
                               QWidget, QTableView, QTableView * {font-size: 9pt} QPushButton {font-size: 6pt}"""
                                      )
    else:
        bLUeTop.Gui.app.setStyleSheet(Path('bLUe.qss').read_text())

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
    window.verticalSlider1, window.verticalSlider2, window.verticalSlider3, window.verticalSlider4 \
        = QbLUeSlider(Qt.Horizontal), QbLUeSlider(Qt.Horizontal), QbLUeSlider(Qt.Horizontal), QbLUeSlider(Qt.Horizontal)
    for slider in (window.verticalSlider1, window.verticalSlider2, window.verticalSlider3, window.verticalSlider4):
        slider.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)
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
    window.brushCombo, window.patternCombo = QComboBox(), QComboBox()
    window.brushCombo.setToolTip('Brush Family')
    window.patternCombo.setToolTip('Patterns')
    window.brushCombo.setIconSize(QSize(50, 50))
    window.patternCombo.setIconSize(QSize(50, 50))
    window.brushCombo.setMinimumWidth(150)
    window.patternCombo.setMinimumWidth(150)
    window.brushes = initBrushes()
    for b in window.brushes[:-1]:  # don't add eraser to combo
        if b.preset is None:
            window.brushCombo.addItem(b.name, b)
        else:
            window.brushCombo.addItem(QIcon(b.preset), b.name, b)
    window.patternCombo.addItem('None', None)
    window.verticalSlider1.sliderReleased.connect(window.label.brushUpdate)
    window.verticalSlider2.sliderReleased.connect(window.label.brushUpdate)
    window.verticalSlider3.sliderReleased.connect(window.label.brushUpdate)
    window.verticalSlider4.sliderReleased.connect(window.label.brushUpdate)
    window.brushCombo.currentIndexChanged.connect(window.label.brushUpdate)
    window.patternCombo.currentIndexChanged.connect(window.label.brushUpdate)
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
    toolBar.addWidget(window.patternCombo)
    # link tooLBar to the group of tool buttons
    for button in window.drawFG.group().buttons():
        button.toolBar = toolBar
    window.addToolBar(toolBar)

    # whatsThis
    window.cropButton.setWhatsThis(
        """To <b>crop</b> the image drag a gray curtain on either side using the 8 small square buttons
        around the image. Use corner buttons to keep the image aspect ratio unchanged.<br>
        To <b>drag</b> the cropping area over the image select the <i>Drag Tool</i> and do <i>Ctrl+Drag
        with the mouse.</i><br>
        To <b>zoom</b> inside the cropping area select the <i>Drag Tool</i> and do <i>Ctrl+Wheel.</i> 
        The image point under the mouse cursor is kept fixed.
        """)
    window.rulerButton.setWhatsThis("""Draw horizontal and vertical rulers over the image""")
    window.fitButton.setWhatsThis("""Reset the image size to the window size""")
    window.eyeDropper.setWhatsThis("""<b>Color Picker</b><br> Click on the image to sample pixel colors""")
    window.toolButton.setWhatsThis("""<b>Pointer Tool</b><br>""")
    window.dragBtn.setWhatsThis(
        """<b>Drag Tool</b><br> Mouse Left Button : drag the whole image<br> Ctrl+Mouse Left 
        Button : drag the active layer only
        """
    )
    window.rectangle.setWhatsThis(
        """<b>Marquee Tool/Selection Rectangle</b><br>
        Draws selection rectangles on the active layer.<br>
        For a segmentation layer only, all pixels outside the rectangle are set to background.<br>
        Ctrl+click inside a rectangle to remove it.<br>
        To refine the selection add a mask to the layer.
        """
    )
    window.drawFG.setWhatsThis(
        """
        <b>Foreground/Unmask tool</b><br>
          Paint on the active layer to <b>unmask</b> a previously masked region or to <b>select foreground pixels</b> (segmentation layer only);
          the mask must be enabled as opacity or color mask in the layer panel.<br>
          With <b>Color Mask</b> enabled, masked pixels are grayed and unmasked pixels are reddish.<br>
          Use the <b>Brush Size slider</b> below to choose the size of the tool. 
        """
    )
    window.drawBG.setWhatsThis(
        """<b>Background/Mask tool</b><br>
          Paint on the active layer to mask a region or to select background pixels (segmentation layer only);
          (the mask must be enabled as opacity or color mask in the layer panel).<br>
          With <b>Color Mask</b> enabled, masked pixels are grayed and unmasked pixels are reddish.<br>
          Use the 'Brush Size' slider below to choose the size of the tool. 
        """
    )
    window.verticalSlider1.setWhatsThis("""Set the diameter of the painting brush""")
    window.verticalSlider2.setWhatsThis("""Set the opacity of the painting brush""")
    window.verticalSlider3.setWhatsThis("""Set the hardness of the painting brush""")
    window.verticalSlider4.setWhatsThis("""Set the flow of the painting brush""")
    window.brushCombo.setWhatsThis(
        """Loaded painting brushes.<br>To add presets use<br><i>Menu File->Load Preset</i>""")
    window.patternCombo.setWhatsThis(
        """Available patterns.<br> Patterns are loaded with brushes using <br><i>Menu File->Load Preset</i>""")

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
    window.menuLoad_Preset.aboutToShow.connect(updateMenuLoadPreset)
    window.menu_File.triggered.connect(lambda a: menuFile(a.objectName()))
    window.menuLayer.triggered.connect(lambda a: menuLayer(a.objectName()))
    window.menuImage.triggered.connect(lambda a: menuImage(a.objectName()))
    window.menuWindow.triggered.connect(lambda a: menuView(a.objectName()))
    window.menuHelp.triggered.connect(lambda a: menuHelp(a.objectName()))

    #  onWidgetChange is called by all main form button and slider slots (cf. QtGui1.py)
    window.onWidgetChange = widgetChange

    # init imageLabel objects
    window.label.window = window
    window.label.splitWin = bLUeTop.Gui.splitWin
    window.label_2.window = window
    window.label_2.splitWin = bLUeTop.Gui.splitWin
    window.label_3.window = window
    window.label_3.splitWin = bLUeTop.Gui.splitWin
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
    window.splitter.currentState = next(bLUeTop.Gui.splitWin.splitViews)
    window.splitter.setSizes([2 ** 20, 2 ** 20])
    window.splitter.setHandleWidth(1)
    window.splitter.hide()
    window.viewState = 'After'
    actionCycle = QAction('cycle', window)
    actionCycle.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_Space))

    # status flags
    window.status_loadingFile = False
    window.status_loadingLayers = False

    def f():
        window.viewState = 'Before/After'
        bLUeTop.Gui.splitWin.nextSplitView()
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


def switchDoc(index, window=bLUeTop.Gui.window):
    """
    tabBarClicked slot : make visble the document in tab index

    :param index: tab index
    :type  index: int
    """
    # clean up
    layer = window.label.img.getActiveLayer()
    if layer.tool is not None:
        layer.tool.hideTool()
    img = window.tabBar.tabData(index)
    setDocumentImage(img)


def closeTab(index):
    """
    tabCloseRequested Slot.
    Tries to save and close a single document.

    :param index: valid tab index
    :type  index: int
    """
    closeTabs(index=index)


def setTabBar(window=bLUeTop.Gui.window):
    tabBar = QTabBar()
    tabBar.currentChanged.connect(switchDoc)
    tabBar.tabCloseRequested.connect(closeTab)
    tabBar.setMaximumHeight(25)
    # tabBar.setAutoHide(True)
    tabBar.setDrawBase(False)  # remove base line
    tabBar.setStyleSheet("QTabBar::tab {height: 15px; width: 100px; border-top-right-radius: 4px}")
    tabBar.setTabsClosable(True)
    vlay = QVBoxLayout()
    hlay2 = QHBoxLayout()
    hlay2.addWidget(tabBar)
    hlay2.addStretch(1)
    vlay.addLayout(hlay2)
    hlay1 = QHBoxLayout()
    hlay1.addWidget(window.label)
    hlay1.addWidget(window.splitter)
    vlay.addLayout(hlay1)
    hlay = window.horizontalLayout_2
    window.groupbox_btn.setParent(None)
    hlay.addWidget(window.groupbox_btn)
    # layoutWidget is silently added by Qt Designer as child of groupbox_btn
    # and parent of its layout. This widget overlaps a part of groupbox_btn border,
    # so we set its background to transparent.
    window.groupbox_btn.setStyleSheet("QWidget#layoutWidget {background-color: transparent}")
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
    # appInit()
    # load UI
    bLUeTop.Gui.window.init()
    bLUeTop.Gui.window.setWindowIcon(QIcon('logo.png'))
    # display splash screen and set app style sheet
    setupGUI()
    setTabBar()

    ###############
    # launching app
    sys.exit(bLUeTop.Gui.app.exec_())
