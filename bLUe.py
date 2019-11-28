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

from bLUeTop import resources_rc  # mandatory

import numpy as np
import multiprocessing
import sys
import threading
from itertools import cycle
from os import path, walk
from io import BytesIO
from time import sleep
import gc

from PIL.ImageCms import ImageCmsProfile
from types import MethodType
import rawpy

from bLUeCore.bLUeLUT3D import HaldArray
from bLUeTop.drawing import brushFamily, initBrushes
from bLUeTop.graphicsDraw import drawForm
from bLUeTop.graphicsHDRMerge import HDRMergeForm
from bLUeTop.graphicsSegment import segmentForm
from PySide2.QtCore import QUrl, QSize, QFileInfo
from PySide2.QtGui import QPixmap, QCursor, QKeySequence, QDesktopServices, QFont, \
    QTransform, QColor, QImage, QPainterPath
from PySide2.QtWidgets import QApplication, QAction, \
    QMainWindow, QDockWidget, QSizePolicy, QScrollArea, QSplashScreen, QWidget, \
    QStyle, QTabWidget, QToolBar, QComboBox
from bLUeTop.QtGui1 import app, window, rootWidget, splittedWin, set_event_handlers, brushUpdate
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
from bLUeTop.utils import UDict
from bLUeGui.tool import cropTool, rotatingTool
from bLUeTop.graphicsTemp import temperatureForm
from bLUeTop.graphicsFilter import filterForm
from bLUeTop.graphicsHspbLUT import graphicsHspbForm
from bLUeTop.graphicsLabLUT import graphicsLabForm

from bLUeCore.demosaicing import demosaic
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
VERSION = "v1.6.1"
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


def loadImageFromFile(f, createsidecar=True, icc=icc):
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
    @param icc:
    @type icc: class icc
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
        metadata = {'SourceFile': f}
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
        # As a workaround we use low-level file buffer and unpack().
        # Relevant RawPy attributes are black_level_per_channel, camera_white_balance, color_desc, color_matrix,
        # daylight_whitebalance, num_colors, raw_colors_visible, raw_image, raw_image_visible, raw_pattern,
        # raw_type, rgb_xyz_matrix, sizes, tone_curve.
        # raw_image and raw_image_visible are sensor data
        rawpyInst = rawRead(f)
        # postprocess raw image, applying default settings (cf. vImage.applyRawPostProcessing)
        rawBuf = rawpyInst.postprocess(use_camera_wb=True)
        # build Qimage : switch to BGR and add alpha channel
        rawBuf = np.dstack((rawBuf[:, :, ::-1], np.zeros(rawBuf.shape[:2], dtype=np.uint8)+255))
        img = imImage(cv2Img=rawBuf, colorSpace=colorSpace, orientation=transformation,
                      rawMetadata=metadata, profile=profile, name=name, rating=rating)
        # keeping a reference to rawBuf along with img is
        # needed to protect the buffer from garbage collector
        img.rawBuf = rawBuf
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


def loadImage(img, window=window):
    """
    load a vImage into bLUe
    @param img:
    @type img: vImage
    @param window:
    @type window: QWidget
    """
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


def openFile(f, window=window):
    """
    Top level function for file opening, used by File Menu actions
    @param f: file name
    @type f: str
    @param window:
    @type window: QWidget
    """
    # close open document, if any
    if not closeFile():
        return
    try:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        # load imImage from file
        img = loadImageFromFile(f)
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


def closeFile(window=window):
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
    defaultImImage = initDefaultImage()
    window.label.img = defaultImImage
    window.label_2.img = defaultImImage
    window.label_3.img = defaultImImage
    gc.collect()
    window.label.update()
    window.label_2.update()
    window.label_3.update()
    return True


def showHistogram(window=window):
    """
    Update and display the histogram of the
    currently opened document
    """
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
    window.histView.cache = QPixmap.fromImage(histView)
    window.histView.Label_Hist.setPixmap(window.histView.cache.scaled(window.histView.width() - 20, window.histView.height()-50))
    window.histView.Label_Hist.repaint()


def adjustHistogramSize(window=window):
    pxm = getattr(window.histView, 'cache', None)
    if pxm is not None:
        window.histView.Label_Hist.setPixmap(pxm.scaled(window.histView.width() - 20, window.histView.height()-50))
        window.histView.Label_Hist.repaint()


def setDocumentImage(img, window=window):
    """
    Inits GUI and displays the current document
    @param img: image
    @type img: imImage
    @param window:
    @type window: QWidget
    """
    window.cropButton.setChecked(False)
    window.rulerButton.setChecked(False)
    window.label.img = img
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

    window.label.img.onImageChanged = f

    ###################################
    # init displayed images
    # label.img : working image
    # label_2.img  : before image (copy of the initial state of working image)
    # label_3.img : reference to working image
    ###################################

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
    window.label.img.window = window.label  # TODO 4/11/18 graphicsForm3DLUT.onReset should be modified to remove this dependency
    window.label_2.img.window = window.label_2
    window.label.img.setModified(True)


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
    window.actionSave_Hald_Cube.setEnabled(window.label.img.isHald)


def menuFile(name, window=window):
    """
    Menu handler
    @param name: action name
    @type name: str
    @param window:
    @type window: QWidget
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
    # new image from clipboard
    if name == 'actionNew':
        # close open document, if any
        if not closeFile():
            return
        cb = QApplication.clipboard()
        img = cb.image()
        if not img.isNull():
            img = imImage(QImg=img)
            loadImage(img)
        else:
            dlgWarn("Clipboard : no image found")
    # display image info
    if name == 'actionImage_info':
        # Format
        s = "Format : %s\n(cf. QImage formats in the doc for more info)" % QImageFormats.get(img.format(), 'unknown')
        # dimensions
        s = s + "\n\ndim : %d x %d" % (img.width(), img.height())
        # profile info
        if img.meta.profile is not None:
            if len(img.meta.profile) > 0:
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
        dock = QDockWidget(window)
        dock.setWidget(grWindow)
        dock.setWindowTitle(grWindow.windowTitle())
        if TABBING:
            # add form to docking area
            forms = [item.view for item in layer.parentImage.layersStack if getattr(item, 'view', None) is not None]
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
            layer = window.label.img.addAdjustmentLayer(name=lname, sourceImg=imgNew, role='Image')  # role='GEOMETRY')
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
        imgNew.fill(QColor(0,0,0,0))
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
        layer = window.label.img.addAdjustmentLayer(name=lname)
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
        w, label = handleTextWindow(parent=window, title='About bLUe', center=False)
        label.setStyleSheet("background-image: url(logo.png); color: white;")
        label.setAlignment(Qt.AlignCenter)
        label.setText(VERSION + "\n" + attributions + "\n" + "http://bernard.virot.free.fr")
        # center window on screen
        w.setGeometry(QStyle.alignedRect(Qt.LeftToRight, Qt.AlignCenter, w.size(),
                                         rootWidget.availableGeometry(w)))
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


def canClose(window=window):
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
        s += '&nbsp;&nbsp;&nbsp;&nbsp;Press Space Bar to toggle Before/After view'
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
            w.setWindowTitle('Hist')
            histViewDock = HistQDockWidget()
            hl = QHBoxLayout()
            #hl.addStretch(1)
            hl.setAlignment(Qt.AlignLeft)
            hl.addWidget(w)
            w.setMaximumSize(140000, 140000)
            wdg = QWidget()
            wdg.setMaximumSize(140000, 140000)
            wdg.setLayout(hl)
            histViewDock.setWidget(wdg)
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
                               QTabBar::tab:selected {border-top-color: white; 
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
    window.verticalSlider1.setToolTip('Brush Size')
    window.verticalSlider2.setAccessibleName('verticalSlider2')
    window.verticalSlider2.setRange(0, 100)
    window.verticalSlider2.setSliderPosition(100)
    window.verticalSlider2.setToolTip('Brush Opacity')
    window.verticalSlider3.setAccessibleName('verticalSlider3')
    window.verticalSlider3.setRange(0, 100)
    window.verticalSlider3.setSliderPosition(100)
    window.verticalSlider3.setToolTip('Brush Hardness')
    window.verticalSlider4.setAccessibleName('verticalSlider4')
    window.verticalSlider4.setRange(0, 100)
    window.verticalSlider4.setSliderPosition(100)
    window.verticalSlider4.setToolTip('Brush Flow')
    brushes = initBrushes()
    window.brushCombo = QComboBox()
    for b in brushes[:-1]:  # don't add eraser to combo
        window.brushCombo.addItem(b.name, b)
    for slider in [window.verticalSlider1, window.verticalSlider2, window.verticalSlider3, window.verticalSlider4]:
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setMaximumSize(100, 15)
        toolBar.addWidget(slider)
        empty = QWidget()
        empty.setFixedHeight(30)
        empty.setFixedWidth(50)
        empty.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolBar.addWidget(empty)
    toolBar.addWidget(window.brushCombo)
    # record the (initial) current brush
    brushUpdate()
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

    setupGUI(window)

    ###############
    # launch app
    ###############
    sys.exit(app.exec_())

