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
import gc
import os
import pickle
import shutil
from datetime import datetime

import tifffile
from ast import literal_eval
from os import path, listdir
from os.path import basename, isfile
from re import search

from PySide6.QtCore import Qt, QMimeData, QPoint, QSize, QBuffer, QIODevice, QRect, QEventLoop, \
    QTimer, QObject, QMargins
from PySide6.QtGui import QKeySequence, QImage, QDrag, QAction, QPixmap, QIcon, QFontMetrics
from PySide6.QtWidgets import QMainWindow, QSizePolicy, QMenu, QListWidget, QAbstractItemView, \
    QApplication, QListWidgetItem, QFileDialog, QWidget, QHBoxLayout, QLabel, QPushButton

from bLUeGui.memory import weakProxy
from bLUeTop import exiftool
from bLUeTop.MarkedImg import imImage
import bLUeTop.Gui
from bLUeTop.heif import readheifFile2QImage
from bLUeTop.imLabel import slideshowLabel
from bLUeTop.utils import stateAwareQDockWidget, imagej_description_metadata, fileExt, sortPushButton, \
    QbLUePushButton, restricted_loads
from bLUeGui.dialog import IMAGE_FILE_EXTENSIONS, RAW_FILE_EXTENSIONS, BLUE_FILE_EXTENSIONS, dlgWarn, \
    HEIF_FILE_EXTENSIONS, QblueFileDialog, IMAGE_FILE_NAME_FILTER

# global variable recording diaporama state
isSuspended = False


def playDiaporama(diaporamaGenerator, parent=None):
    """
    Open a new window and play a slide show.

    :param diaporamaGenerator: generator for file names
    :type  diaporamaGenerator: iterator object
    :param parent:
    :type parent:
    """
    global isSuspended
    isSuspended = False

    # init diaporama window
    newWin = QMainWindow(parent)
    newWin.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    newWin.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    newWin.setWindowTitle(parent.tr('Slide show'))
    label = slideshowLabel(mainForm=parent)
    label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    newWin.setCentralWidget(label)
    newWin.showFullScreen()
    # Pause key shortcut
    actionEsc = QAction('Pause', None)
    actionEsc.setShortcut(QKeySequence(Qt.Key.Key_Escape))
    newWin.addAction(actionEsc)

    # context menu event handler
    def contextMenuHandler(action):
        global isSuspended
        if action.text() == 'Pause':
            isSuspended = True
            # quit full screen mode
            newWin.showMaximized()
        elif action.text() == 'Full Screen':
            if action.isChecked():
                newWin.showFullScreen()
            else:
                newWin.showMaximized()
        elif action.text() == 'Resume':
            newWin.close()
            isSuspended = False
            playDiaporama(diaporamaGenerator, parent=bLUeTop.Gui.window)
        elif action.text() == 'Close':
            newWin.close()
            isSuspended = False
        # rating : the tag is written into the .mie file; the file is
        # created if needed.
        elif action.text() in ['0', '1', '2', '3', '4', '5']:
            with exiftool.ExifTool() as e:
                e.writeXMPTag(name, 'XMP:rating', int(action.text()))

    # connect shortkey action
    actionEsc.triggered.connect(
        lambda checked=False, name=actionEsc: contextMenuHandler(name))  # named arg checked is sent

    # context menu
    def contextMenu(position):
        menu = QMenu()
        actionEsc.setEnabled(not isSuspended)
        action2 = QAction('Full Screen', None)
        action2.setCheckable(True)
        action2.setChecked(newWin.isFullScreen())  # windowState() & Qt.WindowFullScreen)
        action3 = QAction('Resume', None)
        action3.setEnabled(isSuspended)
        action4 = QAction('Close', None)
        for action in [actionEsc, action2, action3, action4]:
            menu.addAction(action)
            action.triggered.connect(
                lambda checked=False, name=action: contextMenuHandler(name))  # named arg checked is sent
        subMenuRating = menu.addMenu('Rating')
        for i in range(6):
            action = QAction(str(i), None)
            subMenuRating.addAction(action)
            action.triggered.connect(
                lambda checked=False, name=action: contextMenuHandler(name))  # named arg checked is sent
        menu.exec(position)

    def testPaused():
        #bLUeTop.Gui.app.processEvents()
        if isSuspended:
            newWin.setWindowTitle(newWin.windowTitle() + ' Paused')
        return isSuspended

    # connect contextMenuRequested
    newWin.customContextMenuRequested.connect(contextMenu)
    newWin.setToolTip("Esc to exit full screen mode")
    newWin.setWhatsThis(
        """ <b>Slide Show</b><br>
        The slide show cycles through the starting directory and its subfolders to display images.
        Photos rated 0 or 1 star are not shown (by default, all photos are rated 5 stars).<br>
        Hit the Esc key to <b>exit full screen mode and pause.</b><br> Use the Context Menu for <b>rating and resuming.</b>
        The rating is saved in the .mie sidecar and the image file is not modified.
        """
    )  # end of setWhatsThis
    # play diaporama
    bLUeTop.Gui.window.modeDiaporama = True
    while True:
        if testPaused():
            break
            
        try:
            if not newWin.isVisible():
                break
            name = next(diaporamaGenerator)
            # search rating in metadata
            rating = 5  # default
            with exiftool.ExifTool() as e:
                try:
                    rt = e.readXMPTag(name, 'XMP:rating')  # raise ValueError if sidecar not found
                    r = search(r'\d', rt)
                    if r is not None:
                        rating = int(r.group(0))
                except ValueError:
                    rating = 5
            # don't display image with low rating
            if rating < 2:
                # app.processEvents()
                continue
            imImg = imImage.loadImageFromFile(name, createsidecar=False, cmsConfigure=True, window=bLUeTop.Gui.window)
            # zoom might be modified by the mouse wheel : remember it
            if label.img is not None:
                imImg.Zoom_coeff = label.img.Zoom_coeff
            coeff = imImg.resize_coeff(label)
            imImg.yOffset -= (imImg.height() * coeff - label.height()) / 2.0
            imImg.xOffset -= (imImg.width() * coeff - label.width()) / 2.0

            if testPaused():
                break

            newWin.setWindowTitle(parent.tr('Slide show') + ' ' + name + ' ' + ' '.join(['*'] * imImg.meta.rating))
            label.img = imImg
            gc.collect()

            if label.prevImg is not None:
                for i in range(41): # range(81):
                    label.prevOpacity = 1.0 -  i * 0.0250 # i * 0.0125  # last prevOpacity must be 0
                    label.repaint()

            label.repaint()  # mandatory to display first image

            if testPaused():
                break

            # wait 2 sec.
            loop = QEventLoop()
            QTimer.singleShot(2000, loop.quit)
            loop.exec()

            if testPaused():
                break

            label.prevImg = label.img
            gc.collect()

        except StopIteration:
            newWin.close()
            bLUeTop.Gui.window.diaporamaGenerator = None
            break
        except ValueError:
            continue
        except RuntimeError:
            bLUeTop.Gui.window.diaporamaGenerator = None
            break
        except:
            bLUeTop.Gui.window.diaporamaGenerator = None
            bLUeTop.Gui.window.modeDiaporama = False
            raise

    bLUeTop.Gui.window.modeDiaporama = False


class blueQListWidgetItem(QListWidgetItem):
    """
    Overrides QListWidgetItem comparison operator <.
    To choose the right operator, instances use
    a weak ref to the containing QListWidget (QListWidgetItem
    has no parent attribute)
    """
    def __init__(self, *args, listwidget=None):
        super().__init__(*args)
        self.wdg = weakProxy(listwidget)

    def __lt__(self, other):
        sortIndex = self.wdg.sortIndex
        sortOrder = self.wdg.sortOrder
        if  sortOrder == Qt.SortOrder.AscendingOrder:
            return self.data(Qt.ItemDataRole.UserRole)[sortIndex] < other.data(Qt.ItemDataRole.UserRole)[sortIndex]
        else:
            return self.data(Qt.ItemDataRole.UserRole)[sortIndex] > other.data(Qt.ItemDataRole.UserRole)[sortIndex]


class dragQListWidget(QListWidget):
    """
    This class overrides mousePressEvent.It initializes
    a convenient QMimeData object for drag and drop events.
    For item comparisons, the < operator is determined by
    the attributes sortIndex and sortOrder.
    sortIndex indicates which component of item's user data will be used
    for comparison.
    sortOrder is a tuple of length maxSortIndex + 1, containing a value
    (AscendingOrder or DescendingOrder) for each index.
    """

    def __init__(self, maxSortIndex=3):
        super().__init__()
        self.maxSortIndex = maxSortIndex
        self.sortIndex = 0
        self.sortOrder = [Qt.SortOrder.AscendingOrder]  * (maxSortIndex + 1)

    def mousePressEvent(self, event):
        # call to super needed for selections
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mimeData = QMimeData()
            item = self.itemAt(event.pos())
            if item is None:
                return
            mimeData.setText(item.data(Qt.ItemDataRole.UserRole)[0])  # should be path to file
            drag.setMimeData(mimeData)
            # set dragging pixmap
            drag.setPixmap(item.icon().pixmap(QSize(160, 120)))
            # roughly center the cursor relative to pixmap
            drag.setHotSpot(QPoint(60, 60))
            dropAction = drag.exec()


class loader(QObject):
    """
    Image loader. It loads image files yielded by a generator as blueQListWidgetItem instances and populates
    a dragQListWidget with them.
    """
    def __init__(self, gen, viewerInstance):
        """
       :param gen: generator for image file names
       :type gen: generator
       :param viewerInstance:
       :type viewerInstance: viewer
        """
        super().__init__()
        self.fileListGen = gen
        self.viewerInstance = viewerInstance

    def populate(self, items):
        for item in items:
            self.viewerInstance.listWdg.addItem(item)
        self.viewerInstance.listWdg.viewport().repaint()  # viewport().update()

    def load(self):
        with exiftool.ExifTool() as e:
            while True:
                try:
                    # next() raises a StopIteration exception when the generator ends.
                    # If this exception is unhandled by run(), it causes thread termination.
                    filename = next(self.fileListGen)
                    # get orientation
                    try:
                        # read metadata from sidecar (.mie) if it exists, otherwise from image file.
                        profile, metadata = e.get_metadata(filename,
                                                           tags=(
                                                               "colorspace",
                                                               "profileDescription",
                                                               "orientation",
                                                               "model",
                                                               "rating",
                                                               "DateTimeOriginal",
                                                               "FileCreateDate"),
                                                           createsidecar=False)
                    except ValueError:
                        metadata = {}

                    # get image info
                    tmp = [value for key, value in metadata.items() if 'orientation' in key.lower()]
                    orientation = tmp[0] if tmp else 1  # metadata.get("EXIF:Orientation", 1)
                    # EXIF:DateTimeOriginal seems to be missing in some files
                    tmp1, tmp2 = metadata.get("DateTimeOriginal", None), metadata.get("FileCreateDate", None)
                    date1, date2 = (datetime.min, ) * 2
                    if tmp1:
                        try:
                            date1 = datetime.strptime(tmp1, "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            pass
                    if tmp2:
                        try:
                            date2 = datetime.strptime(tmp2, "%Y:%m:%d %H:%M:%S%z")
                        except ValueError:
                            pass

                    tmp = [value for key, value in metadata.items() if 'rating' in key.lower()]
                    rating = tmp[0] if tmp else 0  # metadata.get("XMP:Rating", 5)
                    rating = ''.join(['*'] * int(rating))
                    transformation = exiftool.decodeExifOrientation(orientation)

                    # get thumbnail
                    img = e.get_thumbNail(filename, thumbname='thumbnailimage')

                    # no thumbnail found : try preview
                    if img.isNull():
                        img = e.get_thumbNail(filename,
                                              thumbname='PreviewImage'
                                              )  # the order is important : for jpeg PreviewImage is full sized !
                    # may be a bLU file
                    if img.isNull() and fileExt(filename) in BLUE_FILE_EXTENSIONS:
                        with tifffile.TiffFile(filename) as tfile:
                            meta_dict = imagej_description_metadata(tfile.pages[0].description)
                            v = meta_dict.get('thumbnailimage', None)  # lazy loading: need tfile open !
                        if v is not None:
                            ba = restricted_loads(literal_eval(v))
                            buffer = QBuffer(ba)
                            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
                            img = QImage()
                            img.load(buffer, 'JPG')
                    # everything else fails, read file !
                    if img.isNull():
                        if filename.endswith(HEIF_FILE_EXTENSIONS):
                            img = readheifFile2QImage(filename)
                        else:
                            img = QImage(filename)
                        img = img.scaled(QSize(viewer.iconSize, viewer.iconSize),
                                                      aspectMode=Qt.AspectRatioMode.KeepAspectRatio
                                                      )
                    if img.isNull():
                        raise IOError

                    # remove possible black borders, except for .NEF
                    if filename[-3:] not in ['nef', 'NEF']:
                        bBorder = 7
                        img = img.copy(QRect(0, bBorder, img.width(), img.height() - 2 * bBorder))

                    pxm = QPixmap.fromImage(img)
                    if not transformation.isIdentity():
                        pxm = pxm.transformed(transformation)

                    # set item caption and tooltip
                    item = blueQListWidgetItem(QIcon(pxm), basename(filename), listwidget=self.viewerInstance.listWdg)  # icon, text
                    item.setToolTip(basename(filename) +
                                    '\n' +
                                    'Taken:\t\t' + date1.strftime("%Y %m %d %X %p") +
                                    '\n' +
                                    'Created:\t' + date2.strftime("%Y %m %d %X %p")
                                    )
                    # set item mimeData to get filename=item.data(Qt.UserRole)[0] transformation=item.data(Qt.UserRole)[1]
                    item.setData(Qt.ItemDataRole.UserRole, (filename, transformation, date1, date2))
                    item.setSizeHint(QSize(viewer.iconSize, viewer.iconSize + 15))
                    self.populate([item])
                except (OSError, IOError, ValueError, tifffile.TiffFileError, KeyError, SyntaxError,
                        ModuleNotFoundError, pickle.UnpicklingError) as ex:
                    continue
                except StopIteration:
                    break


class viewer(QObject):
    """
    Synchronized QblueFileDialog and dragQListWidget. The file dialog is used to
    select a directory. All image files in this directory are displayed as icons
    in the list widget.
    3 buttons allow sorting the icons by file name, taken date or creation date.
    A context menu allows copying selected files and to set their rating.
    2 dock widgets are used to contain the file dialog and the list widget.
    """

    instance = None
    iconSize = 120

    @classmethod
    def getViewerInstance(cls, mainWin=None):
        """
        Returns a unique viewer instance : a new instance
        is created only if there exists no instance yet.

        :param mainWin: should be the app main window
        :type mainWin: QMainWindow
        :return: viewer instance
        :rtype: viewer
        """
        if cls.instance is None:
            cls.instance = viewer(mainWin=mainWin)
        return cls.instance

    @classmethod
    def isInstanciated(cls):
        return (cls.instance is not None)

    def __init__(self, mainWin=None):
        """

       :param mainWin: should be the app main window
       :type mainWin:  QMainWindow
        """
        super().__init__(parent=mainWin)
        self.listWdg = None
        self.currentDir = '.'
        self.fileDlg = None
        self.fileDlgDock, self.listViewDock = (None,) * 2
        self.sortButtons = ()
        self.cMenu = None
        self.initWins()
        self.initFileDlg()
        self.initCMenu()
        self.refButton.pressed.connect(self.refreshViewer)

    def currentToSettings(self, mainWin=None):
        mainWin.settings.setValue('paths/dlgdir', self.currentDir)
        mainWin.settings.setValue('mainwindow/explistwdg', self.listViewDock.isVisible())
        mainWin.settings.setValue('mainwindow/expfiledlg', self.fileDlg.isVisible())

    def currentFromSettings(self, mainWin=None):
        return mainWin.settings.value('paths/dlgdir', '.')

    def initFileDlg(self):
        #lastDir = self.currentFromSettings(mainWin=bLUeTop.Gui.window)
        fileDlg = QblueFileDialog(bLUeTop.Gui.window, "Select a folder", self.currentDir)
        fileDlg.setNameFilters(IMAGE_FILE_NAME_FILTER + ['All files (*)'])
        fileDlg.setFileMode(QFileDialog.FileMode.Directory)
        fileDlg.setOption(QFileDialog.Option.ShowDirsOnly)
        fileDlg.setLabelText(QFileDialog.DialogLabel.Accept, 'Close')  # accept button
        fileDlg.setWhatsThis(
            """
            The <b>bLUe File Explorer</b> is composed of two synchronized windows.
            <UL>
            <li> The left window is a usual file explorer
            <li> All image files in the current directory, including raw files and blu files,
            are shown as icons in the bottom window.
            </UL>
            Use <i>Ctrl+L</i> to open or reopen the file explorer.
            """
        )
        self.fileDlg = fileDlg
        self.fileDlgDock = fileDlg.setDock()
        bLUeTop.Gui.window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.fileDlgDock)

        fileDlg.directoryEntered.connect(self.showViewer)
        fileDlg.finished.connect(self.recordDir)

    def initWins(self):
        listWdg = dragQListWidget()
        listWdg.setWrapping(False)
        listWdg.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        listWdg.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        listWdg.setViewMode(QListWidget.ViewMode.IconMode)
        listWdg.setIconSize(QSize(self.iconSize, self.iconSize))
        listWdg.setMaximumSize(160000, self.iconSize + 40)
        listWdg.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        listWdg.customContextMenuRequested.connect(self.contextMenu)

        self.listWdg = listWdg

        # dock the viewer
        dock = stateAwareQDockWidget(bLUeTop.Gui.window)
        dock.setWidget(listWdg)
        dock.setWindowFlags(listWdg.windowFlags())
        dock.setWindowTitle(listWdg.windowTitle())
        self.listViewDock = dock

        # add a titleBar widget to dock
        tbw = QWidget()
        dock.setTitleBarWidget(tbw)
        hl = QHBoxLayout()
        hl.setContentsMargins(QMargins(2,2,2,2))
        self.titleLabel = QLabel()
        self.titleLabel.setMargin(2)
        self.selLabel = QLabel()
        self.selLabel.setMargin(2)
        font = self.titleLabel.font()
        metrics = QFontMetrics(font)
        h = metrics.height()
        self.titleLabel.setFixedSize(250, h)
        hl.addWidget(self.titleLabel)
        hl.addStretch()
        hl.addWidget(self.selLabel)
        self.selButton = QbLUePushButton('Select All')
        hl.addWidget(self.selButton)
        self.refButton = QbLUePushButton('Refresh')
        hl.addWidget(self.refButton)
        nameButton = sortPushButton('File name')
        hl.addWidget(nameButton)
        takenButton = sortPushButton('Taken date')
        hl.addWidget(takenButton)
        creationButton = sortPushButton('Creation Date')
        hl.addWidget(creationButton)
        tbw.setLayout(hl)

        self.sortButtons = (nameButton, takenButton, creationButton)

        nameButton.pressed.connect(lambda : self.sort(0))
        takenButton.pressed.connect(lambda: self.sort(1))
        creationButton.pressed.connect(lambda: self.sort(2))

        self.selButton.pressed.connect(self.listWdg.selectAll)

        bLUeTop.Gui.window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        self.listWdg.setWhatsThis(
            """<b>File Explorer</b><br>
            To <b>open an image</b> drag and drop its icon into the main window.<br>
            To <b> sort icons</b> use the 3 rightmost buttons.
            To <b>copy</b> selected files, right clic an icon and use the context menu.
            """
        )  # end setWhatsThis

    def initCMenu(self):
        """
        Context menu initialization
        """
        menu = QMenu()
        actionCopy = QAction("Copy Selected File(s) to Folder...", None)
        menu.addAction(actionCopy)
        actionCopy.triggered.connect(self.hCopy)
        menu.actionCopy = actionCopy
        # selection change slot
        def h():
            selectionList = self.listWdg.selectedItems()
            menu.actionCopy.setEnabled(bool(selectionList))
            self.selLabel.setText('%d file(s) selected' % len(selectionList))
        self.listWdg.itemSelectionChanged.connect(h)
        self.cMenu = menu

    def contextMenu(self, pos):
        """
        customContextMenuRequested signal slot
        """
        globalPos = self.listWdg.mapToGlobal(pos)
        self.cMenu.exec(globalPos)

    def hCopy(self):
        """
        actionCopy slot
        """
        sel = self.listWdg.selectedItems()
        lastDir = str(bLUeTop.Gui.window.settings.value('paths/dlgcopydir', '.'))
        destDir = QFileDialog.getExistingDirectory(bLUeTop.Gui.window,
                                                   "Open Directory",
                                                   lastDir,
                                                   QFileDialog.Option.ShowDirsOnly
                                                   )
        try:
            if not os.path.isdir(destDir):
                raise(OSError('Destination Directory does not exist'))
            count = 0
            for item in sel:
                if os.path.exists(os.path.join(destDir, item.text())):
                    dlgWarn('Cannot copy %s' % item.text())
                else:
                    sourceFile = os.path.join(self.currentDir, item.text())
                    shutil.copy2(sourceFile, destDir)  # copy metadata too
                    count += 1
            dlgWarn('       %d file(s) copied          ' % count)
            bLUeTop.Gui.window.settings.setValue('paths/dlgcopydir', destDir)
        except OSError as e:
            dlgWarn('      Error copying files     ', info=str(e))

    def doGen(self, folder):
        fileListGen = (path.join(folder, filename) for filename in listdir(folder) if
                       isfile(path.join(folder, filename)) and (
                               filename.endswith(IMAGE_FILE_EXTENSIONS) or
                               filename.endswith(RAW_FILE_EXTENSIONS)) or
                               filename.endswith(BLUE_FILE_EXTENSIONS) or
                                filename.endswith(HEIF_FILE_EXTENSIONS)
                       )
        self.currentDir = folder
        return fileListGen

    def sort(self, index):
        self.listWdg.sortItems(self.listWdg.sortOrder[index])

        self.listWdg.sortOrder[index] = Qt.SortOrder.DescendingOrder\
                                        if self.listWdg.sortOrder[index] == Qt.SortOrder.AscendingOrder\
                                        else Qt.SortOrder.AscendingOrder

        for btn in self.sortButtons:
            btn.setText(btn.baseText)  # reset all buttons
        self.sortButtons[index].setText(self.sortButtons[index].baseText + ' \u25bc' if self.listWdg.sortOrder[index] == Qt.SortOrder.DescendingOrder else self.sortButtons[index].baseText + ' \u25b2')

        selectionList = self.listWdg.selectedItems()
        # scroll to selection
        if bool(selectionList):
            self.listWdg.scrollToItem(selectionList[0], QAbstractItemView.ScrollHint.PositionAtTop)

    def playViewer(self, folder):
        """
        Displays all images from folder.

        :param folder: path to folder
        :type folder: str
        """
        try:
            fileListGen = self.doGen(folder)
            self.listWdg.clear()
            self.listViewDock.setWindowTitle(folder)
            self.titleLabel.setText(folder)
            self.listWdg.showMaximized()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()  # to immediately update the listWdg and the GUI
            loader(fileListGen, self).load()
        except (FileNotFoundError, NotADirectoryError):
            pass
        finally:
            QApplication.restoreOverrideCursor()

    def showViewer(self, aDir, forcevisible=True):
        """
        Updates the file list if the directory changes

        :param aDir: new dir
        :type aDir: str
        :param forcevisible:
        :type forcevisible: bool
        """
        if forcevisible:
            self.listViewDock.show()
            self.fileDlgDock.show()

        if self.currentDir != aDir:
            self.fileDlg.setDirectory(aDir)
            self.fileDlg.setWindowTitle(aDir)
            self.fileDlgDock.setWindowTitle(self.fileDlg.windowTitle())
            self.fileDlg.repaint()  # needed to immediately display the file list
            self.playViewer(aDir)

    def refreshViewer(self):
        """
        Reloads the current directory
        """
        self.playViewer(self.currentDir)

    def recordDir(self):
        """
        fileDlg finished signal slot
        """
        self.currentToSettings(mainWin=bLUeTop.Gui.window)
