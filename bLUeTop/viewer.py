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
import tifffile
from ast import literal_eval
from os import walk, path, listdir
from os.path import basename, isfile
from re import search

from PySide6.QtCore import Qt, QMimeData, QPoint, QSize, QBuffer, QIODevice, QRect, QEventLoop, \
    QTimer, QObject
from PySide6.QtGui import QKeySequence, QImage, QDrag, QAction, QPixmap, QIcon
from PySide6.QtWidgets import QMainWindow, QSizePolicy, QMenu, QListWidget, QAbstractItemView, \
    QApplication, QListWidgetItem, QFileDialog

from bLUeTop import exiftool
from bLUeTop.MarkedImg import imImage
import bLUeTop.Gui
from bLUeTop.imLabel import slideshowLabel
from bLUeTop.utils import stateAwareQDockWidget, imagej_description_metadata, compat
from bLUeGui.dialog import IMAGE_FILE_EXTENSIONS, RAW_FILE_EXTENSIONS, BLUE_FILE_EXTENSIONS, dlgWarn

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


class dragQListWidget(QListWidget):
    """
    This class is used by playViewer() instead of QListWidget.
    It reimplements mousePressEvent and inits
    a convenient QMimeData object for drag and drop events.
    """

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
    Image loader
    """
    def __init__(self, gen, viewerInstance):
        """
       :param gen: generator of image file names
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
        self.viewerInstance.listWdg.repaint()  # usual update method is masked by QAbstractItemView.update()

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
                                                               "colorspace", "profileDescription",
                                                               "orientation",
                                                               "model",
                                                               "rating", "FileCreateDate"),
                                                           createsidecar=False)
                    except ValueError:
                        metadata = {}

                    # get image info
                    tmp = [value for key, value in metadata.items() if 'orientation' in key.lower()]
                    orientation = tmp[0] if tmp else 1  # metadata.get("EXIF:Orientation", 1)
                    # EXIF:DateTimeOriginal seems to be missing in many files
                    tmp = [value for key, value in metadata.items() if 'date' in key.lower()]
                    date = tmp[0] if tmp else ''  # metadata.get("EXIF:ModifyDate", '')
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
                    if img.isNull() and filename[-4:] in BLUE_FILE_EXTENSIONS:
                        tfile = tifffile.TiffFile(filename)
                        meta_dict = imagej_description_metadata(tfile.pages[0].description)
                        version = meta_dict.get('version', 'unknown')
                        v = meta_dict.get('thumbnailimage', None)
                        if v is not None:
                            v = compat(v, version)
                            ba = pickle.loads(literal_eval(v))
                            buffer = QBuffer(ba)
                            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
                            img = QImage()
                            img.load(buffer, 'JPG')
                    # everything else fails, read file !
                    if img.isNull():
                        img = QImage(filename).scaled(QSize(viewer.iconSize, viewer.iconSize),
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
                    item = QListWidgetItem(QIcon(pxm), basename(filename))  # icon, text
                    item.setToolTip(basename(filename) + ' ' + date + ' ' + rating)
                    # set item mimeData to get filename=item.data(Qt.UserRole)[0] transformation=item.data(Qt.UserRole)[1]
                    item.setData(Qt.ItemDataRole.UserRole, (filename, transformation))
                    item.setSizeHint(QSize(viewer.iconSize, viewer.iconSize + 15))
                    self.populate([item])
                except (OSError, IOError, ValueError, tifffile.TiffFileError, KeyError, SyntaxError,
                        ModuleNotFoundError, pickle.UnpicklingError) as ex:
                    continue
                except StopIteration:
                    break


class viewer(QObject):
    """
    Dockable folder browser
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
        self.dock, self.fileDlg = (None,) * 2
        self.fileDlg = None
        self.initWins()
        self.initCMenu()

    def currentToSettings(self, mainWin=None):
        mainWin.settings.setValue('paths/dlgdir', self.currentDir)
        mainWin.settings.setValue('mainwindow/explistwdg', self.dock.isVisible())
        mainWin.settings.setValue('mainwindow/expfiledlg', self.fileDlg.isVisible())

    def currentFromSettings(self, mainWin=None):
        return mainWin.settings.value('paths/dlgdir', '.')

    def initWins(self):
        # init listWdg
        listWdg = dragQListWidget()
        listWdg.setWrapping(False)
        listWdg.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        listWdg.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        listWdg.setViewMode(QListWidget.ViewMode.IconMode)
        listWdg.setIconSize(QSize(self.iconSize, self.iconSize))
        listWdg.setMaximumSize(160000, self.iconSize + 40)
        listWdg.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        # listWdg.customContextMenuRequested.connect(self.contextMenu)  # disabled 12/11/24

        # dock the viewer
        dock = stateAwareQDockWidget(bLUeTop.Gui.window)
        dock.setWidget(listWdg)
        dock.setWindowFlags(listWdg.windowFlags())
        dock.setWindowTitle(listWdg.windowTitle())
        self.dock = dock

        bLUeTop.Gui.window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        self.listWdg = listWdg
        self.listWdg.setWhatsThis(
            """<b>Library Viewer</b><br>
            To <b>open an image</b> drag and drop its icon into the main window.<br>
            """
        )  # end setWhatsThis

    def initCMenu(self):
        """
        Context menu initialization
        """
        menu = QMenu()
        menu.addAction("Copy Image to Clipboard", self.hCopy)
        subMenuRating = menu.addMenu('Rating')
        for i in range(6):
            action = QAction(str(i), None)
            subMenuRating.addAction(action)
            action.triggered.connect(
                lambda checked=False, action=action: self.setRating(action))  # named arg checked is sent
        self.cMenu = menu

    def contextMenu(self, pos):
        globalPos = self.listWdg.mapToGlobal(pos)
        self.cMenu.exec(globalPos)

    def hCopy(self):
        """
        Slot for copy files
        Currently unsused
        """
        sel = self.listWdg.selectedItems()
        destDir = QFileDialog.getExistingDirectory(None,
                                               "Open Directory",
                                                "/home",
                                                QFileDialog.Option.ShowDirsOnly
                                                 )
        #dlg.selectedFiles()[0].absolutePath()
        for item in sel:
            if os.path.exists(os.path.join(destDir, item.text())):
                dlgWarn('error')
            else:
                sourceFile = os.path.join(self.currentDir, item.text())
                shutil.copy(sourceFile, destDir)
        return

    # slot for action rating
    def setRating(self, action):
        # rating : the tag is written into the .mie file; the file is
        # created if needed.
        listWdg = self.listWdg
        sel = listWdg.selectedItems()
        if action.text() in ['0', '1', '2', '3', '4', '5']:
            with exiftool.ExifTool() as e:
                value = int(action.text())
                for item in sel:
                    filename = item.data(Qt.ItemDataRole.UserRole)[0]
                    e.writeXMPTag(filename, 'XMP:rating', value)
                    item.setText(basename(filename) + ''.join(['*'] * value))

    def doGen(self, folder, withsub=False):
        if withsub:
            # browse the directory and its subfolders
            fileListGen = (path.join(dirpath, filename) for (dirpath, dirnames, filenames) in walk(folder) for filename
                           in filenames if
                           filename.endswith(IMAGE_FILE_EXTENSIONS) or
                           filename.endswith(RAW_FILE_EXTENSIONS) or
                           filename.endswith(BLUE_FILE_EXTENSIONS))
        else:
            fileListGen = (path.join(folder, filename) for filename in listdir(folder) if
                           isfile(path.join(folder, filename)) and (
                                   filename.endswith(IMAGE_FILE_EXTENSIONS) or
                                   filename.endswith(RAW_FILE_EXTENSIONS)) or
                                   filename.endswith(BLUE_FILE_EXTENSIONS))
        self.currentDir = folder
        return fileListGen

    def playViewer(self, folder):
        """
        Opens a window and displays all images in a folder.

        :param folder: path to folder
        :type folder: str
        :return: loader instance
        :rtype: thread
        """

        # build generator
        try:
            fileListGen = self.doGen(folder)
            self.listWdg.clear()
            self.dock.setWindowTitle(folder)
            self.listWdg.showMaximized()
            QApplication.processEvents()  # needed for instant showing

            ldr = loader(fileListGen, self)
            ldr.load()

            #self.dock.setWindowTitle(self.currentDir)  # currentDir may be changed by loader
        except (FileNotFoundError, NotADirectoryError):
            pass





