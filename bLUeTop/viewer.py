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
import pickle
import threading
import tifffile
from ast import literal_eval
from os import walk, path, listdir
from os.path import basename, isfile
from re import search
from time import sleep

from PySide2.QtCore import Qt, QUrl, QMimeData, QByteArray, QPoint, QSize, QBuffer, QIODevice, QRect
from PySide2.QtGui import QKeySequence, QImage, QDrag, QPixmap, QIcon, QColor
from PySide2.QtWidgets import QMainWindow, QSizePolicy, QMenu, QListWidget, QAbstractItemView, \
    QApplication, QAction, QListWidgetItem

from bLUeTop import exiftool
from bLUeTop.MarkedImg import imImage
from bLUeTop.QtGui1 import app, window
from bLUeTop.imLabel import slideshowLabel, imageLabel
from bLUeTop.utils import stateAwareQDockWidget, imagej_description_metadata, compat
from bLUeGui.dialog import IMAGE_FILE_EXTENSIONS, RAW_FILE_EXTENSIONS, BLUE_FILE_EXTENSIONS

# global variable recording diaporama state
isSuspended = False


def playDiaporama(diaporamaGenerator, parent=None):
    """
    Open a new window and play a slide show.
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
    label = slideshowLabel(mainForm=parent)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    label.img = None
    label.prevImg = None
    label.prevOpacity = 0.0
    newWin.setCentralWidget(label)
    newWin.showFullScreen()
    # Pause key shortcut
    actionEsc = QAction('Pause', None)
    actionEsc.setShortcut(QKeySequence(Qt.Key_Escape))
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
            playDiaporama(diaporamaGenerator, parent=window)
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
        action2.setChecked(newWin.windowState() & Qt.WindowFullScreen)
        action3 = QAction('Resume', None)
        action3.setEnabled(isSuspended)
        for action in [actionEsc, action2, action3]:
            menu.addAction(action)
            action.triggered.connect(
                lambda checked=False, name=action: contextMenuHandler(name))  # named arg checked is sent
        subMenuRating = menu.addMenu('Rating')
        for i in range(6):
            action = QAction(str(i), None)
            subMenuRating.addAction(action)
            action.triggered.connect(
                lambda checked=False, name=action: contextMenuHandler(name))  # named arg checked is sent
        menu.exec_(position)

    def testPaused():
        app.processEvents()
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
    window.modeDiaporama = True
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
                    r = search("\d", rt)
                    if r is not None:
                        rating = int(r.group(0))
                except ValueError:
                    rating = 5
            # don't display image with low rating
            if rating < 2:
                # app.processEvents()
                continue
            imImg = imImage.loadImageFromFile(name, createsidecar=False, cmsConfigure=True, window=window)
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
                for i in range(81):
                    label.prevOpacity = 1.0 - i * 0.0125  # last prevOpacity must be 0
                    label.repaint()
            label.repaint()  # mandatory to display first image
            if testPaused():
                break
            sleep(2)
            if testPaused():
                break
            label.prevImg = label.img
            gc.collect()
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
            window.modeDiaporama = False
            raise
    window.modeDiaporama = False


class dragQListWidget(QListWidget):
    """
    This class is used by playViewer() instead of QListWidget.
    It reimplements mousePressEvent and inits
    a convenient QMimeData object for drag and drop events.
    """

    def mousePressEvent(self, event):
        # call to super needed for selections
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            drag = QDrag(self)
            mimeData = QMimeData()
            item = self.itemAt(event.pos())
            if item is None:
                return
            mimeData.setText(item.data(Qt.UserRole)[0])  # should be path to file
            drag.setMimeData(mimeData)
            # set dragging pixmap
            drag.setPixmap(item.icon().pixmap(QSize(160, 120)))
            # roughly center the cursor relative to pixmap
            drag.setHotSpot(QPoint(60, 60))
            dropAction = drag.exec_()


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
                        profile, metadata = e.get_metadata(filename,
                                                           tags=(
                                                               "colorspace", "profileDescription", "orientation",
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
                                              thumbname='PreviewImage')  # the order is important : for jpeg PreviewImage is full sized !
                    # may be a bLU file
                    if img.isNull() and filename[-4:] in BLUE_FILE_EXTENSIONS:
                        tfile = tifffile.TiffFile(filename)
                        meta_dict = imagej_description_metadata(tfile.pages[0].is_imagej)
                        version = meta_dict.get('version', 'unknown')
                        v = meta_dict.get('thumbnailimage', None)
                        if v is not None:
                            v = compat(v, version)
                            ba = pickle.loads(literal_eval(v))
                            buffer = QBuffer(ba)
                            buffer.open(QIODevice.ReadOnly)
                            img = QImage()
                            img.load(buffer, 'JPG')

                    if img.isNull():
                        img = QImage(QSize(160, 120), QImage.Format_RGB888)
                        img.fill(QColor(128, 128, 140))

                    # remove possible black borders, except for .NEF
                    if filename[-3:] not in ['nef', 'NEF']:
                        bBorder = 7
                        img = img.copy(QRect(0, bBorder, img.width(), img.height() - 2 * bBorder))
                    pxm = QPixmap.fromImage(img)
                    if not transformation.isIdentity():
                        pxm = pxm.transformed(transformation)

                    # set item caption and tooltip
                    item = QListWidgetItem(QIcon(pxm), basename(filename))
                    item.setToolTip(basename(filename) + ' ' + date + ' ' + rating)
                    # set item mimeData to get filename=item.data(Qt.UserRole)[0] transformation=item.data(Qt.UserRole)[1]
                    item.setData(Qt.UserRole, (filename, transformation))
                    self.wdg.addItem(item)
                except (OSError, IOError, ValueError, tifffile.TiffFileError, KeyError, SyntaxError,
                        ModuleNotFoundError, pickle.UnpicklingError) as ex:
                    continue
                except Exception as ex:  # StopIteration and anything else for clean exit
                    break


class viewer:
    """
    Folder browser
    """
    # current viewer instance
    instance = None
    iconSize = 80

    @classmethod
    def getViewerInstance(cls, mainWin=None):
        """
        Returns a unique viewer instance : a new instance
        is created only if there exists no instance yet.
        @param mainWin: should be the app main window
        @type mainWin: QMainWindow
        @return: viewer instance
        @rtype: viewer
        """
        if cls.instance is None:
            cls.instance = viewer(mainWin=mainWin)
        elif cls.instance.mainWin is not mainWin:
            raise ValueError("getViewer: wrong main form")
        return cls.instance

    def __init__(self, mainWin=None):
        """

        @param mainWin: should be the app main window
        @type mainWin:  QMainWindow
        """
        self.mainWin = mainWin
        # init form
        self.initWins()
        actionSub = QAction('Show SubFolders', None)
        actionSub.setCheckable(True)
        actionSub.setChecked(False)
        actionSub.triggered.connect(
            lambda checked=False, action=actionSub: self.hSubfolders(action))  # named arg checked is always sent
        self.actionSub = actionSub
        # init context menu
        self.initCMenu()

    def initWins(self):
        # viewer main form
        newWin = QMainWindow(self.mainWin)
        newWin.setAttribute(Qt.WA_DeleteOnClose)
        newWin.setContextMenuPolicy(Qt.CustomContextMenu)
        self.newWin = newWin
        # image list
        listWdg = dragQListWidget()
        listWdg.setWrapping(False)
        listWdg.setSelectionMode(QAbstractItemView.ExtendedSelection)
        listWdg.setContextMenuPolicy(Qt.CustomContextMenu)
        listWdg.label = None
        listWdg.setViewMode(QListWidget.IconMode)
        # set icon and listWdg sizes
        listWdg.setIconSize(QSize(self.iconSize, self.iconSize))
        listWdg.setMaximumSize(160000, self.iconSize + 40)
        listWdg.setDragDropMode(QAbstractItemView.DragDrop)
        listWdg.customContextMenuRequested.connect(self.contextMenu)
        # dock the form
        dock = stateAwareQDockWidget(window)
        dock.setWidget(newWin)
        dock.setWindowFlags(newWin.windowFlags())
        dock.setWindowTitle(newWin.windowTitle())
        dock.setAttribute(Qt.WA_DeleteOnClose)
        self.dock = dock
        window.addDockWidget(Qt.BottomDockWidgetArea, dock)
        newWin.setCentralWidget(listWdg)
        self.listWdg = listWdg
        self.newWin.setWhatsThis(
            """<b>Library Viewer</b><br>
            To <b>open context menu</b> right click on an icon or a selection.<br>
            To <b>open an image</b> drag it onto the main window.<br>
            <b>Rating</b> is shown as 0 to 5 stars below each icon.<br>
            """
        )  # end setWhatsThis

    def initCMenu(self):
        """
        Context menu initialization
        """
        menu = QMenu()
        menu.addAction("Copy Image to Clipboard", self.hCopy)
        menu.addAction(self.actionSub)
        subMenuRating = menu.addMenu('Rating')
        for i in range(6):
            action = QAction(str(i), None)
            subMenuRating.addAction(action)
            action.triggered.connect(
                lambda checked=False, action=action: self.setRating(action))  # named arg checked is sent
        self.cMenu = menu

    def contextMenu(self, pos):
        globalPos = self.listWdg.mapToGlobal(pos)
        self.cMenu.exec_(globalPos)

    def viewImage(self):
        """
        display full size image in a new window
        Unused yet
        """
        parent = window
        newWin = QMainWindow(parent)
        newWin.setAttribute(Qt.WA_DeleteOnClose)
        newWin.setContextMenuPolicy(Qt.CustomContextMenu)
        label = imageLabel(parent=newWin)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.img = None
        newWin.setCentralWidget(label)
        sel = self.listWdg.selectedItems()
        item = sel[0]
        filename = item.data(Qt.UserRole)[0]
        newWin.setWindowTitle(filename)
        imImg = imImage.loadImageFromFile(filename, createsidecar=False, window=window)
        label.img = imImg
        newWin.showMaximized()

    def hCopy(self):
        """
        # slot for action copy_to_clipboard
        """
        sel = self.listWdg.selectedItems()
        ####################
        # test code
        l = []
        for item in sel:
            # get url from path
            l.append(QUrl.fromLocalFile(item.data(Qt.UserRole)[0]))
        # init clipboard data
        q = QMimeData()
        # set some Windows magic values for copying files from system clipboard : Don't modify
        # 1 : copy; 2 : move
        q.setData("Preferred DropEffect", QByteArray(1, "2"))
        q.setUrls(l)
        # end of test code
        #####################
        # copy image to clipboard
        item = sel[0]
        filename = item.data(Qt.UserRole)[0]
        if filename.endswith(IMAGE_FILE_EXTENSIONS):
            q.setImageData(QImage(sel[0].data(Qt.UserRole)[0]))
        QApplication.clipboard().clear()
        QApplication.clipboard().setMimeData(q)

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
                    filename = item.data(Qt.UserRole)[0]
                    e.writeXMPTag(filename, 'XMP:rating', value)
                    item.setText(basename(filename) + '\n' + ''.join(['*'] * value))

    # slot for subfolders browsing
    def hSubfolders(self, action):
        self.listWdg.clear()
        fileListGen = self.doGen(self.folder, withsub=action.isChecked())
        # launch loader instance
        thr = loader(fileListGen, self.listWdg)
        thr.start()

    def doGen(self, folder, withsub=False):
        self.folder = folder
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
        return fileListGen

    def playViewer(self, folder):
        """
        Opens a window and displays all images in a folder.
        The images are loaded asynchronously by a separate thread.
        @param folder: path to folder
        @type folder: str
        """
        if self.dock.isClosed:
            # reinit form
            self.initWins()
        else:
            # clear form
            self.listWdg.clear()
        self.newWin.showMaximized()
        # build generator:
        fileListGen = self.doGen(folder, withsub=self.actionSub.isChecked())
        self.dock.setWindowTitle(folder)
        # launch loader instance
        thr = loader(fileListGen, self.listWdg)
        thr.start()
