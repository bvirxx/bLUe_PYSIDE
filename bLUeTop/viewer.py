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
from os import walk, path, listdir
from os.path import basename, isfile
from re import search
from time import sleep

from PySide2.QtCore import Qt, QUrl, QMimeData, QByteArray, QPoint, QSize
from PySide2.QtGui import QKeySequence, QImage, QDrag
from PySide2.QtWidgets import QMainWindow, QLabel, QSizePolicy, QAction, QMenu, QListWidget, QAbstractItemView, \
    QApplication

from bLUeTop import exiftool
from bLUeTop.QtGui1 import app, window
from bLUeTop.imLabel import imageLabel
from bLUeTop.utils import loader, stateAwareQDockWidget
from bLUeGui.dialog import IMAGE_FILE_EXTENSIONS, RAW_FILE_EXTENSIONS

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
    label = imageLabel(mainForm=parent)  #QLabel()
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    label.img = None
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
    from bLUe import loadImageFromFile
    window.modeDiaporama = True
    while True:
        if isSuspended:
            newWin.setWindowTitle(newWin.windowTitle() + ' Paused')
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
                app.processEvents()
            imImg = loadImageFromFile(name, createsidecar=False)
            # zoom might be modified by the mouse wheel : remember
            if label.img is not None:
                imImg.Zoom_coeff = label.img.Zoom_coeff
            coeff = imImg.resize_coeff(label)
            imImg.yOffset -= (imImg.height() * coeff - label.height()) / 2.0
            imImg.xOffset -= (imImg.width() * coeff - label.width()) / 2.0
            app.processEvents()
            if isSuspended:
                newWin.setWindowTitle(newWin.windowTitle() + ' Paused')
                break
            newWin.setWindowTitle(parent.tr('Slide show') + ' ' + name + ' ' + ' '.join(['*'] * imImg.meta.rating))
            gc.collect()
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
            window.modeDiaporama = False
            raise
        app.processEvents()
    window.modeDiaporama = False


class dragQListWidget(QListWidget):
    """
    This Class is used by playViewer instead of QListWidget.
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


class viewer:
    """
    Folder browser
    """
    # current viewer instance
    instance = None
    iconSize = 100

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
        actionSub.triggered.connect(lambda checked=False, action=actionSub: self.hSubfolders(action))  # named arg checked is always sent
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
        listWdg.setMaximumSize(160000, self.iconSize+20)
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
        from bLUe import loadImageFromFile
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
        imImg = loadImageFromFile(filename, createsidecar=False)
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
        q.setData("Preferred DropEffect", QByteArray("2"))
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
            fileListGen = (path.join(dirpath, filename) for (dirpath, dirnames, filenames) in walk(folder) for filename in
                           filenames if
                           filename.endswith(IMAGE_FILE_EXTENSIONS) or filename.endswith(RAW_FILE_EXTENSIONS))
        else:
            fileListGen = (path.join(folder, filename) for filename in listdir(folder) if
                           isfile(path.join(folder, filename)) and (
                                       filename.endswith(IMAGE_FILE_EXTENSIONS) or filename.endswith(
                                   RAW_FILE_EXTENSIONS)))
        return fileListGen

    def playViewer(self, folder):
        """
        Opens a window and displays all images from a folder.
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
        # build generator:
        fileListGen = self.doGen(folder, withsub=self.actionSub.isChecked())
        self.dock.setWindowTitle(folder)
        self.newWin.showMaximized()
        # launch loader instance
        thr = loader(fileListGen, self.listWdg)
        thr.start()
