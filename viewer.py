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
from os.path import basename
from re import search
from time import sleep

from PySide2.QtCore import Qt, QUrl, QMimeData, QByteArray, QPoint, QSize
from PySide2.QtGui import QKeySequence, QImage, QPixmap
from PySide2.QtWidgets import QMainWindow, QLabel, QSizePolicy, QAction, QMenu, QListWidget, QAbstractItemView, \
    QApplication

import exiftool
from QtGui1 import app, window
from imgconvert import QImageBuffer
from utils import loader

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
    label = QLabel()
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    label.img = None
    newWin.setCentralWidget(label)
    # newWin.showMaximized()
    newWin.showFullScreen()
    from bLUe import set_event_handlers
    set_event_handlers(label)
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
    # play diaporama
    from bLUe import loadImageFromFile
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
            try:
                with exiftool.ExifTool() as e:
                    rt = e.readXMPTag(name, 'XMP:rating')
                    r = search("\d", rt)
                    if r is not None:
                        rating = int(r.group(0))
            except ValueError:
                rating = 5
            # don't display image with low rating
            if rating < 2:
                app.processEvents()
                #continue
            imImg = loadImageFromFile(name, createsidecar=False)
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
            raise
        app.processEvents()
    newWin.setToolTip("Esc to exit full screen mode")
    newWin.setWhatsThis(
        """ Slide Show
        The diaporama cycles through the starting directory and its subfolders to display images. \
        Photos rated 0 or 1 star are not shown. By default, all photos are rated 5 stars.
        Hit the Esc key to exit full screen mode and pause. Use the Context Menu for rating and resuming. \
        The rating is saved in the .mie sidecar and the image file is not modified.
        """
    )  # end of setWhatsThis

def playViewer(fileListGen, dir, parent=None):
    """
    Opens a window and displays all images from a folder.
    The images are loaded asynchronously by a separate thread.
    @param fileListGen: file name generator
    @type fileListGen: generator object
    @param dir:
    @type dir:
    @param parent:
    @type parent:
    """
    # init form
    newWin = QMainWindow(parent)
    newWin.setAttribute(Qt.WA_DeleteOnClose)
    newWin.setContextMenuPolicy(Qt.CustomContextMenu)
    newWin.setWindowTitle(parent.tr('Library Viewer ' + dir))
    # init viewer
    wdg = QListWidget(parent=parent)
    wdg.setSelectionMode(QAbstractItemView.ExtendedSelection)
    wdg.setContextMenuPolicy(Qt.CustomContextMenu)
    wdg.label = None
    # slot for action copy_to_clipboard
    def hCopy():
        sel = wdg.selectedItems()
        l = []
        for item in sel:
            # get url from path
            l.append(QUrl.fromLocalFile(item.data(Qt.UserRole)))
        # init clipboard data
        q = QMimeData()
        # set some Windows magic values for copying files from system clipboard : Don't modify
        # 1 : copy; 2 : move
        q.setData("Preferred DropEffect", QByteArray("2"))
        q.setUrls(l)
        QApplication.clipboard().clear()
        QApplication.clipboard().setMimeData(q)
    # slot for action zoom
    def hZoom():
        sel = wdg.selectedItems()
        l = []
        # build list of file paths
        for item in sel:
            l.append(item.data(Qt.UserRole)[0])
        if wdg.label is None:
            wdg.label = QLabel(parent=wdg)
            wdg.label.setMaximumSize(500, 500)
        # get selected item bounding rect (global coords)
        rect = wdg.visualItemRect(sel[0])
        # move label close to rect while keeping it visible
        point = QPoint(min(rect.left(), wdg.viewport().width() - 500), min(rect.top(), wdg.viewport().height() - 500))
        wdg.label.move(wdg.mapFromGlobal(point))
        # get correctly oriented image
        img = QImage(l[0]).transformed(item.data(Qt.UserRole)[1])
        img = img.scaled(500, 500, Qt.KeepAspectRatio)
        wdg.label.setPixmap(QPixmap.fromImage(img))
        wdg.label.show()
    # slot for action rating
    def setRating(action):
        # rating : the tag is written into the .mie file; the file is
        # created if needed.
        sel = wdg.selectedItems()
        if action.text() in ['0', '1', '2', '3', '4', '5']:
            with exiftool.ExifTool() as e:
                value = int(action.text())
                for item in sel:
                    filename = item.data(Qt.UserRole)[0]
                    e.writeXMPTag(filename, 'XMP:rating', value)
                    item.setText(basename(filename) + '\n' + ''.join(['*']*value))
    # context menu
    def contextMenu(pos):
        globalPos = wdg.mapToGlobal(pos)
        menu = QMenu()
        menu.addAction("Copy to Clipboard", hCopy)
        menu.addAction("Zoom", hZoom)
        subMenuRating = menu.addMenu('Rating')
        for i in range(6):
            action = QAction(str(i), None)
            subMenuRating.addAction(action)
            action.triggered.connect(
                lambda checked=False, action=action: setRating(action))  # named arg checked is sent
        menu.exec_(globalPos)
    # selection change
    def hChange():
        if wdg.label is not None:
            wdg.label.hide()
    wdg.customContextMenuRequested.connect(contextMenu)
    wdg.itemSelectionChanged.connect(hChange)
    wdg.setViewMode(QListWidget.IconMode)
    wdg.setIconSize(QSize(150, 150))
    newWin.setCentralWidget(wdg)
    newWin.showMaximized()
    newWin.setWhatsThis(
"""Library Viewer
Rating is shown as 0 to 5 stars below each icon.
Right click on an icon or a selection to open the context menu.
"""
                        ) # end setWhatsThis
    # launch loader instance
    thr = loader(fileListGen, wdg)
    thr.start()