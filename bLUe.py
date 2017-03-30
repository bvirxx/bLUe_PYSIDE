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

"""
bLUe - Photo editing software.

Copyright (C) 2017  Bernard Virot <bernard.virot@libertysurf.fr>

With Blue you can enhance and correct the colors of your photos in a few clicks.
No need for complex tools such as lasso, magic wand or masks.
bLUe interactively constructs 3D LUTs (Look Up Tables), adjusting the exact set
of colors you chose.

3D LUTs are widely used by professional film makers, but the lack of
interactive tools maked them poorly useful for photo enhancement, as the shooting conditions
can vary widely from an image to another. With bLUe, in a few clicks, you select the set of
colors to modify, the corresponding 3D LUT is automatically built and applied to the image.
Then, you can fine tune it as you want.
"""

from grabcut import segmentForm
import sys
from PySide.QtCore import Qt, QRect, QEvent, QDir
from PySide.QtGui import QColor,QPainter, QApplication, QMenu, QAction, QCursor, QFileDialog, QMessageBox, QMainWindow, QLabel, QDockWidget, QSizePolicy
from QtGui1 import app, window
import exiftool
from imgconvert import *
from MarkedImg import imImage, metadata

from graphicsLUT import graphicsForm
from graphicsLUT3D import graphicsForm3DLUT
from LUT3D import LUTSIZE
from colorModels import cmHSP, cmHSB
import icc
from os import path

CONST_FG_COLOR = QColor(255, 255, 255,128)
CONST_BG_COLOR = QColor(255, 0, 255,128)

thickness = 30*4
State = {'drag' : False, 'drawing' : False , 'tool_rect' : False, 'rect_over' : False, 'ix' : 0, 'iy' :0, 'rawMask' : None}
# application windows
#Wins = {'3D_LUT': None, 'Brightness-Contrast':None}
rect_or_mask = 0


mask=None
mask_s=None


# paintEvent painter and background color
qp=QPainter()
defaultBgColor=QColor(191,191,191)

def paintEvent(widg, e) :
    """
    Paint event handler for widgets that display a mImage object.
    The widget must have a valid img attribute of type QImage.
    :param widg: widget object with a img attribute
    :param e: paint event
    """
    if not hasattr(widg, 'img'):
        raise ValueError("paintEvent : no image defined")
    if widg.img is None:
        raise ValueError("paintEvent : no image set")
    mimg = widg.img
    r = mimg.resize_coeff(widg)
    qp.begin(widg)
    # pen for selection rectangle
    qp.setPen(QColor(0,255,0))
    # background
    qp.fillRect(QRect(0, 0, widg.width() , widg.height() ), defaultBgColor)
    # draw layers
    for layer in mimg.layersStack :
        if layer.visible:
            qp.setOpacity(layer.opacity)
            if layer.qPixmap is not None:
                qp.drawPixmap(QRect(mimg.xOffset,mimg.yOffset, mimg.width()*r, mimg.height()*r), # target rect
                           layer.transfer() #layer.qPixmap
                         )
            else:
                qp.drawImage(QRect(mimg.xOffset, mimg.yOffset, mimg.width() * r , mimg.height() * r ), # target rect
                              layer  # layer.qPixmap
                              )
    # draw selection rectangle
    if (mimg.getActiveLayer().visible) and (mimg.getActiveLayer().rect is not None ):
        rct = mimg.getActiveLayer().rect
        qp.drawRect(rct.left()*r + mimg.xOffset, rct.top()*r +mimg.yOffset, rct.width()*r, rct.height()*r)
    #if mimg.mask is not None :
        #qp.drawImage(QRect(mimg.xOffset, mimg.yOffset, mimg.width * r-10, mimg.height * r  -10), mimg.mask)
    qp.end()

# mouse event handler for image widgets (dynamically set attribute widget.img, currently label and label_2)

pressed=False
clicked = True
def mouseEvent(widget, event) :
    """
    mouse event handler for QLabel object.
    It handles image positionning and zooming, and
    tool interactions with the active layer.
    :param widget:
    :param event:
    """
    global rect_or_mask, mask, mask_s, Mimg_1, pressed, clicked

    img= widget.img
    layer = img.getActiveLayer()
    r = img.resize_coeff(widget)
    x, y = event.x(), event.y()
    # read keyboard modifiers
    modifier = QApplication.keyboardModifiers()
    """
    if modifier == Qt.ControlModifier:
        if event.type() == QEvent.MouseButtonPress:
            pass
            #showResult(Mimg_p, Mimg_1, turn)
            #turn = (turn + 1) % 2
        return
    """
    # press event
    if event.type() == QEvent.MouseButtonPress :
        pressed=True
        if event.button() == Qt.LeftButton:
            clicked=True
        """
        elif event.button() == Qt.RightButton:
            #State['drag'] = True
            if not State['rect_over']:
                print("first draw rectangle \n")
            else:
                pass
                # State['drawing'] = True
                # # cv2.circle(img.cv2Img, (int(x/r), int(y/r)), thickness, value['color'], -1)
                # cv2.circle(img.mask, (int(x/r), int(y/r)), thickness, value['val'], -1)
                # rect_or_mask = 1
                # mask = cv2.bitwise_or(img.mask, mask)
                # do_grabcut(Mimg_p, Mimg_1, preview=P_SIZE)
        """
        # recording of possible move beginning coordinates
        State['ix'], State['iy'] = x, y
        State['ix_begin'], State['iy_begin'] = x, y
    elif event.type() == QEvent.MouseMove :
        clicked=False
        if pressed :
            # button pressed
            if img.isMouseSelectable:
                # marquee
                if window.btnValues['rectangle']:
                    layer.rect = QRect(min(State['ix_begin'], x)/r -img.xOffset/r, min(State['iy_begin'], y)/r - img.yOffset/r, abs(State['ix_begin'] - x)/r, abs(State['iy_begin'] - y)/r)
                    rect_or_mask = 0
                # brush
                elif (window.btnValues['drawFG'] or window.btnValues['drawBG']):
                    color= CONST_FG_COLOR if window.btnValues['drawFG'] else CONST_BG_COLOR
                    #qp.begin(img._layers['drawlayer'])
                    tmp = layer.mask if layer.maskIsEnabled else layer
                    qp.begin(tmp)
                    qp.setPen(color)
                    qp.setBrush(color)
                    # avoid alpha summation
                    qp.setCompositionMode(qp.CompositionMode_Source)
                    qp.drawEllipse(int(x / r)-img.xOffset/r, int(y / r)- img.yOffset/r, 80, 80)
                    qp.end()
                    rect_or_mask=1
                    window.label.repaint()
                else:
                    img.xOffset+=(x-State['ix'])
                    img.yOffset+=(y-State['iy'])
            else:
                img.xOffset+=(x-State['ix'])
                img.yOffset+=(y-State['iy'])
        #update current coordinates
        State['ix'],State['iy']=x,y
    elif event.type() == QEvent.MouseButtonRelease :
        pressed=False
        if event.button() == Qt.LeftButton:
            # click event
            if img.isMouseSelectable and clicked:
                # adding/removing grid nodes
                # Note : for multilayered images we read pixel color from  the background layer
                c = QColor(img.getActivePixel(State['ix'] / r -  img.xOffset/r, State['iy'] / r - img.yOffset/r))
                cM = QColor(img.getActiveLayer().pixel(State['ix'] / r - img.xOffset / r, State['iy'] / r - img.yOffset / r))
                red, green, blue = c.red(), c.green(), c.blue()
                rM, gM, bM = cM.red(), cM.green(), cM.blue()
                if hasattr(layer, "adjustView") and layer.adjustView is not None:
                    if hasattr(layer.adjustView.widget(), 'selectGridNode'):
                        # adding/removing  nodes
                        layer.adjustView.widget().selectGridNode(red, green, blue, rM,gM,bM)
                window.label.repaint()
            if window.btnValues['rectangle'] and img.isMouseSelectable:
                layer.rect = QRect(min(State['ix_begin'], x)/r-img.xOffset/r, min(State['iy_begin'], y)/r- img.yOffset/r, abs(State['ix_begin'] - x)/r, abs(State['iy_begin'] - y)/r)
                rect_or_mask = 0 #init_with_rect

                rect_or_mask=1 # init with mask
                #print(" Now press the key 'n' a few times until no further change \n")
        """
        elif event.button() == Qt.RightButton:
            State['drag'] = False
            if window.btnValues['drawFG']:
                #State['drawFG'] = False
                #cv2.circle(img.cv2Img, (x, y), thickness, value['color'], -1)
                #cv2.circle(img.mask, (int(x/r), int(y/r)), thickness, value['val'], -1)
                qp.drawEllipse(int(x/r), int(y/r), 10,10)
                rect_or_mask=1
                #cv2.bitwise_or(img.mask, mask)
                #window.label_2.img=do_grabcut(Mimg_p, preview=P_SIZE)
                #tmp = img.mask
                #if not (mask is None):
                    #np.copyto(mask, tmp, where=(tmp == 1))
                window.label.repaint()
        """
    widget.repaint()

def wheelEvent(widget,img, event):
    """
    Mouse wheel event handler for zooming
    imImage objects.
    :param widget: widget displaying image
    :param img: imImage object to display
    :param event: mouse wheel event (type QWheelEvent)
    """
    pos = event.pos()
    # delta unit is 1/8 of degree
    # Most mice have a resolution of 15 degrees
    numDegrees = event.delta() / 8
    numSteps = numDegrees / 150.0
    r=img.Zoom_coeff
    img.Zoom_coeff *= (1.0 + numSteps)
    # correct image offset to keep unchanged the image point
    # under the cursor : (pos - offset) / resize_coeff should be invariant
    img.xOffset = -pos.x() * numSteps + (1.0+numSteps)*img.xOffset
    img.yOffset = -pos.y() * numSteps + (1.0+numSteps)*img.yOffset
    widget.repaint()

def set_event_handler(widg):
    """
    Pythonic way for redefining event handlers, without
    subclassing or overridding.
    :param widg:
    """
    #widg.paintEvent = new.instancemethod(lambda e, wdg=widg : paintEvent(wdg,e), widg, QLabel)
    widg.paintEvent = lambda e, wdg=widg: paintEvent(wdg, e)
    widg.mousePressEvent = lambda e, wdg=widg : mouseEvent(wdg, e)
    widg.mouseMoveEvent = lambda e, wdg=widg : mouseEvent(wdg, e)
    widg.mouseReleaseEvent = lambda e, wdg=widg : mouseEvent(wdg, e)
    widg.wheelEvent = lambda e, wdg=widg : wheelEvent(wdg, wdg.img, e)

def button_change(button):
    if str(button.accessibleName()) == "Fit_Screen" :
        window.label.img.fit_window(window.label)
        window.label.repaint()
        #do_grabcut(Mimg_p, mode=cv2.GC_INIT_WITH_MASK, again=(rect_or_mask==0))
        #window.label_2.img=do_grabcut(window.label.img, mode=cv2.GC_INIT_WITH_MASK, again=(rect_or_mask==0))

def contextMenu(widget):
    qmenu = QMenu("Context menu")
    for k in widget.img.layers.keys():
        action1 = QAction(k, qmenu, checkable=True)
        qmenu.addAction(action1)
        action1.triggered[bool].connect(lambda b, widget=widget, layer=widget.img.layers[k]: toggleLayer(widget, layer,b))
        action1.setChecked(widget.img.layers[k].visible)
    qmenu.exec_(QCursor.pos())

def toggleLayer(widget, layer, b):
    layer.visible = b
    widget.repaint()

def menuFile(name):
    """

    :param name: name of calling action
    :return:
    """
    window._recentFiles = window.settings.value('paths/recent', [])
    # update menu and actions
    updateMenuOpenRecent()

    if name == 'actionOpen' :
        if window.label.img.isModified:
            ret = savingDialog(window.label.img)
            if ret == QMessageBox.Yes:
                save()
            elif ret == QMessageBox.Cancel:
                return
        lastDir = window.settings.value('paths/dlgdir', '.')
        dlg =QFileDialog(window, "select", lastDir)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            newDir = dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
            # update list of recent files
            filter(lambda a: a != filenames[0], window._recentFiles)
            window._recentFiles.append(filenames[0])
            if len(window._recentFiles) > 10:
                window._recentFiles.pop(0)
            window.settings.setValue('paths/recent', window._recentFiles)
            # update menu and actions
            updateMenuOpenRecent()
            openFile(filenames[0])
    elif name == 'actionSave':
        save()
    updateEnabledActions()

def openFile(f):

    # convert QString object to string
    #if isinstance(f, QString):
        #f=str(f.toUtf8())

    # extract embedded profile and metadata, if any.
    # metadata is a list of dicts with len(metadata) >=1.
    # metadata[0] contains at least 'SourceFile' : path.
    # profile is a string containing the profile binary data.
    # Currently, we do not use these data : standard profiles
    # are loaded from disk, non standard profiles are ignored.
    with exiftool.ExifTool() as e:
        profile, metadata = e.get_metadata(f)

    # trying to get color space info
    # 1 = sRGB
    colorSpace = metadata[0].get("EXIF:ColorSpace", -1)
    if colorSpace <0:
        # sRGBIEC61966-2.1
        desc_colorSpace = metadata[0].get("ICC_Profile:ProfileDescription", '')
        if isinstance(desc_colorSpace, unicode) or isinstance(desc_colorSpace, str):
            if 'sRGB' in desc_colorSpace:
                colorSpace = 1

    orientation = metadata[0].get("EXIF:Orientation", 0)
    transformation = exiftool.decodeExifOrientation(orientation)

    name = path.basename(f)
    img = imImage(filename=f, colorSpace=colorSpace, orientation=transformation, rawMetadata=metadata, profile=profile, name=name)
    if img.format() < 4:
        msg = QMessageBox()
        msg.setText("Cannot edit indexed formats\nConvert image to a non indexed mode first")
        msg.exec_()
        return
    if colorSpace < 0:
        print 'colorspace', colorSpace
        msg = QMessageBox()
        msg.setText("Color profile missing\nAssigning sRGB profile")
        msg.exec_()
        img.meta.colorSpace = 1
        img.updatePixmap()

    window.label.img = img
    window.label.img.onModify = lambda : updateEnabledActions()
    window.label_2.img = imImage(QImg=img.copy(), meta=img.meta)
    # no mouse drawing or painting
    window.label_2.img.isMouseSelectable = False
    window.tableView.addLayers(window.label.img)
    window.label.repaint()
    window.label_2.repaint()
    # used by graphicsForm3DLUT.onReset
    window.label.img.window = window.label
    window.label_2.img.window = window.label_2

def updateMenuOpenRecent():
    window.menuOpen_recent.clear()
    for f in window._recentFiles :
        window.menuOpen_recent.addAction(f, lambda x=f: window.execFileOpen(x))

def updateEnabledActions():
    window.actionSave.setEnabled(window.label.img.isModified)

def menuWindow(x, name):

    if name == 'actionShow_hide_left_window' :
        pass
    elif name == 'actionShow_hide_right_window_3' :
        if window.label_2.isHidden() :
            window.label_2.show()
        else:
            window.label_2.hide()
    elif name == 'actionDiaporama':
        handleNewWindow(imImg=window.label.img, title='Diaporama', show_maximized=True, parent=window)

def menuImage(x, name) :
    img = window.label.img
    # display image info
    if name == 'actionImage_info' :
        # Format
        s = "Format : %s\n(cf. QImage formats in the doc for more info)" % QImageFormats.get(img.format(), 'unknown')
        # dimensions
        s = s + "\n\ndim : %d x %d" % (img.width(), img.height())
        # working profile
        if img.colorTransformation is not None:
            workingProfileInfo = img.colorTransformation.fromProfile.info
        else:
            workingProfileInfo = 'None'
        s = s + "\n\nWorking Profile : %s" % workingProfileInfo
        # embedded profile
        if len(img.meta.profile) > 0:
            s = s +"\n\nEmbedded profile found, length %d" % len(img.meta.profile)
        # raw meta data
        l = img.meta.rawMetadata
        s = s + "\nMETADATA :\n"
        for d in l:
            s = s + '\n'.join('%s : %s' % (k,v) for k, v in d.iteritems())
        w, label = handleTextWindow(parent=window, title='Image info')
        label.setWordWrap(True)
        label.setText(s)
    elif name == 'actionColor_manage':
        icc.COLOR_MANAGE = window.actionColor_manage.isChecked()
        img.updatePixmap()
        window.label_2.img.updatePixmap()
    elif name == 'actionWorking_profile':
        w, label = handleTextWindow(parent=window, title='profile info')
        s = 'Working profile\n' + icc.workingProfile.info
        s = s + '-------------\n' + 'Monitor profile\n'+ icc.monitorProfile.info
        s = s + '\nNote : The working profile is the color profile currently associated\n with the opened image.'
        s = s + '\nThe monitor profile is associated with your monitor'
        s = s + '\nBoth profles are used in conjunction to display exact colors'
        s = s + '\n\nIf the monitor profile listed above is not the right profile\nfor your monitor, check the system settings for color management'
        label.setText(s)
    elif name == 'actionSnap':
        snap = img.snapshot()
        snap.setView(*window.label_2.img.view())
        window.label_2.img = snap
        window.label_2.repaint()


def menuLayer(x, name):
    if name == 'actionBrightness_Contrast':
        grWindow=graphicsForm.getNewWindow()
        #Wins['Brightness-Contrast'] = grWindow
        grWindow.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored);
        grWindow.setGeometry(QRect(100, 40, 156, 102))

        dock=QDockWidget()
        dock.setWidget(grWindow)
        window.addDockWidget(Qt.RightDockWidgetArea, dock)
        #l=QLayer(QImg=testLUT(grWindow.LUTXY))
        layerName='Brightness Contrast'
        l=window.label.img.addAdjustmentLayer(name=layerName)
        l.inputImg = window.label.img
        l.applyLUT(grWindow.graphicsScene.LUTXY, widget=window.label)
        #window.label.img.addLayer(l, name='Brightness/Contrast')
        window.tableView.addLayers(window.label.img)
        grWindow.graphicsScene.onUpdateLUT = lambda options={} : l.applyLUT(grWindow.graphicsScene.LUTXY, widget=window.label, options=options)
        window.label.repaint()
    elif name in ['action3D_LUT', 'action3D_LUT_HSB']:
        ccm = cmHSP if name == 'action3D_LUT' else cmHSB
        layerName = '3D LUT HSpB' if name == 'action3D_LUT' else '3D LUT HSB'
        l = window.label.img.addAdjustmentLayer(name=layerName)
        window.tableView.addLayers(window.label.img)
        grWindow = graphicsForm3DLUT.getNewWindow(ccm, size=800, targetImage=window.label.img, LUTSize=LUTSIZE, layer=l, parent=window)
        # add a dockable widget
        dock = QDockWidget(window)
        # link dock with adjustment layer
        l.adjustView = dock
        dock.setWidget(grWindow)
        dock.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowStaysOnTopHint)
        dock.setWindowTitle(grWindow.windowTitle())
        dock.move(900, 40)
        #window.addDockWidget(Qt.RightDockWidgetArea, dock)
        window.tableView.update()
        grWindow.graphicsScene.onUpdateLUT = lambda options={} : l.apply3DLUT(grWindow.graphicsScene.LUT3D, widget=window.label, options=options)
        window.label.repaint()
    elif name == 'actionNew_segmentation_layer':
        l=window.label.img.addSegmentationLayer(name='Segmentation')
        window.tableView.addLayers(window.label.img)
        # link to grabcut form
        l.segmentView = segmentForm.getNewWindow(targetImage=window.label.img)
        # TODO continue


def handleNewWindow(imImg=None, parent=None, title='New window', show_maximized=False, event_handler=True):
    """
    Show a floating window with a QLabel object. It can be used
    to display text or iamge. If the parameter event_handler is True (default)
    the QLabel object redefines its handlers for paint and mouse events to display
    the imImage object label.img
    :param parent:
    :param title:
    :param event_handler:
    """

    newwindow = QMainWindow(parent)
    newwindow.setAttribute(Qt.WA_DeleteOnClose)
    newwindow.setWindowTitle(parent.tr(title))
    label=QLabel()
    newwindow.setCentralWidget(label)
    # The attribute img is used by event handlers
    label.img = imImg
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    if event_handler:
        set_event_handler(label)
    if show_maximized:
        newwindow.showMaximized()
    else:
        newwindow.show()
    return newwindow, label

def handleTextWindow(parent=None, title=''):
    """
    Display a floating modal text window
    :param parent:
    :param title:
    """
    w, label = handleNewWindow(parent=parent, title=title, event_handler=False)
    w.setFixedSize(500,500)
    #label.setStyleSheet("QLabel { background-color: blue }")
    label.setAlignment(Qt.AlignTop)
    w.hide()
    w.setWindowModality(Qt.WindowModal)
    w.show()
    return w, label

def initMenuAssignProfile():
    window.menuAssign_profile.clear()
    for f in PROFILES_LIST:
        window.menuAssign_profile.addAction(f[0]+" "+f[1], lambda x=f : window.execAssignProfile(x))

def savingDialog(img):
    reply = QMessageBox()
    reply.setText("%s has been modified" % img.meta.name if len(img.meta.name) > 0 else 'unnamed image')
    reply.setInformativeText("Save your changes ?")
    reply.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
    reply.setDefaultButton(QMessageBox.Save)
    ret = reply.exec_()
    return ret

###########
# app init
##########
window.setStyleSheet("background-color: rgb(200, 200, 200);")

def save():
    lastDir = window.settings.value('paths/dlgdir', QDir.currentPath()).toString()
    dlg = QFileDialog(window, "select", lastDir)
    if dlg.exec_():
        filenames = dlg.selectedFiles()
        # Note : class vImage overrides QImage.save()
        img = window.label.img
        img.save(filenames[0], quality=100)
        with exiftool.ExifTool() as e:
            e.restoreMetadata(img.filename, str(filenames[0]))

def close(e):
    if window.label.img.isModified:
        ret = savingDialog(window.label.img)
        if ret == QMessageBox.Yes:
            save()
            return True
        else:
            return False
    return True

window.onCloseEvent = close

#get mouse hover events
window.label.setMouseTracking(True)
#window.label_2.setMouseTracking(True)

# set GUI Slot hooks
window.onWidgetChange = button_change
window.onShowContextMenu = contextMenu
window.onExecMenuFile = menuFile
window.onExecFileOpen = openFile
window.onExecMenuWindow = menuWindow
window.onExecMenuImage = menuImage
window.onExecMenuLayer = menuLayer
#window.onUpdateMenuAssignProfile = menuAssignProfile

# load current settings
window.readSettings()

window._recentFiles = window.settings.value('paths/recent', [])

set_event_handler(window.label)
set_event_handler(window.label_2)

img=QImage(200, 200, QImage.Format_ARGB32)
img.fill(Qt.darkGray)
defaultImImage = imImage(QImg=img, meta=metadata(name='noName'))

PROFILES_LIST = icc.getProfiles()
initMenuAssignProfile()
updateMenuOpenRecent()

window.label.img = defaultImImage
window.label_2.img = defaultImImage

window.showMaximized()

sys.exit(app.exec_())