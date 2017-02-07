"""
Copyright (C) 2017  Bernard Virot

PeLUT - Photo editing software using adjustment layers with 1D and 3D Look Up Tables.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""


import sys
import cv2
from PyQt4.QtCore import Qt, QRect, QEvent, QDir, QSettings, QSize, QString
from PyQt4.QtGui import QWidget, QSplitter, QPixmap, QImage, QAbstractItemView, QColor,QPainter, QApplication, QMenu, QAction, QCursor, QFileDialog, QMessageBox, QColorDialog, QMainWindow, QLabel, QDockWidget, QHBoxLayout, QSizePolicy
from QtGui1 import app, window
import PyQt4.Qwt5 as Qwt
import time
import exiftool
from imgconvert import *
from MarkedImg import mImage, imImage, QLayer

from graphicsLUT import graphicsForm
from graphicsLUT3D import graphicsForm3DLUT
from LUT3D import LUTSIZE
from math import floor
from colorModels import hueSatModel
import icc
from os import path

P_SIZE=4000000

CONST_FG_COLOR = QColor(255, 255, 255,128)
CONST_BG_COLOR = QColor(255, 0, 255,128)

thickness = 30*4
State = {'drag' : False, 'drawing' : False , 'tool_rect' : False, 'rect_over' : False, 'ix' : 0, 'iy' :0, 'rawMask' : None}
# application windows
Wins = {'3D_LUT': None, 'Brightness-Contrast':None}
rect_or_mask = 0

def QRect2tuple(qrect):
    return (qrect.left(), qrect.top(), qrect.right()-qrect.left(), qrect.bottom()-qrect.top())

def waitBtnEvent():
    global btn_pressed
    btn_pressed=False
    while not btn_pressed:
        time.sleep(0.1)
        print 'waiting'
    btn_pressed = False


mask=None
mask_s=None

def do_grabcut(img0, preview=-1, nb_iter=1, mode=cv2.GC_INIT_WITH_RECT, again=False):
    """
    segment source MImage instance.

    :param img0: source Mimage, unmodified.
    :param preview:
    :param nb_iter:
    :param mode
    :return:
    """
    #img0.rect = QRect(500, 400, Mimg.width() - 2000, Mimg.height() - 1000)

    print '********* do_grabCut call'
    mask_s = State['rawMask']
    global rect_or_mask

    #if preview>0:
        #img0_r=img0.resize(preview)
    #else:
    img0_r=img0

    # set rect mask
    rectMask = np.zeros((img0_r.height(), img0_r.width()), dtype=np.uint8)
    rectMask[img0_r.rect.top():img0_r.rect.bottom(), img0_r.rect.left():img0_r.rect.right()] = cv2.GC_PR_FGD

    if not again:
        #get painted values in BGRA order
        paintedMask = QImageBuffer(img0_r._layers['drawlayer'])

        paintedMask[paintedMask==255]=cv2.GC_FGD
        paintedMask[paintedMask==0]=cv2.GC_BGD

        np.copyto(rectMask, paintedMask[:,:,1], where=(paintedMask[:,:,3]>0)) # copy  painted (A > 0) pixels (G value only)

        if mask_s is not None:
            np.copyto(rectMask, mask_s, where=(np.logical_and((mask_s==0),(paintedMask[:,:,0]==0))))

        mask_s=rectMask
        rect_or_mask=0
    else:
        if mask_s is None:
            mask_s=rectMask
            print "None mask"
        else:
            print "reuse mask"

    bgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the background model
    fgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the foreground model

    t0 = time.time()
    if preview >0:
        img0_r=img0_r.resize(preview)
        mask_s=cv2.resize(mask_s, (img0_r.width(), img0_r.height()), interpolation=cv2.INTER_NEAREST)
        #a=img0_r.cv2Img()
    #cv2.grabCut_mtd(img0_r.cv2Img()[:,:,:3],
    cv2.grabCut_mtd(img0_r.cv2Img()[:, :, :3],
                mask_s,
                None,#QRect2tuple(img0_r.rect),
                bgdmodel, fgdmodel,
                nb_iter,
                mode)
    print 'grabcut_mtd time :', time.time()-t0

    img0_r = img0
    if preview >0:
        mask_s=cv2.resize(mask_s, (img0.width(), img0.height()), interpolation=cv2.INTER_NEAREST)

    State['rawMask'] = mask_s
    # apply mask
    current_mask = mask_s

    mask_s = np.where((current_mask == cv2.GC_FGD) + (current_mask == cv2.GC_PR_FGD), 1, 0)
    mask_s1 = np.where((current_mask == cv2.GC_FGD) + (current_mask == cv2.GC_PR_FGD), 1, 0.4)

    tmp = np.copy(img0_r.cv2Img())

    tmp[:, :, 3] = tmp[:, :, 3] * mask_s1 # cast float to uint8

    img1= imImage(cv2Img=tmp, cv2mask=current_mask)
    #display
    #window.label_2.repaint()

    b=np.zeros((img0_r.height(), img0_r.width()), dtype=np.uint8)
    c=np.zeros((img0_r.height(), img0_r.width()), dtype=np.uint8)
    b[:,:]=255
    alpha = ((1 - mask_s) * 255).astype('uint8')
    #cv2mask = cv2.resize(np.dstack((b, c, c, alpha)), (img0.qImg.width(), img0.qImg.height()), interpolation=cv2.INTER_NEAREST)
    cv2mask = np.dstack((c, c, b, alpha))
    img0._layers['masklayer']=QLayer(QImg=ndarrayToQImage(cv2mask))
    #img0.drawLayer=mImage(QImg=ndarrayToQImage(cv2mask))
    #img1=imImage(cv2Img=cv2.inpaint(img1.cv2Img[:,:,:3], mask_s, 20, cv2.INPAINT_NS), format=QImage.Format_RGB888)
    return img1

def canny(img0, img1) :
   low = window.slidersValues['low']
   high = window.slidersValues['high']

   l= [k[-1] for k in window.btnValues if window.btnValues[k] == 1]
   if l :
       aperture = int(l[0])
   else :
       aperture =3

   print 'Canny edges', 'low=%d, high=%d aperture=%d' % (low, high, aperture)

   edges = cv2.Canny(img0.cv2Img, low, high, L2gradient=True, apertureSize=aperture) #values 3,5,7

   contours= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   edges[:]=0
   print len(contours)
   cv2.drawContours(edges, contours[0], -1, 255, 2)
   img1.__set_cv2Img(edges)
   window.label_2.repaint()

# paintEvent painter
qp=QPainter()
bgColor=QColor(100,100,100)

def paintEvent(widg, e) :
    """
    Paint event handler for widgets that display a mImage object.
    The widget must have a valid img attribute of type QImage.
    :param widg: widget object with a img attribute
    :param e: paint event
    """
    if not hasattr(widg, 'img'):
        raise ValueError("paintEvent : no image")
    if widg.img is None:
        raise ValueError("paintEvent : no image")
    qp.begin(widg)
    #qp.translate(5, 5)
    #qp.setClipRect(QRect(0,0, widg.width()-10, widg.height()-10))
    #qp.setCompositionMode(qp.CompositionMode_DestinationIn)  # avoid alpha summation

    mimg= widg.img
    r=mimg.resize_coeff(widg)
    # pen for selection rectangle
    qp.setPen(QColor(0,255,0))
    # background
    qp.fillRect(QRect(0, 0, widg.width() , widg.height() ), bgColor)
    # draw layers
    for layer in mimg._layersStack :
        if layer.visible:
            if layer.qPixmap is not None:
                qp.drawPixmap(QRect(mimg.xOffset,mimg.yOffset, mimg.width()*r, mimg.height()*r), # target rect
                           layer.transfer() #layer.qPixmap
                         )
            else:
                qp.drawImage(QRect(mimg.xOffset, mimg.yOffset, mimg.width() * r , mimg.height() * r ), # target rect
                              layer  # layer.qPixmap
                              )
    # draw selection rectangle
    if (mimg.activeLayer.visible) and (mimg.activeLayer.rect is not None ):
        rct = mimg.activeLayer.rect
        qp.drawRect(rct.left()*r + mimg.xOffset, rct.top()*r +mimg.yOffset, rct.width()*r, rct.height()*r)
    #if mimg.mask is not None :
        #qp.drawImage(QRect(mimg.xOffset, mimg.yOffset, mimg.width * r-10, mimg.height * r  -10), mimg.mask)
    qp.end()

# mouse eventd handler for image widgets (currently label and label_2)
turn = 0
pressed=False
clicked = True
def mouseEvent(widget, event) :

    global rect_or_mask, mask, mask_s, turn,Mimg_1, pressed, clicked
    img= widget.img

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
        State['ix'], State['iy'] = x, y
        State['ix_begin'], State['iy_begin'] = x, y
    elif event.type() == QEvent.MouseMove :
        clicked=False
        if pressed :
            # button pressed
            if img.isMouseSelectable:
                if window.btnValues['rectangle']:
                    img.activeLayer.rect = QRect(min(State['ix_begin'], x)/r -img.xOffset/r, min(State['iy_begin'], y)/r - img.yOffset/r, abs(State['ix_begin'] - x)/r, abs(State['iy_begin'] - y)/r)
                    rect_or_mask = 0
                elif (window.btnValues['drawFG'] or window.btnValues['drawBG']):
                    color= CONST_FG_COLOR if window.btnValues['drawFG'] else CONST_BG_COLOR
                    #qp.begin(img._layers['drawlayer'])
                    qp.begin(img.activeLayer)
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
        #update
        State['ix'],State['iy']=x,y
    elif event.type() == QEvent.MouseButtonRelease :
        pressed=False
        if event.button() == Qt.LeftButton:
            # click event
            if img.isMouseSelectable and clicked:
                # select grid node
                # Note : for multilayered images we read pixel color from  the background layer
                c = QColor(img.pixel(State['ix'] / r -  img.xOffset/r, State['iy'] / r - img.yOffset/r))
                r, g, b = c.red(), c.green(), c.blue()
                if hasattr(img.activeLayer, "adjustView") and img.activeLayer.adjustView is not None:
                    if hasattr(img.activeLayer.adjustView.widget(), 'selectGridNode'):
                        mode = 'add' if modifier == Qt.ControlModifier else ''
                        img.activeLayer.adjustView.widget().selectGridNode(r, g, b, mode=mode)
                window.label.repaint()
            if window.btnValues['rectangle'] and img.isMouseSelectable:

                img.activeLayer.rect = QRect(min(State['ix_begin'], x)/r-img.xOffset/r, min(State['iy_begin'], y)/r- img.yOffset/r, abs(State['ix_begin'] - x)/r, abs(State['iy_begin'] - y)/r)
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
    numDegrees = event.delta() / 8
    numSteps = numDegrees / 150.0
    img.Zoom_coeff += numSteps
    widget.repaint()


def set_event_handler(widg):
    """
    Pythonic way for redefining event handlers. In contrast to
    subclassing and overriding, we can add convenient parameters to
    our handlers.
    :param widg:
    """
    widg.paintEvent = lambda e, wdg=widg : paintEvent(wdg,e)
    widg.mousePressEvent = lambda e, wdg=widg : mouseEvent(wdg, e)
    widg.mouseMoveEvent = lambda e, wdg=widg : mouseEvent(wdg, e)
    widg.mouseReleaseEvent = lambda e, wdg=widg : mouseEvent(wdg, e)
    widg.wheelEvent = lambda e, wdg=widg : wheelEvent(wdg, wdg.img, e)


def button_change(widg):
    if str(widg.accessibleName()) == "Apply" :
        print "grabcut"
        #do_grabcut(Mimg_p, mode=cv2.GC_INIT_WITH_MASK, again=(rect_or_mask==0))
        window.label_2.img=do_grabcut(window.label.img, mode=cv2.GC_INIT_WITH_MASK, again=(rect_or_mask==0))
    elif str(widg.accessibleName()) == "Preview" :
        print "grabcut preview"
        window.label_2.img = do_grabcut(window.label.img, preview=P_SIZE, mode=cv2.GC_INIT_WITH_MASK, again=(rect_or_mask==0))
    print "done"
    window.label_2.repaint()

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
    window._recentFiles = window.settings.value('paths/recent', [], QString)

    # update menu and actions
    updateMenuOpenRecent()

    def save():
        lastDir = window.settings.value('paths/dlgdir', QDir.currentPath()).toString()
        dlg = QFileDialog(window, "select", lastDir)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            window.label.img.save(filenames[0])

    if name == 'actionOpen' :
        if window.label.img.isModified:
            quit_msg = "%s is modified. Save?" % window.label.img.meta.name
            reply = QMessageBox.question(window, 'Message', quit_msg, QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                save()
            elif reply == QMessageBox.Cancel:
                return

        lastDir = window.settings.value('paths/dlgdir', 'F:/bernard').toString()
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
    if isinstance(f, QString):
        f=str(f.toUtf8())

    # extract embedded profile and metadata, if any
    # metadata is a list of dicts with len(metadata) >=1.
    # metadata[0] contains at least 'SourceFile' : path
    with exiftool.ExifTool() as e:
        profile, metadata = e.get_metadata(f)

    # trying to get color space info  1 = sRGB
    colorSpace = metadata[0].get("EXIF:ColorSpace", -1)
    if colorSpace <0:
        colorSpace = metadata[0].get("ICC_Profile:ProfileDescription", -1)

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
        img.updatePixmaps()

    window.label.img = img
    window.label.img.onModify = lambda : updateEnabledActions()
    window.label_2.img = imImage(QImg=img.copy(), meta=img.meta)
    # no mouse drawing or painting
    window.label_2.img.isMouseSelectable = False
    window.tableView.addLayers(window.label.img)
    window.label.repaint()
    window.label_2.repaint()

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
        handleNewWindow(window)

def menuImage(x, name) :

    img = window.label.img
    if name == 'actionImage_info' :
        # Format
        s = "Format : %s\n(cf. QImage formats in the doc for more info)" % QImageFormats.get(img.format(), 'unknown')
        # dimensions
        s = s + "\n\ndim : %d x %d" % (img.width(), img.height())
        # raw meta data
        l = img.meta.rawMetadata
        s = s + "\n\nMETADATA :\n"
        for d in l:
            s = s + '\n'.join('%s : %s' % (k,v) for k, v in d.iteritems())
        w, label = handleTextWindow(parent=window, title='Image info')
        label.setText(QString(s))
    elif name == 'actionColor_manage':
        icc.COLOR_MANAGE = window.actionColor_manage.isChecked()
        img.updatePixmaps()
        window.label_2.img.updatePixmaps()
    elif name == 'actionWorking_profile':
        w, label = handleTextWindow(parent=window, title='Working profile info')
        label.setText(QString(icc.MONITOR_PROFILE_INFO))
    elif name == 'actionSnap':
        img= img.snapshot()
        img.setView(*window.label_2.img.view())
        window.label_2.img = img
        window.label_2.repaint()
        print 'snap done'


def menuLayer(x, name):

    if name == 'actionBrightness_Contrast' :
        grWindow=graphicsForm.getNewWindow()
        Wins['Brightness-Contrast'] = grWindow
        grWindow.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored);
        grWindow.setGeometry(QRect(100, 40, 156, 102))

        dock=QDockWidget()
        dock.setWidget(grWindow)
        window.addDockWidget(Qt.RightDockWidgetArea, dock)
        #l=QLayer(QImg=testLUT(grWindow.LUTXY))
        l=QLayer(QImg=window.label.img)
        l.inputImg = window.label.img
        l.applyLUT(grWindow.graphicsScene.LUTXY, widget=window.label)

        window.label.img.addLayer(l, 'Brightness/Contrast')
        window.tableView.addLayers(window.label.img)
        grWindow.graphicsScene.onUpdateLUT = lambda options={} : l.applyLUT(grWindow.graphicsScene.LUTXY, widget=window.label, options=options)
        window.label.repaint()
    elif name == 'action3D_LUT':
        l = window.label.img.addAdjustmentLayer(name='3D LUT')
        window.tableView.addLayers(window.label.img)
        grWindow = graphicsForm3DLUT.getNewWindow(size=800, LUTSize=LUTSIZE, title= l.name, parent=window)
        Wins[l.name] = grWindow
        dock = QDockWidget(window)
        # link to colorwheel
        l.adjustView = dock
        dock.setWidget(grWindow)
        dock.setWindowFlags(Qt.WindowStaysOnTopHint)
        dock.setWindowTitle(grWindow.windowTitle())
        dock.move(500, 40)
        #window.addDockWidget(Qt.RightDockWidgetArea, dock)
        grWindow.graphicsScene.onUpdateLUT = lambda options={} : l.apply3DLUT(grWindow.graphicsScene.LUT3D, widget=window.label, options=options)
        window.label.repaint()

def handleNewWindow(parent=None, title='New window', set_event_handler=True):

    newwindow = QMainWindow(parent)
    newwindow.setAttribute(Qt.WA_DeleteOnClose)
    newwindow.setWindowTitle(parent.tr(title))
    label_3=QLabel()
    newwindow.setCentralWidget(label_3)
    label_3.img = window.label.img
    label_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding);
    if set_event_handler:
        set_event_handler(label_3)
    newwindow.show()
    return newwindow, label_3

def handleTextWindow(parent=None, title=''):

    w, label = handleNewWindow(parent=parent, title=title, set_event_handler=False)
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



###########
# app init
##########
window.setStyleSheet("background-color: rgb(200, 200, 200);")

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

window.readSettings()

window._recentFiles = window.settings.value('paths/recent', [], QString)

set_event_handler(window.label)
set_event_handler(window.label_2)

img=QImage(200, 200, QImage.Format_ARGB32)
img.fill(Qt.darkGray)
defaultImImage = imImage(QImg=img)

PROFILES_LIST = icc.getProfiles()
initMenuAssignProfile()
updateMenuOpenRecent()

window.label.img = defaultImImage
window.label_2.img = defaultImImage

window.show()

sys.exit(app.exec_())