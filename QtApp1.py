import sys
import cv2
from PyQt4.QtCore import Qt, QRect, QEvent, QSettings, QSize, QString
from PyQt4.QtGui import QWidget, QSplitter, QPixmap, QImage, QColor,QPainter, QApplication, QMenu, QAction, QCursor, QFileDialog, QColorDialog, QMainWindow, QLabel, QDockWidget, QHBoxLayout, QSizePolicy
from QtGui1 import app, window
import PyQt4.Qwt5 as Qwt
import time
import exiftool
from imgconvert import *
from MarkedImg import mImage, imImage, QLayer

from graphicsLUT import graphicsForm
from graphicsLUT3D import graphicsForm3DLUT
from LUT3D import rgb2hsB,hsp2rgb, LUT3D
from math import floor
from colorModels import hueSatModel
from icc import COLOR_MANAGE

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
    #mask= np.bitwise_and(mask_s , 12)
    #mask= np.right_shift(mask, 2)
    #mask[:200,:200]=1
    #mask_s=np.bitwise_and(mask_s , 3)
    #current_mask=mask_s
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

# painter for images
qp=QPainter()

def paintEvent(widg, e) :
    """
    paint event handler for a widget dispaying a mImage
    :param widg:
    :param e: paint event
    """

    qp.begin(widg)
    qp.translate(5, 5)
    qp.setClipRect(QRect(0,0, widg.width()-10, widg.height()-10))
    #qp.setCompositionMode(qp.CompositionMode_DestinationIn)  # avoid alpha summation

    mimg= widg.img
    r=mimg.resize_coeff(widg)
    qp.setPen(QColor(0,255,0))
    qp.fillRect(QRect(0, 0, widg.width() - 10, widg.height() - 10), QColor(255, 128, 0, 50));

    for layer in mimg._layersStack :
        if layer.visible:
            if layer.qPixmap is not None:
                qp.drawPixmap(QRect(mimg.xOffset,mimg.yOffset, mimg.width()*r-10, mimg.height()*r-10), # target rect
                           layer.transfer() #layer.qPixmap
                         )
            else:
                qp.drawImage(QRect(mimg.xOffset, mimg.yOffset, mimg.width() * r - 10, mimg.height() * r - 10), # target rect
                              layer  # layer.qPixmap
                              )
    if mimg.rect is not None :
        qp.drawRect(mimg.rect.left()*r + mimg.xOffset, mimg.rect.top()*r +mimg.yOffset,
                    mimg.rect.width()*r, mimg.rect.height()*r
                    )
    #if mimg.mask is not None :
        #qp.drawImage(QRect(mimg.xOffset, mimg.yOffset, mimg.width * r-10, mimg.height * r  -10), mimg.mask)
    """
    for layer in mimg.layers.values() :
        if layer.visible:
            qp.drawImage(QRect(mimg.xOffset, mimg.yOffset, mimg.qImg.width() * r - 10, mimg.qImg.height() * r - 10), layer.qImg)
    """
    qp.end()

def showResult(img0, img1, turn):
    global mask, mask_s
    print 'turn', turn
    img0=img0.resize(P_SIZE)
    #mask = np.zeros(img0.cv2Img.shape[:2], dtype=np.uint8)
    if turn ==0:
        current_mask = mask
    else:
        current_mask= mask_s

    mask2 = np.where((current_mask == cv2.GC_FGD) + (current_mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
    #img1.set_cv2Img_(img0.cv2Img)
    img1=imImage(cv2Img=img0.cv2Img)
    img1.cv2Img[:, :, 3] = img1.cv2Img[:, :, 3] * mask2
    #img1.set_cv2Img_(img1.cv2Img)
    img1 = imImage(cv2Img=img0.cv2Img)
    window.label_2.img=img1
    window.label_2.repaint()


# mouse eventd handler for image widgets (currently label and label_2)
turn = 0
pressed=False
clicked = True
def mouseEvent(widget, event) :

    global rect_or_mask, mask, mask_s, turn,Mimg_1, pressed, clicked
    img= widget.img

    r = img.resize_coeff(widget)
    x, y = event.x(), event.y()
    modifier = QApplication.keyboardModifiers()

    if modifier == Qt.ControlModifier:
        if event.type() == QEvent.MouseButtonPress:
            showResult(Mimg_p, Mimg_1, turn)
            turn = (turn + 1) % 2
        return

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

    elif event.type() == QEvent.MouseMove :
        clicked=False
        if pressed :
            if window.btnValues['rectangle'] :
                img.rect = QRect(min(State['ix'], x)/r -img.xOffset/r, min(State['iy'], y)/r - img.yOffset/r, abs(State['ix'] - x)/r, abs(State['iy'] - y)/r)
                rect_or_mask = 0
            elif (window.btnValues['drawFG'] or window.btnValues['drawBG']):
                color= CONST_FG_COLOR if window.btnValues['drawFG'] else CONST_BG_COLOR
                #qp.begin(img.mask)
                qp.begin(img._layers['drawlayer'])
                qp.setPen(color)
                qp.setBrush(color);
                qp.setCompositionMode(qp.CompositionMode_Source)  # avoid alpha summation
                qp.drawEllipse(int(x / r)-img.xOffset/r, int(y / r)- img.yOffset/r, 80, 80)
                qp.end()
                rect_or_mask=1

                #mask=cv2.bitwise_or(img.resize(40000).mask, mask)

                #window.label_2.img=do_grabcut(Mimg_p, preview=P_SIZE)
                #window.label_2.repaint()
                window.label.repaint()
            else:
                img.xOffset+=(x-State['ix'])
                img.yOffset+=(y-State['iy'])
        #update
        State['ix'],State['iy']=x,y
        #c = QColor(img.pixel(State['ix'] / r -  img.xOffset/r,  State['iy'] / r - img.yOffset/r))
        #r,g,b=c.red(), c.green(), c.blue()
        #h,s,p=rgb2hsv(r,g,b, perceptual=True)
        #print 'picker rgb :', r,g,b, 'hsp :', h,s,p
        #print hs2rgbList(h,s)

    elif event.type() == QEvent.MouseButtonRelease :
        pressed=False
        if event.button() == Qt.LeftButton:
            #click event
            if clicked:
                # Note : for multilayered images we read pixel color from  the background layer
                c = QColor(img.pixel(State['ix'] / r -  img.xOffset/r, State['iy'] / r - img.yOffset/r))
                r, g, b = c.red(), c.green(), c.blue()
                #hsModel.colorPickerSetmark(r,g,b, LUT3D)
                #h, s, p = rgb2hsv(r, g, b, perceptual=True)
                #i,j= hsModel.colorPickerGetPoint(h,s)
                # The selected node corresponds to the background layer : it does not take into account
                # the modifications induced by adjustment layers
                if hasattr(img.activeLayer.window.widget(), 'selectGridNode'):
                    #Wins['3D_LUT'].selectGridNode(r, g, b)
                    img.activeLayer.window.widget().selectGridNode(r, g, b)
                else:
                    print type(img.activeLayer.window.widget())
                #Wins['3D_LUT'].select(h,s,p)
                #window.label_2.img.apply3DLUT(LUT3D)
                window.label.repaint()
            if window.btnValues['rectangle']:
                #State['tool_rect'] = False
                #State['rect_over'] = True
                #cv2.rectangle(img, (State['ix'], State['iy']), (x, y), BLUE, 2)
                img.rect = QRect(min(State['ix'], x)/r-img.xOffset/r, min(State['iy'], y)/r- img.yOffset/r, abs(State['ix'] - x)/r, abs(State['iy'] - y)/r)
                rect_or_mask = 0 #init_with_rect
                #tmp=np.zeros((img.height, img.width), dtype=np.uint8)
                #tmp[img.rect.top():img.rect.bottom(), img.rect.left():img.rect.right()] = cv2.GC_PR_FGD
                #img.mask=ndarrayToQimage(tmp)
                #mask=tmp
                #mask_s=tmp
                #window.label_2.img=do_grabcut(Mimg_p, preview=P_SIZE, mode =cv2.GC_INIT_WITH_MASK)
                #window.label_2.repaint()
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

"""
app = QApplication(sys.argv)
window = QtGui1.Form1()
set_event_handler(window.label)
set_event_handler(window.label_2)

window.label.setStyleSheet("background-color: rgb(200, 200, 200);")
"""

#img_0=Mimg_0.cv2Img()
#Mimg.rect = QRect(500, 400, Mimg.qImg.width()-2000, Mimg.qImg.height()-1000)

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
    #get list of recent files and update menu
    window._recentFiles = window.settings.value('paths/recent', [], QString)
    window.updateMenuOpenRecent()
    print 'updated'
    if name == 'actionOpen' :
        lastDir = window.settings.value('paths/dlgdir', 'F:/bernard').toString()
        dlg =QFileDialog(window, "select", lastDir)

        if dlg.exec_():
            filenames = dlg.selectedFiles()
            newDir = dlg.directory().absolutePath()
            window.settings.setValue('paths/dlgdir', newDir)
            filter(lambda a: a != filenames[0], window._recentFiles)
            window._recentFiles.append(filenames[0])
            if len(window._recentFiles) > 5:
                window._recentFiles.pop(0)
            window.settings.setValue('paths/recent', window._recentFiles)
            window.updateMenuOpenRecent()
            openFile(filenames[0])
    elif name == 'actionSave':
        lastDir = window.settings.value('paths/dlgdir', 'F:/bernard').toString()
        dlg = QFileDialog(window, "select", lastDir)

        if dlg.exec_():
            filenames = dlg.selectedFiles()
            window.label.img.save(filenames[0])

# hsModel= hueSatModel.colorWheel(500, 500)

def openFile(f):

    # convert QString object to string
    if isinstance(f, QString):
        f=str(f.toUtf8())
    #get exif data
    with exiftool.ExifTool() as e:
        profile, metadata = e.get_metadata(f)
    metadata=metadata[0]
    #for k, v in metadata.iteritems():
        #print k, v
    colorSpace = metadata.get("EXIF:ColorSpace", -1) # sRGB: 1
    orientation = metadata.get("EXIF:Orientation", 0)
    b=exiftool.decodeExifOrientation(orientation)

    window.label.img = imImage(filename=f, colorSpace=colorSpace, orientation=b, metadata=metadata, profile=profile).resize(250000)

    #window.label.img = hsModel
    #window.label.img=imImage(QImg=convert(f))
    #window.label_2.img = window.label.img
    window.label_2.img = imImage(filename=f, orientation=b, metadata=metadata)
    #print 'Orientation', metadata['EXIF:Orientation']
    #print 'ICC', metadata['EXIF:ColorSpace']
    window.tableView.addLayers(window.label.img)
    window.label.repaint()
    window.label_2.repaint()


def menuWindow(x, name):

    if name == 'actionShow_hide_left_window' :
        if window.label.isHidden() :
            window.label.show()
        else:
            window.label.hide()
    elif name == 'actionShow_hide_right_window' :
        if window.label_2.isHidden() :
            window.label_2.show()
        else:
            window.label_2.hide()
    elif name == 'actionDiaporama':
        handleNewWindow(window)

def menuImage(x, name) :

    if name == 'actionImage_info' :
        print window.label.img.metadata
    elif name == 'actionColor_manage':
        COLOR_MANAGE = window.actionColor_manage.isChecked()
        window.label.img.updatePixmaps()
        window.label_2.img.updatePixmaps()



def menuLayer(x, name):

    if name == 'actionBrightness_Contrast' :
        grWindow=graphicsForm.getNewWindow()
        Wins['Brightness-Contrast'] = grWindow
        grWindow.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored);
        grWindow.setGeometry(QRect(100, 40, 156, 102));

        dock=QDockWidget()
        dock.setWidget(grWindow)
        window.addDockWidget(Qt.RightDockWidgetArea, dock);
        #window.horizontalLayout_2.addWidget(grWindow)
        #window.verticalLayout_2.addWidget(grWindow)
        #l=QLayer(QImg=testLUT(grWindow.LUTXY))
        l=QLayer(QImg=window.label.img)
        l.inputImg = window.label.img
        l.applyLUT(grWindow.graphicsScene.LUTXY, widget=window.label)

        window.label.img.addLayer(l, 'Brightness/Contrast')
        window.tableView.addLayers(window.label.img)
        grWindow.graphicsScene.onUpdateScene = lambda : l.applyLUT(grWindow.graphicsScene.LUTXY, widget=window.label)
        window.label.repaint()
    elif name == 'action3D_LUT':
        l = window.label.img.addAdjustmentLayer(name='3D LUT')
        window.tableView.addLayers(window.label.img)
        grWindow = graphicsForm3DLUT.getNewWindow(size=800, title= l.name, parent=window)
        Wins[l.name] = grWindow
        dock = QDockWidget(window)
        l.window = dock
        dock.setWidget(grWindow)
        dock.setWindowFlags(Qt.WindowStaysOnTopHint)
        #dock.setAttribute(Qt.WA_DeleteOnClose)
        dock.setWindowTitle(grWindow.windowTitle())
        dock.move(500, 40)
        #window.addDockWidget(Qt.RightDockWidgetArea, dock);

        grWindow.graphicsScene.onUpdateScene = lambda: l.apply3DLUT(grWindow.graphicsScene.LUT3D, widget=window.label)
        window.label.repaint()
"""
def testLUT(LUT) :
    img=window.label.img
    a=  QImageBuffer(img)
    b=np.array(LUT)
    a=b[a]
    r,g,b=a[:,:,0], a[:,:,1], a[:,:,2]
    #a=cv2.LUT(a, np.array(LUT))
    #a=cv2.convertScaleAbs(a)


    a=a[:,:,::-1]
    a = np.ascontiguousarray(a[:, :, 1:4], dtype=np.uint8)
    #a=np.dstack((r, g, b))
    print 'shape', a.shape

    return QPixmap.fromImage(mImage(cv2Img=a, format=QImage.Format_RGB888))
"""


def handleNewWindow(parent):
    newwindow = QMainWindow(parent)
    newwindow.setAttribute(Qt.WA_DeleteOnClose)
    newwindow.setWindowTitle(parent.tr('New Window'))
    label_3=QLabel()
    newwindow.setCentralWidget(label_3)
    label_3.img = window.label.img
    label_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding);
    set_event_handler(label_3)
    newwindow.show()


set_event_handler(window.label)
set_event_handler(window.label_2)

window.label.setStyleSheet("background-color: rgb(200, 200, 200);")
#get mouse hover events
window.label.setMouseTracking(True)
window.label_2.setMouseTracking(True)

# set button and slider change handler
window.onWidgetChange = button_change
window.onShowContextMenu = contextMenu
window.onExecMenuFile = menuFile
window.onExecFileOpen = openFile
window.onExecMenuWindow = menuWindow
window.onExecMenuImage = menuImage
window.onExecMenuLayer = menuLayer




window.readSettings()

window._recentFiles = window.settings.value('paths/recent', [], QString)

openFile('orch2-2-2.jpg')

window.show()

sys.exit(app.exec_())