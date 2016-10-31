import sys
import cv2
from PyQt4.QtCore import Qt, QString, QRect
import QtGui1
import PyQt4.Qwt5 as Qwt
import time

from imgconvert import *
from MarkedImg import MImage
#P_SIZE=6400000
P_SIZE=24000000
BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG
DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

BLUE =(255,0,0)
thickness = 30*4
State = {'drag' : False, 'drawing' : False , 'tool_rect' : False, 'rect_over' : False, 'ix' : 0, 'iy' :0}
value = DRAW_FG
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

def resize_coeff(mimg, widget) :
    """
    computes the resizing coefficient
    to be applied to mimg to display a non distorted view
    in widget.

    :param mimg: MarkedImg
    :param widget: Qwidget object
    :return: the (multiplicative) resizing coefficient
    """
    r_w, r_h = float(widget.width()) / mimg.cv2Img.shape[1], float(widget.height()) / mimg.cv2Img.shape[0]
    r = max(r_w, r_h)
    return r*mimg.Zoom_coeff

"""
def cv2img2qpixmap ( cv2image) :
    if len(cv2image.shape) >= 3 :
        qtImage = ndarrayToQimage(cv2image)
    else:
        qtImage=gray2qimage(cv2image)
    qtPixmap = QtGui.QPixmap.fromImage(qtImage)
    return qtPixmap
"""
mask=None
mask_s=None

def do_grabcut(img0, preview=-1, nb_iter=1, mode=cv2.GC_INIT_WITH_RECT, test=False):
    """
    segment source MImage instance.

    :param img0: source Mimage, unmodified.
    :param img1: MImage instance, will hold the segmented image.
    :param preview:
    :param nb_iter:
    :return:
    """
    #img0=MImage('orch2-2-2.jpg')
    img0.rect = QRect(500, 400, Mimg.width - 2000, Mimg.height - 1000)

    print '********* do_grabCut call'
    print "input shape ", img0.cv2Img.shape
    global mask, mask_s

    if preview>0:
        img0_r=img0.resize(preview)
        img0_r.ROI = QtCore.QRect(0, 0, img0_r.width, img0_r.height)

    #nb_iter = nb_iter
    #if rect_or_mask == 0:
    #   mask = np.zeros(img0.cv2Img.shape[:2], dtype=np.uint8)
    #else:
    #    mask=img0.mask[:,:,1].astype(np.uint8)

    #if rect_or_mask==0: # init_with_rect
    bgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the background model
    fgdmodel = np.zeros((1, 13 * 5), np.float64)  # Temporary array for the foreground model

    mask = np.zeros(img0_r.cv2Img.shape[:2], dtype=np.uint8)
    #mask_s = np.zeros(img0_r.cv2Img.shape[:2], dtype=np.uint8)
    #data=img0.mask.bits().asarray(img0.height*img0.width*4)
    #mask_s=np.ascontiguousarray(np.array(data, dtype=np.uint8).reshape(img0.height, img0.width,4)[:,:,3])
    #mask = np.ascontiguousarray(np.array(data, dtype=np.uint8).reshape(img0.height, img0.width,4)[:,:,3])

    tmp = np.zeros((img0_r.height, img0_r.width), dtype=np.uint8)
    if img0_r.rect is None:
        img0_r.rect = QRect(1000, 800, img0_r.width-1000, img0_r.height-800)
        print "none rect modify"
    tmp[img0_r.rect.top():img0_r.rect.bottom(), img0_r.rect.left():img0_r.rect.right()] = cv2.GC_PR_FGD
    tmp1=Qimage2array(img0_r.mask)
    #print Qimage2array(img0_r.mask)[:10,:10,]
    #exit()

    tmp1[tmp1==255]=cv2.GC_FGD
    tmp1[tmp1==0]=cv2.GC_BGD
    np.copyto(tmp, tmp1[:,:,1], where=(tmp1[:,:,0]>0)) # copy to tmp paint pixels only

    if not (mask_s is None):
        np.copyto(tmp, mask_s, where=(np.logical_and((mask_s==0),(tmp1[:,:,0]==0))))

    #mask = np.ascontiguousarray(Qimage2array(img0.mask)[:,:,3])
    mask_s=tmp


    print mask.shape, img0_r.cv2Img.shape

    t0 = time.time()
    cv2.grabCut_mtd(np.copy(img0_r.cv2Img[:,:,:3]),
                mask_s,
                None,#QRect2tuple(img0_r.rect),
                bgdmodel, fgdmodel,
                nb_iter,
                mode)
    print 'grabcut_mtd time :', time.time()-t0


    """
    else: # init with mask
        t0 = time.time()
        cv2.grabCut_slim(img0.cv2Img[:, :, :3], mask_s, QRect2tuple(img0.rect), bgdmodel, fgdmodel, nb_iter, cv2.GC_INIT_WITH_MASK)
        print 'grabcut_slim time :', time.time() - t0
        t0 = time.time()
        cv2.grabCut(img0.cv2Img[:, :, :3], mask, QRect2tuple(img0.rect), bgdmodel, fgdmodel, nb_iter,
                     cv2.GC_INIT_WITH_MASK)
        print 'grabcut time :', time.time() - t0
    """
    # apply mask
    current_mask = mask_s
    #mask= np.bitwise_and(mask_s , 12)
    #mask= np.right_shift(mask, 2)
    #mask[:200,:200]=1
    #mask_s=np.bitwise_and(mask_s , 3)
    #current_mask=mask_s
    mask_s = np.where((current_mask == cv2.GC_FGD) + (current_mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

    tmp = np.copy(img0_r.cv2Img)

    #tmp[:, :, 3]=tmp[:,:,3]*mask2
    tmp[:, :, 3] = tmp[:, :, 3] * mask_s

    img1= MImage(cv2Img=tmp, cv2mask=mask_s)
    #display
    #window.label_2.repaint()
    a = (mask_s * 255).astype('uint8')
    b=np.zeros((img0_r.height, img0_r.width), dtype=np.uint8)
    c=np.zeros((img0_r.height, img0_r.width), dtype=np.uint8)
    b[:,:]=128
    cv2mask = cv2.resize(np.dstack((a, c, c, b) ), (img0.width, img0.height), interpolation=cv2.INTER_NEAREST)
    img0.layers=[]
    img0.layers.append(ndarrayToQimage(cv2mask))
    #img0.mask=ndarrayToQimage(cv2mask)
    # change
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

qp=QtGui.QPainter()


def paintEvent(widg, e) :
    qp.begin(widg)
    qp.translate(5, 5)
    qp.setClipRect(QRect(0,0, widg.width()-10, widg.height()-10))
    #qp.setCompositionMode(qp.CompositionMode_DestinationIn)  # avoid alpha summation

    mimg= widg.img
    r=resize_coeff(mimg, widg)
    qp.setPen(QtGui.QColor(0,255,0))
    qp.fillRect(QRect(0, 0, widg.width() - 10, widg.height() - 10), QtGui1.QtGui.QColor(255, 128, 0, 50));
    qp.drawImage(QRect(mimg.xOffset,mimg.yOffset, mimg.width*r-10, mimg.height*r-10), # target rect
                  mimg.qImg
                 )
    if mimg.rect is not None :
        qp.drawRect(mimg.rect.left()*r + mimg.xOffset, mimg.rect.top()*r +mimg.yOffset,
                    mimg.rect.width()*r, mimg.rect.height()*r
                    )
    #if mimg.mask is not None :
        #qp.drawImage(QRect(mimg.xOffset, mimg.yOffset, mimg.width * r-10, mimg.height * r  -10), mimg.mask)

    for lay in mimg.layers :
        qp.drawImage(QRect(mimg.xOffset, mimg.yOffset, mimg.width * r - 10, mimg.height * r - 10), lay)

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
    img1=MImage(cv2Img=img0.cv2Img)
    img1.cv2Img[:, :, 3] = img1.cv2Img[:, :, 3] * mask2
    #img1.set_cv2Img_(img1.cv2Img)
    img1 = MImage(cv2Img=img0.cv2Img)
    window.label_2.img=img1
    window.label_2.repaint()



turn = 0
def mouseEvent(widget, event) :

    global rect_or_mask, mask, mask_s, turn,Mimg_1

    img= widget.img

    r = resize_coeff(img, widget)
    x,y = event.x(), event.y()
    modifier = QtGui.QApplication.keyboardModifiers()

    if modifier == QtCore.Qt.ControlModifier:
        if event.type() == QtCore.QEvent.MouseButtonPress:
            showResult(Mimg_p, Mimg_1, turn)
            turn = (turn + 1) % 2
        return

    if event.type() == QtCore.QEvent.MouseButtonPress :
        if event.button() == QtCore.Qt.LeftButton:
            pass #State['tool_rect'] = True
        elif event.button() == QtCore.Qt.RightButton:
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
        State['ix'], State['iy'] = x, y

    elif event.type() == QtCore.QEvent.MouseMove :
        if window.btnValues['rectangle'] :
            img.rect = QRect(min(State['ix'], x)/r -img.xOffset/r, min(State['iy'], y)/r - img.yOffset/r, abs(State['ix'] - x)/r, abs(State['iy'] - y)/r)
            rect_or_mask = 0
        elif (window.btnValues['drawFG'] or window.btnValues['drawBG']):
            color= QtGui.QColor(255, 255, 255,128) if window.btnValues['drawFG'] else QtGui.QColor(255, 0, 255,128)
            qp.begin(img.mask)
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
            State['ix'],State['iy']=x,y
            print x,y,img.xOffset, img.yOffset
    elif event.type() == QtCore.QEvent.MouseButtonRelease :
        if event.button() == QtCore.Qt.LeftButton:
            if window.btnValues['rectangle']:
                #State['tool_rect'] = False
                #State['rect_over'] = True
                #cv2.rectangle(img, (State['ix'], State['iy']), (x, y), BLUE, 2)
                img.rect = QtCore.QRect(min(State['ix'], x)/r-img.xOffset/r, min(State['iy'], y)/r- img.yOffset/r, abs(State['ix'] - x)/r, abs(State['iy'] - y)/r)
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
        elif event.button() == QtCore.Qt.RightButton:
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

    widget.repaint()

def wheelEvent(widget,img, event):
    numDegrees = event.delta() / 8
    numSteps = numDegrees / 150.0
    img.Zoom_coeff += numSteps
    widget.repaint()

app = QtGui.QApplication(sys.argv)
window = QtGui1.Form1()

window.show()

Mimg = MImage('orch2-2-2.jpg')
#Mimg = MImage('F:\Bernard\epreuves\loiret.jpg')

w, h = Mimg.width, Mimg.height
w1, h1 = window.label.width(), window.label.height()
rs_coeff = min(float(w1)/float(w), float(h1)/float(h) )
rs_w, rs_h = int(w * rs_coeff), int(h * rs_coeff)

Mimg_p=Mimg
Mimg_0=MImage(cv2Img=Mimg_p.cv2Img, copy=True)
window.label.img=Mimg_p

Mimg_1=MImage(cv2Img=Mimg_p.cv2Img, copy=True)
window.label_2.img= Mimg_1
#Mimg_2=MImage(cv2Img=Mimg_p.cv2Img)

def set_event_handler(label):
    label.paintEvent = lambda e, widg=label : paintEvent(widg,e)

    label.mousePressEvent = lambda e, wdg=label : mouseEvent(wdg, e)
    label.mouseMoveEvent = lambda e, wdg=label : mouseEvent(wdg, e)
    label.mouseReleaseEvent = lambda e, wdg=label : mouseEvent(wdg, e)

    label.wheelEvent = lambda e, wdg=label : wheelEvent(wdg, wdg.img, e)

set_event_handler(window.label)
set_event_handler(window.label_2)

window.label.setStyleSheet("background-color: rgb(200, 200, 200);")


img_0=Mimg_0.cv2Img
Mimg.rect = QRect(500, 400, Mimg.width-2000, Mimg.height-1000)

def button_change(widg,img0=Mimg_p, img1=Mimg_1):
    if str(widg.accessibleName()) == "Apply" :
        print "grabcut"
        do_grabcut(img0)
    elif str(widg.accessibleName()) == "Preview" :
        print "grabcut preview"
        window.label_2.img = do_grabcut(Mimg_p, preview=P_SIZE, mode=cv2.GC_INIT_WITH_MASK)
    print "done"
    window.label_2.repaint()

#window.onChange  = lambda widg ,img0=Mimg_p, img1=Mimg_1 : do_grabcut(img0, img1, preview=P_SIZE)

window.onChange = button_change


sys.exit(app.exec_())