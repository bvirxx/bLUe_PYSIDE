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
import os
from types import MethodType

import numpy as np
from PySide2 import QtWidgets, QtCore
from PySide2.QtCore import QSettings, Qt, QRect, QRectF, QEvent, QPointF
import sys

from PySide2.QtGui import QScreen, QPainter, QFont, QPainterPath, QColor, QBrush, QContextMenuEvent, QImage, QPen, \
    QCursor
from PySide2.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy

from bLUeGui.dialog import dlgWarn
from bLUeTop.pyside_dynamicLoader import loadUi
from bLUeTop.settings import MAX_ZOOM
from bLUeTop.splittedView import splittedWindow
from bLUeTop.utils import hideConsole, showConsole, QbLUeColorDialog, colorInfoView
from bLUeTop.versatileImg import vImage


class Form1(QMainWindow):
    """
    Main form class.
    """
    # screen changed signal
    screenChanged = QtCore.Signal(QScreen)

    def __init__(self):
        super(Form1, self).__init__()
        self.settings = QSettings("bLUe.ini", QSettings.IniFormat)
        # we presume that the form will be shown first on screen 0;
        # No detection possible before it is effectively shown !
        self.currentScreenIndex = 0
        self.__colorChooser, self.__infoView = (None,) * 2

    @property
    def colorChooser(self):
        if self.__colorChooser is None:
            self.__colorChooser = QbLUeColorDialog(parent=self)
        return self.__colorChooser

    @property
    def infoView(self):
        if self.__infoView is None:
            self.__infoView = colorInfoView()
            self.__infoView.label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
            self.__infoView.label.setMaximumSize(400, 80)
        return self.__infoView

    def init(self):
        """
        Load the form from the .ui file

        """
        from bLUeTop.graphicsHist import histForm
        from bLUeTop.layerView import QLayerView
        loadUi('bLUe.ui', baseinstance=self,
               customWidgets={'QLayerView': QLayerView, 'QLabel': QLabel, 'histForm': histForm})
        # hook called by event slots
        # should be redefined later
        self.onWidgetChange = lambda b: None
        # State recording.
        self.slidersValues = {}
        self.btnValues = {}
        # connect slider and button signals to slots
        for slider in self.findChildren(QtWidgets.QSlider):
            slider.valueChanged.connect(
                lambda value, slider=slider: self.handleSliderMoved(value, slider)
            )
            self.slidersValues[str(slider.accessibleName())] = slider.value()

        for button in self.findChildren(QtWidgets.QPushButton):
            # signal clicked has a default argument checked=False,
            # so we consume all passed args
            button.clicked.connect(
                lambda *args, button=button: self.handlePushButtonClicked(button)
            )
            self.btnValues[str(button.accessibleName())] = button.isChecked()

        for button in self.findChildren(QtWidgets.QToolButton):
            button.toggled.connect(
                lambda state, button=button: self.handleToolButtonClicked(button)
            )
            if not button.isCheckable():
                # signal clicked has a default argument checked=False
                # so we consume all args passed.
                button.clicked.connect(
                    lambda *args, button=button: self.handleToolButtonClicked(button)
                )
            self.btnValues[str(button.accessibleName())] = button.isChecked()

    def handlePushButtonClicked(self, button):
        """
        button clicked/toggled slot.
        @param button:
        @type button:
        """
        self.onWidgetChange(button)

    def handleToolButtonClicked(self, button):
        """
        button clicked/toggled slot.
        The toggled signal is triggered only by checkable buttons,
        when the button state changes. Thus, the method is called
        by all auto exclusive buttons in a group to correctly update
        the btnValues dictionary.
        @param button:
        @type button: QButton
        """
        self.btnValues[str(button.accessibleName())] = button.isChecked()
        self.onWidgetChange(button)

    def handleSliderMoved(self, value, slider):
        """
        Slider valueChanged slot
        @param value:
        @param slider:
        @type slider : QSlider
        """
        self.slidersValues[slider.accessibleName()] = value
        self.onWidgetChange(slider)

    def moveEvent(self, event):
        """
        Overriding moveEvent to emit
        a screenChanged signal
        when a screen change is detected.
        @param event:
        @type event:
        """
        super(Form1, self).moveEvent(event)
        # detecting screen changes :
        # getting current QScreen instance
        sn = self.windowHandle().screen()
        if sn is not self.currentScreenIndex:
            # screen changed detected
            self.currentScreenIndex = sn
            self.screenChanged.emit(sn)

    def closeEvent(self, event):
        if self.onCloseEvent():
            # close
            event.accept()
        else:
            # don't close
            event.ignore()
            return
        self.settings.sync()
        if getattr(sys, 'frozen', False):
            showConsole()
        super(Form1, self).closeEvent(event)


def enumerateMenuActions(menu):
    """
    Recursively builds the list of actions contained in a menu
    and in its submenus.
    @param menu: Qmenu object
    @return: list of actions
    """
    actions = []
    for action in menu.actions():
        # subMenu
        if action.menu():
            actions.extend(enumerateMenuActions(action.menu()))
            action.menu().parent()
        else:
            actions.append(action)
    return actions


########################
# Add plugin path to library path : mandatory to enable
# the loading of imageformat dlls for reading and writing QImage objects.
#######################
plugin_path = os.path.join(os.path.dirname(QtCore.__file__), "plugins")
QtCore.QCoreApplication.addLibraryPath(plugin_path)

######################
# Hide console for frozen app
# Pass an argument to program to keep console showing
#####################
if getattr(sys, 'frozen', False) and len(sys.argv) <= 1:
    hideConsole()

############
# launch app
############
QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)  # needed when a plugin initializes a Qt WebEngine
app = QApplication(sys.argv)
# get root widget
rootWidget = app.desktop()

#################
# init main form
# the UI is not loaded yet
window = Form1()
#################

# init global QPainter for paint event
qp = QPainter()
# stuff for Before/After tags
qp.font = QFont("Arial", 8)
qp.markPath = QPainterPath()
qp.markRect = QRect(0, 0, 50, 20)
qp.markPath.addRoundedRect(qp.markRect, 5, 5)


def paintEvent(widg, e, qp=qp):
    """
    Paint event handler.
    It displays the presentation layer of a vImage object in a Qlabel,
    with current offset and zooming coefficient.
    The widget must have a valid img attribute of type vImage.
    The handler should be used to override the paintEvent method of widg. This can be done
    by subclassing (not directly compatible with Qt Designer) or by assigning paintEvent
    to the method widg.paintEvent (cf. the function set_event_handler below).
    @param widg: widget
    @type widg: QLabel with a img attribute of type vImage
    @param e: paint event
    @type e:
    @param qp:
    @type qp: QPainter
    """
    mimg = getattr(widg, 'img', None)
    if mimg is None:
        return
    r = mimg.resize_coeff(widg)
    qp.begin(widg)
    # smooth painting
    qp.setRenderHint(QPainter.SmoothPixmapTransform)  #  may be useless
    # fill background
    qp.fillRect(QRect(0, 0, widg.width(), widg.height()), vImage.defaultBgColor)
    # draw the presentation layer.
    # As offsets can be float numbers, we use QRectF instead of QRect
    # r is relative to the full resolution image, so we use mimg width and height
    w, h = mimg.width() * r, mimg.height() * r
    rectF = QRectF(mimg.xOffset, mimg.yOffset, w, h)
    px = mimg.prLayer.qPixmap
    if px is not None:
        qp.drawPixmap(rectF, px, px.rect())
    else:
        currentImage = mimg.prLayer.getCurrentImage()
        qp.drawImage(rectF, currentImage, QImage.rect(currentImage))  # CAUTION : vImage.rect() is overwritten by attribute rect
    # draw the selection rectangle of the active layer, if any
    layer = mimg.getActiveLayer()
    rect = layer.rect
    if layer.visible and rect is not None:
        qp.setPen(QColor(0, 255, 0))
        qp.drawRect(rect.left()*r + mimg.xOffset, rect.top()*r + mimg.yOffset, rect.width()*r, rect.height()*r)
    # draw the cropping marks
    lm, rm, tm, bm = 0, 0, 0, 0
    if mimg.isCropped:
        c = QColor(128, 128, 128, 192)
        lm = window.cropTool.btnDict['left'].margin*r
        rm = window.cropTool.btnDict['right'].margin*r
        tm = window.cropTool.btnDict['top'].margin*r
        bm = window.cropTool.btnDict['bottom'].margin*r
        # left
        qp.fillRect(QRectF(mimg.xOffset, mimg.yOffset, lm, h), c)
        # top
        qp.fillRect(QRectF(mimg.xOffset+lm, mimg.yOffset, w - lm, tm), c)
        # right
        qp.fillRect(QRectF(mimg.xOffset+w-rm, mimg.yOffset+tm, rm, h-tm),  c)
        # bottom
        qp.fillRect(QRectF(mimg.xOffset+lm, mimg.yOffset+h-bm, w-lm-rm, bm), c)
    # draw rulers
    if mimg.isRuled:
        deltaX, deltaY = (w-lm-rm)//3, (h-tm-bm)//3
        qp.drawLine(lm+mimg.xOffset, deltaY+tm+mimg.yOffset, w-rm+mimg.xOffset, deltaY+tm+mimg.yOffset)
        qp.drawLine(lm+mimg.xOffset, 2*deltaY+tm+mimg.yOffset, w-rm+mimg.xOffset, 2*deltaY+tm+mimg.yOffset)
        qp.drawLine(deltaX+lm+mimg.xOffset, tm+mimg.yOffset, deltaX+lm+mimg.xOffset, h-bm+mimg.yOffset)
        qp.drawLine(2*deltaX+lm+mimg.xOffset, tm+mimg.yOffset, 2*deltaX+lm+mimg.xOffset, h-bm+mimg.yOffset)
    # tag before/after views
    name = widg.objectName()
    if name == "label_2" or name == "label_3":
        # draw filled rect
        qp.fillPath(qp.markPath, QBrush(Qt.gray))
        # draw text
        qp.setPen(Qt.white)
        qp.setFont(qp.font)
        qp.drawText(qp.markRect, Qt.AlignCenter | Qt.AlignVCenter, "Before" if name == "label_2" else "After")
    qp.end()


##############################################################
# global state variables used in mouseEvent.
pressed = False
clicked = True
# Recording of state and mouse coordinates (relative to widget)
State = {'ix': 0, 'iy': 0, 'ix_begin': 0, 'iy_begin': 0}
###############################################################

# Before/After view
splittedWin = splittedWindow(window)


def mouseEvent(widget, event, qp=qp, window=window):  # TODO split into 3 handlers
    """
    Mouse event handler.
    The handler implements mouse actions on a vImage in a QLabel.
    It handles image positioning, zooming, and
    tool actions. It must be called by the mousePressed,
    mouseMoved and mouseReleased methods of the QLabel. This can be done by
    subclassing (not directly compatible with Qt Designer) or by
    dynamically assigning mouseEvent to the former three methods
    (cf. the function set_event_handler below).
    NOTE 1. Mouse hover generates mouse move events
    NOTE 2. Due to wheeelEvent, xOffset and yOffset are float numbers
    @param widget:
    @type widget: QLabel object with img attribute of type mImage
    @param event: mouse event
    @type event: QMouseEvent
    @param window:
    @type window: QMainWidget
    """
    global pressed, clicked
    if type(event) == QContextMenuEvent:
        return
    # get image and active layer
    img = widget.img
    layer = img.getActiveLayer()
    r = img.resize_coeff(widget)
    # x, y coordinates (relative to widget)
    x, y = event.x(), event.y()
    modifiers = event.modifiers() # app.keyboardModifiers()
    eventType = event.type()
    ###################
    # mouse press event
    ###################
    if eventType == QEvent.MouseButtonPress:
        # Mouse hover generates mouse move events,
        # so, we set pressed to select only non hovering events
        pressed = True
        if event.button() == Qt.LeftButton:
            # no move yet
            clicked = True
        State['ix'], State['iy'] = x, y
        State['ix_begin'], State['iy_begin'] = x, y
        State['x_imagePrecPos'], State['y_imagePrecPos'] = (x - img.xOffset) // r, (y - img.yOffset) // r
        # add current mask to history
        if window.btnValues['drawFG'] or window.btnValues['drawBG']:
            if layer.maskIsEnabled:
                layer.historyListMask.addItem(layer.mask.copy())
        # dragBtn or arrow
        elif modifiers == Qt.ControlModifier | Qt.AltModifier:
            if layer.isCloningLayer():
                layer.updateCloningMask()
                layer.updateSourcePixmap()
        return
    ##################
    # mouse move event
    ##################
    elif eventType == QEvent.MouseMove:
        # hover event
        if not pressed:
            x_img, y_img = (x - img.xOffset) / r, (y - img.yOffset) / r
            # read input and current colors from active layer (coordinates are relative to the full-sized image)
            clr = img.getActivePixel(x_img, y_img, qcolor=True)
            clrC = img.getActivePixel(x_img, y_img, fromInputImg=False, qcolor=True)
            window.infoView.setText(clr, clrC)
            return
        clicked = False
        if img.isMouseSelectable:
            # don't draw on a non visible layer
            if window.btnValues['rectangle'] or window.btnValues['drawFG'] or window.btnValues['drawBG']:
                if not layer.visible:
                    dlgWarn('Select a visible layer for drawing or painting')
                    pressed = False
                    return
                elif not window.btnValues['rectangle'] and not layer.maskIsEnabled:
                    dlgWarn('Enable the mask before painting')
                    pressed = False
                    return
            # marquee tool
            if window.btnValues['rectangle']:
                # rectangle coordinates are relative to full image
                x_img = (min(State['ix_begin'], x) - img.xOffset) // r
                y_img = (min(State['iy_begin'], y) - img.yOffset) // r
                w = abs(State['ix_begin'] - x) // r
                h = abs(State['iy_begin'] - y) // r
                layer.rect = QRect(x_img, y_img, w, h)
            # drawing tools
            elif window.btnValues['drawFG'] or window.btnValues['drawBG']:
                if layer.maskIsEnabled:
                    toolOpacity = window.verticalSlider2.value() / 100
                    if modifiers == Qt.NoModifier:
                        if layer.isSegmentLayer():
                            color = vImage.defaultColor_UnMasked_SM if \
                                window.btnValues['drawFG'] else vImage.defaultColor_Masked_SM
                        else:
                            color = vImage.defaultColor_UnMasked if \
                                window.btnValues['drawFG'] else vImage.defaultColor_Masked
                    else:
                        color = vImage.defaultColor_UnMasked_Invalid
                    qp.begin(layer.mask)
                    # get pen width (relative to image)
                    w_pen = window.verticalSlider1.value() // r
                    # mode source : result is source (=pen) pixel color and opacity
                    qp.setCompositionMode(qp.CompositionMode_Source)
                    tmp_x = (x - img.xOffset) // r
                    tmp_y = (y - img.yOffset) // r
                    qp.setPen(QPen(color, w_pen))
                    qp.setOpacity(toolOpacity)
                    # paint the brush tips spaced by 0.25 * w_pen
                    # use 1-norm for performance
                    a_x, a_y = tmp_x - State['x_imagePrecPos'], tmp_y - State['y_imagePrecPos']
                    d = abs(a_x) + abs(a_y)
                    x, y = State['x_imagePrecPos'], State['y_imagePrecPos']
                    radius = w_pen / 2
                    if d == 0:
                        qp.drawEllipse(QPointF(x, y), radius, radius)  # center, radius : QPointF mandatory, else bounding rect topleft and size
                    else:
                        step = w_pen * 0.25 / d
                        for i in range(int(1 / step) + 1):
                            qp.drawEllipse(QPointF(x, y), radius, radius)  # center, radius : QPointF mandatory, else bounding rect topleft and size
                            x, y = x + a_x * step, y + a_y * step
                    qp.end()
                    State['x_imagePrecPos'], State['y_imagePrecPos'] = tmp_x, tmp_y
                    ############################
                    # update upper stack
                    # should be layer.applyToStack() if any upper layer visible : too slow
                    layer.updatePixmap()  # maskOnly=True not used: removed
                    #############################
                    img.prLayer.applyNone()
                    window.label.repaint()
            # dragBtn or arrow
            else:
                # drag image
                if modifiers == Qt.NoModifier:
                    img.xOffset += x - State['ix']
                    img.yOffset += y - State['iy']
                    if window.btnValues['Crop_Button']:
                        window.cropTool.drawCropTool(img)
                # drag active layer only
                elif modifiers == Qt.ControlModifier:
                    layer.xOffset += (x - State['ix'])
                    layer.yOffset += (y - State['iy'])
                    layer.updatePixmap()
                    img.prLayer.applyNone()
                # drag cloning virtual layer
                elif modifiers == Qt.ControlModifier | Qt.AltModifier:
                    if layer.isCloningLayer():
                        layer.xAltOffset += (x - State['ix'])
                        layer.yAltOffset += (y - State['iy'])
                        layer.vlChanged = True
                        if layer.maskIsSelected or not layer.maskIsEnabled:
                            layer.setMaskEnabled(color=False)
                        layer.applyCloning(seamless=False, showTranslated=True, moving=True)
        # not mouse selectable widget : probably before window alone !
        else:
            if modifiers == Qt.NoModifier:
                img.xOffset += (x - State['ix'])
                img.yOffset += (y - State['iy'])
            elif modifiers == Qt.ControlModifier:
                layer.xOffset += (x - State['ix'])
                layer.yOffset += (y - State['iy'])
                layer.updatePixmap()
        # update current coordinates
        State['ix'], State['iy'] = x, y
        if layer.isGeomLayer():
            layer.tool.moveRotatingTool()
    #####################
    # mouse release event
    # mouse click event
    ####################
    elif eventType == QEvent.MouseButtonRelease:
        pressed = False
        if event.button() == Qt.LeftButton:
            if layer.maskIsEnabled \
                    and layer.getUpperVisibleStackIndex() != -1\
                    and (window.btnValues['drawFG'] or window.btnValues['drawBG']):
                layer.applyToStack()
            if img.isMouseSelectable:
                # click event
                if clicked:
                    x_img, y_img = (x - img.xOffset) / r, (y - img.yOffset) / r
                    # read input and current colors from active layer (coordinates are relative to the full-sized image)
                    clr = img.getActivePixel(x_img, y_img, qcolor=True)
                    clrC = img.getActivePixel(x_img, y_img, fromInputImg=False, qcolor=True)
                    red, green, blue = clr.red(), clr.green(), clr.blue()
                    # read color from presentation layer
                    redP, greenP, blueP = img.getPrPixel(x_img, y_img)
                    # color chooser : when visible the colorPicked signal is not emitted
                    if getattr(window, 'colorChooser', None) and window.colorChooser.isVisible():
                        if (modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier):
                            window.colorChooser.setCurrentColor(clr)
                        elif modifiers & Qt.ControlModifier:
                            window.colorChooser.setCurrentColor(clrC)
                        else:
                            window.colorChooser.setCurrentColor(QColor(redP, greenP, blueP))
                    else:
                        # emit colorPicked signal
                        layer.colorPicked.sig.emit(x_img, y_img, modifiers)
                        # select grid node for 3DLUT form
                        if layer.is3DLUTLayer():
                            layer.getGraphicsForm().selectGridNode(red, green, blue)
                        # rectangle selection
                        if window.btnValues['rectangle'] and (modifiers == Qt.ControlModifier):
                                layer.rect = None
                                layer.selectionChanged.sig.emit()
                        # for raw layer, set multipliers to get selected pixel as White Point
                        if layer.isRawLayer() and window.btnValues['colorPicker']:
                            # get demosaic buffer and sample raw pixels
                            bufRaw = layer.parentImage.demosaic
                            nb = QRect(x_img-2, y_img-2, 4, 4)
                            r = QImage.rect(layer.parentImage).intersected(nb)
                            if not r.isEmpty():
                                color = np.sum(bufRaw[r.top():r.bottom()+1, r.left():r.right()+1], axis=(0, 1))/(r.width()*r.height())
                            else:
                                color = bufRaw[y_img, x_img, :]
                            color = [color[i] - layer.parentImage.rawImage.black_level_per_channel[i] for i in range(3)]
                            form = layer.getGraphicsForm()
                            if form.sampleMultipliers:
                                row, col = 3*y_img//layer.height(), 3*x_img//layer.width()
                                if form.samples:
                                    form.setRawMultipliers(*form.samples[3*row + col], sampling=False)
                            else:
                                form.setRawMultipliers(1/color[0], 1/color[1], 1/color[2], sampling=True)
                else:  # not clicked
                    if window.btnValues['rectangle']:
                        layer.selectionChanged.sig.emit()
                    # cloning layer
                    elif layer.isCloningLayer():
                        if layer.vlChanged:
                            # the virtual layer was moved : clone
                            layer.applyCloning(seamless=True, showTranslated=True, moving=True)
                            layer.vlChanged = False
    # updates
    widget.repaint()
    # sync split views
    linked = True
    if widget.objectName() == 'label_2':
        splittedWin.syncSplittedView(window.label_3, window.label_2, linked)
        window.label_3.repaint()
    elif widget.objectName() == 'label_3':
        splittedWin.syncSplittedView(window.label_2, window.label_3, linked)
        window.label_2.repaint()


def wheelEvent(widget, img, event, window=window):
    """
    Mouse wheel event handler : zooming
    for imImage objects.
    @param widget: widget displaying image
    @type widget: QWidget
    @param img: imImage object to display
    @type img:
    @param event: mouse wheel event
    @type event: QWheelEvent
    @param window:
    @type window: QMainWidget
    """
    pos = event.pos()
    modifiers = event.modifiers()
    # delta unit is 1/8 of degree
    # Most mice have a resolution of 15 degrees
    numSteps = event.delta() / 1200.0
    layer = img.getActiveLayer()
    if modifiers == Qt.NoModifier:
        img.Zoom_coeff *= (1.0 + numSteps)
        # max Zoom for previews
        if img.Zoom_coeff > MAX_ZOOM:
            img.Zoom_coeff /= (1.0 + numSteps)
            return
        # correct image offset to keep unchanged the image point
        # under the cursor : (pos - offset) / resize_coeff is invariant
        img.xOffset = -pos.x() * numSteps + (1.0+numSteps)*img.xOffset
        img.yOffset = -pos.y() * numSteps + (1.0+numSteps)*img.yOffset
        if window.btnValues['Crop_Button']:
            window.cropTool.drawCropTool(img)
        if layer.isGeomLayer():
            # layer.view.widget().tool.moveRotatingTool()
            layer.tool.moveRotatingTool()
    elif modifiers == Qt.ControlModifier:
        layer.Zoom_coeff *= (1.0 + numSteps)
        layer.updatePixmap()
    # cloning layer zoom
    elif layer.isCloningLayer and modifiers == Qt.ControlModifier | Qt.AltModifier:
        layer.AltZoom_coeff *= (1.0 + numSteps)
        layer.applyCloning(seamless=False, showTranslated=True, moving=True)
    widget.repaint()
    # sync split views
    linked = True
    if widget.objectName() == 'label_2':
        splittedWin.syncSplittedView(window.label_3, window.label_2, linked)
        window.label_3.repaint()
    elif widget.objectName() == 'label_3':
        splittedWin.syncSplittedView(window.label_2, window.label_3, linked)
        window.label_2.repaint()


def enterEvent(widget, img, event, window=window):
    """
    Mouse enter event handler
    @param widget:
    @type widget: QWidget
    @param img:
    @type img:
    @param event:
    @type event
    @param window:
    @type window: QMainWidget
    """
    if QApplication.overrideCursor():
        # don't stack multiple cursors
        return
    if window.btnValues['drawFG'] or window.btnValues['drawBG']:
        w = window.verticalSlider1.value()
        if w > 5:
            QApplication.setOverrideCursor(QCursor(window.cursor_Circle_Pixmap.scaled(w * 2.0, w * 2.0), hotX=w, hotY=w))
        else:
            QApplication.setOverrideCursor(Qt.CrossCursor)
    elif window.btnValues['drag']:
        QApplication.setOverrideCursor(Qt.OpenHandCursor)
    elif window.btnValues['colorPicker']:
        layer = window.label.img.getActiveLayer()
        if layer.isAdjustLayer():
            if layer.view.isVisible():
                QApplication.setOverrideCursor(window.cursor_EyeDropper)


def leaveEvent(widget, img, event):
    QApplication.restoreOverrideCursor()


def set_event_handlers(widg, enterAndLeave=True):
    """
    Pythonic redefinition of event handlers, without
    subclassing or overriding. However, the PySide dynamic
    ui loader requires that we set the corresponding classes as customWidget
    (cf. file QtGui1.py and pyside_dynamicLoader.py).
    if enterAndLeave is False enter and leave event handlers are not set,
    otherwise all mouse event handlers are set.
    @param widg:
    @type widg : QWidget
    @param enterAndLeave:
    @type enterAndLeave: boolean
    """
    widg.paintEvent = MethodType(lambda instance, e, wdg=widg: paintEvent(wdg, e), widg.__class__)
    widg.mousePressEvent = MethodType(lambda instance, e, wdg=widg: mouseEvent(wdg, e), widg.__class__)
    widg.mouseMoveEvent = MethodType(lambda instance, e, wdg=widg: mouseEvent(wdg, e), widg.__class__)
    widg.mouseReleaseEvent = MethodType(lambda instance, e, wdg=widg: mouseEvent(wdg, e), widg.__class__)
    widg.wheelEvent = MethodType(lambda instance, e, wdg=widg: wheelEvent(wdg, wdg.img, e), widg.__class__)
    if enterAndLeave:
        widg.enterEvent = MethodType(lambda instance, e, wdg=widg: enterEvent(wdg, wdg.img, e), widg.__class__)
        widg.leaveEvent = MethodType(lambda instance, e, wdg=widg: leaveEvent(wdg, wdg.img, e), widg.__class__)





