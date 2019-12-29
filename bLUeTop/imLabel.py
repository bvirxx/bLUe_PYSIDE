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
from PySide2.QtCore import QRect, QRectF, Qt, QPointF
from PySide2.QtGui import QPainter, QImage, QColor, QBrush, QContextMenuEvent, QCursor, QPen, QFont, QPainterPath
from PySide2.QtWidgets import QLabel, QApplication

from bLUeGui.dialog import dlgWarn
from bLUeTop.drawing import bLUeFloodFill
from bLUeTop.settings import MAX_ZOOM
from bLUeTop.versatileImg import vImage


class imageLabel(QLabel):

    # global state variables used by mouseEvent.
    #pressed = False
    #clicked = True
    #State = {'ix': 0, 'iy': 0, 'ix_begin': 0, 'iy_begin': 0, 'cloning': ''}

    qp = QPainter()
    qp.font = QFont("Arial", 8)
    qp.markPath = QPainterPath()
    qp.markRect = QRect(0, 0, 50, 20)
    qp.markPath.addRoundedRect(qp.markRect, 5, 5)

    def brushUpdate(self):
        """
        Records the current brush/eraser in self.State
        @return: current brush size
        @rtype: int
        """
        window = self.window
        bSize = window.verticalSlider1.value()
        bOpacity = window.verticalSlider2.value() / 100
        bHardness = window.verticalSlider3.value() / 100
        bFlow = window.verticalSlider4.value() / 100
        if getattr(window, 'colorChooser', None):
            bColor = window.colorChooser.selectedColor()
        else:
            bColor = Qt.black
        if window.btnValues['eraserButton']:
            self.State['brush'] = window.brushes[-1].getBrush(bSize, bOpacity, bColor, bHardness, bFlow)
        else:
            self.State['brush'] = window.brushCombo.currentData().getBrush(bSize, bOpacity, bColor, bHardness, bFlow)
        return bSize

    def syncBrush(self, zooming):
        """
        Sync current brush with tool bar and zoom coeff.
        Overrides or changes the application cursor.
        @param zooming:
        @type zooming: float
        """
        minSize = 16
        bSize = self.brushUpdate()
        w = bSize * zooming
        if w >= minSize:
            cursor = QCursor(self.State['brush']['cursor'].scaled(w, w), hotX=0, hotY=0)
        else:
            d = int((minSize - w) / 2)
            cursor = QCursor(self.State['brush']['cursor'].scaled(minSize, minSize), hotX=d, hotY=d)
        if QApplication.overrideCursor():
            QApplication.changeOverrideCursor(cursor)
        else:
            QApplication.setOverrideCursor(cursor)

    def __init__(self, mainForm=None, splitWin=None, enterAndLeave=False, parent=None):
        super().__init__(parent=parent)
        self.window = mainForm
        self.splitWin = splitWin
        self.enterAndLeave = enterAndLeave
        # global state variables used in mouseEvent.
        self.pressed = False
        self.clicked = True
        self.State = {'ix': 0, 'iy': 0, 'ix_begin': 0, 'iy_begin': 0, 'cloning': ''}
        self.img = None

    def paintEvent(self, e):
        """
        Overrides QLabel paintEvent().
        Displays the presentation layer of a vImage object,
        with its current offset and zooming coefficient.
        @param e: paint event
        @type e:
        """
        mimg = self.img
        if mimg is None:
            return
        r = mimg.resize_coeff(self)
        qp = self.qp
        window = self.window
        qp.begin(self)
        # smooth painting
        qp.setRenderHint(QPainter.SmoothPixmapTransform)  # TODO may be useless
        # fill background
        qp.fillRect(QRect(0, 0, self.width(), self.height()), vImage.defaultBgColor)
        # draw presentation layer.
        # As offsets can be float numbers, we use a QRectF instead of a QRect.
        # r is relative to the full resolution image, so we use mimg width and height
        w, h = mimg.width() * r, mimg.height() * r
        rectF = QRectF(mimg.xOffset, mimg.yOffset, w, h)
        px = mimg.prLayer.qPixmap
        if px is not None:
            qp.drawPixmap(rectF, px, px.rect())
        else:
            currentImage = mimg.prLayer.getCurrentImage()
            qp.drawImage(rectF, currentImage,
                         QImage.rect(currentImage))  # CAUTION : vImage.rect() is overwritten by the attribute rect
        # draw selection rectangle and cloning marker of the active layer, if any
        layer = mimg.getActiveLayer()
        rect, mark = layer.rect, layer.marker
        if layer.visible:
            if rect is not None:
                qp.setPen(Qt.green)
                qp.drawRect(rect.left() * r + mimg.xOffset, rect.top() * r + mimg.yOffset, rect.width() * r,
                            rect.height() * r)
            if not (mark is None or layer.sourceFromFile):
                qp.setPen(Qt.white)
                qp.drawEllipse(mark.x() * r + mimg.xOffset, mark.y() * r + mimg.yOffset, 10, 10)
        # draw the cropping marks
        lm, rm, tm, bm = 0, 0, 0, 0
        if mimg.isCropped:
            c = QColor(128, 128, 128, 192)
            lm = window.cropTool.btnDict['left'].margin * r
            rm = window.cropTool.btnDict['right'].margin * r
            tm = window.cropTool.btnDict['top'].margin * r
            bm = window.cropTool.btnDict['bottom'].margin * r
            qp.fillRect(QRectF(mimg.xOffset, mimg.yOffset, lm, h), c)  # left
            qp.fillRect(QRectF(mimg.xOffset + lm, mimg.yOffset, w - lm, tm), c)  # top
            qp.fillRect(QRectF(mimg.xOffset + w - rm, mimg.yOffset + tm, rm, h - tm), c)  # right
            qp.fillRect(QRectF(mimg.xOffset + lm, mimg.yOffset + h - bm, w - lm - rm, bm), c)  # bottom
        # draw rulers
        if mimg.isRuled:
            deltaX, deltaY = (w - lm - rm) // 3, (h - tm - bm) // 3
            qp.drawLine(lm + mimg.xOffset, deltaY + tm + mimg.yOffset, w - rm + mimg.xOffset,
                        deltaY + tm + mimg.yOffset)
            qp.drawLine(lm + mimg.xOffset, 2 * deltaY + tm + mimg.yOffset, w - rm + mimg.xOffset,
                        2 * deltaY + tm + mimg.yOffset)
            qp.drawLine(deltaX + lm + mimg.xOffset, tm + mimg.yOffset, deltaX + lm + mimg.xOffset,
                        h - bm + mimg.yOffset)
            qp.drawLine(2 * deltaX + lm + mimg.xOffset, tm + mimg.yOffset, 2 * deltaX + lm + mimg.xOffset,
                        h - bm + mimg.yOffset)
        # tag before/after views
        name = self.objectName()
        if name == "label_2" or name == "label_3":
            # draw filled rect
            qp.fillPath(qp.markPath, QBrush(Qt.gray))
            # draw text
            qp.setPen(Qt.white)
            qp.setFont(qp.font)
            qp.drawText(qp.markRect, Qt.AlignCenter | Qt.AlignVCenter, "Before" if name == "label_2" else "After")
        qp.end()

    def mousePressEvent(self, event):
        """
        Mouse event handlers.
        The handlers implement mouse actions on an imImage displayed in a QLabel.
        It handles image positioning, zooming, and
        tool actions.
        NOTE. Due to wheeelEvent, xOffset and yOffset are float numbers
        @param event: mouse event
        @type event: QMouseEvent
        """
        State = self.State
        window = self.window
        eventType = event.type()
        # no context menu
        if eventType == QContextMenuEvent:
            return
        # get image and active layer
        img = self.img
        layer = img.getActiveLayer()
        r = img.resize_coeff(self)
        ############################################################
        # get mouse x, y coordinates (relative to widget).
        # The mouse coordinates relative to the (full size) image are
        # (x - img.xOffset) / r, (y - img.yOffset) / r
        #############################################################
        x, y = event.x(), event.y()
        modifiers = event.modifiers()
        # Mouse hover generates mouse move events,
        # so, we set pressed to select only non hovering events
        self.pressed = True
        if event.button() == Qt.LeftButton:
            # no move yet
            self.clicked = True
        State['ix'], State['iy'] = x, y
        State['ix_begin'], State['iy_begin'] = x, y
        State['x_imagePrecPos'], State['y_imagePrecPos'] = (x - img.xOffset) // r, (y - img.yOffset) // r
        if layer.isDrawLayer():
            layer.history.addItem(layer.sourceImg.copy())
            if window.btnValues['brushButton'] or window.btnValues['bucket']:
                # starting new stroke : save initial image for atomic stroke painting  and init intermediate layer
                layer.strokeDest = layer.sourceImg.copy()
                layer.stroke.fill(QColor(0, 0, 0, 0))
                if window.btnValues['brushButton']:
                    self.syncBrush(r)
            """
            elif window.btnValues['eraserButton']:
                layer.stroke = layer.sourceImg.copy()
                layer.strokeDest = layer.sourceImg.copy()
            """
        # add current mask to history
        if window.btnValues['drawFG'] or window.btnValues['drawBG']:
            if layer.maskIsEnabled:
                layer.historyListMask.addItem(layer.mask.copy())
        # dragBtn or arrow
        if layer.isCloningLayer():
            if not (window.btnValues['drawFG'] or window.btnValues['drawBG']):
                if modifiers == Qt.ControlModifier | Qt.AltModifier:  # prevent unwanted clicks
                    if layer.cloningState == 'continue':
                        dlgWarn('Layer already cloned', 'To start a new cloning operation, add another cloning layer')
                        return
                    # set source starting point (coordinates are relative to full size image)
                    layer.sourceX, layer.sourceY = (x - img.xOffset) / r, (y - img.yOffset) / r
                    layer.cloningState = 'start'
            elif layer.cloningState == 'start':
                # set the virtual layer translation (relative to full size image)
                layer.xAltOffset, layer.yAltOffset = (x - img.xOffset) / r - layer.sourceX,\
                                                     (y - img.yOffset) / r - layer.sourceY
                layer.cloningState = 'continue'
                layer.updateCloningMask()
                layer.updateSourcePixmap()

    def mouseMoveEvent(self, event):
        """
       Mouse event handlers.
       The handlers implement mouse actions on an imImage displayed in a QLabel.
       It handles image positioning, zooming, and
       tool actions.
       NOTE 1. Mouse hover generates mouse move events
       NOTE 2. Due to wheeelEvent, xOffset and yOffset are float numbers
       @param event: mouse event
       @type event: QMouseEvent
       """
        window = self.window
        State = self.State
        qp = self.qp
        eventType = event.type()
        # no context menu
        if eventType == QContextMenuEvent:
            return
        # get image and active layer
        img = self.img
        layer = img.getActiveLayer()
        r = img.resize_coeff(self)
        ############################################################
        # get mouse x, y coordinates (relative to widget).
        # The mouse coordinates relative to the (full size) image are
        # (x - img.xOffset) / r, (y - img.yOffset) / r
        #############################################################
        x, y = event.x(), event.y()
        modifiers = event.modifiers()
        # hover event
        if not self.pressed:
            x_img, y_img = (x - img.xOffset) / r, (y - img.yOffset) / r
            # read input and current colors from active layer (coordinates are relative to the full-sized image)
            clr = img.getActivePixel(x_img, y_img, qcolor=True)
            clrC = img.getActivePixel(x_img, y_img, fromInputImg=False, qcolor=True)
            window.infoView.setText(clr, clrC)
            if layer.isCloningLayer():
                if layer.cloningState == 'continue':
                    mx, my = x_img - layer.xAltOffset, y_img - layer.yAltOffset
                elif layer.cloningState == 'start':
                    mx, my = layer.sourceX , layer.sourceY
                else:
                    mx, my = x_img, y_img
                if layer.sourceFromFile:
                    mx, my = mx * layer.getGraphicsForm().sourcePixmap.width() / layer.width(), my * layer.getGraphicsForm().sourcePixmap.height() / layer.height()
                layer.marker = QPointF(mx, my)
                window.label.repaint()
                if layer.sourceFromFile:
                    layer.getGraphicsForm().widgetImg.repaint()
            return
        self.clicked = False
        if img.isMouseSelectable:
            # don't draw on a non visible layer
            if window.btnValues['rectangle'] or window.btnValues['drawFG'] or window.btnValues['drawBG']:
                if not layer.visible:
                    dlgWarn('Select a visible layer for drawing or painting')
                    self.pressed = False
                    return
                elif not window.btnValues['rectangle'] and not layer.maskIsEnabled:
                    dlgWarn('Enable the mask before painting')
                    self.pressed = False
                    return
            # marquee tool
            if window.btnValues['rectangle']:
                # rectangle coordinates are relative to full image
                x_img = (min(State['ix_begin'], x) - img.xOffset) // r
                y_img = (min(State['iy_begin'], y) - img.yOffset) // r
                w = abs(State['ix_begin'] - x) // r
                h = abs(State['iy_begin'] - y) // r
                layer.rect = QRect(x_img, y_img, w, h)
            # drawing
            elif layer.isDrawLayer() and (window.btnValues['brushButton'] or window.btnValues['eraserButton']):
                self.__strokePaint(layer, x, y, r)
            # mask
            elif window.btnValues['drawFG'] or window.btnValues['drawBG']:
                if layer.maskIsEnabled:
                    if layer.isCloningLayer:
                        layer.vlChanged = True  # TODO added 16/12/19
                        # layer.setMaskEnabled(color=True)  # set mask to color mask
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
                        qp.drawEllipse(QPointF(x, y), radius,
                                       radius)  # center, radius : QPointF mandatory, else bounding rect topleft and size
                    else:
                        step = w_pen * 0.25 / d
                        for i in range(int(1 / step) + 1):
                            qp.drawEllipse(QPointF(x, y), radius,
                                           radius)  # center, radius : QPointF mandatory, else bounding rect topleft and size
                            x, y = x + a_x * step, y + a_y * step
                    qp.end()
                    if layer.isCloningLayer():
                        if not layer.sourceFromFile:
                            layer.marker = QPointF(tmp_x - layer.xAltOffset, tmp_y - layer.yAltOffset)
                        else:
                            layer.marker = QPointF((tmp_x - layer.xAltOffset) * layer.getGraphicsForm().sourcePixmap.width() / layer.width(), (tmp_y - layer.yAltOffset) * layer.getGraphicsForm().sourcePixmap.height() / layer.height())
                            layer.getGraphicsForm().widgetImg.repaint()
                    State['x_imagePrecPos'], State['y_imagePrecPos'] = tmp_x, tmp_y
                    ############################
                    # update upper stack
                    # should be layer.applyToStack() if any upper layer visible : too slow
                    # layer.applyToStack()
                    layer.updatePixmap()
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
                            layer.setMaskEnabled(color=False)  # set to opacity mask
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
        # updates
        self.repaint()
        # sync split views
        linked = True
        if self.objectName() == 'label_2':
            self.splitWin.syncSplitView(window.label_3, window.label_2, linked)
            window.label_3.repaint()
        elif self.objectName() == 'label_3':
            self.splitWin.syncSplitView(window.label_2, window.label_3, linked)
            window.label_2.repaint()

    def mouseReleaseEvent(self, event):
        """
       Mouse event handlers.
       The handlers implement mouse actions on an imImage displayed in a QLabel.
       It handles image positioning, zooming, and
       tool actions.
       @param event: mouse event
       @type event: QMouseEvent
       """
        window = self.window
        p = self.qp
        eventType = event.type()
        # no context menu
        if eventType == QContextMenuEvent:
            return
        # get image and active layer
        img = self.img
        layer = img.getActiveLayer()
        r = img.resize_coeff(self)
        ############################################################
        # get mouse x, y coordinates (relative to widget).
        # The mouse coordinates relative to the (full size) image are
        # (x - img.xOffset) / r, (y - img.yOffset) / r
        #############################################################
        x, y = event.x(), event.y()
        modifiers = event.modifiers()
        self.pressed = False
        if event.button() == Qt.LeftButton:
            if layer.maskIsEnabled \
                    and layer.getUpperVisibleStackIndex() != -1 \
                    and (window.btnValues['drawFG'] or window.btnValues['drawBG']):
                layer.applyToStack()
            if img.isMouseSelectable:
                # click event
                if self.clicked:
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
                        # Flood fill tool
                        if layer.isDrawLayer() and window.btnValues['bucket']:
                            if getattr(window, 'colorChooser', None):
                                bucketColor = window.colorChooser.selectedColor()
                            else:
                                bucketColor = Qt.black
                            bLUeFloodFill(layer, int(x_img), int(y_img), bucketColor)
                            layer.applyToStack()
                        """
                        # for raw layer, set multipliers to get selected pixel as White Point : NOT USED YET
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
                        """
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
        self.repaint()
        # sync split views
        linked = True
        if self.objectName() == 'label_2':
            self.splitWin.syncSplitView(window.label_3, window.label_2, linked)
            window.label_3.repaint()
        elif self.objectName() == 'label_3':
            self.splitWin.syncSplitView(window.label_2, window.label_3, linked)
            window.label_2.repaint()

    def wheelEvent(self, event):
        """
        Mouse wheel event handler : zooming
        for imImage objects.
        @param event: mouse wheel event
        @type event: QWheelEvent
        """
        img = self.img
        window = self.window
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
            img.xOffset = -pos.x() * numSteps + (1.0 + numSteps) * img.xOffset
            img.yOffset = -pos.y() * numSteps + (1.0 + numSteps) * img.yOffset
            if layer.isDrawLayer() and (window.btnValues['brushButton'] or window.btnValues['eraserButton']):
                self.syncBrush(img.resize_coeff(self))
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
            layer.applyCloning(seamless=False, showTranslated=True, moving=True)  # autocloning (seamless=true) too slow
        self.repaint()
        # sync split views
        linked = True
        if self.objectName() == 'label_2':
            self.splitWin.syncSplitView(window.label_3, window.label_2, linked)
            window.label_3.repaint()
        elif self.objectName() == 'label_3':
            self.splitWin.syncSplitView(window.label_2, window.label_3, linked)
            window.label_2.repaint()

    def enterEvent(self, event):
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
        if not self.enterAndLeave:
            return
        window = self.window
        if QApplication.overrideCursor():
            # don't stack multiple cursors
            return
        # tool cursors
        w = window.verticalSlider1.value()
        layer = window.label.img.getActiveLayer()
        if window.btnValues['drawFG'] or window.btnValues['drawBG']:
            if w > 10:
                QApplication.setOverrideCursor(
                    QCursor(window.cursor_Circle_Pixmap.scaled(w * 2.0, w * 2.0), hotX=w, hotY=w))
            else:
                QApplication.setOverrideCursor(Qt.CrossCursor)
        elif layer.isDrawLayer() and (window.btnValues['brushButton'] or window.btnValues['eraserButton']):
            self.syncBrush(self.img.resize_coeff(self))
        elif window.btnValues['drag']:
            QApplication.setOverrideCursor(Qt.OpenHandCursor)
        elif window.btnValues['colorPicker']:
            if layer.isAdjustLayer():
                if layer.view.isVisible():
                    QApplication.setOverrideCursor(window.cursor_EyeDropper)

    def leaveEvent(self, event):
        if not self.enterAndLeave:
            return
        QApplication.restoreOverrideCursor()

    def __movePaint(self, x, y, r, radius, pxmp=None):
        """
        Private drawing function.
        Base function for painting tools. The radius and pixmap
        of the tool are passed by the parameters radius and pxmp.
        The parameter qp must be an active QPainter.
        Starting coordinates of the move are recorded in State, ending coordinates
        are passed by parameters x, y.
        Successive brush tips, spaced by 0.25 * radius, are paint on qp.
        @param x: move event x-coord
        @type x: float
        @param y: move event y-coord
        @type y: float
        @param r: image resizing coeff
        @type r: float
        @param radius: tool radius
        @type radius: float
        @param pxmp: brush pixmap
        @type pxmp: QPixmap
        @return: last painted position
        @rtype: 2-uple of float
        """
        img = self.img
        State = self.State
        qp = self.qp
        tmp_x = (x - img.xOffset) // r
        tmp_y = (y - img.yOffset) // r
        # vector of the move
        a_x, a_y = tmp_x - State['x_imagePrecPos'], tmp_y - State['y_imagePrecPos']
        # move length : use 1-norm for performance
        d = abs(a_x) + abs(a_y)
        step = 1 if d == 0 else radius * 0.25 / d
        p_x, p_y = State['x_imagePrecPos'], State['y_imagePrecPos']
        count = 0
        while count * step < 1:
            count += 1
            if pxmp is None:
                qp.drawEllipse(QPointF(p_x, p_y), radius, radius)
            else:
                qp.drawPixmap(QPointF(p_x, p_y), pxmp)
            p_x, p_y = p_x + a_x * step, p_y + a_y * step
        # return last painted position
        return p_x, p_y

    def __strokePaint(self, layer, x, y, r):
        """
        Private drawing function. Should be called only by the mouse event handler
        @param img:
        @type img:
        @param layer:
        @type layer:
        @param x:
        @type x:
        @param y:
        @type y:
        @param r:
        @type r:
        @param window:
        @type window:
        @param State:
        @type State:
        """
        img = self.img
        State = self.State
        qp = self.qp
        radius = State['brush']['size'] / 2
        # drawing into stroke intermediate layer
        if self.window.btnValues['brushButton']:
            cp = layer.stroke
            qp.begin(cp)
            qp.setCompositionMode(qp.CompositionMode_SourceOver)
            State['x_imagePrecPos'], State['y_imagePrecPos'] = self.__movePaint(x, y, r, radius,
                                                                           pxmp=State['brush']['pixmap'])
            qp.end()
            # paint the whole stroke with current brush opacity
            layer.sourceImg = layer.strokeDest.copy()
            qp.begin(layer.sourceImg)
            qp.setOpacity(State['brush']['opacity'])
            qp.setCompositionMode(qp.CompositionMode_SourceOver)
            qp.drawImage(QPointF(0, 0), layer.stroke)
            qp.end()
        elif self.window.btnValues['eraserButton']:
            cp = layer.sourceImg  # layer.stroke
            qp.begin(cp)
            qp.setCompositionMode(qp.CompositionMode_DestinationIn)
            State['x_imagePrecPos'], State['y_imagePrecPos'] = self.__movePaint(x, y, r, radius,
                                                                           pxmp=State['brush']['pixmap'])
            qp.end()
        # update layer - should be layer.applyToStack() if any upper layer visible : too slow !
        # layer.updatePixmap()
        layer.execute()
        img.prLayer.applyNone()
        self.window.label.repaint()







