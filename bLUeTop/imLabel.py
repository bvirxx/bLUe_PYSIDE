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
from math import sqrt
from random import choice

from PySide2.QtCore import QRect, QRectF, Qt, QPointF
from PySide2.QtGui import QPainter, QImage, QColor, QBrush, QContextMenuEvent, QCursor, QPen, QFont, QPainterPath, \
    QTransform
from PySide2.QtWidgets import QLabel, QApplication

from bLUeGui.dialog import dlgWarn
from bLUeTop.drawing import bLUeFloodFill, brushFamily
from bLUeTop.settings import MAX_ZOOM
from bLUeTop.utils import checkeredImage
from bLUeTop.versatileImg import vImage

class imageLabel(QLabel):

    qp = QPainter()
    qp.font = QFont("Arial", 8)
    qp.markPath = QPainterPath()
    qp.markRect = QRect(0, 0, 50, 20)
    qp.markPath.addRoundedRect(qp.markRect, 5, 5)

    checkerBrush = QBrush(checkeredImage())

    def brushUpdate(self):
        """
        Sync the current brush/eraser with self.State
        and update the current brush sample.
        """
        bSpacing, bJitter, bOrientation  = 1.0, 0.0, 0
        current = self.State.get('brush', None)
        if current is not None:
            bSpacing = current['spacing']
            bJitter = current['jitter']
            bOrientation = current['orientation']
        s = self.sender()
        if s is not None:
            name = s.objectName()
            if name == 'spacingSlider':
                bSpacing = s.value() / 10
            elif name == 'jitterSlider':
                bJitter = s.value() / 10
            elif name == 'orientationSlider':
                bOrientation = s.value() - 180
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
            self.State[''] = window.brushes[-1].getBrush(bSize, bOpacity, bColor, bHardness, bFlow)
        else:
            pattern = window.patternCombo.currentData()
            self.State['brush'] = window.brushCombo.currentData().getBrush(bSize, bOpacity, bColor, bHardness, bFlow, spacing=bSpacing, jitter=bJitter, orientation=bOrientation, pattern=pattern)
        # record current brush into layer brushDict
        if self.img is not None:
            layer = self.img.getActiveLayer()
            if layer.isDrawLayer():
                layer.brushDict = self.State['brush'].copy()
                grForm = layer.getGraphicsForm()
                if grForm is not None:
                    grForm.updateSample()

    def syncBrush(self, zooming):
        """
        Sync current brush with tool bar and zoom coeff.
        Overrides or changes the application cursor.
        @param zooming:
        @type zooming: float
        """
        minSize = 16
        # bSize = self.brushUpdate()
        bSize = self.State['brush']['size']  # TODO modified 26/01/20 validate
        w = bSize * zooming
        if w >= minSize:
            cursor = QCursor(self.State['brush']['cursor'].scaled(w, w), hotX=w/2, hotY=w/2)
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
        # draw a checker background to view (semi-)transparent images
        qp.fillRect(rectF, imageLabel.checkerBrush)
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
                # starting a new stroke : save initial image for atomic stroke painting  and init intermediate layer
                layer.strokeDest = layer.sourceImg.copy()
                layer.stroke.fill(QColor(0, 0, 0, 0))
                if window.btnValues['brushButton']:
                    self.syncBrush(r)
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
                    mx, my = layer.sourceX, layer.sourceY
                else:
                    mx, my = x_img, y_img
                if layer.sourceFromFile:
                    pxmp = layer.getGraphicsForm().sourcePixmap
                    mx, my = mx * pxmp.width() / layer.width(), my * pxmp.height() / layer.height()
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
                            pxmp = layer.getGraphicsForm().sourcePixmap
                            layer.marker = QPointF((tmp_x - layer.xAltOffset) * pxmp.width() / layer.width(),
                                                   (tmp_y - layer.yAltOffset) * pxmp.height() / layer.height())
                            layer.getGraphicsForm().widgetImg.repaint()
                    State['x_imagePrecPos'], State['y_imagePrecPos'] = tmp_x, tmp_y
                    ############################
                    # update upper stack
                    # should be layer.applyToStack() if any upper layer visible : too slow
                    # layer.applyToStack()
                    layer.updatePixmap()
                    img.prLayer.update()  # =applyNone()
                    #############################
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
                    img.prLayer.update()  # =applyNone()
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
                        if modifiers == Qt.ControlModifier | Qt.ShiftModifier:  # (modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier):
                            window.colorChooser.setCurrentColor(clr)
                        elif modifiers == Qt.ControlModifier:  # modifiers & Qt.ControlModifier:
                            window.colorChooser.setCurrentColor(clrC)
                        else:
                            window.colorChooser.setCurrentColor(QColor(redP, greenP, blueP))
                    else:
                        # emit colorPicked signal
                        layer.colorPicked.sig.emit(x_img, y_img, modifiers)
                        # select grid node for 3DLUT form
                        if layer.is3DLUTLayer():
                            movedNodes = layer.getGraphicsForm().selectGridNode(red, green, blue)
                            if movedNodes:
                                layer.applyToStack()
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
        elif layer.isCloningLayer() and modifiers == Qt.ControlModifier | Qt.AltModifier:
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
        @param event:
        @type event
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

    """
    def __movePaint(self, x, y, r, radius, pxmp=None):
        
        Private drawing function.
        Base function for painting tools. The radius and pixmap
        of the tool are passed by the parameters radius and pxmp.
        Starting coordinates of the move are recorded in State, ending coordinates
        are passed by parameters x, y.
        Brush tips are drawn on the intermediate layer self.stroke,
        using self.qp.
        Successive brush tips, spaced by 0.25 * radius, are paint on qp.
        Note that self.qp must have been previously activated.
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
        
        img = self.img
        State = self.State
        qp = self.qp
        tmp_x = (x - img.xOffset) // r
        tmp_y = (y - img.yOffset) // r
        # vector of the move
        a_x, a_y = tmp_x - State['x_imagePrecPos'], tmp_y - State['y_imagePrecPos']
        # move length : use 1-norm for performance
        d = sqrt(a_x * a_x + a_y * a_y)
        spacing, jitter = State['brush']['spacing'], State['brush']['jitter']
        step = 1 if d == 0 else radius * 0.3 * spacing / d  # 0.25
        if jitter != 0.0:
            step *= (1.0 + choice(brushFamily.jitterRange) * jitter / 100.0)
        p_x, p_y = State['x_imagePrecPos'], State['y_imagePrecPos']
        if d != 0.0:
            cosTheta, sinTheta = a_x / d, a_y / d
            transform = QTransform(cosTheta, sinTheta, -sinTheta, cosTheta, 0, 0)  # Caution: angles > 0 correspond to counterclockwise rotations of pxmp
            pxmp = pxmp.transformed(transform)
        count = 0
        maxCount = int( 1.0 / step)
        while count < maxCount:
            count += 1
            if pxmp is None:
                qp.drawEllipse(QPointF(p_x, p_y), radius, radius)
            else:
                qp.drawPixmap(QPointF(p_x - radius, p_y - radius), pxmp)  # TODO radius added 19/03/20 validate
            p_x, p_y = p_x + a_x * step, p_y + a_y * step
        # return last painted position
        return p_x, p_y
    """

    def __strokePaint(self, layer, x, y, r):
        """
        Private drawing function. Should be called only by the mouse event handler
        @param layer:
        @type layer:
        @param x:
        @type x:
        @param y:
        @type y:
        @param r:
        @type r:
        """
        img = self.img
        State = self.State
        qp = self.qp
        # get image coordinates
        x_img = (x - img.xOffset) // r
        y_img = (y - img.yOffset) // r
        # draw the stroke
        if self.window.btnValues['brushButton']:
            # drawing onto stroke intermediate layer
            qp.begin(layer.stroke)
            qp.setCompositionMode(qp.CompositionMode_SourceOver)
            # draw move
            State['x_imagePrecPos'], State['y_imagePrecPos'] = brushFamily.brushStrokeSeg(qp,
                                                                                          State['x_imagePrecPos'],
                                                                                          State['y_imagePrecPos'],
                                                                                          x_img, y_img,
                                                                                          State['brush'])
            qp.end()
            # draw texture aligned with image
            strokeTex = layer.stroke
            p = State['brush']['pattern']
            if p is not None:
                if p.pxmp is not None:
                    strokeTex = layer.stroke.copy()
                    qp1 = QPainter(strokeTex)
                    qp1.setCompositionMode(qp.CompositionMode_DestinationIn)
                    qp1.setBrush(QBrush(p.pxmp))
                    qp1.fillRect(QRect(0, 0, strokeTex.width(), strokeTex.height()), QBrush(p.pxmp))
                    qp1.end()
            # restore source image and paint
            # the whole stroke with current brush opacity.
            # Restoring source image enables iterative calls showing
            # stroke progress
            qp.begin(layer.sourceImg)
            qp.setCompositionMode(qp.CompositionMode_Source)
            qp.drawImage(QPointF(), layer.strokeDest)
            qp.setOpacity(State['brush']['opacity'])
            qp.setCompositionMode(qp.CompositionMode_SourceOver)
            qp.drawImage(QPointF(), strokeTex)  # layer.stroke)
            qp.end()
        elif self.window.btnValues['eraserButton']:
            qp.begin(layer.sourceImg)
            qp.setCompositionMode(qp.CompositionMode_DestinationIn)
            State['x_imagePrecPos'], State['y_imagePrecPos'] = brushFamily.brushStrokeSeg(qp,
                                                                                          State['x_imagePrecPos'],
                                                                                          State['y_imagePrecPos'],
                                                                                          x_img, y_img,
                                                                                          State['brush'])
            qp.end()
        # update layer - should be layer.applyToStack() if any upper layer visible : too slow !
        layer.execute()
        img.prLayer.update()
        self.window.label.repaint()
