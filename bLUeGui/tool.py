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
import numpy as np

from PySide6.QtGui import QImage, QTransform, QPolygonF, QPainter, QFontMetrics, QColor, QBrush, QPalette, QAction, \
    QActionGroup, QTextCursor, QTextCharFormat, QFontInfo, QFont, QPainterPath, QTextOption
from PySide6.QtWidgets import QWidget, QToolButton, QPlainTextEdit, QMenu, QFontDialog
from PySide6.QtCore import Qt, QPoint, QObject, QPointF, QRectF, QRect, QSize

from bLUeGui.dialog import dlgWarn
from bLUeGui.memory import weakProxy
from bLUeTop.utils import QbLUeColorDialog


class baseHandle(QToolButton):
    """
    Base class for interactive tool handles. Their
    positions in image may be modified by the user, dragging them
    with the mouse. They can be kept invariant under
    image zooming and panning.
    """

    def __init__(self, role='', tool=None, pos=QPointF(0,0), parent=None):
        """

        :param role:
        :type role: str
        :param tool: collector tool
        :type tool: baseTool
        :param pos: position relative to full size image
        :type pos: QPointF
        :param parent: parent Widget
        :type parent: QWidget
        """
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation)  # avoid interference with parent mouse event
        self.role = role
        self.margin = 0.0
        # back link to the tool
        self.tool = weakProxy(tool)
        self.setVisible(False)
        self.setGeometry(0, 0, 10, 10)
        self.setAutoFillBackground(True)
        self.setAutoRaise(True)
        self.posRelImg = pos  # position relative to full size image
        self.setStyleSheet('QToolButton:hover {background-color:#00FF00} QToolButton {background-color:#555555}')

    def mouseMoveEvent(self, e):
        """

        :param e:
        :type  e:
        """
        # skip hover events and programmatic moves
        if e.buttons() == Qt.MouseButton.NoButton:
            return
        widget = self.parent()
        img = widget.img
        r = img.resize_coeff(widget)
        self.posRelImg = (self.mapToParent(e.position()) - QPointF(img.xOffset, img.yOffset)) / r
        self.move(QPoint(img.xOffset, img.yOffset) + (self.posRelImg * r).toPoint())
        if self.tool:
            self.tool.syncToolWithLayer()


class baseTool(QObject):
    """
    Base class for interactive image tools.
    A tool is a collection of baseHandle buttons, recorded in a dictionary.
    Each button is draggable with the mouse and holds a role attribute,
    defining the type of action executed when the button is moved.
    For example, tools are used to perform interactive geometric transformations
    of an image.
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.btnDict = {}

    def showTool(self):
        for btn in self.btnDict.values():
            btn.show()

    def hideTool(self):
        for btn in self.btnDict.values():
            btn.hide()

    def setVisible(self, value):
        for btn in self.btnDict.values():
            btn.setVisible(value)

    def syncToolWithLayer(self, **kwargs):
        pass

class markTool(baseTool):
    """
    Simple tool to position an interactive text area on a text layer.
    """

    def __init__(self, layer=None,  parent=None):
        """

        :param layer:
        :type layer: QTextLayer
        :param parent:
        :type parent: QWidget
        """
        super().__init__(parent=parent)
        self.layer = layer
        if layer:
            self.layer.tool = self  # add tool to layer
        self.textWdg = transparentWidget(tool=self, targetLayer=layer, parent=parent)
        self.img = layer.parentImage
        # rotation/translation tool for text
        self.rottool = rotatingKbTool(parent=parent)
        self.rottool.addTool(layer, setattr=False)  # don't change layer.tool
        self.rottool.showTool()

        self.layer.visibilityChanged.sig.connect(self.setVisible)
        self.addMarker(QPointF(50, 50), role='topleft') # position relative to full size image
        self.addMarker(QPointF(150, 100), role='bottomright')
        self.syncToolWithLayer()
        self.textWdg.show()

    def showTool(self):
        super().showTool()
        self.textWdg.show()

    def hideTool(self):
        super().hideTool()
        if self.textWdg.editorInstance:
            self.textWdg.editorInstance.hide()
        self.textWdg.hide()

    def setVisible(self, value):
        super().setVisible(value)
        if self.textWdg.editorInstance:
            self.textWdg.editorInstance.setVisible(value)
        self.textWdg.setVisible(value)

    def addMarker(self, pos, role=''):
        """
        Adds a marker at position pos (relative to full size image).

        :param role:
        :type role: str
        :param pos: position relative to full size image
        :type  pos: QPointF
        """
        btn = baseHandle(role=role , pos=pos, tool=self, parent=self.parent())
        self.btnDict[btn.role] = btn
        r = self.img.resize_coeff(self.parent())
        btn.move(QPoint(self.layer.parentImage.xOffset, self.layer.parentImage.yOffset) + (pos * r).toPoint())

    def syncToolWithLayer(self, zooming=False):
        """

        :param zooming:
        :type zooming: bool
        """
        img = self.layer.parentImage
        r = img.resize_coeff(self.parent())
        for btn in self.btnDict.values():
            posRelToParent = QPoint(img.xOffset, img.yOffset) + (btn.posRelImg * r).toPoint()
            #btn.move(posRelToParent)
            match btn.role:
                case 'topleft':
                    # keep button bottom right corner fixed on image
                    btn.move(posRelToParent - QPoint(btn.width(), btn.height()))
                    self.textWdg.move(posRelToParent) # + QPoint(btn.width(), btn.height()))
                case 'bottomright':
                    # keep button top left corner fixed on image
                    btn.move(posRelToParent)
                    s = btn.pos() - self.btnDict['topleft'].pos() - QPoint(btn.width(), btn.height())
                    self.textWdg.setFixedSize(s.x() , s.y())

        if self.textWdg.editorInstance:
            self.textWdg.editorInstance.syncWithTool(zooming=zooming)

        if self.rottool:
            self.rottool.syncToolWithLayer()

class croppingHandle(baseHandle):
    """
    Simple active button, draggable with the mouse.
    When moved, it updates the cropping margins of an image

    """

    def __init__(self, role='', tool=None, pos=QPointF(0,0), parent=None):
        """
        parent should be the widget showing the edited imImage.
        roles are 'left', 'right', 'top', 'bottom',
        'topRight', 'topLeft', 'bottomRight', 'bottomLeft'

        :param role:
        :type role: str
        :param tool: collector tool
        :type tool: baseTool
        :param pos: position relative to full size image
        :type pos: QPointF
        :param parent: parent Widget
        :type parent: QWidget
        """
        super().__init__(role=role, tool=tool, pos=pos, parent=parent)
        self.setStyleSheet("QToolButton:hover {background-color:#00FF00} QToolButton {background-color:#555555}")

    def setPosition(self, p):
        """
        Updates button margins in response to a mouse move event.

        :param p: mouse cursor position (relative to parent widget)
        :type  p: QPoint

        """
        widg = self.parent()
        img = widg.img
        r = img.resize_coeff(widg)
        lMargin, rMargin, tMargin, bMargin = img.cropLeft, img.cropRight, img.cropTop, img.cropBottom
        # middle buttons
        if self.role == 'left':
            margin = int((p.x() - img.xOffset + self.width()) / r)
            if margin < 0 or margin >= img.width() - self.tool.btnDict['right'].margin:
                return
            self.margin = margin
            lMargin = margin
        elif self.role == 'right':
            margin = img.width() - int((p.x() - img.xOffset) / r)
            if margin < 0 or margin >= img.width() - self.tool.btnDict['left'].margin:
                return
            self.margin = margin
            rMargin = margin
        elif self.role == 'top':
            margin = int((p.y() - img.yOffset + self.height()) / r)
            if margin < 0 or margin >= img.height() - self.tool.btnDict['bottom'].margin:
                return
            self.margin = margin
            tMargin = margin
        elif self.role == 'bottom':
            margin = img.height() - int((p.y() - img.yOffset) / r)
            if margin < 0 or margin >= img.height() - self.tool.btnDict['top'].margin:
                return
            self.margin = margin
            bMargin = margin
        # vertex buttons: keep current form factor
        elif self.role == 'topRight':
            rMargin = img.width() - (p.x() - img.xOffset) / r
            lMargin = self.tool.btnDict['left'].margin
            bMargin = self.tool.btnDict['bottom'].margin
            w = img.width() - rMargin - lMargin
            h = w * self.tool.formFactor
            tMargin = img.height() - h - bMargin
            if rMargin < 0 or rMargin >= img.width() - lMargin or tMargin < 0 or tMargin >= img.height() - bMargin:
                return
            self.tool.btnDict['right'].margin = rMargin
            self.tool.btnDict['top'].margin = tMargin
        elif self.role == 'topLeft':
            lBtn = self.tool.btnDict['left']
            lMargin = (p.x() - img.xOffset + lBtn.width()) / r
            rMargin = self.tool.btnDict['right'].margin
            bMargin = self.tool.btnDict['bottom'].margin
            w = img.width() - lMargin - rMargin
            h = w * self.tool.formFactor
            tMargin = img.height() - h - bMargin
            if lMargin < 0 or lMargin >= img.width() - rMargin or tMargin < 0 or tMargin >= img.height() - bMargin:
                return
            self.tool.btnDict['top'].margin = tMargin
            self.tool.btnDict['left'].margin = lMargin
        elif self.role == 'bottomLeft':
            lBtn = self.tool.btnDict['left']
            lMargin = (p.x() - img.xOffset + lBtn.width()) / r
            rMargin = self.tool.btnDict['right'].margin
            tMargin = self.tool.btnDict['top'].margin
            w = img.width() - lMargin - rMargin
            h = w * self.tool.formFactor
            bMargin = img.height() - h - tMargin
            if lMargin < 0 or lMargin >= img.width() - rMargin or bMargin < 0 or tMargin >= img.height() - bMargin:
                return
            self.tool.btnDict['bottom'].margin = bMargin
            self.tool.btnDict['left'].margin = lMargin
        elif self.role == 'bottomRight':
            btn = self.tool.btnDict['right']
            rMargin = img.width() - (p.x() - img.xOffset) / r
            lMargin = self.tool.btnDict['left'].margin
            tMargin = self.tool.btnDict['top'].margin
            w = img.width() - lMargin - rMargin
            h = w * self.tool.formFactor
            bMargin = img.height() - h - tMargin
            if rMargin < 0 or rMargin >= img.width() - lMargin or bMargin < 0 or bMargin >= img.height() - tMargin:
                return
            self.tool.btnDict['right'].margin = rMargin
            self.tool.btnDict['bottom'].margin = bMargin
        img.cropLeft, img.cropRight, img.cropTop, img.cropBottom = lMargin, rMargin, tMargin, bMargin

    def mousePressEvent(self, event):
        img = self.parent().img
        self.tool.crHeight = img.height() - int(img.cropTop + img.cropBottom)
        self.tool.crWidth = img.width() - int(img.cropLeft + img.cropRight)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.NoButton:
            return
        img = self.parent().img
        pos = self.mapToParent(event.position().toPoint())
        oldPos = self.pos()
        if self.role in ['left', 'right']:
            self.setPosition(self.pos() + QPoint((pos - oldPos).x(), 0))
        elif self.role in ['top', 'bottom']:
            self.setPosition(self.pos() + QPoint(0, (pos - oldPos).y()))
        # vertex buttons
        else:
            self.setPosition(pos)
        self.tool.setCropTool(self.parent().img)
        self.tool.crHeight = img.height() - int(img.cropTop + img.cropBottom)
        self.tool.crWidth = img.width() - int(img.cropLeft + img.cropRight)
        self.parent().updateStatus()
        self.parent().repaint()


class cropTool(baseTool):
    """

    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        cropButtonLeft = croppingHandle(role='left', tool=self, parent=parent)
        cropButtonRight = croppingHandle(role='right', tool=self, parent=parent)
        cropButtonTop = croppingHandle(role='top', tool=self, parent=parent)
        cropButtonBottom = croppingHandle(role='bottom', tool=self, parent=parent)
        cropButtonTopLeft = croppingHandle(role='topLeft', tool=self, parent=parent)
        cropButtonTopRight = croppingHandle(role='topRight', tool=self, parent=parent)
        cropButtonBottomLeft = croppingHandle(role='bottomLeft', tool=self, parent=parent)
        cropButtonBottomRight = croppingHandle(role='bottomRight', tool=self, parent=parent)
        btnList = [cropButtonLeft, cropButtonRight, cropButtonTop, cropButtonBottom,
                   cropButtonTopLeft, cropButtonTopRight, cropButtonBottomLeft, cropButtonBottomRight]
        self.btnDict = {btn.role: btn for btn in btnList}
        self.crHeight, self.crWidth = 1, 1

    def fit(self, img):
        """

        :param img:
        :type  img: vImage
        """
        for role, margin in zip(['left', 'top', 'right', 'bottom'],
                                [img.cropLeft, img.cropTop, img.cropRight, img.cropBottom]):
            self.btnDict[role].margin = margin

    def setCropTool(self, img):
        """
        Positions the 8 crop buttons around the image,
        using their current margin values.

        :param img:
        :type  img: QImage
        """
        r = self.parent().img.resize_coeff(self.parent())
        self.formFactor = img.height() / img.width()
        left = self.btnDict['left']
        top = self.btnDict['top']
        bottom = self.btnDict['bottom']
        right = self.btnDict['right']
        # get cropping rectangle (image coord.)
        cRect = QRectF(left.margin, top.margin, img.width() - right.margin - left.margin,
                       img.height() - bottom.margin - top.margin)
        # get widget coord. of cRect
        p = cRect.topLeft() * r + QPoint(img.xOffset, img.yOffset)
        x, y = p.x(), p.y()
        w, h = cRect.width() * r, cRect.height() * r
        # move buttons to their right position
        left.move(x - left.width(), y + h // 2)
        right.move(x + w, y + h // 2)
        top.move(x + w // 2, y - top.height())
        bottom.move(x + w // 2, y + h)
        topLeft = self.btnDict['topLeft']
        topLeft.move(x - topLeft.width(), y - topLeft.height())
        topRight = self.btnDict['topRight']
        topRight.move(x + w, y - topRight.height())
        bottomLeft = self.btnDict['bottomLeft']
        bottomLeft.move(x - bottomLeft.width(), y + h)
        bottomRight = self.btnDict['bottomRight']
        bottomRight.move(x + w, y + h)
        self.crWidth, self.crHeight = img.width() - int(img.cropLeft + img.cropRight), \
                                      img.height() - int(img.cropTop + img.cropBottom)

    def moveCrop(self, deltaX, deltaY, img):
        """
        Translates the crop tool.
        Coordinates are relative to widget.

        :param deltaX: move x-coord.
        :type  deltaX: int
        :param deltaY: move y-coord.
        :type  deltaY: int
        :param img:
        :type  img:
        """

        speed = 4  # deltaX = 1 or deltaY = 1 may lead to an incorrect move, due to int/float conversions
        # in setPosition() and setCropTool()
        deltaX, deltaY = speed * deltaX, speed * deltaY  # parent widget coord.

        lm, rm, tm, bm = self.btnDict['left'].margin + deltaX, self.btnDict['right'].margin - deltaX, \
                         self.btnDict['top'].margin + deltaY, self.btnDict['bottom'].margin - deltaY

        if lm >= 0 and rm >= 0 and tm >= 0 and bm >= 0:
            self.btnDict['left'].margin += deltaX
            self.btnDict['right'].margin -= deltaX
            self.btnDict['top'].margin += deltaY
            self.btnDict['bottom'].margin -= deltaY

            self.setCropTool(img)

    def zoomCrop(self, pos, numSteps, img):
        """
        Crop tool aware zooming.
        Img is zoomed, keeping cropTool and the point of img under cursor both fixed on the screen.

        :param pos: mouse cursor position (relative to widget)
        :type  pos: QPoint
        :param numSteps: relative wheel rotation
        :type  numSteps: float
        :param img:
        :type  img: vImage
        """

        btnList = [self.btnDict[name] for name in ['left', 'right', 'top', 'bottom']]

        img.xOffset = -pos.x() * numSteps + (1.0 + numSteps) * img.xOffset
        img.yOffset = -pos.y() * numSteps + (1.0 + numSteps) * img.yOffset

        img.Zoom_coeff *= 1.0 + numSteps

        for btn in btnList:
            btn.setPosition(btn.pos())

        self.setCropTool(img)


class rotatingHandle(baseHandle):
    """
    Active button for interactive (geometric) transformations
    """

    def __init__(self, role=None, tool=None, pos=QPointF(0, 0), parent=None):
        """

        :param role:
        :type  role: str
        :param tool:
        :type  tool: rotatingTool
        :param pos:
        :type  pos: QPointF
        :param parent: parent widget
        :type  parent: QWidget
        """
        super().__init__(role=role, tool=tool, pos=pos,  parent=parent)
        # set coordinates (relative to the full resolution image)
        self.posRelImg_ori = pos  # starting pos, never modified if a transformation is in progress (cf. rotatingTool.getOriginQuad)
        self.posRelImg = pos  # current pos (cf. rotatingTool.getTargetQuad)
        self.posRelImg_frozen = pos  # starting pos for the current type of transformation
        self.setStyleSheet("QToolButton:hover {background-color:#00FF00} QToolButton {background-color:#AA0000}")

    def mouseMoveEvent(self, event):
        """
        Mouse move event handler.

        :param event:
        :type  event:
        """
        # skip hover events and programmatic moves
        if event.buttons() == Qt.MouseButton.NoButton:
            return

        modifiers = event.modifiers()

        # get new button position (relative to image
        pos = self.mapToParent(event.position())
        img = self.tool.layer.parentImage
        r = img.resize_coeff(self.parent())
        self.posRelImg = (pos - QPointF(img.xOffset, img.yOffset)) / r

        # Ctrl+Alt : set the current position as starting position
        if modifiers == Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier:
            if self.tool.isModified():
                dlgWarn("A transformation is in progress", "Reset first")
                return
            # update the new starting  position
            self.posRelImg_ori = self.posRelImg  # (pos - QPoint(img.xOffset, img.yOffset)) / r
            self.posRelImg_frozen = self.posRelImg
            self.tool.syncToolWithLayer()  # self.tool.moveRotatingTool()
            self.tool.parent().repaint()
            return

        form = self.tool.getForm()
        if form.options['Free']:
            pass
        elif form.options['Rotation']:
            center = self.tool.getTargetQuad().boundingRect().center()
            v = QPointF(self.posRelImg.x() - center.x(), self.posRelImg.y() - center.y())
            v0 = QPointF(self.posRelImg_frozen.x() - center.x(), self.posRelImg_frozen.y() - center.y())
            theta = (np.arctan2(v.y(), v.x()) - np.arctan2(v0.y(), v0.x())) * 180.0 / np.pi
            T = QTransform()  # self.tool.geoTrans_ori)
            T.translate(center.x(), center.y()).rotate(theta).translate(-center.x(), -center.y())

        elif form.options['Translation']:
            # translation vector (coordinates are relative to the full size image)
            p = QPointF(self.posRelImg) - QPointF(self.posRelImg_frozen)
            T = QTransform()
            T.translate(p.x(), p.y())

        # update all button positions
        q = T.map(self.tool.getFrozenQuad())
        for i, role in enumerate(['topLeft', 'topRight', 'bottomRight', 'bottomLeft']):
            self.tool.btnDict[role].posRelImg = q.at(i)
            self.tool.btnDict[role].posRelImg_frozen = self.tool.btnDict[role].posRelImg

        self.tool.syncToolWithLayer()  # self.tool.moveRotatingTool()
        self.tool.modified = True
        self.tool.layer.applyToStack()
        self.parent().repaint()

class rotatingKbHandle(rotatingHandle):
    """
    Active button for interactive rotations and translations  of text layer
    """

    def mouseMoveEvent(self, event):
        """
        Mouse move event handler.

        :param event:
        :type  event:
        """
        # skip hover events and programmatic moves
        if event.buttons() == Qt.MouseButton.NoButton:
            return

        modifiers = event.modifiers()

        # get new button position (relative to image
        pos = self.mapToParent(event.position())
        img = self.tool.layer.parentImage
        r = img.resize_coeff(self.parent())
        self.posRelImg = (pos - QPointF(img.xOffset, img.yOffset)) / r

        deltaT = QTransform()

        # Shift: rotation
        if modifiers == Qt.KeyboardModifier.ShiftModifier:
            center = self.tool.getTargetQuad().boundingRect().center()
            v = self.posRelImg - center  #QPointF(self.posRelImg.x() - center.x(), self.posRelImg.y() - center.y())
            v0 = self.posRelImg_frozen - center  #QPointF(self.posRelImg_frozen.x() - center.x(), self.posRelImg_frozen.y() - center.y())
            # get the angle of rotation
            delta = (np.arctan2(v.y(), v.x()) - np.arctan2(v0.y(), v0.x())) * 180.0 / np.pi
            # get the rotation, centered on center
            #deltaT = QTransform()
            deltaT.translate(center.x(), center.y()).rotate(delta).translate(-center.x(), -center.y())
        # Alt: translation
        elif modifiers == Qt.KeyboardModifier.AltModifier:
            # translation vector (coordinates are relative to the full size image)
            deltaV = self.posRelImg - self.posRelImg_frozen
            #deltaT = QTransform()
            deltaT.translate(deltaV.x(), deltaV.y())

        # update all button positions
        q = deltaT.map(self.tool.getFrozenQuad())
        for i, role in enumerate(['topLeft', 'topRight', 'bottomRight', 'bottomLeft']):
            self.tool.btnDict[role].posRelImg = q.at(i)
            self.tool.btnDict[role].posRelImg_frozen = self.tool.btnDict[role].posRelImg

        self.tool.syncToolWithLayer()  # self.tool.moveRotatingTool()
        self.tool.modified = True

        #self.tool.layer.applyToStack()
        self.tool.layer.execute(l=self.tool.layer)  #, bRect=layer.uRect)
        img.prLayer.update() #bRect=layer.uRect)
        self.parent().repaint()

class rotatingTool(baseTool):
    """
    Provides interactive modifications of a base geometric transformation..
    Different types of transformations can be performed in a cumulative way.
    The tool buttons can be positioned anywhere in the image : this can be useful
    in particular for perspective correction.
    """
    rotatingHandleType = rotatingHandle

    def __init__(self, parent=None): #, layer=None):
        """
        Inits a rotatingTool instance and adds it to the parent widget.

        :param parent: parent widget
        :type  parent: QWidget
        :param layer: image layer
        :type  layer: QLayer
        """
        super().__init__(parent=parent)
        self.modified = False
        self.currentTransformation = QTransform()
        #self.layer = layer
        #if self.layer is None:
        w, h = 1.0, 1.0
        """
        else:
            self.layer.tool = self
            self.img = layer.parentImage
            self.layer.visibilityChanged.sig.connect(self.setVisible)
            w, h = self.img.width(), self.img.height()
        """
        # init tool buttons. The parameter pos is relative to the full size image.
        rotatingButtonLeft = self.rotatingHandleType(role='topLeft', tool=self, pos=QPointF(0, 0), parent=parent)
        rotatingButtonRight = self.rotatingHandleType(role='topRight', tool=self, pos=QPointF(w, 0), parent=parent)
        rotatingButtonTop = self.rotatingHandleType(role='bottomLeft', tool=self, pos=QPointF(0, h), parent=parent)
        rotatingButtonBottom = self.rotatingHandleType(role='bottomRight', tool=self, pos=QPointF(w, h), parent=parent)
        # init button dictionary
        btnList = [rotatingButtonLeft, rotatingButtonRight, rotatingButtonTop, rotatingButtonBottom]
        self.btnDict = {btn.role: btn for btn in btnList}
        #if self.layer is not None:
            #self.syncToolWithLayer()  # self.moveRotatingTool()

    def addTool(self, layer, setattr=True):
        """
        Adds tool to layer.

        :param layer:
        :type layer: QLayer
        """
        if setattr:
            layer.tool = self
        self.modified = False
        self.layer = weakProxy(layer)
        try:
            self.layer.visibilityChanged.sig.disconnect()
        except RuntimeError:
            pass
        self.layer.visibilityChanged.sig.connect(self.setVisible)
        self.img = weakProxy(layer.parentImage)
        w, h = self.img.width(), self.img.height()
        for role, pos in zip(['topLeft', 'topRight', 'bottomRight', 'bottomLeft'],
                             [QPointF(0, 0), QPointF(w, 0), QPointF(w, h), QPointF(0, h)]):
            self.btnDict[role].posRelImg = pos
            self.btnDict[role].posRelImg_ori = pos
            self.btnDict[role].posRelImg_frozen = pos
        self.syncToolWithLayer()  # self.moveRotatingTool()

    def getForm(self):
        if self.layer is not None:
            return self.layer.getGraphicsForm()
        return None

    def setBaseTransform(self):
        """
        Saves the current quad.
        """
        q = self.getTargetQuad()
        for i, role in enumerate(['topLeft', 'topRight', 'bottomRight', 'bottomLeft']):
            self.btnDict[role].posRelImg_frozen = q.at(i)

    def syncToolWithLayer(self, **kwargs):
        self.moveRotatingTool()

    def isModified(self):
        return self.modified

    def getTargetQuad(self):
        """
        Returns the current quad, as defined by the 4 buttons.
        Coordinates are relative to the full size image.

        :return:
        :rtype: QPolygonF
        """
        poly = QPolygonF()
        for role in ['topLeft', 'topRight', 'bottomRight', 'bottomLeft']:
            poly.append(self.btnDict[role].posRelImg)
        return poly

    def getOriginQuad(self):
        """
        Returns the origin quad.
        Coordinates are relative to the full size image

        :return:
        :rtype: QPolygonF
        """
        poly = QPolygonF()
        for role in ['topLeft', 'topRight', 'bottomRight', 'bottomLeft']:
            poly.append(self.btnDict[role].posRelImg_ori)
        return poly

    def getFrozenQuad(self):
        """
        Returns the starting quad for the current type of transformation.

        :return:
        :rtype: QPolygonF
        """
        poly = QPolygonF()
        for role in ['topLeft', 'topRight', 'bottomRight', 'bottomLeft']:
            poly.append(self.btnDict[role].posRelImg_frozen)
        return poly

    def getTransform(self):
        """
        Returns the current geometric transformation T,
        mapping the source quad to the target quad.
        If the computation fails, returns the identity transformation.

        :return:
        :rtype: QTransform
        """
        sourceQuad = self.getOriginQuad()
        targetQuad = self.getTargetQuad()
        T = QTransform()
        ok = QTransform.quadToQuad(sourceQuad, targetQuad, T)
        if not ok:
            dlgWarn("Cannot compute transformation", "Using identity")
            T = QTransform()
        return T

    def restore(self):
        for i, role in enumerate(['topLeft', 'topRight', 'bottomRight', 'bottomLeft']):  # order matters
            self.btnDict[role].posRelImg = self.targetQuad_old.at(i)
        self.syncToolWithLayer()  # moveRotatingTool()

    def moveRotatingTool(self):
        """
        Moves the tool buttons to the vertices of  the displayed image.
        Should be called every time that posRelImg is changed or the
        image zooming coeff. or position in widget
        are modified.
        """
        # get parent widget
        parent = self.parent()
        r = parent.img.resize_coeff(parent)
        topLeft = self.btnDict['topLeft']
        topRight = self.btnDict['topRight']
        bottomLeft = self.btnDict['bottomLeft']
        bottomRight = self.btnDict['bottomRight']
        # move buttons : coordinates are relative to parent widget
        #p = QPoint(self.img.xOffset, self.img.yOffset)
        x, y = self.img.xOffset, self.img.yOffset
        bottomLeft.move(x + bottomLeft.posRelImg.x() * r, y - bottomLeft.height() + bottomLeft.posRelImg.y() * r)
        bottomRight.move(x - bottomRight.width() + bottomRight.posRelImg.x() * r,
                         y - bottomRight.height() + bottomRight.posRelImg.y() * r)
        topLeft.move(x + topLeft.posRelImg.x() * r, y + topLeft.posRelImg.y() * r)
        topRight.move(x - topRight.width() + topRight.posRelImg.x() * r, y + topRight.posRelImg.y() * r)
        """
        T = self.currentTransformation
        #transformedImg = self.layer.sourceImg.transformed(self.currentTransformation, Qt.TransformationMode.SmoothTransformation)
        transformedImg = self.layer.sourceImg.copy(int(-T.m31()), int(-T.m32()), self.layer.rect().width(), self.layer.rect().height())
        self.layer.sourceImg = transformedImg  #.copy(self.layer.sourceImg.rect())
        """

    def resetTrans(self):
        self.modified = False
        w, h = self.img.width(), self.img.height()
        for role, pos in zip(['topLeft', 'topRight', 'bottomRight', 'bottomLeft'],
                             [QPoint(0, 0), QPoint(w, 0), QPoint(w, h), QPoint(0, h)]):
            self.btnDict[role].posRelImg = pos
            self.btnDict[role].posRelImg_ori = pos
            self.btnDict[role].posRelImg_frozen = pos
        self.syncToolWithLayer()  # moveRotatingTool()
        # self.frozenQuad = self.getTargetQuad()
        self.layer.applyToStack()
        self.parent().repaint()


class rotatingKbTool(rotatingTool):
    """
    Rotating tool with keyboard modifiers support.
    """

    rotatingHandleType = rotatingKbHandle

    def moveRotatingTool(self):
        """
        Moves the tool buttons to their new positions, and updates
        the transformed image.
        Must be called every time that posRelImg is changed, or the
        image zooming coefficient or positions in widget
        are modified.
        """
        # get parent widget
        parent = self.parent()
        r = parent.img.resize_coeff(parent)
        topLeft = self.btnDict['topLeft']
        topRight = self.btnDict['topRight']
        bottomLeft = self.btnDict['bottomLeft']
        bottomRight = self.btnDict['bottomRight']
        # move buttons : coordinates are relative to parent widget
        # p = QPoint(self.img.xOffset, self.img.yOffset)
        x, y = self.img.xOffset, self.img.yOffset
        bottomLeft.move(x + bottomLeft.posRelImg.x() * r, y - bottomLeft.height() + bottomLeft.posRelImg.y() * r)
        bottomRight.move(x - bottomRight.width() + bottomRight.posRelImg.x() * r,
                         y - bottomRight.height() + bottomRight.posRelImg.y() * r)
        topLeft.move(x + topLeft.posRelImg.x() * r, y + topLeft.posRelImg.y() * r)
        topRight.move(x - topRight.width() + topRight.posRelImg.x() * r, y + topRight.posRelImg.y() * r)

        #if self.layer.tool.textWdg.editorInstance:
            #self.layer.tool.textWdg.editorInstance.strokeText()  #editorInstance.snap()

        self.layer.sourceImg  = self.transformedSourceImg()

    def transformedSourceImg(self):
        """
        Computes the transformation defined by the origin and target quads,
        and applies it to the layer source image.
        :return: The transformed source image
        :rtype: QImage
        """
        if self.layer.tool.textWdg.savedSourceImg:
            inImg = self.layer.tool.textWdg.savedSourceImg.copy() #self.layer.sourceImg
        else:
            inImg = self.layer.sourceImg.copy()

        w, h = inImg.width(), inImg.height()
        #s = 1 # w / self.width()  # equal to h / self.height()
        #D = QTransform().scale(s, s)
        #DInv = QTransform().scale(1 / s, 1 / s)
        q1Full, q2Full = self.getOriginQuad(), self.getTargetQuad()
        # map Quads to the current image coordinate system
        #q1, q2 = D.map(q1Full), D.map(q2Full)
        # build transformation
        T = QTransform()
        #res = QTransform.quadToQuad(q1, q2, T)
        res = QTransform.quadToQuad(q1Full, q2Full, T)
        if not res:
            logger.warning('applyTransform : no possible transformation')
            self.tool.restore()
            return
        """
        # neutral point
        if T.isIdentity():
            buf1 = QImageBuffer(inImg)
            buf0[:, :, :] = buf1
            self.updatePixmap()
            return
        """
        # get the bounding rect of the transformed image (in the full size image coordinate system)
        # (Avoid the conversion of QTransforms to QMatrix4x4 and matrix product)
        #rectTrans = DInv.map(T.map(D.map(self.layer.rect()))).boundingRect()
        rectTrans = T.map(self.layer.rect()).boundingRect()
        # apply the transformation and re-translate the transformed image
        # so that the resulting transformation is T and NOT that given by QImage.trueMatrix()
        # img = (inImg.transformed(T)).copy(QRect(-rectTrans.x() * s, -rectTrans.y() * s, w, h))
        img = (inImg.transformed(T)).copy(QRect(-rectTrans.topLeft(), QSize(w, h)))

        return img


class blueTextEdit(QPlainTextEdit):  #QLabel):
    """
    Text editor.
    """

    def __init__(self, tool=None, parent=None):
        """

        :param tool:
        :type tool: markTool
        :param parent: parent widget
        :type parent: Qwidget
        """
        super().__init__( parent=parent)
        self.tool = tool
        self.targetLayer = self.tool.layer
        #self.setWindowFlags(Qt.Window)  # | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_PaintOnScreen)
        self.setAttribute(Qt.WA_NoMousePropagation)  # otherwise, right click propagates to parent, I don't know why !
        pl = self.palette()
        textColor = pl.color(QPalette.WindowText )
        pl.setBrush(QPalette.Window, QBrush(QColor(255, 0, 255, 255)))
        pl.setColor(QPalette.WindowText, QColor(0,255,0))
        self.setPalette(pl)
        #self.setPalette(Pl)
        #self.setMouseTracking(True)
        #self.setStyleSheet("QPlainTextEdit {background-color: white; color: red;}")
        self.iniFont = QFont('Arial', 40)
        #self.setCurrentFont(self.iniFont)
        self.setFont(self.iniFont)
        self.iniFontInfo = QFontInfo(self.iniFont)
        self.setPlainText("Enter text here")
        self.setTextColor(QColor(255, 255, 255), selected=False)
        self.current_resize_coeff = 1.0
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.contextMenu = None
        self.syncWithTool()

    def syncSizeWithTool(self, font_coeff):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)  #, QTextCursor.KeepAnchor)
        endPos = cursor.position()
        cursor.movePosition(QTextCursor.Start) # , QTextCursor.KeepAnchor)
        while cursor.position() < endPos:
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor) # select one symbol
            fmt = cursor.charFormat()
            ps = fmt.fontPointSize()
            fmt.setFontPointSize(ps * font_coeff)
            """
            font = fmt.font()
            if font.pixelSize() == -1:
                font.setPixelSize(QFontInfo(font).pixelSize())
            s = font.pixelSize()
            font.setPixelSize(int(s * font_coeff))
            fmt.setFont(font)
            """
            cursor.mergeCharFormat(fmt)
            cursor.setPosition(cursor.position())  # move anchor to current position
        cursor.clearSelection()
        self.setTextCursor(cursor)  # update visible cursor
        return


        block = self.document().firstBlock()
        while block.isValid():
            cursor.setPosition(block.position(),  QTextCursor.KeepAnchor)
            cursor.setPosition(block.position() + block.length())
            fmt = cursor.charFormat()
            font = fmt.font()
            if font.pixelSize() == -1:
                font.setPixelSize(QFontInfo(font).pixelSize())
            s = font.pixelSize()
            font.setPixelSize(int(s * font_coeff))
            fmt.setFont(font)
            cursor.setCharFormat(fmt)
            block = block.next()

    def syncWithTool(self, zooming=False):
        """
        Syncs the text editor window with the tool and target layer.
        Adjusts position, size and font size.
        """
        img = self.targetLayer.parentImage
        r = img.resize_coeff(self.parent())
        font_coeff = r / self.current_resize_coeff
        self.current_resize_coeff = r
        for btn in self.tool.btnDict.values():
            posRelToParent = QPoint(img.xOffset, img.yOffset) + (btn.posRelImg * r).toPoint()
            #btn.move(posRelToParent)
            match btn.role:
                case 'topleft':
                    # keep button bottom right corner fixed on image
                    btn.move(posRelToParent - QPoint(btn.width(), btn.height()))
                    self.move(posRelToParent)  # + QPoint(btn.width(), btn.height()))
                case 'bottomright':
                    # keep button top left corner fixed on image
                    btn.move(posRelToParent)
                    s = btn.pos() - self.tool.btnDict['topleft'].pos() - QPoint(btn.width(), btn.height())
                    self.setFixedSize(s.x() , s.y())
        if zooming:
            self.syncSizeWithTool(font_coeff)

        fs = self.iniFontInfo.pixelSize()
        #font = self.currentFont()
        font = self.font()
        font.setPixelSize(int(fs * r))
        #self.setCurrentFont(font)
        self.setFont(font)

    def initContextMenu(self):
        menu = self.createStandardContextMenu()
        action = QAction("Hide Editor", self)
        menu.addAction(action)
        action.triggered.connect(self.hide)

        action = QAction("Snap Text", self)
        menu.addAction(action)
        action.triggered.connect(self.tool.textWdg.editorInstance.strokeText)

        action = QAction("Change Selection Color", self)
        menu.addAction(action)
        action.triggered.connect(lambda checked: self.setTextColor())

        action = QAction("Font", self)
        menu.addAction(action)
        action.triggered.connect(lambda checked: self.setSelectedTextFont())

        menuAlign = menu.addMenu("Text Alignment")
        alignGroup = QActionGroup(self)
        alignGroup.setExclusive(True)
        for align in ["Left", "Right", "Justify", "Center"]:
            action = QAction(align, self, checkable=True, checked=(align == "Left"))
            alignGroup.addAction(action)
            menuAlign.addAction(action)
            action.triggered.connect(lambda checked, a=align: self.setAlignment(a))

        self.contextMenu = menu

    def contextMenuEvent(self, event):
        if not self.contextMenu:
            self.initContextMenu()
        self.contextMenu.exec(event.globalPos())

    def setAlignment(self, alignment):
        document = self.document()
        if document:
            match alignment:
                case "Left":
                    document.setDefaultTextOption(QTextOption(Qt.AlignmentFlag.AlignLeft))
                case "Right":
                    document.setDefaultTextOption(QTextOption(Qt.AlignmentFlag.AlignRight))
                case "Justify":
                    document.setDefaultTextOption(QTextOption(Qt.AlignmentFlag.AlignJustify))
                case "Center":
                    document.setDefaultTextOption(QTextOption(Qt.AlignmentFlag.AlignCenter))

    def setSelectedTextFont(self):
        """
        Opens a font selection dialog and sets the selected font
        to the selected text.
        """
        cursor = self.textCursor()
        if not cursor.hasSelection():
            dlgWarn("Select text first", "No text selected")
            return
        fmt = cursor.charFormat()  # )QTextCharFormat()
        ok, font = QFontDialog.getFont(fmt.font(), self, "Select Font")
        if ok:
            img = self.targetLayer.parentImage
            font.setPointSize(font.pointSize() * img.resize_coeff(self.parent()))
            fmt.setFont(font)
            cursor.mergeCharFormat(fmt)  # apply to selected text
            #img = self.targetLayer.parentImage
            #self.syncSizeWithTool(img.resize_coeff(self.parent()))

    def setTextColor(self, color=QColor(), selected=True):
        """
        Sets the text color.
        If selected is True, applies to selected text only, else to all text.
        If color is invalid (default), opens a QColorDialog.
        Note: QPlainTextEdit has no native method setTextColor.

        :param color:
        :type  color: QColor
        :param selected:
        :type  selected: bool
        """
        if not color.isValid():
            color = self.parent().State['brush']['color']  # current brush color
            color = QbLUeColorDialog.getColor(initial=color, parent=self, title="Select Text Color")
        if not color.isValid():
            return

        if not selected:
            self.selectAll()
        cursor = self.textCursor()
        fmt = QTextCharFormat()
        fmt.setForeground(QBrush(color))
        cursor.mergeCharFormat(fmt)  # apply to selected text
        self.parent().brushUpdate(color=color)  # sync brush color with text color
        if not selected:
            cursor.clearSelection()

    def fragmentCount(self, block):
        """
        Counts the number of fragments contained in a QTextBlock instance.
        :param block:
        :type block: QTextBlock
        :return:
        :rtype: int
        """
        it = block.begin()  # iterator
        count = 0
        while not it.atEnd():
            count += 1
            it += 1
        return count

    def snapBlock(self, qp, block):
        """

        :param qp:
        :type qp:
        :param block:
        :type block:
        """
        if self.fragmentCount(block) > 1:
            raise ValueError
        cursor = QTextCursor(block)
        it = cursor.block().begin()
        fmt = it.fragment().charFormat()
        text = it.fragment().text()
        font = fmt.font()
        layout = block.layout()
        layoutbr = layout.boundingRect()
        qpbr = layoutbr.adjusted(0, QFontMetrics(font).ascent(), 0, QFontMetrics(font).ascent())
        path = QPainterPath(QPointF(0.0, 0.0))
        path.addText(qpbr.topLeft(), font, text)
        ##qp.drawPath(path)  # contour color : pen  fill color :brush
        qp.save()
        qp.setFont(font)
        ##qp.drawText(qpbr.topLeft(), text)  # color: brush
        qp.restore()

    def snap(self, qp):
        """
        Paints the text editor content on the given QPainter.

        :param qp:
        :type qp: QPainter
        """
        qp.save()
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        #qp.save()
        #qp.setPen(QPen(QColor(255, 0, 0,0)))
        offset = 0
        #qpoffset = offset + QFontMetrics(self.font()).ascent()
        block = self.document().firstBlock()
        while block.isValid():
            blockbr = self.blockBoundingRect(block)
            layout = block.layout()  # QTextLayout
            try:
                self.snapBlock(qp, block)
            except ValueError:
                pass


            #cursor = QTextCursor(block)
            #it = cursor.block().begin()
            #cursor.setPosition(it.fragment().position())
            #cursor.setPosition(it.fragment().position() + it.fragment().length(), QTextCursor.KeepAnchor)

            """
            while not it.atEnd():
                fmt = it.fragment().charFormat()
                font = fmt.font()
                qp.setFont(font)
                 #offset))
                #path.addText(QPointF(0, qpoffset), self.font(), text)
                path.addText(qpbr.topLeft(), font, it.fragment().text())
                it += 1
            """

            #ef =QGraphicsDropShadowEffect()
            #grpath = QGraphicsPathItem(path)
            #grpath.setGraphicsEffect(ef)
            #qp.drawPath(grpath)
            #qp.drawText(QPointF(0, qpoffset), text)  # color: brush



            layout.draw(qp, QPointF(0, offset))
            offset += blockbr.height()
            #qpoffset += blockbr.height()
            block = block.next()
        qp.restore()

    def snapOnLayerToRemovexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(self):
        """
        Strokes the text on the target layer.
        """
        p = QPainter(self.targetLayer.sourceImg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        r = self.tool.img.resize_coeff(self.parent())

        rectangle = QRect(0,0, self.width()/r, self.height()/r)
        #p.translate(self.tool.img.xOffset / r, self.tool.img.yOffset / r)
        p.translate(self.tool.textWdg.getRectRelImg().topLeft())
        p.scale(1.0 / r, 1.0 / r)
        self.snap(p)
        p.end()
        layer = self.targetLayer
        img = layer.parentImage
        layer.execute(l=layer, bRect=layer.rect())
        img.prLayer.update(bRect=layer.rect())
        self.parent().repaint()

    def strokeText(self):
        """
        Paint text on layer
        """
        layer = self.targetLayer

        p = QPainter(layer.sourceImg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        targetRect = self.tool.textWdg.getRectRelImg()
        p.save()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        # p.fillRect(self.targetLayer.sourceImg.rect(), QColor(0, 0, 0, 0))
        p.fillRect(targetRect, QColor(0, 0, 0, 0))
        p.restore()
        # p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        # p.setOpacity(0.5)
        p.setPen(Qt.red)  # default text color
        p.setBrush(Qt.yellow)  # ????

        p.translate(targetRect.topLeft())
        r = self.tool.img.resize_coeff(self.parent())
        p.scale(1.0 / r, 1.0 / r)
        self.snap(p)
        p.end()

        self.tool.layer.tool.textWdg.savedSourceImg = layer.sourceImg.copy(layer.sourceImg.rect())

        img = layer.parentImage
        layer.execute(l=layer, bRect=layer.rect())
        img.prLayer.update(bRect=layer.rect())
        self.parent().repaint()


class transparentWidget(QWidget):
    """
    Simple transparent widget displaying some text.
    """

    def __init__(self, tool=None, targetLayer=None, parent=None):
        """
        Inits the transparent widget.

        :param parent:
        :type  parent: QWidget
        """
        super().__init__(parent)
        #self.setAttribute(Qt.WA_NoMousePropagation)  # propagation to parent is needed for image zooming an panning
        #self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        #self.setAttribute(Qt.WA_NoSystemBackground)
        #self.setAttribute(Qt.WA_TranslucentBackground)
        #self.setAttribute(Qt.WA_PaintOnScreen)
        #Pl = self.palette()
        #Pl.setBrush(QPalette.Window, QBrush(QColor(0, 0, 0, 0)))
        #self.setPalette(Pl)
        #self.setMouseTracking(True)
        self.tool = tool
        self.text = 'processing...  '
        self.paintText = False

        self.setWhatsThis(
            """
            Right click to open the editor
            """
        )
        self.setToolTip(
            """
            Right click to open the editor
            """
        )
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.menu = QMenu(self)
        #exgroup = QActionGroup(self.menu)
        action = QAction("Show Editor", self)
        #exgroup.addAction(action)
        self.menu.addAction(action)
        action.triggered.connect(self.showEditor)

        #action = QAction("Hide Editable Text", self, triggered=self.hideEditor)
        #action.setCheckable(True)
        #self.menu.addAction(action)
        #exgroup.addAction(action)
        #action = QAction("Clear Text", self, checkable=False)
        #self.menu.addAction(action)  # triggered=self.clearText))
        #action.triggered.connect(self.clearText)

        #action = QAction("Stroke Text", self, checkable=False)
        #self.menu.addAction(action)
        #action.triggered.connect(self.strokeText)

        self.editorInstance = None
        self.savedSourceImg = None
        self.targetLayer = targetLayer
        #self.getEditorInstance().textChanged.connect(self.strokeText)

    def openFontDialogToremoveXXXXXXXXXXXXXXXXXXXXXXX(self):
        ok, font = QFontDialog.getFont(self.font(), self, "Select Font")
        if ok:
            #self.setFont(font)
            if self.editorInstance:
                self.editorInstance.setFont(font)
                self.editorInstance.iniFont = font
                self.editorInstance.iniFontInfo = QFontInfo(self.editorInstance.iniFont)
            #self.repaint()

    def clearTextToRemovexxxxxxxxxxxxxxxxxxxxxxx(self):
        self.text = ''
        self.repaint()

    def clearStrokedText(self):
        p = QPainter(self.targetLayer.sourceImg)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.fillRect(self.targetLayer.sourceImg.rect(), QColor(0, 0, 0, 0))
        p.end()
        layer = self.targetLayer
        img = layer.parentImage
        layer.execute(l=layer, bRect=layer.rect())
        img.prLayer.update(bRect=layer.rect())
        self.parent().repaint()

    def updateText(self):
        self.text = self.editorInstance.toPlainText()
        self.repaint()

    def showEditor(self):
        #self.paintText = True
        if not self.editorInstance:
            self.editorInstance = self.getEditorInstance()
            #self.editorInstance.setPlainText(self.text)
            #self.editorInstance.setTextColor(selected=False)
        self.editorInstance.show()
        self.tool.syncToolWithLayer()

    def hideEditortoremoveXXXXXXXXXXXXXXX(self):
        self.paintText = False
        #if self.editorInstance:
            #self.editorInstance.hide()
        self.repaint()

    def getEditorInstance(self):
        """
        Factory method returning a textEditor instance.

        :return:
        :rtype: blueTextEdit
        """
        if not self.editorInstance:
            self.editorInstance = blueTextEdit(tool=self.tool, parent=self.parent())
            self.editorInstance.textChanged.connect(lambda: self.updateText())
        return self.editorInstance

    def getRectRelImg(self):
        """
        Returns the rectangle of the widget,
        coordinates relative to the full size image.

        :return:
        :rtype: QRect
        """
        r = self.tool.img.resize_coeff(self.parent())
        p = QPointF(self.pos() - QPoint(self.tool.img.xOffset, self.tool.img.yOffset)) / r
        s = QSize(self.width() / r, self.height() / r)
        return QRect(p.toPoint(), s)

    def paintEvent(self, event):
        """
        Paint event handler.

        :param event:
        :type  event: QPaintEvent
        """
        p = QPainter(self)
        if self.editorInstance:
            if self.editorInstance.isVisible():
                color = Qt.GlobalColor.gray
            else:
                color = Qt.GlobalColor.red
        else:
            color = Qt.GlobalColor.red
        p.setPen(color)
        pen = p.pen()
        pen.setWidth(3)
        p.setPen(pen)
        p.drawRect(self.rect())
        #fontMetrics = QFontMetrics(p.font())  # widget font
        #self.resize(fontMetrics.size(0, self.text))
        r = self.tool.img.resize_coeff(self.parent())
        p.scale(r, r)
        rectangle = QRect(0,0, self.width()/r, self.height()/r)
        if self.paintText:
            p.drawText(rectangle, Qt.AlignmentFlag.AlignCenter, self.text)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            #self.menu.popup(event.globalPosition().toPoint())
            self.showEditor()
        event.ignore()  # propagate all events to parent

    def strokeTexttoremovexxxxxxxxxxxxxxxxxxxxxxxxxxxx(self):
        """
        Paint text on layer
        """
        layer = self.targetLayer

        p = QPainter(layer.sourceImg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        targetRect = self.getRectRelImg()
        p.save()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        #p.fillRect(self.targetLayer.sourceImg.rect(), QColor(0, 0, 0, 0))
        p.fillRect(targetRect, QColor(0, 0, 0, 0))
        p.restore()
        #p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        #p.setOpacity(0.5)
        p.setPen(Qt.red)  # default text color
        p.setBrush(Qt.yellow) #  ????

        p.translate(targetRect.topLeft())
        r = self.tool.img.resize_coeff(self.parent())
        p.scale(1.0 / r, 1.0 / r)
        self.editorInstance.snap(p)
        p.end()

        img = layer.parentImage
        layer.execute(l=layer, bRect=layer.rect())
        img.prLayer.update(bRect=layer.rect())
        self.parent().repaint()



"""
QTextCursor cursor(myTextEdit->textCursor());

// change block format (will set the yellow background)
QTextBlockFormat blockFormat = cursor.blockFormat();
blockFormat.setBackground(QColor("yellow"));
blockFormat.setNonBreakableLines(true);
blockFormat.setPageBreakPolicy(QTextFormat::PageBreak_AlwaysBefore);
cursor.setBlockFormat(blockFormat);

// change font for current block's fragments
for (QTextBlock::iterator it = cursor.block().begin(); !(it.atEnd()); ++it)
{
    QTextCharFormat charFormat = it.fragment().charFormat();
    charFormat.setFont(QFont("Times", 15, QFont::Bold));

    QTextCursor tempCursor = cursor;
    tempCursor.setPosition(it.fragment().position());
    tempCursor.setPosition(it.fragment().position() + it.fragment().length(), QTextCursor::KeepAnchor);
    tempCursor.setCharFormat(charFormat);

// For block management
QTextDocument *doc = new QTextDocument(this);
ui->textEdit->setDocument(doc);  // from QTextEdit created by the Designer
//-------------------------------------------------
// Locate the 1st block
QTextBlock block = doc->findBlockByNumber(0);

// Initiate a copy of cursor on the block
// Notice: it won't change any cursor behavior of the text editor, since it 
//         just another copy of cursor, and it's "invisible" from the editor.
QTextCursor cursor(block);

// Set background color
QTextBlockFormat blockFormat = cursor.blockFormat();
blockFormat.setBackground(QColor(Qt::yellow));
cursor.setBlockFormat(blockFormat);

// Set font
for (QTextBlock::iterator it = cursor.block().begin(); !(it.atEnd()); ++it)
{
    QTextCharFormat charFormat = it.fragment().charFormat();
    charFormat.setFont(QFont("Times", 15, QFont::Bold));

    QTextCursor tempCursor = cursor;
    tempCursor.setPosition(it.fragment().position());
    tempCursor.setPosition(it.fragment().position() + it.fragment().length(), QTextCursor::KeepAnchor);
    tempCursor.setCharFormat(charFormat);
    
QTextEdit myEdit;
QTextDocument* myDocument = new QTextDocument(&myEdit);
myEdit.setDocument(myDocument);
QTextCursor* myCursor = new QTextCursor(myDocument);

QTextBlockFormat format;
format.setBackground(Qt::red);
myCursor->setBlockFormat(format);

myCursor->insertText("the ");

format.setBackground(Qt::green);
myCursor->insertBlock(format);
myCursor->insertText("fish ");

format.setBackground(Qt::yellow);
myCursor->insertBlock(format);
myCursor->insertText("are ");

format.setBackground(Qt::red);
myCursor->insertBlock(format);
myCursor->insertText("coming!");

format.setBackground(Qt::green);
myCursor->insertBlock(format);
myCursor->insertText(QString(%1 blocks").arg(myDocument->blockCount()));
myEdit.show();

def handleSelectionChanged(self):
    cursor = self.edit.textCursor()
    print ("Selection start: %d end: %d" % 
           (cursor.selectionStart(), cursor.selectionEnd()))
           
void TextEditor::snap(QPixmap &map)
{
    QPainter painter(&map);

    int offset = 0;
    block = document()->firstBlock();

    while (block.isValid())
    {
        QRectF r = blockBoundingRect(block);
        QTextLayout *layout = block.layout();

        if (!block.isVisible())
        {
            offset += r.height();
            block = block.next();
            continue;
        }
        else
        {
            layout->draw(&painter, QPoint(0,offset));
        }

        offset += r.height();

        block = block.next();
    }
}

cursor.beginEditBlock();
for (QTextBlock block = startBlock; block != endBlock; block = block.next()) {
    cursor.setPosition(block.position());
    cursor.setPosition(block.position() + 2, QTextCursor::KeepAnchor);
    cursor.removeSelectedText();
}
cursor.endEditBlock();


 virtual void paintEvent(QPaintEvent *event) override
    {
        QPainter painter(this);
        painter.setRenderHint( QPainter::Antialiasing, true );
        painter.drawRect(rect());
        QFont font;

        font.setPointSize( 12 );
        font.setStyleStrategy( QFont::StyleStrategy::PreferAntialias );
        painter.setFont( font );

        QRect m_window = QRect(- width() / 2, - height() / 2, width(), height());
        painter.setWindow( m_window );
        QRect m_viewport = QRect(0, 0, width(), height());
        painter.setViewport( m_viewport );
        // rotate
        painter.rotate( 80 );
        //draw text
        QPainterPath glyphPath;
        glyphPath.addText( 0, 0, painter.font(), "TEXT" );
        painter.fillPath( glyphPath, painter.pen().color() );

    }
"""