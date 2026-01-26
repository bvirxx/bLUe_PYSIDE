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

from PySide6.QtGui import QImage, QTransform, QPolygonF, QPainter, QFontMetrics, QColor, QBrush, QAction, \
    QActionGroup, QTextCursor, QTextCharFormat, QFontInfo, QFont, QPainterPath, QTextOption, QPen
from PySide6.QtWidgets import QWidget, QToolButton, QPlainTextEdit, QMenu, QFontDialog
from PySide6.QtCore import Qt, QPoint, QObject, QPointF, QRectF, QRect, QSize

from bLUeGui.dialog import dlgWarn
from bLUeGui.logginit import logger
from bLUeGui.memory import weakProxy
from bLUeTop.imLabel import imageLabel
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
        if type(parent) is not imageLabel:
            raise TypeError("baseHandle parent must be an imageLabel")
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
        self.posRelImg = pos  # control point : corner position relative to full size image
        self.setStyleSheet('QToolButton:hover {background-color:#00FF00} QToolButton {background-color:#555555}')

    def mouseMoveEvent(self, e):
        """
        moves button and updates posRelImg.
        :param e:
        :type  e:
        """
        # skip hover events and programmatic moves
        if e.buttons() == Qt.MouseButton.NoButton:
            return
        img = self.parent().img
        r = img.resize_coeff( self.parent())
        posreltowdg = self.mapToParent(e.position())
        self.move(posreltowdg.toPoint())
        self.posRelImg = (posreltowdg - QPointF(img.xOffset, img.yOffset)) / r  # control point is topleft corner
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
        if not layer:
            return
        self.layer = layer
        self.layer.tool = self  # add tool to layer
        self.textWdg = textAreaWidget(tool=self, targetLayer=layer, parent=parent)
        self.img = layer.parentImage
        # rotation/translation tool for text
        self.rottool = rotatingKbTool(parent=parent)
        self.rottool.addTool(layer, setattr=False)  # don't oerwrite layer.tool
        self.rottool.showTool()

        self.layer.visibilityChanged.sig.connect(self.setVisible)

        rect = self.img.rect().toRectF()
        d = min(rect.width(), rect.height()) * 0.1
        self.addMarker(rect.topLeft() + QPointF(d, d), role='topleft') # position relative to full size image
        self.addMarker(rect.bottomRight() - QPointF(d, d), role='bottomright')
        self.syncToolWithLayer()
        self.textWdg.show()

    @property
    def editorInstance(self):
        return self.textWdg.editorInstance

    def showTool(self):
        super().showTool()
        self.rottool.showTool()
        self.textWdg.show()

    def hideTool(self):
        super().hideTool()
        self.rottool.hideTool()
        if self.editorInstance:
            self.editorInstance.hide()
        self.textWdg.hide()

    def setVisible(self, value):
        super().setVisible(value)
        if self.editorInstance:
            self.editorInstance.setVisible(value)
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

    def syncToolWithLayer(self, zooming=False, syncsourceimg=True):
        """

        :param zooming:
        :type zooming: bool
        """
        img = self.layer.parentImage
        r = img.resize_coeff(self.parent())
        for btn in self.btnDict.values():
            posRelToParent = QPoint(img.xOffset, img.yOffset) + (btn.posRelImg * r).toPoint()
            match btn.role:
                case 'topleft':
                    # control point is rightbottom button corner
                    btn.move(posRelToParent - QPoint(btn.width(), btn.height()))
                    self.textWdg.move(posRelToParent) # + QPoint(btn.width(), btn.height()))
                case 'bottomright':
                    # control point is topleft button corner
                    btn.move(posRelToParent)
                    s = btn.pos() - self.btnDict['topleft'].pos() - QPoint(btn.width(), btn.height())
                    self.textWdg.setFixedSize(s.x() , s.y())

        if self.textWdg.editorInstance:
            self.textWdg.editorInstance.syncWithTool(zooming=zooming)

        if self.rottool:
            self.rottool.syncToolWithLayer(syncsourceimg=(syncsourceimg and not self.editorInstance.isVisible()))

class croppingHandle(baseHandle):
    """
    Active button, draggable with the mouse.
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
        # skip hover events and programmatic moves
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
    Updates cropping margins of an image.
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
        self.posRelImg = pos  # current pos
        self.posRelImg_frozen = pos  # saved pos
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
            v = self.posRelImg - center
            v0 = self.posRelImg_frozen - center
            # get the angle of rotation
            delta = (np.arctan2(v.y(), v.x()) - np.arctan2(v0.y(), v0.x())) * 180.0 / np.pi
            # get the rotation, centered on center
            deltaT.translate(center.x(), center.y()).rotate(delta).translate(-center.x(), -center.y())
        # Alt: translation
        elif modifiers == Qt.KeyboardModifier.AltModifier:
            # translation vector (coordinates are relative to the full size image)
            deltaV = self.posRelImg - self.posRelImg_frozen
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
    Provides interactive modifications of a base geometric transformation.
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
        w, h = 1.0, 1.0

        # init tool buttons. The parameter pos is relative to the full size image.
        rotatingButtonLeft = self.rotatingHandleType(role='topLeft', tool=self, pos=QPointF(0, 0), parent=parent)
        rotatingButtonRight = self.rotatingHandleType(role='topRight', tool=self, pos=QPointF(w, 0), parent=parent)
        rotatingButtonTop = self.rotatingHandleType(role='bottomLeft', tool=self, pos=QPointF(0, h), parent=parent)
        rotatingButtonBottom = self.rotatingHandleType(role='bottomRight', tool=self, pos=QPointF(w, h), parent=parent)
        # init button dictionary
        btnList = [rotatingButtonLeft, rotatingButtonRight, rotatingButtonTop, rotatingButtonBottom]
        self.btnDict = {btn.role: btn for btn in btnList}

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
        self.syncToolWithLayer()

    def  syncToolWithLayer(self, syncsourceimg=True, **kwargs):
        """
        Moves the tool buttons to the vertices of  the displayed image.
        Should be called every time that posRelImg is changed or the
        image zooming coeff. or position in widget
        are modified.
        """
        parent = self.parent()
        r = parent.img.resize_coeff(parent)
        # move buttons : coordinates are relative to parent widget
        x, y = self.img.xOffset, self.img.yOffset
        for btn in self.btnDict.values():
            match btn.role:
                case 'bottomleft': # control point is bottomleft
                    btn.move(x + btn.posRelImg.x() * r, y - btn.height() + btn.posRelImg.y() * r)
                case 'bottomright': # control point is bottomright
                    btn.move(x - btn.width() + btn.posRelImg.x() * r, y - btn.height() + btn.posRelImg.y() * r)
                case 'topleft': # control point is topleft
                    btn.move(x + btn.posRelImg.x() * r, y + btn.posRelImg.y() * r)
                case 'topright': # control point is topright
                    btn.move(x - btn.width() + btn.posRelImg.x() * r, y + btn.posRelImg.y() * r)

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

    def  syncToolWithLayer(self, syncsourceimg=True, **kwargs):  # moveRotatingTool(self, syncsourceimg=True):
        """
        Moves the tool buttons to their new positions, and updates
        the transformed image.
        """
        super().syncToolWithLayer(syncsourceimg=syncsourceimg, **kwargs)

        if syncsourceimg:
            self.layer.sourceImg  = self.transformedSourceImg()

            self.layer.execute(l=self.layer)  # , bRect=layer.uRect)
            self.layer.parentImage.prLayer.update()  # bRect=layer.uRect)
            self.parent().repaint()

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
        q1Full, q2Full = self.getOriginQuad(), self.getTargetQuad()
        T = QTransform()
        res = QTransform.quadToQuad(q1Full, q2Full, T)
        if not res:
            logger.warning('applyTransform : no possible transformation')
        # get the bounding rect of the transformed image (in the full size image coordinate system)
        rectTrans = T.map(self.layer.rect()).boundingRect()
        # apply the transformation and re-translate the transformed image.
        # The resulting transformation is T and NOT that given by QImage.trueMatrix()
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
        :type parent: QWidget
        """
        super().__init__( parent=parent)
        self.tool = tool
        self.targetLayer = self.tool.layer
        #self.setWindowFlags(Qt.Window)  # | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_PaintOnScreen)
        self.setAttribute(Qt.WA_NoMousePropagation)  # otherwise, right click propagates to parent, I don't know why !
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #pl = self.palette()
        #textColor = pl.color(QPalette.WindowText )
        #pl.setBrush(QPalette.Window, QBrush(QColor(255, 0, 255, 255)))
        #pl.setColor(QPalette.WindowText, QColor(0,255,0))
        #self.setPalette(pl)
        #self.setPalette(Pl)
        #self.setMouseTracking(True)
        #self.setStyleSheet("QPlainTextEdit {background-color: white; color: red;}")
        self.iniFont = QFont('Arial', 40)
        self.setFont(self.iniFont)
        self.iniFontInfo = QFontInfo(self.iniFont)
        self.setPlainText("Enter text here")
        self.setSelectedTextFont(selectall=True, font=self.iniFont)
        self.setTextColor(QColor(255, 255, 255), selected=False)
        self.setAlignment('Center')
        self.current_resize_coeff = 1.0
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.contextMenu = None
        self.syncWithTool(zooming=True)  #####################
        self.modificationChanged.connect(self.onModificationChanged)
        #self.textChanged.connect(lambda b=True : self.onModificationChanged(b))
        self.document().setModified(False)

    def onModificationChanged(self, changed):
        """
        Slot called when  the modification state changes.

        :param changed: True if the document has been modified
        :type changed: bool
        """
        if not changed:
            return
        # check if text width exceeds editor width
        w, h = self.maxBlockWidth()
        if w > self.width() or h > self.height():
            dlgWarn("on modif Text size exceeds editor size", "Please reduce font size or remove some text")
            self.undo()
        self.document().setModified(False) # reset modification state

    def syncSizeWithTool(self, font_coeff):
        """
        Zooms all editor font sizes.

        :param font_coeff:
        :type font_coeff: Float
        """
        #self.document().setModified(True)
        try:
            self.modificationChanged.disconnect()
        except RuntimeError:
            pass
        margin = self.document().documentMargin()
        self.document().setDocumentMargin(margin * font_coeff)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        endPos = cursor.position()
        cursor.movePosition(QTextCursor.Start)
        while cursor.position() < endPos:
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor) # select one symbol
            fmt = cursor.charFormat()
            ps = fmt.fontPointSize()
            fmt.setFontPointSize(ps * font_coeff)
            cursor.mergeCharFormat(fmt)
            cursor.setPosition(cursor.position())  # move anchor to current position
            w, h = self.maxBlockWidth()
            if w > self.width() or h > self.height():
                if font_coeff > 1.0 and self.isVisible():
                    dlgWarn("synsize Text size exceeds editor size", "Please reduce font size or remove some text")
                    self.document().setModified(False)
                    break
        cursor.clearSelection()
        self.setTextCursor(cursor)  # update visible cursor
        self.document().setModified(False)
        self.modificationChanged.connect(self.onModificationChanged)

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
            match btn.role:
                case 'topleft':
                    # keep button bottom right corner fixed on image
                    btn.move(posRelToParent - QPoint(btn.width(), btn.height()))
                    self.move(posRelToParent)  # + QPoint(btn.width(), btn.height()))
                case 'bottomright':
                    # keep button top left corner fixed on image
                    btn.move(posRelToParent)
                    s = btn.pos() - self.tool.btnDict['topleft'].pos() - QPoint(btn.width(), btn.height())
                    shrink = (s.x() < self.width()) or (s.y() < self.height())
                    self.setFixedSize(s.x() , s.y())
        if zooming:
            self.syncSizeWithTool(font_coeff)

        w, h = self.maxBlockWidth()
        if w > self.width() or h > self.height():
            if shrink and self.isVisible():
                dlgWarn("sync Text size exceeds editor size", "Please reduce font size or remove some text")

        textwdgtopleft = self.tool.textWdg.getRectRelImg().topLeft()

        # self.targetLayer.textTransform = QTransform().translate(textwdgtopleft.x(), textwdgtopleft.y())

        center = self.tool.textWdg.getRectRelImg().center()
        T = self.targetLayer.textTransform
        a = np.atan2(T.m12(), T.m11())
        rot = QTransform().translate(-center.x(), -center.y()) * QTransform().rotateRadians(a) * QTransform().translate(center.x(), center.y())
        self.targetLayer.textTransform = QTransform().translate(textwdgtopleft.x(), textwdgtopleft.y()) * rot

    def maxBlockWidth(self):
        """
        Computes the maximum width of text blocks in the document.

        :return:
        :rtype: int
        """
        doc = self.document()
        block = doc.firstBlock()
        maxWidth = 0
        height = 0
        while block.isValid():
            layout = block.layout()
            br = layout.boundingRect()
            if br.width() > maxWidth:
                maxWidth = br.width()
            height += br.height()
            block = block.next()
        return int(maxWidth), int(height)

    def hide(self):
        """
        Hides the text editor and paint text on layer.
        """
        #self.drawText()
        for btn in self.tool.btnDict.values():
            btn.hide()
        rottool_visible = self.targetLayer.visible and self.targetLayer.isActiveLayer()
        self.tool.rottool.setVisible(rottool_visible)
        if rottool_visible:
            self.tool.rottool.syncToolWithLayer()
        super().hide()

    def show(self):
        for btn in self.tool.btnDict.values():
            btn.show()
        super().show()

    def initContextMenu(self):
        menu = self.createStandardContextMenu()
        action = QAction("Hide Editor", self)
        menu.addAction(action)
        action.triggered.connect(self.hide)

        action = QAction("Snap Text", self)
        menu.addAction(action)
        action.triggered.connect(self.tool.textWdg.editorInstance.drawText)

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
            action = QAction(align, self, checkable=True, checked=(align == "Center"))
            alignGroup.addAction(action)
            menuAlign.addAction(action)
            action.triggered.connect(lambda checked, a=align: self.setAlignment(a))

        self.contextMenu = menu

    def contextMenuEvent(self, event):
        if not self.contextMenu:
            self.initContextMenu()
        self.contextMenu.exec(event.globalPos())

    def setAlignment(self, alignment):
        """

        :param alignment:
        :type alignment: str
        """

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

        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

    def setSelectedTextFont(self, selectall=False, font=None):
        """
        Opens a font selection dialog and sets the selected font
        to the selected text.
        """
        if selectall:
            self.selectAll()
        cursor = self.textCursor()
        if not cursor.hasSelection():
            dlgWarn("Select text first", "No text selected")
            return
        fmt = cursor.charFormat()
        ok = True
        if not font:
            dlgfont = fmt.font()
            img = self.targetLayer.parentImage
            dlgfont.setPointSize(dlgfont.pointSize() / img.resize_coeff(self.parent()))
            ok, font = QFontDialog.getFont(dlgfont, self, "Select Font")
        if ok:
            fmt.setFont(font)
            cursor.mergeCharFormat(fmt)  # apply to selected text
            w, h = self.maxBlockWidth()
            if w > self.width() or h > self.height():
                dlgWarn("font Text size exceeds editor size", "Please reduce font size or remove some text")
                self.undo()
        if selectall:
            cursor.clearSelection()

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
            color = self.currentBrushDict['color']
            color = QbLUeColorDialog.getColor(initial=color, parent=self, title="Select Text Color")
        if not color.isValid():
            return

        if not selected:
            self.selectAll()
        cursor = self.textCursor()
        fmt = QTextCharFormat()
        fmt.setForeground(QBrush(color))
        cursor.mergeCharFormat(fmt)  # apply to selected text
        #self.parent().brushUpdate(color=color)  # sync brush color with text color
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

    def snapBlock(self, qp, offset, block):
        """
        Paints a single text block using the given QPainter.
        :param qp:
        :type qp: QPainter
        :param block:
        :type block: QTextBlock
        """

        layout = block.layout()
        layoutbr = layout.boundingRect()

        """
        layout.beginLayout()
        line = layout.createLine()
        if (line.isValid()):
            line.setLineWidth(500)  # Set a generous width
            line.setPosition(QPointF(0, 0))
        layout.endLayout()
        layoutbr = layout.boundingRect()

        runs = layout.glyphRuns()  # to ensure layout is built
        fmt = layout.glyphRuns()[0].format()
        text = layout.glyphRuns()[0].text()
        font = fmt.font()
        """
        itemlist = []
        cursor = QTextCursor(block)
        it = cursor.block().begin()
        margin = self.document().documentMargin()
        current = margin # self.document().documentMargin() #10 #0
        #offset += margin
        maxdescent = 0
        while not it.atEnd():
            fmt = it.fragment().charFormat()
            text = it.fragment().text()
            font = fmt.font()
            brush = fmt.foreground()
            metrics = QFontMetrics(font)
            #textrect = metrics.boundingRect(text)
            #qpbr = layoutbr.adjusted(0, metrics.ascent(), 0, metrics.ascent())
            qpbr = layoutbr  #.adjusted(0, 0, 0, -metrics.descent())
            #path.addText(current + qpbr.center().x() - textrect.width() / 2, qpbr.bottom() + offset, font, text)
            #path.addText(current, qpbr.bottom() + offset, font, text)
            path = QPainterPath(QPointF(0,0))  #(QPointF(current, qpbr.bottom() + offset))
            path.addText(current, qpbr.bottom() + offset, font, text)
            itemlist.append((path, brush))
            current += metrics.horizontalAdvance(text)
            if metrics.descent() > maxdescent:
                maxdescent = metrics.descent()
            it += 1

        totalwidth = current
        # align = block.blockFormat().alignment()  # does not work for QPlainTextEdit
        option = self.document().defaultTextOption()
        align = option.alignment()
        match align:
            case Qt.AlignmentFlag.AlignCenter:
                centerOffset = (layoutbr.width() - totalwidth) / 2
            case Qt.AlignmentFlag.AlignRight:
                centerOffset = layoutbr.width() - totalwidth
            case _:
                centerOffset = 0
        T = QTransform()
        #T.translate(centerOffset+1.5, offset - maxdescent)  #  +1 to avoid antialiasing artifacts on left edge
        T.translate(centerOffset + 1.5, - maxdescent)  # +1 to avoid antialiasing artifacts on left edge

        qp.save()

        for item in itemlist:
            path = T.map(item[0])  # map path
            if self.targetLayer.getGraphicsForm().listWidget1.options['Font Color']:  # Brush Fill
                qp.setBrush(item[1].color())
                qp.setPen(item[1].color())
            elif self.targetLayer.getGraphicsForm().listWidget1.options['Brush Fill']:
                color = self.currentBrushDict['color']
                color.setAlphaF(self.currentBrushDict['opacity'])  #parent().State['brush']['opacity'])
                brush = QBrush(color)
                qp.setBrush(brush)
                pen = QPen(item[1].color())
                #pen.setWidth(4)
                qp.setPen(pen)

            """
            layout = block.layout()
            layoutbr = layout.boundingRect()
            metrics = QFontMetrics(font)
            textrect = metrics.boundingRect(text)
            qpbr = layoutbr.adjusted(0, metrics.ascent(), 0, metrics.ascent())
            """
            qp.drawPath(path)  # contour color : pen  fill color :brush
        #qp.save()
        #qp.setFont(font)
        #qp.drawText(qpbr.topLeft(), text)  # color: brush
        qp.restore()

    def snap(self, qp):
        """
        Paints the text editor content on the given QPainter.

        :param qp:
        :type qp: QPainter
        """
        qp.save()
        #qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        #qp.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        offset = 1.5 # self.document().documentMargin()  +1  # to avoid antialiasing artifacts on top edge
        #qpoffset = offset + QFontMetrics(self.font()).ascent()
        block = self.document().firstBlock()
        while block.isValid():
            # paint block
            blockbr = self.blockBoundingRect(block)
            layout = block.layout()  # QTextLayout
            fmt = block.blockFormat()
            #fmt.setAlignment(Qt.AlignmentFlag.AlignCenter)
            try:
                self.snapBlock(qp, offset, block)
            except ValueError:
                pass

            #layout.draw(qp, QPointF(0, offset))
            offset += blockbr.height() #/ 2  # because of newline
            #qpoffset += blockbr.height()
            block = block.next()
        qp.restore()

    def drawText(self):
        """
        Paint text on target layer
        """
        layer = self.targetLayer

        qp = QPainter(layer.sourceImg)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        targetRect = self.tool.textWdg.getRectRelImg()
        qp.save()
        qp.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        # clear target
        qp.fillRect(self.targetLayer.sourceImg.rect(), QColor(0, 0, 0, 0))
        #qp.fillRect(targetRect, QColor(0, 0, 0, 0))
        qp.restore()

        # p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        # p.setOpacity(0.5)
        #qp.setPen(Qt.red)  # default text color
        #qp.setBrush(QColor(0,0,0,0)) #(Qt.yellow)  # default text fill color
        #qp.translate(targetRect.topLeft())

        r = self.tool.img.resize_coeff(self.parent())

        textwdgtopleft = self.tool.textWdg.getRectRelImg().topLeft()

        #self.targetLayer.textTransform = QTransform().translate(textwdgtopleft.x(), textwdgtopleft.y())
        #self.targetLayer.textTransform.translate(-self.targetLayer.textTransform.dx(), -self.targetLayer.textTransform.dy())
        if self.targetLayer.textTransform.isIdentity():
            self.targetLayer.textTransform.translate(textwdgtopleft.x(), textwdgtopleft.y())

        qp.scale(1.0 / r, 1.0 / r)  # now, qp uses editor (widget)  coordinates

        # paint text
        self.snap(qp)

        qp.end()

        # save the source image with unmodified text (for further transformations)
        self.tool.layer.tool.textWdg.savedSourceImg = layer.sourceImg.copy(layer.sourceImg.rect())

        # update layer stack
        img = layer.parentImage
        layer.execute(l=layer, bRect=layer.rect())
        img.prLayer.update(bRect=layer.rect())
        self.parent().repaint()

    @property
    def currentBrushDict(self):
        """
        Returns the current brush settings from parent widget (imageLabel).
        :return:
        :rtype: dict
        """
        return self.parent().State['brush']

class textAreaWidget(QWidget):
    """
    Text area widget with transparent background.
    1) Draws a rectangle around the text area
    2) On right click, opens a text editor.
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
        self.tool = tool
        self.text = 'processing...  '
        self.paintText = False
        self.State = {'ix': 0, 'iy': 0}

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
        action = QAction("Show Editor", self)
        self.menu.addAction(action)
        action.triggered.connect(self.showEditor)
        self.__editorInstance = None
        self.savedSourceImg = None
        self.targetLayer = targetLayer

    def clearStrokedText(self):
        p = QPainter(self.targetLayer.sourceImg)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        #p.fillRect(self.targetLayer.sourceImg.rect(), QColor(0, 0, 0, 0))
        # clear text area
        p.fillRect(self.getRectRelImg(), QColor(0, 0, 0, 0))
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
        #if not self.editorInstance:
            #self.editorInstance = self.getEditorInstance()
        self.clearStrokedText()
        self.tool.rottool.hideTool()
        self.editorInstance.show()
        self.tool.syncToolWithLayer(syncsourceimg=False)

    @property
    def editorInstance(self):
        """
        Factory property returning a unique (for each text area instance) textEditor instance.

        :return:
        :rtype: blueTextEdit
        """
        if not self.__editorInstance:
            self.__editorInstance = blueTextEdit(tool=self.tool, parent=self.parent())
            self.__editorInstance.textChanged.connect(lambda: self.updateText())
        return self.__editorInstance

    def getRectRelImg(self):
        """
        Returns the bounding rectangle of the widget,
        coordinates are relative to the full size image.

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
        """
        r = self.tool.img.resize_coeff(self.parent())
        p.scale(r, r)
        rectangle = QRect(0,0, self.width()/r, self.height()/r)
        if self.paintText:
            p.drawText(rectangle, Qt.AlignmentFlag.AlignCenter, self.text)
        """

    def mousePressEvent(self, event):
        globalpos = event.globalPosition().toPoint()
        self.State['ix'] = globalpos.x()
        self.State['iy'] = globalpos.y()
        if event.button() == Qt.MouseButton.RightButton:
            self.showEditor()
        event.accept()

    def mouseMoveEvent(self, event):
        globalpos = event.globalPosition().toPoint()
        delta = globalpos - QPoint(self.State['ix'], self.State['iy'])

        img = self.parent().img
        r = img.resize_coeff(self.parent())
        center = (self.rect().center() + self.pos() - QPoint(img.xOffset, img.yOffset)).toPointF() / r  # image coordinates
        center = center.toPoint()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # rotation
            globalcenter = self.mapToGlobal(self.rect().center())
            angle1 = np.atan2(self.State['iy'] - globalcenter.y(), self.State['ix'] - globalcenter.x())
            angle2 = np.atan2(globalpos.y() - globalcenter.y(), globalpos.x() - globalcenter.x())
            # rotation around center
            rot = QTransform().translate(-center.x(), -center.y()) * QTransform().rotateRadians(angle2 - angle1) * QTransform().translate(center.x(), center.y())
            self.targetLayer.textTransform = self.targetLayer.textTransform * rot # rot must be right = last applied
            #self.targetLayer.textTransform.translate(center.x(), center.y()).rotateRadians(angle2 - angle1).translate(-center.x(), -center.y())

        else:
            # translation
            self.move(self.pos() + delta)
            delta = delta.toPointF() / r  # image coordinates
            #a = np.acos(self.targetLayer.textTransform.m11())
            #center = (self.rect().center() - QPoint(img.xOffset, img.yOffset)).toPointF() / r
            #center = center.toPoint()
            #dx, dy = self.targetLayer.textTransform.dx(), self.targetLayer.textTransform.dy()
            #T = QTransform().translate(delta.x()+dx, delta.y()+dy)
            #T.translate(center.x(), center.y()).rotateRadians(a).translate(-center.x(), -center.y())
            #T.rotateRadians(a)
            self.targetLayer.textTransform = self.targetLayer.textTransform * QTransform().translate(delta.x(), delta.y())
            #self.targetLayer.textTransform = QTransform().translate(delta.x(), delta.y()) * self.targetLayer.textTransform  #test ???
            #T = self.targetLayer.textTransform.translate(delta.x(), delta.y())
            #self.targetLayer.textTransform = T

        self.State['ix'] = globalpos.x()
        self.State['iy'] = globalpos.y()

        self.targetLayer.execute(l=self.tool.layer)  # , bRect=layer.uRect)
        self.parent().img.prLayer.update()  # bRect=layer.uRect)
        self.parent().repaint()

        event.accept()

    def mouseReleaseEvent(self, event):
        self.syncTool()
        event.accept()

    def syncTool(self):
        widget = self.parent()
        img = widget.img
        r = img.resize_coeff(widget)
        for btn in self.tool.btnDict.values():
            if btn.role == 'topleft': # control point is button bottomright corner
                btn.move(self.pos() - QPoint(btn.width(), btn.height()))
                btn.posRelImg = (btn.pos().toPointF() + QPointF(btn.width(), btn.height()) - QPointF(img.xOffset, img.yOffset)) / r
            elif btn.role == 'bottomright': # control point is button topleft corner
                btn.move(self.pos() + QPoint(self.width(), self.height()))
                btn.posRelImg = (btn.pos().toPointF() - QPointF(img.xOffset, img.yOffset)) / r
        self.editorInstance.move(self.pos())
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