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

from PySide2.QtGui import QImage, QTransform, QPolygonF
from PySide2.QtWidgets import QWidget, QToolButton
from PySide2.QtCore import Qt, QPoint, QObject, QRect, QPointF

from bLUeGui.dialog import dlgWarn


class baseHandle(QToolButton):
    """
    Base class for tool handles.
    Tools are used to perform interactive geometric transformations
    of an image.
    A tool is a collection of buttons, recorded in a dictionary.
    Each button is draggable with the mouse and holds a role attribute,
    defining the type of action executed when the button is moved.


    """
    def __init__(self, role='', tool=None, parent=None):
        super().__init__(parent=parent)
        self.role = role
        self.margin = 0.0
        # back link to the tool
        self.tool = tool
        self.setVisible(False)
        self.setGeometry(0, 0, 10, 10)
        self.setAutoFillBackground(True)
        self.setAutoRaise(True)


class croppingHandle(baseHandle):
    """
    Simple active button, draggable with the mouse.
    When moved, it updates the cropping margins of an image

    """
    def __init__(self, role='', tool=None, parent=None):
        """
        parent should be the widget showing the edited imImage.
        role is 'left', 'right', 'top', 'bottom',
        'topRight', 'topLeft', 'bottomRight', 'bottomLeft'
        @param role:
        @type role: str
        @param parent:
        @type parent: QWidget
        """
        super().__init__(role=role, tool=tool, parent=parent)
        self.setStyleSheet("QToolButton:hover {background-color:#00FF00} QToolButton {background-color:#555555}")

    def setPosition(self, p):
        """
        Updates button margins in response to a mouse move event
        @param p: mouse cursor position (relative to parent widget)
        @type p: QPoint

        """
        widg = self.parent()
        img = widg.img
        r = img.resize_coeff(widg)
        lMargin, rMargin, tMargin, bMargin = img.cropLeft, img.cropRight, img.cropTop, img.cropBottom
        # middle buttons
        if self.role == 'left':
            margin = (p.x() - img.xOffset + self.width()) / r
            if margin < 0 or margin >= img.width() - self.tool.btnDict['right'].margin:
                return
            self.margin = margin
            lMargin = margin
        elif self.role == 'right':
            margin = img.width() - (p.x() - img.xOffset) / r
            if margin < 0 or margin >= img.width() - self.tool.btnDict['left'].margin:
                return
            self.margin = margin
            rMargin = margin
        elif self.role == 'top':
            margin = (p.y() - img.yOffset + self.height()) / r
            if margin < 0 or margin >= img.height() - self.tool.btnDict['bottom'].margin:
                return
            self.margin = margin
            tMargin = margin
        elif self.role == 'bottom':
            margin = img.height() - (p.y() - img.yOffset) / r
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
        self.tool.crHeight = img.height() - int(img.cropTop) - int(img.cropBottom)  # self.tool.btnDict['top'].margin - self.tool.btnDict['bottom'].margin
        self.tool.crWidth = img.width() - int(img.cropLeft) - int(img.cropRight)  # self.tool.btnDict['left'].margin - self.tool.btnDict['right'].margin

    def mouseMoveEvent(self, event):
        img = self.parent().img
        pos = self.mapToParent(event.pos())
        oldPos = self.pos()
        if self.role in ['left', 'right']:
            self.setPosition(self.pos() + QPoint((pos - oldPos).x(), 0))
        elif self.role in ['top', 'bottom']:
            self.setPosition(self.pos() + QPoint(0, (pos - oldPos).y()))
        # vertex buttons
        else:
            self.setPosition(pos)
        self.tool.drawCropTool(self.parent().img)
        self.tool.crHeight = img.height() - int(img.cropTop) - int(img.cropBottom)  # self.tool.btnDict['top'].margin - self.tool.btnDict['bottom'].margin
        self.tool.crWidth = img.width() - int(img.cropLeft) - int(img.cropRight)  # self.tool.btnDict['left'].margin - self.tool.btnDict['right'].margin
        self.parent().updateStatus()
        self.parent().repaint()


class cropTool(QObject):
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

    def drawCropTool(self, img):
        """
        Draws the 8 crop buttons around the displayed image,
        with their current margins.
        @param img:
        @type img: QImage
        """
        r = self.parent().img.resize_coeff(self.parent())
        left = self.btnDict['left']
        top = self.btnDict['top']
        bottom = self.btnDict['bottom']
        right = self.btnDict['right']
        cRect = QRect(round(left.margin), round(top.margin), img.width() - round(right.margin + left.margin),
                      img.height() - round(bottom.margin + top.margin))
        p = cRect.topLeft() * r + QPoint(img.xOffset, img.yOffset)
        x, y = p.x(), p.y()
        w, h = cRect.width() * r, cRect.height() * r
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
        self.crWidth, self.crHeight = img.width() - int(img.cropLeft) - int(img.cropRight), img.height() - int(img.cropTop) - int(img.cropBottom)


class rotatingHandle(baseHandle):
    """
    Active button for interactive (geometric) transformations
    """

    def __init__(self, role=None, tool=None, pos=QPoint(0, 0), parent=None):
        """

        @param role:
        @type role: str
        @param tool:
        @type tool: rotatingTool
        @param pos:
        @type pos: QPoint
        @param parent: parent widget
        @type parent: QWidget
        """
        super().__init__(role=role, tool=tool, parent=parent)
        # set coordinates (relative to the full resolution image)
        self.posRelImg_ori = pos  # starting pos, never modified if a transformation is in progress (cf. rotatingTool.getSourceQuad)
        self.posRelImg = pos  # current pos (cf. rotatingTool.getTargetQuad)
        self.posRelImg_frozen = pos  # starting pos for the current type of transformation
        self.setStyleSheet("QToolButton:hover {background-color:#00FF00} QToolButton {background-color:#AA0000}")

    def mousePressEvent(self, event):
        """
        Mouse press event handler: sets the tool resizing
        coefficient for the current move.
        @param event:
        @type event:
        """
        widget = self.tool.parent()
        # get the current resizing coeff.
        self.tool.resizingCoeff = self.tool.layer.parentImage.resize_coeff(widget)

    def mouseMoveEvent(self, event):
        """
        Mouse move event handler
        @param event:
        @type event:
        """
        modifiers = event.modifiers()  # QApplication.keyboardModifiers()
        # mouse coordinates, relative to parent widget
        pos = self.mapToParent(event.pos())
        img = self.tool.layer.parentImage
        r = self.tool.resizingCoeff
        self.tool.targetQuad_old = self.tool.getTargetQuad()
        self.posRelImg = (pos - QPoint(img.xOffset, img.yOffset)) / r
        if modifiers == Qt.ControlModifier | Qt.AltModifier:
            if self.tool.isModified():
                dlgWarn("A transformation is in progress", "Reset first")
                return
            # update the new starting  position
            self.posRelImg_ori = self.posRelImg  # (pos - QPoint(img.xOffset, img.yOffset)) / r
            self.posRelImg_frozen = self.posRelImg
            self.tool.moveRotatingTool()
            self.tool.parent().repaint()
            return
        curimg = self.tool.layer.getCurrentImage()
        w, h = curimg.width(), curimg.height()
        s = w / self.tool.img.width()
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
            q = T.map(self.tool.getFrozenQuad())
            for i, role in enumerate(['topLeft', 'topRight', 'bottomRight', 'bottomLeft']):
                self.tool.btnDict[role].posRelImg = q.at(i)
        elif form.options['Translation']:
            # translation vector (coordinates are relative to the full size image)
            p = QPointF(self.posRelImg) - QPointF(self.posRelImg_frozen)
            T = QTransform()
            T.translate(p.x(), p.y())
            q = T.map(self.tool.getFrozenQuad())
            for i, role in enumerate(['topLeft', 'topRight', 'bottomRight', 'bottomLeft']):
                self.tool.btnDict[role].posRelImg = q.at(i)
        self.tool.moveRotatingTool()
        self.tool.modified = True
        self.tool.layer.applyToStack()
        self.parent().repaint()


class rotatingTool(QObject):
    """
    Provides interactive modifications of a base geometric transformation.
    When applied to an image ABCD, a transformation T
    acts in the image coordinate system, and the resulting image A'B'C'D' is
    translated by -(A'B'C'D').boundingRect().topLeft().
    A'B'C'D' is represented by a QPolygonF instance (cf. the methods getSourceQuad and getTargetQuad).
    Different types of transformations can be performed in a cumulative way.
    The tool buttons can be positioned anywhere in the image : this can be useful
    in particular for perspective correction.
    """

    def __init__(self, parent=None, layer=None, form=None):
        """
        Inits a rotatingTool instance and adds it to the parent widget
        @param parent: parent widget
        @type parent: QWidget
        @param layer: image layer
        @type layer: QLayer
        @param form: GUI form
        @type form: transform
        """
        self.modified = False
        self.layer = layer
        # self.form = form
        # dynamic attribute
        # if self.form is not None:
        # self.form.tool = self
        super().__init__(parent=parent)
        if self.layer is None:
            w, h = 1.0, 1.0
        else:
            self.layer.tool = self
            self.img = layer.parentImage
            self.layer.visibilityChanged.sig.connect(self.setVisible)
            w, h = self.img.width(), self.img.height()
        # init tool buttons. The parameter pos is relative to the full size image.
        rotatingButtonLeft = rotatingHandle(role='topLeft', tool=self, pos=QPoint(0, 0), parent=parent)
        rotatingButtonRight = rotatingHandle(role='topRight', tool=self, pos=QPoint(w, 0), parent=parent)
        rotatingButtonTop = rotatingHandle(role='bottomLeft', tool=self, pos=QPoint(0, h), parent=parent)
        rotatingButtonBottom = rotatingHandle(role='bottomRight', tool=self, pos=QPoint(w, h), parent=parent)
        # init button dictionary
        btnList = [rotatingButtonLeft, rotatingButtonRight, rotatingButtonTop, rotatingButtonBottom]
        self.btnDict = {btn.role: btn for btn in btnList}
        if self.layer is not None:
            self.moveRotatingTool()

    def showTool(self):
        for btn in self.btnDict.values():
            btn.show()

    def hideTool(self):
        for btn in self.btnDict.values():
            btn.hide()

    def setVisible(self, value):
        for btn in self.btnDict.values():
            btn.setVisible(value)

    def getForm(self):
        if self.layer is not None:
            return self.layer.getGraphicsForm()
        return None

    def setBaseTransform(self):
        """
        Save the current quad as starting quad
        for the current type of transformation
        """
        q = self.getTargetQuad()
        for i, role in enumerate(['topLeft', 'topRight', 'bottomRight', 'bottomLeft']):
            self.btnDict[role].posRelImg_frozen = q.at(i)

    def isModified(self):
        return self.modified

    def getTargetQuad(self):
        """
        Returns the current quad, as defined by the 4 buttons.
        Coordinates are relative to the full size image
        @return:
        @rtype: QPolygonF
        """
        poly = QPolygonF()
        for role in ['topLeft', 'topRight', 'bottomRight', 'bottomLeft']:
            poly.append(self.btnDict[role].posRelImg)
        return poly

    def getSourceQuad(self):
        """
        Returns the starting quad for the transformation in progress
        @return:
        @rtype: QPolygonF
        """
        poly = QPolygonF()
        for role in ['topLeft', 'topRight', 'bottomRight', 'bottomLeft']:
            poly.append(self.btnDict[role].posRelImg_ori)
        return poly

    def getFrozenQuad(self):
        """
        Returns the starting quad for the current type of transformation
        @return:
        @rtype: QPolygonF
        """
        poly = QPolygonF()
        for role in ['topLeft', 'topRight', 'bottomRight', 'bottomLeft']:
            poly.append(self.btnDict[role].posRelImg_frozen)
        return poly

    def restore(self):
        for i, role in enumerate(['topLeft', 'topRight', 'bottomRight', 'bottomLeft']):  # order matters
            self.btnDict[role].posRelImg = self.targetQuad_old.at(i)
        self.moveRotatingTool()

    def moveRotatingTool(self):
        """
        Moves the tool buttons to the vertices of  the displayed image.
        Should be called every time that posRelImg is changed or the
        image zooming coeff. or position in widget
        are modified (cf. blue.mouseEvent and blue.wheelEvent)
        """
        # get parent widget
        parent = self.parent()
        r = parent.img.resize_coeff(parent)
        topLeft = self.btnDict['topLeft']
        topRight = self.btnDict['topRight']
        bottomLeft = self.btnDict['bottomLeft']
        bottomRight = self.btnDict['bottomRight']
        # move buttons : coordinates are relative to parent widget
        p = QPoint(self.img.xOffset, self.img.yOffset)
        x, y = p.x(), p.y()
        bottomLeft.move(x + bottomLeft.posRelImg.x() * r, y - bottomLeft.height() + bottomLeft.posRelImg.y() * r)
        bottomRight.move(x - bottomRight.width() + bottomRight.posRelImg.x() * r,
                         y - bottomRight.height() + bottomRight.posRelImg.y() * r)
        topLeft.move(x + topLeft.posRelImg.x() * r, y + topLeft.posRelImg.y() * r)
        topRight.move(x - topRight.width() + topRight.posRelImg.x() * r, y + topRight.posRelImg.y() * r)

    def resetTrans(self):
        self.modified = False
        w, h = self.img.width(), self.img.height()
        for role, pos in zip(['topLeft', 'topRight', 'bottomRight', 'bottomLeft'],
                             [QPoint(0, 0), QPoint(w, 0), QPoint(w, h), QPoint(0, h)]):
            self.btnDict[role].posRelImg = pos
            self.btnDict[role].posRelImg_ori = pos
            self.btnDict[role].posRelImg_frozen = pos
        self.moveRotatingTool()
        # self.frozenQuad = self.getTargetQuad()
        self.layer.applyToStack()
        self.parent().repaint()
