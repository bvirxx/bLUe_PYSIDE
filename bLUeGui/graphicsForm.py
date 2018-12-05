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
import weakref

from PySide2 import QtCore
from PySide2.QtCore import QPoint
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QSizePolicy, QGraphicsPathItem, QWidget
from PySide2.QtGui import QColor, QPen, QPainterPath
from PySide2.QtCore import Qt

#################################################
# All layer control graphic forms must inherit
# from either baseForm or graphicsCurveForm below.
#################################################
from bLUeGui.memory import weakProxy


class baseForm(QWidget):
    """
    Base class for non graphic (i.e. without scene) forms
    """
    dataChanged = QtCore.Signal()

    def __init__(self, parent=None, layer=None):
        super().__init__(parent=parent)
        self.layer = layer
        # accept click focus (needed by whatsthis)
        self.setFocusPolicy(Qt.ClickFocus)
        # connect layerPicked signal
        if self.layer is not None:
            self.layer.colorPicked.sig.connect(self.colorPickedSlot)

    @property
    def layer(self):
        return self.__layer

    @layer.setter
    def layer(self, aLayer):
        if aLayer is None:
            self.__layer = None
        else:
            # link back to image layer :
            # using weak ref for back links
            if type(aLayer) in weakref.ProxyTypes:
                self.__layer = aLayer
            else:
                self.__layer = weakref.proxy(aLayer)

    def colorPickedSlot(self, x, y, modifiers):
        """
        Should be overridden in derived classes
        """
        pass

    def setDefaults(self):
        self.dataChanged.connect(self.updateLayer)


    def reset(self):
        self.setDefaults()
        self.dataChanged.emit(True)

    def updateLayer(self, *args, **kwargs):  # cacheInvalidate)
        """
        data changed event handler.
        Should be overridden
        by subclasses.
        @param cacheInvalidate:
        @type cacheInvalidate:
        """
        pass

    def updateHists(self):
        """
        Update the input histograms displayed
        on the form. Should be overridden
        by subclasses.
        """
        pass


class baseGraphicsForm(QGraphicsView):
    """
    Base class for graphics (with scene) forms
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def updateHists(self):
        """
        Update the input histograms displayed
        on the form. Should be overridden
        by subclasses.
        """
        pass


class graphicsCurveForm(baseGraphicsForm):  # QGraphicsView):  # TODO modified 30/11/18
    """
    Base class for interactive curve forms
    """

    @classmethod
    def drawPlotGrid(cls, axeSize, gradient=None):
        """
        Rerturns a QGraphicsPathItem initialized with
        a square grid.
        @param axeSize:
        @type axeSize:
        @param gradient:
        @type gradient
        @return:
        @rtype: QGraphicsPathItem
        """
        lineWidth = 1
        item = QGraphicsPathItem()
        if gradient is None:
            item.setPen(QPen(Qt.darkGray, lineWidth, Qt.DashLine))
        else:
            item.setPen(QPen(gradient, lineWidth, Qt.DashLine))
        qppath = QPainterPath()
        qppath.moveTo(QPoint(0, 0))
        qppath.lineTo(QPoint(axeSize, 0))
        qppath.lineTo(QPoint(axeSize, -axeSize))
        qppath.lineTo(QPoint(0, -axeSize))
        qppath.closeSubpath()
        qppath.lineTo(QPoint(axeSize, -axeSize))
        # draw grid
        for i in range(1, 5):
            a = (axeSize * i) / 4
            qppath.moveTo(a, -axeSize)
            qppath.lineTo(a, 0)
            qppath.moveTo(0, -a)
            qppath.lineTo(axeSize, -a)
        item.setPath(qppath)
        return item

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        self.layer = layer
        # additional inactive curve to draw (QPolyLineF or list of QPointF)
        self.baseCurve = None
        self.setMinimumSize(axeSize + 60, axeSize + 140)
        self.setAttribute(Qt.WA_DeleteOnClose)
        graphicsScene = QGraphicsScene()
        self.setScene(graphicsScene)
        # back links to image
        graphicsScene.targetImage = weakProxy(targetImage)
        graphicsScene.layer = weakProxy(layer)
        graphicsScene.bgColor = QColor(200, 200, 200)
        self.mainForm = mainForm
        graphicsScene.axeSize = axeSize
        # add axes and grid
        graphicsScene.defaultAxes = self.drawPlotGrid(axeSize)
        graphicsScene.addItem(graphicsScene.defaultAxes)
        # connect layer colorPicked signal
        self.scene().layer.colorPicked.sig.connect(self.colorPickedSlot)
        # default WhatsThis for interactive curves
        self.setWhatsThis(
            """
            The background histogram is the <i>input</i> histogram; it is refreshed only
            when the curve is reset.<br>
            <b>Drag control points</b> with the mouse.<br>
            <b>Add a control point</b> by clicking on the curve.<br>
            <b>Remove a control point</b> by clicking it.<br>
            <b>Zoom</b> with the mouse wheel.<br>
            <b>Set black, white and neutral points</b> in the image by clicking the corresponding pixels
            while pressing one of the following key combination (RGB and Lab curves only):<br>
            &nbsp;&nbsp;<b>Black Point</b> : Ctrl+Shift<br>
            &nbsp;&nbsp;<b>White Point</b> : Ctrl<br>
            &nbsp;&nbsp;<b>Grey Neutral Point (Lab only)</b></br> : Shift<br>
            <b>Caution</b> : Selecting a black, white or neutral point in an image is enabled only when
            the Color Chooser is closed.
            """)  # end setWhatsThis

    @property
    def layer(self):
        return self.__layer

    @layer.setter
    def layer(self, aLayer):
        if aLayer is None:
            self.__layer = None
        else:
            # link back to image layer :
            # using weak ref for back links
            if type(aLayer) in weakref.ProxyTypes:
                self.__layer = aLayer
            else:
                self.__layer = weakref.proxy(aLayer)

    @property
    def baseCurve(self):
        return self.__baseCurve

    @baseCurve.setter
    def baseCurve(self, points):
        self.__baseCurve = points

    def colorPickedSlot(self, x, y, modifiers):
        """
        Should be overridden in derived classes
        """
        pass

    def wheelEvent(self, e):
        """
        Overrides QGraphicsView wheelEvent
        Zoom the scene
        @param e:
        @type e:
        """
        # delta unit is 1/8 of degree
        # Most mice have a resolution of 15 degrees
        numSteps = 1 + e.delta() / 1200.0
        self.scale(numSteps, numSteps)

    def updateHists(self):
        """
        Update the input histograms displayed
        under the curves. Should be overridden
        by subclasses.
        """
        pass
