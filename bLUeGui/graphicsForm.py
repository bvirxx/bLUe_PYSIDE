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
from PySide2 import QtCore
from PySide2.QtCore import QPoint
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QSizePolicy, QGraphicsPathItem, QWidget, QVBoxLayout
from PySide2.QtGui import QColor, QPen, QPainterPath, QBrush
from PySide2.QtCore import Qt
from bLUeGui.memory import weakProxy
from bLUeTop.utils import stateAwareQDockWidget


class bottomWidget(QWidget):
    """
    ad hoc container to add non-zoomable
    buttons and options below a scene.
    """

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setMaximumSize(400, 150)
        self.setMinimumSize(200, 50)
        self.setObjectName('container')
        ss = """QWidget#container{background-color: black;}
                               QListWidget {font-size: 7pt;}
                               QListWidget::item{color: white;}
                               QListWidget::item:disabled {color: gray;}"""
        self.setStyleSheet(ss)


class abstractForm:
    """
    Base properties and methods
    for graphic forms.
    This container is designed for multiple
    inheritance only and should never be instantiated.
    """

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=200, layer=None, parent=None, mainForm=None):
        wdgt = cls(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.mainForm = weakProxy(mainForm)
        if layer is not None:
            wdgt.setWindowTitle(getattr(layer, 'name', 'noname'))
        return wdgt

    @property
    def layer(self):
        return self.__layer

    @layer.setter
    def layer(self, aLayer):
        self.__layer = weakProxy(aLayer)

    @property
    def targetImage(self):
        return self.__targetImage

    @targetImage.setter
    def targetImage(self, aTargetImage):
        self.__targetImage = weakProxy(aTargetImage)

    def __del__(self):
        print('*********** %s' % type(self))

    def colorPickedSlot(self, x, y, modifiers):
        """
        A colorPicked signal is emitted when a mouse click
        occurs on the image under edition (cf. bLUe.mouseEvent()).
        (x,y) coordinates are supposed to be relative to the full size image.
        Should be overridden in subclasses.
        @param x:
        @type x: int
        @param y:
        @type y: int
        @param modifiers:
        @type modifiers: Qt.KeyboardModifiers

        """
        pass

    def setDefaults(self):
        """
        Set the initial state of the form.
        This is an outline that must
        be overridden in subclasses.
        """
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        # set default values here
        self.dataChanged.connect(self.updateLayer)

    def reset(self):
        """
        Reset the form and update
        """
        self.setDefaults()
        self.dataChanged.emit()

    def updateLayer(self):
        """
        data changed slot.
        Must be overridden in subclasses.
        """
        pass

    def updateHists(self):
        """
        Update the input histograms possibly displayed
        on the form. Should be overridden
        in subclasses.
        """
        pass


#################################################
# Base graphic forms.
# All graphic forms should inherit
# from baseForm, baseGraphicsForm or graphicsCurveForm below.
#################################################

class baseForm(QWidget, abstractForm):
    """
    Base class for all graphic forms.

    """
    # Form state changed signal
    # Subclasses may redefine it with a different signature.
    # In that case, they must override abstractForm.reset() accordingly.
    dataChanged = QtCore.Signal()

    def __init__(self, layer=None, targetImage=None, parent=None):
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # accept click focus (needed by whatsthis)
        self.setFocusPolicy(Qt.ClickFocus)
        # back link to image layer (weak ref)
        self.layer = layer  # property setter
        self.targetImage = targetImage  # property setter
        # list of subcontrols
        # The visibility of subcontrols is managed by QLayerView
        # together with the visibility of graphic forms
        self.subControls = []
        self.dataChanged.connect(self.updateLayer)
        # layer color picked signal
        if layer is not None:
            layer.colorPicked.sig.connect(self.colorPickedSlot)
        self.setStyleSheet("QListWidget, QLabel, QGroupBox {font-size : 7pt;}")

    def addSubcontrol(self, parent=None):
        dock = stateAwareQDockWidget(parent)
        self.subControls.append(dock)
        return dock

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class baseGraphicsForm(QGraphicsView, abstractForm):
    """
    Base class for graphic forms using a scene.
    """
    # Form state changed signal
    # Subclasses may redefine it with a different signature.
    # In that case, they must override abstractForm.reset() accordingly.
    dataChanged = QtCore.Signal()

    def __init__(self, layer=None, targetImage=None, parent=None):
        super().__init__(parent=parent)  # QGraphicsView __init__ is mandatory : don't rely on MRO !
        self.setAlignment(Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setFocusPolicy(Qt.ClickFocus)
        # back link to image layer (weak ref)
        self.layer = layer  # property setter
        self.targetImage = targetImage  # property setter
        # list of subcontrols
        self.subControls = []
        self.setScene(QGraphicsScene())
        # convenience attributes
        self.graphicsScene = weakProxy(self.scene())
        self.graphicsScene.options = None
        self.graphicsScene.layer = self.layer
        self.graphicsScene.targetImage = self.targetImage
        self.dataChanged.connect(self.updateLayer)
        # layer color picked signal
        if layer is not None:
            layer.colorPicked.sig.connect(self.colorPickedSlot)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass

    def wheelEvent(self, e):
        """
        Overrides QGraphicsView wheelEvent.
        Zoom the scene.
        @param e:
        @type e:
        """
        # delta unit is 1/8 of degree
        # Most mice have a resolution of 15 degrees
        numSteps = 1 + e.delta() / 2400.0  # 1200.0
        self.scale(numSteps, numSteps)

    def addSubcontrol(self, parent=None):
        dock = stateAwareQDockWidget(parent)
        self.subControls.append(dock)
        return dock

    def addCommandLayout(self, glayout):
        """
        Add a layout below the scene.
        That layout (e.g. a grid layout) is opaque and
        non-zoomable. It is supposed
        to contain option lists, buttons,...needed by
        the form.
        @param glayout:
        @type glayout: Qlayout
        @return:
        @rtype QWidget
        """
        container = bottomWidget()
        container.setLayout(glayout)
        vl1 = QVBoxLayout()
        vl1.setAlignment(Qt.AlignBottom)
        vl1.addWidget(container)
        self.setLayout(vl1)
        return container


class graphicsCurveForm(baseGraphicsForm):
    """
    Base class for interactive curve forms
    """

    @staticmethod
    def drawPlotGrid(axeSize, gradient=None):
        """
        Return a QGraphicsPathItem initialized with
        a square grid.
        @param axeSize:
        @type axeSize: int
        @param gradient:
        @type gradient: QGradient
        @return:
        @rtype: QGraphicsPathItem
        """
        lineWidth = 1
        item = QGraphicsPathItem()
        if gradient is None:
            item.setPen(QPen(Qt.darkGray, lineWidth, Qt.DashLine))
        else:
            item.setPen(QPen(QBrush(gradient), lineWidth, Qt.DashLine))
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

    def __init__(self, layer=None, targetImage=None, axeSize=500, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        # additional inactive curve to draw (QPolyLineF or list of QPointF)
        self.baseCurve = None
        self.setMinimumSize(axeSize + 60, axeSize + 140)
        self.graphicsScene.bgColor = QColor(200, 200, 200)
        self.graphicsScene.axeSize = axeSize
        self.axeSize = axeSize
        # add axes and grid to scene
        self.graphicsScene.defaultAxes = self.drawPlotGrid(axeSize)
        self.graphicsScene.addItem(self.graphicsScene.defaultAxes)
        # default WhatsThis for interactive curves
        self.setWhatsThis(
            """
            The background histogram is the <i>input</i> histogram.<br>
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
    def baseCurve(self):
        return self.__baseCurve

    @baseCurve.setter
    def baseCurve(self, points):
        self.__baseCurve = points
