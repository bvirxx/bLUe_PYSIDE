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
from PySide2.QtCore import Qt, QRect
from PySide2.QtGui import QFontMetrics, QColor
from PySide2.QtWidgets import QLabel, QGridLayout

from bLUeCore.bLUeLUT3D import DeltaLUT3D
from bLUeGui.graphicsSpline import graphicsCurveForm, activeBSpline
from bLUeTop.graphicsLUT3D import activeMarker
from bLUeTop.utils import QbLUeSlider


class HVLUT2DForm(graphicsCurveForm):
    """
    Form for interactive HV 2D LUT
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        newWindow = HVLUT2DForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        newWindow.setWindowTitle(layer.name)
        return newWindow

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        graphicsScene = self.scene()
        # connect layer selectionChanged signal
        self.layer.selectionChanged.sig.connect(self.updateLayer)
        # Init curves
        dSplineItem = activeBSpline(axeSize, period=axeSize, yZero=-3*axeSize//4)
        graphicsScene.addItem(dSplineItem)
        dSplineItem.setVisible(True)
        dSplineItem.initFixedPoints()
        self.dSplineItemB = dSplineItem
        graphicsScene.dSplineItemB = dSplineItem
        text = graphicsScene.addText('Hue (0-360)')
        text.setDefaultTextColor(Qt.white)
        text.setPos(-40, 10)
        text = graphicsScene.addText('delta H')
        text.setDefaultTextColor(Qt.white)
        text.setPos(-40, -self.axeSize // 4)
        text = graphicsScene.addText('delta B')
        text.setDefaultTextColor(Qt.white)
        text.setPos(-40, -(self.axeSize * 3) // 4)

        dSplineItem = activeBSpline(axeSize, period=axeSize, yZero=-axeSize // 4)
        graphicsScene.addItem(dSplineItem)
        dSplineItem.setVisible(True)
        dSplineItem.initFixedPoints()
        self.dSplineItemH = dSplineItem
        graphicsScene.dSplineItemH = dSplineItem

        # init 3D LUT
        self.LUT = DeltaLUT3D((34, 32, 32))

        self.marker = activeMarker.fromTriangle(parent=self.dSplineItemB)
        self.marker.setPos(0, 0)  # -(axeSize * 3) // 4)
        self.marker.setMoveRange(QRect(0, 0, axeSize, 0))
        self.scene().addItem(self.marker)

        def showPos(e, x, y):
            self.markerLabel.setText("%d" % (x * 360 // axeSize))

        self.marker.onMouseMove = showPos

        self.markerLabel = QLabel()
        font = self.markerLabel.font()
        metrics = QFontMetrics(font)
        w = metrics.width("0000")
        h = metrics.height()
        self.markerLabel.setMinimumSize(w, h)
        self.markerLabel.setMaximumSize(w, h)

        self.sliderSat = QbLUeSlider(Qt.Horizontal)
        self.sliderSat.setMinimumWidth(200)

        def satUpdate(value):
            self.satValue.setText(str("{:d}".format(value)))
            # move not yet terminated or values not modified
            if self.sliderSat.isSliderDown() or value == self.satThr:
                return
            try:
                self.sliderSat.valueChanged.disconnect()
                self.sliderSat.sliderReleased.disconnect()
            except RuntimeError:
                pass
            self.satThr = value
            self.dataChanged.emit()
            self.sliderSat.valueChanged.connect(satUpdate)
            self.sliderSat.sliderReleased.connect(lambda: satUpdate(self.sliderSat.value()))

        self.sliderSat.valueChanged.connect(satUpdate)
        self.sliderSat.sliderReleased.connect(lambda: satUpdate(self.sliderSat.value()))

        self.satValue = QLabel()
        font = self.markerLabel.font()
        metrics = QFontMetrics(font)
        w = metrics.width("0000")
        h = metrics.height()
        self.satValue.setMinimumSize(w, h)
        self.satValue.setMaximumSize(w, h)

        # layout
        gl = QGridLayout()
        gl.addWidget(QLabel('Hue '), 0, 0)
        gl.addWidget(self.markerLabel, 0, 1)
        gl.addWidget(QLabel('Sat Thr '), 1, 0)
        gl.addWidget(self.satValue, 1, 1)
        gl.addWidget(self.sliderSat, 1, 2, 4, 1)
        self.addCommandLayout(gl)

        self.setDefaults()

        self.setWhatsThis(
            """<b>3D LUT Shift HSV</b><br>
            All pixel colors are changed by the specific hue and brightness shifts corresponding to their hue.<br>
            x-axis represents hue values from 0 to 360. 
            The upper curve shows brightness multiplicative shifts (initially 1) and
            the lower curve hue additive shifts (initially 0). <br>
            Each curve is controlled by bump triangles.<br>
            To <b>add a bump triangle</b> to the curve click anywhere on the curve.
            To <b>remove the triangle</b> click on any vertex.<br>
            Drag the triangle vertices to move the bump along the x-axis and to change 
            its height and orientation. Use the <b> Sat Thr</b> slider 
            to preserve low saturated colors.<br>
            To <b>set the Hue Value Marker</b> Ctrl+click on the image.<br>
            To limit the shift corrections to a region of the image select the desired area
            with the rectangular marquee tool.<br>
            <b>Zoom</b> the curves with the mouse wheel.<br>
            """)

    def setDefaults(self):
        try:
            self.dataChanged.disconnect()
            self.dSplineItemB.curveChanged.sig.disconnect()
            self.dSplineItemH.curveChanged.sig.disconnect()
        except RuntimeError:
            pass
        self.satThr = 10
        self.sliderSat.setValue(self.satThr)
        self.dataChanged.connect(self.updateLayer)
        self.dSplineItemB.curveChanged.sig.connect(self.updateLayer)
        self.dSplineItemH.curveChanged.sig.connect(self.updateLayer)

    def updateLUT(self):
        """
        Updates the displacement LUT
        """
        data = self.LUT.data
        axeSize = self.axeSize
        hdivs = self.LUT.divs[0]
        # sat threshold
        sThr = int(self.LUT.divs[1] * self.satThr / 100)
        hstep = 360 / hdivs
        activeSpline = self.dSplineItemB
        sp = activeSpline.spline[activeSpline.periodViewing:]
        d = activeSpline.yZero
        # reinit the LUT
        data[...] = 0, 1, 1
        # update brightness
        for i in range(hdivs):
            pt = sp[int(i * hstep * axeSize/360)]
            data[i, sThr:, :, 2] = 1.0 - (pt.y() - d) / 100

        activeSpline = self.dSplineItemH
        sp = activeSpline.spline[activeSpline.periodViewing:]
        d = activeSpline.yZero
        for i in range(hdivs):
            pt = sp[int(i * hstep * axeSize / 360)]
            data[i, sThr:, :, 0] = - (pt.y() - d) / 5

    def updateLayer(self):
        self.updateLUT()
        l = self.scene().layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def colorPickedSlot(self, x, y, modifiers):
        """
        Updates cursor from the hue of an image pixel.
        (x,y) coordinates are relative to the full size image.
        @param x:
        @type x:
        @param y:
        @type y:
        @param modifiers:
        @type modifiers:
        """
        color = self.scene().targetImage.getActivePixel(x, y, qcolor=True)
        h = color.hsvHue()
        if modifiers == QtCore.Qt.ControlModifier:
            self.marker.setPos(h * 300/360, 0)  # -(self.axeSize * 3) // 4)
            self.markerLabel.setText("%d" % h)
            self.update()


