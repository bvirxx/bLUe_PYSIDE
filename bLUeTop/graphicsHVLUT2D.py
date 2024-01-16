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
from PySide6 import QtCore
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QGridLayout, QToolTip, QFrame

from bLUeCore.bLUeLUT3D import DeltaLUT3D
from bLUeGui.colorPatterns import cmHSP, graphicsHueShiftPattern, graphicsHueBrShiftPattern
from bLUeGui.graphicsSpline import graphicsCurveForm, activeBSpline
from bLUeGui.graphicsSpline import activeMarker
from bLUeTop.utils import QbLUeSlider


class HVLUT2DForm(graphicsCurveForm):
    """
    Form for interactive HV 2D LUT
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        graphicsScene = self.scene()

        # init brightness curve
        margin = 30  # space between B-Axes and H-Axes
        dSplineItem = activeBSpline(axeSize, period=axeSize, yZero=axeSize // 2 + margin)  # -3 * axeSize // 4)
        graphicsScene.addItem(dSplineItem)
        dSplineItem.setVisible(True)
        dSplineItem.initFixedPoints()
        self.dSplineItemB = dSplineItem
        graphicsScene.dSplineItemB = dSplineItem
        text = graphicsScene.addText('dV ')
        text.setDefaultTextColor(Qt.white)
        text.setPos(-25, -15 + axeSize // 2 + margin)

        baxes = graphicsCurveForm.drawPlotGrid(axeSize)
        graphicsScene.addItem(baxes)
        baxes.setPos(0, axeSize + margin)



        # init hue curve
        dSplineItem = activeBSpline(axeSize, period=axeSize, yZero=-axeSize // 2)
        graphicsScene.addItem(dSplineItem)
        dSplineItem.setVisible(True)
        dSplineItem.initFixedPoints()
        self.dSplineItemH = dSplineItem
        graphicsScene.dSplineItemH = dSplineItem
        text = graphicsScene.addText('dH ')
        text.setDefaultTextColor(Qt.white)
        text.setPos(-25, - (15 + self.axeSize // 2))

        # init 3D LUT
        self.LUT = DeltaLUT3D((34, 32, 32))

        self.hueBand = graphicsHueShiftPattern(self.axeSize, self.axeSize, cmHSP, 1, 1)
        self.hueBand.setPos(0, - self.axeSize)
        self.hueBand.setZValue(-100)
        self.scene().addItem(self.hueBand)

        self.BrBand = graphicsHueBrShiftPattern(self.axeSize, self.axeSize, cmHSP, 1, 1)
        self.BrBand.setPos(0, margin)
        self.BrBand.setZValue(-100)
        self.scene().addItem(self.BrBand)

        def showPos(e, x, y):
            QToolTip.showText(e.screenPos(), "%d" % (x * 360 // axeSize))

        # HueBand marker
        self.markerH = activeMarker.fromTriangle()
        self.markerH.setParentItem(self.hueBand)
        self.markerH.setPos(0, self.axeSize / 2 + 5)
        self.markerH.setMoveRange(QRect(0, self.axeSize / 2 + 5, axeSize, 0))

        self.scene().addItem(self.markerH)
        self.markerH.onMouseMove = showPos

        # BrBand  marker
        self.markerB = activeMarker.fromTriangle()
        self.markerB.setParentItem(self.BrBand)
        self.markerB.setPos(0, self.axeSize / 2 + 5)
        self.markerB.setMoveRange(QRect(0, self.axeSize / 2 + 5, axeSize, 0))

        self.scene().addItem(self.markerB)
        self.markerB.onMouseMove = showPos

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
        font = self.font()
        metrics = QFontMetrics(font)
        w = metrics.horizontalAdvance("0000")
        h = metrics.height()
        self.satValue.setMinimumSize(w, h)
        self.satValue.setMaximumSize(w, h)

        # layout
        gl = QGridLayout()

        q = QFrame()
        q.setFrameShape(QFrame.Shape.HLine)
        q.setStyleSheet("border : 1px solid rgb(128, 128, 128)")
        gl.addWidget(q, 0, 0, 1, 6)
        gl.addWidget(QLabel('Sat Thr '), 1, 0)
        gl.addWidget(self.satValue, 1, 1)
        gl.addWidget(self.sliderSat, 1, 2, 1, 3)  # 6 columns
        container = self.addCommandLayout(gl)
        container.adjustSize()

        self.setViewportMargins(10, 10, 10, container.height() + 25)  # left, top, right, bottom

        self.setDefaults()

        self.setWhatsThis(
            """<b>3D LUT HV Shift</b><br>
            Hue values are used to select image pixels. x-axis shows hue values from 0 to 360.<br>
            The upper curve (dH) shows hue additive shift (initially 0).<br> 
            The lower curve (dV) shows brightness multiplicative shifts (initially 1).<br>
            The curves are periodic (period 360) and displayed as gray lines.<br>
            Pixel colors are shifted by the hue and brightness shift values corresponding to their hue.<br>
            Each curve is controlled by <i>bump triangles</i>.<br>
            To <b>add a triangle</b> to the curve click anywhere on the curve.
            To <b>remove the triangle</b> Ctrl+Click on any vertex.<br>
            Drag the triangle vertices to move the bump along the x-axis and to change 
            its height and orientation. Use the <b> Sat Thr</b> slider 
            to preserve low saturated colors.<br>
            To <b>select a hue</b> Ctrl+Click on the image.<br>
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

        # update brightness
        activeSpline = self.dSplineItemB
        sp = activeSpline.spline[activeSpline.periodViewing:]
        d = activeSpline.yZero
        # reinit the LUT
        data[...] = 0, 1, 1
        for i in range(hdivs):
            pt = sp[int(i * hstep * axeSize / 360)]
            data[i, sThr:, :, 2] = 1.0 - (pt.y() - d) / 100

        # update hue
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

        :param x:
        :type x:
        :param y:
        :type y:
        :param modifiers:
        :type modifiers:
        """
        color = self.scene().targetImage.getActivePixel(x, y, qcolor=True)
        h = color.hsvHue()
        if modifiers == QtCore.Qt.ControlModifier:
            self.markerH.setPos(h * self.axeSize / 360, self.markerH.pos().y())
            self.markerB.setPos(h * self.axeSize / 360, self.markerB.pos().y())
            self.update()

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [QbLUeSlider, activeBSpline]:
                d[a] = obj.__getstate__()
        return d

    def __setstate__(self, d):
        # prevent multiple updates
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        for name in d['state']:
            obj = getattr(self, name, None)
            if type(obj) in [QbLUeSlider, activeBSpline]:
                obj.__setstate__(d['state'][name])
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()