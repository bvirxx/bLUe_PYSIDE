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
from PySide2.QtCore import QRect
from PySide2.QtCore import Qt, QRectF
from PySide2.QtWidgets import QGraphicsScene, QGridLayout

from bLUeCore.bLUeLUT3D import DeltaLUT3D
from bLUeGui.graphicsSpline import activeCubicSpline, graphicsCurveForm, activeSplinePoint, channelValues, activeBSpline
from graphicsLUT3D import activeMarker
from utils import optionsWidget, QbLUePushButton


class HVLUT2DForm(graphicsCurveForm) :
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
        # Init curve
        dSpline = activeBSpline(axeSize, period=axeSize)
        graphicsScene = self.scene()
        graphicsScene.addItem(dSpline)

        dSpline.initFixedPoints()

        self.LUT = DeltaLUT3D((34, 32, 32))

        # set current curve to dsplacement spline
        self.cubicItem = dSpline
        graphicsScene.cubicItem = dSpline
        graphicsScene.cubicItem.setVisible(True)

        self.marker = activeMarker.fromTriangle(parent=self.cubicItem)
        self.scene().addItem(self.marker)

        self.setWhatsThis(
            """<b>(Hue, Brightness) correction curve</b><br>
            The curve represents a brightness correction (initially 0) for 
            each value of the hue. The specific brightness correction corresponding 
            to its hue is applied to each image pixel.<br> 
            To <b>add a bump triangle</b> click anywhere on the curve.<br>
            Drag the triangle vertices to move the bump along the x-axis and to change 
            its height and orientation.<br>
            To <b>set the Hue Value Marker</b> Ctrl+click on the image.
            """)

        dSpline.curveChanged.sig.connect(self.updateLayer)

    def updateLUT(self):
        """
        Updates the displacement LUT
        """
        data = self.LUT.data
        axeSize = self.axeSize
        hdivs = self.LUT.divs[0]
        sThr = self.LUT.divs[1] // 4  # unsaturated color preservation threshold
        hstep = 360 / hdivs
        activeSpline = self.cubicItem
        sp = activeSpline.spline[activeSpline.periodViewing:]
        d = axeSize // 2
        for i in range(hdivs):
            pt = sp[int(i * hstep * axeSize/360)]
            data[i, sThr:, :, 2] = 1.0 -(pt.y() + d) / 100

    def updateLayer(self):
        self.updateLUT()
        l = self.scene().layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def colorPickedSlot(self, x, y, modifiers):
        """
        sets black/white points
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
        if (modifiers & QtCore.Qt.ControlModifier):
            self.marker.setPos(h * 300/360, -self.axeSize//2)

    def writeToStream(self, outStream):
        """

        @param outStream:
        @type outStream: QDataStream
        @return:
        @rtype: QDataStream
        """
        graphicsScene = self.scene()
        layer = graphicsScene.layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        if layer.actionName in ['actionBrightness_Contrast', 'actionCurves_HSpB', 'actionCurves_Lab']:
            outStream.writeQString(self.listWidget1.selectedItems()[0].text())
            graphicsScene.cubicRGB.writeToStream(outStream)
            graphicsScene.cubicR.writeToStream(outStream)
            graphicsScene.cubicG.writeToStream(outStream)
            graphicsScene.cubicB.writeToStream(outStream)
        return outStream

    def readFromStream(self, inStream):
        actionName = inStream.readQString()
        name = inStream.readQString()
        sel = inStream.readQString()
        graphicsScene = self.scene()
        graphicsScene.cubicRGB.readFromStream(inStream)
        graphicsScene.cubicR.readFromStream(inStream)
        graphicsScene.cubicG.readFromStream(inStream)
        graphicsScene.cubicB.readFromStream(inStream)
        return inStream

