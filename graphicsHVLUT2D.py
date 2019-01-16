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
        dSpline = activeBSpline(axeSize)
        graphicsScene = self.scene()
        graphicsScene.addItem(dSpline)

        dSpline.initFixedPoints()

        self.LUT = DeltaLUT3D((34, 32, 32))

        # set current curve to dsplacement spline
        self.cubicItem = dSpline
        graphicsScene.cubicItem = dSpline
        graphicsScene.cubicItem.setVisible(True)

        # buttons
        pushButton1 = QbLUePushButton("Reset Curve")
        pushButton1.clicked.connect(self.resetCurve)

        # options
        options = ['RGB', 'Red', 'Green', 'Blue']
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        self.listWidget1.setGeometry(0, 10, self.listWidget1.sizeHintForColumn(0) + 5, self.listWidget1.sizeHintForRow(0)*len(options) + 5)

        # layout
        gl = QGridLayout()
        gl.addWidget(self.listWidget1, 0, 0, 2, 1)
        for i, button in enumerate([pushButton1]):
            gl.addWidget(button, i, 1)
        self.addCommandLayout(gl)

        self.setWhatsThis(
            """<b>HV displacement curve</b><br>
            """ + self.whatsThis())

        dSpline.curveChanged.sig.connect(self.updateLayer)

    def updateLUT(self):
        """
        Updates the displacement LUT
        """
        data = self.LUT.data
        axeSize = self.axeSize
        hdivs = self.LUT.divs[0]
        hstep = 360 / hdivs
        activeSpline = self.cubicItem
        sp = activeSpline.spline
        d = axeSize // 2
        for i in range(hdivs):
            pt = sp[int(i * hstep * axeSize/360)]
            data[i, :, :, 2] = 1.0 -(pt.y() + d) / 100

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
        r,g,b= self.scene().targetImage.getActivePixel(x, y)
        if (modifiers & QtCore.Qt.ControlModifier) and (modifiers & QtCore.Qt.ShiftModifier):
            self.setBlackPoint(r,g,b)
        elif (modifiers & QtCore.Qt.ControlModifier):
            self.setWhitePoint(r, g, b)

    def setBlackPoint(self, r, g ,b):
        """

        @param r:
        @type r:
        @param g:
        @type g:
        @param b:
        @type b:
        """
        sc = self.scene()
        bPoint = min(r, g, b)
        # don't set black point to white !
        if bPoint >= 255:
            bPoint -= 10.0
        cubicRGB, cubicR, cubicG, cubicB = sc.cubicRGB, sc.cubicR, sc.cubicG, sc.cubicB
        for cubic in [cubicRGB, cubicR, cubicG, cubicB]:
            scale = cubic.size / 255.0
            fp = cubic.fixedPoints
            # find current white point
            wPoint = cubic.size
            tmp = [p.x() for p in fp if p.y() == -cubic.size]
            if tmp:
                wPoint = min(tmp)
            # remove control points at the left of wPoint, but the first
            for p in list(fp[1:-1]):
                if p.x() < wPoint:
                    fp.remove(p)
                    sc.removeItem(p)
            # add new black point if needed
            if bPoint > 0.0:
                a = activeSplinePoint(bPoint * scale, 0.0, parentItem=cubic)
                cubic.fixedPoints.append(a)
            cubic.fixedPoints.sort(key=lambda z: z.scenePos().x())
            cubic.updatePath()
            cubic.updateLUTXY()
        l = self.scene().layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def setWhitePoint(self, r, g, b):
        """

        @param r:
        @type r:
        @param g:
        @type g:
        @param b:
        @type b:
        """
        sc = self.scene()
        cubicRGB, cubicR, cubicG, cubicB = sc.cubicRGB, sc.cubicR, sc.cubicG, sc.cubicB
        for i, cubic in enumerate([cubicRGB, cubicR, cubicG, cubicB]):
            scale = cubic.size / 255.0
            fp = cubic.fixedPoints
            wPoint = max(r, g, b) if i==0 else r if i==1 else g if i==2 else b
            # don't set white point to black!
            if wPoint <= 10:
                wPoint += 10.0
            # find black point
            bPoint = 0.0
            tmp = [p.x() for p in fp if p.y() == 0.0]
            if tmp:
                bPoint = max(tmp)
            # remove control points at the right of bPoint
            for p in list(fp[1:-1]):
                if p.x() > bPoint:
                    cubic.fixedPoints.remove(p)
                    sc.removeItem(p)
            # add new white point if needed
            if wPoint < cubic.size:
                p = activeSplinePoint(wPoint * scale, -cubic.size, parentItem=cubic)
                cubic.fixedPoints.append(p)
            cubic.fixedPoints.sort(key=lambda z: z.scenePos().x())
            cubic.updatePath()
            cubic.updateLUTXY()
        l = self.scene().layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def drawBackground(self, qp, qrF):
        """
        Overrides QGraphicsView.drawBackground
        @param qp:
        @type qp: QPainter
        @param qrF:
        @type qrF: QRectF
        """
        super().drawBackground(qp, qrF)
        graphicsScene = self.scene()
        s = graphicsScene.axeSize
        if graphicsScene.cubicItem.histImg is not None:
            qp.drawImage(QRect(0, -s, s, s), graphicsScene.cubicItem.histImg)

    def resetCurve(self):
        """
        Button event handler
        Reset the current curve
        """
        graphicsScene = self.scene()
        graphicsScene.cubicItem.reset()
        self.updateHist(graphicsScene.cubicItem)
        # self.scene().onUpdateLUT()
        l = graphicsScene.layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def resetAllCurves(self):
        """
        Button event handler
        Reset R,G,B curves
        """
        graphicsScene = self.scene()
        for cubicItem in [graphicsScene.cubicR, graphicsScene.cubicG, graphicsScene.cubicB]:
            cubicItem.reset()
        self.updateHists()
        l = graphicsScene.layer
        l.applyToStack()
        l.parentImage.onImageChanged()

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

