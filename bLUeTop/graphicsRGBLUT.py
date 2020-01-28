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

from bLUeGui.graphicsSpline import activeCubicSpline, graphicsCurveForm, activeSplinePoint, channelValues
from bLUeTop.utils import optionsWidget, QbLUePushButton


class graphicsForm(graphicsCurveForm):
    """
    Form for interactive RGB curves
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        newWindow = graphicsForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        newWindow.setWindowTitle(layer.name)
        return newWindow

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        # Brightness curve
        cubic = activeCubicSpline(axeSize)
        graphicsScene = self.scene()
        graphicsScene.addItem(cubic)
        graphicsScene.cubicRGB = cubic
        cubic.channel = channelValues.RGB
        cubic.histImg = self.scene().layer.inputImg().histogram(size=graphicsScene.axeSize,
                                                                bgColor=graphicsScene.bgColor, chans=[], mode='Luminosity')
        cubic.initFixedPoints()
        # Red curve
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicR = cubic
        cubic.channel = channelValues.Red
        cubic.histImg = self.scene().layer.inputImg().histogram(size=graphicsScene.axeSize,
                                                                bgColor=graphicsScene.bgColor, chans=channelValues.Red)
        cubic.initFixedPoints()
        # Green curve
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicG = cubic
        cubic.channel = channelValues.Green
        cubic.histImg = self.scene().layer.inputImg().histogram(size=graphicsScene.axeSize,
                                                                bgColor=graphicsScene.bgColor, chans=channelValues.Green)
        cubic.initFixedPoints()
        # Blue curve
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicB = cubic
        cubic.channel = channelValues.Blue
        cubic.histImg = self.scene().layer.inputImg().histogram(size=graphicsScene.axeSize,
                                                                bgColor=graphicsScene.bgColor, chans=channelValues.Blue)
        cubic.initFixedPoints()
        # set current curve to brightness
        graphicsScene.cubicItem = graphicsScene.cubicRGB
        graphicsScene.cubicItem.setVisible(True)

        # buttons
        pushButton1 = QbLUePushButton("Reset Current")
        pushButton1.clicked.connect(self.resetCurve)
        pushButton2 = QbLUePushButton("Reset R,G,B")
        pushButton2.clicked.connect(self.resetAllCurves)

        # options
        options = ['RGB', 'Red', 'Green', 'Blue']
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        self.listWidget1.setGeometry(0, 0, self.listWidget1.sizeHintForColumn(0) + 5,
                                     self.listWidget1.sizeHintForRow(0)*len(options) + 5)
        # selection changed handler
        curves = [graphicsScene.cubicRGB, graphicsScene.cubicR, graphicsScene.cubicG, graphicsScene.cubicB]
        curveDict = dict(zip(options, curves))

        def onSelect1(item):
            self.scene().cubicItem.setVisible(False)
            self.scene().cubicItem = curveDict[item.text()]
            pushButton2.setEnabled(item.text() != 'RGB')
            self.scene().cubicItem.setVisible(True)
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
            # Force redraw histogram
            self.scene().invalidate(QRectF(0.0, -self.scene().axeSize, self.scene().axeSize, self.scene().axeSize),
                                    QGraphicsScene.BackgroundLayer)
        self.listWidget1.onSelect = onSelect1
        # set initial selection to RGB
        item = self.listWidget1.items[options[0]]
        item.setCheckState(Qt.Checked)
        self.listWidget1.select(item)

        # layout
        gl = QGridLayout()
        gl.addWidget(self.listWidget1, 0, 0, 2, 1)
        for i, button in enumerate([pushButton1, pushButton2]):
            gl.addWidget(button, i, 1)
        self.addCommandLayout(gl)

        self.setWhatsThis("""<b>RGB curves</b><br>""" + self.whatsThis())

        def f():
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        self.scene().cubicRGB.curveChanged.sig.connect(f)
        self.scene().cubicR.curveChanged.sig.connect(f)
        self.scene().cubicG.curveChanged.sig.connect(f)
        self.scene().cubicB.curveChanged.sig.connect(f)

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
        r, g, b = self.scene().targetImage.getActivePixel(x, y)
        if modifiers == QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier:
            self.setBlackPoint(r, g, b)
        elif modifiers == QtCore.Qt.ControlModifier:
            self.setWhitePoint(r, g, b)

    def setBlackPoint(self, r, g, b):
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
            wPoint = max(r, g, b) if i == 0 else r if i == 1 else g if i == 2 else b
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

    def updateHist(self, curve, redraw=True):
        """
        Update the channel histogram displayed under the curve
        @param curve:
        @type curve:
        @param redraw:
        @type redraw
        """
        sc = self.scene()
        if curve is sc.cubicRGB:
            curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize, bgColor=sc.bgColor, chans=[], mode='Luminosity')
        elif curve is sc.cubicR:
            curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize, bgColor=sc.bgColor, chans=channelValues.Red)
        elif curve is sc.cubicG:
            curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize, bgColor=sc.bgColor, chans=channelValues.Green)
        elif curve is sc.cubicB:
            curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize, bgColor=sc.bgColor, chans=channelValues.Blue)
        # Force to redraw histogram
        if redraw:
            sc.invalidate(QRectF(0.0, -sc.axeSize, sc.axeSize, sc.axeSize),
                            sc.BackgroundLayer)

    def updateHists(self):
        """
        Updates all histograms
        """
        sc = self.scene()
        for curve in [sc.cubicRGB, sc.cubicR, sc.cubicG, sc.cubicB]:
            self.updateHist(curve, redraw=False)
        # Force to redraw histogram
        sc.invalidate(QRectF(0.0, -sc.axeSize, sc.axeSize, sc.axeSize),
                        sc.BackgroundLayer)

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
    """
    def writeToStream(self, outStream):
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
    """



