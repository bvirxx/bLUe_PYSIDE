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
from PySide2 import QtCore
from PySide2.QtCore import QRect, QPoint
from PySide2.QtWidgets import QGraphicsScene, QPushButton
from PySide2.QtGui import QPixmap, QRadialGradient
from PySide2.QtCore import Qt, QRectF

from colorConv import sRGB2LabVec
from bLUeGui.graphicsSpline import activeCubicSpline, graphicsCurveForm, activePoint, channelValues
from utils import optionsWidget

class graphicsLabForm(graphicsCurveForm):
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        newWindow = graphicsLabForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        newWindow.setWindowTitle(layer.name)
        return newWindow
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        graphicsScene = self.scene()
        #########
        # L curve
        #########
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicR = cubic
        cubic.channel = channelValues.L
        # get histogram as a Qimage
        cubic.histImg = graphicsScene.layer.inputImg().histogram(size=graphicsScene.axeSize,
                                                                    bgColor=graphicsScene.bgColor, range=(0, 1),
                                                                    chans=channelValues.L, mode='Lab')
        # L curve use the default axes
        cubic.axes = graphicsScene.defaultAxes
        cubic.initFixedPoints()
        cubic.axes.setVisible(False)
        cubic.setVisible(False)
        ##########
        # a curve (Green--> Magenta axis)
        #########
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicG = cubic
        cubic.channel = channelValues.a
        cubic.histImg = graphicsScene.layer.inputImg().histogram(size=graphicsScene.axeSize,
                                                                    bgColor=graphicsScene.bgColor, range=(-100, 100),
                                                                    chans=channelValues.a, mode='Lab')
        #  add specific axes
        gradient = QRadialGradient()
        gradient.setCenter(QPoint(0, 1))
        gradient.setRadius(axeSize*1.4)
        gradient.setColorAt(0.0, Qt.green)
        gradient.setColorAt(1.0, Qt.magenta)
        cubic.axes = self.drawPlotGrid(axeSize, gradient)
        graphicsScene.addItem(cubic.axes)
        cubic.initFixedPoints()
        cubic.axes.setVisible(False)
        cubic.setVisible(False)

        # b curve (Blue-->Yellow axis)
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicB = cubic
        cubic.channel = channelValues.b
        cubic.histImg = graphicsScene.layer.inputImg().histogram(size=graphicsScene.axeSize,
                                                                    bgColor=graphicsScene.bgColor, range=(-100, 100),
                                                                    chans=channelValues.b, mode='Lab')
        # add specific axes
        gradient.setColorAt(0.0, Qt.blue)
        gradient.setColorAt(1.0, Qt.yellow)
        cubic.axes = self.drawPlotGrid(axeSize, gradient)
        graphicsScene.addItem(cubic.axes)
        cubic.initFixedPoints()
        cubic.axes.setVisible(False)
        cubic.setVisible(False)

        # set current to L curve and axes
        graphicsScene.cubicItem = graphicsScene.cubicR
        graphicsScene.cubicItem.setVisible(True)
        graphicsScene.cubicItem.axes.setVisible(True)
        # buttons
        pushButton1 = QPushButton("Reset Current")
        pushButton1.move(100, 20)
        pushButton1.adjustSize()
        pushButton1.clicked.connect(self.resetCurve)
        graphicsScene.addWidget(pushButton1)
        pushButton2 = QPushButton("Reset All")
        pushButton2.move(100, 50)
        pushButton2.adjustSize()
        pushButton2.clicked.connect(self.resetAllCurves)
        graphicsScene.addWidget(pushButton2)
        # options
        options = ['L', 'a', 'b']
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        self.listWidget1.setGeometry(0, 10, self.listWidget1.sizeHintForColumn(0) + 5, self.listWidget1.sizeHintForRow(0) * len(options) + 5)
        graphicsScene.addWidget(self.listWidget1)

        # selection changed handler
        curves = [graphicsScene.cubicR, graphicsScene.cubicG, graphicsScene.cubicB]
        curveDict = dict(zip(options, curves))
        def onSelect1(item):
            cubicItem = self.scene().cubicItem
            cubicItem.setVisible(False)
            cubicItem.axes.setVisible(False)
            self.scene().cubicItem = curveDict[item.text()]
            self.scene().cubicItem.setVisible(True)
            self.scene().cubicItem.axes.setVisible(True)
            # Force to redraw  histogram
            self.scene().invalidate(QRectF(0.0, -self.scene().axeSize, self.scene().axeSize, self.scene().axeSize),
                                    QGraphicsScene.BackgroundLayer)
        self.listWidget1.onSelect = onSelect1
        # set initial selection to L
        item = self.listWidget1.items[options[0]]
        item.setCheckState(Qt.Checked)
        self.listWidget1.select(item)
        self.setWhatsThis("""<b>Lab curves</b><br>""" + self.whatsThis())

    def colorPickedSlot(self, x, y, modifiers):
        """
        sets black/white points
        @param x:
        @type x:
        @param y:
        @type y:
        @param modifiers:
        @type modifiers:
        """
        r,g,b= self.scene().targetImage.getActivePixel(x, y)
        if (modifiers & QtCore.Qt.ControlModifier):
            if modifiers & QtCore.Qt.ShiftModifier:
                self.setBlackPoint(r,g,b)
            else:
                self.setWhitePoint(r, g, b, luminance=True, balance=False)
        elif modifiers & QtCore.Qt.ShiftModifier:
            self.setWhitePoint(r, g, b, luminance=False, balance=True)

    def setBlackPoint(self, r, g ,b):
        """
        Sets the L curve
        @param r:
        @type r:
        @param g:
        @type g:
        @param b:
        @type b:
        """
        l = self.scene().layer
        if not l.isActiveLayer():
            return
        sc = self.scene()
        tmp = np.zeros((1,1,3,), dtype=np.uint8)
        tmp[0,0,:] = (r, g, b)
        L, a, b = sRGB2LabVec(tmp)[0,0,:]
        cubicL = sc.cubicR
        scale = cubicL.size
        bPoint = L * scale
        # don't set black point to white !
        if bPoint >= cubicL.size:
            bPoint -= 10.0
        fp = cubicL.fixedPoints
        # find current white point
        wPoint = cubicL.size
        tmp = [p.x() for p in fp if p.y() == -cubicL.size]
        if tmp:
            wPoint = min(tmp)
        # remove control points at the left of wPoint, but the first
        for p in list(fp[1:-1]):
            if p.x() < wPoint:
                fp.remove(p)
                sc.removeItem(p)
        # add new black point if needed
        if bPoint > 0.0 :
            a = activePoint(bPoint, 0.0, parentItem=cubicL)
            fp.append(a)
        fp.sort(key=lambda z: z.scenePos().x())
        cubicL.updatePath()
        cubicL.updateLUTXY()
        l.applyToStack()
        l.parentImage.onImageChanged()

    def setWhitePoint(self, r, g, b, luminance=True, balance=True):
        """
        for a, b curves, the method sets first and
        @param r:
        @type r:
        @param g:
        @type g:
        @param b:
        @type b:
        """
        l = self.scene().layer
        if not l.isActiveLayer():
            return
        sc = self.scene()
        tmp = np.zeros((1, 1, 3,), dtype=np.uint8)
        tmp[0, 0, :] = (r, g, b)
        L, a, b = sRGB2LabVec(tmp)[0, 0, :]
        cubicL, cubica, cubicb = sc.cubicR, sc.cubicG, sc.cubicB
        if luminance:
            ##########
            # L curve
            ##########
            cubic = cubicL
            scale = cubic.size
            fp = cubic.fixedPoints
            wPoint =  L * scale
            # don't set white point to black!
            if wPoint <= 10:
                wPoint += 10.0
            # find black point
            bPoint = 0.0
            tmp = [p.x() for p in fp if p.y()==0.0]
            if tmp:
                bPoint = max(tmp)
            # remove control points at the right of bPoint
            for p in list(fp[1:-1]):
                if p.x() > bPoint:
                    cubic.fixedPoints.remove(p)
                    sc.removeItem(p)
            # add new white point if needed
            if wPoint < cubic.size:
                p = activePoint(wPoint, -cubic.size, parentItem=cubic)
                cubic.fixedPoints.append(p)
                cubic.fixedPoints.sort(key=lambda z: z.scenePos().x())
            cubic.updatePath()
            cubic.updateLUTXY()
        if balance:
            #############
            # a, b curves
            #############
            corr = cubicL.size/8
            for i, cubic in enumerate([cubica, cubicb]):
                fp = cubic.fixedPoints
                scale = cubic.size / (127 * 2.0)
                wPoint = a * scale if i==0 else b * scale
                # remove all control points but the first and the last
                for p in list(fp[1:-1]):
                        fp.remove(p)
                        sc.removeItem(p)
                # reset first and last points
                #fp[0].setPos(0.0, 0.0)
                #fp[-1].setPos(scale, -scale)
                # according to the sign of wPoint, shift horizontally
                # first or last control point by 2*wPoint
                wPoint *= 2.0
                p = cubic.fixedPoints[0]
                p.setPos(max(0, wPoint) + corr, 0)
                p = cubic.fixedPoints[-1]
                p.setPos(min(cubic.size, cubic.size + wPoint) - corr, -cubic.size)
                cubic.updatePath()
                cubic.updateLUTXY()
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
        graphicsScene = self.scene()
        s = graphicsScene.axeSize
        if graphicsScene.cubicItem.histImg is not None:
            qp.drawPixmap(QRect(0, -s, s, s), QPixmap.fromImage(graphicsScene.cubicItem.histImg))

    def updateHist(self, curve, redraw=True):
        """
        Updates the channel histogram displayed under the curve

        @param curve:
        @type curve:

        """
        sc = self.scene()
        if curve is sc.cubicR:
            curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize,
                                                        bgColor=sc.bgColor, range=(0, 1),
                                                        chans=channelValues.L, mode='Lab')
        elif curve is sc.cubicG:
            curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize,
                                                        bgColor=sc.bgColor, range=(-100, 100),
                                                        chans=channelValues.a, mode='Lab')
        elif curve is sc.cubicB:
            curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize,
                                                        bgColor=sc.bgColor, range=(-100, 100),
                                                        chans=channelValues.b, mode='Lab')
        # Force to redraw histogram
        if redraw:
            sc.invalidate(QRectF(0.0, -sc.axeSize, sc.axeSize, sc.axeSize),
                            sc.BackgroundLayer)

    def updateHists(self):
        """
        Updates all histograms
        @return:
        @rtype:
        """
        sc = self.scene()
        for curve in [sc.cubicR, sc.cubicG, sc.cubicB]:
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
        l = graphicsScene.layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def resetAllCurves(self):
        """
        Button event handler
        Reset all curves
        """
        graphicsScene = self.scene()
        for cubicItem in [graphicsScene.cubicR, graphicsScene.cubicG, graphicsScene.cubicB]:
            cubicItem.reset()
        self.updateHists()
        l = graphicsScene.layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def writeToStream(self, outStream):
        graphicsScene = self.scene()
        layer = graphicsScene.layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        if layer.actionName in ['actionBrightness_Contrast', 'actionCurves_HSpB', 'actionCurves_Lab']:
            outStream.writeQString(self.listWidget1.selectedItems()[0].text())
            graphicsScene.cubicR.writeToStream(outStream)
            graphicsScene.cubicG.writeToStream(outStream)
            graphicsScene.cubicB.writeToStream(outStream)
        return outStream

    def readFromStream(self, inStream):
        actionName = inStream.readQString()
        name = inStream.readQString()
        sel = inStream.readQString()
        cubics = []
        #for i in range(3):
            #cubic = cubicItem.readFromStream(inStream)
            #cubics.append(cubic)
        #kwargs = dict(zip(['cubicR', 'cubicG', 'cubicB'], cubics))
        #self.setEntries(sel=sel, **kwargs)
        graphicsScene = self.scene()
        graphicsScene.cubicR.readFromStream(inStream)
        graphicsScene.cubicG.readFromStream(inStream)
        graphicsScene.cubicB.readFromStream(inStream)
        return inStream

