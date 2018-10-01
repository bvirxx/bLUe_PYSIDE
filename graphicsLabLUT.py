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
from PySide2.QtCore import QRect
from PySide2.QtWidgets import QGraphicsScene, QPushButton
from PySide2.QtGui import QPixmap
from PySide2.QtCore import Qt, QRectF

from colorConv import sRGB2LabVec
from graphicsLUT import activeCubicSpline, graphicsCurveForm, activePoint
from utils import optionsWidget, channelValues

class graphicsLabForm(graphicsCurveForm):
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        newWindow = graphicsLabForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        newWindow.setWindowTitle(layer.name)
        return newWindow
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        graphicsScene = self.scene()
        # L curve
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicR = cubic
        cubic.channel = channelValues.L
        # get histogram as Qimage
        cubic.histImg = graphicsScene.layer.inputImg().histogram(size=graphicsScene.axeSize,
                                                                    bgColor=graphicsScene.bgColor, range=(0, 1),
                                                                    chans=channelValues.L, mode='Lab')
        cubic.initFixedPoints()
        # a curve (Green--> Red axis)
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicG = cubic
        cubic.channel = channelValues.a
        cubic.histImg = graphicsScene.layer.inputImg().histogram(size=axeSize,
                                                                    bgColor=graphicsScene.bgColor, range=(-100, 100),
                                                                    chans=channelValues.a, mode='Lab')
        cubic.initFixedPoints()
        # b curve (Blue-->Yellow axis)
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicB = cubic
        cubic.channel = channelValues.b
        cubic.histImg = graphicsScene.layer.inputImg().histogram(size=axeSize,
                                                                    bgColor=graphicsScene.bgColor, range=(-100, 100),
                                                                    chans=channelValues.b, mode='Lab')
        cubic.initFixedPoints()
        # set current curve to L curve
        graphicsScene.cubicItem = graphicsScene.cubicR
        graphicsScene.cubicItem.setVisible(True)
        # buttons
        pushButton1 = QPushButton("Reset Current")
        pushButton1.move(100, 20)
        pushButton1.adjustSize()
        pushButton1.clicked.connect(self.onResetCurve)
        graphicsScene.addWidget(pushButton1)
        pushButton2 = QPushButton("Reset All")
        pushButton2.move(100, 50)
        pushButton2.adjustSize()
        pushButton2.clicked.connect(self.onResetAllCurves)
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
            self.scene().cubicItem.setVisible(False)
            self.scene().cubicItem = curveDict[item.text()]
            self.scene().cubicItem.setVisible(True)
            # Force redraw  histogram
            self.scene().invalidate(QRectF(0.0, -self.scene().axeSize, self.scene().axeSize, self.scene().axeSize),
                                    QGraphicsScene.BackgroundLayer)

        self.listWidget1.onSelect = onSelect1
        # set initial selection to L
        item = self.listWidget1.items[options[0]]
        item.setCheckState(Qt.Checked)
        self.listWidget1.select(item)
        self.setWhatsThis("""<b>Lab curves</b><br>""" + self.whatsThis())

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
        tmp = np.zeros((1,1,3,), dtype=np.uint8)
        tmp[0,0,:] = (r, g, b)
        L, a, b = sRGB2LabVec(tmp)[0,0,:]
        cubicL = sc.cubicR
        scale = cubicL.size
        bPoint = L * scale
        # don't set black point to white !
        if bPoint >= cubicL.size:
            bPoint -= 10.0
        for p in list(cubicL.fixedPoints):
            if (p.x() > 0.0 and p.x() <= bPoint) or (p.y() == 0.0 and p.x() > bPoint):
                    cubicL.fixedPoints.remove(p)
                    sc.removeItem(p)
        try:
            a = activePoint(bPoint, 0.0, parentItem=cubicL)
            cubicL.fixedPoints.append(a)
            cubicL.fixedPoints.sort(key=lambda z: z.scenePos().x())
            cubicL.updatePath()
            cubicL.updateLUTXY()
        except ValueError: # empty list
            pass
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
        tmp = np.zeros((1, 1, 3,), dtype=np.uint8)
        tmp[0, 0, :] = (r, g, b)
        L, a, b = sRGB2LabVec(tmp)[0, 0, :]
        cubicL, cubica, cubicb = sc.cubicR, sc.cubicG, sc.cubicB
        scale = cubicL.size
        for i, cubic in enumerate([cubicL, cubica, cubicb]):
            scale = cubic.size / 1.0 if i == 0 else 127.0
            wPoint =  L * scale if i==0 else a * scale if i==1 else b * scale
            # don't set white point to black!
            if wPoint <= 10:
                wPoint += 10.0
            for p in list(cubic.fixedPoints):
                if (p.x() > 255.0 and p.x() >= wPoint) or (p.y() == 255.0 and p.x() < wPoint):
                    cubic.fixedPoints.remove(p)
                    sc.removeItem(p)
            try:
                p = activePoint(wPoint, -cubic.size if i==0 else -cubic.size//2, parentItem=cubic)
                cubic.fixedPoints.append(p)
                cubic.fixedPoints.sort(key=lambda z: z.scenePos().x())
                cubic.updatePath()
                cubic.updateLUTXY()
            except ValueError:  # empty list
                pass
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
        graphicsScene = self.scene()
        s = graphicsScene.axeSize
        if graphicsScene.cubicItem.histImg is not None:
            qp.drawPixmap(QRect(0, -s, s, s), QPixmap.fromImage(graphicsScene.cubicItem.histImg))

    def onResetCurve(self):
        """
        Button event handler
        Reset the selected curve
        """
        graphicsScene = self.scene()
        graphicsScene.cubicItem.reset()
        # self.scene().onUpdateLUT()
        l = graphicsScene.layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def onResetAllCurves(self):
        """
        Button event handler
        Reset all curves
        """
        graphicsScene = self.scene()
        for cubicItem in [graphicsScene.cubicR, graphicsScene.cubicG, graphicsScene.cubicB]:
            cubicItem.reset()
        # self.scene().onUpdateLUT()
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

