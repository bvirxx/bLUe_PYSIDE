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
from PySide2.QtCore import QRect
from PySide2.QtWidgets import QGraphicsScene, QPushButton
from PySide2.QtCore import Qt, QRectF

from graphicsLUT import activeCubicSpline, graphicsCurveForm
from utils import optionsWidget, channelValues

class graphicsHspbForm(graphicsCurveForm) :
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        newWindow = graphicsHspbForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        newWindow.setWindowTitle(layer.name)
        return newWindow
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        graphicsScene = self.scene()
        # hue curve
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicR = cubic
        cubic.channel = channelValues.Hue
        cubic.histImg = graphicsScene.layer.histogram(size=axeSize,
                                                       bgColor=graphicsScene.bgColor, range=(0, 360),
                                                       chans=channelValues.Hue, mode='HSpB')
        cubic.initFixedPoints()
        # sat curve
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicG = cubic
        cubic.channel = channelValues.Sat
        cubic.histImg = graphicsScene.layer.histogram(size=axeSize,
                                                      bgColor=graphicsScene.bgColor, range=(0,1),
                                                      chans=channelValues.Sat, mode='HSpB')
        cubic.initFixedPoints()
        # brightness curve
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicB = cubic
        cubic.channel = channelValues.Br
        cubic.histImg = graphicsScene.layer.histogram(size=axeSize,
                                                      bgColor=graphicsScene.bgColor,
                                                      range=(0,1), chans=channelValues.Br, mode='HSpB')
        cubic.initFixedPoints()
        # set current curve
        graphicsScene.cubicItem = graphicsScene.cubicB
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
        options = ['H', 'S', 'B']
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
            # draw  histogram
            self.scene().invalidate(QRectF(0.0, -self.scene().axeSize, self.scene().axeSize, self.scene().axeSize), QGraphicsScene.BackgroundLayer)

        self.listWidget1.onSelect = onSelect1
        # set initial selection to Saturation
        item = self.listWidget1.items[options[1]]
        item.setCheckState(Qt.Checked)
        self.listWidget1.select(item)

    def drawBackground(self, qp, qrF):
        graphicsScene = self.scene()
        s = graphicsScene.axeSize
        if graphicsScene.cubicItem.histImg is not None:
            qp.drawImage(QRect(0, -s, s, s), graphicsScene.cubicItem.histImg)

    def onResetCurve(self):
        """
        Button event handler
        Reset the selected curve
        """
        graphicsScene = self.scene()
        graphicsScene.cubicItem.reset()
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
        # for i in range(3):
        # cubic = cubicItem.readFromStream(inStream)
        # cubics.append(cubic)
        # kwargs = dict(zip(['cubicR', 'cubicG', 'cubicB'], cubics))
        # self.setEntries(sel=sel, **kwargs)
        graphicsScene = self.scene()
        graphicsScene.cubicR.readFromStream(inStream)
        graphicsScene.cubicG.readFromStream(inStream)
        graphicsScene.cubicB.readFromStream(inStream)
        return inStream