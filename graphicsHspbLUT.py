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

from graphicsLUT import cubicSpline, graphicsCurveForm
from utils import optionsWidget, channelValues

class graphicsHspbForm(graphicsCurveForm) :
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        newWindow = graphicsHspbForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        newWindow.setWindowTitle(layer.name)
        return newWindow

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        # hue curve
        cubic = cubicSpline(axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicR = cubic
        cubic.channel = channelValues.Hue
        cubic.histImg = self.scene().layer.histogram(size=self.scene().axeSize,
                                                                    bgColor=self.scene().bgColor, range=(0, 360),
                                                                    chans=channelValues.Hue, mode='HSpB')
        cubic.initFixedPoints()
        # sat curve
        cubic = cubicSpline(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicG = cubic
        cubic.channel = channelValues.Sat
        cubic.histImg = self.scene().layer.histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, range=(0,1), chans=channelValues.Sat, mode='HSpB')
        cubic.initFixedPoints()
        # brightness curve
        cubic = cubicSpline(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicB = cubic
        cubic.channel = channelValues.Br
        cubic.histImg = self.scene().layer.histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, range=(0,1), chans=channelValues.Br, mode='HSpB')
        cubic.initFixedPoints()
        # set current curve
        self.scene().cubicItem = self.graphicsScene.cubicB
        self.scene().cubicItem.setVisible(True)

        # buttons
        pushButton1 = QPushButton("Reset Current")
        pushButton1.move(100, 20)
        pushButton1.adjustSize()
        pushButton1.clicked.connect(self.onResetCurve)
        self.graphicsScene.addWidget(pushButton1)
        pushButton2 = QPushButton("Reset All")
        pushButton2.move(100, 50)
        pushButton2.adjustSize()
        pushButton2.clicked.connect(self.onResetAllCurves)
        self.graphicsScene.addWidget(pushButton2)
        # options
        options = ['H', 'S', 'B']
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        self.listWidget1.setGeometry(0, 10, self.listWidget1.sizeHintForColumn(0) + 5, self.listWidget1.sizeHintForRow(0) * len(options) + 5)
        self.graphicsScene.addWidget(self.listWidget1)

        # selection changed handler
        curves = [self.graphicsScene.cubicR, self.graphicsScene.cubicG, self.graphicsScene.cubicB]
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
        s = self.graphicsScene.axeSize
        if self.scene().cubicItem.histImg is not None:
            qp.drawImage(QRect(0, -s, s, s), self.scene().cubicItem.histImg)

    def onResetCurve(self):
        """
        Button event handler
        Reset the selected curve
        """
        self.scene().cubicItem.reset()
        l = self.scene().layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def onResetAllCurves(self):
        """
        Button event handler
        Reset all curves
        """
        for cubicItem in [self.graphicsScene.cubicR, self.graphicsScene.cubicG, self.graphicsScene.cubicB]:
            cubicItem.reset()
        l = self.scene().layer
        l.applyToStack()
        l.parentImage.onImageChanged()

    def writeToStream(self, outStream):
        layer = self.scene().layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        if layer.actionName in ['actionBrightness_Contrast', 'actionCurves_HSpB', 'actionCurves_Lab']:
            outStream.writeQString(self.listWidget1.selectedItems()[0].text())
            self.graphicsScene.cubicR.writeToStream(outStream)
            self.graphicsScene.cubicG.writeToStream(outStream)
            self.graphicsScene.cubicB.writeToStream(outStream)
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
        self.graphicsScene.cubicR.readFromStream(inStream)
        self.graphicsScene.cubicG.readFromStream(inStream)
        self.graphicsScene.cubicB.readFromStream(inStream)
        return inStream