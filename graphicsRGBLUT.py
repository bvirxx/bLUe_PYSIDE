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
from PySide2.QtCore import Qt, QRectF
from PySide2.QtWidgets import QPushButton, QGraphicsScene

from graphicsLUT import cubicSpline, graphicsCurveForm
from utils import optionsWidget, channelValues

class graphicsForm(graphicsCurveForm) :
    """
    Form for interactive RGB curves
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        newWindow = graphicsForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        newWindow.setWindowTitle(layer.name)
        return newWindow

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        # Brightness curve
        cubic = cubicSpline(axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicRGB = cubic
        cubic.channel = channelValues.RGB
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, chans=[], mode='Luminosity')
        cubic.initFixedPoints()
        # Red curve
        cubic = cubicSpline(axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicR = cubic
        cubic.channel = channelValues.Red
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, chans=channelValues.Red)
        cubic.initFixedPoints()
        # Green curve
        cubic = cubicSpline(axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicG = cubic
        cubic.channel = channelValues.Green
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, chans=channelValues.Green)
        cubic.initFixedPoints()
        # Blue curve
        cubic = cubicSpline(axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicB = cubic
        cubic.channel = channelValues.Blue
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize, bgColor=self.scene().bgColor, chans=channelValues.Blue)
        cubic.initFixedPoints()
        # set current curve to brightness
        self.scene().cubicItem = self.graphicsScene.cubicRGB
        self.scene().cubicItem.setVisible(True)

        # buttons
        pushButton1 = QPushButton("Reset Current")
        pushButton1.move(100,20)
        pushButton1.adjustSize()
        pushButton1.clicked.connect(self.onResetCurve)
        self.graphicsScene.addWidget(pushButton1)
        pushButton2 = QPushButton("Reset R,G,B")
        pushButton2.move(100, 50)
        pushButton2.adjustSize()
        pushButton2.clicked.connect(self.onResetAllCurves)
        self.graphicsScene.addWidget(pushButton2)

        # options
        options = ['RGB', 'Red', 'Green', 'Blue']
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        self.listWidget1.setGeometry(0, 10, self.listWidget1.sizeHintForColumn(0)+5, self.listWidget1.sizeHintForRow(0)*len(options) + 5)
        self.graphicsScene.addWidget(self.listWidget1)
        # selection changed handler
        curves = [self.graphicsScene.cubicRGB, self.graphicsScene.cubicR, self.graphicsScene.cubicG, self.graphicsScene.cubicB]
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

    def drawBackground(self, qp, qrF):
        """
        Overrides QGraphicsView.drawBackground
        @param qp:
        @type qp: QPainter
        @param qrF:
        @type qrF: QRectF
        """
        super().drawBackground(qp, qrF)
        s = self.graphicsScene.axeSize
        if self.scene().cubicItem.histImg is not None:
            qp.drawImage(QRect(0, -s, s, s), self.scene().cubicItem.histImg)

    def onResetCurve(self):
        """
        Button event handler
        Reset the selected curve
        """
        self.scene().cubicItem.reset()
        # self.scene().onUpdateLUT()
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
        """

        @param outStream:
        @type outStream: QDataStream
        @return:
        @rtype: QDataStream
        """
        layer = self.scene().layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        if layer.actionName in ['actionBrightness_Contrast', 'actionCurves_HSpB', 'actionCurves_Lab']:
            outStream.writeQString(self.listWidget1.selectedItems()[0].text())
            self.graphicsScene.cubicRGB.writeToStream(outStream)
            self.graphicsScene.cubicR.writeToStream(outStream)
            self.graphicsScene.cubicG.writeToStream(outStream)
            self.graphicsScene.cubicB.writeToStream(outStream)
        return outStream

    def readFromStream(self, inStream):
        actionName = inStream.readQString()
        name = inStream.readQString()
        sel = inStream.readQString()
        self.graphicsScene.cubicRGB.readFromStream(inStream)
        self.graphicsScene.cubicR.readFromStream(inStream)
        self.graphicsScene.cubicG.readFromStream(inStream)
        self.graphicsScene.cubicB.readFromStream(inStream)
        return inStream



