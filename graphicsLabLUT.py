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
from PySide2.QtWidgets import QSizePolicy, QGraphicsScene, QPushButton
from PySide2.QtGui import  QColor, QPixmap
from PySide2.QtCore import Qt, QRectF

from graphicsLUT import cubicItem, graphicsCurveForm
from utils import optionsWidget, channelValues, drawPlotGrid

class graphicsLabForm(graphicsCurveForm):
    strokeWidth = 3
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        newWindow = graphicsLabForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        newWindow.setWindowTitle(layer.name)
        return newWindow

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        # curves
        cubic = cubicItem(axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicR = cubic
        cubic.channel = channelValues.L
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize,
                                                                    bgColor=self.scene().bgColor, range=(0, 1),
                                                                    chans=channelValues.L, mode='Lab')
        cubic.initFixedPoints()
        cubic = cubicItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicG = cubic
        cubic.channel = channelValues.a
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize,
                                                                    bgColor=self.scene().bgColor, range=(-100, 100),
                                                                    chans=channelValues.a, mode='Lab')
        cubic.initFixedPoints()
        cubic = cubicItem(self.graphicsScene.axeSize)
        self.graphicsScene.addItem(cubic)
        self.graphicsScene.cubicB = cubic
        cubic.channel = channelValues.b
        cubic.histImg = self.scene().layer.inputImg().histogram(size=self.scene().axeSize,
                                                                    bgColor=self.scene().bgColor, range=(-100, 100),
                                                                    chans=channelValues.b, mode='Lab')
        cubic.initFixedPoints()
        # set current
        self.scene().cubicItem = self.graphicsScene.cubicR
        self.scene().cubicItem.setVisible(True)

        def onResetCurve():
            """
            Reset the selected curve
            """
            self.scene().cubicItem.reset()
            #self.scene().onUpdateLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        def onResetAllCurves():
            """
            Reset all curves
            """
            for cubicItem in [self.graphicsScene.cubicR, self.graphicsScene.cubicG, self.graphicsScene.cubicB]:
                cubicItem.reset()
            #self.scene().onUpdateLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()
        """
        def updateStack():
            layer.applyToStack()
            targetImage.onImageChanged()
        """

        # buttons
        pushButton1 = QPushButton("Reset Curve")
        pushButton1.setMinimumSize(1, 1)
        pushButton1.setGeometry(80, 20, 100, 30)  # x,y,w,h
        pushButton1.adjustSize()
        pushButton1.clicked.connect(onResetCurve)
        self.graphicsScene.addWidget(pushButton1)
        pushButton2 = QPushButton("Reset All Curves")
        pushButton2.setMinimumSize(1, 1)
        pushButton2.setGeometry(80, 50, 100, 30)  # x,y,w,h
        pushButton2.adjustSize()
        pushButton2.clicked.connect(onResetAllCurves)
        self.graphicsScene.addWidget(pushButton2)
        """
        pushButton3 = QPushButton("Update Top Layers")
        pushButton3.setObjectName("btn_reset_channel")
        pushButton3.setMinimumSize(1, 1)
        pushButton3.setGeometry(100, 80, 80, 30)  # x,y,w,h
        pushButton3.adjustSize()
        pushButton3.clicked.connect(updateStack)
        self.graphicsScene.addWidget(pushButton3)
        """
        # options
        options = ['L', 'a', 'b']
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        self.listWidget1.setGeometry(0, 10, self.listWidget1.sizeHintForColumn(0) + 5, self.listWidget1.sizeHintForRow(0) * len(options) + 5)
        self.graphicsScene.addWidget(self.listWidget1)

        # self.options is for convenience only
        self.options = {option: True for option in options}

        def onSelect1(item):
            self.scene().cubicItem.setVisible(False)
            #if item.isSelected():
            if item.text() == 'L':
                self.scene().cubicItem = self.graphicsScene.cubicR
            elif item.text() == 'a':
                self.scene().cubicItem = self.graphicsScene.cubicG
            elif item.text() == 'b':
                self.scene().cubicItem = self.graphicsScene.cubicB

            self.scene().cubicItem.setVisible(True)
            # no need for update, but for color mode RGB.
            # self.scene().onUpdateLUT()

            # draw  histogram
            self.scene().invalidate(QRectF(0.0, -self.scene().axeSize, self.scene().axeSize, self.scene().axeSize),
                                    QGraphicsScene.BackgroundLayer)

        self.listWidget1.onSelect = onSelect1

        # set initial selection to L
        item = self.listWidget1.items[options[0]]
        item.setCheckState(Qt.Checked)
        self.listWidget1.select(item)
    """
    def showEvent(self, e):
        self.mainForm.tableView.setEnabled(False)

    def hideEvent(self, e):
        self.mainForm.tableView.setEnabled(True)
    """
    def drawBackground(self, qp, qrF):
        s = self.graphicsScene.axeSize
        if self.scene().cubicItem.histImg is not None:
            qp.drawPixmap(QRect(0, -s, s, s), QPixmap.fromImage(self.scene().cubicItem.histImg))

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
        #for i in range(3):
            #cubic = cubicItem.readFromStream(inStream)
            #cubics.append(cubic)
        #kwargs = dict(zip(['cubicR', 'cubicG', 'cubicB'], cubics))
        #self.setEntries(sel=sel, **kwargs)
        self.graphicsScene.cubicR.readFromStream(inStream)
        self.graphicsScene.cubicG.readFromStream(inStream)
        self.graphicsScene.cubicB.readFromStream(inStream)
        return inStream

    #unused
    def setEntries(self, sel='', cubicR=None, cubicG=None, cubicB=None):
        listWidget = self.listWidget1
        self.graphicsScene.cubicR = cubicR
        self.graphicsScene.cubicG = cubicG
        self.graphicsScene.cubicB = cubicB
        for r in range(listWidget.count()):
            currentItem = listWidget.item(r)
            if currentItem.text() == sel:
                listWidget.select(currentItem)
        self.repaint()
