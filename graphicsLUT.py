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
from PySide2.QtWidgets import QPushButton

from bLUeGui.graphicsSpline import graphicsCurveForm, activeQuadricSpline, channelValues

class graphicsQuadricForm(graphicsCurveForm) :
    """
    Form for interactive quadratic splines.
    Dynamic attributes are added to the scene in order
    to provide links to arbitrary graphics items:
        self.scene().cubicItem : current active curve
        self.scene().targetImage
        self.scene().layer
        self.scene().bgColor
    """
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        newWindow = graphicsQuadricForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        newWindow.setWindowTitle(layer.name)
        return newWindow
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        graphicsScene = self.scene()
        # init the curve
        quadric = activeQuadricSpline(graphicsScene.axeSize)
        graphicsScene.addItem(quadric)
        graphicsScene.quadricB = quadric
        quadric.channel = channelValues.Br
        quadric.histImg = graphicsScene.layer.histogram(size=graphicsScene.axeSize, bgColor=graphicsScene.bgColor,
                                                       range=(0,255), chans=channelValues.Br, mode='Luminosity')
        quadric.initFixedPoints()
        # set current curve
        graphicsScene.cubicItem = graphicsScene.quadricB
        graphicsScene.cubicItem.setVisible(True)
        self.setWhatsThis(
"""<b>Contrast Curve</b><br>
Drag <b>control points</b> and <b>tangents</b> with the mouse.<br>
<b>Add</b> a control point by clicking on the curve.<br>
<b>Remove</b> a control point by clicking on it.<br>
<b>Zoom</b> with the mouse wheel.<br>
"""
                           )

        def onResetCurve():
            """
            Reset the selected curve
            """
            self.scene().cubicItem.reset()
            #self.scene().onUpdateLUT()
            l = self.scene().layer
            l.applyToStack()
            l.parentImage.onImageChanged()

        # buttons
        pushButton1 = QPushButton("Reset to Auto Curve")
        pushButton1.setGeometry(10, 20, 100, 30)  # x,y,w,h
        pushButton1.adjustSize()
        pushButton1.clicked.connect(onResetCurve)
        graphicsScene.addWidget(pushButton1)

    def drawBackground(self, qp, qrF):
        graphicsScene = self.scene()
        s = graphicsScene.axeSize
        if graphicsScene.cubicItem.histImg is not None:
            qp.drawImage(QRect(0, -s, s, s), graphicsScene.cubicItem.histImg)

    def writeToStream(self, outStream):
        graphicsScene = self.scene()
        layer = graphicsScene.layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        if layer.actionName in ['actionBrightness_Contrast', 'actionCurves_HSpB', 'actionCurves_Lab']:
            outStream.writeQString(self.listWidget1.selectedItems()[0].text())
            graphicsScene.quadricR.writeToStream(outStream)
            graphicsScene.quadricG.writeToStream(outStream)
            graphicsScene.quadricB.writeToStream(outStream)
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
        graphicsScene.quadricR.readFromStream(inStream)
        graphicsScene.quadricG.readFromStream(inStream)
        graphicsScene.quadricB.readFromStream(inStream)
        return inStream