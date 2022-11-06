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
from PySide2.QtWidgets import QGraphicsScene, QPushButton, QGridLayout
from PySide2.QtCore import Qt, QRectF

from bLUeGui.const import channelValues
from bLUeGui.graphicsSpline import activeCubicSpline, graphicsCurveForm
from bLUeTop.utils import optionsWidget


class graphicsHspbForm(graphicsCurveForm):
    """
    Form for HSV/HSpB curves
    """

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, colorModel='HSV'):
        super().__init__(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        graphicsScene = self.scene()
        graphicsScene.colorModel = colorModel
        # for the sake of simplicity attributes are still named cubicR, cubicG, cubicB instead of cubicH, cubicS, cubicV
        # hue curve init.
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicR = cubic
        cubic.channel = channelValues.Hue
        cubic.initFixedPoints()

        # sat curve init.
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicG = cubic
        cubic.channel = channelValues.Sat
        cubic.initFixedPoints()

        # brightness curve init.
        cubic = activeCubicSpline(axeSize)
        graphicsScene.addItem(cubic)
        graphicsScene.cubicB = cubic
        cubic.channel = channelValues.Br
        cubic.initFixedPoints()

        # init histograms
        self.updateHists()

        # set current curve to sat
        graphicsScene.cubicItem = graphicsScene.cubicG
        graphicsScene.cubicItem.setVisible(True)

        # buttons
        pushButton1 = QPushButton("Reset Current")
        pushButton1.clicked.connect(self.resetCurve)
        pushButton2 = QPushButton("Reset All")
        pushButton2.clicked.connect(self.resetAllCurves)
        # options
        options = ['H', 'S', 'B']
        self.listWidget1 = optionsWidget(options=options, exclusive=True)
        self.listWidget1.setGeometry(0, 0, self.listWidget1.sizeHintForColumn(0) + 5,
                                     self.listWidget1.sizeHintForRow(0) * len(options) + 5)

        # selection changed handler
        curves = [graphicsScene.cubicR, graphicsScene.cubicG, graphicsScene.cubicB]
        curveDict = dict(zip(options, curves))

        def onSelect1(item):
            self.scene().cubicItem.setVisible(False)
            self.scene().cubicItem = curveDict[item.text()]
            self.scene().cubicItem.setVisible(True)
            # draw  histogram
            self.scene().invalidate(QRectF(0.0, -self.scene().axeSize, self.scene().axeSize,
                                           self.scene().axeSize), QGraphicsScene.BackgroundLayer)

        self.listWidget1.onSelect = onSelect1

        # layout
        gl = QGridLayout()
        container = self.addCommandLayout(gl)
        gl.addWidget(self.listWidget1, 0, 0, 2, 1)
        for i, button in enumerate([pushButton1, pushButton2]):
            gl.addWidget(button, i, 1)
        container.adjustSize()
        self.setViewportMargins(0, 0, 0, container.height() + 15)

        self.setWhatsThis("""<b>HSV curves</b><br>""" + self.whatsThis())

        for item in [self.scene().cubicR, self.scene().cubicG, self.scene().cubicB]:
            item.curveChanged.sig.connect(self.dataChanged)

        self.setDefaults()

    def updateLayer(self):
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def setDefaults(self):
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        # set initial selection to Saturation
        item = self.listWidget1.items['S']
        item.setCheckState(Qt.Checked)
        self.listWidget1.select(item)
        self.dataChanged.connect(self.updateLayer)

    def drawBackground(self, qp, qrF):
        graphicsScene = self.scene()
        s = graphicsScene.axeSize
        if graphicsScene.cubicItem.histImg is not None:
            qp.drawImage(QRect(0, -s, s, s), graphicsScene.cubicItem.histImg)

    def updateHist(self, curve, redraw=True):
        """
        Updates the channel histogram displayed under the curve.

        :param curve:
        :type curve:
        :param redraw:
        :type redraw:

        """
        sc = self.scene()
        if curve is sc.cubicR:
            curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize,
                                                          bgColor=sc.bgColor, range=(0, 255),
                                                          # opencv convention for 8 bits image
                                                          chans=channelValues.Hue, mode=sc.colorModel)
        elif curve is sc.cubicG:
            curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize,
                                                          bgColor=sc.bgColor, range=(0, 255),
                                                          # opencv convention for 8 bits image
                                                          chans=channelValues.Sat, mode=sc.colorModel)
        elif curve is sc.cubicB:
            curve.histImg = sc.layer.inputImg().histogram(size=sc.axeSize,
                                                          bgColor=sc.bgColor, range=(0, 255),
                                                          # opencv convention for 8 bits image
                                                          chans=channelValues.Br, mode=sc.colorModel)
        # Force to redraw histogram
        if redraw:
            sc.invalidate(QRectF(0.0, -sc.axeSize, sc.axeSize, sc.axeSize),
                          sc.BackgroundLayer)

    def updateHists(self):
        """
        Updates all histograms
        :return:
        :rtype:
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
        Reset the selected curve
        """
        graphicsScene = self.scene()
        graphicsScene.cubicItem.reset()
        self.updateHist(graphicsScene.cubicItem)
        self.dataChanged.emit()

    def resetAllCurves(self):
        """
        Button event handler
        Reset all curves
        """
        graphicsScene = self.scene()
        for cubicItem in [graphicsScene.cubicR, graphicsScene.cubicG, graphicsScene.cubicB]:
            cubicItem.reset()
        self.updateHists()
        self.dataChanged.emit()

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [optionsWidget]:
                d[a] = obj.__getstate__()
        sc = self.scene()
        for a in ['cubicR', 'cubicG', 'cubicB']:
            d[a] = getattr(sc, a).__getstate__()
        return d

    def __setstate__(self, d):
        # prevent multiple updates
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        for name in d['state']:
            obj = getattr(self, name, None)
            if type(obj) in [optionsWidget]:
                obj.__setstate__(d['state'][name])
        sc = self.scene()
        for name in ['cubicR', 'cubicG', 'cubicB']:
            getattr(sc, name).__setstate__(d['state'][name])
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()
