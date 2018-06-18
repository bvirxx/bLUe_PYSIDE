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
from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QGraphicsView, QSizePolicy, QVBoxLayout, QLabel, QHBoxLayout, QGroupBox

from graphicsLUT import graphicsQuadricForm
from utils import optionsWidget, QbLUeSlider, UDict, QbLUeLabel


class CoBrSatForm (QGraphicsView):

    dataChanged = QtCore.Signal()
    layerTitle = "Cont/Bright/Sat"
    contrastDefault = 0.0
    brightnessDefault = 0.0
    saturationDefault = 0.0

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        wdgt = CoBrSatForm(axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt
    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        self.setStyleSheet('QRangeSlider * {border: 0px; padding: 0px; margin: 0px}')
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize+100)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.layer = layer
        # contrast spline viewer
        self.contrastForm = None
        # options
        optionList1, optionNames1 = ['Multi-Mode', 'CLAHE'], ['Multi-Mode', 'CLAHE']
        self.listWidget1 = optionsWidget(options=optionList1, optionNames=optionNames1, exclusive=True, changed=lambda: self.dataChanged.emit())
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.listWidget1.setStyleSheet("QListWidget {border: 0px;} QListWidget::item {border: 0px; padding-left: 0px;}")
        optionList2, optionNames2 = ['High', 'manualCurve'], ['Preserve Highlights', 'Show Contrast Curve']
        def optionList2Change(item):
            if item.internalName == 'High':
                # force to recalculate the spline
                self.layer.autoSpline = True
            self.dataChanged.emit()
        self.listWidget2 = optionsWidget(options=optionList2, optionNames=optionNames2, exclusive=False, changed=optionList2Change)
        self.listWidget2.checkOption(self.listWidget2.intNames[0])
        self.listWidget2.setStyleSheet("QListWidget {border: 0px;} QListWidget::item {border: 0px; padding-left: 0px;}")
        self.options = UDict(self.listWidget1.options, self.listWidget2.options)

        # contrast slider
        self.sliderContrast = QbLUeSlider(Qt.Horizontal)
        self.sliderContrast.setStyleSheet(QbLUeSlider.bLueSliderDefaultIBWStylesheet)
        self.sliderContrast.setRange(0, 10)
        self.sliderContrast.setSingleStep(1)

        contrastLabel = QbLUeLabel()
        contrastLabel.setMaximumSize(150, 30)
        contrastLabel.setText("Contrast Level")
        contrastLabel.doubleClicked.connect(lambda: self.sliderContrast.setValue(self.contrast2Slider(self.contrastDefault)))

        self.contrastValue = QLabel()
        font = self.contrastValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("100")
        h = metrics.height()
        self.contrastValue.setMinimumSize(w, h)
        self.contrastValue.setMaximumSize(w, h)
        self.contrastValue.setText(str("{:d}".format(self.sliderContrast.value())))

        # contrast changed  event handler
        def contrastUpdate(value):
            self.contrastValue.setText(str("{:d}".format(self.sliderContrast.value())))
            # move not yet terminated or value not modified
            if self.sliderContrast.isSliderDown() or self.slider2Contrast(value) == self.contrastCorrection:
                return
            self.sliderContrast.valueChanged.disconnect()
            self.sliderContrast.sliderReleased.disconnect()
            self.contrastCorrection = self.slider2Contrast(self.sliderContrast.value())
            # force to recalculate the spline
            self.layer.autoSpline = True
            self.dataChanged.emit()
            self.sliderContrast.valueChanged.connect(contrastUpdate)
            self.sliderContrast.sliderReleased.connect(lambda: contrastUpdate(self.sliderContrast.value()))

        self.sliderContrast.valueChanged.connect(contrastUpdate)
        self.sliderContrast.sliderReleased.connect(lambda: contrastUpdate(self.sliderContrast.value()))

        # saturation slider
        self.sliderSaturation = QbLUeSlider(Qt.Horizontal)
        self.sliderSaturation.setStyleSheet(QbLUeSlider.bLueSliderDefaultColorStylesheet)
        self.sliderSaturation.setRange(0, 10)
        self.sliderSaturation.setSingleStep(1)

        saturationLabel = QbLUeLabel()
        saturationLabel.setMaximumSize(150, 30)
        saturationLabel.setText("Saturation")
        saturationLabel.doubleClicked.connect(lambda: self.sliderSaturation.setValue(self.saturation2Slider(self.saturationDefault)))

        self.saturationValue = QLabel()
        font = self.saturationValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("100")
        h = metrics.height()
        self.saturationValue.setMinimumSize(w, h)
        self.saturationValue.setMaximumSize(w, h)
        self.saturationValue.setText(str("{:+d}".format(self.sliderContrast.value())))

        # saturation changed  event handler
        def saturationUpdate(value):
            self.saturationValue.setText(str("{:+d}".format(int(self.slidersaturation2User(self.sliderSaturation.value())))))
            # move not yet terminated or value not modified
            if self.sliderSaturation.isSliderDown() or self.slider2Saturation(value) == self.satCorrection:
                return
            self.sliderSaturation.valueChanged.disconnect()
            self.sliderSaturation.sliderReleased.disconnect()
            self.satCorrection = self.slider2Saturation(self.sliderSaturation.value())
            self.dataChanged.emit()
            self.sliderSaturation.valueChanged.connect(saturationUpdate)
            self.sliderSaturation.sliderReleased.connect(lambda: saturationUpdate(self.sliderSaturation.value()))
        self.sliderSaturation.valueChanged.connect(saturationUpdate)
        self.sliderSaturation.sliderReleased.connect(lambda: saturationUpdate(self.sliderSaturation.value()))

        # brightness slider
        self.sliderBrightness = QbLUeSlider(Qt.Horizontal)
        self.sliderBrightness.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)
        self.sliderBrightness.setRange(0, 10)
        self.sliderBrightness.setSingleStep(1)

        brightnessLabel = QbLUeLabel()
        brightnessLabel.setMaximumSize(150, 30)
        brightnessLabel.setText("Brightness")
        brightnessLabel.doubleClicked.connect(lambda : self.sliderBrightness.setValue(self.brightness2Slider(self.brightnessDefault)))

        self.brightnessValue = QLabel()
        font = self.brightnessValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("100")
        h = metrics.height()
        self.brightnessValue.setMinimumSize(w, h)
        self.brightnessValue.setMaximumSize(w, h)
        self.brightnessValue.setText(str("{:+d}".format(self.sliderContrast.value())))

        # brightness changed  event handler
        def brightnessUpdate(value):
            self.brightnessValue.setText(str("{:+d}".format(int(self.sliderBrightness2User(self.sliderBrightness.value())))))
            # move not yet terminated or value not modified
            if self.sliderBrightness.isSliderDown() or self.slider2Brightness(value) == self.brightnessCorrection:
                return
            self.sliderBrightness.valueChanged.disconnect()
            self.sliderBrightness.sliderReleased.disconnect()
            self.brightnessCorrection = self.slider2Brightness(self.sliderBrightness.value())
            self.dataChanged.emit()
            self.sliderBrightness.valueChanged.connect(brightnessUpdate)
            self.sliderBrightness.sliderReleased.connect(lambda: brightnessUpdate(self.sliderBrightness.value()))

        self.sliderBrightness.valueChanged.connect(brightnessUpdate)
        self.sliderBrightness.sliderReleased.connect(lambda: brightnessUpdate(self.sliderBrightness.value()))

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        gb1 = QGroupBox()
        gb1.setStyleSheet("QGroupBox {border: 1px solid gray; border-radius: 4px}")
        l1 = QVBoxLayout()
        ct = QLabel()
        ct.setText('Contrast')
        l1.addWidget(ct)
        l1.addWidget(self.listWidget1)
        gb1.setLayout(l1)
        l.addWidget(gb1)
        l.addWidget(self.listWidget2)
        l.addWidget(contrastLabel)
        hl = QHBoxLayout()
        hl.addWidget(self.contrastValue)
        hl.addWidget(self.sliderContrast)
        l.addLayout(hl)
        l.addWidget(brightnessLabel)
        hl3 = QHBoxLayout()
        hl3.addWidget(self.brightnessValue)
        hl3.addWidget(self.sliderBrightness)
        l.addLayout(hl3)
        l.addWidget(saturationLabel)
        hl2 = QHBoxLayout()
        hl2.addWidget(self.saturationValue)
        hl2.addWidget(self.sliderSaturation)
        l.addLayout(hl2)
        self.setLayout(l)
        self.adjustSize()
        self.dataChanged.connect(self.updateLayer)
        self.setStyleSheet("QListWidget, QLabel {font : 7pt;}")
        self.setDefaults()
        self.setWhatsThis(
"""
Brightness and saturation are increased or decreased using the corresponding sliders.

Contrast is enhanced using one of these two methods:
  - CLAHE : increases the local contrast.
  - Multi-Mode : increases the local contrast and the contrast between regions of the image.
For both methods the correction is calculated automatically; the contrast slider controls \
the level of the correction.

With Multi-Mode enabled, use option Show Contrast Curve to modify the correction curve.

Sliders can be reset to their default value by double clicking the name of the slider.
"""
                        )
    def setContrastSpline(self, a, b, d, T):
        """
        Updates and displays the contrast spline
        @param a: x_ccordinates
        @type a:
        @param b: y-coordinates
        @type b:
        @param d: tangent slopes
        @type d:
        @param T: spline
        @type T: ndarray dtype=float
        """
        axeSize = 500
        if self.contrastForm is None:
            form = graphicsQuadricForm.getNewWindow(targetImage=None, axeSize=axeSize, layer=self.layer, parent=None, mainForm=None)
            form.setWindowFlags(Qt.WindowStaysOnTopHint)
            form.setAttribute(Qt.WA_DeleteOnClose, on=False)
            form.setWindowTitle('Contrast curve')
            self.contrastForm = form
        else:
            form = self.contrastForm
        form.scene().quadricB.setCurve(a*axeSize,b*axeSize,d,T*axeSize)
        form.showNormal()

    def enableSliders(self):
        self.sliderContrast.setEnabled(True)
        self.sliderSaturation.setEnabled(True)
        self.sliderBrightness.setEnabled(True)

    def setDefaults(self):  #TODO 13/06/18 put all initial connect in setDefaults to minimize updates
        self.listWidget1.unCheckAll()
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.listWidget2.unCheckAll()
        self.listWidget2.checkOption(self.listWidget2.intNames[0])
        self.enableSliders()
        self.contrastCorrection = self.contrastDefault
        self.sliderContrast.setValue(round(self.contrast2Slider(self.contrastCorrection)))
        self.satCorrection = self.saturationDefault
        self.sliderSaturation.setValue(round(self.saturation2Slider(self.satCorrection)))
        self.brightnessCorrection = self.brightnessDefault
        self.sliderBrightness.setValue(round(self.brightness2Slider(self.brightnessCorrection)))
        self.dataChanged.emit()

    def updateLayer(self):
        """
        data changed event handler.
        """
        self.enableSliders()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()
        # enable/disable options relative to multi-mode
        for intname in ['High', 'manualCurve']:
            item = self.listWidget2.items[intname]
            if self.options['Multi-Mode']:
                item.setFlags(item.flags() | Qt.ItemIsEnabled)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
        cf = self.contrastForm
        if cf is None:
            return
        if self.options['manualCurve']:
            cf.showNormal()
        else:
            cf.hide()

    def slider2Contrast(self, v):
        return v / 10

    def contrast2Slider(self, v):
        return v * 10

    def slider2Saturation(self, v):
        return v / 10 - 0.5

    def saturation2Slider(self, v):
        return v * 10 + 5

    def slidersaturation2User(selfself, v):
        return v - 5.0

    def slider2Brightness(self, v):
        return v / 10 - 0.5

    def brightness2Slider(self, v):
        return v * 10 + 5

    def sliderBrightness2User(selfself, v):
        return v - 5.0

    def writeToStream(self, outStream):
        layer = self.layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        outStream.writeQString(self.listWidget1.selectedItems()[0].text())
        outStream.writeInt32(self.sliderContrast.value())
        return outStream

    def readFromStream(self, inStream):
        actionName = inStream.readQString()
        name = inStream.readQString()
        sel = inStream.readQString()
        temp = inStream.readInt32()
        for r in range(self.listWidget1.count()):
            currentItem = self.listWidget1.item(r)
            if currentItem.text() == sel:
                self.listWidget.select(currentItem)
        self.sliderContrast.setValue(temp)
        self.update()
        return inStream