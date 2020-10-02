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

from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QGroupBox

from bLUeGui.graphicsSpline import graphicsSplineForm
from bLUeGui.graphicsForm import baseForm
from bLUeTop.utils import optionsWidget, QbLUeSlider, UDict, QbLUeLabel, stateAwareQDockWidget


class CoBrSatForm(baseForm):
    """
    Contrast, Brightness, Saturation adjustment form
    """
    layerTitle = "Cont/Bright/Sat"
    contrastDefault = 0.0
    brightnessDefault = 0.0
    saturationDefault = 0.0

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = CoBrSatForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    @classmethod
    def slider2Contrast(cls, v):
        return v / 10

    @classmethod
    def contrast2Slider(cls, v):
        return v * 10

    @classmethod
    def slider2Saturation(cls, v):
        return v / 100 - 0.5

    @classmethod
    def saturation2Slider(cls, v):
        return v * 100 + 50

    @classmethod
    def slidersaturation2User(cls, v):
        return v - 50

    @classmethod
    def slider2Brightness(cls, v):
        return v / 100 - 0.5

    @classmethod
    def brightness2Slider(cls, v):
        return v * 100 + 50

    @classmethod
    def sliderBrightness2User(cls, v):
        return v - 50

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)
        self.setMinimumSize(axeSize, axeSize+100)
        # contrast spline viewer
        self.contrastForm = None
        # options
        optionList1, optionNames1 = ['Multi-Mode', 'CLAHE'], ['Multi-Mode', 'CLAHE']
        self.listWidget1 = optionsWidget(options=optionList1, optionNames=optionNames1, exclusive=True,
                                         changed=self.dataChanged)
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        optionList2, optionNames2 = ['High', 'manualCurve'], ['Preserve Highlights', 'Show Contrast Curve']

        def optionList2Change(item):
            if item.internalName == 'High':
                # force to recalculate the spline
                self.layer.autoSpline = True
            self.dataChanged.emit()
        self.listWidget2 = optionsWidget(options=optionList2, optionNames=optionNames2, exclusive=False,
                                         changed=optionList2Change)
        self.listWidget2.checkOption(self.listWidget2.intNames[0])
        self.options = UDict((self.listWidget1.options, self.listWidget2.options))

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

        # contrast changed  event handler.
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
        self.sliderSaturation.setRange(0, 100)
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
        self.sliderBrightness.setRange(0, 100)
        self.sliderBrightness.setSingleStep(1)

        brightnessLabel = QbLUeLabel()
        brightnessLabel.setMaximumSize(150, 30)
        brightnessLabel.setText("Brightness")
        brightnessLabel.doubleClicked.connect(lambda: self.sliderBrightness.setValue(self.brightness2Slider(self.brightnessDefault)))

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

        # attributes initialized in setDefaults, declared here
        # for the sake of correctness
        self.contrastCorrection = None    # range
        self.satCorrection = None         # range -0.5..0.5
        self.brightnessCorrection = None  # range -0.5..0.5

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        gb1 = QGroupBox()
        gb1.setTitle('Contrast')
        l1 = QVBoxLayout()
        l1.setAlignment(Qt.AlignTop)
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
        l.addStretch(1)
        self.setLayout(l)
        self.adjustSize()
        self.setDefaults()
        self.setWhatsThis(
            """<b>Contrast Brightness Saturation</b><br>
            <b>Contrast</b> is enhanced using one of these two methods:<br>
              - <b>CLAHE</b> : increases the local contrast.<br>
              - <b>Multi-Mode</b> : increases the local contrast and the contrast between regions of the image.<br>
            For both methods the contrast slider controls the level of the correction.<br>
            With Multi-Mode enabled, use the option <b>Show Contrast Curve</b> to edit the correction curve and check
            <b>Preserve Highlights</b> for softer highlights.<br>
            <b>Brightness</b> and <b>Saturation</b> corrections are non linear to limit clipping.<br>
            Sliders are <b>reset</b> to their default value by double clicking the name of the slider.<br>
            """
                        )  # end setWhatsThis

    def setContrastSpline(self, a, b, d, T):
        """
        Updates and displays the contrast spline viewer.
        The form is created only once.
        (Cf also rawForm.setCoBrSat.setContrastSpline).
        @param a: x_coordinates
        @type a:
        @param b: y-coordinates
        @type b:
        @param d: tangent slopes
        @type d:
        @param T: spline
        @type T: ndarray dtype=float
        """
        axeSize = 200
        if self.contrastForm is None:
            form = graphicsSplineForm.getNewWindow(targetImage=None, axeSize=axeSize, layer=self.layer, parent=None)
            form.setAttribute(Qt.WA_DeleteOnClose, on=False)
            form.setWindowTitle('Contrast Curve')
            form.setFixedHeight(axeSize+140)
            self.contrastForm = form
            self.contrastForm.optionName = 'manualCurve'
            # dock contrastForm
            window = self.parent().parent()
            dock = self.addSubcontrol(self.parent())  # stateAwareQDockWidget(self.parent())
            dock.setWidget(form)
            dock.setWindowFlags(form.windowFlags())
            dock.setWindowTitle(form.windowTitle())
            dock.setStyleSheet("QGraphicsView{margin: 10px; border-style: solid; border-width: 1px; border-radius: 1px;}")
            window.addDockWidget(Qt.LeftDockWidgetArea, dock)
            self.dock = dock

            # curve changed slot
            def f():
                self.layer.applyToStack()
                self.layer.parentImage.onImageChanged()
            form.scene().quadricB.curveChanged.sig.connect(f)
        else:
            form = self.contrastForm
        # update the curve
        form.scene().setSceneRect(-25, -axeSize-25, axeSize+50, axeSize+50)
        form.scene().quadricB.setCurve(a*axeSize, b*axeSize, d, T*axeSize)
        self.dock.showNormal()

    def updateHists(self):
        """
        Update the histogram displayed under
        the contrast spline.
        """
        if self.contrastForm is not None:
            self.contrastForm.updateHists()

    def enableSliders(self):
        self.sliderContrast.setEnabled(True)
        self.sliderSaturation.setEnabled(True)
        self.sliderBrightness.setEnabled(True)

    def setDefaults(self):
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
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
        self.dataChanged.connect(self.updateLayer)

    def updateLayer(self):
        """
        data changed slot.
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
        # show/hide contrast curve
        cf = getattr(self, 'dock', None)
        if cf is None:
            return
        if self.options['manualCurve'] and self.options['Multi-Mode']:
            cf.showNormal()
        else:
            cf.hide()

