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
import cv2

from collections import OrderedDict
from math import log
from os.path import basename

from PySide2 import QtCore
from PySide2.QtCore import Qt, QPointF
from PySide2.QtGui import QFontMetrics, QBrush, QPolygonF
from PySide2.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QFrame, QGroupBox, QComboBox, QGraphicsPolygonItem
from bLUeGui.graphicsSpline import graphicsSplineForm
from bLUeGui.graphicsForm import baseForm
from bLUeTop.dng import getDngProfileList, getDngProfileDict, dngProfileToneCurve
from bLUeTop.utils import optionsWidget, UDict, QbLUeSlider, stateAwareQDockWidget
from bLUeGui.multiplier import *


class graphicsToneForm(graphicsSplineForm):
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, curveType='quadric'):
        newWindow = graphicsToneForm(targetImage=targetImage, axeSize=axeSize, layer=layer, parent=parent, curveType=curveType)
        newWindow.setWindowTitle(layer.name)
        # init marker
        triangle = QPolygonF()
        s = 10
        triangle.append(QPointF(-s, s))
        triangle.append(QPointF(0, 0))
        triangle.append(QPointF(s, s))
        newWindow.inputMarker = QGraphicsPolygonItem(triangle)
        newWindow.scene().addItem(newWindow.inputMarker)
        newWindow.inputMarker.setBrush(QBrush(Qt.white))
        return newWindow

    def colorPickedSlot(self, x, y, modifiers):
        """
        Move the marker to the position corresponding to
        the color picked on the input image of the active layer.
        (x,y) coordinates are relative to the full size image.
        @param x:
        @type x:
        @param y:
        @type y:
        @param modifiers:
        @type modifiers:
        """
        rImg = self.scene().targetImage.getActiveLayer()
        if rImg.parentImage.useThumb:
            x, y = x//2, y//2
        color = rImg.linearImg.pixelColor(x, y)
        r, g, b = color.red(), color.green(), color.blue()
        h, s, v = cv2.cvtColor((np.array([r, g, b])/255).astype(np.float32)[np.newaxis, np.newaxis, :],
                               cv2.COLOR_RGB2HSV)[0, 0, :]
        self.inputMarker.setPos(v*self.scene().axeSize, 0.0)


class rawForm (baseForm):
    """
    Postprocessing of raw files.
    """
    dataChanged = QtCore.Signal(int)

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None):
        wdgt = rawForm(axeSize=axeSize, targetImage=targetImage, layer=layer, parent=parent)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    @staticmethod
    def slider2Temp(v):
        return 2000 + v * v

    @staticmethod
    def temp2Slider(T):
        return np.sqrt(T - 2000)

    @staticmethod
    def slider2Tint(v):
        return 0.1 + 0.0125 * v  # 0.2 + 0.0125 * v  # wanted range : 0.2...2.5
        # coeff = (self.tempCorrection / 4000 - 1) * 1.2 # experimental formula
        # eturn coeff + 0.01*v

    @staticmethod
    def tint2Slider(t):
        return (t - 0.1) / 0.0125
        # coeff = (self.tempCorrection / 4000 - 1) * 1.2 # experimental formula
        # return (t-coeff)/0.01
        # displayed value

    @staticmethod
    def sliderTint2User(v):
        return v - 75  # ((slider2Tint(v) - 1)*100)

    @staticmethod
    def slider2Exp(v):
        return 2**((v - 50) / 15.0)

    @staticmethod
    def exp2Slider(e):
        return round(15 * np.log2(e) + 50)

    @staticmethod
    def sliderExp2User(v):
        return (v - 50) / 15

    @staticmethod
    def slider2Cont(v):
        return v

    @staticmethod
    def cont2Slider(e):
        return e

    @staticmethod
    def slider2Br(v):
        return (np.power(3, v/50) - 1) / 2

    @staticmethod
    def br2Slider(v):
        return 50 * log(2*v + 1, 3)  # int(round(50.0 * e))

    @staticmethod
    def brSlider2User(v):
        return v - 50

    @staticmethod
    def slider2Sat(v):
        return v - 50

    @staticmethod
    def sat2Slider(e):
        return e + 50

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None):
        super().__init__(layer=layer, targetImage=targetImage, parent=parent)

        #######################################
        # Libraw correspondences:
        # rgb_xyz_matrix is libraw cam_xyz
        # camera_whitebalance is libraw cam_mul
        # daylight_whitebalance is libraw pre_mul
        # dng correspondences:
        # ASSHOTNEUTRAL tag value is (X,Y,Z) =  1 / rawpyObj.camera_whitebalance
        ##########################################
        rawpyObj = layer.parentImage.rawImage

        # constants and as shot values
        self.XYZ2CameraMatrix = rawpyObj.rgb_xyz_matrix[:3, :]
        self.XYZ2CameraInverseMatrix = np.linalg.inv(self.XYZ2CameraMatrix)
        # initial post processing multipliers (as shot)
        m1, m2, m3, m4 = rawpyObj.camera_whitebalance
        self.asShotMultipliers = (m1/m2, 1.0, m3/m2, m4/m2)  # normalization is mandatory : for nef files white balance is around 256
        self.asShotTemp, self.asShotTint = multipliers2TemperatureAndTint(*1 / np.array(self.asShotMultipliers[:3]), self.XYZ2CameraMatrix)
        self.rawMultipliers = self.asShotMultipliers  # rawpyObj.camera_whitebalance # = 1/(dng ASSHOTNEUTRAL tag value)
        self.sampleMultipliers = False
        self.samples = []
        ########################################
        # XYZ-->Camera conversion matrix:
        # Last row is zero for RGB cameras (cf. rawpy and libraw docs).
        # type ndarray, shape (4,3)
        #########################################

        # attributes initialized in setDefaults, declared here for the sake of correctness
        self.tempCorrection, self.tintCorrection, self.expCorrection, self.highCorrection,\
                                                   self.contCorrection, self.satCorrection, self.brCorrection = [None] * 7
        # contrast spline vie (initialized by setContrastSpline)
        self.contrastForm = None
        # tone spline view (initialized by setToneSpline)
        self.toneForm = None
        # dock containers for contrast and tome forms
        self.dockC, self.dockT = None, None
        # options
        optionList0, optionNames0 = ['Auto Brightness', 'Preserve Highlights'], ['Auto Expose', 'Preserve Highlights']
        optionList1, optionNames1 = ['Auto WB', 'Camera WB', 'User WB'], ['Auto', 'Camera (As Shot)', 'User']
        optionList2, optionNames2 = ['cpLookTable', 'cpToneCurve', 'manualCurve'], ['Use Camera Profile Look Table',
                                                                                    'Show Tone Curves', 'Show Contrast Curve']
        self.listWidget1 = optionsWidget(options=optionList0, optionNames=optionNames0, exclusive=False,
                                         changed=lambda: self.dataChanged.emit(1))
        self.listWidget2 = optionsWidget(options=optionList1, optionNames=optionNames1,  exclusive=True,
                                         changed=lambda: self.dataChanged.emit(1))
        self.listWidget3 = optionsWidget(options=optionList2, optionNames=optionNames2, exclusive=False,
                                         changed=lambda: self.dataChanged.emit(2))
        self.options = UDict((self.listWidget1.options, self.listWidget2.options, self.listWidget3.options))
        # display the 'as shot' temperature
        item = self.listWidget2.item(1)
        item.setText(item.text() + ' : %d' % self.asShotTemp)

        # temperature slider
        self.sliderTemp = QbLUeSlider(Qt.Horizontal)
        self.sliderTemp.setStyleSheet(QbLUeSlider.bLueSliderDefaultColorStylesheet)
        self.sliderTemp.setRange(0, 100)
        self.sliderTemp.setSingleStep(1)

        self.tempLabel = QLabel()
        self.tempLabel.setText("Temp")

        self.tempValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("10000")
        h = metrics.height()
        self.tempValue.setMinimumSize(w, h)
        self.tempValue.setMaximumSize(w, h)
        self.tempValue.setText(str("{:.0f}".format(self.slider2Temp(self.sliderTemp.value()))))

        self.sliderTemp.valueChanged.connect(self.tempUpdate)  # signal send new value as parameter
        self.sliderTemp.sliderReleased.connect(lambda: self.tempUpdate(self.sliderTemp.value()))  # signal pass no parameter

        # tint slider
        self.sliderTint = QbLUeSlider(Qt.Horizontal)
        self.sliderTint.setStyleSheet(QbLUeSlider.bLueSliderDefaultIMGColorStylesheet)
        self.sliderTint.setRange(0, 150)

        self.sliderTint.setSingleStep(1)

        self.tintLabel = QLabel()
        self.tintLabel.setText("Tint")

        self.tintValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("100")
        h = metrics.height()
        self.tintValue.setMinimumSize(w, h)
        self.tintValue.setMaximumSize(w, h)
        self.tintValue.setText(str("{:.0f}".format(self.sliderTint2User(self.sliderTint.value()))))

        self.sliderTint.valueChanged.connect(self.tintUpdate)
        self.sliderTint.sliderReleased.connect(lambda: self.tintUpdate(self.sliderTint.value()))  # signal pass no parameter)

        ######################
        # From libraw and dcraw sources:
        # Exposure and brightness are curve transformations.
        # Exposure curve is y = alpha*x, with cubic root ending; it is applied before demosaicing.
        # Brightness is (similar to) y = x**alpha and part of gamma transformation from linear sRGB to RGB.
        # Exposure and brightness both dilate the histogram towards highlights.
        # Exposure dilatation is uniform (homothety), brightness dilataion is
        # maximum for the midtones and the highlghts are preserved.
        # As a consequence, normal workflow begins with the adjustment of exposure,
        # to fill the entire range of the histogram and to adjust the highlights. Next,
        # one adjusts the brightness to put the midtones at the level we want them to be.
        # Cf. https://www.cambridgeincolour.com/forums/thread653.htm
        #####################

        # profile combo
        self.dngDict = self.setCameraProfilesCombo()
        # cameraProfilesCombo index changed event handler

        def cameraProfileUpdate(value):
            self.dngDict = self.cameraProfilesCombo.itemData(value)
            if self.options['cpToneCurve']:
                toneCurve = dngProfileToneCurve(self.dngDict.get('ProfileToneCurve', []))
                self.toneForm.baseCurve = [QPointF(x * axeSize, -y * axeSize) for x, y in zip(toneCurve.dataX, toneCurve.dataY)]
                self.toneForm.update()
            # recompute as shot temp and tint using new profile
            self.asShotTemp, self.asShotTint = multipliers2TemperatureAndTint(*1 / np.array(self.asShotMultipliers[:3]),
                                                                              self.XYZ2CameraMatrix, dngDict=self.dngDict)  # TODO 6/12/19 added keyword dngDict validate
            # display updated as shot temp
            item = self.listWidget2.item(1)
            item.setText(item.text().split(":")[0] + ': %d' % self.asShotTemp)
            # invalidate cache
            self.layer.bufCache_HSV_CV32 = None
            self.dataChanged.emit(2)  # 2 = no postprocessing

        self.cameraProfilesCombo.currentIndexChanged.connect(cameraProfileUpdate)

        # denoising combo
        self.denoiseCombo = QComboBox()
        items = OrderedDict([('Off', 0), ('Medium', 1), ('Full', 2)])
        for key in items:
            self.denoiseCombo.addItem(key, items[key])

        # denoiseCombo index changed event handler
        def denoiseUpdate(value):
            self.denoiseValue = self.denoiseCombo.itemData(value)
            self.dataChanged.emit(1)

        self.denoiseCombo.currentIndexChanged.connect(denoiseUpdate)

        # overexposed area restoration
        self.overexpCombo = QComboBox()
        items = OrderedDict([('Clip', 0), ('Ignore', 1), ('Blend', 2), ('Reconstruct', 3)])
        for key in items:
            self.overexpCombo.addItem(key, items[key])

        # overexpCombo index changed event handler
        def overexpUpdate(value):
            self.overexpValue = self.overexpCombo.itemData(value)
            self.dataChanged.emit(1)

        self.overexpCombo.currentIndexChanged.connect(overexpUpdate)

        # exp slider
        self.sliderExp = QbLUeSlider(Qt.Horizontal)
        self.sliderExp.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)
        self.sliderExp.setRange(0, 100)

        self.sliderExp.setSingleStep(1)

        self.expLabel = QLabel()
        self.expLabel.setText("Exp.")

        self.expValue = QLabel()
        font = self.expValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("+1.0")
        h = metrics.height()
        self.expValue.setMinimumSize(w, h)
        self.expValue.setMaximumSize(w, h)
        self.expValue.setText(str("{:.1f}".format(self.slider2Exp(self.sliderExp.value()))))

        # exp done event handler
        def expUpdate(value):
            self.expValue.setText(str("{:+.1f}".format(self.sliderExp2User(self.sliderExp.value()))))
            # move not yet terminated or value not modified
            if self.sliderExp.isSliderDown() or self.slider2Exp(value) == self.expCorrection:
                return
            try:
                self.sliderExp.valueChanged.disconnect()
                self.sliderExp.sliderReleased.disconnect()
            except RuntimeError:
                pass
            # rawpy: expCorrection range is -2.0...3.0, boiling down to exp_shift range 2**(-2)=0.25...2**3=8.0
            self.expCorrection = self.slider2Exp(self.sliderExp.value())
            self.dataChanged.emit(1)
            self.sliderExp.valueChanged.connect(expUpdate)  # send new value as parameter
            self.sliderExp.sliderReleased.connect(lambda: expUpdate(self.sliderExp.value()))  # signal pass no parameter
        self.sliderExp.valueChanged.connect(expUpdate)  # send new value as parameter
        self.sliderExp.sliderReleased.connect(lambda: expUpdate(self.sliderExp.value()))      # signal pass no parameter

        # brightness slider
        brSlider = QbLUeSlider(Qt.Horizontal)
        brSlider.setRange(1, 101)

        self.sliderExp.setSingleStep(1)

        brSlider.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)

        self.sliderBrightness = brSlider
        brLabel = QLabel()
        brLabel.setText("Bright.")

        self.brValue = QLabel()
        font = self.expValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("+99")
        h = metrics.height()
        self.brValue.setMinimumSize(w, h)
        self.brValue.setMaximumSize(w, h)
        self.brValue.setText(str("{:+d}".format(int(self.brSlider2User(self.sliderBrightness.value())))))

        # brightness done event handler
        def brUpdate(value):
            self.brValue.setText(str("{:+d}".format(int(self.brSlider2User(self.sliderBrightness.value())))))
            # move not yet terminated or value not modified
            if self.sliderBrightness.isSliderDown() or self.slider2Br(value) == self.brCorrection:
                return
            try:
                self.sliderBrightness.valueChanged.disconnect()
                self.sliderBrightness.sliderReleased.disconnect()
            except RuntimeError:
                pass
            self.brCorrection = self.slider2Br(self.sliderBrightness.value())
            self.dataChanged.emit(1)
            self.sliderBrightness.sliderReleased.connect(lambda: brUpdate(self.sliderBrightness.value()))
            self.sliderBrightness.valueChanged.connect(brUpdate)  # send new value as parameter
        self.sliderBrightness.valueChanged.connect(brUpdate)  # send new value as parameter
        self.sliderBrightness.sliderReleased.connect(lambda: brUpdate(self.sliderBrightness.value()))

        # contrast slider
        self.sliderCont = QbLUeSlider(Qt.Horizontal)
        self.sliderCont.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)
        self.sliderCont.setRange(0, 20)

        self.sliderCont.setSingleStep(1)

        self.contLabel = QLabel()
        self.contLabel.setText("Cont.")

        self.contValue = QLabel()
        font = self.contValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("100")
        h = metrics.height()
        self.contValue.setMinimumSize(w, h)
        self.contValue.setMaximumSize(w, h)
        self.contValue.setText(str("{:.0f}".format(self.slider2Cont(self.sliderCont.value()))))

        # cont done event handler
        def contUpdate(value):
            self.contValue.setText(str("{:.0f}".format(self.slider2Cont(self.sliderCont.value()))))
            # move not yet terminated or value not modified
            if self.sliderCont.isSliderDown() or self.slider2Cont(value) == self.tempCorrection:
                return
            try:
                self.sliderCont.valueChanged.disconnect()
                self.sliderCont.sliderReleased.disconnect()
            except RuntimeError:
                pass
            self.contCorrection = self.slider2Cont(self.sliderCont.value())
            self.contValue.setText(str("{:+d}".format(self.contCorrection)))
            # force to recalculate the spline
            self.layer.autoSpline = True
            self.dataChanged.emit(3)  # no postprocessing and no camera profile stuff
            self.sliderCont.valueChanged.connect(contUpdate)  # send new value as parameter
            self.sliderCont.sliderReleased.connect(lambda: contUpdate(self.sliderCont.value()))  # signal has no parameter
        self.sliderCont.valueChanged.connect(contUpdate)  # send new value as parameter
        self.sliderCont.sliderReleased.connect(lambda: contUpdate(self.sliderCont.value()))  # signal has no parameter

        # saturation slider
        self.sliderSat = QbLUeSlider(Qt.Horizontal)
        self.sliderSat.setStyleSheet(QbLUeSlider.bLueSliderDefaultColorStylesheet)
        self.sliderSat.setRange(0, 100)

        self.sliderSat.setSingleStep(1)

        satLabel = QLabel()
        satLabel.setText("Sat.")

        self.satValue = QLabel()
        font = self.satValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("+10")
        h = metrics.height()
        self.satValue.setMinimumSize(w, h)
        self.satValue.setMaximumSize(w, h)
        self.satValue.setText(str("{:+d}".format(self.slider2Sat(self.sliderSat.value()))))

        """sat done event handler"""
        def satUpdate(value):
            self.satValue.setText(str("{:+d}".format(self.slider2Sat(self.sliderSat.value()))))
            # move not yet terminated or value not modified
            if self.sliderSat.isSliderDown() or self.slider2Sat(value) == self.satCorrection:
                return
            try:
                self.sliderSat.valueChanged.disconnect()
                self.sliderSat.sliderReleased.disconnect()
            except RuntimeError:
                pass
            self.satCorrection = self.slider2Sat(self.sliderSat.value())
            self.dataChanged.emit(3)  # no post processing and no camera profile stuff
            self.sliderSat.valueChanged.connect(satUpdate)  # send new value as parameter
            self.sliderSat.sliderReleased.connect(lambda: satUpdate(self.sliderSat.value()))  # signal has no parameter
        self.sliderSat.valueChanged.connect(satUpdate)  # send new value as parameter
        self.sliderSat.sliderReleased.connect(lambda: satUpdate(self.sliderSat.value()))  # signal has no parameter

        self.setStyleSheet("QListWidget, QLabel {font : 7pt;}")

        # layout
        l = QVBoxLayout()
        l.addWidget(self.listWidget3)
        hl01 = QHBoxLayout()
        hl01.addWidget(QLabel('Camera Profile'))
        hl01.addWidget(self.cameraProfilesCombo)
        l.addLayout(hl01)
        hl0 = QHBoxLayout()
        hl0.addWidget(QLabel('Denoising'))
        hl0.addWidget(self.denoiseCombo)
        l.addLayout(hl0)
        hl00 = QHBoxLayout()
        hl00.addWidget(QLabel('Overexp. Restoration'))
        hl00.addWidget(self.overexpCombo)
        l.addLayout(hl00)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.expLabel)
        hl1.addWidget(self.expValue)
        hl1.addWidget(self.sliderExp)
        l.addLayout(hl1)
        hl8 = QHBoxLayout()
        hl8.addWidget(brLabel)
        hl8.addWidget(self.brValue)
        hl8.addWidget(self.sliderBrightness)
        l.addLayout(hl8)
        l.addWidget(self.listWidget1)
        self.listWidget2.setStyleSheet("QListWidget {border: 0px;} QListWidget::item {border: 0px; padding-left: 20px;}")
        vl1 = QVBoxLayout()
        vl1.addWidget(QLabel('White Balance'))
        vl1.addWidget(self.listWidget2)
        gb1 = QGroupBox()
        gb1.setStyleSheet("QGroupBox {border: 1px solid gray; border-radius: 4px}")
        hl2 = QHBoxLayout()
        hl2.addWidget(self.tempLabel)
        hl2.addWidget(self.tempValue)
        hl2.addWidget(self.sliderTemp)
        hl3 = QHBoxLayout()
        hl3.addWidget(self.tintLabel)
        hl3.addWidget(self.tintValue)
        hl3.addWidget(self.sliderTint)
        vl1.addLayout(hl2)
        vl1.addLayout(hl3)
        gb1.setLayout(vl1)
        l.addWidget(gb1)
        hl4 = QHBoxLayout()
        hl4.addWidget(self.contLabel)
        hl4.addWidget(self.contValue)
        hl4.addWidget(self.sliderCont)
        hl7 = QHBoxLayout()
        hl7.addWidget(satLabel)
        hl7.addWidget(self.satValue)
        hl7.addWidget(self.sliderSat)

        # separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        l.addWidget(sep)
        l.addLayout(hl4)
        l.addLayout(hl7)
        l.addStretch(1)
        self.setLayout(l)
        self.adjustSize()
        self.setDefaults()
        self.setWhatsThis(
                    """<b>Development of raw files</b><br>
                    <b>Default settings</b> are a good starting point.<br>
                    A <b>Tone Curve</b> is applied to the raw image prior to postprocessing.<br> 
                    The cuvre can be edited by checking the option
                    <b>Show Tone Curve</b>; this option works best with manual exposure.<br>
                    <b>Contrast</b> correction is based on an automatic algorithm 
                    well suited to multi-mode histograms.<br>
                    <b>Brightness, Contrast</b> and <b>Saturation</b> levels</b> are 
                    adjustable with the correponding sliders.<br>
                    The <b>Contrast Curve</b> can be edited manually by checking 
                    the option <b>Show Contrast Curve</b>.<br>
                    Uncheck <b>Auto Expose</b> to adjust the exposure manually.<br>
                    The <b>OverExp. Rest.</b> slider controls the mode of restoration of overexposed areas. 
                    Valid values are 0 to 3 (0=clip;1=unclip;2=blend;3=rebuild); (with Auto Exposed 
                    checked the mode is clip).<br>
                    """
                        )  # end of setWhatsThis

    def close(self):
        """
        Overrides QWidget.close to
        close toneForm and contrastForm
        @return:
        @rtype: boolean
        """
        delete = self.testAttribute(Qt.WA_DeleteOnClose)
        for attr in ['toneForm', 'contrastForm']:
            form = getattr(self, attr, None)
            if delete and form is not None:
                dock = form.parent()
                dock.setAttribute(Qt.WA_DeleteOnClose)
                form.setAttribute(Qt.WA_DeleteOnClose)
                dock.close()
                form.close()
        return super().close()

    def showToneSpline(self):
        """
        On first call, init and show the Tone Curve form.
        Otherwise, show the form.
        Return True if called for the first time, False otherwise.
        @return:
        @rtype: boolean
        """
        axeSize = 200
        if self.toneForm is None:
            form = graphicsToneForm.getNewWindow(targetImage=self.targetImage, axeSize=axeSize, layer=self.layer, parent=self,
                                                  curveType='cubic')
            form.setWindowFlags(Qt.WindowStaysOnTopHint)
            form.setAttribute(Qt.WA_DeleteOnClose, on=False)
            form.setWindowTitle('Cam Tone Curve')
            form.setButtonText('Reset Curve')
            # get base curve from profile
            toneCurve = dngProfileToneCurve(self.dngDict.get('ProfileToneCurve', []))
            form.baseCurve = [QPointF(x*axeSize, -y*axeSize) for x, y in zip(toneCurve.dataX, toneCurve.dataY)]

            def f():
                layer = self.layer
                layer.bufCache_HSV_CV32 = None
                layer.applyToStack()
                layer.parentImage.onImageChanged()
            form.scene().quadricB.curveChanged.sig.connect(f)
            self.toneForm = form
            dockT = stateAwareQDockWidget(self.parent())
            dockT.setWindowFlags(form.windowFlags())
            dockT.setWindowTitle(form.windowTitle())
            dockT.setStyleSheet(
                "QGraphicsView{margin: 10px; border-style: solid; border-width: 1px; border-radius: 1px;}")
            window = self.parent().parent()
            window.addDockWidget(Qt.LeftDockWidgetArea, dockT)
            self.dockT = dockT
            dockT.setWidget(form)
            showFirst = True
            form.setWhatsThis(
                            """<b>Camera Profile Tone Curve</b><br>
                            The profile curve, if any, is applied as a starting point for user adjustments,
                            after raw post-processing.
                            Its input and output are in <b>linear</b> gamma.
                            The curve is shown in red and cannot be changed.<br>
                            A user curve, shown in black, is editable and is applied right after the
                            former.<br>         
                            """
            )  # end of setWhatsThis
        else:
            form = self.toneForm
            showFirst = False
        form.scene().setSceneRect(-25, -axeSize - 25, axeSize + 50, axeSize + 50)
        self.dockT.showNormal()
        return showFirst

    def setContrastSpline(self, a, b, d, T):
        """
        Updates and displays the contrast spline Form.
        The form is created if needed.
        (Cf. also CoBrStaForm setContrastSpline).
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
            form.setWindowFlags(Qt.WindowStaysOnTopHint)
            form.setAttribute(Qt.WA_DeleteOnClose, on=False)
            form.setWindowTitle('Contrast Curve')

            def f():
                layer = self.layer
                layer.applyToStack()
                layer.parentImage.onImageChanged()
            form.scene().quadricB.curveChanged.sig.connect(f)
            self.contrastForm = form
            dockC = stateAwareQDockWidget(self.parent())
            dockC.setWindowFlags(form.windowFlags())
            dockC.setWindowTitle(form.windowTitle())
            dockC.setStyleSheet("QGraphicsView{margin: 10px; border-style: solid; border-width: 1px; border-radius: 1px;}")
            window = self.parent().parent()
            window.addDockWidget(Qt.LeftDockWidgetArea, dockC)
            self.dockC = dockC
            dockC.setWidget(form)
        else:
            form = self.contrastForm
        # update the curve
        form.scene().setSceneRect(-25, -axeSize - 25, axeSize + 50, axeSize + 50)
        form.scene().quadricB.setCurve(a * axeSize, b * axeSize, d, T * axeSize)
        self.dockC.showNormal()

    # temp changed  event handler
    def tempUpdate(self, value):
        self.tempValue.setText(str("{:.0f}".format(self.slider2Temp(self.sliderTemp.value()))))
        # move not yet terminated or value not modified
        if self.sliderTemp.isSliderDown() or self.slider2Temp(value) == self.tempCorrection:
            return
        try:
            self.sliderTemp.valueChanged.disconnect()
            self.sliderTemp.sliderReleased.disconnect()
        except RuntimeError:
            pass
        self.tempCorrection = self.slider2Temp(self.sliderTemp.value())
        # get multipliers (temperatureAndTint2Multipliers returns the camera neutral)
        multipliers = [1/m for m in temperatureAndTint2Multipliers(self.tempCorrection, 1.0, self.XYZ2CameraMatrix, dngDict=self.dngDict)]  # TODO 6/12/19 added keyword dngDict validate
        multipliers[1] *= self.tintCorrection
        self.rawMultipliers = multipliers
        m = multipliers[1]
        self.rawMultipliers = [self.rawMultipliers[i] / m for i in range(4)]
        self.dataChanged.emit(1)
        self.sliderTemp.valueChanged.connect(self.tempUpdate)  # send new value as parameter
        self.sliderTemp.sliderReleased.connect(lambda: self.tempUpdate(self.sliderTemp.value()))  # signal has no parameter

    # tint change event handler
    def tintUpdate(self, value):
        self.tintValue.setText(str("{:.0f}".format(self.sliderTint2User(self.sliderTint.value()))))
        # move not yet terminated or value not modified
        if self.sliderTint.isSliderDown() or self.slider2Tint(value) == self.tintCorrection:
            return
        try:
            self.sliderTint.valueChanged.disconnect()
            self.sliderTint.sliderReleased.disconnect()
        except RuntimeError:
            pass
        self.tintCorrection = self.slider2Tint(self.sliderTint.value())
        # get multipliers (temperatureAndTint2Multipliers returns the camera neutral)
        multipliers = [1/m for m in temperatureAndTint2Multipliers(self.tempCorrection, 1.0,
                                                                   self.XYZ2CameraMatrix, dngDict=self.dngDict)]  # TODO 6/12/19 added keyword dngDict validate
        multipliers[1] *= self.tintCorrection
        self.rawMultipliers = multipliers
        m = multipliers[1]
        self.rawMultipliers = [self.rawMultipliers[i] / m for i in range(4)]
        self.dataChanged.emit(1)
        self.sliderTint.valueChanged.connect(self.tintUpdate)
        self.sliderTint.sliderReleased.connect(lambda: self.tintUpdate(self.sliderTint.value()))  # signal has no parameter)

    def setRawMultipliers(self, m0, m1, m2, sampling=True):
        mi = min(m0, m1, m2)
        m0, m1, m2 = m0/mi, m1/mi, m2/mi
        self.rawMultipliers = [m0, m1, m2, m1]
        invMultipliers = [1 / self.rawMultipliers[i] for i in range(3)]
        try:
            self.sliderTemp.valueChanged.disconnect()
            self.sliderTint.valueChanged.disconnect()
        except RuntimeError:
            pass
        # get temp and tint
        temp, tint = multipliers2TemperatureAndTint(*invMultipliers, self.XYZ2CameraMatrix)
        self.tintCorrection = tint
        self.sliderTemp.setValue(self.temp2Slider(temp))
        self.sliderTint.setValue(self.tint2Slider(tint))
        self.tempValue.setText(str("{:.0f}".format(self.slider2Temp(self.sliderTemp.value()))))
        self.tintValue.setText(str("{:.0f}".format(self.sliderTint2User(self.sliderTint.value()))))
        self.sliderTemp.valueChanged.connect(self.tempUpdate)
        self.sliderTint.valueChanged.connect(self.tintUpdate)
        self.sampleMultipliers = sampling
        self.dataChanged.emit(1)

    def updateLayer(self, level):
        """
        data changed event handler.
        @param level: 3: redo contrast and saturation, 2: previous + camera profile stuff, 1: all
        @type level: int
        """
        if level == 1:
            # force all
            self.layer.bufCache_HSV_CV32 = None
            self.layer.postProcessCache = None
        elif level == 2:
            # force camera profile stuff
            self.layer.bufCache_HSV_CV32 = None
        elif level == 3:
            # keep the 2 cache buffers
            pass
        # contrast curve
        cf = getattr(self, 'dockC', None)
        if cf is not None:
            if self.options['manualCurve']:
                cf.showNormal()
            else:
                cf.hide()
        # tone curve
        ct = getattr(self, 'dockT', None)
        if ct is not None:
            if self.options['cpToneCurve']:
                ct.showNormal()
            else:
                ct.hide()
        self.enableSliders()
        self.layer.applyToStack()
        self.layer.parentImage.onImageChanged()

    def enableSliders(self):
        useUserWB = self.listWidget2.options["User WB"]
        useUserExp = not self.listWidget1.options["Auto Brightness"]
        self.sliderTemp.setEnabled(useUserWB)
        self.sliderTint.setEnabled(useUserWB)
        self.sliderExp.setEnabled(useUserExp)
        # self.sliderHigh.setEnabled(useUserExp)
        self.tempValue.setEnabled(self.sliderTemp.isEnabled())
        self.tintValue.setEnabled(self.sliderTint.isEnabled())
        self.expValue.setEnabled(self.sliderExp.isEnabled())
        # self.highValue.setEnabled(self.sliderHigh.isEnabled())
        self.tempLabel.setEnabled(self.sliderTemp.isEnabled())
        self.tintLabel.setEnabled(self.sliderTint.isEnabled())
        self.expLabel.setEnabled(self.sliderExp.isEnabled())
        # self.highLabel.setEnabled(self.sliderHigh.isEnabled())

    def setDefaults(self):
        self.dngDict = self.cameraProfilesCombo.itemData(0)
        self.listWidget1.unCheckAll()
        self.listWidget2.unCheckAll()
        self.listWidget1.checkOption(self.listWidget1.intNames[0])
        self.listWidget1.checkOption(self.listWidget1.intNames[1])
        self.listWidget2.checkOption(self.listWidget2.intNames[1])
        self.enableSliders()
        self.denoiseValue = 0  # denoising off
        self.overexpValue = 0  # clip
        self.tempCorrection = self.asShotTemp
        self.tintCorrection = 1.0
        self.expCorrection = 1.0
        # self.highCorrection = 3.0  # restoration of overexposed highlights. 0: clip 1:unclip, 2: blend, 3...: rebuild
        self.contCorrection = 0.0
        # self.noiseCorrection = 0
        self.satCorrection = 0.0
        self.brCorrection = 1.0
        # prevent multiple updates
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        self.sliderTemp.setValue(round(self.temp2Slider(self.tempCorrection)))
        self.sliderTint.setValue(round(self.tint2Slider(self.tintCorrection)))
        self.sliderExp.setValue(self.exp2Slider(self.expCorrection))
        # self.sliderHigh.setValue(self.highCorrection)
        self.sliderCont.setValue(self.cont2Slider(self.contCorrection))
        self.sliderBrightness.setValue(self.br2Slider(self.brCorrection))
        self.sliderSat.setValue(self.sat2Slider(self.satCorrection))
        self.dataChanged.connect(self.updateLayer)

    def setCameraProfilesCombo(self):
        """
        # for each item, text is the filename and data is the corresponding dict
        @return: the currently selected item data
        @rtype: dict
        """
        self.cameraProfilesCombo = QComboBox()
        files = [self.targetImage.filename]
        files.extend(getDngProfileList(self.targetImage.cameraModel()))
        # load profiles
        items = OrderedDict(
            [(basename(f)[:-4] if i > 0 else 'Embedded Profile', getDngProfileDict(f)) for i, f in enumerate(files)])
        # add 'None' and all found profiles for the current camera model: 'None' will be the default selection

        # filter items to eliminate empty entries and
        # add non empty dicts to cameraProfileCombo
        for key in items:
            # filter items[key]
            d = {k: items[key][k] for k in items[key] if items[key][k] != ''}
            if d:
                self.cameraProfilesCombo.addItem(key, d)
        self.cameraProfilesCombo.addItem('None', {})
        self.cameraProfilesCombo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cameraProfilesCombo.setMaximumWidth(150)
        self.cameraProfilesCombo.setStyleSheet("QComboBox QAbstractItemView { min-width: 250px;}")
        # return the currently selected item
        return self.cameraProfilesCombo.itemData(0)
    """
    def writeToStream(self, outStream):
        layer = self.layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        outStream.writeQString(self.listWidget1.selectedItems()[0].text())
        outStream.writeInt32(self.sliderExp.value())
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
        self.sliderExp.setValue(temp)
        self.update()
        return inStream
    """
