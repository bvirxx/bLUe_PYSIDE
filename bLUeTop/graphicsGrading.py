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
import cv2

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QSizePolicy, QLabel

from bLUeCore.bLUeLUT3D import LUT3D
from bLUeGui.colorPatterns import colorWheelChooser
from bLUeGui.gradient import gradient2Img, gradientArray
from bLUeTop.utils import QbLUeSlider

from PySide6.QtCore import Qt

from bLUeGui.graphicsForm import baseForm
from bLUeTop.lutUtils import LUTSIZE


class graphicsFormGrading(baseForm):

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, LUTSize=LUTSIZE, layer=None, parent=None, mainForm=None):
        """
        build a graphicsFormGrading object. The parameter axeSize represents the size of
        the color wheel, border not included (the size of the window is adjusted).

        :param cModel: color Model converter
        :type cModel: cmConverter
        :param targetImage
        :type targetImage:
        :param axeSize: size of the color wheel (default 500)
        :type axeSize:
        :param LUTSize: size of the LUT
        :type LUTSize:
        :param layer: layer of targetImage linked to graphics form
        :type layer:
        :param parent: parent widget
        :type parent:
        :param mainForm:
        :type mainForm:
        :return: graphicsForm3DLUT object
        :rtype:
        """
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            newWindow = graphicsFormGrading(targetImage=targetImage, axeSize=axeSize, LUTSize=LUTSize,
                                              layer=layer, parent=parent, mainForm=mainForm)
            newWindow.setWindowTitle(layer.name)
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
        return newWindow

    def __init__(self, targetImage=None, axeSize=500, LUTSize=LUTSIZE, layer=None, parent=None, mainForm=None):
        """
       :param cModel: color space used by colorPicker, slider2D and colorPicker
       :type cModel: cmConverter object
       :param axeSize: size of the color wheel
       :type axeSize: int
       :param targetImage:
       :type targetImage: imImage
       :param LUTSize:
       :type LUTSize: int
       :param layer: layer of targetImage linked to graphics form
       :type layer : QLayer
       :param parent:
       :type parent:
        """
        super().__init__(targetImage=targetImage, layer=layer, parent=parent)

        # size of gradient sample
        gradW, gradH = 200, 30

        # init LUT3D and convert it to HSV
        self.LUT3Dgrad = LUT3D(None, size=LUTSize, alpha=False)
        aux = self.LUT3Dgrad.LUT3DArray[..., :3].reshape(33*33, 33, 3)
        aux = cv2.cvtColor((aux / 255).astype(np.float32), cv2.COLOR_BGR2HSV)
        self.LUT3D_ori2hsv = aux.reshape(33, 33, 33, 3)

        self.mainForm = mainForm  # used by saveLUT()

        # init array of brightness correction coefficients
        self.brCoeffs = np.ones((255,), dtype=float)

        # Help tag
        self.helpId = "GradingForm"
        self.border = 20
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.size = axeSize

        self.predLabel1 = QLabel()
        self.predLabel1.setText('Shadow thr')
        self.predLabel2 = QLabel()
        self.predLabel2.setText('Highlight thr')
        self.predLabel3 = QLabel()
        self.predLabel3.setText('Overlap')

        # init color wheel choosers
        s = 120
        self.chooser1 = colorWheelChooser(s, s, name='Midtone')
        self.chooser2 = colorWheelChooser(s, s, name='Shadow')
        self.chooser3 = colorWheelChooser(s, s, name='Highlight')

        self.slider1 = QbLUeSlider(Qt.Horizontal)
        hlay1 = QHBoxLayout()
        hlay1.addWidget(self.predLabel1)
        hlay1.addWidget(self.slider1)

        self.slider2 = QbLUeSlider(Qt.Horizontal)
        hlay2 = QHBoxLayout()
        hlay2.addWidget(self.predLabel2)
        hlay2.addWidget(self.slider2)

        self.slider3 = QbLUeSlider(Qt.Horizontal)
        hlay3 = QHBoxLayout()
        hlay3.addWidget(self.predLabel3)
        hlay3.addWidget(self.slider3)

        hlay4 = QHBoxLayout()
        hlay4.addWidget(self.chooser2)
        hlay4.addWidget(self.chooser3)

        def initSlider(slider):
            slider.setMinimum(1)
            slider.setMaximum(254)

        sliders = [self.slider1, self.slider2, self.slider3]

        for s in sliders:
            initSlider(s)

        self.slider1.setSliderPosition(40)
        self.slider2.setSliderPosition(180)
        self.slider3.setSliderPosition(10)

        self.grad = gradientArray([self.chooser2.sampler.currentColor,
                                   self.chooser1.sampler.currentColor,
                                   self.chooser3.sampler.currentColor],
                                  [self.slider1.value() * 2, gradW - self.slider1.value() * 2]
                                  )
        self.gradSample = QLabel()
        self.gradSample.setMinimumSize(gradW, gradH)
        self.gradSample.setScaledContents(True)

        # layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        layout.addWidget(self.chooser1)  # midtones
        layout.setAlignment(self.chooser1, Qt.AlignHCenter)
        for lay in [hlay4, hlay1, hlay2, hlay3]:
            layout.addLayout(lay)
        layout.addWidget(self.gradSample)

        self.setLayout(layout)

        def f():
            self.dataChanged.emit()

        def g():
            """
            Gradient and brightness corrections update slot
            """
            # build gradient array. Its size is truncated to 255.
            # No truncation occurs if shadow_thr + overlap <= highlight_thr and highlight_thr + overlap <= 255
            overlap = self.slider3.value()
            self.grad = gradientArray([self.chooser2.sampler.currentColor, self.chooser2.sampler.currentColor,
                                       self.chooser1.sampler.currentColor, self.chooser1.sampler.currentColor,
                                       self.chooser3.sampler.currentColor, self.chooser3.sampler.currentColor],
                                      [self.slider1.value(),
                                       overlap,
                                       max(0, self.slider2.value() - self.slider1.value() - overlap),
                                       overlap,
                                       max(0, 255 - self.slider2.value() - overlap)]
                                      )
            self.grad = self.grad[:255]

            # update brightness coefficients array
            self.brCoeffs[:self.slider1.value()] = self.chooser2.getBr(self.chooser2.brSlider.value())
            self.brCoeffs[self.slider1.value():self.slider2.value()] = self.chooser1.getBr(self.chooser1.brSlider.value())
            self.brCoeffs[self.slider2.value():] = self.chooser3.getBr(self.chooser3.brSlider.value())

            # update gradient image
            img = gradient2Img(self.grad)
            self.gradSample.setPixmap(QPixmap.fromImage(img))

        for s in sliders:
            s.sliderReleased.connect(f)
            s.sliderMoved.connect(g)

        for chooser in [self.chooser1, self.chooser2, self.chooser3]:
            chooser.sampler.samplerReleased.connect(f)
            chooser.sampler.samplerReleased.connect(g)
            chooser.sampler.colorChanged.connect(g)
            chooser.brSlider.sliderReleased.connect(g)
            chooser.brSlider.sliderReleased.connect(f)

        self.adjustSize()
        self.setWhatsThis(
            """<b>Color Grading</b><br>
            Use shadows, midtones and highlights color wheels to pick the 3 corresponding colors.
            Use sliders to choose the shadows and highlights thresholds.<br>
            Your choices are reflected by the gradient image shown at the bottom<br>. Overlap between colors
            is controlled by the <i>overlap</i> slider.<br> 
            the three small sliders below the color wheels allow you to selectively correct the brightnesses.
            """
        )  # end setWhatsThis

    def updateLayer(self):
        """
        data changed slot
        """
        layer = self.layer
        layer.applyToStack()
        layer.parentImage.onImageChanged()

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [QbLUeSlider, colorWheelChooser]:
                d[a] = obj.__getstate__()
        return d

    def __setstate__(self, d):
        # prevent multiple updates
        try:
            self.dataChanged.disconnect()
        except RuntimeError:
            pass
        for name in d['state']:
            obj = getattr(self, name, None)
            if type(obj) in [QbLUeSlider, colorWheelChooser]:
                obj.__setstate__(d['state'][name])
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()
