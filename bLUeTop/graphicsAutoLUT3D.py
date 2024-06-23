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

from PySide6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QSizePolicy, QLabel, QPushButton

from bLUeGui.dialog import save3DLUTDlg, dlgWarn, isfileError, dlgInfo
from bLUeTop import Gui
from bLUeTop.utils import QbLUeSlider

from PySide6.QtCore import Qt

from bLUeGui.graphicsForm import baseForm
from bLUeTop.lutUtils import LUTSIZE


class graphicsFormAuto3DLUT(baseForm):

    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, LUTSize=LUTSIZE, layer=None, parent=None, mainForm=None):
        """
        Build a graphicsForm3DLUT object. The parameter axeSize represents the size of
        the color wheel, border not included (the size of the window is adjusted).

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
            newWindow = graphicsFormAuto3DLUT(targetImage=targetImage, axeSize=axeSize, LUTSize=LUTSize,
                                              layer=layer, parent=parent, mainForm=mainForm)
            newWindow.setWindowTitle(layer.name)
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
        return newWindow

    def __init__(self, targetImage=None, axeSize=500, LUTSize=LUTSIZE, layer=None, parent=None, mainForm=None):
        """
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
        self.mainForm = mainForm  # used by saveLUT()
        # context help tag
        self.helpId = "AutoLUT3DForm"
        self.border = 20
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # self.setMinimumSize(axeSize + 90, axeSize + 250)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.size = axeSize

        self.predLabel1 = QLabel()
        self.predLabel2 = QLabel()
        self.predLabel3 = QLabel()

        def initSlider(slider):
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setSliderPosition(0)

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

        self.exportBtn = QPushButton('Export 3D LUT')

        sliders = [self.slider1, self.slider2, self.slider3]
        for s in sliders:
            initSlider(s)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        self.setLayout(layout)
        for lay in [hlay1, hlay2, hlay3]:
            layout.addLayout(lay)
        layout.addWidget(self.exportBtn)

        def f():
            self.dataChanged.emit()

        def g():
            try:
                filename = save3DLUTDlg(Gui.window)
                self.lut3D.writeToTextFile(filename)
                dlgInfo('3D LUT written')
            except (isfileError, IOError) as e:
                dlgWarn('Export Failure', info=str(e))

        for s in sliders:
            s.sliderReleased.connect(f)

        self.exportBtn.clicked.connect(g)

        self.adjustSize()
        self.setWhatsThis(
            """<b>Auto 3D LUT</b><br>
            Use the sliders to add a personal touch to the image (auto corresponds to central positions).<br>
            """
        )  # end setWhatsThis

    def updateLayer(self):
        """
        data changed slot
        """
        self.exportBtn.setEnabled(False)
        layer = self.layer
        layer.applyToStack()
        layer.parentImage.onImageChanged()
        self.exportBtn.setEnabled(True)

    def __getstate__(self):
        d = {}
        for a in self.__dir__():
            obj = getattr(self, a)
            if type(obj) in [QbLUeSlider]:
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
            if type(obj) in [QbLUeSlider]:
                obj.__setstate__(d['state'][name])
        self.dataChanged.connect(self.updateLayer)
        self.dataChanged.emit()
