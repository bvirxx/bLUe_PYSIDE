import weakref

from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QSlider, QLabel, QHBoxLayout

from bLUeGui.graphicsSpline import baseForm
from utils import QbLUeSlider, QbLUeLabel


class ExpForm (baseForm):
    defaultExpCorrection = 0.0
    DefaultStep = 0.1
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        wdgt = ExpForm(axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        super().__init__(parent=parent)
        #self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        # link back to image layer
        # using weak ref for back links
        if type(layer) in weakref.ProxyTypes:
            self.layer = layer
        else:
            self.layer = weakref.proxy(layer)

        # options
        self.options = None

        # clipLimit slider
        self.sliderExp = QbLUeSlider(Qt.Horizontal)
        self.sliderExp.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)
        self.sliderExp.setTickPosition(QSlider.TicksBelow)
        self.sliderExp.setRange(-20, 20)
        self.sliderExp.setSingleStep(1)

        expLabel = QbLUeLabel()
        expLabel.setMaximumSize(150, 30)
        expLabel.setText("Exposure Correction")
        expLabel.doubleClicked.connect(lambda: self.sliderExp.setValue(self.defaultExpCorrection))

        self.expValue = QLabel()
        font = self.expValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.expValue.setMinimumSize(w, h)
        self.expValue.setMaximumSize(w, h)
        self.expValue.setStyleSheet("QLabel {background-color: white;}")

        # exp done event handler
        def f():
            self.sliderExp.setEnabled(False)
            self.expValue.setText(str("{:+.1f}".format(self.sliderExp.value() * self.DefaultStep)))
            self.onUpdateExposure(self.layer, self.sliderExp.value() * self.DefaultStep)
            self.sliderExp.setEnabled(True)

        # exp value changed slot
        def g():
            self.expValue.setText(str("{:+.1f}".format(self.sliderExp.value() * self.DefaultStep)))

        self.sliderExp.valueChanged.connect(g)
        self.sliderExp.sliderReleased.connect(f)

        self.sliderExp.setValue(self.defaultExpCorrection / self.DefaultStep)
        self.expValue.setText(str("{:+.1f}".format(self.defaultExpCorrection)))

        #layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        l.addWidget(expLabel)
        hl = QHBoxLayout()
        hl.addWidget(self.expValue)
        hl.addWidget(self.sliderExp)
        l.addLayout(hl)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        #l.addStretch(1)
        self.setLayout(l)
        self.adjustSize()
        self.setWhatsThis(
"""<b>Exposure Correction</b>
Multiplicative correction in the linear sRGB color space.<br>
"""
                         )  # end setWhatsThis

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