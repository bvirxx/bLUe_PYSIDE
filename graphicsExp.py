from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QWidget


class ExpForm (QWidget): # (QGraphicsView): TODO modified 25/06/18 validate
    defaultExpCorrection = 0.0
    DefaultStep = 0.1
    @classmethod
    def getNewWindow(cls, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None):
        wdgt = ExpForm(axeSize=axeSize, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        """
        pushButton = QPushButton('apply', parent=wdgt)
        hLay = QHBoxLayout()
        wdgt.setLayout(hLay)
        hLay.addWidget(pushButton)
        pushButton.clicked.connect(lambda: wdgt.execute())
        """
        return wdgt

    def __init__(self, targetImage=None, axeSize=500, layer=None, parent=None, mainForm=None): # TODO 01/12/17 remove param targetImage
        super().__init__(parent=parent)
        #self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(axeSize, axeSize)
        self.setAttribute(Qt.WA_DeleteOnClose)
        #self.img = targetImage
        self.layer = layer
        #self.defaultClip = self.defaultClipLimit

        # options
        self.options = None

        # clipLimit slider
        self.sliderClip = QSlider(Qt.Horizontal)
        self.sliderClip.setTickPosition(QSlider.TicksBelow)
        self.sliderClip.setRange(-20, 20)
        self.sliderClip.setSingleStep(1)

        tempLabel = QLabel()
        tempLabel.setMaximumSize(150, 30)
        tempLabel.setText("Exposure Correction")

        self.tempValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.tempValue.setMinimumSize(w, h)
        self.tempValue.setMaximumSize(w, h)
        self.tempValue.setStyleSheet("QLabel {background-color: white;}")

        # exp done event handler
        def f():
            self.sliderClip.setEnabled(False)
            self.tempValue.setText(str("{:+.1f}".format(self.sliderClip.value()*self.DefaultStep)))
            self.onUpdateExposure(self.layer, self.sliderClip.value() * self.DefaultStep)
            self.sliderClip.setEnabled(True)

        # exp value changed event handler
        def g():
            self.tempValue.setText(str("{:+.1f}".format(self.sliderClip.value()*self.DefaultStep)))

        self.sliderClip.valueChanged.connect(g)
        self.sliderClip.sliderReleased.connect(f)

        self.sliderClip.setValue(self.defaultExpCorrection / self.DefaultStep)
        self.tempValue.setText(str("{:+.1f}".format(self.defaultExpCorrection )))

        #layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)
        l.addWidget(tempLabel)
        hl = QHBoxLayout()
        hl.addWidget(self.tempValue)
        hl.addWidget(self.sliderClip)
        l.addLayout(hl)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        l.addStretch(1)
        self.setLayout(l)
        self.adjustSize()
        self.setWhatsThis(
"""Exposure correction
"""
                         )  # end setWhatsThis

    def writeToStream(self, outStream):
        layer = self.layer
        outStream.writeQString(layer.actionName)
        outStream.writeQString(layer.name)
        outStream.writeQString(self.listWidget1.selectedItems()[0].text())
        outStream.writeInt32(self.sliderClip.value())
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
        self.sliderClip.setValue(temp)
        self.update()
        return inStream