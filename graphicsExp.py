from PySide2.QtCore import Qt
from PySide2.QtGui import QFontMetrics
from PySide2.QtWidgets import QSizePolicy, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QGraphicsView


class ExpForm (QGraphicsView):
    defaultExpCorrection = 0.0
    DefaultStep = 0.1
    @classmethod
    def getNewWindow(cls, targetImage=None, size=500, layer=None, parent=None, mainForm=None):
        wdgt = ExpForm(targetImage=targetImage, size=size, layer=layer, parent=parent, mainForm=mainForm)
        wdgt.setWindowTitle(layer.name)
        """
        pushButton = QPushButton('apply', parent=wdgt)
        hLay = QHBoxLayout()
        wdgt.setLayout(hLay)
        hLay.addWidget(pushButton)
        pushButton.clicked.connect(lambda: wdgt.execute())
        """
        return wdgt

    def __init__(self, targetImage=None, size=500, layer=None, parent=None, mainForm=None):
        super(ExpForm, self).__init__(parent=parent)
        self.targetImage = targetImage
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumSize(size+100, size)  # default width 200 doesn't fit the length of option names
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.img = targetImage
        self.layer = layer
        #self.defaultClip = self.defaultClipLimit
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)

        # f is defined later, but we need to declare it righjt now
        def f():
            pass

        # options
        self.options = None

        # clipLimit slider
        self.sliderClip = QSlider(Qt.Horizontal)
        self.sliderClip.setTickPosition(QSlider.TicksBelow)
        self.sliderClip.setRange(-10, 10)
        self.sliderClip.setSingleStep(1)

        tempLabel = QLabel()
        tempLabel.setMaximumSize(150, 30)
        tempLabel.setText("Exposure Correction")
        l.addWidget(tempLabel)
        hl = QHBoxLayout()
        self.tempValue = QLabel()
        font = self.tempValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("1000 ")
        h = metrics.height()
        self.tempValue.setMinimumSize(w, h)
        self.tempValue.setMaximumSize(w, h)
        self.tempValue.setStyleSheet("QLabel {background-color: white;}")
        hl.addWidget(self.tempValue)
        hl.addWidget(self.sliderClip)
        l.addLayout(hl)
        l.setContentsMargins(20, 0, 20, 25)  # left, top, right, bottom
        l.addStretch(1)
        self.setLayout(l)
        self.adjustSize()

        # temp done event handler
        def f():
            self.sliderClip.setEnabled(False)
            self.tempValue.setText(str("{:+.1f}".format(self.sliderClip.value()*self.DefaultStep)))
            self.onUpdateExposure(self.sliderClip.value() * self.DefaultStep)
            self.sliderClip.setEnabled(True)

        # temp value changed event handler
        def g():
            self.tempValue.setText(str("{:+.1f}".format(self.sliderClip.value()*self.DefaultStep)))
            #self.previewWindow.setPixmap()

        self.sliderClip.valueChanged.connect(g)
        self.sliderClip.sliderReleased.connect(f)

        self.sliderClip.setValue(self.defaultExpCorrection / self.DefaultStep)
        self.tempValue.setText(str("{:+.1f}".format(self.defaultExpCorrection )))

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