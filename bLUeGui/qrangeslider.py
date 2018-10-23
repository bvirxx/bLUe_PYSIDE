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

###############################
QRangeSlider copyright notice
##############################

Copyright (c) 2011-2012, Ryan Galloway (http://rsgalloway.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of the author nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt, QEvent
from PySide2.QtGui import QColor, QPainter, QFont

DEFAULT_CSS = """
QRangeSlider * {
    border: 0px;
    padding: 0px;
}
QRangeSlider #Head {
    background: #222;
    margin: 2px;
}
QRangeSlider #Span {
    background: #393;
    margin: 2px;
}
QRangeSlider #Span:active {
    background: #282;
}
QRangeSlider #Tail {
    background: #222;
    margin: 2px;
}
QRangeSlider > QSplitter::handle:vertical {
    height: 8px;
}
QRangeSlider > QSplitter::handle:pressed {
    background: #ca5;
}
QRangeSlider > QSplitter::handle {
    background: #393;
}
"""
def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return int(((val - src[0]) / float(src[1]-src[0])) * (dst[1]-dst[0]) + dst[0])

class Filter(QtCore.QObject):
    def eventFilter(self, obj, event):
        if type(obj) == Handle:
            rs = obj.parent().parent().parent()
        else:
            rs = obj.parent().parent()
        if event.type() not in [QEvent.MouseButtonPress, QEvent.MouseButtonRelease]:
            return False
        if event.type() == QEvent.MouseButtonPress:
            rs.pressed = True
            return False
        elif event.type() == QEvent.MouseButtonRelease:
            rs.pressed = False
            rs.rangeDone.emit(*rs.getRange())
        return False

class Ui_Form(object):
    """default range slider form"""
    def setupUi(self, Form):
        Form.setObjectName("QRangeSlider")
        Form.resize(300, 10)
        Form.setStyleSheet(DEFAULT_CSS)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self._splitter = QtWidgets.QSplitter(Form)
        # QT Bug workaround to show hover events on the handle (trigger polish)
        self._splitter.setStyleSheet("QSplitterHandle:hover {}  QSplitter::handle:hover {background-color:red;}")
        self._splitter.setMinimumSize(QtCore.QSize(0, 0))
        self._splitter.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self._splitter.setOrientation(QtCore.Qt.Horizontal)
        self._splitter.setObjectName("splitter")
        self._head = QtWidgets.QGroupBox(self._splitter)
        self._head.setTitle("")
        self._head.setObjectName("Head")
        self._handle = QtWidgets.QGroupBox(self._splitter)
        self._handle.setTitle("")
        self._handle.setObjectName("Span")
        self._tail = QtWidgets.QGroupBox(self._splitter)
        self._tail.setTitle("")
        self._tail.setObjectName("Tail")
        self.gridLayout.addWidget(self._splitter, 0, 0, 1, 1)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        # reset default cursor
        self._splitter.handle(1).setCursor(Qt.ArrowCursor)
        self._splitter.handle(2).setCursor(Qt.ArrowCursor)
        #self._head.setAttribute( Qt.WA_TransparentForMouseEvents)
        #self._tail.setAttribute(Qt.WA_TransparentForMouseEvents)
        #self._handle.setAttribute(Qt.WA_TransparentForMouseEvents)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("QRangeSlider", "QRangeSlider"))

class Element(QtWidgets.QGroupBox):
    def __init__(self, parent, main):
        super().__init__(parent)
        self.main = main

    def setStyleSheet(self, style):
        """redirect style to parent groupbox"""
        self.parent().setStyleSheet(style)

    def textColor(self):
        """text paint color"""
        return getattr(self, '__textColor', QColor(125, 125, 125))

    def setTextColor(self, color):
        """set the text paint color"""
        if type(color) == tuple and len(color) == 3:
            color = QColor(color[0], color[1], color[2])
        elif type(color) == int:
            color = QColor(color, color, color)
        setattr(self, '__textColor', color)

    def paintEvent(self, event):
        """overrides paint event to handle text"""
        qp = QPainter()
        qp.begin(self)
        if self.main.drawValues():
            self.drawText(event, qp)
        qp.end()
###############
FONT_SIZE = 6
###############
class Head(Element):
    """area before the handle"""
    def __init__(self, parent, main):
        super().__init__(parent, main)

    def drawText(self, event, qp):
        qp.setPen(self.textColor())
        qp.setFont(QFont('Arial', FONT_SIZE))
        qp.drawText(event.rect(), QtCore.Qt.AlignLeft, str(self.main.min()))

class Tail(Element):
    def __init__(self, parent, main):
        super().__init__(parent, main)

    def drawText(self, event, qp):
        qp.setPen(self.textColor())
        qp.setFont(QFont('Arial', FONT_SIZE))
        qp.drawText(event.rect(), QtCore.Qt.AlignRight, str(self.main.max()))

class Handle(Element):
    """handle area"""
    def __init__(self, parent, main):
        super().__init__(parent, main)

    def drawText(self, event, qp):
        qp.setPen(self.textColor())
        qp.setFont(QFont('Arial', FONT_SIZE))
        qp.drawText(event.rect(), QtCore.Qt.AlignLeft, str(self.main.start()))
        qp.drawText(event.rect(), QtCore.Qt.AlignRight, str(self.main.end()))

    def mouseMoveEvent(self, event):
        event.accept()
        mx = event.globalX()
        _mx = getattr(self, '__mx', None)
        if not _mx:
            setattr(self, '__mx', mx)
            dx = 0
        else:
            dx = mx - _mx
        setattr(self, '__mx', mx)
        if dx == 0:
            event.ignore()
            return
        elif dx > 0:
            dx = 1
        elif dx < 0:
            dx = -1
        s = self.main.start() + dx
        e = self.main.end() + dx
        if s >= self.main.min() and e <= self.main.max():
            self.main.setRange(s, e)

class QRangeSlider(QtWidgets.QWidget, Ui_Form):
    """
       The QRangeSlider class implements a horizontal range slider widget.
       Inherits QWidget.
       Methods
           * __init__ (self, QWidget parent = None)
           * bool drawValues (self)
           * int end (self)
           * (int, int) getRange (self)
           * int max (self)
           * int min (self)
           * int start (self)
           * setBackgroundStyle (self, QString styleSheet)
           * setDrawValues (self, bool draw)
           * setEnd (self, int end)
           * setStart (self, int start)
           * setRange (self, int start, int end)
           * setSpanStyle (self, QString styleSheet)
       Signals
           * endValueChanged (int)
           * maxValueChanged (int)
           * minValueChanged (int)
           * startValueChanged (int)
       Customizing QRangeSlider
       You can style the range slider as below:
       ::
           QRangeSlider * {
               border: 0px;
               padding: 0px;
           }
           QRangeSlider #Head {
               background: #222;
           }
           QRangeSlider #Span {
               background: #393;
           }
           QRangeSlider #Span:active {
               background: #282;
           }
           QRangeSlider #Tail {
               background: #222;
           }
       Styling the range slider handles follows QSplitter options:
       ::
           QRangeSlider > QSplitter::handle {
               background: #393;
           }
           QRangeSlider > QSplitter::handle:vertical {
               height: 4px;
           }
           QRangeSlider > QSplitter::handle:pressed {
               background: #ca5;
           }

       """
    # Slider boundary values change
    maxValueChanged = QtCore.Signal(int)
    minValueChanged = QtCore.Signal(int)
    # range boundary values change
    startValueChanged = QtCore.Signal(int,int)
    endValueChanged = QtCore.Signal(int,int)
    # mouse released over handle
    rangeDone = QtCore.Signal(int, int)

    _SPLIT_START = 1
    _SPLIT_END = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        #self.setMouseTracking(False)
        self.setMouseTracking(True)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)
        self._head_layout = QtWidgets.QHBoxLayout()
        self._head_layout.setSpacing(0)
        self._head_layout.setContentsMargins(0, 0, 0, 0)
        self._head.setLayout(self._head_layout)
        self.head = Head(self._head, main=self)
        self._head_layout.addWidget(self.head)
        self._handle_layout = QtWidgets.QHBoxLayout()
        self._handle_layout.setSpacing(0)
        self._handle_layout.setContentsMargins(0, 0, 0, 0)
        self._handle.setLayout(self._handle_layout)
        self.handle = Handle(self._handle, main=self)
        self.handle.setTextColor((150, 255, 150))
        self._handle_layout.addWidget(self.handle)
        self._tail_layout = QtWidgets.QHBoxLayout()
        self._tail_layout.setSpacing(0)
        self._tail_layout.setContentsMargins(0, 0, 0, 0)
        self._tail.setLayout(self._tail_layout)
        self.tail = Tail(self._tail, main=self)
        self._tail_layout.addWidget(self.tail)
        self.setMin(0)
        self.setMax(99)
        self.setStart(0)
        self.setEnd(99)
        self.setDrawValues(True)
        for i in [1, 2]:
            self._splitter.handle(i).installEventFilter(Filter(self))
        self.handle.installEventFilter(Filter(self))
        self.pressed = False

    def isSliderDown(self):
        """
        test if a mouse button is pressed over a handle
        @return:
        @rtype: boolean
        """
        return self.pressed

    def min(self):
        return getattr(self, '__min', None)

    def max(self):
        return getattr(self, '__max', None)

    def setMin(self, value):
        setattr(self, '__min', value)
        self.minValueChanged.emit(value)

    def setMax(self, value):
        setattr(self, '__max', value)
        self.maxValueChanged.emit(value)

    def start(self):
        return getattr(self, '__start', None)

    def end(self):
        return getattr(self, '__end', None)

    def _setStart(self, value):
        setattr(self, '__start', value)
        self.startValueChanged.emit(value, getattr(self, '__end', -1))

    def setStart(self, value):
        v = self._valueToPos(value)
        self._splitter.splitterMoved.disconnect()
        self._splitter.moveSplitter(v, self._SPLIT_START)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)
        self._setStart(value)

    def _setEnd(self, value):
        setattr(self, '__end', value)
        self.endValueChanged.emit(getattr(self, '__start', -1), value)

    def setEnd(self, value):
        v = self._valueToPos(value)
        self._splitter.splitterMoved.disconnect()
        self._splitter.moveSplitter(v, self._SPLIT_END)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)
        self._setEnd(value)

    def drawValues(self):
        return getattr(self, '__drawValues', None)

    def setDrawValues(self, draw):
        setattr(self, '__drawValues', draw)

    def getRange(self):
        return (self.start(), self.end())

    def setRange(self, start, end):
        self.setStart(start)
        self.setEnd(end)

    def keyPressEvent(self, event):
        """overrides key press event to move range left and right"""
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            s = self.start()-1
            e = self.end()-1
        elif key == QtCore.Qt.Key_Right:
            s = self.start()+1
            e = self.end()+1
        else:
            event.ignore()
            return
        event.accept()
        if s >= self.min() and e <= self.max():
            self.setRange(s, e)

    def setBackgroundStyle(self, style):
        self._tail.setStyleSheet(style)
        self._head.setStyleSheet(style)

    def setSpanStyle(self, style):
        self._handle.setStyleSheet(style)

    def _valueToPos(self, value):
        """converts slider value to local pixel x coord"""
        return scale(value, (self.min(), self.max()), (0, self.width()))

    def _posToValue(self, xpos):
        """converts local pixel x coord to slider value"""
        return scale(xpos, (0, self.width()), (self.min(), self.max()))

    def _handleMoveSplitter(self, xpos, index):
        """
        splitterMoved handler. Triggers start/endValueChanged signals
        @param xpos:
        @type xpos:
        @param index:
        @type index:
        @return:
        @rtype:
        """
        hw = self._splitter.handleWidth()
        def _lockWidth(widget):
            width = widget.size().width()
            widget.setMinimumWidth(width)
            widget.setMaximumWidth(width)
        def _unlockWidth(widget):
            widget.setMinimumWidth(0)
            widget.setMaximumWidth(16777215)
        v = self._posToValue(xpos)
        if index == self._SPLIT_START:
            _lockWidth(self._tail)
            if v >= self.end():
                return
            offset = -20
            w = xpos + offset
            self._setStart(v)
        elif index == self._SPLIT_END:
            _lockWidth(self._head)
            if v <= self.start():
                return
            offset = -40
            w = self.width() - xpos + offset
            self._setEnd(v)
        _unlockWidth(self._tail)
        _unlockWidth(self._head)
        _unlockWidth(self._handle)
