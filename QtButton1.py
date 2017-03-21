from PyQt4.QtCore import Qt, QObject, pyqtSignal, QPoint
from PyQt4.QtGui import QToolButton

"""
override mouse event handlers.
ev type is QMouseEvent
unused in project CS4
"""
class myButton(QToolButton):
    sig = pyqtSignal(QPoint)

    def __init__(self, *args):
        super(QToolButton, self).__init__(*args)
        self.sig.connect(self.test)

    def test(self, p):
        print "gane",p

    def mousePressEvent(self, ev):
        if (ev.buttons() == Qt.LeftButton):
            print ev.pos()
            self.sig.emit(ev.pos())
    """
    def mouseReleaseEvent(self, ev):
        if (ev.button() == Qt.LeftButton):
            print ev.pos()
    """

