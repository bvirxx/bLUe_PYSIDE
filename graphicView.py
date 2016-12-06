import sys
from PyQt4.QtGui import QApplication, QPainter, QWidget
from PyQt4.QtGui import QGraphicsView, QGraphicsScene, QGraphicsPathItem , QPainterPath, QPainterPathStroker, QColor, QMainWindow, QLabel, QSizePolicy
from PyQt4.QtCore import Qt, QPoint, QPointF
import numpy as np

strokeWidth = 5

def updateScene():
    for item in window.graphicsScene.items():
        item.updatePath()

class myGraphicsPathItem (QGraphicsPathItem):
    def updatePath(self):
        pass

class activePoint(myGraphicsPathItem):
    def __init__(self, x,y):
        super(QGraphicsPathItem, self).__init__()
        self.position = QPointF(x,y)
        self.updatePath()

    def pos(self):
        return self.position

    def updatePath(self):
        qpp = QPainterPath()
        qpp.addEllipse(self.position, 5, 5)
        self.setPath(qpp)

    def mousePressEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        self.position = e.pos()
        updateScene()

    def mouseReleaseEvent(self, e):
        fixedPoints.sort(key=lambda x : x.pos().x())
        if e.lastPos() == e.pos():
            print 'click'


axeSize = 500
fixedPoints = [activePoint(0, 0), activePoint(axeSize / 2, -axeSize / 2), activePoint(axeSize, -axeSize)]


class Bezier(myGraphicsPathItem) :

    def __init__(self):
        super(QGraphicsPathItem, self).__init__()
        self.qpp = QPainterPath()

        #build curve
        self.qpp.lineTo(QPoint(axeSize, -axeSize))
        #for p in fixedPoints :
            #self.qpp.addEllipse(p.pos(),3,3)

        # stroke curve
        stroker=QPainterPathStroker()
        stroker.setWidth(strokeWidth)
        self.mboundingPath = stroker.createStroke(self.qpp)


        #self.qppath1.lineTo(QPoint(10, 10))
        #self.qppath1.closeSubpath()

        self.setPath(self.mboundingPath)

        self.clicked=QPoint(0,0)
        self.selected = False

    def updatePath(self):
        lfixedPoints = fixedPoints

        t = 0.1
        """
        if not self.selected:
            cp = e.pos()
            mvptIndex = 1
        else:
            x, y = e.pos().x(), e.pos().y()
            if x < lfixedPoints[1].pos().x():
                lfixedPoints.insert(1, activePoint(e.pos().x(), e.pos().y()))
                mvptIndex = 1
            else:
                lfixedPoints.insert(2, activePoint(e.pos().x(), e.pos().y()))
                mvptIndex = 2
        """
        mvptIndex = 1
        qpp = QPainterPath()
        qpp1 = QPainterPath()
        qpp.moveTo(0, 0)
        if lfixedPoints[mvptIndex].x() < -lfixedPoints[mvptIndex].y():
            initila = QPointF(-1, 1) * t * lfixedPoints[mvptIndex].y()
        else:
            initila = QPointF(1, -1) * t * lfixedPoints[mvptIndex].y()

        cp = lfixedPoints[mvptIndex - 1].pos() + initila

        qpp.moveTo(lfixedPoints[mvptIndex - 1].pos())

        for i in range(0, len(lfixedPoints) - mvptIndex):
            qpp.quadTo(cp, lfixedPoints[mvptIndex + i].pos())
            # self.qpp.addEllipse(lfixedPoints[mvptIndex + i].pos(), 3, 3)
            qpp1.addEllipse(cp, 3, 3)
            qpp.moveTo(lfixedPoints[mvptIndex + i].pos())
            print 'f', mvptIndex, mvptIndex + i, lfixedPoints[mvptIndex + i].pos(), cp, initila
            cp = 2 * lfixedPoints[mvptIndex + i].pos() - cp

        qpp.moveTo(lfixedPoints[mvptIndex - 1].pos())
        cp = lfixedPoints[mvptIndex - 1].pos() - initila
        for i in range(0, 0):  # range(2, mvptIndex+1):
            qpp.quadTo(cp, lfixedPoints[mvptIndex - i].pos())

            qpp1.addEllipse(cp, 3, 3)
            qpp.moveTo(lfixedPoints[mvptIndex - i].pos())
            print 'b', mvptIndex, mvptIndex - i, lfixedPoints[mvptIndex - i].pos(), cp, initila
            # equation (1-t)^2 *startPoint + 2*t*(1-t)*cp + t^2*endPoint, 0<=t<=1
            cp = 2 * lfixedPoints[mvptIndex - i].pos() - cp

        #stroke path
        stroker = QPainterPathStroker()
        stroker.setWidth(5);
        mboundingPath = QPainterPath(qpp)
        mboundingPath = stroker.createStroke(qpp);
        # self.setPath(mboundingPath + qpp1)
        self.setPath(mboundingPath + qpp1)

    def mousePressEvent(self, e):
        print "clicked"

        self.selected= True


    def mouseMoveEvent(self, e):
        #self.updatePath()
        updateScene()
        return
        lfixedPoints = []
        lfixedPoints.extend(fixedPoints)
        t = 0.1
        if not self.selected :
            cp=e.pos()
            mvptIndex = 1
        else:
            x, y = e.pos().x(), e.pos().y()
            if x < lfixedPoints[1].pos().x():
                lfixedPoints.insert(1, activePoint(e.pos().x(), e.pos().y()))
                mvptIndex=1
            else :
                lfixedPoints.insert(2, activePoint(e.pos().x(), e.pos().y()))
                mvptIndex=2

        self.qpp = QPainterPath()
        self.qpp1=QPainterPath()
        self.qpp.moveTo(0, 0)
        if lfixedPoints[mvptIndex].x() < -lfixedPoints[mvptIndex].y():
            initila = QPointF(-1, 1) * t*lfixedPoints[mvptIndex].y()
        else:
            initila = QPointF(1, -1) * t*lfixedPoints[mvptIndex].y()

        if self.selected :
            cp = lfixedPoints[mvptIndex - 1].pos() + initila

        self.qpp.moveTo(lfixedPoints[mvptIndex - 1].pos())

        for i in range(0, len(lfixedPoints) -mvptIndex):
            self.qpp.quadTo(cp, lfixedPoints[mvptIndex + i].pos())
            #self.qpp.addEllipse(lfixedPoints[mvptIndex + i].pos(), 3, 3)
            self.qpp1.addEllipse(cp, 3,3)
            self.qpp.moveTo(lfixedPoints[mvptIndex + i].pos())
            print 'f', mvptIndex, mvptIndex + i, lfixedPoints[mvptIndex + i].pos(), cp, initila
            cp = 2 * lfixedPoints[mvptIndex + i].pos() - cp


        self.qpp.moveTo(lfixedPoints[mvptIndex-1].pos())
        cp = lfixedPoints[mvptIndex - 1].pos() - initila
        for i in range(0,0) :#range(2, mvptIndex+1):
            self.qpp.quadTo(cp, lfixedPoints[mvptIndex - i].pos())

            self.qpp1.addEllipse(cp, 3, 3)
            self.qpp.moveTo(lfixedPoints[mvptIndex - i].pos())
            print 'b', mvptIndex, mvptIndex - i, lfixedPoints[mvptIndex - i].pos(), cp, initila
            #equation (1-t)^2 *startPoint + 2*t*(1-t)*cp + t^2*endPoint, 0<=t<=1
            cp= 2*lfixedPoints[mvptIndex-i].pos()- cp

        stroker = QPainterPathStroker()
        stroker.setWidth(5);
        self.mboundingPath = QPainterPath(self.qpp)
        self.mboundingPath = stroker.createStroke(self.qpp);
        self.setPath(self.mboundingPath+self.qpp1)

    def mouseReleaseEvent(self, e):
        self.selected = False
        if e.lastPos() == e.pos():
            p=e.pos()
            a=activePoint(p.x(), p.y())
            fixedPoints.append(a)
            fixedPoints.sort(key=lambda z : z.pos().x())
            window.graphicsScene.addItem(a)
            updateScene()
        return
        qpp = QPainterPath()
        qpp.cubicTo(self.clicked, -e.scenePos(),  QPoint(-100, 100))
        self.setPath(qpp)


class graphicsForm(QGraphicsView) :

    @classmethod
    def getNewWindow(cls):
        newwindow = graphicsForm()
        newwindow.setAttribute(Qt.WA_DeleteOnClose)
        newwindow.setWindowTitle('New Window')
        return newwindow

    def __init__(self):
        super(graphicsForm, self).__init__()
        self.graphicsScene = QGraphicsScene();
        self.setScene(self.graphicsScene);

        # draw axes
        item=myGraphicsPathItem()
        qppath = QPainterPath()
        qppath.moveTo(QPoint(0, 0))
        qppath.lineTo(QPoint(axeSize, 0))
        qppath.lineTo(QPoint(axeSize, -axeSize))
        qppath.lineTo(QPoint(0, -axeSize))
        qppath.closeSubpath()
        qppath.lineTo(QPoint(axeSize, -axeSize))
        item.setPath(qppath)
        self.graphicsScene.addItem(item)

        #draw curve
        item = Bezier()
        self.graphicsScene.addItem(item)

        for p in fixedPoints :
            self.graphicsScene.addItem(p)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = graphicsForm.getNewWindow()
    window.show()

    sys.exit(app.exec_())