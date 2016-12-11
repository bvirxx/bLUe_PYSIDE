import sys
from PyQt4.QtGui import QApplication, QPainter, QWidget
from PyQt4.QtGui import QGraphicsView, QGraphicsScene, QGraphicsPathItem , QPainterPath, QPainterPathStroker, QPen, QBrush, QColor, QMainWindow, QLabel, QSizePolicy
from PyQt4.QtCore import Qt, QPoint, QPointF
import numpy as np
from bisect import bisect

strokeWidth = 3
controlPoints =[]
computeControlPoints = True

def buildLUT(lut):
    """

    :param lut:
    :return:
    """
    LUTX = [p.x() for p in lut]

    #build LUT array
    LUTXY = []
    for i in range(256):
        # get smallest index j s.t. LUTX[j] > i (j=len(LUTX) if it exists, j=len(LUTX) otherwise)
        j = bisect(LUTX, i)

        if j < len(lut):
            p1 = lut[j-1]
            p2 = lut[j]
        else:
            p1 = lut[j-2]
            p2 = lut[j-1]

        y=np.interp(i, [p1.x(), p2.x()], [p1.y(), p2.y()])

        LUTXY.append(int(round(-y)))
    print LUTXY
    return LUTXY

def updateScene(grScene):
    """
    Update all curves in the scene.
    """
    for item in grScene.items():
        item.updatePath()

class myGraphicsPathItem (QGraphicsPathItem):
    """
    Base class for GraphicsPathItems
    """
    def updatePath(self):
        pass

class activePoint(myGraphicsPathItem):
    def __init__(self, x,y):
        super(QGraphicsPathItem, self).__init__()
        self.setPen(QPen(QBrush(QColor(0, 0, 255)), 2))
        self.position_ = QPointF(x,y)
        self.moveStart=QPointF(0,0)
        self.updatePath()

    def position(self):
        return self.position_

    def updatePath(self):
        qpp = QPainterPath()
        qpp.addEllipse(self.position_, 5, 5)
        self.setPath(qpp)

    def mousePressEvent(self, e):
        self.moveStart = e.pos()

    def mouseMoveEvent(self, e):
        self.position_ = e.pos()
        updateScene(self.scene())

    def mouseReleaseEvent(self, e):
        fixedPoints.sort(key=lambda p : p.position().x())
        if self.moveStart == e.pos():
            fixedPoints.remove(self)
            window.graphicsScene.removeItem(self)
            for t in tangents :
                if t.contactPoint == self.position() :
                    print "removed"
                    tangents.remove(t)
                    window.graphicsScene.removeItem(t)
            updateScene(self.scene())

class activeTangent(myGraphicsPathItem):
    def __init__(self, controlPoint=QPointF(), contactPoint=QPointF()):
        super(QGraphicsPathItem, self).__init__()
        self.setPen(QPen(QBrush(QColor(255, 0, 0)), 2))
        self.controlPoint = controlPoint
        self.contactPoint = contactPoint
        self.updatePath()

    def updatePath(self):
        qpp = QPainterPath()
        qpp.addEllipse(self.controlPoint, 5, 5)
        qpp.moveTo(self.controlPoint)
        qpp.lineTo(self.contactPoint)
        self.setPath(qpp)

    def mousePressEvent(self, e):
        global computeControlPoints
        computeControlPoints = False

    def mouseMoveEvent(self, e):
        self.controlPoint = e.pos()
        updateScene(self.scene())

    def mouseReleaseEvent(self, e):
        global computeControlPoints
        if e.lastPos() == e.pos():
            print 'tangent click'

        computeControlPoints = True


axeSize = 255
fixedPoints = [activePoint(0, 0), activePoint(axeSize / 2, -axeSize / 2), activePoint(axeSize, -axeSize)]
tangents = []
for i in range(2*len(fixedPoints)):
    tangents.append(activeTangent())
sampleSize = 400
tSample = [ float(i) / sampleSize for i in range(sampleSize+1) ]
tSample1 = np.array([(1-t)**2 for t in tSample])
tSample2=np.array([2*t*(1-t) for t in tSample])
tSample3=np.array([t**2 for t in tSample])
LUT = []
LUTXY=[]

def qBezierLen(p0, p1, p2) :
    """
    Compute the length of a quadratic Bezier curve.
    cf. http://www.malczak.linuxpl.com/blog/quadratic-bezier-curve-length
    :param p0: QPointF starting point
    :param p1: QPointF control point
    :param p2: QpointF end point
    :return: flaot the curve length
    """

    a = QPointF(p0 - 2*p1 + p2)
    b = QPointF(2*(p1 - p0))

    """
    a.x = p0.x() - 2*p1.x() + p2.x()
    a.y = p0.y() - 2*p1.y() + p2.y()
    b.x = 2*p1.x() - 2*p0.x()
    b.y = 2*p1.y() - 2*p0.y()
    """

    A = 4*(a.x()*a.x() + a.y()*a.y())
    B = 4*(a.x()*b.x() + a.y()*b.y())
    C = b.x()*b.x() + b.y()*b.y()

    Sabc = 2*np.sqrt(A+B+C)

    A_2 = np.sqrt(A)
    A_32 = 2 * A* A_2
    C_2 = 2 * np.sqrt(C)
    BA = B / A_2

    return (
               A_32 * Sabc + A_2 * B * (Sabc - C_2) + (4*C*A - B*B) * np.log( (2*A_2 + BA + Sabc ) / ( BA + C_2) )
           ) / (4 * A_32)

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

        t = 0.2
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
        #qpp.moveTo(0, 0)
        if lfixedPoints[mvptIndex].position().x() < -lfixedPoints[mvptIndex].position().y():
            initila = QPointF(-1, 0) * t * lfixedPoints[mvptIndex].position().y()
        else:
            initila = QPointF(0, -1) * t * lfixedPoints[mvptIndex].position().y()

        if computeControlPoints :
            cp = lfixedPoints[mvptIndex - 1].position() + initila
            tangents[0].controlPoint=cp
            tangents[0].contactPoint=lfixedPoints[mvptIndex-1].position()
            #tangents[1].controlPoint = cp
            #tangents[1].contactPoint = lfixedPoints[mvptIndex].position()
            print 'tangent0', tangents[0].contactPoint, tangents[0].controlPoint
        else :
            cp = tangents[0].controlPoint

        qpp.moveTo(lfixedPoints[mvptIndex - 1].position())
        LUT[:]=[]
        for i in range(0, len(lfixedPoints) - mvptIndex):
            #print "initila", initila, lfixedPoints[mvptIndex].position().y()
            qpp.quadTo(cp, lfixedPoints[mvptIndex + i].position())

            #ecart = abs(lfixedPoints[mvptIndex + i].position().x() - lfixedPoints[mvptIndex +i - 1].position().x()) + abs(cp.x())  #take abs and add abs(cp.x())

            lgth = qBezierLen(lfixedPoints[mvptIndex + i - 1].position(), cp, lfixedPoints[mvptIndex + i].position())
            gap = int(sampleSize/lgth)
            idx = np.arange(0, len(tSample1), max(1, gap), dtype=int)
            print 'spread', int(sampleSize/lgth)


            LUT.extend(tSample1[idx] * lfixedPoints[mvptIndex +i - 1].position() + tSample2[idx]* cp + tSample3[idx] * lfixedPoints[mvptIndex + i].position())
            # self.qpp.addEllipse(lfixedPoints[mvptIndex + i].position(), 3, 3)
            qpp1.moveTo(lfixedPoints[mvptIndex + i-1].position())
            qpp1.lineTo(cp)
            qpp1.moveTo(lfixedPoints[mvptIndex + i].position())
            qpp1.lineTo(cp)
            #qpp1.lineTo(lfixedPoints[mvptIndex + i].pos())
            qpp1.addEllipse(cp, 3, 3)
            #tangents[i].setPath(qpp1)
            qpp.moveTo(lfixedPoints[mvptIndex + i].position())
            #print 'f', mvptIndex, mvptIndex + i, lfixedPoints[mvptIndex + i].pos(), cp, initila
            if computeControlPoints and i < len(lfixedPoints) - mvptIndex - 1:
                cp = 2 * lfixedPoints[mvptIndex + i].position() - cp
                tangents[2*(i+1)].controlPoint=cp
                tangents[2*(i+1)].contactPoint=lfixedPoints[mvptIndex+i].position()
                #tangents[2 * (i + 1)+1].controlPoint = cp
                #tangents[2 * (i + 1)+1].contactPoint = lfixedPoints[mvptIndex + i+1].position()
            else:
                cp = tangents[2*(i+1)].controlPoint
        #qpp.moveTo(lfixedPoints[mvptIndex - 1].position())
        #cp = lfixedPoints[mvptIndex - 1].position() - initila
        """
        for i in range(0, 0):  # range(2, mvptIndex+1):
            qpp.quadTo(cp, lfixedPoints[mvptIndex - i].pos())

            qpp1.addEllipse(cp, 3, 3)
            qpp.moveTo(lfixedPoints[mvptIndex - i].pos())
            print 'b', mvptIndex, mvptIndex - i, lfixedPoints[mvptIndex - i].pos(), cp, initila
            # equation (1-t)^2 *startPoint + 2*t*(1-t)*cp + t^2*endPoint, 0<=t<=1
            cp = 2 * lfixedPoints[mvptIndex - i].pos() - cp
        """
        #stroke path
        stroker = QPainterPathStroker()
        stroker.setWidth(5);
        #mboundingPath = QPainterPath(qpp)
        mboundingPath = stroker.createStroke(qpp);
        # self.setPath(mboundingPath + qpp1)
        self.setPath(mboundingPath)
        LUTXY[:] = buildLUT(LUT)
        #print len(LUTX), max([abs(p.x() - q.x()) for p,q in zip(LUT[1:], LUT[:-1])])


    def mousePressEvent(self, e):
        print "clicked"
        self.selected= True

    def mouseMoveEvent(self, e):
        #self.updatePath()
        updateScene(self.scene())
        return
        """
        lfixedPoints = []
        lfixedPoints.extend(fixedPoints)
        t = 0.1
        if not self.selected :
            cp=e.pos()
            mvptIndex = 1
        else:
            x, y = e.pos().x(), e.pos().y()
            if x < lfixedPoints[1].position().x():
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
        """
    def mouseReleaseEvent(self, e):
        self.selected = False
        if e.lastPos() == e.pos():
            #add curve point
            p=e.pos()
            a=activePoint(p.x(), p.y())
            fixedPoints.append(a)
            fixedPoints.sort(key=lambda z : z.position().x())
            c,d=activeTangent(), activeTangent()
            tangents.extend([c,d])
            for x in [a,c,d]:
                window.graphicsScene.addItem(x)
            updateScene(self.scene())

#main class
class graphicsForm(QGraphicsView) :

    @classmethod
    def getNewWindow(cls, *args, **kwargs):
        newwindow = graphicsForm(*args, **kwargs)
        newwindow.setAttribute(Qt.WA_DeleteOnClose)
        newwindow.setWindowTitle('New Window')
        return newwindow

    def __init__(self, *args, **kwargs):
        super(graphicsForm, self).__init__(*args, **kwargs)
        self.graphicsScene = QGraphicsScene()
        self.setScene(self.graphicsScene)

        # draw axes
        item=myGraphicsPathItem()
        item.setPen(QPen(QBrush(QColor(255, 0, 0)), 1, style=Qt.DashLine))
        qppath = QPainterPath()
        qppath.moveTo(QPoint(0, 0))
        qppath.lineTo(QPoint(axeSize, 0))
        qppath.lineTo(QPoint(axeSize, -axeSize))
        qppath.lineTo(QPoint(0, -axeSize))
        qppath.closeSubpath()
        qppath.lineTo(QPoint(axeSize, -axeSize))

        #add axes
        item.setPath(qppath)
        self.graphicsScene.addItem(item)

        #self.graphicsScene.addPath(qppath, QPen(Qt.DashLine))  #create and add QGraphicsPathItem

        #add curve
        item = Bezier()
        self.graphicsScene.addItem(item)

        #add fixed points
        for p in fixedPoints :
            #p.setPen(QPen(QBrush(QColor(0, 0, 255)), 2))
            self.graphicsScene.addItem(p)

        # add tangents
        for p in tangents:
            self.graphicsScene.addItem(p)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = graphicsForm.getNewWindow()
    window.show()

    sys.exit(app.exec_())




"""
float blen(v* p0, v* p1, v* p2)
{
 v a,b;
 a.x = p0->x - 2*p1->x + p2->x;
 a.y = p0->y - 2*p1->y + p2->y;
 b.x = 2*p1->x - 2*p0->x;
 b.y = 2*p1->y - 2*p0->y;
 float A = 4*(a.x*a.x + a.y*a.y);
 float B = 4*(a.x*b.x + a.y*b.y);
 float C = b.x*b.x + b.y*b.y;

 float Sabc = 2*sqrt(A+B+C);
 float A_2 = sqrt(A);
 float A_32 = 2*A*A_2;
 float C_2 = 2*sqrt(C);
 float BA = B/A_2;

 return ( A_32*Sabc +
          A_2*B*(Sabc-C_2) +
          (4*C*A-B*B)*log( (2*A_2+BA+Sabc)/(BA+C_2) )
        )/(4*A_32);
};
"""