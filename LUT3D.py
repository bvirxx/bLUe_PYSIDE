import numpy as np
from PyQt4.QtCore import Qt, QPoint, QPointF

class QPoint3D(object):
    def __init__(self, x,y,z):
        self.x_ =x
        self.y_=y
        self.z_=z

    def x(self):
        return self.x_
    def y(self):
        return self.y_
    def z(self):
        return self.z_

    def __add__(self, other):
        return QPoint3D(self.x_ + other.x_, self.y_ + other.y_, self.z_ + other.z_)

    def __radd__(self, other):
        return QPoint3D(self.x_ + other.x_, self.y_ + other.y_, self.z_ + other.z_)

    def __mul__(self, scalar):
        return QPoint3D(scalar*self.x_, scalar.self.y_, scalar.self.z_)

    def __rmul__(self, scalar):
        return QPoint3D(scalar*self.x_, scalar.self.y_, scalar.self.z_)


LUTSIZE = 17

LUT=np.array([[[(i*16,j*16,k*16) for k in range(LUTSIZE)] for j in range(LUTSIZE)] for i in range(LUTSIZE)])

def interp(i,j,k):
    """
    Trilinear interpolation

                                              k    I12
                                           F1 |---------- D1
                                              |           |
                                       E1 |   |   I11    |
                                          |   |           |
                                          |C0/--------------j
                                          | /      /I22    E0
                                          |/      /
                                          /      /I2
                                      D0 /------/-------F0
                                        /        I21
                                        i


    :param i:
    :param j:
    :param k:
    :return:
    """

    C0 = (i/16 , j/16, k/16)
    D0 = (i/16 +1 , j/16, k/16)   #C0 + QPoint3D(1,0,0)
    E0 = (i/16 , j/16+1, k/16)    #C0 + QPoint3D(0,1,0)
    F0 = (i/16+1, j/16+1, k/16)   # C0 + QPoint3D(1, 1, 0)

    C1= (i/16 +1 , j/16+1, k/16+1)    #C0 + QPoint3D(1,1,1)
    D1 = (i/16, j/16+1, k/16+1)       #C1 - QPoint3D(1, 0, 0)
    E1 = (i/16+1, j/16, k/16+1)     #C1 -  QPoint3D(0, 1, 0)
    F1 =  (i/16, j/16, k/16+1)     #C1 - QPoint3D(1, 1, 0)

    iP=float(i)/16
    jP=float(j)/16
    I1 = (iP,jP,C1[2])
    I2= (iP,jP, C0[2])
    I11 = (C1[0], jP, C1[2])  # C1.x(), C0.y(),  C1.z()  and C1
    I12 = (C0[0], jP, C1[2])  # C0.x(), C0.y(), C1.z() and C0.x(), C1.y(),C1.z()
    I21 = (C1[0], jP, C0[2])  # C1.x(), C0.y(),  C0.z()  and C1.x(), C1.y(),  C0.z()
    I22 = (C0[0], jP, C0[2])  # C0.x(), C0.y(),  C0.z()  and C0.x(), C1.y(),  C0.z()

    alpha = float(j-LUT[E1][1])/(LUT[C1][1]-LUT[E1][1])
    I11Value= LUT[E1] + alpha*(LUT[C1]-LUT[E1])

    alpha = float(j - LUT[F1][1]) / (LUT[D1][1] - LUT[F1][1])
    I12Value = LUT[F1] + alpha * (LUT[D1] - LUT[F1])

    alpha = float(j - LUT[D0][1]) / (LUT[F0][1] - LUT[D0][1])
    I21Value = LUT[D0] + alpha * (LUT[F0] - LUT[D0])

    alpha = float(j - LUT[C0][1]) / (LUT[E0][1] - LUT[C0][1])
    I22Value = LUT[C0] + alpha * (LUT[E0] - LUT[C0])

    alpha = float(i - I11Value[0]) / (I12Value[0] - I11Value[0])
    I1Value = I11Value + alpha * (I12Value - I11Value)

    alpha = float(i - I21Value[0]) / (I22Value[0] - I21Value[0])
    I2Value = I21Value + alpha * (I22Value - I21Value)

    alpha = float(k - I1Value[2]) / (I2Value[2] - I1Value[2])
    IValue = I1Value + alpha * (I2Value - I1Value)

    return IValue

if __name__=='__main__':

    interp(41,25, 3)