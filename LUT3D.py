import numpy as np
from PyQt4.QtCore import Qt, QPoint, QPointF
from PyQt4.QtGui import QImage, QColor
#from MarkedImg import mImage, imImage


Pr = 0.299
Pg = 0.587
Pb = 0.114
"""
Pr=0.2126
Pg=0.7152
Pb=0.0722
"""
# interpolated from 3D lUT Creator
Pr=0.2338
Pg=0.6880
Pb=0.0782

"""
Pr=0.79134178
Pg=2.31839104
Pb=0.25510923
"""
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

def rgb2hsv(r,g,b, perceptual = False):

    cMax = max(r,g,b)
    cMin = min(r, g, b)
    delta = cMax - cMin

    # get hue
    if delta == 0:
        H = 0.0
    elif cMax == r:
        H = 60.0 * float(g-b)/delta if g - b >=0 else 360 + 60.0 * float(g-b)/delta
    elif cMax == g:
        H = 60.0 * (2.0 + float(b-r)/delta)
    elif cMax == b:
        H = 60.0 * (4.0 + float(g-b)/delta)
    #get saturation
    S = 0.0 if cMax == 0 else float(delta)/cMax

    # get brightness
    if perceptual:
        V = np.sqrt(0.299*r*r + 0.587*g*g + 0.114*b*b)
        V = V / 255.0
    else:
        V = cMax/255.0

    return H,S,V

def hsv2rgb(h,s,v):

    c = v * s

    slice = int(h)/60

    slice1 = int(slice) % 2

    #x = c * (1 - abs(slice1 - 1))
    x = c * slice1

    if  h < 60 :
        r1, g1, b1 = c, x , 0
    elif h < 120:
        r1, g1, b1 = x, c , 0
    elif h < 180:
        r1, g1, b1 = 0, c, x
    elif h < 240:
        r1, g1, b1 = 0,x,c
    elif h < 300:
        r1, g1, b1 = x, 0, c
    else :
        r1, g1, b1 = c, 0, x

    m = v - c
    r = (r1 + m) * 255
    g = (g1 + m) * 255
    b = (b1 + m) * 255

    return r,g,b

def hsp2rgb(h,s,p):
    """

    :param h: hue in range 0..360
    :param s:
    :param p:
    :return:
    """

    hd, sd, pd = h,s,p
    # m = min(r,g,b) and M = max(r,g,b) s = (M - m) / M, , 1 - s = m/M
    mM = 1.0 - s
    h = h /60.0  # 0<=h<=6

    if s == 1.0:
        if h < 1.0 :  # r > g > b=0
            h = h#/60.0
            r = np.sqrt(p * p / (Pr + Pg * h * h))
            g = r * h
            b = 0.0
        elif h < 2.0:  # g>r>b=0
            h = (-h + 2.0)#/60.0
            g = np.sqrt(p * p / (Pg + Pr * h * h))
            r = g * h
            b = 0.0
        elif h < 3.0: # g>b>r=0
            h = (h - 2.0)#/60.0
            g = np.sqrt(p * p / (Pg + Pb * h * h))
            b = g * h
            r = 0.0
        elif h < 4.0 : # b>g>r=0
            h = (-h + 4.0)# / 60.0
            b = np.sqrt(p * p / (Pb + Pg * h * h))
            g = b * h
            r = 0.0
        elif h < 5.0 :  # b>r>g=0
            h = (h - 4.0)#/60.0
            b = np.sqrt(p * p / (Pb + Pr * h * h))
            r = b * h
            g = 0.0
        else : # r>b>g=0
            h = (-h + 6.0) #/ 60.0
            r = np.sqrt(p * p / (Pr + Pb * h * h))
            b = r * h
            g = 0.0
    else:  # s !=1
        Mm = 1.0 / (1.0 - s)  #Mm >= 1
        if h < 1.0 :  # r > g > b
            h = h#/60.0
            part = 1.0 + h * (Mm - 1.0)  # part >=1 part = g/b
            b = p / np.sqrt(Pr*Mm*Mm + Pg*part*part + Pb)  # b<=p
            r = b * Mm
            g = b + h * (r - b)
        elif h < 2.0: #g>r>b
            h = (-h + 2.0)# / 60.0
            part = 1.0 + h * (Mm - 1.0) #part = r/b
            b = p / np.sqrt(Pg*Mm*Mm + Pr*part*part + Pb)
            g = b * Mm
            r = b + h * (g - b)
        elif h < 3.0: # g>b>r
            h = (h - 2.0)# / 60.0
            part = 1.0 + h * (Mm - 1.0) # part = b/r
            r = p / np.sqrt(Pg*Mm*Mm + Pb*part*part + Pr)
            g = r * Mm
            b = r + h * (g - r)
        elif h < 4.0: # b>g>r
            h = (-h + 4.0)# / 60.0
            part = 1.0 + h * (Mm - 1.0)
            r = p / np.sqrt(Pb*Mm*Mm + Pg*part*part + Pr)
            b = r * Mm
            g = r  + h * (b - r)
        elif h < 5.0: # b>r>g
            h = (h - 4.0)# / 60.0
            part = 1.0 + h * (Mm - 1.0)
            g = p / np.sqrt(Pb*Mm*Mm + Pr*part*part + Pg)
            b = g * Mm
            r = g + h * (b - g)
        else: # r>b>g
            h = (-h + 6.0) #/ 60.0
            part = 1.0 + h * (Mm - 1.0)
            g = p / np.sqrt(Pr*Mm*Mm + Pb*part*part + Pg)
            r = g * Mm
            b = g + h * (r - g)

    if r<0 or g<0 or b<0 :
        print 'neg value found', h,s,p, r,g,b

    pc=Pr*r*r+Pg*g*g+Pb*b*b
    if abs(p*p - pc)> 0.000001:
        print 'hsp2rgb error', p, np.sqrt(pc)
    return min(255,int(round(r*255))), min(255,int(round(g*255))), min(255, int(round(b*255)))

"""
def colorPicker(w,h):

    img = QImage(w,h, QImage.Format_ARGB32)
    cx=w/2
    cy=h/2
    for i in range(w):
        for j in range(h):
            i1=i-cx
            j1=-j+cy
            m = max(abs(i1), abs(j1))
            hue = np.arctan2(j1,i1)*180.0/np.pi + 315
            hue = hue - floor(hue/360.0)*360.0
            #sat = np.sqrt(i1*i1 + j1*j1)/np.sqrt(w*w/2.0)
            #sat = float(m) /cx
            sat = sat = np.sqrt(i1*i1 + j1*j1)/cx
            sat = min(1.0, sat)
            c = QColor(*hsp2rgb(hue,sat,0.45))
            #if i == w-1 and j == 0:#0:#h - 90:
            if hue==0.0:
                r=c.red()
                g=c.green()
                b=c.blue()
                print hue,sat, r,g,b, i, j
                print np.sqrt((Pr*r*r+Pg*g*g+Pb*b*b))/255.0
            img.setPixel(i,j,c.rgb())
    img = imImage(QImg=img)
    return img
"""
def hs2rgbList(h,s):

    return [hsp2rgb(h,s,p/100.0) for p in range(100)]


#LUTSIZE = 17

#LUT=np.array([[[(i*16,j*16,k*16) for k in range(LUTSIZE)] for j in range(LUTSIZE)] for i in range(LUTSIZE)])

def interp(LUT, i,j,k):
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
    i16, j16, k16 = i/16, j/16, k/16

    C0 = (i16 , j16, k16)
    D0 = (i16 +1 , j16, k16)   #C0 + QPoint3D(1,0,0)
    E0 = (i16 , j16+1, k16)    #C0 + QPoint3D(0,1,0)
    F0 = (i16+1, j16+1, k16)   # C0 + QPoint3D(1, 1, 0)

    C1= (i16 +1 , j16+1, k16+1)    #C0 + QPoint3D(1,1,1)
    D1 = (i16, j16+1, k16+1)       #C1 - QPoint3D(1, 0, 0)
    E1 = (i16+1, j16, k16+1)     #C1 -  QPoint3D(0, 1, 0)
    F1 =  (i16, j16, k16+1)     #C1 - QPoint3D(1, 1, 0)

    iP=float(i)/16
    jP=float(j)/16
    kP = float(k) / 16

    I1 = (iP,jP,C1[2])
    I2= (iP,jP, C0[2])
    I11 = (C1[0], jP, C1[2])  # C1.x(), C0.y(),  C1.z()  and C1
    I12 = (C0[0], jP, C1[2])  # C0.x(), C0.y(), C1.z() and C0.x(), C1.y(),C1.z()
    I21 = (C1[0], jP, C0[2])  # C1.x(), C0.y(),  C0.z()  and C1.x(), C1.y(),  C0.z()
    I22 = (C0[0], jP, C0[2])  # C0.x(), C0.y(),  C0.z()  and C0.x(), C1.y(),  C0.z()

    alpha = float(jP-E1[1])/(C1[1]-E1[1])
    I11Value= LUT[E1] + alpha*(LUT[C1]-LUT[E1])

    alpha = float(jP - F1[1]) / (D1[1] - F1[1])
    I12Value = LUT[F1] + alpha * (LUT[D1] - LUT[F1])

    alpha = float(jP - D0[1]) / (F0[1] - D0[1])
    I21Value = LUT[D0] + alpha * (LUT[F0] - LUT[D0])

    alpha = float(jP - C0[1]) / (E0[1] - C0[1])
    I22Value = LUT[C0] + alpha * (LUT[E0] - LUT[C0])

    alpha = float(iP - I11[0]) / (I12[0] - I11[0])
    I1Value = I11Value + alpha * (I12Value - I11Value)

    alpha = float(iP - I21[0]) / (I22[0] - I21[0])
    I2Value = I21Value + alpha * (I22Value - I21Value)

    alpha = float(kP - I1[2]) / (I2[2] - I1[2])
    IValue = I1Value + alpha * (I2Value - I1Value)

    #print "ivalue", IValue, i,j,k
    return IValue

def lutNN(LUT, r,g,b):

    x = 0 if i % 16 < 8 else 1
    y = 0 if j % 16 < 8 else 1
    z = 0 if k % 16 < 8 else 1

    NN = (r/16 + x, g / 16 + y , k /16 + z)

    return NN

if __name__=='__main__':

    interp(41,25, 3)

    print 'hsp', rgb2hsv(200,20,20, perceptual=True)

    print hsp2rgb(*rgb2hsv(200,20,20, perceptual=True))


"""

#define  Pr  .299
#define  Pg  .587
#define  Pb  .114



//  public domain function by Darel Rex Finley, 2006
//
//  This function expects the passed-in values to be on a scale
//  of 0 to 1, and uses that same scale for the return values.
//
//  See description/examples at alienryderflex.com/hsp.html

void RGBtoHSP(
double  R, double  G, double  B,
double *H, double *S, double *P) {

  //  Calculate the Perceived brightness.
  *P=sqrt(R*R*Pr+G*G*Pg+B*B*Pb);

  //  Calculate the Hue and Saturation.  (This part works
  //  the same way as in the HSV/B and HSL systems???.)
  if      (R==G && R==B) {
    *H=0.; *S=0.; return; }
  if      (R>=G && R>=B) {   //  R is largest
    if    (B>=G) {
      *H=6./6.-1./6.*(B-G)/(R-G); *S=1.-G/R; }
    else         {
      *H=0./6.+1./6.*(G-B)/(R-B); *S=1.-B/R; }}
  else if (G>=R && G>=B) {   //  G is largest
    if    (R>=B) {
      *H=2./6.-1./6.*(R-B)/(G-B); *S=1.-B/G; }
    else         {
      *H=2./6.+1./6.*(B-R)/(G-R); *S=1.-R/G; }}
  else                   {   //  B is largest
    if    (G>=R) {
      *H=4./6.-1./6.*(G-R)/(B-R); *S=1.-R/B; }
    else         {
      *H=4./6.+1./6.*(R-G)/(B-G); *S=1.-G/B; }}}



//  public domain function by Darel Rex Finley, 2006
//
//  This function expects the passed-in values to be on a scale
//  of 0 to 1, and uses that same scale for the return values.
//
//  Note that some combinations of HSP, even if in the scale
//  0-1, may return RGB values that exceed a value of 1.  For
//  example, if you pass in the HSP color 0,1,1, the result
//  will be the RGB color 2.037,0,0.
//
//  See description/examples at alienryderflex.com/hsp.html

void HSPtoRGB(
double  H, double  S, double  P,
double *R, double *G, double *B) {

  double  part, minOverMax=1.-S ;

  if (minOverMax>0.) {
    if      ( H<1./6.) {   //  R>G>B
      H= 6.*( H-0./6.); part=1.+H*(1./minOverMax-1.);
      *B=P/sqrt(Pr/minOverMax/minOverMax+Pg*part*part+Pb);
      *R=(*B)/minOverMax; *G=(*B)+H*((*R)-(*B)); }
    else if ( H<2./6.) {   //  G>R>B
      H= 6.*(-H+2./6.); part=1.+H*(1./minOverMax-1.);
      *B=P/sqrt(Pg/minOverMax/minOverMax+Pr*part*part+Pb);
      *G=(*B)/minOverMax; *R=(*B)+H*((*G)-(*B)); }
    else if ( H<3./6.) {   //  G>B>R
      H= 6.*( H-2./6.); part=1.+H*(1./minOverMax-1.);
      *R=P/sqrt(Pg/minOverMax/minOverMax+Pb*part*part+Pr);
      *G=(*R)/minOverMax; *B=(*R)+H*((*G)-(*R)); }
    else if ( H<4./6.) {   //  B>G>R
      H= 6.*(-H+4./6.); part=1.+H*(1./minOverMax-1.);
      *R=P/sqrt(Pb/minOverMax/minOverMax+Pg*part*part+Pr);
      *B=(*R)/minOverMax; *G=(*R)+H*((*B)-(*R)); }
    else if ( H<5./6.) {   //  B>R>G
      H= 6.*( H-4./6.); part=1.+H*(1./minOverMax-1.);
      *G=P/sqrt(Pb/minOverMax/minOverMax+Pr*part*part+Pg);
      *B=(*G)/minOverMax; *R=(*G)+H*((*B)-(*G)); }
    else               {   //  R>B>G
      H= 6.*(-H+6./6.); part=1.+H*(1./minOverMax-1.);
      *G=P/sqrt(Pr/minOverMax/minOverMax+Pb*part*part+Pg);
      *R=(*G)/minOverMax; *B=(*G)+H*((*R)-(*G)); }}
  else {
    if      ( H<1./6.) {   //  R>G>B
      H= 6.*( H-0./6.); *R=sqrt(P*P/(Pr+Pg*H*H)); *G=(*R)*H; *B=0.; }
    else if ( H<2./6.) {   //  G>R>B
      H= 6.*(-H+2./6.); *G=sqrt(P*P/(Pg+Pr*H*H)); *R=(*G)*H; *B=0.; }
    else if ( H<3./6.) {   //  G>B>R
      H= 6.*( H-2./6.); *G=sqrt(P*P/(Pg+Pb*H*H)); *B=(*G)*H; *R=0.; }
    else if ( H<4./6.) {   //  B>G>R
      H= 6.*(-H+4./6.); *B=sqrt(P*P/(Pb+Pg*H*H)); *G=(*B)*H; *R=0.; }
    else if ( H<5./6.) {   //  B>R>G
      H= 6.*( H-4./6.); *B=sqrt(P*P/(Pb+Pr*H*H)); *R=(*B)*H; *G=0.; }
    else               {   //  R>B>G
      H= 6.*(-H+6./6.); *R=sqrt(P*P/(Pr+Pb*H*H)); *B=(*R)*H; *G=0.; }}}


"""