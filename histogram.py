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
########################################################################################################################
# This module implements a (nearly) automatic histogram stretching and warping
# algorithm, well suited for multimodal histograms.
# It is based on a paper from Grundland M., A. Dodgson N. (2006) AUTOMATIC CONTRAST ENHANCEMENT BY HISTOGRAM WARPING.
# In: Wojciechowski K., Smolka B., Palus H., Kozera R., Skarbek W., Noakes L. (eds) Computer Vision and
# Graphics. Computational Imaging and Vision, vol 32. Springer, Dordrecht
#########################################################################################################################
import numpy as np

from imgconvert import QImageBuffer
from utils import distribution

"""
def distribution(hist, bins):
    
    This function Interpolates the distribution ("density") of an image, given its histogram.
    It computes the list of Proba(image=k) for k in range(256).
    The interpolation attributes equal probabilities to all points in the same bin.
    @param hist:
    @type hist:
    @param bins:
    @type bins:
    @return: The interpolated distribution
    @rtype: list

    dist = []
    for i in range(256):
        r = np.argmax(bins > i)  # if r>0 bins[r-1]<= i <bins[r]
        # if r==0, i< bins[0] or i >= bins[-1], however the last bin is a
        # closed interval, so we must correct r if i = bins[-1]
        if r == 0:
            if i > bins[-1] or i < bins[0]:
                dist += [0.0]
                continue
            else:
                r = len(bins) - 1
        # calculate the number n of integers contained in the bin.
        lg = bins[r] - bins[r-1]
        n = np.floor(bins[r]) - np.ceil(bins[r-1]) + (1 if (np.floor(bins[r]) != bins[r] or r == len(bins) - 1) else 0)
        # suppose equal probabilities for all these integers
        dist += [hist[r-1] * lg / n]
    if np.sum(dist) != 1:
        print('distribution', np.sum(dist))
    return dist
"""

def gaussianDistribution(x, hist, bins, h):
    """
    Build a kernel density estimation for the distribution (hist, bins)
    by a mixture of gaussian variables. The parameter x is in range 0..255
    and h is the bandwidth.
    @param img:
    @type img:
    @param x:
    @type x:
    @param hist:
    @type hist:
    @param bins:
    @type bins:
    @param h:
    @type h:
    @return:
    @rtype:
    """
    dist = distribution(hist, bins)
    values = np.array(range(256), dtype=np.float)/256
    valuesx = x/256 -values
    valuesx = - valuesx*valuesx/(2*h*h)
    expx = np.exp(valuesx)/ (h*np.sqrt(2*np.pi)) * dist
    dx = np.sum(expx)
    return dx

def distWins(CDF, delta):
    """
    Return the sorted list of the windows of (probability) width 2*delta
    for the distribution (hist,bins). Parameter delta should be in range 0..0.5 : 0<delta<0.5
    @param CDF: CDF table
    @type CDF: ndarray, shape (256,), dtype=np.float
    @param delta: window half width
    @type delta: float
    @return: the list of 3-uples (j, k, i) corresponding to the windows (k+j, k, k+i), Note j is < 0
    @rtype:
    """
    def histDistBin(k):
        # search for a window of (probability) width 2*delta, centered at k.
        i, j = 1, -1
        while  CDF[k + i] - CDF[k] < delta and k+i < len(CDF)-1:
            i += 1
        while CDF[k] - CDF[k + j]< delta and k+j > 0:
            j -= 1
        return j, k, i
    # cut the delta-queues of distribution.
    m, M = np.argmax(CDF > delta), np.argmax(CDF > 1 - delta)
    return [histDistBin(k) for k in range(m, M)]

def valleys(imgBuf, delta):
    """
    Search for valleys (local minima) in the distribution of the image histogram. A valley is a
    window of (probability) width 2*delta, whose central point gives the minimum value.
    The function returns the ordered list of these local minima.

    @param imgBuf:
    @type imgBuf:
    @param delta:
    @type delta:
    @return:
    @rtype:
    """
    hist, bins = np.histogram(imgBuf, bins=100, density=True)

    dist = distribution(hist, bins, 255)
    V = []
    # get windows
    hDB = distWins(dist.CDFTable, delta)
    # search for valleys
    for j, k, i in hDB:
        if np.all([(dist.DTable[l] > dist.DTable[k]) for l in range(k+j, k)]) and np.all([(dist.DTable[l] > dist.DTable[k]) for l in range(k+1, k+i+1)]) :
            V.append(k)
    #V = [0.0]+V+[255.0]
    V = np.array(V, dtype=np.float)
    return V, dist

def interpolationSpline(a, b, d):
    """
    Build a monotonic transformation curve T from [0,1] onto b[0], b[-1],
    as a piecewise rational quadratic interpolation spline to a set of (a[k], b[k]) 2D nodes.
    Coefficients d[k] are the slopes at nodes. The function returns
    the list of T[k/255] for k in range(256).
    a and b must be non decreasing sequences in range 0..1, with a strictly increasing.
    The 3 parameter lists must have the same length.
    for k < a[0], T[k]=b[0] and, for k > a[-1], T[k]=b[-1]
    @param a:
    @type a:
    @param b:
    @type b:
    @param d:
    @type d:
    @return:
    @rtype:
    """
    if np.min(a[1:] - a[:-1]) <= 0:
        raise ValueError('InterpolationSpline : a must be strictly increasing')
    x = np.array(range(256), dtype=np.float)/255
    x = np.clip(x, a[0], a[-1])
    tmp = np.array([[(a[j]> x[i]) for i in range(len(x))] for j in range(len(a))])
    # for each i, get smallest j s.t. a[j] > x[i]
    k = np.argmax(tmp, axis=0)                     # a[k[i]-1]<= x[i] < a[k[i]] if k[i] > 0, and x[i] out of a[0],..a[-1] otherwise
    # k = np.where(k == 0, len(a)-1, k)
    k = np.where(x >= a[-1], len(a) - 1, k)
    #r = (b[k] - b[k-1])/(a[k]-a[k-1])
    r = (b[1:] - b[:-1]) / (a[1:] - a[:-1])       # r[k] = (b[k] - b[k-1]) / (a[k] - a[k-1])
    r = np.concatenate(([0], r))
    t = (x-a[k-1])/(a[k]-a[k-1])                  # t[k] = (x - a[k-1]) / (a[k] - a[k-1]) for x in a[k-1]..a[k]
    assert np.all(t>=0)
    assert np.all(t<=1)
    T = b[k-1] + (r[k]*t*t + d[k-1]*(1-t)*t)*(b[k]-b[k-1]) / (r[k]+(d[k]+d[k-1] - 2*r[k])*(1-t)*t)
    # plot curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    G = T*255
    M = np.int(a[-1]*255)
    ax.plot(T[:M+1]*255)
    ax.plot([i for i in range(256)])
    ax.plot(a*255, b*255, 'ro')
    #ax.plot(a*255, a*255, 'bo')
    fig.savefig('temp.png')
    return T

def warpHistogram(imgBuf, valleyAperture=0.01, warp=0):
    """
    Stretch and warp the distribution of imgBuf.
    The parameter delta controls the (probability) width
    of histogram valleys: it should be >0 and < 0.5
    @param imgBuf: single channel image (luminance), range 0..1
    @type imgBuf: ndarray, shape(h,w), dtype=uint8 or int or float
    @param valleyAperture:
    @type valleyAperture:
    @return: transformed image channel, range 0..1
    @rtype: ndarray same shape as imgBuf, dtype=np.flaot,
    """
    ###################
    # outlier threshold
    tau = 0.01
    ###################
    # get valleys and histogram
    V0, dist = valleys(imgBuf*255, valleyAperture)                    # len(V0) = K-1 is the count of valleys
    print('valleys ', V0)
    V0 = V0/255
    # discard black/white images
    if dist.FInv(0) == dist.FInv(1):
        raise ValueError('Image is uniform: no contrast correction possible')
    # add the distribution end points to the valley array,
    # and calculate the array of modeCenters.
    # each interval [V[k-1], V[k] is called a mode.
    # Modes are viewed as groups of objects with similar lighting conditions,
    # (background and foreground for example).
    # The center points of modeCenters partition the pixel range in regions and
    # each region will be transformed by a single quadratic spline.
    V0 = np.concatenate((dist.FInv(0), V0, dist.FInv(1)))
    modeCenters = (V0[1:] + V0[:-1]) / 2
    # init the array of data points
    a = np.zeros(len(V0)+3, dtype=np.float)                      # len(a) = K+2
    # put modeCenters into a[2:K-1], and count
    # valleys from V[1] to get a[k] = (V[k-1] + V[k])/2, k>=2
    a[2:-2] = modeCenters
    V = np.concatenate(([0], V0))
    # init the array of mappings a[k]-->b[k], add end points
    # and map dynamic range to [0, 1]
    b = a.copy()
    a[0], a[1], a[-2], a[-1] = dist.FInv(0), dist.FInv(tau), dist.FInv(1-tau), dist.FInv(1)
    b[0], b[1], b[-2], b[-1] = 0, tau, 1 - tau, 1
    # discard outliers
    if a[1]<=a[0] or a[1]>=a[2]:
        a[1]= (a[0]+a[2])/2
    if a[-2]>=a[-1] or a[-2]<=a[-3]:
        a[-2] = (a[-1]+a[-3])/2
    if  np.min(a[1:]-a[:-1]) <= 0:
        raise ValueError('warpHistogram : array must be strictly increasing')
    # move b[k] within [v[k-1], v[k]] to equalize the histogram
    # s controls the move
    s = warp
    for k in range(2, len(b)-2):                                # 2..K-1
        b[k] = (dist.F(V[k]) - dist.F(a[k])) * V[k-1] + (dist.F(a[k]) - dist.F(V[k-1])) * V[k]
        b[k] = b[k] / (dist.F(V[k]) - dist.F(V[k-1]))*s  + a[k]*(1-s)     # F(V[k]) - F(V[k-1] >= valleyAperture
    if np.min(b[1:] - b[:-1]) < 0:
        raise ValueError('warpHistogram : array b must be non decreasing')

    # calculate curve slopes at (a[k], b[k]).
    bPlus = (b[:-1] + b[1:])/ 2                                 # bPlus[k] = (b[k] + b[k+1])/ 2
    bPlus = np.concatenate((bPlus, [0.0]))
    bMinus = np.concatenate( ([0.0], bPlus[:-1]))               # bMinus[k] = (b[k] + b[k-1]) / 2 , k>=1

    tmpMid = (dist.FVec(a[:-1]) + dist.FVec(a[1:]))/2                     # tmpMid[k] = (F(a[k]) + F(a[k+1]) / 2
    aPlus = dist.FInv(tmpMid)                                        # aPLus[k] = FInv( (F(a[k] + F(a[k+1])) / 2 )
    aPlus = np.concatenate((aPlus, [1.0]))
    aMinus = np.concatenate(([0.0], aPlus[:-1]))                # aMinus[k] = FInv( (F(a[k] + F(a[k-1])) / 2 ), k>=1
    eps = np.finfo(np.float64).eps

    alpha = (dist.FVec(a[1:-1]) - dist.FVec(a[:-2])) /(dist.FVec(a[2:]) - dist.FVec(a[:-2]))
    alpha = np.concatenate(([0], alpha))                       # alpha[k] = (F(a[k]) - F(a[k-1])/(F(a[k+1]) - F(a[k-1]), K-1>=k>=1
    beta = (dist.FVec(a[2:]) - dist.FVec(a[1:-1])) / (dist.FVec(a[2:]) - dist.FVec(a[:-2]))
    beta = np.concatenate(([0], beta))                         # beta[k] = (F(a[k+1] - F(a[k])/(F(a[k+1]) - F([a[k-1])), K-1>=k>=1
    print('a', a)
    print('b', b)
    print('alpha', alpha)
    print('beta', beta)
    r1 = np.zeros(len(a), dtype=np.float)
    r2 = np.zeros(len(a), dtype=np.float)
    r1[1:-1] = (b[1:-1] - bMinus[1:-1]) / (a[1:-1] - aMinus[1:-1])  # r1[k] = (b[k] - bMinus[k])/((a[k] - aMinus[k]), k>=1
    r2[1:-1] = (bPlus[1:-1] - b[1:-1]) / (aPlus[1:-1] - a[1:-1])   # r2[k] = (bPlus[k] - b[k])/((aPlus[k] - a[k]), k>=1
    print('r1', r1)
    print('r2', r2)
    # array of slopes
    d = np.zeros(len(a), dtype = np.float)                       # len(a) = len(b) = len(d) = K+2
    d[1:-1] = np.power( r1[1:-1], alpha[1:]) * np.power(r2[1:-1], beta[1:])
    d[0] = bPlus[0] / aPlus[0]
    d[-1] = (1-bMinus[-1])/ (1 -aMinus[-1])                      # d[K+1] = (1 - bMinus[K+1]) / ((1-aMinus[K+1])
    print('slopes', d)
    d=np.clip(d, 0.25, 5)
    # build the transformation curve.
    T = interpolationSpline(a, b, d)
    # apply the transformation to the image
    return T[(imgBuf*255).astype(np.int)]

if __name__ == '__main__':
    img = (np.array(range(1000*800))/800000).reshape(1000, 800)
    print(gaussianDistribution(img))