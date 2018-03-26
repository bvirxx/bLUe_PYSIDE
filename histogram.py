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

import numpy as np

from imgconvert import QImageBuffer

def distribution(hist, bins):
    """
    Interpolate the distribution ("density") of an image, given its histogram,
    i.e. compute the list [Proba(image=k) for k in range(256)].
    We attribute equal probabilities to all points in the same bin.
    (Cf. the equivalent function interpHist above)
    @param hist:
    @type hist:
    @param bins:
    @type bins:
    @return:
    @rtype:
    """
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

def gaussianDist(x, hist, bins, h):
    """
    build the approximation of the histogram hist
    by a mixture of gaussian variables. The parameter x is in range 0..255
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

def CDF(x, hist, bins, h, gaussian=False):
    """
    return the value of the CDF of the gaussian approximation of the histogram hist at point x.
    x is an integer in range 0..255
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
    if gaussian:
        value = np.sum([gaussianDist(y, hist, bins, h) for y in range(int(x))]) / 256
    else:
        value = np.sum([distribution(hist, bins)[y] for y in range(int(x))]) / 256
    print ('CDF', x, value)
    return value

def distWins(hist, bins, delta):
    """
    Return the list of 2*delta windows in hist distribution
    @param hist:
    @type hist:
    @param bins:
    @type bins:
    @param delta:
    @type delta:
    @return:
    @rtype:
    """
    # get CDF
    F = np.cumsum(hist*(bins[1:] - bins[:-1]))
    #F = np.cumsum(distribution(hist, bins))
    m, M = np.argmax(F> delta), np.argmax(F>1-delta)
    def histDistBin(k):
        i, j = 1, -1
        while  F[k+i] - F[k] < delta and k+i < len(F)-1:
            i += 1
        while F[k] - F[k+j]< delta and k+j > 0:
            j -= 1
        return j, k, i
    return [histDistBin(k) for k in range(m, M)]

def valleys(imgBuf, delta):
    """
    Search for valleys in histogram distribution. A valley is a
    window of width 2*delta, whose central point is the minimum value.
    The function returns the list of the central points of the valleys,
    completed with the two values 0 and 255.

    @param imgBuf:
    @type imgBuf:
    @param delta:
    @type delta:
    @return:
    @rtype:
    """
    hist, bins = np.histogram(imgBuf, bins=100, density=True)
    V = []
    # get windows
    hDB = distWins(hist, bins, delta)
    # search for valleys
    for j, k, i in hDB:
        if np.all([(hist[l] > hist[k]) for l in range(k+j, k)]) and np.all([(hist[l] > hist[k]) for l in range(k+1, k+i+1)]) :
            V.append(k)
    V = [0.0]+V+[255.0]
    V = np.array(V, dtype=np.float)
    return V, hist, bins

def interpolationSpline(a, b, d):
    """
    Build a transformation curve T from [0,1] onto itself,
    as piecewise rational quadratic interpolation to a set of (a[k], b[k]) 2D-points.
    d[k] controls the slope of the segments: if 0<=d[k]<1 the kth segment
    compresses the corresponding portion of the histogram, if d[k] > 1 it is stretched.
    The function returns the list of T[k] values for k in range(256).
    The 3 parameter lists maust ahave the same length.
    @param a:
    @type a:
    @param b:
    @type b:
    @param d:
    @type d:
    @return:
    @rtype:
    """
    x = np.array(range(256), dtype=np.float)/256
    #
    tmp = np.array([[(a[j]> x[i]) for i in range(len(x))] for j in range(len(a))])
    # for each i, get smallest a[j] > x[i]
    k = np.argmax(tmp, axis=0) # a[k[i]-1]<= x[i] < a[k[i]] if k[i] > 0, and x[i] out of a[0],..a[-1] otherwise
    k = np.where(k == 0, len(a)-1, k)

    r = (b[k] - b[k-1])/(a[k]-a[k-1])
    t = (x-a[k-1])/(a[k]-a[k-1])
    assert np.all(t>=0)
    assert np.all(t<=1)

    T = b[k-1] + (r[k]*t*t + d[k-1]*(1-t)*t)*(b[k]-b[k-1]) / (r[k]+(d[k]+d[k-1] - 2*r[k])*(1-t)*t)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(T)
    ax.plot([i/256 for i in range(256)])
    fig.savefig('temp.png')
    return T

def stretchHist(imgBuf, delta):
    """

    @param imgBuf:
    @type imgBuf:
    @param delta:
    @type delta:
    @return:
    @rtype:
    """
    n = np.prod(imgBuf.shape)
    # build histogram and get valleys
    V, hist, bins = valleys(imgBuf, delta)
    F = np.cumsum(hist)
    F1, F2 = np.argmax(F > 1/4), np.argmax(F>3/4)
    h = 0.7816774 / np.power(n, 1/7) * (F2 - F1)/256
    V = V/255.0
    a = (V[1:] + V[:-1]) / 2
    a = np.concatenate(([0.0], a, [1.0])) # len(a) = len(V) + 1
    # get CDF
    CDF = np.cumsum(np.array(distribution(hist, bins)))
    def F(x):
        #return CDF(x*255, hist, bins, h)
        return CDF[int(x*255)]
    b = a.copy()
    """
    b[-1]=250/255
    # warp midpoints
    for k in range(1, len(V)):
        b[k] = (F(V[k]) - F(a[k])) * V[k-1] + (F(a[k]) - F(V[k-1])) * V[k]
        b[k] = b[k] / (F(V[k]) - F(V[k-1]))
    """
    d=np.array([0.2]*4 + [0.2]*256)
    T = interpolationSpline(a, b, d)
    imgBuf[:,:]= T[(imgBuf.astype(np.int))]*255.0


if __name__ == '__main__':
    img = (np.array(range(1000*800))/800000).reshape(1000, 800)
    print(gaussianDist(img))