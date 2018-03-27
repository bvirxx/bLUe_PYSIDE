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

def distribution(hist, bins):
    """
    This function Interpolates the distribution ("density") of an image, given its histogram.
    It computes the list of Proba(image=k) for k in range(256).
    The interpolation attributes equal probabilities to all points in the same bin.
    @param hist:
    @type hist:
    @param bins:
    @type bins:
    @return: The interpolated distribution
    @rtype: list
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

def distWins(hist, bins, delta):
    """
    Return the sorted list of the windows of (probability) width 2*delta
    for the distribution (hist,bins).
    @param hist:
    @type hist:
    @param bins:
    @type bins:
    @param delta:
    @type delta:
    @return: the list of 3-uples (j, k, i) corresponding to the windows (k+j, k, k+i), j is < 0
    @rtype:
    """
    # get CDF
    F = np.cumsum(hist * (bins[1:] - bins[:-1]))
    # F = np.cumsum(distribution(hist, bins))
    def histDistBin(k):
        i, j = 1, -1
        while  F[k+i] - F[k] < delta and k+i < len(F)-1:
            i += 1
        while F[k] - F[k+j]< delta and k+j > 0:
            j -= 1
        return j, k, i
    # cut the delta-queues of distribution.
    m, M = np.argmax(F > delta), np.argmax(F > 1 - delta)
    return [histDistBin(k) for k in range(m, M)]

def valleys(imgBuf, delta):
    """
    Search for valleys (local minima) in the distribution of the image histogram. A valley is a
    window of (probability) width 2*delta, whose central point gives the minimum value.
    The function returns the ordered list of these local minima in 2nd and 3rd quartiles, completed by the two values 0 and 255.

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
    as a piecewise rational quadratic interpolation to a set of (a[k], b[k]) 2D-points (nodes).
    d[k] controls the slope at the node points. The function returns
     the list of T[k] values for k in range(256).
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
    ax.plot(T*256)
    ax.plot([i for i in range(256)])
    ax.plot(a*256, b*256, 'ro')
    ax.plot(a*256, a*256, 'bo')
    fig.savefig('temp.png')
    return T

def warpHistogram(imgBuf, valleyAperture=0.01, warp=0):
    """
    Stretch and warp the distribution of imgBuf.
    The parameter delta controls the (probability) width
    of histogram valleys: it should be >0 and < 0.5
    @param imgBuf: single channel image (luminance), values in 0..255
    @type imgBuf: ndarray, shape(h,w), dtype=uint8 (or int in range 0..255)
    @param valleyAperture:
    @type valleyAperture:
    @return: transformed image.
    @rtype: ndarray same shape as imgBuf, dtype=np.int
    """
    # build the histogram and get valleys
    V, hist, bins = valleys(imgBuf, valleyAperture)
    """
    n = np.prod(imgBuf.shape)
    cs = np.cumsum(hist)
    F1, F2 = np.argmax(cs > 1/4), np.argmax(cs>3/4)
    h = 0.7816774 / np.power(n, 1/7) * (F2 - F1)/256
    """
    V = V/255.0
    # get the centers of the intervals between valleys.
    a = (V[1:] + V[:-1]) / 2
    a = np.concatenate(([0.0], a, [1.0])) # len(a) = len(V) + 1
    # get CDF of distribution
    CDF = np.cumsum(np.array(distribution(hist, bins)))
    def F(x):
        return CDF[int(x*255)]
    b = a.copy()
    # move b[k]'s to equalize the histogram.
    # s controls the move
    s = warp
    for k in range(1, len(V)):
        b[k] = (F(V[k]) - F(a[k])) * V[k-1] + (F(a[k]) - F(V[k-1])) * V[k]
        b[k] = b[k] / (F(V[k]) - F(V[k-1]))*s  + a[k]*(1-s)
    # curve slopes at nodes (a[k], b[k]).
    # They control stretching.
    d=np.array([0.3]*2 + [1] + [0]*256)
    # build transformation curve.
    T = interpolationSpline(a, b, d)
    return T[(imgBuf.astype(np.int))]*255.0


if __name__ == '__main__':
    img = (np.array(range(1000*800))/800000).reshape(1000, 800)
    print(gaussianDistribution(img))