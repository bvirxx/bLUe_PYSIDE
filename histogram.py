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
# Graphics. Computational Imaging and Vision, vol 32. Springer, Dordrecht.
# https://link.springer.com/chapter/10.1007/1-4020-4179-9_42
# However, in contrast with this paper we do not use a mixture gaussian estimation for the density of the distribution, due to a
# too high computational cost. We simply keep the discrete distribution given by the histogram, and the corresponding CDF is
# taken constant on each bin. Valleys are found as local minima of the discrete distribution, by an exhaustive search.
# Our method can also be viewed as pre-discretizing the image, to an integer type (8 bits, 16bits...) before
# applying the contrast correction. The algorithm is simplified and the speedup is consequent.
#########################################################################################################################
import numpy as np

from debug import tdec
from spline import interpolationQuadSpline


class dstb(object):
    """
    Represents a discrete distribution over an interval 0...maxVal of positive
    integers, estimated from an histogram partition of the interval.
    For continuous distributions, 1/maxVal ca be viewed as  the size
    of a discretization mesh.
    """
    interpolateCDF = False
    plotDist = False

    @classmethod
    def FromImage(cls, imgBuf):
        hist, bins = np.histogram(imgBuf, bins=256, range=(0, 255), density=True)
        return  dstb(hist, bins, 255)

    def __init__(self, hist, bins, maxVal=0):
        self.maxVal = max(maxVal, np.ceil(bins[-1]))
        self.DTable, self.CDFTable = self.setDist(hist=hist, bins=bins, maxVal=maxVal)
        self.bins = bins
        self.hist = hist
        # plot curve
        """
        if self.plotDist:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            T = [self.F(k / 256) for k in range(257)]
            TInv = [self.FInv(k / 256) for k in range(257)]
            ax.plot(T, markersize=1, linestyle='', marker='o', color='r')
            ax.plot(TInv, markersize=1, linestyle='', marker='o', color='b')
            #ax.plot(Tdummy, markersize=1, linestyle=None, marker='X')
            ax.plot([self.DTable[k]*100 for k in range(len(self.DTable))])
            fig.savefig('tempdist.png')
            plt.close()
        """

    def setDist(self, hist=None, bins=None, maxVal=0):
        """
        The distribution is estimated from the histogram. If maxVal is 0 (default), the distribution
        represents the probablities of each bin. If maxVal is a positive integer,
        the distribution represents the probabilities of all successive integers in the range 0..ceil(maxVal).
        It is not smoothed: all integers in the same bin get equal probabilities.
        All bins should be between 0 and maxVal.
        @param hist: histogram
        @type hist: list
        @param bins: histogram bins
        @type bins: list
        """
        if bins is None:
            bins = []
        if hist is None:
            hist = []
        if maxVal == 0:
            self.maxVal = bins[-1]                                          # len(self.DTable) = maxVal + 1
            self.DTable = hist * (bins[1:]-bins[:-1])
        else:
            dist = []
            for i in range(maxVal+1):
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
                lg = bins[r] - bins[r - 1]
                n = np.floor(bins[r]) - np.ceil(bins[r - 1]) + (
                    1 if (np.floor(bins[r]) != bins[r] or r == len(bins) - 1) else 0)
                # assign equal probabilities to all these integers
                dist += [hist[r - 1] * lg / n]
            DTable = dist
        CDFTable= np.cumsum(DTable)                             # len(CDFTable) = len(DTable) = maxVal + 1
        # sanity check
        if np.abs(CDFTable[-1] - 1) > 0.00000001:
            raise ValueError('setDistribution: invalid distribution')
        return DTable, CDFTable

    def F(self, x):
        """
        Calculate the CDF.
        For convenience, the argument x is normalized to 0..1.
        We have CDF(x) = Proba(int i: i <= x*maxVal).  for all
        integers k = 0..maxVal, CDF(k/s) = CDFTable(k); If interpolate
        is False (default) the CDF is constant on the intervals [k/s, (k+1)/s[,
        otherwise its value is interpolated between CDF(k/s) and CDF((k+1)/s.
        @param x:
        @type x: float
        @return: CDF(x)
        @rtype: float
        """
        if not np.isscalar(x):
            raise ValueError('dstb.F : argument is not a scalar')
        s = self.maxVal
        xs = x*s
        k1 = int(xs)
        v1 = self.CDFTable[k1]
        if (k1 < s) and self.interpolateCDF:
            k2 = k1 + 1
            v2 = self.CDFTable[k2]
            return (k2 - xs) * v1 + (xs - k1) * v2
        return v1 #self.CDFTable[int(x * s)]

    def FVec(self, x):
        """
        Vectorized version of the CDF function.
        @param x: array of arguments
        @type x: ndarray, dtype float
        @return: array of CDF(x) values
        @rtype: ndarray, dtype=np.float,  same shape as x
        """
        s = self.maxVal
        xs = x * s
        k1 = xs.astype(np.int)
        v1 = self.CDFTable[k1]
        if self.interpolateCDF:
            k2 = np.minimum(k1+1, s)
            v2 = self.CDFTable[k2]
            return (k2 - xs) * v1 + (xs - k1) * v2
        return v1

    def FInvVec(self, x):
        """
        Inverse CDF. The argument x and the
        returned value are normalized to 0..1. if interpolateCDF is False,
        Finv(x) = k/maxVal, with k the smallest integer s. t.
        CDFTable[k] >= x if x > 0 and CDFTable[k+1] > 0  if x = 0.
        if interpolateCDFis True
        Note. As F is neither continuous nor strictly increasing, FInv is not the
        mathematical inverse function of F.
        @param x:
        @type x: float or array of float
        @return:
        @rtype: array of float
        """
        if np.isscalar(x):
            raise ValueError('dstb.FInvVec : argument is a scalar')
        s = self.maxVal
        CDFTable = self.CDFTable
        FVec = self.FVec
        xs = x * s
        # get the smallest integer k s.t. CDFTable[k] >= x
        k1 = np.argmax(CDFTable[:, np.newaxis] >= x, axis=0)
        # CDFTable[-1] should be equal to 1. However, we try to prevent an eventual error
        # caused by a floating point precision pb : k is 0 if 1 >= x > CDFTable[-1]
        M = np.argmax((CDFTable / CDFTable[-1]) == 1.0)
        k1 = np.where(x > CDFTable[-1], M, k1)
        # CDFTable[k1] is >= x, we increase
        # k1 to get CDFTTable[k1+1] > 0
        m = np.argmax(CDFTable > 0)
        k1 = np.maximum(k1, m-1)
        k1_over_s = k1/s
        # interpolation
        if self.interpolateCDF:
            k2 = np.maximum(k1 - 1, 0)                                                   # FVec(k2_over_s) = CDFTable[k1-1] <= x <= CDFTable[k1]
            #k3 = (FVec(k1_over_s) -x) * k2 + (x - FVec(k2overs)) * k1
            k3 = (CDFTable[k1] - x) * k2 + (x - CDFTable[k2]) * k1
            # ignore floating point warnings
            # old_settings = np.seterr(all='ignore')
            k4 = k3 /(CDFTable[k1]-CDFTable[k2])
            # np.seterr(**old_settings)
            k1 = np.where(np.isfinite(k4), k4, k1 )
            k1_over_s = k1/s
        return k1_over_s

    def FInv(self, x):
        """
        Convenience inverse CDF function with scalar argument and value.
        Inverse CDF. For convenience the argument x and the
        returned value are normalized to 0..1
        @param x:
        @type x: float
        @return:
        @rtype: float
        """
        if not np.isscalar(x):
            raise ValueError('dstb.FInv : argument is not a scalar')
        return np.asscalar(self.FInvVec(np.array([x])))

def gaussianDistribution(x, hist, bins, h):
    """
    Computes the kernel density estimation at point x, by a mixture of gaussian variables
    for the distribution (hist, bins). The parameter x is in range 0..256
    and h is the bandwidth.
    @param x: x-value for estimation
    @type x: float, range 0..256
    @param hist: histogram
    @type hist: ndarray
    @param bins: bins of histogram
    @type bins: ndarray
    @param h: bandwidth
    @type h: float
    @return: density estimation value at x
    @rtype: float
    """
    dist = dstb(hist, bins)
    values = np.arange(256, dtype=np.float)/256
    valuesx = x/256 -values
    valuesx = - valuesx*valuesx/(2*h*h)
    expx = np.exp(valuesx)/ (h*np.sqrt(2*np.pi)) * dist
    dx = np.sum(expx)
    return dx

def distWins(dist, delta):
    """
    Returns the sorted list of the windows of (probability) width >= 2*delta
    for a disribution dist. Parameter delta should be in range 0..0.5.
    @param dist: distribution
    @type dist: distribution object
    @param delta: half probablity width of window
    @type delta: float
    @return: the list of 3-uples (j, k, i) corresponding to the windows (k+j, k, k+i), Note j is < 0
    @rtype: list
    """
    mv = dist.maxVal
    CDF = dist.CDFTable
    W = []
    for k in range(mv+1):
        # search for a window of (probability) width >= 2*delta, centered at k.
        foundi, foundj = False, False
        i, j = 1, -1
        while  k+i < len(CDF)-1:
            if CDF[k + i] - CDF[k] >= delta:
                foundi=True
                break
            i += 1
        while  k+j > 0:
            if CDF[k] - CDF[k + j] >= delta:
                foundj = True
                break
            j -= 1
        if foundi and foundj:
            W.append((j, k, i))
    return W

def valleys(imgBuf, delta):
    """
    Searches for valleys (local minima) in the distribution of the image. A valley is a
    window of (probability) width 2*delta, whose central point gives its minimum value.
    The function returns the ordered list of these local minima and a distribution object
    representing the distribution of image data.
    Note. In contrast to Grundland and Dodgson U{https://link.springer.com/chapter/10.1007/1-4020-4179-9_42},
    a valley may contain other (higher) local minima.

    @param imgBuf: image buffer, one single channel, range 0..255
    @type imgBuf: ndarray, dtype uint8 or int or float
    @param delta:
    @type delta: float
    @return: V, dist
    @rtype: ndarray, distribution
    """
    # build image histogram and init distribution
    hist, bins = np.histogram(imgBuf, bins=256, range=(0,255), density=True)
    dist = dstb(hist, bins, 255)
    # get candidate windows
    hDB = distWins(dist, delta)
    # search for valleys
    V = []
    DTable = dist.DTable
    for j, k, i in hDB:
        if np.all([(DTable[l] > DTable[k]) for l in range(k+j, k)]) and np.all([(DTable[l] > DTable[k]) for l in range(k+1, k+i+1)]) :
            V.append(k)
    V = np.fromiter(V, dtype=np.float, count=len(V))
    return V, dist

def autoQuadSpline(imgBuf, valleyAperture=0.05, warp=1.0, preserveHigh=True):
    """
    Calculates a quadratic spline from ImgBuf histogram, for automatic
    contrats enhancement.
    We mainly use the algorithm proposed by  Grundland and Dodgson
    Cf. U{https://link.springer.com/chapter/10.1007/1-4020-4179-9_42},
    with a supplementary correction for highlights.
    The parameter valleyAperture is the (probability) width
    of histogram valleys: it should be > 0 and < 0.5. The parameter warp controls
    the correction level : it should be between 0 (no correction and 1 (full correction).
    If preserveHigh is True (default) a final correction is applied to preserve highlights.
    @param imgBuf:
    @type imgBuf:
    @param valleyAperture:
    @type valleyAperture:
    @param warp: control the amplitude of the control point moves
    @type warp: float, range 0..1
    @param preserveHigh: final highlight correction
    @type preserveHigh: boolean
    @return: x-coordinates, y-coordinates, tangent slopes, spline array (tabulation)
    @rtype: ndarray dtype=float, ranges 0..1
    """
    ###################
    # outlier threshold
    tau = 0.01
    ###################
    # get valleys and distribution object
    V0, dist = valleys(imgBuf*255, valleyAperture)                    # len(V0) = K-1 is the valley count
    V0 = V0/255
    # discard images with a too narrow dynamic range
    if dist.FInv(1) - dist.FInv(0) < 0.01:
        raise ValueError('warphistogram: dynamic range too narrow')
    # add the distribution end points to the valley array,
    # and calculate the array of mode centers :
    # each interval [V[k-1], V[k] is called a mode.
    # Modes are viewed as groups of objects with similar lighting conditions,
    # (background and foreground for example).
    # The center points of mode intervals partition the pixel range in regions and
    # each region will be transformed by a single quadratic spline.
    V0 = np.concatenate(([dist.FInv(0)], V0, [dist.FInv(1)]))
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
    a[0], a[1], a[-2], a[-1] = dist.FInv(0), dist.FInv(tau), dist.FInv(1 - tau), dist.FInv(1)
    b[0], b[1], b[-2], b[-1] = 0, tau, 1 - tau, 1
    # make a strictly increasing if needed
    if a[1]<=a[0] or a[1]>=a[2]:
        a[1]= (a[0]*99+a[2])/100
    if a[-2]>=a[-1] or a[-2]<=a[-3]:
        a[-2] = (a[-1]*99+a[-3])/100
    if  np.min(a[1:]-a[:-1]) <= 0:
        raise ValueError('warpHistogram : array must be strictly increasing')
    # move b[k] within [v[k-1], v[k]] to equalize the histogram.
    # The parameter s controls the move.
    # b[k] is interpolated between V[k-1], V[k] and a[k], hence b[k]
    # lies in [V(k-1], V[k]], which in turn guaranties that b[2:-2]  is non decreasing.
    s = np.clip(warp, 0, 1)
    for k in range(2, len(b)-2):                                # 2..K-1
        b[k] = (dist.F(V[k]) - dist.F(a[k])) * V[k-1] + (dist.F(a[k]) - dist.F(V[k-1])) * V[k]
        b[k] = b[k] / (dist.F(V[k]) - dist.F(V[k-1]))*s  + a[k]*(1-s)     # F(V[k]) - F(V[k-1] >= valleyAperture
    if np.min(b[1:] - b[:-1]) < 0:
        raise ValueError('warpHistogram : array b must be non decreasing')

    # calculate curve slopes at (a[k], b[k]).
    bPlus = (b[:-1] + b[1:])/ 2                                 # bPlus[k] = (b[k] + b[k+1])/ 2
    bPlus = np.concatenate((bPlus, [0.0]))
    bMinus = np.concatenate( ([0.0], bPlus[:-1]))               # bMinus[k] = (b[k] + b[k-1]) / 2 , k>=1

    tmpMid = (dist.FVec(a[:-1]) + dist.FVec(a[1:]))/2           # tmpMid[k] = (F(a[k]) + F(a[k+1]) / 2
    aPlus = dist.FInvVec(tmpMid)                                # aPLus[k] = FInv( (F(a[k] + F(a[k+1])) / 2 )
    aPlus = np.concatenate((aPlus, [1.0]))
    aMinus = np.concatenate(([0.0], aPlus[:-1]))                # aMinus[k] = FInv( (F(a[k] + F(a[k-1])) / 2 ), k>=1
    #eps = np.finfo(np.float64).eps

    alpha = (dist.FVec(a[1:-1]) - dist.FVec(a[:-2])) /(dist.FVec(a[2:]) - dist.FVec(a[:-2]))
    alpha = np.concatenate(([0], alpha))                       # alpha[k] = (F(a[k]) - F(a[k-1])/(F(a[k+1]) - F(a[k-1]), K-1>=k>=1
    beta = (dist.FVec(a[2:]) - dist.FVec(a[1:-1])) / (dist.FVec(a[2:]) - dist.FVec(a[:-2]))
    beta = np.concatenate(([0], beta))                         # beta[k] = (F(a[k+1] - F(a[k])/(F(a[k+1]) - F([a[k-1])), K-1>=k>=1
    r1 = np.zeros(len(a), dtype=np.float)
    r2 = np.zeros(len(a), dtype=np.float)
    r1[1:-1] = (b[1:-1] - bMinus[1:-1]) / (a[1:-1] - aMinus[1:-1])  # r1[k] = (b[k] - bMinus[k])/((a[k] - aMinus[k]), k>=1
    r2[1:-1] = (bPlus[1:-1] - b[1:-1]) / (aPlus[1:-1] - a[1:-1])    # r2[k] = (bPlus[k] - b[k])/((aPlus[k] - a[k]), k>=1
    # array of slopes
    d = np.zeros(len(a), dtype = np.float)                          # len(a) = len(b) = len(d) = K+2
    d[1:-1] = np.power( r1[1:-1], alpha[1:]) * np.power(r2[1:-1], beta[1:])
    d[0] = bPlus[0] / aPlus[0]
    d[-1] = (1-bMinus[-1])/ (1 -aMinus[-1])                      # d[K+1] = (1 - bMinus[K+1]) / ((1-aMinus[K+1])
    d = np.where(d==np.NaN, 1.0, d)
    d=np.clip(d, 0.25, 5)
    # highlights correction
    if preserveHigh:
        skyInd = np.argmax(a > 0.75*a[-1])
        #if a[-3] > 0.75*a[-1]:  # may be V0[-2]>=0.75*V0[-1]??
        b[-1]=0.98
        for i in range(len(b) - skyInd - 1):  # TODO 31/05/18 skyInd is array, should be int
            b[-i-2]= min(np.power(a[-i-2], 0.30), b[-i-1]) - 0.02
        b[skyInd-1] = np.power(b[skyInd-1], 0.9)
        d[skyInd:] = 0.2
        d[-1]=2
    # build and apply the spline.
    T = interpolationQuadSpline(a, b, d, plot=True)
    return a, b, d, T

def warpHistogram(imgBuf, valleyAperture=0.05, warp=1.0, preserveHigh=True, spline=None):
    """
    Stretches and warps the distribution of imgBuf to enhance the contrast.
    If a spline is given, it is applied to imgBuf, otherwise an "automatic"
    spline is deduced from the image histogram. We mainly use the algorithm
    proposed by Grundland and Dodgson,
    Cf. U{https://link.springer.com/chapter/10.1007/1-4020-4179-9_42},
    with a supplementary correction for highlights.
    The parameter valleyAperture is the (probability) width
    of histogram valleys: it should be > 0 and < 0.5. The parameter warp controls
    the correction level : it should be between 0 (no correction and 1 (full correction).
    If preserveHigh is True (default) a final correction is applied to preserve highlights.
    @param imgBuf: single channel image (luminance), range 0..1
    @type imgBuf: ndarray, shape(h,w), dtype=uint8 or int or float
    @param valleyAperture:
    @type valleyAperture: float
    @param warp:
    @type warp: float
    @param preserveHigh:
    @type preserveHigh: boolean
    @param spline: spline, range 0..256 --> 0..1
    @type spline: activeSpline
    @return: the transformed image channel, range 0..1 and the quadratic spline
    @rtype: image ndarray same shape as imgBuf, dtype=np.float, a, b, T are in range 0..1
    """
    if spline is None:
        a, b, d, T = autoQuadSpline(imgBuf, valleyAperture=valleyAperture, warp=warp, preserveHigh=preserveHigh)
    else:
        a, b, d, T = [p.x() for p in spline.fixedPoints], [p.y() for p in spline.fixedPoints], spline.fixedTangents, spline.LUTXY/256
    im = imgBuf*255
    im1 = im.astype(np.int)
    im2 = im1+1
    B1 = T[im1]
    # extrapolate T to handle eventual value 256 in im2
    T1 = np.hstack((T, [T[-1]]))
    B2 = T1[im2] # clearer but slower : T[np.minimum(im2, 255)]
    # interpolate B1, B2
    B = (im2 - im) * B1 + (im - im1) * B2
    return np.clip(B, 0, 1, out=B), a, b, d, T

if __name__ == '__main__':
    img = (np.arange(1000*800, dtype=np.float)/800000).reshape(1000, 800)
    img = np.zeros((1000, 800))
    dist = dstb.FromImage(img)
