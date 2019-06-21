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
import pywt
from .rollingStats import movingAverage

def noiseEstimation(DWT_coeffs):
    """
    Returns an estimation of the noise variance, using the Mean
    Absolute Deviation (MAD) method of Donoho.
    of
    @param DWT_coeffs: DWT coefficients, as returned by wavedecn
    @type DWT_coeffs:
    @return: noise variance estimation
    @rtype: float
    """
    a, s = pywt.coeffs_to_array(DWT_coeffs)
    flattened_coeffs = a[s[-1]['dd'][0], s[-1]['dd'][1]].ravel()
    MAD = np.median(abs(flattened_coeffs))
    return MAD*MAD / (0.6745*0.6745)

def dwtDenoiseChan(image, chan=0, thr=1.0, thrmode='hard', wavelet='haar', level=None):
    """
    Denoise a channel of image, using a Discrete Wavelet Transform. The three following
    filtering methods can be used:
       1 - hard threshold,
       2 - soft threshod
       3 - Local Wiener Filter described in
            M. Mihcak, I. Kozintsev, K. Ramchandran, and P. Moulin,
            "Low-complexity image denoising based on statistical
            modeling of wavelet coefficients," IEEE Signal Processing
            Letters, vol. 6, 1999, pp. 300-303.
    Our implementation  follows the lines of J. Fridrich's paper:
    Fridrich, "Digital Image Forensics," IEEE Signal Processing
    Magazine, vol. 26, 2009, pp. 26-37,
    and also :
    M. K. Mihcak, I. Kozintsev, and K. Ramchandran, - Spatially adaptive statistical
    modeling of wavelet image coefficients and its application to denoising - , in
    Proc. IEEE Int. Conf. Acoustics, Speech, and Signal Processing, Phoenix, AZ,
    Mar. 1999, vol. 6, pp. 3253–3256
    See also U{https://github.com/stefanv} for a similar approach, differing in
    the final computation of the filter coefficients.
    @param image: image array
    @type image: ndarray, shape(w,h,d), dtype= float
    @param chan: channel to denoise
    @type chan: int
    @param thr: filtering threshold parameter A larger value should result in a smoother output.
    @type thr: float
    @param thrmode: one among 'hard', 'soft', 'wiener'
    @type thrmode: str
    @param wavelet: wavelet family
    @type wavelet: str
    @param level: max level of decomposition, automatic if level is None (default)
    @type level: int or None
    @return: the denoised channel
    @rtype: ndarray, same shape as the image channel, dtype= np.float

    """
    imArray = image[:,:,chan]
    w,h = imArray.shape[1], imArray.shape[0]
    #################
    # apply DWT
    # DWT_coeffs is the list of DWT coefficients :
    # DWT_coeffs[0] : array and for i>=1, DWT_coeffs[i] : dict of arrays(wavedecn),
    # or t-uple of arrays (wavedec2)
    # For each array a, a.ndims = imArray.ndims
    ###############
    DWT_coeffs = pywt.wavedecn(imArray, wavelet, level=level)
    if thrmode == 'hard' or thrmode == 'soft':
        # stack all arrays from coeffs in a single ndims-dimensional array
        a, s = pywt.coeffs_to_array(DWT_coeffs)  # a:array, s:strides
        # keep details coefficients only
        mask = np.ones(a.shape, dtype = np.bool)
        mask[s[0][0],s[0][1]] = False
        # hard threshold: cut coeffs under thr
        if thrmode == 'hard':
            a[mask] = np.where(np.abs(a[mask])<thr, 0, a[mask])
            DWT_coeffs = pywt.array_to_coeffs(a, s)
        # soft threshold: filter h = max(0, (|a| - thr)) * sgn(a) / a
        elif thrmode == 'soft':
            a[mask] = np.where(np.abs(a[mask]) <= thr, 0, (np.sign(a[mask])) * (np.abs(a[mask])-thr))
            DWT_coeffs = pywt.array_to_coeffs(a, s)
    else:  # local Wiener Filter
        # we do not estimate the noise variance sigma2 (a priori value or
        # Mean Absolute Deviation method for instance).
        # Instead, we use a variable interactive threshold set by the user
        thr = thr/100
        ###################################################
        # Estimation of the variance of the coefficients of the
        # DWT transform.
        # we use an adaptative window-based estimation procedure
        # to capture the effect of edges
        ####################################################
        win_sizes = (3, 5, 7, 9)
        # skip approximation level and scan other levels
        for all_coeff in DWT_coeffs[1:]:
            nY2_est = np.empty(all_coeff['ad'].shape, dtype=float)
            nY2_est.fill(np.inf)
            # walk through H,V,D coefficients (2D arrays) at level i,
            for coeff in all_coeff.values():
                # for each coeff Y, estimate E(Y**2) as the minimum of
                # moving averages of coeff**2 over window sizes.
                for win_size in win_sizes:
                    nY2 = movingAverage(coeff*coeff, win_size, version='strides')
                    minmask = (nY2 < nY2_est)
                    nY2_est[minmask] = nY2[minmask]
                # The Wiener Estimator for a noisy signal Y with
                # noise variance sigma is ~ max(0,E(Y**2) - sigma**2)/ (max(0, E(Y**2)-sigma**2) + sigma**2)
                # here sigma**2 is the interactive threshold
                coeff *= np.where(nY2_est> thr, 1.0 - thr / nY2_est, 0)
    # apply inverse DWT
    imArray = pywt.waverecn(DWT_coeffs, wavelet)
    # waverecn sometimes returns a padded array
    return imArray[:h, :w]

def dwtDenoise(image, thr=1.0, thrmode='hard', wavelet='haar', level=None):
    for chan in image.shape[2]:
        image[:,:,chan] = dwtDenoiseChan(image, chan=chan, thr=thr, thrmode=thrmode, wavelet=wavelet, level=level)

if __name__== '__main__':
   dwtDenoiseChan(np.arange(10000).reshape(100, 100), wavelet='haar', level=3)
