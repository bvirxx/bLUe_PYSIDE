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
from utils import movingAverage, movingVariance


def dwtDenoiseChan(image, chan=0, thr=1.0, thrmode='hard', wavelet='haar', level=1):
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
            Magazine, vol. 26, 2009, pp. 26-37.
           See also U{https://github.com/stefanv} for a similar approach, differing in
           the final computation of the filter coefficients.
    @param imArray: image
    @type imArray: ndarray, shape(w,h,d), dtype= float
    @param chan: channel to denoise
    @type chan: int
    @param thr: filtering threshold parameter A larger value should result in a smoother output.
    @type thr: float
    @param thrMode: one of 'hard', 'soft', 'wiener'
    @type thrMode: str
    @param wavelet: wavelet family
    @type wavelet: str
    @param level: max level of decomposition
    @type level: int
    @return: the denoised channel
    @rtype: ndarray, same shape as the image channel, dtype= np.float

    """
    imArray = image[:,:,chan]
    #imArray = 2 * np.sqrt(imArray + 3 / 8)
    # coeffs is the list of DWT coefficients :
    # coeffs[0] : array and for i>=1, coeffs[i] : dict of arrays(wavedecn),
    # or t-uple of arrays (wavedec2)
    # For each array a, a.ndims = imArray.ndims
    coeffs=pywt.wavedecn(imArray, wavelet, level=level)
    if thrmode == 'hard' or thrmode == 'soft':
        # stack all arrays from coeffs in a single ndims-dimensional array
        a, s = pywt.coeffs_to_array(coeffs)  # a:array, s:strides
        # keep details coefficients only
        mask = np.ones(a.shape, dtype = np.bool)
        mask[s[0][0],s[0][1]] = False
        # hard threshold: cut coeffs under thr
        if thrmode == 'hard':
            a[mask] = np.where(np.abs(a[mask])<thr, 0, a[mask])
            coeffs = pywt.array_to_coeffs(a, s)
        # soft threshold: filter h = max(0, (|a| - thr)) * sgn(a) / a
        elif thrmode == 'soft':
            a[mask] = np.where(np.abs(a[mask]) <= thr, 0, (np.sign(a[mask])) * (np.abs(a[mask])-thr))
            coeffs = pywt.array_to_coeffs(a, s)
    else:  # local Wiener Filter
        imV = movingVariance(imArray[9:-9, 9:-9], 9, version='strides')
        imV[image[9:-9,9:-9,0] > 200]=np.inf
        x = np.argmin(imV)
        x_u = np.unravel_index(x, imV.shape)
        print('argmin', x_u, imV[x_u])
        # stack coefficients,  # a:array, s:strides
        a, s = pywt.coeffs_to_array(coeffs)
        coefV = movingVariance(a[s[3]['dd'][0], s[3]['dd'][1]], 3, version='strides')
        y = np.argmin(coefV)
        y_u = np.unravel_index(y, coefV.shape)
        print('argmin coeff', y_u, coefV[y_u])

        flattened_coeffs = a[s[-1]['dd'][0], s[-1]['dd'][1]].ravel()
        median = np.median(abs(flattened_coeffs))

        print('median', median)

        thr = thr/100 #((median/0.6745)**2) * thr/500
        win_sizes = (3, 5, 7, 9)
        # skip approximation and walk through levels
        for all_coeff in coeffs[1:]:
            nY2_est = np.empty(all_coeff['ad'].shape, dtype=float)
            nY2_est.fill(np.inf)
            # walk through H,V,D coefficients (2D arrays) at level i,
            for coeff in all_coeff.values():
                for win_size in win_sizes:
                    nY2 = movingAverage(coeff*coeff, win_size, version='strides')
                    minmask = (nY2 < nY2_est)
                    nY2_est[minmask] = nY2[minmask]
                # The Wiener Estimator for a noisy signal Y with
                # noise variance sigma2 is ~ max(0,E(Y**2) - sigma**2)/ (max(0, (E(Y**2)-sigma**2) + sigma**2)
                coeff *= np.where(nY2_est> thr, 1.0-thr/nY2_est, 0)
    # inverse DWT
    imArray = pywt.waverecn(coeffs, wavelet)
    return imArray

def dwtDenoise(image, thr=1.0, thrmode='hard', wavelet='haar', level=1):
    for chan in image.shape[2]:
        image[:,:,chan] = dwtDenoiseChan(image, chan=chan, thr=thr, thrmode=thrmode, wavelet=wavelet, level=level)

if __name__== '__main__':
   #out=dwt_denoise(np.array(range(10000)).reshape(100,100), 'db1',1)
   #print(out)
   #w2d("toto.jpg",1,'db1',10)
   dwtDenoiseChan(np.arange(10000).reshape(100, 100), wavelet='haar', level=3)