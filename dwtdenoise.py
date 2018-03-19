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
from utils import movingAverage

def dwtDenoise_thr(imArray, thr=1.0, thrmode='hard', wavelet='haar', level=1):
    """
    Denoise a one-channel image using a Discrete Wavelet Transform. The three following
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
           See also https://github.com/stefanv for a similar approach, differing in
           the computation of the filter coefficients.
    @param imArray: image
    @type imArray: ndarray, shape(w,h), dtype= float
    @param thr: filtering threshold parameter A larger value should result in a smoother output.
    @type thr: float
    @param thrMode: one of 'hard', 'soft', 'wiener'
    @type thrMode: str
    @param wavelet: wavelet family
    @type wavelet: str
    @param level: max level of decomposition
    @type level: int
    @return: the denoised image
    @rtype: ndarray same shape as imArray
    """
    # DWT coeffs is a list coeffs[0] : array and for i>=1, coeffs[i] : dict of arrays/t-uple of arrays for wavedec2
    # For all arrays, ndims = imArray.ndims
    coeffs=pywt.wavedecn(imArray, wavelet, level=level)
    # stack all arrays from coeffs in a single ndims-dimensional array
    a, s = pywt.coeffs_to_array(coeffs)
    # mask approximation coefficients
    mask = np.ones(a.shape, dtype = np.bool)
    mask[s[0][0],s[0][1]] = False
    # hard threshold
    if thrmode == 'hard':
        a[mask] = np.where(np.abs(a[mask])<thr, 0, a[mask])
        coeffs = pywt.array_to_coeffs(a, s)
    # soft threshold: h = (|y| - thr) * sgn(y) / y
    elif thrmode == 'soft':
        a[mask] = np.where(np.abs(a[mask]) <= thr, 0, (np.sign(a[mask])) * (np.abs(a[mask])-thr))
        coeffs = pywt.array_to_coeffs(a, s)
    else:
        thr = thr/100
        win_sizes = (3, 5, 7, 9)
        nY2_est = np.empty(a.shape, dtype=float)
        nY2_est.fill(np.inf)
        for all_coeff in coeffs[1:]:
            # ith all coeff is a 3_uple of arrays for level maxlevel - i
            nY2_est = np.empty(all_coeff['ad'].shape, dtype=float)
            nY2_est.fill(np.inf)
            for coeff in all_coeff.values():
                for win_size in win_sizes:
                    nY2 = movingAverage(coeff * coeff, win_size)
                    minmask = (nY2 < nY2_est)
                    nY2_est[minmask] = nY2[minmask]
                # The Wiener Estimator for a noisy signal Y with
                # noise variance sigma2 is ~ max(0,E(Y**2) - sigma**2)/ (max(0, (E(Y**2)-sigma**2) + sigma**2)
                coeff *= np.where(nY2_est> thr, 1.0-thr/nY2_est, 0)
    # inverse DWT
    imArray = pywt.waverecn(coeffs, wavelet)
    return imArray

if __name__== '__main__':
   #out=dwt_denoise(np.array(range(10000)).reshape(100,100), 'db1',1)
   #print(out)
   #w2d("toto.jpg",1,'db1',10)
   dwtDenoise_thr(np.array(range(10000)).reshape(100,100), wavelet='haar', level=3)