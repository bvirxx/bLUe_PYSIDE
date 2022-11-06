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
from .rollingStats import movingAverage, movingVariance


def noiseSTD_Est(DWT_coeffs, version='DON'):
    """
    Estimates the standard deviation of the noise.
    The MAD estimation highlights a possibly lower noise reduction.

    See D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
    by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.

    :param DWT_coeffs: DWT coeffs as yielded by pywt.wavedecn()
    :type DWT_coeffs:
    :return: noise STD
    :rtype: float

    """
    if version == 'DON':
        # use median
        detail_coeffs = DWT_coeffs[1:][-1]['dd']
        # removing 0's seems to introduce some discontinuities in noise STD estimation
        # detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]
        if detail_coeffs.size > 0:
            const = 0.6745  # could be calculated as  scipy.stats.norm.ppf(0.75)
            sigma = np.median(np.abs(detail_coeffs)) / const
        else:
            sigma = 0.0

    else:
        # use Mean Absolute Deviation (MAD)
        coeff_array = pywt.coeffs_to_array(DWT_coeffs)[0]
        m = np.mean(coeff_array)
        MAD = np.mean(np.absolute(coeff_array - m))
        sigma = ((np.pi / 2) ** 0.5) * MAD

    return sigma


def chooseDWTLevel(imArray, wavelet):
    """
    Returns a convenient level for the DWT decomposition
    of imArray

    :param imArray:
    :type imArray: ndarray
    :param wavelet: wavelet family
    :type wavelet: str
    :return: max level
    :rtype: int
    """

    w, h = imArray.shape[1], imArray.shape[0]
    wavelet = pywt.Wavelet(wavelet)
    dlen = wavelet.dec_len
    wavelet_levels = np.min([pywt.dwt_max_level(s, dlen) for s in [w, h]])

    level = max(wavelet_levels - 3, 1)
    return level


def dwtDenoiseChan(image, chan=0, thr=1.0, thrmode='hard', wavelet='haar', level=None):
    """
    Denoises a channel of an image, using a Discrete Wavelet Transform. The three following
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
    See also Stefan van der Walt https://github.com/stefanv for a similar approach, differing in
    the final computation of the DWT coefficients using the Wiener Estimator.

    :param image: image array
    :type image: ndarray, shape(w,h,d), dtype= float
    :param chan: channel to denoise
    :type chan: int
    :param thr: filtering threshold parameter A larger value should result in a smoother output.
    :type thr: float
    :param thrmode: one among 'hard', 'soft', 'wiener'
    :type thrmode: str
    :param wavelet: wavelet family
    :type wavelet: str
    :param level: max level of decomposition, automatic if level is None (default)
    :type level: int or None
    :return: the denoised channel
    :rtype: ndarray, same shape as the image channel, dtype= float

    """

    imArray = image[:, :, chan]

    #################
    # DWT_coeffs is the list of DWT coefficients :
    # DWT_coeffs[0] : array
    # for i>=1, DWT_coeffs[i] : dict of arrays (wavedecn), or t-uple of arrays (wavedec2)
    ###############
    DWT_coeffs = pywt.wavedecn(imArray, wavelet, level=level)
    sigma = noiseSTD_Est(DWT_coeffs)

    if thrmode == 'hard' or thrmode == 'soft':
        # stack all coeffs in a single array
        a, s = pywt.coeffs_to_array(DWT_coeffs)  # a:array, s:strides
        # keep details coefficients only
        mask = np.ones(a.shape, dtype=np.bool)
        mask[s[0][0], s[0][1]] = False

        # hard threshold: cut coeffs under thr
        if thrmode == 'hard':
            a[mask] = np.where(np.abs(a[mask]) < thr, 0, a[mask])
            DWT_coeffs = pywt.array_to_coeffs(a, s)

        # soft threshold: filter h = max(0, (|a| - thr)) * sgn(a) / a
        elif thrmode == 'soft':
            a[mask] = np.where(np.abs(a[mask]) <= thr, 0, (np.sign(a[mask])) * (np.abs(a[mask]) - thr))
            DWT_coeffs = pywt.array_to_coeffs(a, s)

    else:  # local Wiener Filter
        ###################################################
        # local variance of coeffs is used to build an adapted filter.
        # we use a window-based estimation procedure
        # to capture the effect of edges.
        ####################################################
        win_sizes = (3, 5, 7, 9)
        thr = sigma * thr / 5
        thr = thr ** 2  # set middle value to sigma**2
        for all_coeff in DWT_coeffs[1:]:
            nY2_est = np.empty(all_coeff['ad'].shape, dtype=float)
            nY2_est.fill(np.inf)
            # walk through coeffs (2D arrays) at level i,
            for coeff in all_coeff.values():
                # calculate the minimum of moving variances
                # of coeff over window sizes.
                for win_size in win_sizes:
                    nY2 = movingVariance(coeff, win_size, version='strides')
                    # nY2 = movingAverage(coeff * coeff, win_size, version='strides')
                    minmask = (nY2 < nY2_est)
                    nY2_est[minmask] = nY2[minmask]
                # The Wiener Estimator for a noisy signal Y with
                # noise variance sigma is ~ max(0,E(Y**2) - sigma**2)/ (max(0, E(Y**2)-sigma**2) + sigma**2)
                # We replace sigma**2 by the interactive threshold.
                coeff *= np.where(nY2_est > thr, 1.0 - thr / nY2_est, 0)
                ###################################################
                # Wiener Estimator according to the code of Stefan van der Walt.
                # coeff *= (nY2_est / (nY2_est + thr))
                ###################################################

        sigma1 = noiseSTD_Est(DWT_coeffs)
        print(sigma, sigma1)

    # apply inverse DWT
    w, h = imArray.shape[1], imArray.shape[0]
    imArray = pywt.waverecn(DWT_coeffs, wavelet)
    np.clip(imArray, 0, 255, out=imArray)
    # waverecn may return a padded array
    return imArray[:h, :w]


def dwtDenoise(source, dest, thr=1.0, thrmode='hard', wavelet='haar', level=None):
    for chan in range(source.shape[2]):
        dest[:, :, chan] = dwtDenoiseChan(source, chan=chan, thr=thr, thrmode=thrmode, wavelet=wavelet, level=level)
