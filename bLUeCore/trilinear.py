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


def interpTriLinear(lut, lut_step, nd_img, convert=True):
    """
    Vectorized trilinear interpolation for 3D LUT color mapping.

    Convert an array nd_img with shape (h, w, dIn)  with dIn >=3 by interpolating
    its values from a 3D array lut with shape s = (s1, s2, s3, 3).
    The color channels of nd_img[:,:,:3] are interpolated from lut.
    Identical orderings (BGR, RGB, ...) are assumed for lut axes and color channels.
    The alpha channel, if any, is deleted.

    lut_step is the integer (or the 3-uple of integers) representing the unitary interpolation
    step for each axis of lut.
    All input values for channel i must be in the (right opened)
    interval [0, max[ with max = (s[i] - 1) * lut_step[i].

    If convert is True (default), the output array is clipped to (0, 255) and converted
    to dtype=np.uint8, otherwise the output array has dtype= np.float32.

    :param lut: 3D LUT array
    :type lut: ndarray, dtype float or int, shape(s1, s2, s3, dIn), dIn >= 3
    :param lut_step: interpolation step
    :type lut_step: number or 3-uple of numbers
    :param nd_img: input array
    :type nd_img: ndarray, dtype float or int, shape (h, w, dOut), dOut >= 3
    :param convert: convert the output to dtype=np.uint8
    :type convert: boolean
    :return: interpolated array
    :rtype: ndarray, shape (h, w, 3)
    """

    if not lut.flags['C_CONTIGUOUS']:
        raise ValueError('interpTriLinear : LUT array must be contiguous')

    lut = lut.astype(np.float32)
    #lut_step = np.broadcast_to(np.array(lut_step), (3,))
    img_scaled = nd_img[:, :, :3] / lut_step

    base_idx = img_scaled.astype(np.int16)
    r0, g0, b0 = base_idx[..., 0], base_idx[..., 1], base_idx[..., 2]
    shape = lut.shape
    strides = np.array(lut.strides) // lut.itemsize

    # Flattened indices
    idx_base = np.ravel_multi_index(
        (r0[..., None], g0[..., None], b0[..., None], np.arange(shape[-1])),
        shape
    )

    # Helper function for optimized linear interpolation
    def lerp_in(A, alpha, B, C):
        B -= C
        B *= alpha
        A += B
        return A

    # Get the LUT values at the 8 corners of the cube
    nd00 = np.take(lut, idx_base)
    nd01 = np.take(lut, idx_base + strides[0])
    nd02 = np.take(lut, idx_base + strides[1])
    nd03 = np.take(lut, idx_base + (strides[0] + strides[1]))
    nd10 = np.take(lut, idx_base + strides[2])
    nd11 = np.take(lut, idx_base + (strides[0] + strides[2]))
    nd12 = np.take(lut, idx_base + (strides[1] + strides[2]))
    nd13 = np.take(lut, idx_base + (strides[0] + strides[1] + strides[2]))

    # Interpolation along 1st axis (alpha)
    alpha = (img_scaled[..., 1] - g0)[..., None]
    i11 = lerp_in(nd11, alpha, nd13, nd11)
    i12 = lerp_in(nd10, alpha, nd12, nd10)
    i21 = lerp_in(nd01, alpha, nd03, nd01)
    i22 = lerp_in(nd00, alpha, nd02, nd00)

    # Interpolation along 2nd axis (beta)
    beta = (img_scaled[..., 0] - r0)[..., None]
    j1 = lerp_in(i12, beta, i11, i12)
    j2 = lerp_in(i22, beta, i21, i22)

    # Interpolation along 3rd axis (gamma)
    gamma = (img_scaled[..., 2] - b0)[..., None]
    result = lerp_in(j2, gamma, j1, j2)

    if convert:
        np.clip(result, 0, 255, out=result)
        result = result.astype(np.uint8)

    return result
