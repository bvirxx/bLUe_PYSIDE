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


def interpTetra(lut, lut_step, nd_img, convert=True):
    """
    Vectorized tetrahedral interpolation for 3D LUT color mapping.

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
        raise ValueError("LUT must be a contiguous array")

    lut = lut.astype(np.float32)
    lut_step = np.array(lut_step, dtype=np.float32)

    # Normalize image to LUT grid coordinates
    ndImgF = nd_img / lut_step
    base_coords = ndImgF.astype(np.int16)
    r0, g0, b0 = base_coords[:, :, 0], base_coords[:, :, 1], base_coords[:, :, 2]

    # Compute flattened LUT index
    shape = lut.shape
    strides = np.array(lut.strides) // lut.itemsize
    lut_index = np.ravel_multi_index(
        (r0[..., np.newaxis], g0[..., np.newaxis], b0[..., np.newaxis], np.arange(shape[-1])),
        shape
    )

    # Get the LUT values at the 8 corners of the cube
    ndImg00 = np.take(lut, lut_index)
    ndImg01 = np.take(lut, lut_index + strides[0])
    ndImg02 = np.take(lut, lut_index + strides[1])
    ndImg03 = np.take(lut, lut_index + (strides[0] + strides[1]))
    ndImg10 = np.take(lut, lut_index + strides[2])
    ndImg11 = np.take(lut, lut_index + (strides[0] + strides[2]))
    ndImg12 = np.take(lut, lut_index + (strides[1] + strides[2]))
    ndImg13 = np.take(lut, lut_index + (strides[0] + strides[1] + strides[2]))

    # Compute fractional deltas
    fR = ndImgF[:, :, 0] - r0
    fG = ndImgF[:, :, 1] - g0
    fB = ndImgF[:, :, 2] - b0
    fR, fG, fB = fR[..., np.newaxis], fG[..., np.newaxis], fB[..., np.newaxis]

    one_minus_fR = (1 - fR) * ndImg00
    one_minus_fG = (1 - fG) * ndImg00
    one_minus_fB = (1 - fB) * ndImg00

    # Intermediate weights
    fRG, fGB, fBR = fR - fG, fG - fB, fB - fR

    # Conditions for region selection
    C1 = fR > fG
    C2 = fG > fB
    C3 = fB > fR

    # Compute interpolated colors for each region
    X0 = one_minus_fG + fGB * ndImg02 + fBR * ndImg12 + fR * ndImg13
    X1 = one_minus_fB + fBR * ndImg10 + fRG * ndImg11 + fG * ndImg13
    X2 = one_minus_fB - fGB * ndImg10 - fRG * ndImg12 + fR * ndImg13
    X3 = one_minus_fR + fRG * ndImg01 + fGB * ndImg03 + fB * ndImg13
    X4 = one_minus_fG - fRG * ndImg02 - fBR * ndImg03 + fB * ndImg13
    X5 = one_minus_fR - fBR * ndImg01 - fGB * ndImg11 + fG * ndImg13

    result = np.select(
        [C2 & C3, C3 & C1, ~(C1 | C2), C1 & C2, ~(C1 | C3)],
        [X0, X1, X2, X3, X4],
        default=X5
    )

    if convert:
        np.clip(result, 0, 255, out=result)
        result = result.astype(np.uint8)

    return result
