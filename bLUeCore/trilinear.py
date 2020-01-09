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


def interpTriLinear(LUT, LUTSTEP, ndImg, convert=True):
    """
    Implement a vectorized version of trilinear interpolation.

    Convert an array ndImg with shape (h, w, dIn)  with dIn >=3 by interpolating
    its values from a 3D LUT array LUT with shape s = (s1, s2, s3, dOut).
    Values from the third axis of ndImg[:,:,:3] are input to
    the three first axes of LUT, keeping the same ordering (i.e. v[i] is input to axis i).
    Output values are interpolated from LUT.

    LUTSTEP is the integer or the 3-uple of integers representing the unitary interpolation
    step for each axis of LUT.

    All input values for axis i must be in the (right opened)
    interval [0, max[ with max = (s[i] - 1) * LUTSTEP[i]. Closed intervals
    [0, max] can be used instead by adding a sentinel to each axis. Then, with
    max = (s[i] - 2) * LUTSTEP[i], the LUT values for sentinel sides are not used.

    if convert is True (default), the output array is clipped to (0, 255) and converted
    to dtype=np.uint8, otherwise the output array has dtype= np.float32.

    @param LUT: 3D LUT array
    @type LUT: ndarray, dtype float or int, shape(s1, s2, s3, dIn), dIn >= 3
    @param LUTSTEP: interpolation step
    @type LUTSTEP: number or 3-uple of numbers
    @param ndImg: input array
    @type ndImg: ndarray dtype float or int, shape (h, w, dOut), dOut >= 3
    @param convert: convert the output to dtype=np.uint8
    @type convert: boolean
    @return: interpolated array
    @rtype: ndarray, shape (h, w, dOut)
    """
    # Probably due to a numpy bug, ravel_multi_index sometimes returns wrong indices
    # for non contiguous arrays.
    if not LUT.flags['C_CONTIGUOUS']:
        raise ValueError('interpTriLinear : LUT array must be contiguous')
    # As interpolation computes differences, we switch to a signed type,
    # minimizing memory usage and implicit conversions.
    # LUT = LUT.astype(np.int16)
    LUT = LUT.astype(np.float32)
    # We will use the bounding unit cube around each point (r, g, b)/LUTSTEP :
    # get its vertex closest to the origin and the corresponding channel colors.
    ndImgF = ndImg / LUTSTEP
    a = ndImgF.astype(np.int16)
    r0, g0, b0 = a[:, :, 0], a[:, :, 1], a[:, :, 2]

    # get indices of the vertex channels in the flattened LUT
    s = LUT.shape
    st = np.array(LUT.strides)
    st = st // st[-1]  # we count items instead of bytes
    flatIndex = np.ravel_multi_index((r0[..., np.newaxis],
                                      g0[..., np.newaxis],
                                      b0[..., np.newaxis],
                                      np.arange(s[-1])),
                                      s)  # broadcasted to shape (w,h,3)

    # apply LUT to the vertices of the bounding cube.
    # np.take uses the the flattened LUT, but keeps the shape of flatIndex
    ndImg00 = np.take(LUT, flatIndex)                            # = LUT[r0, g0, b0] but faster
    ndImg01 = np.take(LUT, flatIndex + st[0])                    # = LUT[r1, g0, b0] where r1 = r0 + 1
    ndImg02 = np.take(LUT, flatIndex + st[1])                    # = LUT[r0, g1, b0]
    ndImg03 = np.take(LUT, flatIndex + (st[0] + st[1]))          # = LUT[r1, g1, b0]
    ndImg10 = np.take(LUT, flatIndex + st[2])                    # = LUT[r0, g0, b1]
    ndImg11 = np.take(LUT, flatIndex + (st[0] + st[2]))          # = LUT[r1, g0, b1]
    ndImg12 = np.take(LUT, flatIndex + (st[1] + st[2]))          # = LUT[r0, g1, b1]
    ndImg13 = np.take(LUT, flatIndex + (st[0] + st[1] + st[2]))  # = LUT[r1, g1, b1]

    # interpolation
    alpha = ndImgF[:, :, 1] - g0
    # alpha=np.dstack((alpha, alpha, alpha))
    alpha = alpha[:, :, np.newaxis]  # broadcasting tested slower

    I11Value = ndImg11 + alpha * (ndImg13 - ndImg11)  # oneMinusAlpha * ndImg11 + alpha * ndImg13
    I12Value = ndImg10 + alpha * (ndImg12 - ndImg10)  # oneMinusAlpha * ndImg10 + alpha * ndImg12
    I21Value = ndImg01 + alpha * (ndImg03 - ndImg01)  # oneMinusAlpha * ndImg01 + alpha * ndImg03
    I22Value = ndImg00 + alpha * (ndImg02 - ndImg00)  # oneMinusAlpha * ndImg00 + alpha * ndImg02
    # allowing to free memory
    del ndImg00, ndImg01, ndImg02, ndImg03, ndImg10, ndImg11, ndImg12, ndImg13

    beta = ndImgF[:, :, 0] - r0
    # beta = np.dstack((beta, beta, beta))
    beta = beta[..., np.newaxis]

    I1Value = I12Value + beta * (I11Value - I12Value)  # oneMinusBeta * I12Value + beta * I11Value
    I2Value = I22Value + beta * (I21Value - I22Value)  # oneMinusBeta * I22Value + beta * I21Value
    # allowing to free memory
    del I11Value, I12Value, I21Value, I22Value

    gamma = ndImgF[:, :, 2] - b0
    # gamma = np.dstack((gamma, gamma, gamma))
    gamma = gamma[..., np.newaxis]

    IValue = I2Value + gamma * (I1Value - I2Value)  # (1 - gamma) * I2Value + gamma * I1Value

    if convert:
        # clip in place
        np.clip(IValue, 0, 255, out=IValue)
        IValue = IValue.astype(np.uint8)
    return IValue
