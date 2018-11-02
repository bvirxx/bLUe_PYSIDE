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

    Convert a array ndImg with shape (w, h, 3) by interpolating
    its values in a 3D LUT array LUT with shape s = (s1, s2, s3, 3).
    3-uple values v taken from the third axis of ndImg1 are input to
    the three first axes of the LUT, keeping the ordering (i.e. v[i] is input to axis i).

    LUTSTEP is the number or the 3-uple of numbers giving the unitary interpolation
    steps for each axis of the LUT table.

    All input values for axis i of the LUT must be in the (right opened)
    interval [0, max[ with max = (s[i] - 1) * LUTSTEP[i].

    if convert is True (default), the output array is clipped to (0, 255) and converted
    to dtype=np.uint8, otherwise the output array has the same shape as ndImg and
    dtype= np.float32.

    @param LUT: 3D LUT array
    @type LUT: ndarray, dtype float or int, shape(s1, s2, s3, 3)
    @param LUTSTEP: interpolation step
    @type LUTSTEP: number or 3-uple of numbers
    @param ndImg: input array
    @type ndImg: ndarray dtype float or int, shape (w, h, 3)
    @param convert: convert the output to dtype=np.uint8
    @type convert: boolean
    @return: interpolated array
    @rtype: ndarray, same shape as the input image
    """
    # Probably due to a numpy bug, ravel_multi_index sometimes returns wrong indices
    # for non contiguous arrays.
    if not LUT.flags['C_CONTIGUOUS']:
        raise ValueError('interpTriLinear : LUT array must be contiguous')
    # As interpolation computes differences, we switch to a signed type,
    # minimizing memory usage and implicit conversions.
    #LUT = LUT.astype(np.int16)
    LUT = LUT.astype(np.float32)
    # We will use the bounding unit cube around each point (r, g, b)/LUTSTEP :
    # get its vertex closest to the origin and the corresponding channel colors.
    ndImgF = ndImg / LUTSTEP
    a = ndImgF.astype(np.int16)
    r0, g0, b0 = a[:,:,0], a[:,:,1], a[:,:,2]

    # get indices of the vertex channels in the flattened LUT
    s = LUT.shape
    st = np.array(LUT.strides)
    st = st // st[-1]  # we count items instead of bytes
    flatIndex = np.ravel_multi_index((r0[...,np.newaxis], g0[...,np.newaxis], b0[...,np.newaxis], np.arange(3)), s) # broadcasted to shape (w,h,3)

    # apply LUT to the vertices of the bounding cube.
    # np.take uses the the flattened LUT, but keeps the shape of flatIndex
    ndImg00 = np.take(LUT, flatIndex)                           # = LUT[r0, g0, b0] but faster
    ndImg01 = np.take(LUT, flatIndex + st[0])                   # = LUT[r1, g0, b0] where r1 = r0 + 1
    ndImg02 = np.take(LUT, flatIndex + st[1])                   # = LUT[r0, g1, b0]
    ndImg03 = np.take(LUT, flatIndex + (st[0] + st[1]))         # = LUT[r1, g1, b0]
    ndImg10 = np.take(LUT, flatIndex + st[2])                   # = LUT[r0, g0, b1]
    ndImg11 = np.take(LUT, flatIndex + (st[0] + st[2]))         # = LUT[r1, g0, b1]
    ndImg12 = np.take(LUT, flatIndex + (st[1] + st[2]))         # = LUT[r0, g1, b1]
    ndImg13 = np.take(LUT, flatIndex + (st[0] + st[1] + st[2])) # = LUT[r1, g1, b1]

    # interpolation
    alpha =  ndImgF[:,:,1] - g0
    alpha=np.dstack((alpha, alpha, alpha))
    #alpha = alpha[:,:,np.newaxis] # broadcasting tested slower

    I11Value = ndImg11 + alpha * (ndImg13 - ndImg11)  #oneMinusAlpha * ndImg11 + alpha * ndImg13
    I12Value = ndImg10 + alpha * (ndImg12 - ndImg10)  #oneMinusAlpha * ndImg10 + alpha * ndImg12
    I21Value = ndImg01 + alpha * (ndImg03 - ndImg01)  #oneMinusAlpha * ndImg01 + alpha * ndImg03
    I22Value = ndImg00 + alpha * (ndImg02 - ndImg00)  # oneMinusAlpha * ndImg00 + alpha * ndImg02

    del ndImg00, ndImg01, ndImg02, ndImg03, ndImg10, ndImg11, ndImg12, ndImg13

    beta = ndImgF[:,:,0] - r0
    beta = np.dstack((beta, beta, beta))
    #beta = beta[...,np.newaxis]

    I1Value = I12Value + beta * (I11Value - I12Value) #oneMinusBeta * I12Value + beta * I11Value
    I2Value = I22Value + beta * (I21Value - I22Value) #oneMinusBeta * I22Value + beta * I21Value
    # allowing to free memory
    del I11Value, I12Value, I21Value, I22Value

    gamma = ndImgF[:,:,2] - b0
    gamma = np.dstack((gamma, gamma, gamma))
    #gamma = gamma[...,np.newaxis]

    IValue = I2Value + gamma * (I1Value - I2Value)  #(1 - gamma) * I2Value + gamma * I1Value

    if convert:
        # clip in place
        np.clip(IValue, 0, 255, out=IValue)
        IValue = IValue.astype(np.uint8)
    return IValue

def interpTriLinearPara(LUT, LUTSTEP, ndImg):
    """
    Converts a color image by interpolating the values in a 3D LUT array.

    Implements a vectorized version of trilinear interpolation.

    The output image has the same type as the input image.
    All pixel colors to interpolate must be in the (right opened)
    interval [0, (s - 1) * LUTSTEP[, with s denoting the size of the LUT axes,
    All values in the LUT should be in range 0..255.
    The role (R or G or B) of the LUT axes follows the ordering of color channels.

    @param LUT: 3D LUT array
    @type LUT: ndarray, dtype float or int (faster)
    @param LUTSTEP: interpolation steps
    @type LUTSTEP: 3-uple
    @param ndImg: image array, 3 channels, same order as LUT channels and axes
    @type ndImg: ndarray dtype=np.uint8
    @return: RGB image with the same type as the input image
    @rtype: ndarray dtype=np.uint8
    """
    s = LUT.shape
    # As interpolation computes differences, we switch to a signed type,
    # minimizing memory usage and implicit conversions.
    #LUT = LUT.astype(np.int16)
    # We will use the bounding unit cube around each point (r, g, b)/LUTSTEP :
    # get its vertex closest to the origin and the corresponding channel colors.
    ndImgF = ndImg / LUTSTEP
    a = ndImgF.astype(np.int16)
    # values should be STRICTLY lower than s -1 for all axes. Value s-1 can for ndIMg values > = (s-1) * LUTSTEP
    r0, g0, b0 = np.clip(a[:,:,0], 0, s[0]-2), np.clip(a[:,:,1], 0, s[1] - 2), np.clip(a[:,:,2], 0, s[2] - 2)

    # get indices of the vertex channels in the flattened LUT

    st = np.array(LUT.strides)
    st = st // st[-1]  # we count items instead of bytes
    flatIndex = np.ravel_multi_index((r0[...,np.newaxis], g0[...,np.newaxis], b0[...,np.newaxis], np.arange(3)), s) # broadcasted to shape (w,h,3)

    # apply LUT to the vertices of the bounding cube.
    # np.take uses the the flattened LUT, but keeps the shape of flatIndex
    ndImg00 = np.take(LUT, flatIndex)                           # = LUT[r0, g0, b0] but faster
    ndImg01 = np.take(LUT, flatIndex + st[0])                   # = LUT[r1, g0, b0] where r1 = r0 + 1
    ndImg02 = np.take(LUT, flatIndex + st[1])                   # = LUT[r0, g1, b0]
    ndImg03 = np.take(LUT, flatIndex + (st[0] + st[1]))         # = LUT[r1, g1, b0]
    ndImg10 = np.take(LUT, flatIndex + st[2])                   # = LUT[r0, g0, b1]
    ndImg11 = np.take(LUT, flatIndex + (st[0] + st[2]))         # = LUT[r1, g0, b1]
    ndImg12 = np.take(LUT, flatIndex + (st[1] + st[2]))         # = LUT[r0, g1, b1]
    ndImg13 = np.take(LUT, flatIndex + (st[0] + st[1] + st[2])) # = LUT[r1, g1, b1]

    # interpolation
    alpha =  ndImgF[:,:,1] - g0
    alpha=np.dstack((alpha, alpha, alpha))
    #alpha = alpha[:,:,np.newaxis] # broadcasting tested slower

    I11Value = ndImg11 + alpha * (ndImg13 - ndImg11)  #oneMinusAlpha * ndImg11 + alpha * ndImg13
    I12Value = ndImg10 + alpha * (ndImg12 - ndImg10)  #oneMinusAlpha * ndImg10 + alpha * ndImg12
    I21Value = ndImg01 + alpha * (ndImg03 - ndImg01)  #oneMinusAlpha * ndImg01 + alpha * ndImg03
    I22Value = ndImg00 + alpha * (ndImg02 - ndImg00)  # oneMinusAlpha * ndImg00 + alpha * ndImg02

    del ndImg00, ndImg01, ndImg02, ndImg03, ndImg10, ndImg11, ndImg12, ndImg13

    beta = ndImgF[:,:,0] - r0
    beta = np.dstack((beta, beta, beta))
    #beta = beta[...,np.newaxis]

    I1Value = I12Value + beta * (I11Value - I12Value) #oneMinusBeta * I12Value + beta * I11Value
    I2Value = I22Value + beta * (I21Value - I22Value) #oneMinusBeta * I22Value + beta * I21Value
    # allowing to free memory
    del I11Value, I12Value, I21Value, I22Value

    gamma = ndImgF[:,:,2] - b0
    gamma = np.dstack((gamma, gamma, gamma))
    #gamma = gamma[...,np.newaxis]

    IValue = I2Value + gamma * (I1Value - I2Value)  #(1 - gamma) * I2Value + gamma * I1Value

    # clip in place
    #np.clip(IValue, 0, 255, out=IValue)
    return IValue#.astype(np.uint8)
