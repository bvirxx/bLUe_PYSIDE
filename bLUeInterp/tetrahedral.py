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

def interpTetra(LUT, LUTSTEP, ndImg):
    """
       Converts a color image by interpolating the values in a 3D LUT array.

       Implements and uses a vectorized version of tetrahedral interpolation.
       Cf.
       U{https://www.filmlight.ltd.uk/pdf/whitepapers//FL-TL-TN-0057-SoftwareLib.pdf}
       page 57.

       The output image has the same type as the input image.
       With s denoting the size of the LUT axes, all pixel colors to
       interpolate must be in the (right opened) interval [0, (s - 1) * LUTSTEP[.
       All values in the LUT should be in range 0..255.
       The role (R or G or B) of the LUT axes follows the ordering of color channels.

       It turns out that tetrahedral interpolation is 2 times slower
       than trilinear.
       @param LUT: 3D LUT array
       @type LUT: ndarray, shape=(s,s,s,3), dtype float or int (faster)
       @param LUTSTEP: interpolation step
       @type LUTSTEP: int
       @param ndImg: image array, 3 channels, same order as LUT channels and axes
       @type ndImg: ndarray dtype=np.uint8
       @return: RGB image with the same type as the input image
       @rtype: ndarray dtype=np.uint8
       """
    # As interpolation computes differences, we switch to a signed type,
    # minimizing memory usage and implicit conversions.
    LUT = LUT.astype(np.int16)
    # We will use the bounding unit cube around each point (r, g, b)/LUTSTEP :
    # get its vertex closest to the origin and the corresponding channel colors.
    ndImgF = ndImg / float(LUTSTEP)
    a = ndImgF.astype(np.int16)
    # RGB channels
    r0, g0, b0 = a[:, :, 0], a[:, :, 1], a[:, :, 2]

    # get indices of vertex channels in the flattened LUT
    s = LUT.shape
    st = np.array(LUT.strides)
    st = st // st[-1]  # we count items instead of bytes
    flatIndex = np.ravel_multi_index((r0[..., np.newaxis], g0[..., np.newaxis], b0[..., np.newaxis], np.arange(3)), s)  # broadcasted to shape (w,h,3)

    # apply LUT to the vertices of the bounding cube
    # np.take uses the the flattened LUT, but keeps the shape of flatIndex
    ndImg00 = np.take(LUT, flatIndex)                            # = LUT[r0, g0, b0] but faster
    ndImg01 = np.take(LUT, flatIndex + st[0])                    # = LUT[r1, g0, b0] where r1 = r0 + 1
    ndImg02 = np.take(LUT, flatIndex + st[1])                    # = LUT[r0, g1, b0]
    ndImg03 = np.take(LUT, flatIndex + (st[0] + st[1]))          # = LUT[r1, g1, b0]
    ndImg10 = np.take(LUT, flatIndex + st[2])                    # = LUT[r0, g0, b1]
    ndImg11 = np.take(LUT, flatIndex + (st[0] + st[2]))          # = LUT[r1, g0, b1]
    ndImg12 = np.take(LUT, flatIndex + (st[1] + st[2]))          # = LUT[r0, g1, b1]
    ndImg13 = np.take(LUT, flatIndex + (st[0] + st[1] + st[2]))  # = LUT[r1, g1, b1]

    fR = ndImgF[: , :,0] - a[:, :, 0]
    fG = ndImgF[:, :, 1] - a[:, :, 1]
    fB = ndImgF[:, :, 2] - a[:, :, 2]
    oneMinusFR = (1 - fR)[..., np.newaxis] * ndImg00
    oneMinusFG = (1 - fG)[..., np.newaxis] * ndImg00
    oneMinusFB = (1 - fB)[..., np.newaxis] * ndImg00

    fRG = (fR - fG)[..., np.newaxis]
    fGB = (fG - fB)[..., np.newaxis]
    fBR = (fB - fR)[..., np.newaxis]
    fR = fR[..., np.newaxis]
    fG = fG[..., np.newaxis]
    fB = fB[..., np.newaxis]

    # regions
    C1 = fR > fG
    C2 = fG > fB
    C3 = fB > fR

    fR = fR * ndImg13
    fG = fG * ndImg13
    fB = fB * ndImg13

    X0 = oneMinusFG + fGB * ndImg02 + fBR * ndImg12 + fR  # fG > fB > fR
    X1 = oneMinusFB + fBR * ndImg10 + fRG * ndImg11 + fG  # fB > fR > fG
    X2 = oneMinusFB - fGB * ndImg10 - fRG * ndImg12 + fR  # fB >=fG >=fR
    X3 = oneMinusFR + fRG * ndImg01 + fGB * ndImg03 + fB  # fR > fG > fB
    X4 = oneMinusFG - fRG * ndImg02 - fBR * ndImg03 + fB  # fG >=fR >=fB
    X5 = oneMinusFR - fBR * ndImg01 - fGB * ndImg11 + fG  # fR >=fB >=fG

    Y1 = np.select(
                [C2 * C3, C3 * C1, np.logical_not(np.logical_or(C1,C2)), C1 * C2, np.logical_not(np.logical_or(C1,C3))],
                [X0, X1, X2, X3, X4],  # clockwise ordering: X3, X5, X1, X2, X0, X4
                default = X5
                )

    np.clip(Y1, 0, 255, out=Y1)
    return Y1.astype(np.uint8)
