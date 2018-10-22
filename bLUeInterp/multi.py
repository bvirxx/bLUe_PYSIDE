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
from functools import partial
import numpy as np

from bLUeInterp.tetrahedral import interpTetra
from bLUeInterp.trilinear import interpTriLinear

def interpMulti(LUT, LUTSTEP, ndImg, pool=None, use_tetra=False):
    """
    Parallel trilinear/tetrahedral interpolation, using
    a pool of workers.
    Converts a color image using a 3D LUT.
    The roles (R or G or B) of the three first LUT channels
    must follow the ordering of the color channels.
    The output image is interpolated from the LUT.
    It has the same type as the input image.
    @param LUT: 3D LUT array
    @type LUT: ndarray, dtype=np.float
    @param LUTSTEP: interpolation step
    @type LUTSTEP: int
    @param ndImg: image array, RGB channels
    @type ndImg: ndarray dtype=np.uint8
    @param pool: multiprocessing pool
    @type pool: multiprocessing.Pool
    @return: RGB image with same type as the input image
    @rtype: ndarray dtype=np.uint8
    """
    w, h = ndImg.shape[1], ndImg.shape[0]
    SLF = 4
    sl_w = [slice((w * i) // SLF, (w * (i+1)) // SLF) for i in range(SLF)]  # python 3 // for integer quotient
    sl_h = [slice((h * i) // SLF, (h * (i + 1)) // SLF) for i in range(SLF)]

    slices = [ (s1, s2) for s1 in sl_w for s2 in sl_h]
    imgList = [ndImg[s2, s1] for s1, s2 in slices]
    if pool is None:
        raise ValueError('interpMulti: no processing pool')
    # get vectorized interpolation as partial function
    partial_f = partial(interpTetra if use_tetra else interpTriLinear, LUT, LUTSTEP)
    # parallel interpolation
    res = pool.map(partial_f, imgList)
    outImg = np.empty(ndImg.shape)
    # collect results
    for i, (s1, s2) in enumerate(slices):
            outImg[s2, s1] = res[i]
    # np.clip(outImg, 0, 255, out=outImg) # chunks are already clipped
    return outImg.astype(np.uint8)  # TODO 07/09/18 validate
