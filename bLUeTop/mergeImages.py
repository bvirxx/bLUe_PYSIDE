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
import cv2


def expFusion(imList):
    """
    Computes the exposure fusion of a list of images with identical sizes.
    Cf. Exposure Fusion: A Simple and Practical Alternative to High Dynamic Range Photography.
    Tom Mertens, Jan Kautz and Frank Van Reeth In Computer Graphics Forum, 28 (1) 161 - 171, 2009
    @param imList:
    @type imList: list of ndarray
    @return:
    @rtype: ndarray
    """
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(imList, imList)

    mergeMertens = cv2.createMergeMertens()
    fusion = mergeMertens.process(imList)
    np.clip(fusion, 0.0, 1.0, out=fusion)
    fusion *= 255
    return fusion
