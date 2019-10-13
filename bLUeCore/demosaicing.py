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

def demosaic(raw_image_visible, raw_colors_visible, black_level_per_channel):
    """
    demosaic a sensor bitmap. The input array raw_image_visble is the image from sensor. It has
    the same dimensions as the image, BUT NO CHANNEL. The array raw_colors_visible (identical shape)
    gives the color channels (0=R, 1=G, 2=B) corresponding to each point.

    Black levels are subtracted from raw_image_visible before the conversion.

    @param raw_image_visible: image from sensor
    @type raw_image_visible: nd_array, dtype uint16, shape(img_h, img_w)
    @param raw_colors_visible:
    @type raw_colors_visible: nd_array, dtype u1, shape(img_h, img_w)
    @param black_level_per_channel:
    @type black_level_per_channel: list or array, dtype= int
    @return: demosaic array
    @rtype: ndarray, dtype uint16, shape (img_width, img_height, 3)
    """
    black_level_per_channel = np.array(black_level_per_channel, dtype=np.uint16)
    # Bayer bitmap (16 bits), subtract black level for each channel
    if np.any(black_level_per_channel!=0):
        bayerBuf = raw_image_visible - black_level_per_channel[raw_colors_visible]
    else:
        bayerBuf = raw_image_visible
    # encode Bayer pattern to opencv constant
    tmpdict = {0:'R', 1:'G', 2:'B'}
    pattern = 'cv2.COLOR_BAYER_' + tmpdict[raw_colors_visible[1,1]] + tmpdict[raw_colors_visible[1,2]] + '2RGB'
    # demosaic
    demosaicBuffer = cv2.cvtColor(bayerBuf, eval(pattern))
    return demosaicBuffer