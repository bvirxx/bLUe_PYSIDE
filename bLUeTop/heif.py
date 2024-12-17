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
import pillow_heif

from PySide6.QtGui import QImage
from bLUeGui.bLUeImage import ndarrayToQImage


def readheifFile(f):
    """
    Reads a HEIF file into a numpy array buffer.
    An alpha channel is added when needed.
    :param f: file name
    :type f: str
    :return: buffer
    :rtype: numpy array
    """
    img = QImage()
    if pillow_heif.is_supported(f):
        heif_file = pillow_heif.open_heif(f, bgr_mode=True)  # convert to 8 bits
        iobuf = np.asarray(heif_file[0], copy=True)  # read primary image only
        if iobuf.shape[2] < 4:
            # add alpha channel
            aux = np.zeros((iobuf.shape[0], iobuf.shape[1], 4), dtype=np.uint8)
            aux[..., :3] = iobuf
            aux[..., 3] = 255
            iobuf = aux
    else:
        raise IOError('Pillow_heif: unsupported file % s' % f)
    return iobuf

def readheifFile2QImage(f):
    """
    Reads a HEIF file to a QImage.
    :param f: file name
    :type f: str
    :return:
    :rtype: QImage, format ARGB32
    """
    iobuf = readheifFile(f)
    return ndarrayToQImage(iobuf, format=QImage.Format.Format_ARGB32)
