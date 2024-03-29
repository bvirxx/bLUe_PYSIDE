#
# The Python Imaging Library.
# $Id$
#
# a simple Qt image interface.
#
# history:
# 2006-06-03 fl: created
# 2006-06-04 fl: inherit from QImage instead of wrapping it
# 2006-06-05 fl: removed toimage helper; move string support to ImageQt
# 2013-11-13 fl: add support for Qt5 (aurelien.ballier@cyclonit.com)
#
# Copyright (c) 2006 by Secret Labs AB
# Copyright (c) 2006 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
###############################
# NOTICE
# The Python Imaging Library (PIL) is

# Copyright © 1997-2011 by Secret Labs AB
# Copyright © 1995-2011 by Fredrik Lundh

# By obtaining, using, and/or copying this software and/or its associated documentation, you agree that you have read, understood, and will comply
# with the following terms and conditions:

# Permission to use, copy, modify, and distribute this software and its associated documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appears in all copies, and that both that copyright notice and this permission notice appear
# in supporting documentation, and that the name of Secret Labs AB or the author not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.

# SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
# ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
################################

# The PIL.ImageQt module looks for PySide installed and does not find PySide6.
# This modified version of ImageQt.py is a workaround for PySide6.
# Do not use the functions fromqimage, fromqpixmap, toqimage, toqpixmap
# from PIL.Image as they do local imports of ImageQt. Use the corresponding functions
# from this module instead.

from PIL import Image
from PIL._util import isPath
from io import BytesIO

qt_is_installed = True
qt_version = None

try:
    from PyQt5.QtGui import QImage, qRgba, QPixmap
    from PyQt5.QtCore import QBuffer, QIODevice

    qt_version = '5'
except (ImportError, RuntimeError):
    try:
        from PyQt4.QtGui import QImage, qRgba, QPixmap
        from PyQt4.QtCore import QBuffer, QIODevice

        qt_version = '4'
    except (ImportError, RuntimeError):
        try:
            from PySide6.QtGui import QImage, qRgba, QPixmap
            from PySide6.QtCore import QBuffer, QIODevice

            qt_version = 'side'
        except ImportError:
            qt_is_installed = False
"""
from PySide6.QtGui import QImage, qRgba, QPixmap
from PySide6.QtCore import QBuffer, QIODevice
qt_version = 'side'
"""


def rgb(r, g, b, a=255):
    """(Internal) Turns an RGB color into a Qt compatible color integer."""
    # use qRgb to pack the colors, and then turn the resulting long
    # into a negative integer with the same bitpattern.
    return (qRgba(r, g, b, a) & 0xffffffff)


def fromqimage(im):
    """
    :param im: A PIL Image object, or a file name
    (given either as Python string or a PyQt string object)
    """
    buffer = QBuffer()
    buffer.open(QIODevice.ReadWrite)
    # preserve alpha channel with png
    # otherwise ppm is more friendly with Image.open
    if im.hasAlphaChannel():
        im.save(buffer, 'png')
    else:
        im.save(buffer, 'ppm')

    b = BytesIO()
    try:
        b.write(buffer.data())
    except TypeError:
        # workaround for Python 2
        b.write(str(buffer.data()))
    buffer.close()
    b.seek(0)

    return Image.open(b)


def fromqpixmap(im):
    return fromqimage(im)
    # buffer = QBuffer()
    # buffer.open(QIODevice.ReadWrite)
    # # im.save(buffer)
    # # What if png doesn't support some image features like animation?
    # im.save(buffer, 'ppm')
    # bytes_io = BytesIO()
    # bytes_io.write(buffer.data())
    # buffer.close()
    # bytes_io.seek(0)
    # return Image.open(bytes_io)


def align8to32(bytes, width, mode):
    """
    converts each scanline of data from 8 bit to 32 bit aligned
    """

    bits_per_pixel = {
        '1': 1,
        'L': 8,
        'P': 8,
    }[mode]

    # calculate bytes per line and the extra padding if needed
    bits_per_line = bits_per_pixel * width
    full_bytes_per_line, remaining_bits_per_line = divmod(bits_per_line, 8)
    bytes_per_line = full_bytes_per_line + (1 if remaining_bits_per_line else 0)

    extra_padding = -bytes_per_line % 4

    # already 32 bit aligned by luck
    if not extra_padding:
        return bytes

    new_data = []
    for i in range(len(bytes) // bytes_per_line):
        new_data.append(bytes[i * bytes_per_line:(i + 1) * bytes_per_line] + b'\x00' * extra_padding)

    return b''.join(new_data)


def _toqclass_helper(im):
    data = None
    colortable = None

    # handle filename, if given instead of image name
    if hasattr(im, "toUtf8"):
        # FIXME - is this really the best way to do this?
        if str is bytes:
            im = unicode(im.toUtf8(), "utf-8")
        else:
            im = str(im.toUtf8(), "utf-8")
    if isPath(im):
        im = Image.open(im)

    if im.mode == "1":
        format = QImage.Format_Mono
    elif im.mode == "L":
        format = QImage.Format_Indexed8
        colortable = []
        for i in range(256):
            colortable.append(rgb(i, i, i))
    elif im.mode == "P":
        format = QImage.Format_Indexed8
        colortable = []
        palette = im.getpalette()
        for i in range(0, len(palette), 3):
            colortable.append(rgb(*palette[i:i + 3]))
    elif im.mode == "RGB":
        data = im.tobytes("raw", "BGRX")
        format = QImage.Format_RGB32
    elif im.mode == "RGBA":
        try:
            data = im.tobytes("raw", "BGRA")
        except SystemError:
            # workaround for earlier versions
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (b, g, r, a))
        format = QImage.Format_ARGB32
    else:
        raise ValueError("unsupported image mode %r" % im.mode)

    __data = data or align8to32(im.tobytes(), im.size[0], im.mode)
    return {
        'data': __data, 'im': im, 'format': format, 'colortable': colortable
    }


def PilImgToRaw(im):  # rename for external use
    return _toqclass_helper(im)


if qt_is_installed:
    class ImageQt(QImage):

        def __init__(self, im):
            """
            A PIL image wrapper for Qt.

           :param im: A PIL Image object, or a file name (given either as Python
                string or a PyQt string object).
            """
            im_data = _toqclass_helper(im)
            # must keep a reference, or Qt will crash!
            # Fixes https://github.com/python-pillow/Pillow/issues/1370
            self.__data = im_data['data']
            QImage.__init__(self,
                            self.__data, im_data['im'].size[0],
                            im_data['im'].size[1], im_data['format'])
            if im_data['colortable']:
                self.setColorTable(im_data['colortable'])


def toqimage(im):
    return ImageQt(im)


def toqpixmap(im):
    # # This doesn't work. For now using a dumb approach.
    # im_data = _toqclass_helper(im)
    # result = QPixmap(im_data['im'].size[0], im_data['im'].size[1])
    # result.loadFromData(im_data['data'])
    # Fix some strange bug that causes
    if im.mode == 'RGB':
        im = im.convert('RGBA')

    qimage = toqimage(im)
    return QPixmap.fromImage(qimage)
