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

# the code of the class ExifTool was taken
# from the Stackoverflow answer http://stackoverflow.com/questions/10075115/call-exiftool-from-a-python-script
# We gratefully acknowledge the contribution of the author.
# startupinfo was added to prevent subprocess from opening a window. This is
# needed when the application is freezed with Pyinstaller

import subprocess
import os
import json

from PySide2.QtCore import QByteArray
from PySide2.QtGui import QTransform, QImage
from os.path import isfile
from settings import EXIFTOOL_PATH
from utils import dlgWarn

class ExifTool(object):
    """
    exiftool wrapper
    """
    sentinel = "{ready}"
    # exiftool flags
    # -n : print numerical values
    # -j : json output
    # -a : extract duplicate tags
    # -S : very short output format
    # -G0 : print group name for each tag
    flags = ["-j", "-a", "-XMP:all", "-EXIF:all", "-n", "-S", "-G0", "-Orientation", "-ProfileDescription", "-colorSpace", "-InteropIndex", "-WhitePoint", "-PrimaryChromaticities", "-Gamma"]#, "-ICC_Profile:all"]
    extract_meta_flags = ["-icc_profile", "-b"] #["-b"] #["-icc_profile", "-b"]
    #copy_meta_flags = ["-tagsFromFile", "-all:all"]

    def __init__(self, executable = EXIFTOOL_PATH):
        self.executable = executable

    def __enter__(self):
        """
        enter "with" block
        @return: 
        """
        try:
            # hide sub-window to prevent flashing console
            # when the program is freezed by PyInstaller with
            # console=False
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            self.process = subprocess.Popen(
                                        [self.executable, "-stay_open", "True",  "-@", "-"],
                                        stdin =subprocess.PIPE, stdout =subprocess.PIPE, stderr =subprocess.STDOUT,
                                        startupinfo=startupinfo
                                       )
        except OSError:
            dlgWarn("cannot execute exiftool :\nset EXIFTOOL_PATH in settings.py")
            exit()
        return self

    def  __exit__(self, exc_type, exc_value, traceback):
        """
        exit "with" block
        @param exc_type: 
        @param exc_value: 
        @param traceback: 
        @return: 
        """
        self.process.stdin.write(bytearray("-stay_open\nFalse\n", 'ascii'))
        self.process.stdin.flush()
        self.process.terminate()

    def execute(self, *args, inData=None, ascii=True):
        """
        Executes the exiftool command defined by *args and returns the
        command output.
        @param args:
        @type args:
        @param indata: data sent to exiftool stdin
        @type inData: bytearray
        @param ascii:
        @type ascii:
        @return: command output
        @rtype: str or bytes
        """
        args = args + ("-execute\n",)
        stdin = self.process.stdin
        stdin.write(bytearray(str.join("\n", args), 'ascii'))
        stdin.flush()
        if inData is not None:
            stdin.write(inData)
            stdin.flush()
            return None
        output = bytearray()
        # get stdout and stdin file descriptors
        fdout = self.process.stdout.fileno()
        # encode sentinel to bytes
        sb = self.sentinel.encode('ascii')
        # read stdout up to sentinel
        while not output[:-2].endswith(sb):
            output.extend(os.read(fdout, 4096))
        # cut off sentinel and CRLF
        output = output[:-len(self.sentinel) - 2]
        if ascii:
            output = str(output, encoding='ascii')
        else:
            output = bytes(output[:-len(self.sentinel)-2])
        return output

    def writeThumbnail(self, filename, thumbnail):
        """
        Writes a bytearray containing thumbnail data to an image
        or sidecar file. Thumbnail should be a valid jpeg image
        160x120 or 120x160.
        @param filename: path to image or sidecar file
        @type filename: str
        @param thumbnail: path to thumbnail jpg file
        @type thumbnail: str
        """
        self.execute(*(['-%s<=%s' % ('ThumbnailImage', thumbnail)] + [filename]))
        #self.execute(*([filename] + ['-%s<=-' % 'thumbnailimage']), inData=bytes)

    def writeOrientation(self, filename, value):
        """
        Writes orientation to file (image or sidecar)
        @param filename: destination file
        @type filename: str
        @param value: orientation code (range 1..8)
        @type value: str
        """
        self.execute(*(['-%s=%s' % ('Orientation', value)] + ['-n'] + [filename]))

    def writeXMPTag(self, filename, tagName, value):
        """
        Writes tag info to the sidecar (.mie) file. If the sidecar
        does not exist it is created from the image file.
        @param filename: image file name
        @type filename: str
        @param tagName: tag name
        @type tagName: str
        @param value: tag value
        @type value: str or number
        """
        fmie = filename[:-4] + '.mie'
        # if sidecar does not exist create it
        if not isfile(fmie):
            self.saveMetadata(filename)
        # write tag to sidecar
        self.execute(*(['-%s=%s' % (tagName, value)] + [fmie]))

    def readXMPTag(self, filename, tagName):
        """
        Reads tag info from the sidecar (.mie) file. Despite its name, the method can read
        a tag of any type. A ValueError exception is raised if the file does not exist.
        @param filename: image or sidecar path
        @type filename: str
        @param tagName:
        @type tagName: str
        @return: tag info
        @rtype: str
        """
        filename = filename[:-4] + '.mie'
        if not isfile(filename):
            raise ValueError
        res = self.execute(*(['-%s' % tagName] + [filename]))
        return res

    def get_metadata(self, f, createsidecar=True):
        """
        Reads metadata from file : data are read from the sidecar
        (.mie) file if it exists. Otherwise data are read
        from the image file and the sidecar file is created if
        createsidecar is True (default).
        @param f: file name
        @type f: str
        @return: profile, metadata
        @rtype: 2-uple profile: bytes, metadata: (length 1) list of dict
        """
        # Using PIL _getexif is simpler.
        # However, exiftool is much more powerful
        """
        with open(f, 'rb') as fd:
            img = Image.open(fd)
        exif_data = img._getexif()
        """
        fmie = f[:-4]+'.mie'
        if isfile(fmie):
            # get profile as bytes
            profile = self.execute(*(self.extract_meta_flags + [fmie]))
            # get data as (length 1) list of dict :
            # execute returns a string [{aa:bb, cc:dd,...}],
            # and json.loads deserializes it.
            data = json.loads(self.execute(*(self.flags + [fmie])))
        else:
            profile = self.execute(*(self.extract_meta_flags + [f]), ascii=False)
            data = json.loads(self.execute(*(self.flags + [f])))
            # create sidecar file
            if createsidecar:
                self.saveMetadata(f)
        return profile, data

    def get_thumbNail(self, f, thumbname='ThumbnailImage'):
        thumbNail = self.execute(*[ '-b', '-m', '-'+thumbname, f], ascii=False)  # -m disables output of warnings
        return QImage.fromData(QByteArray.fromRawData(bytes(thumbNail)), 'JPG')

    def saveMetadata(self, f):
        """
        saves all metadata and icc profile to sidecar (.mie) file.
        An existing sidecar is overwritten.
        @param f: image file to process
        @type f: str
        """
        self.execute(*(["-tagsFromFile", f, "-overwrite_original", f[:-4] + ".mie"]))

    def restoreMetadata(self, source, dest, removesidecar=False):
        """
        Copy all metadata and icc profile from sidecar .mie to image file.
        if removesidecar is True (default False), the sidecar file is removed after copying.
        @param source: file the image was loaded from
        @param dest: file the image is saved to
        @param removesidecar: if True remove sidecar file after restoration. Default is False
        @return True if sidecar file exists, False otherwise
        """
        sidecar = source[:-4] + '.mie'
        if isfile(sidecar):
            # copy metadata from sidecar to image file
            self.execute(*(["-tagsFromFile", sidecar, "-overwrite_original", dest]))
        else:
            return False
        if removesidecar:
            os.remove(sidecar)
        return True

def decodeExifOrientation(value):
    """
    Returns a QTransform object representing the
    image transformation corresponding to the orientation tag value.
    @param value: orientation tag
    @type value: int
    @return: Qtransform object
    @rtype: QTransform
    """
    # identity transformation
    tr = QTransform()
    if value == 0:
        pass
    elif value == 1 :
        pass
    elif value == 6:   # TODO complete
        tr.rotate(90)  # clockwise
    elif value == 8:
        tr.rotate(-90) # counterclockwise
    else :
        raise ValueError("decodeExifOrientation : unhandled orientation tag: %d" % value)
    return tr

"""
case `jpegexiforient -n "$i"` in
 1) transform="";;
 2) transform="-flip horizontal";;
 3) transform="-rotate 180";;
 4) transform="-flip vertical";;
 5) transform="-transpose";;
 6) transform="-rotate 90";;
 7) transform="-transverse";;
 8) transform="-rotate 270";;
 *) transform="";;
 esac
"""