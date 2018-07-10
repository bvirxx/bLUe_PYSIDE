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
from time import sleep

from PySide2.QtCore import QByteArray
from PySide2.QtGui import QTransform, QImage
from os.path import isfile, basename
from settings import EXIFTOOL_PATH
from utils import dlgWarn

class ExifTool(object):
    """
    This class implements the exiftool communication and synchronization protocol as
    described in https://www.sno.phy.queensu.ca/~phil/exiftool/exiftool_pod.html
    (cf. the paragraph -stay_open FLAG).
    The implementation follows the guidelines of Sven Marnach answer found in
    https://stackoverflow.com/questions/10075115/call-exiftool-from-a-python-script
    We gratefully acknowledge the contribution of the author.

    """
    # exiftool synchronization token
    sentinel = "{ready}"
    # exiftool useful flags
    # -n : print numerical values
    # -j : json output
    # -a : extract duplicate tags
    # -S : very short output format
    # -G0 : print group name for each tag

    def __init__(self, executable = EXIFTOOL_PATH):
        self.executable = executable

    def __enter__(self):
        """
        entering "with" block:
        launch exiftool. According to the documentation stdin, stdout
        znd stderr are open in binary mode.
        """
        try:
            # hide sub-window to prevent flashing console
            # when the program is freezed by PyInstaller with
            # console set to False.
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            # -@ FILE : read command line args from FILE
            # -stay_open True: keep reading -@ argFILE even after EOF
            self.process = subprocess.Popen(
                                        [self.executable, "-stay_open", "True",  "-@", "-"],  # "-stay_open"
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,#subprocess.DEVNULL, #subprocess.STDOUT,
                                        startupinfo=startupinfo
                                       )
        except OSError:
            dlgWarn("cannot execute exiftool :\nset EXIFTOOL_PATH in settings.py")
            # exit program
            exit()
        return self

    def  __exit__(self, exc_type, exc_value, traceback):
        """
        exit "with" block:
        Terminate process. The function is always executed,
        even if an exception has occured within the block.
        Return True to catch the exception.
        @param exc_type: type of exception (if any)
        @param exc_value: 
        @param traceback: 
        @return: True to catch exceptions
        """
        if exc_type is ValueError:
            print('Exiftool.__exit__: ', exc_value)
            self.process.terminate()
            return True
        self.process.stdin.write(bytearray("-stay_open\nFalse\n", 'ascii'))
        self.process.stdin.flush()
        self.process.terminate()

    def execute(self, *args, ascii=True):
        """
        Class main method. It executes
        the exiftool commands defined by *args and returns
        the command output. The value of the flag ascii must
        correspond to the expected type of exiftool output.
        @param args:
        @type args: tuple of str
        @param indata: data sent to exiftool stdin
        @type inData: bytearray
        @param ascii: flag for exiftool expected output data type : binary/str
        @type ascii: boolean
        @return: command output
        @rtype: str or bytes according to the ascii flag.
        """
        # append synchronization token to args (args type is tuple)
        args = args + ("-execute\n",)
        # convert command to bytes and write it to process stdin
        stdin = self.process.stdin
        stdin.write(bytearray(str.join("\n", args), 'ascii'))
        # flush and sync stdin - both are mandatory here
        stdin.flush()
        os.fsync(stdin.fileno())
        # get exiftool response : data, if any, followed by sentinel
        output = bytearray()
        fdout = self.process.stdout.fileno()
        # encode sentinel to bytes
        sb = self.sentinel.encode('ascii')
        # read stdout up to sentinel
        # NOTE: os.read is blocking; termination is granted by the sentinel
        while not output[:-2].endswith(sb):
            output.extend(os.read(fdout, 4096))
        # cut off sentinel and CRLF
        output = output[:-len(self.sentinel) - 2]
        if ascii:
            output = str(output, encoding='ascii')
        else:
            output = bytes(output[:-len(self.sentinel)-2])
        return output

    def writeThumbnail(self, filename, thumbfile):
        """
        Writes a bytearray containing thumbnail data to an image
        or sidecar file. Thumbnail should be a valid jpeg image
        160x120 or 120x160.
        @param filename: path to image or sidecar file
        @type filename: str
        @param thumbnail: path to thumbnail jpg file
        @type thumbnail: str
        """
        #self.execute(*(['-%s<=%s' % ('ThumbnailImage', thumbnail)] + [filename]))
        self.execute(*([filename] + ['-%s<=%s' % ('thumbnailimage', thumbfile)]))

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
        PIL _getexif() example
        with open(f, 'rb') as fd:
            img = Image.open(fd)
        exif_data = img._getexif()
        """
        flags = ["-j", "-a", "-XMP:all", "-EXIF:all", "-n", "-S", "-G0", "-Orientation", "-ProfileDescription", \
                 "-colorSpace", "-InteropIndex", "-WhitePoint", "-PrimaryChromaticities", "-Gamma"]
        extract_meta_flags = ["-icc_profile", "-b"]
        fmie = f[:-4]+'.mie'
        if isfile(fmie):
            # get profile as bytes
            profile = self.execute(*(extract_meta_flags + [fmie]))
            # get data as (length 1) list of dict :
            # execute returns a string [{aa:bb, cc:dd,...}],
            # and json.loads deserializes it.
            data = json.loads(self.execute(*(flags + [fmie])))
        else:
            profile = self.execute(*(extract_meta_flags + [f]), ascii=False)
            data = json.loads(self.execute(*(flags + [f])))
            # create sidecar file
            if createsidecar:
                self.saveMetadata(f)
        return profile, data

    def get_thumbNail(self, f, thumbname='thumbnailimage'):
        """
        Extracts the thumbnail from image file
        @param f: path to image file
        @type f: str
        @param thumbname: tag name
        @type thumbname: str
        @return: thumbnail
        @rtype: QImage
        """
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