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

# The class ExifTool implements the exiftool communication and synchronization protocol as
# described in https://www.sno.phy.queensu.ca/~phil/exiftool/exiftool_pod.html
# (cf. the paragraph -stay_open FLAG).
# The implementation as a context manager follows the guidelines of Sven Marnach answer found in
# https://stackoverflow.com/questions/10075115/call-exiftool-from-a-python-script
# We gratefully acknowledge the contribution of the author.

import re
import subprocess
import os
import json
from sys import platform

from PySide2.QtCore import QByteArray
from PySide2.QtGui import QTransform, QImage
from os.path import isfile
from bLUeTop.settings import EXIFTOOL_PATH
from bLUeGui.dialog import dlgWarn


class ExifTool(object):
    """
    # exiftool useful flags
    # -v : formatted output
    # -n : print numerical values
    # -j : json output
    # -a : extract duplicate tags
    # -S : very short output format
    # -G0 : print group name for each tag
    """
    # exiftool synchronization token
    sentinel = "{ready}"

    def __init__(self, executable=EXIFTOOL_PATH):
        self.executable = executable

    def __enter__(self):
        """
        entering "with" block: launch exiftool.
        According to the documentation stdin, stdout and stderr are open in binary mode.
        """
        try:
            startupinfo = None

            if platform == 'win32':
                # hide sub-window to prevent flashing console when the program is frozen by PyInstaller with
                # console set to False.
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # prevent subprocess from opening a window.
                startupinfo.wShowWindow = subprocess.SW_HIDE  # This is needed when the app is frozen with PyInstaller

            self.process = subprocess.Popen(
                [self.executable, "-stay_open", "True", "-@", "-"],
                # -@ FILE : read command line args from FILE, -stay_open True: keep reading -@ argFILE even after EOF
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # subprocess.DEVNULL
                startupinfo=startupinfo
            )

        except (AttributeError, OSError):
            dlgWarn("cannot execute exiftool :\nset EXIFTOOL_PATH in config.json")
            # exit program
            exit()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        exit "with" block:
        Terminate process. The function is always executed,
        even if an exception occurred within the block.
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
        self.process.wait()  # mandatory to prevent defunct on linux

    def execute(self, *args, ascii=True):
        """
        Main ExifTool method. It executes
        the exiftool commands defined by *args and returns
        exif output. If ascii is True, output is decoded as str,
        and is a bytes object otherwise.
        @param args:
        @type args: tuple of str
        @param ascii: flag for the type of returned data
        @type ascii: boolean
        @return: command output
        @rtype: str or bytes according to the ascii flag.
        """
        args = args + ("-execute\n",)
        # convert command to bytes and write it to process stdin
        stdin = self.process.stdin
        try:
            stdin.write(bytearray(str.join("\n", args), 'ascii'))
        except UnicodeEncodeError as e:
            dlgWarn(str.join("\n", args), str(e))
        # flush and sync stdin : both are mandatory on Windows
        stdin.flush()
        if platform == 'win32':
            os.fsync(stdin.fileno())
        # get exiftool response : data, if any, followed by sentinel
        output = bytearray()
        fdout = self.process.stdout.fileno()
        # encode sentinel to bytes
        sb = self.sentinel.encode('ascii')
        # read stdout up to sentinel
        # NOTE: os.read is blocking; termination is granted by the sentinel
        if platform == 'win32':
            eol = 2  # CRLF
        else:
            eol = 1  # CR
        while not output[:-eol].endswith(sb):
            output.extend(os.read(fdout, 4096))
        # cut off sentinel and CRLF
        output = output[:-len(self.sentinel) - eol]
        if ascii:
            output = str(output, encoding='ascii')
        else:
            output = bytes(output)
        return output

    ##################
    # Convenience methods
    #################
    def createSidecar(self, f):
        """
        Copy all metadata and icc profile from image file
        to sidecar (.mie) file. An existing sidecar is overwritten.
        @param f: path to image file
        @type f: str
        """
        # following exif doc, wild cards do not copy icc_profile : we must specify it explicitly
        # Tag ImageDescription is added by tifffile to .blu files to hold the layer stack.
        # Copying it to sidecar will restore old stack.
        command = ["-tagsFromFile", f, "-all", "-icc_profile", "-overwrite_original", "--ImageDescription",
                   f[:-4] + ".mie"]
        self.execute(*command)

    def copySidecar(self, source, dest, removesidecar=False):
        """
        Copy all metadata and icc profile from sidecar to image file.
        if removesidecar is True (default False), the sidecar file is removed after copying.
        Should be called only while editing the file.
        @param source: path to sidecar file
        @type source: str
        @param dest: path to image file
        @type dest: str
        @param removesidecar: if True remove sidecar file after restoration. Default is False
        @type removesidecar: bool
        @return True if sidecar file exists, False otherwise
        @rtype bool
        """
        sidecar = source[:-4] + '.mie'
        if isfile(sidecar):
            # copy metadata from sidecar to image file
            # following exif doc, wild cards do not copy icc_profile : we must specify it explicitly
            command = ["-tagsFromFile", sidecar, "-all", "-icc_profile", "-overwrite_original", dest]
            self.execute(*command)
        else:
            return False
        if removesidecar:
            os.remove(sidecar)
        return True

    def readBinaryData(self, f, tagname='xx'):
        """
        Read binary metadata value of tagname from an image file or
        a raw iamge file or a sidecar file.
        @param f: path to file
        @type f: str
        @param tagname:
        @type tagname: str
        @return: data
        @rtype: bytes
        """
        command = ['-b', '-m', '-' + tagname, f]
        buf = self.execute(*command, ascii=False)  # -m : disables output of warnings
        return bytes(buf)

    def readBinaryDataAsDict(self, f, taglist=None):
        """
        Read tag values from a list of tag names in an image file,
        a raw imamge file, a sidecar file or a dng/dcp profile.
        tag values can be binary data or strings.
        The method returns a dictionary of (str) decoded buffers.
        @param f: file name
        @type f: str
        @param taglist: tag names
        @type taglist: list of str
        @return: data
        @rtype: dict of str
        """
        d = {}
        if taglist is None:
            return d
        for tagname in taglist:
            command = ['-b', '-m', '-' + tagname, f]  # -m : disables output of warnings
            buf = self.execute(*command, ascii=False)
            # decode to str
            d[tagname] = buf.decode()
        return d

    def get_thumbNail(self, f, thumbname='thumbnailimage'):
        """
        Extract the (jpg) thumbnail from an image or sidecar file
        and returns it as a QImage.
        @param f: path to image or sidecar file
        @type f: str
        @param thumbname: tag name
        @type thumbname: str
        @return: thumbnail
        @rtype: QImage
        """
        thumbnail = self.readBinaryData(f, tagname=thumbname)
        return QImage.fromData(QByteArray.fromRawData(thumbnail), 'JPG')  # Pyside2 fromRawData takes 1 arg only

    def writeThumbnail(self, filename, thumbfile):
        """
        Write a bytearray containing thumbnail data to an image
        or sidecar file. Thumbnail data should be a valid jpeg image
        with dimensions 160x120 or 120x160.
        For an image file, should be called only while editing the file.
        @param filename: path to image or sidecar file
        @type filename: str
        @param thumbfile: path to thumbnail jpg file
        @type thumbfile: str
        """
        command = [filename, '-overwrite_original'] + ['-%s<=%s' % ('thumbnailimage', thumbfile)]
        self.execute(*command)

    def writeOrientation(self, filename, value):
        """
        Writes orientation tag to file (image or sidecar).
        For an image file, should be called only while editing the file.
        @param filename: path to file
        @type filename: str
        @param value: orientation code (range 1..8)
        @type value: str or int
        """
        command = ['-%s=%s' % ('Orientation', value)] + ['-n'] + [filename, '-overwrite_original']
        self.execute(*command)

    def readXMPTag(self, filename, tagName, ext='.mie'):
        """
        Read a tag from a sidecar (.mie) file. Despite its name, the method can read
        a tag of any type. A ValueError exception is raised if the file does not exist.
        @param filename: image or sidecar path
        @type filename: str
        @param tagName:
        @type tagName: str
        @param ext:
        @type ext: str
        @return: tag info
        @rtype: str
        """
        filename = filename[:-4] + ext
        if not isfile(filename):
            raise ValueError
        command = ['-%s' % tagName] + [filename]
        res = self.execute(*command)
        return res

    ###############################
    # The two next methods create
    # the sidecar if it does not exist
    ###############################
    def writeXMPTag(self, filename, tagName, value):
        """
        Write a tag to a sidecar (.mie) file. If the sidecar
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
            self.createSidecar(filename)
        # write tag to sidecar
        command = ['-%s=%s' % (tagName, value)] + [fmie, '-overwrite_original']
        self.execute(*command)

    def get_metadata(self, f, tags=None, createsidecar=True):
        """
        Read metadata from file : data are read
        from the image file and the sidecar file is created if
        createsidecar is True (default).
        @param f: file name
        @type f: str
        @param tags:
        @type tags:
        @param createsidecar: flag
        @type createsidecar: bool
        @return: profile, metadata
        @rtype: 2-uple profile: bytes, metadata: dict
        """
        # Using PIL _getexif is simpler.
        # However, exiftool is much more powerful
        if tags is None:
            flags = ["-j", "-a", "-n", "-S"]  # -j = json export format
        else:
            flags = ["-j", "-n", "-S"] + ['-' + tag for tag in tags]
        # try to extract profile
        extract_meta_flags = ["-icc_profile", "-b"]
        command = extract_meta_flags + [f]
        profile = self.execute(*command, ascii=False)
        # extract tags
        command = flags + [f]
        data = json.loads(self.execute(*command))
        # create sidecar file
        if createsidecar:
            self.createSidecar(f)
        # data is a length 1 list
        return profile, data[0]

    def get_formatted_metadata(self, f):
        """
        read all metadata from file f and return
        a formatted string.
        @param f: path to file
        @type f: str
        @return:
        @rtype: str
        """
        command = ["-a", "-G0:1", "--ImageDescription",
                   f]  # remove Imagej tag from output, G0:1 add group names to output
        out = self.execute(*command)
        return out


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
    elif value == 1:
        pass
    elif value == 6:  # TODO complete
        tr.rotate(90)  # clockwise
    elif value == 8:
        tr.rotate(-90)  # counterclockwise
    else:
        raise ValueError("decodeExifOrientation : unhandled orientation tag: %d" % value)
    return tr


def readExpTime(filename):
    """
    Convenience method. It returns the exposure time.
    @param filename: path to image file
    @type filename: str
    @return: exposure time
    @rtype: float
    """
    p = re.compile('[0-9]+/?[0-9]+')
    with ExifTool() as e:
        expTime = e.readXMPTag(filename, 'ExposureTime', ext='.jpg')
    s = p.findall(expTime)
    # time may be recorded as a fraction, so we use eval
    return eval(s[0])


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
