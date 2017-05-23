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

import subprocess
import os
import json
from tempfile import NamedTemporaryFile

from PySide.QtGui import QTransform, QMessageBox
from os.path import isfile

from settings import EXIFTOOL_PATH

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
            self.process = subprocess.Popen(
                                        [self.executable, "-stay_open", "True",  "-@", "-"],
                                        stdin =subprocess.PIPE, stdout =subprocess.PIPE, stderr =subprocess.STDOUT
                                       )
        except OSError:
            msg = QMessageBox()
            msg.setText("cannot execute exiftool :\nset EXIFTOOL_PATH in file settings.py")
            msg.exec_()
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
        self.process.stdin.write("-stay_open\nFalse\n")
        self.process.stdin.flush()
        self.process.terminate()

    def execute(self, *args):
        args = args + ("-execute\n",)
        self.process.stdin.write(str.join("\n", args))
        self.process.stdin.flush()
        output = ""
        fd = self.process.stdout.fileno()
        while not output[:-2].endswith(self.sentinel):
            output += os.read(fd, 4096)
        return output[:-len(self.sentinel)-2]

    def writeXMPTag(self, filename, tagName, value):
        """
        
        @param filename: 
        @type filename: str
        @param tagName: tag name
        @type tagName: str
        @param value: tag value
        @type value: str or number
        @return: 
        """
        filename = filename[:-4] + '.mie'
        self.execute(*(['-%s=%s' % (tagName, value)] + [filename]))

    def readXMPTag(self, filename, tagName):
        """
        Reads tag info from the sidecar (.mie) file. Despite its name, the method can read
        any type of tag. The file extension is modified.
        @param filename: 
        @param tagName: 
        @return: 
        """
        filename = filename[:-4] + '.mie'
        res = self.execute(*(['-%s' % tagName] + [filename]))
        return res

    def get_metadata(self, f):
        """
        Reads metadata from file : data are read from the sidecar 
        .mie file if it exists, otherwise data are read
        from the image file and a sidecar file is created.
        @param f: file name
        @type f: str
        @return: profile, metadata
        @rtype: 
        """
        fmie = f[:-4]+'.mie'
        if isfile(fmie):
            profile = self.execute(*(self.extract_meta_flags + [fmie]))
            data = json.loads(self.execute(*(self.flags + [fmie])))
        else:
            profile = self.execute(*(self.extract_meta_flags + [f]))
            data = json.loads(self.execute(*(self.flags + [f])))
            # write sidecar file
            self.saveMetadata(f)
        return profile, data

    def saveMetadata(self, f):
        """
        save all metadata and icc profile to sidecar .mie file
        @param f: file name to process
        """
        #temp = NamedTemporaryFile(delete=False)
        #temp.close()
        #self.execute(*(["-tagsFromFile", f, "-all:all", "-overwrite_original", "-icc_profile", f[:-4] + ".mie"]))
        self.execute(*(["-tagsFromFile", f, "-overwrite_original", f[:-4] + ".mie"]))

    def restoreMetadata(self, source, dest, removesidecar=False):
        """
        Copy all metadata and icc profile from sidecar .mie to image file.
        if removesidecar is True, the sidecar file is removed after copying.
        @param source: file the image was loaded from
        @param dest: file the image is saved to
        @param removesidecar: if True remove sidecar file after restoration. Default is False
        @return True if sidecar file exists, False otherwise
        """
        sidecar = source[:-4] + '.mie'
        if isfile(sidecar):
            #self.execute(*(["-tagsFromFile", sidecar, "-all:all", "-overwrite_original", "-icc_profile", dest]))
            self.execute(*(["-tagsFromFile", sidecar, "-overwrite_original", dest]))
        else:
            return False
        if removesidecar:
            os.remove(sidecar)
        return True

def decodeExifOrientation(value):
    """
    Returns a QTransform object representing the
    image transformation corresponding to the orientation tag value
    @param value: orientation tag
    @return: Qtransform object
    """
    # identity transformation
    tr = QTransform()
    if value == 0:
        pass
    elif value == 1 :
        pass
    elif value == 6:   # TODO complete
        tr.rotate(90)
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