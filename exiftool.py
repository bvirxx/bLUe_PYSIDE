"""
Copyright (C) 2017  Bernard Virot

bLUe - Photo editing software.

With Blue you can enhance and correct the colors of your photos in a few clicks.
No need for complex tools such as lasso, magic wand or masks.
bLUe interactively constructs 3D LUTs (Look Up Tables), adjusting the exact set
of colors you want.

3D LUTs are widely used by professional film makers, but the lack of
interactive tools maked them poorly useful for photo enhancement, as the shooting conditions
can vary widely from an image to another. With bLUe, in a few clicks, you select the set of
colors to modify, the corresponding 3D LUT is automatically built and applied to the image.
You can then fine tune it as you want.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
# the code of the class ExifTool was taken from the Stackoverflow answer http://stackoverflow.com/questions/10075115/call-exiftool-from-a-python-script
# We gratefully acknoledge the contribution of the author.

import subprocess
import os
import json
from PyQt4.QtGui import QTransform, QMessageBox
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
    flags = ["-j", "-a", "-n", "-S", "-G0", "-Orientation", "-ProfileDescription", "-colorSpace", "-InteropIndex", "-WhitePoint", "-PrimaryChromaticities", "-Gamma"]#, "-ICC_Profile:all"]
    extract_meta_flags = ["-icc_profile", "-b"] #["-b"] #["-icc_profile", "-b"]
    #copy_meta_flags = ["-tagsFromFile", "-all:all"]

    def __init__(self, executable = EXIFTOOL_PATH):
        self.executable = executable

    # enter/exit "with" block
    def __enter__(self):
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
        self.process.stdin.write("-stay_open\nFalse\n")
        self.process.stdin.flush()

    def execute(self, *args):
        args = args + ("-execute\n",)
        self.process.stdin.write(str.join("\n", args))
        self.process.stdin.flush()
        output = ""
        fd = self.process.stdout.fileno()
        while not output[:-2].endswith(self.sentinel):
            output += os.read(fd, 4096)
        return output[:-len(self.sentinel)-2]

    def get_metadata(self, f, save=True):
        profile=self.execute(*(self.extract_meta_flags + [f]))
        if save:
            self.saveMetadata(f)
        return profile, json.loads(self.execute(*(self.flags + [f])))

    def saveMetadata(self, f):
        """
        save all metadata and icc profile to sidecar .mie files.
        :param f: arbitrary number of file names to process
        """
        self.execute(*(["-tagsFromFile", f, "-all:all", "-icc_profile", f+".mie"]))

    def restoreMetadata(self, source, dest, removesidecar=False):
        """
        restore all metadata and icc profile from sidecar .mie to image files.
        if removesidecar is True, the sidecar file is removed after
        restoration.
        :param source: file the image was loaded from
        :param dest: file the image is saved to
        :param removesidecar: if True (default) remove sidecar file after restoration
        """
        sidecar = source + '.mie'
        self.execute(*(["-tagsFromFile", sidecar, "-all:all", "-icc_profile", dest]))
        if removesidecar:
            os.remove(sidecar)

def decodeExifOrientation(value):
    """
    Returns a QTransform object representing the
    image transformation corresponding to the orientation tag value
    :param value: orientation tag
    :return: Qtransform object
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