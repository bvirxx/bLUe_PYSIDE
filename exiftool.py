import subprocess
import os
import json
from PyQt4.QtGui import QTransform

class ExifTool(object):
    """
    exiftool wrapper
    """

    sentinel = "{ready}"
    # exiftool flags
    # -n : print numerical values
    # -j : json output
    # -a : extarct duplicate tags
    # -S : very short output format
    # -G0 : print group name for each tag
    flags = ["-j", "-a", "-n", "-S", "-G0", "-Orientation", "-ColorSpace", "-InteropIndex", "-WhitePoint", "-PrimaryChromaticities", "-Gamma", "-ICC_Profile:all"]

    def __init__(self, executable = "H:\standalone\exiftool\exiftool(-k)"):
        self.executable = executable

    def __enter__(self):
        self.process = subprocess.Popen(
                                        [self.executable, "-stay_open", "True",  "-@", "-"],
                                        stdin =subprocess.PIPE, stdout =subprocess.PIPE, stderr =subprocess.STDOUT
                                       )
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

    def get_metadata(self, *filenames):
        #return json.loads(self.execute("-G", "-j", "-n", *filenames))
        print self.flags+list(filenames)
        return json.loads(self.execute(*(self.flags+list(filenames))))


def decodeExifOrientation(value):
    tr = QTransform()
    if value == 1 :
        pass
    elif value == 6:
        tr.rotate(90)
    else :
        print "unhandled orientation tag: ", value
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