"""
Copyright (C) 2017  Bernard Virot

PeLUT - Photo editing software using adjustment layers with 1D and 3D Look Up Tables.

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

from PIL import Image
from PIL.ImageCms import profileToProfile, getOpenProfile, get_display_profile, getProfileDescription, getProfileInfo, buildTransformFromOpenProfiles, applyTransform
from PIL.ImageQt import ImageQt
import numpy as np
from PyQt4.QtGui import QPixmap, QImage
from imgconvert import QImageBuffer, PilImageToQImage, QImageToPilImage
from StringIO import StringIO
from os import path
#help(PIL.ImageCms)

# global flag
COLOR_MANAGE = True

###########################
# Paths to installed profiles
ADOBE_RGB_PROFILE_PATH = "C:\Windows\System32\spool\drivers\color\AdobeRGB1998.icc"
SRGB_PROFILE_PATH = "C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm"
MONITOR_PROFILE_PATH = "C:\Windows\System32\spool\drivers\color\CS240(34381075)00000002.icc"  #photography CS240 calibration
#############################

monitorProfile = get_display_profile()
MONITOR_PROFILE_INFO=getProfileInfo(monitorProfile)

workingProfile = getOpenProfile(SRGB_PROFILE_PATH)



# ICC transform (type : Cms transform object)
workToMonTransform = buildTransformFromOpenProfiles(workingProfile, monitorProfile, "RGB", "RGB")

def getProfiles():
    profileDir = "C:\Windows\System32\spool\drivers\color"
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(profileDir) if isfile(join(profileDir, f)) and ('icc' in f or 'icm' in f)]
    desc = [getProfileDescription(join(profileDir, f)) for f in onlyfiles]
    profileList = zip(onlyfiles, desc)
    return profileList

def convertQImage(image, transformation = workToMonTransform):
    """
    Convert a QImage from profile fromProfile to profile toProfile.
    :param image: source QImage
    :param transformation : a CmsTransform object
    :return: The converted QImage
    """
    # conversion in the PIL context
    converted_image = applyTransform(QImageToPilImage(image), transformation, 0)
    return PilImageToQImage(converted_image)