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

from PIL import Image
from PIL.ImageCms import profileToProfile, getOpenProfile, get_display_profile, getProfileDescription, getProfileInfo, buildTransformFromOpenProfiles, applyTransform
#from PIL.ImageQt import ImageQt
import numpy as np
from PySide2.QtGui import QPixmap, QImage
from imgconvert import QImageBuffer, PilImageToQImage, QImageToPilImage
#from StringIO import StringIO
from os import path
#help(PIL.ImageCms)
from settings import ADOBE_RGB_PROFILE_PATH, SRGB_PROFILE_PATH
# global flag
COLOR_MANAGE = True


monitorProfile = get_display_profile()
monitorProfile.info = getProfileInfo(monitorProfile)
workingProfile = getOpenProfile(SRGB_PROFILE_PATH)
workingProfile.info = getProfileInfo(workingProfile)

# ICC transform (type : Cms transform object plus 2 dynamic attributes)
# TODO workToMonTransform should be an image attribute
workToMonTransform = buildTransformFromOpenProfiles(workingProfile, monitorProfile, "RGB", "RGB")
workToMonTransform.fromProfile  = workingProfile
workToMonTransform.toProfile = monitorProfile

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
    @param image: source QImage
    @param transformation : a CmsTransform object (default workToMonTransform)
    @return: The converted QImage
    """
    # sRGB
    if transformation is not None: #  or image.meta.colorSpace == 1
        # conversion in the PIL context
        converted_image = applyTransform(QImageToPilImage(image), transformation, 0)
        image.colorTransformation = transformation
        return PilImageToQImage(converted_image)
    else :
        return image