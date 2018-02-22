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

from PIL.ImageCms import getOpenProfile, get_display_profile, getProfileDescription, getProfileInfo, \
            buildTransformFromOpenProfiles, applyTransform, INTENT_RELATIVE_COLORIMETRIC

from imgconvert import PilImageToQImage, QImageToPilImage

from settings import SRGB_PROFILE_PATH

###################
# Color management init
###################
# global flags
COLOR_MANAGE = False  # no color management
HAS_COLOR_MANAGE = False # menu action will be disabled

try:
    # get CmsProfile object, None if profile not known
    monitorProfile = get_display_profile()
    # get profile info (type str).
    # a PyCmsError exception raised if monitorProfile is invalid
    monitorProfile.info = getProfileInfo(monitorProfile)
    # get CmsProfile object and info from ICC profile name,
    # a PyCmsError exception is raised if invalid path
    workingProfile = getOpenProfile(SRGB_PROFILE_PATH)
    workingProfile.info = getProfileInfo(workingProfile)
    # get CmsTransform object from working profile to monitor profile
    workToMonTransform = buildTransformFromOpenProfiles(workingProfile, monitorProfile, "RGB", "RGB",
                                                        renderingIntent=INTENT_RELATIVE_COLORIMETRIC)
    workToMonTransform.fromProfile = workingProfile
    workToMonTransform.toProfile = monitorProfile
    """
                INTENT_PERCEPTUAL            = 0 (DEFAULT) (ImageCms.INTENT_PERCEPTUAL)
                INTENT_RELATIVE_COLORIMETRIC = 1 (ImageCms.INTENT_RELATIVE_COLORIMETRIC)
                INTENT_SATURATION            = 2 (ImageCms.INTENT_SATURATION)
                INTENT_ABSOLUTE_COLORIMETRIC = 3 (ImageCms.INTENT_ABSOLUTE_COLORIMETRIC)
    """
    COLOR_MANAGE = True
    HAS_COLOR_MANAGE = True
except:
    COLOR_MANAGE = False
    # profile(s) missing : we will
    # definitely disable menu action.
    HAS_COLOR_MANAGE = False
    workToMonTransform = None

def getProfiles():
    profileDir = "C:\Windows\System32\spool\drivers\color"
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(profileDir) if isfile(join(profileDir, f)) and ('icc' in f or 'icm' in f)]
    desc = [getProfileDescription(join(profileDir, f)) for f in onlyfiles]
    profileList = zip(onlyfiles, desc)
    return profileList

def convertQImage(image, transformation=None):
    """
    Convert a QImage from a profile to another. The transformation
    parameter is a Cms transformation object built from the input and
    output profiles.
    @param image: source QImage
    @type image: QImage
    @param transformation : Cms transformation
    @type transformation: CmsTransform
    @return: The converted QImage
    @rtype: QImage
    """
    if transformation is not None:
        # conversion to the PIL context
        converted_image = applyTransform(QImageToPilImage(image), transformation, 0)  # time 0.65s for full res.
        image.colorTransformation = transformation
        return PilImageToQImage(converted_image)
    else :
        return image