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
import win32api
import win32gui
import numpy as np
from ctypes.wintypes import HWND

from PIL.ImageCms import getOpenProfile, get_display_profile, getProfileDescription, getProfileInfo, \
            buildTransformFromOpenProfiles, applyTransform, INTENT_RELATIVE_COLORIMETRIC
from PIL.ImageWin import HDC
from PySide2.QtCore import QByteArray
from PySide2.QtGui import QGuiApplication
from PySide2.QtWidgets import QWidget

from imgconvert import PilImageToQImage, QImageToPilImage

from settings import SRGB_PROFILE_PATH, SYSTEM_PROFILE_PATH

class icc:
    HAS_COLOR_MANAGE = False  # menu action "color manage" will be disabled
    COLOR_MANAGE = False  # no color management
    monitorProfile, workingProfile, workToMonTransform = None, None, None
    @classmethod
    def configure(cls, qscreen=None):
        try:
            # get monitor profile as CmsProfile object, None if profile not found
            if qscreen is not None:
                #device = win32api.EnumDisplayDevices(Device=None, DevNum=0)
                #dc = win32gui.CreateDC(device.DeviceName, None, None)
                dc = win32gui.CreateDC(qscreen.name(), None, None)
                cls.monitorProfile = get_display_profile(HDC(dc))
            else:
                cls.monitorProfile = get_display_profile()
            # get profile info (type str).
            # a PyCmsError exception raised if monitorProfile is invalid
            cls.monitorProfile.info = getProfileInfo(cls.monitorProfile)
            # get working profile
            cls.workingProfile = getOpenProfile(SRGB_PROFILE_PATH)
            cls.workingProfile.info = getProfileInfo(cls.workingProfile)
            # init CmsTransform object : working profile ---> monitor profile
            cls.workToMonTransform = buildTransformFromOpenProfiles(cls.workingProfile, cls.monitorProfile,
                                                                     "RGB", "RGB", renderingIntent=INTENT_RELATIVE_COLORIMETRIC)
            cls.workToMonTransform.fromProfile = cls.workingProfile
            cls.workToMonTransform.toProfile = cls.monitorProfile
            """
                        INTENT_PERCEPTUAL            = 0 (DEFAULT) (ImageCms.INTENT_PERCEPTUAL)
                        INTENT_RELATIVE_COLORIMETRIC = 1 (ImageCms.INTENT_RELATIVE_COLORIMETRIC)
                        INTENT_SATURATION            = 2 (ImageCms.INTENT_SATURATION)
                        INTENT_ABSOLUTE_COLORIMETRIC = 3 (ImageCms.INTENT_ABSOLUTE_COLORIMETRIC)
            """
            cls.HAS_COLOR_MANAGE = (cls.monitorProfile is not None) and (cls.workingProfile is not None) and (cls.workToMonTransform is not None)
            cls.COLOR_MANAGE = cls.HAS_COLOR_MANAGE and cls.COLOR_MANAGE
        except:
            cls.COLOR_MANAGE = False
            # profile(s) missing : we will disable menu action "color manage".
            cls.HAS_COLOR_MANAGE = False
            cls.workToMonTransform = None

def getProfiles():
    profileDir = SYSTEM_PROFILE_PATH
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
        # convert to the PIL context and transform
        converted_image = applyTransform(QImageToPilImage(image), transformation, 0)  # time 0.65s for full res.
        image.colorTransformation = transformation
        return PilImageToQImage(converted_image)
    else :
        return image