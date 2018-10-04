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
import sys

from PIL.ImageCms import getOpenProfile, get_display_profile, getProfileInfo, \
    buildTransformFromOpenProfiles, applyTransform, INTENT_PERCEPTUAL
from PySide2.QtGui import QImage

from debug import tdec
from imgconvert import PilImageToQImage, QImageToPilImage
from settings import SRGB_PROFILE_PATH

if sys.platform == 'win32':
    import win32gui

class icc:
    """
    Container for color management related variables and methods.
    Never instantiated
    """
    HAS_COLOR_MANAGE = False  # menu action "color manage" will be disabled
    COLOR_MANAGE = False  # no color management
    monitorProfile, workingProfile, workToMonTransform = (None,)*3
    workingProfileInfo, monitorProfileInfo = '', ''

    @classmethod
    def configure(cls, qscreen=None):
        """
        Try to configure color management for the monitor
        specified by QScreen, and build an image transformation
        from the working profile (default sRGB) to the monitor profile.
        This transformation is convenient to match image colors to the screen.
        @param qscreen: QScreen instance
        @type qscreen: QScreen
        """
        try:
            # monitor profile
            cls.monitorProfile = cls.getMonitorProfile(qscreen=qscreen)
            # get profile info, a PyCmsError exception is raised if monitorProfile is invalid
            cls.monitorProfileInfo = getProfileInfo(cls.monitorProfile)
            # working profile
            cls.workingProfile = getOpenProfile(SRGB_PROFILE_PATH)
            cls.workingProfileInfo = getProfileInfo(cls.workingProfile)
            # init CmsTransform object : working profile ---> monitor profile
            cls.workToMonTransform = buildTransformFromOpenProfiles(cls.workingProfile, cls.monitorProfile,
                                                                     "RGB", "RGB", renderingIntent=INTENT_PERCEPTUAL)
            """
                                    INTENT_PERCEPTUAL            = 0 (DEFAULT) (ImageCms.INTENT_PERCEPTUAL)
                                    INTENT_RELATIVE_COLORIMETRIC = 1 (ImageCms.INTENT_RELATIVE_COLORIMETRIC)
                                    INTENT_SATURATION            = 2 (ImageCms.INTENT_SATURATION)
                                    INTENT_ABSOLUTE_COLORIMETRIC = 3 (ImageCms.INTENT_ABSOLUTE_COLORIMETRIC)
            """
            cls.HAS_COLOR_MANAGE = (cls.monitorProfile is not None) and (cls.workingProfile is not None) and (cls.workToMonTransform is not None)
            cls.COLOR_MANAGE = cls.HAS_COLOR_MANAGE and cls.COLOR_MANAGE
        except OSError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        except ValueError:
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    @classmethod
    def getMonitorProfile(cls, qscreen=None):
        """
        Try to retrieve the default color profile
        associated to the monitor specified by QScreen
        (the system main display if qscreen is None).
        The method returns None if no profile can be found.
        @param qscreen: QScreen instance
        @type qscreen: QScreen
        @return: monitor profile
        @rtype: CmsProfile
        """
        try:
            if qscreen is not None and sys.platform == 'win32':  #TODO added 04/10/18 validate
                dc = win32gui.CreateDC(qscreen.name(), None, None)
                monitorProfile = get_display_profile(dc)    # TODO modified 21/05/18 -- #HDC(dc))
                                                            # cf. imageCms.get_display_profile_win32 v5.1.0 patch
            else:
                monitorProfile = get_display_profile()
        except :
            monitorProfile = None
        return monitorProfile

def convertQImage(image, transformation=None):
    """
    Applies a Cms transformation to a QImage and returns the transformed image.
    If transformation is None, the input image is returned.
    Caution: The alpha chanel is not preserved.
    @param image: image
    @type image: QImage
    @param transformation : Cms transformation
    @type transformation: CmsTransform
    @return: The converted QImage
    @rtype: QImage
    """
    if transformation is not None:
        # convert to the PIL context and apply transformation
        converted_image = applyTransform(QImageToPilImage(image), transformation, 0)  # time 0.85s for a 15 Mpx image.
        return PilImageToQImage(converted_image)
    else :
        return image