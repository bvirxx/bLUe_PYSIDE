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

import numpy as np
from PIL import Image

from PIL.ImageCms import getOpenProfile, get_display_profile, getProfileInfo, \
    buildTransformFromOpenProfiles, applyTransform, INTENT_PERCEPTUAL, ImageCmsProfile
from PySide2.QtGui import QImage

from bLUeGui.bLUeImage import QImageBuffer
from compat import PilImgToRaw
from debug import tdec

from settings import SRGB_PROFILE_PATH, ADOBE_RGB_PROFILE_PATH

if sys.platform == 'win32':
    import win32gui


def PilImageToQImage(pilimg) :
    """
    Converts a PIL image (mode RGB) to a QImage (format RGB32)
    @param pilimg: The PIL image, mode RGB
    @type pilimg: PIL image
    @return: the converted image
    @rtype: QImage
    """
    ############################################
    # CAUTION: PIL ImageQt causes a memory leak!!!
    # return ImageQt(pilimg)
    ############################################
    im_data = PilImgToRaw(pilimg)
    Qimg = QImage(im_data['im'].size[0], im_data['im'].size[1], im_data['format'])
    buf = QImageBuffer(Qimg).ravel()
    buf[:] = np.frombuffer(im_data['data'], dtype=np.uint8)
    return Qimg


def QImageToPilImage(qimg) :
    """
    Converts a QImage (format ARGB32or RGB32) to a PIL image
    @param qimg: The Qimage to convert
    @type qimg: Qimage
    @return: PIL image  object, mode RGB
    @rtype: PIL Image
    """
    a = QImageBuffer(qimg)
    if (qimg.format() == QImage.Format_ARGB32) or (qimg.format() == QImage.Format_RGB32):
        # convert pixels from BGRA or BGRX to RGB
        a = np.ascontiguousarray(a[:,:,:3][:,:,::-1]) #ascontiguousarray is mandatory to speed up Image.fromArray (x3)
    else :
        raise ValueError("QImageToPilImage : unrecognized format : %s" %qimg.Format())
    return Image.fromarray(a)


class icc:
    """
    Container for color management related flags and methods.
    Should never be instantiated
    """
    HAS_COLOR_MANAGE = False  # menu action "color manage" will be disabled
    COLOR_MANAGE = False  # no color management
    monitorProfile, workingProfile, workToMonTransform = (None,)*3
    workingProfileInfo, monitorProfileInfo = '', ''


    @classmethod
    def configure(cls, qscreen=None, colorSpace=-1, workingProfile=None):
        """
        Try to configure color management for the monitor
        specified by QScreen, and build an image transformation
        from the working profile (default sRGB) to the monitor profile.
        This transformation is convenient to match image colors to screen colors.
        @param qscreen: QScreen instance
        @type qscreen: QScreen
        """
        try:
            # get monitor profile as CmsProfile object.
            if qscreen is not None:
                cls.monitorProfile = cls.getMonitorProfile(qscreen=qscreen)
                # get profile info, a PyCmsError exception is raised if monitorProfile is invalid
                cls.monitorProfileInfo = getProfileInfo(cls.monitorProfile)
            # get working profile as CmsProfile object
            if colorSpace == 1:
                cls.workingProfile = getOpenProfile(SRGB_PROFILE_PATH)
            elif colorSpace==2:
                cls.workingProfile = getOpenProfile(ADOBE_RGB_PROFILE_PATH)
            elif type(workingProfile) is ImageCmsProfile:
                cls.workingProfile = workingProfile
            else:
                cls.workingProfile = getOpenProfile(SRGB_PROFILE_PATH)  # default

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
        except (OSError, IOError) as e:
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
    Caution: The format is kept, but the alpha chanel is not preserved.
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
        # convert back to QImage
        img = PilImageToQImage(converted_image)
        # restore format
        return img.convertToFormat(image.format())
    else :
        return image