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

from PIL import Image, ImageCms

from PySide6.QtGui import QImage

from bLUeTop import Gui
from PIL import Image, ImageCms

from PySide6.QtGui import QImage, QColorSpace, QColorTransform

from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.dialog import dlgWarn

from bLUeTop.settings import COLOR_MANAGE_OPT, SRGB_PROFILE_PATH, ADOBE_RGB_PROFILE_PATH, DEFAULT_MONITOR_PROFILE_PATH

# python-gi flag
HAS_GI = False

if COLOR_MANAGE_OPT:
    if sys.platform == 'win32':
        import win32gui
    else:
        try:
            from gi.repository import GLib, Gio, Colord
            HAS_GI = True
        except ImportError:
            pass
        if not HAS_GI:
            dlgWarn(
                "Automatic detection of monitor profile needs gi installed.\n trying to use %s instead" % DEFAULT_MONITOR_PROFILE_PATH)
            try:
                getOpenProfile(DEFAULT_MONITOR_PROFILE_PATH)
            except PyCMSError:
                dlgWarn("Invalid profile %s" % DEFAULT_MONITOR_PROFILE_PATH, info="Color management is disabled")


def getProfile(path, mode='Cms'):
    """
    Loads color profile from file.
    The parameter mode determines the type
    of the returned object.
    :raises Union[IOError, ValueError, ImageCms.PyCMSError]

    :param path:
    :type path: str
    :param mode: 'QCS' for QColorSpace or 'Cms' for CmsProfile
    :type mode: str
    :rtype: Union[QColorSpace, CmsProfile, None]
    """
    profile = None
    if mode == 'Cms':
        profile = ImageCms.getOpenProfile(path)
    elif mode == 'QCS':
        try:
            with open(path, 'rb') as pf:
                profile = QColorSpace.fromIccProfile(pf.read())
            if not profile.isValid():
                raise ValueError
        except:
            raise
    return profile


def get_default_working_profile(mode='Cms'):
    """
    tries to find a default image profile.
    The parameter mode determines the type
    of the returned object.

    :return: profile
    :rtype: Union[QColorSpace, CmsProfile, None]
    """
    path = SRGB_PROFILE_PATH
    try:
        profile = getProfile(path, mode=mode)
    except (ImageCms.PyCMSError, ValueError, IOError):
        dlgWarn(
            'No valid sRGB color profile found.\nSet SYSTEM_PROFILE_DIR and SRGB_PROFILE_NAME in your config.json',
            info='Invalid profile %s' % path)
        sys.exit()
    return profile


class icc:
    """
    Container for color management related options and methods.
    Should never be instantiated.
    """
    HAS_COLOR_MANAGE = False  # menu action "color manage" will be disabled
    COLOR_MANAGE = False  # no color management

    monitorProfile, workingProfile, workToMonTransform = (None,) * 3
    workingProfileInfo, monitorProfileInfo = '', ''

    softProofingProfile = None

    # a (default) working profile is always needed
    defaultWorkingProfile = get_default_working_profile(mode='Cms')  # CmsProfile
    defaultWorkingProfile_QCS = get_default_working_profile(mode='QCS')  # QColorSpace

    @staticmethod
    def get_default_monitor_profile(cls, mode='Cms'):
        """
        try to find a default monitor profile.

        :param mode: 'QCS or 'Cms'
        :type mode: str
        :return: monitor profile or None
        :rtype: Union[CmsProfile, QColorSpace, None]
        """
        profile = None
        try:
            profile = getProfile(DEFAULT_MONITOR_PROFILE_PATH, mode=mode)
        except (IOError, ValueError, ImageCms.PyCMSError):
            pass
        return profile

    @staticmethod
    def B_get_display_profilePath(handle=None, device_id=None):
        """
        bLUe version of ImageCms get_display_profile.

        :param handle: screen handle (Windows)
        :type handle: int
        :param device_id: name of display
        :type device_id: str
        :return: monitor profile path
        :rtype: str
        """

        profile_path = DEFAULT_MONITOR_PROFILE_PATH
        if sys.platform == "win32":
            profile_path = ImageCms.core.get_display_profile_win32(handle, 1)
        elif HAS_GI:
            try:
                GIO_CANCELLABLE = Gio.Cancellable.new()
                client = Colord.Client.new()
                client.connect_sync(GIO_CANCELLABLE)
                device = client.find_device_sync('xrandr-' + device_id, GIO_CANCELLABLE)
                device.connect_sync(GIO_CANCELLABLE)
                default_profile = device.get_default_profile()
                default_profile.connect_sync(GIO_CANCELLABLE)
                profile_path = default_profile.get_filename()
            except (NameError, ImportError, GLib.GError) as e:
                dlgWarn('Cannot detect monitor profile', info=str(e), parent=Gui.window)
        return profile_path

    @staticmethod
    def B_get_display_profile(handle=None, device_id=None):
        """
        Returns the display CmsProfile instance

        :param handle: screen handle (Windows)
        :type handle: int
        :param device_id: name of display
        :type device_id: str
        :return: monitor profile or None
        :rtype: Union[CmsProfile, None]
        """

        profile_path = icc.B_get_display_profilePath(handle=handle, device_id=device_id)
        try:
            Cms_profile = getOpenProfile(profile_path)
        except PyCMSError:
            Cms_profile = get_default_monitor_profile()
        return Cms_profile

    @classmethod
    def getMonitorProfile(cls, qscreen=None, mode='Cms'):
        """
        Tries to retrieve the current color profile
        associated with the monitor specified by QScreen
        (the system main display if qscreen is None).
        THe parameter mode determines the type of the returned
        value (CmsProfile or QColorSpace).
        The method returns None if no valid profile can be found.

        :param qscreen: QScreen instance
        :type qscreen: QScreen
        :param mode: 'QCS' or 'Cms'
        :type mode: str
        :return: monitor profile or None
        :rtype: Union[CmsProfile, QColorSpace, None]
        """

        monitorProfile = None
        # detect profile
        if qscreen is not None:
            try:
                if sys.platform == 'win32':
                    dc = win32gui.CreateDC('DISPLAY', str(qscreen.name()), None)
                    monitorProfile = cls.B_get_display_profile(handle=dc, mode=mode)
                else:
                    monitorProfile = cls.B_get_display_profile(device_id=qscreen.name(), mode=mode)
            except (RuntimeError, OSError, TypeError, ImageCms.PyCMSError):
                pass
            if isinstance(monitorProfile, QColorSpace):
                if not monitorProfile.isValid():
                    monitorProfile = None

        return monitorProfile

    @classmethod
    def configure(cls, qscreen=None, colorSpace=-1, workingProfile=None, softproofingwp=-1, useqcs=True):
        """
        Try to configure color management for the monitor
        specified by QScreen, and build an image transformation
        from the working profile (default sRGB) to the monitor profile.
        if softproofingwp is provided, toggle soft proofing mode off (if it is None) and
        on (if it is a valid ImageCmsProfile).

        :param qscreen: QScreen instance
        :type qscreen: QScreep
        :param colorSpace:
        :type colorSpace
        :param workingProfile:
        :type workingProfile: ImageCmsProfile
        :param softproofingwp: profile for the device to simulate
        :type softproofingwp:
        """
        cls.HAS_COLOR_MANAGE = False

        cls.workingProfile = cls.defaultWorkingProfile
        cls.workingProfileInfo = ImageCms.getProfileInfo(cls.workingProfile)
        cls.workingProfile_QCS = cls.defaultWorkingProfile_QCS

        if not COLOR_MANAGE_OPT:
            return
        # looking for specific profiles
        try:
            # get monitor profile as CmsProfile object.
            if qscreen is not None:
                cls.monitorProfile = cls.getMonitorProfile(qscreen=qscreen)
                if useqcs:
                    cls.monitorProfile_QCS = cls.getMonitorProfile(qscreen=qscreen, mode='QCS')
                if cls.monitorProfile is None:  # not handled by PIL
                    raise ValueError
                # get profile info, a PyCmsError exception is raised if monitorProfile is invalid
                cls.monitorProfileInfo = ImageCms.getProfileInfo(cls.monitorProfile)

            # get working profile as CmsProfile object : priority to known Color Spaces
                cls.monitorProfileInfo = ImageCms.getProfileInfo(cls.monitorProfile)

            # get working profile as CmsProfile object
            if colorSpace == 1:
                cls.workingProfile = cls.defaultWorkingProfile
            elif colorSpace == 2:
                cls.workingProfile = getProfile(ADOBE_RGB_PROFILE_PATH)
            elif isinstance(workingProfile, ImageCms.ImageCmsProfile):
                cls.workingProfile = workingProfile
            else:
                cls.workingProfile = getProfile(SRGB_PROFILE_PATH)  # default

            if useqcs:
                cls.workingProfile_QCS = QColorSpace.fromIccProfile(cls.workingProfile.tobytes())

            cls.workingProfileInfo = getProfileInfo(cls.workingProfile)
            # init CmsTransform object : working profile ---> monitor profile
            if softproofingwp == -1:
                softproofingwp = cls.softProofingProfile  # default : do not change the current soft proofing mode
            if type(softproofingwp) is ImageCmsProfile:
                cls.softProofingProfile = softproofingwp
                cls.workToMonTransform = buildProofTransformFromOpenProfiles(cls.workingProfile, cls.monitorProfile,
                                                                             softproofingwp,
                                                                             "RGB", "RGB",
                                                                             renderingIntent=INTENT_PERCEPTUAL,
                                                                             proofRenderingIntent=INTENT_RELATIVE_COLORIMETRIC,
                                                                             flags=FLAGS['SOFTPROOFING'] | FLAGS[
                                                                                 'BLACKPOINTCOMPENSATION'])  # | FLAGS['GAMUTCHECK'])
                cls.softProofingProfile = softproofingwp
            else:
                cls.workToMonTransform = ImageCms.buildTransformFromOpenProfiles(cls.workingProfile,
                                                                                 cls.monitorProfile,
                                                                                 "RGB",
                                                                                 "RGB",
                                                                                 renderingIntent=ImageCms.Intent.PERCEPTUAL
                                                                                 )
                cls.softProofingProfile = None
            """
                                    INTENT_PERCEPTUAL            = 0 (DEFAULT) (ImageCms.INTENT_PERCEPTUAL)
                                    INTENT_RELATIVE_COLORIMETRIC = 1 (ImageCms.INTENT_RELATIVE_COLORIMETRIC)
                                    INTENT_SATURATION            = 2 (ImageCms.INTENT_SATURATION)
                                    INTENT_ABSOLUTE_COLORIMETRIC = 3 (ImageCms.INTENT_ABSOLUTE_COLORIMETRIC)
            """
            cls.HAS_COLOR_MANAGE = (cls.monitorProfile is not None) and \
                                   (cls.workingProfile is not None) and (cls.workToMonTransform is not None)
            cls.COLOR_MANAGE = cls.HAS_COLOR_MANAGE and cls.COLOR_MANAGE
        except (OSError, IOError) as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        except (ValueError, TypeError, ImageCms.PyCMSError):
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    @staticmethod
    def cmsConvertQImage(image, cmsTransformation=None, inPlace=False):
        """
        Apply a Cms transformation to a copy of a QImage and
        return the transformed image.
        If cmsTransformation is None, the input image is returned (no copy).

        :param image: image to transform
        :type image: QImage
        :param cmsTransformation : Cms transformation
        :type cmsTransformation: ImageCmsTransform
        :return: The converted QImage
        :rtype: QImage
        """

        if cmsTransformation is None:
            return image
        if not inPlace:
            image = image.copy()
        buf = QImageBuffer(image)[:, :, :3][:, :, ::-1]
        # convert to the PIL context and apply cmsTransformation
        bufC = np.ascontiguousarray(buf)
        PIL_img = Image.frombuffer('RGB', (image.width(), image.height()), bufC, 'raw',
                                   'RGB', 0, 1)  # these 3 weird parameters are recommended by a runtime warning !!!
        ImageCms.applyTransform(PIL_img, cmsTransformation, 1)  # 1=in place
        # back to the image buffer
        buf[...] = np.frombuffer(PIL_img.tobytes(), dtype=np.uint8).reshape(buf.shape)
        return image

    @classmethod
    def convertQImage(cls, image, transformation=None, inPlace=False):
        """
        Apply a color transformation to a copy of a QImage and
        return the transformed image.
        If transformation is None, the input image is returned (no copy).

        :param image: image to transform
        :type image: QImage
        :param transformation: color transformation
        :type transformation:
        :return: The converted QImage
        :rtype: QImage
        """
        if type(transformation) is ImageCms.ImageCmsTransform:
            image = cls.cmsConvertQImage(image, cmsTransformation=transformation, inPlace=inPlace)
        return image
