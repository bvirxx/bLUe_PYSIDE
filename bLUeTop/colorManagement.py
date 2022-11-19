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
from PIL.ImageCms import getOpenProfile, getProfileInfo, \
    buildTransformFromOpenProfiles, buildProofTransformFromOpenProfiles, applyTransform, INTENT_PERCEPTUAL, \
    INTENT_ABSOLUTE_COLORIMETRIC, INTENT_RELATIVE_COLORIMETRIC, \
    FLAGS, ImageCmsProfile, PyCMSError, core
from PySide2.QtGui import QImage, QColorSpace
import bLUeTop.Gui
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


def getProfile_QCS(path):
    """
    gets QColorSpace instance from file

    :param path:
    :type path: str
    :rtype: Union[QColorSpace, None]
    """
    profile = None
    try:
        with open(path, 'rb') as pf:
            profile = QColorSpace.fromIccProfile(pf.read())
        if not profile.isValid():
            raise ValueError
    except (IOError, ValueError):
        raise
    return profile


def get_default_working_profile():
    """
    try to find a default image profile.

    :return: profile
    :rtype: ImageCmsProfile
    """
    try:
        profile = getOpenProfile(SRGB_PROFILE_PATH)
    except PyCMSError:
        dlgWarn('No valid sRGB color profile found.\nSet SYSTEM_PROFILE_DIR and SRGB_PROFILE_NAME in your config.json',
                info='Invalid profile %s' % SRGB_PROFILE_PATH)
        sys.exit()
    return profile


def get_default_working_profile_QCS():
    """
    try to find a default image profile.

    :return: profile
    :rtype: QColorSpace instance
    """
    path = SRGB_PROFILE_PATH
    try:
        profile = getProfile_QCS(path)
    except (IOError, ValueError):
        dlgWarn('No valid sRGB color profile found.\nSet SYSTEM_PROFILE_DIR and SRGB_PROFILE_NAME in your config.json',
                info='Invalid profile %s' % path)
        sys.exit()
    return profile


def get_default_monitor_profile():
    """
    try to find a default monitor profile.

    :return: profile or None
    :rtype: ImageCmsProfile or None
    """
    profile = None
    try:
        profile = getOpenProfile(DEFAULT_MONITOR_PROFILE_PATH)
    except PyCMSError:
        pass
    return profile


def get_default_monitor_profile_QCS():
    """
    try to find a default monitor profile.

    :return: profile or None
    :rtype: ImageCmsProfile or None
    """
    profile = None
    try:
        profile = getProfile_QCS(DEFAULT_MONITOR_PROFILE_PATH)
    except (IOError, ValueError):
        pass
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

    # a (default) working image profile is always needed, at least for RGB<-->XYZ conversions
    defaultWorkingProfile = get_default_working_profile()
    defaultWorkingProfile_QCS = get_default_working_profile_QCS()

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
            profile_path = core.get_display_profile_win32(handle, 1)
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
                dlgWarn('Cannot detect monitor profile', info=str(e), parent=bLUeTop.Gui.window)
        return profile_path

    @staticmethod
    def B_get_display_profile(handle=None, device_id=None):
        """
        Returns the display ImageCmsProfile instance

        :param handle: screen handle (Windows)
        :type handle: int
        :param device_id: name of display
        :type device_id: str
        :return: monitor profile or None
        :rtype: ImageCmsProfile or None
        """
        profile_path = icc.B_get_display_profilePath(handle=handle, device_id=device_id)
        try:
            Cms_profile = getOpenProfile(profile_path)
        except PyCMSError:
            Cms_profile = get_default_monitor_profile()
        return Cms_profile

    @staticmethod
    def B_get_display_profile_QCS(handle=None, device_id=None):
        """
        Returns the display QColorSpace instance

        :param handle: screen handle (Windows)
        :type handle: int
        :param device_id: name of display
        :type device_id: str
        :return: monitor profile or None
        :rtype: Union[QColorSpace, None]
        """
        profile_path = icc.B_get_display_profilePath(handle=handle, device_id=device_id)
        try:
            with open(profile_path, 'rb') as ppf:
                profile = QColorSpace.fromIccProfile(ppf.read())
        except IOError:
            with open(DEFAULT_MONITOR_PROFILE_PATH, 'rb') as ppf:
                profile = QColorSpace.fromIccProfile(ppf.read())
        if profile.isValid():
            return profile
        else:
            return None

    @classmethod
    def getMonitorProfile(cls, qscreen=None):
        """
        Try to retrieve the default color profile
        associated to the monitor specified by QScreen
        (the system main display if qscreen is None).
        The method returns None if no profile can be found.

        :param qscreen: QScreen instance
        :type qscreen: QScreen
        :return: monitor profile or None
        :rtype: CmsProfile or None
        """
        monitorProfile = None
        # detecting profile
        if qscreen is not None:
            try:
                if sys.platform == 'win32':
                    dc = win32gui.CreateDC('DISPLAY', str(qscreen.name()), None)
                    monitorProfile = cls.B_get_display_profile(handle=dc)
                else:
                    monitorProfile = cls.B_get_display_profile(device_id=qscreen.name())
            except (RuntimeError, OSError, TypeError):
                pass
        return monitorProfile

    @classmethod
    def getMonitorProfile_QCS(cls, qscreen=None):
        """
        Try to retrieve the default color profile
        associated to the monitor specified by QScreen
        (the system main display if qscreen is None).
        The method returns None if no profile can be found.

        :param qscreen: QScreen instance
        :type qscreen: QScreen
        :return: monitor profile or None
        :rtype: Union[QColorSpace, None]
        """
        monitorProfile = None
        # detecting profile
        if qscreen is not None:
            try:
                if sys.platform == 'win32':
                    dc = win32gui.CreateDC('DISPLAY', str(qscreen.name()), None)
                    monitorProfile = cls.B_get_display_profile_QCS(handle=dc)
                else:
                    monitorProfile = cls.B_get_display_profile_QCS(device_id=qscreen.name())
            except (RuntimeError, OSError, TypeError):
                pass
        return monitorProfile

    @classmethod
    def configure(cls, qscreen=None, colorSpace=-1, workingProfile=None, softproofingwp=-1):
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
        cls.workingProfileInfo = getProfileInfo(cls.workingProfile)
        if not COLOR_MANAGE_OPT:
            return
        # looking for specific profiles
        try:
            # get monitor profile as CmsProfile object.
            if qscreen is not None:
                cls.monitorProfile = cls.getMonitorProfile(qscreen=qscreen)
                if cls.monitorProfile is None:  # not handled by PIL
                    raise ValueError
                # get profile info, a PyCmsError exception is raised if monitorProfile is invalid
                cls.monitorProfileInfo = getProfileInfo(cls.monitorProfile)
            # get working profile as CmsProfile object
            if colorSpace == 1:
                cls.workingProfile = cls.defaultWorkingProfile  # getOpenProfile(SRGB_PROFILE_PATH)
            elif colorSpace == 2:
                cls.workingProfile = getOpenProfile(ADOBE_RGB_PROFILE_PATH)
            elif type(workingProfile) is ImageCmsProfile:
                cls.workingProfile = workingProfile
            else:
                cls.workingProfile = getOpenProfile(SRGB_PROFILE_PATH)  # default

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
                cls.workToMonTransform = buildTransformFromOpenProfiles(cls.workingProfile, cls.monitorProfile,
                                                                        "RGB", "RGB", renderingIntent=INTENT_PERCEPTUAL)
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
        except (ValueError, TypeError, PyCMSError):
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    @classmethod
    def configure_QCS(cls, qscreen=None, colorSpace=-1, workingProfile=None, softproofingwp=-1):
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

        cls.workingProfile = cls.defaultWorkingProfile_QCS
        # cls.workingProfileInfo = getProfileInfo(cls.workingProfile)
        if not COLOR_MANAGE_OPT:
            return
        # looking for specific profiles
        try:
            # get monitor profile as CmsProfile object.
            if qscreen is not None:
                cls.monitorProfile = cls.getMonitorProfile_QCS(qscreen=qscreen)
                if cls.monitorProfile is None:  # not handled by PIL
                    raise ValueError
                # get profile info, a PyCmsError exception is raised if monitorProfile is invalid
                # cls.monitorProfileInfo = getProfileInfo(cls.monitorProfile)
            # get working profile as CmsProfile object
            if colorSpace == 1:
                cls.workingProfile = cls.defaultWorkingProfile_QCS  # getOpenProfile(SRGB_PROFILE_PATH)
            elif colorSpace == 2:
                cls.workingProfile = getProfile_QCS(ADOBE_RGB_PROFILE_PATH)
            elif type(workingProfile) is QColorSpace:
                cls.workingProfile = workingProfile
            else:
                cls.workingProfile = getProfile_QCS(SRGB_PROFILE_PATH)  # default

            # cls.workingProfileInfo = getProfileInfo(cls.workingProfile)
            # init CmsTransform object : working profile ---> monitor profile
            if softproofingwp == -1:
                softproofingwp = cls.softProofingProfile  # default : do not change the current soft proofing mode
            if type(softproofingwp) is QColorSpace:
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
                cls.workToMonTransform = cls.workingProfile.transformationToColorSpace(cls.monitorProfile)
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
        except (ValueError, TypeError, PyCMSError):
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise


def cmsConvertQImage(image, cmsTransformation=None):
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
    image = image.copy()
    buf = QImageBuffer(image)[:, :, :3][:, :, ::-1]
    # convert to the PIL context and apply cmsTransformation
    bufC = np.ascontiguousarray(buf)
    PIL_img = Image.frombuffer('RGB', (image.width(), image.height()), bufC, 'raw',
                               'RGB', 0, 1)  # these 3 weird parameters are recommended by a runtime warning !!!
    applyTransform(PIL_img, cmsTransformation, 1)  # 1=in place
    # back to the image buffer
    buf[...] = np.frombuffer(PIL_img.tobytes(), dtype=np.uint8).reshape(buf.shape)
    return image


icc.configure = icc.configure_QCS
