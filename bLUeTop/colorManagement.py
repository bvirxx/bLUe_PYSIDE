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

from PySide6.QtGui import QImage, QColorSpace, QColorTransform

from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.dialog import dlgWarn

from bLUeTop import Gui
from bLUeTop.settings import COLOR_MANAGE_OPT, SRGB_PROFILE_PATH, ADOBE_RGB_PROFILE_PATH, DEFAULT_MONITOR_PROFILE_PATH
from bLUeTop.utils import bLUeDialogCombo, getDisplayProfileWin, enumDisplayProfilesWin

HAS_WIN32 = False
HAS_GI = False
HAS_COLORSYNC = False

if COLOR_MANAGE_OPT:
    try:
        if sys.platform == 'win32':
            modulestr = 'pywin32'
            import win32gui
            HAS_WIN32 = True

        elif sys.platform == 'linux':
            modulestr = 'gi and colord'
            import gi
            gi.require_version('Colord', '1.0')
            from gi.repository import GLib, Gio, Colord
            HAS_GI = True

        elif sys.platform == 'darwin':
            modulestr = 'ColorSync'
            import Foundation , Quartz, ColorSync
            HAS_COLORSYNC = True

    except ImportError as e:
        dlgWarn(str(e))

    if not (HAS_WIN32 or HAS_GI or HAS_COLORSYNC):
        dlgWarn('Automatic detection of monitor profile needs %s installed' % modulestr,
                info='trying to use %s instead' % DEFAULT_MONITOR_PROFILE_PATH,
                parent=Gui.window
                )
        try:
            ImageCms.getOpenProfile(DEFAULT_MONITOR_PROFILE_PATH)
        except ImageCms.PyCMSError:
            dlgWarn('Invalid default monitor profile path %s' % DEFAULT_MONITOR_PROFILE_PATH,
                    info='Edit your configxxx.json file',
                    parent=Gui.window
                    )


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
        with open(path, 'rb') as pf:
            profile = QColorSpace.fromIccProfile(pf.read())
        if not profile.isValid():
            raise ValueError

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
        dlgWarn('No valid sRGB color profile found.\nSet SYSTEM_PROFILE_DIR and SRGB_PROFILE_NAME in config*.json',
                info='Invalid profile %s' % path,
                parent=None  # main window not shown yet: center on screen
                )
        sys.exit()
    return profile

def get_default_monitor_profile(mode='Cms'):
    """
    tries to find a default monitor profile.
    The parameter mode determines the type
    of the returned object.

    :return: profile
    :rtype: Union[QColorSpace, CmsProfile, None]
    """

    profile = None
    try:
        profile = getProfile(DEFAULT_MONITOR_PROFILE_PATH, mode=mode)
    except (IOError, ValueError, ImageCms.PyCMSError) as e:
        pass
        """
        dlgWarn(
         'No valid monitor profile found.\nSet SYSTEM_PROFILE_DIR and DEFAULT_MONITOR_PROFILE_NAME in config*.json',
         info='Invalid profile %s' % path,
         parent=Gui.window
                )
        """

    return profile


class icc:
    """
    Container for color management related options and methods.
    Should never be instantiated.
    The current state of color management system is defined by the
    following class variables (see configure() below)
        workingProfile, workingProfileInfo
        monitorProfile, monitorProfileInfo
        softProofingProfile
        colorSpace
        workToMonTransform

    """
    HAS_COLOR_MANAGE = False  # menu action "color manage" will be disabled
    COLOR_MANAGE = False  # no color management

    monitorProfile, workingProfile, softProofingProfile, workToMonTransform = (None,) * 4
    monitor_Profile_Path = DEFAULT_MONITOR_PROFILE_PATH
    workingProfileInfo, monitorProfileInfo = ('',) * 2

    # a (default) working profile is always needed
    defaultWorkingProfile = get_default_working_profile(mode='Cms')

    defaultMonitorProfile = get_default_monitor_profile(mode='Cms')

    # init current working profile
    workingProfile = defaultWorkingProfile
    workingProfileInfo = ImageCms.getProfileInfo(workingProfile)

    @classmethod
    def B_get_display_profilePath(cls, handle=None, device_id=None, enumicmprofiles=False):
        """
        Tries to detect current display profile path.

        :param handle: screen handle (Windows)
        :type handle: int
        :param device_id: name of display
        :type device_id: str
        :param enumicmprofiles: force usage of EnumICMProfiles
        :type enumicmprofiles: bool
        :return: monitor profile path
        :rtype: str
        """

        profile_path = cls.monitor_Profile_Path  # DEFAULT_MONITOR_PROFILE_PATH
        profile_path_list = []

        if HAS_WIN32:
            """
            We use GetICMProfileW. Unfortunately, during a session, GetICMProfileW does
            not see external (e.g. using ColorNavigator with a EIZO monitor) changes in
            display profile. As a workaround we call EnumICMProfilesW to get the list of 
            available profiles and open a dialog to manually set the profile. 
            """
            try:
                profile_path = getDisplayProfileWin(handle)

                if enumicmprofiles:
                    profile_path_list = enumDisplayProfilesWin(handle)

            except (OSError, RuntimeError, ValueError) as e:
                dlgWarn('Cannot detect monitor profile',
                        info=str(e)
                        )

        elif HAS_GI:
            # platform is 'linux'
            try:
                GIO_CANCELLABLE = Gio.Cancellable.new()
                client = Colord.Client.new()
                client.connect_sync(GIO_CANCELLABLE)
                device = client.find_device_sync('xrandr-' + device_id, GIO_CANCELLABLE)
                device.connect_sync(GIO_CANCELLABLE)
                default_profile = device.get_default_profile()
                default_profile.connect_sync(GIO_CANCELLABLE)
                profile_path = default_profile.get_filename()

            except (NameError, ImportError, GLib.GError, AttributeError) as e:
                dlgWarn('Cannot detect default monitor profile',
                        info=str(e)
                        )

            if enumicmprofiles:
                profile_list = []
                try:
                    profile_list = client.get_profiles_sync(GIO_CANCELLABLE)
                except (NameError, GLib.GError, AttributeError):
                    pass
                for p in profile_list:
                    try:
                        p.connect_sync(GIO_CANCELLABLE)
                        profile_path_list.append(p.get_filename())
                    except (NameError, GLib.GError, AttributeError):
                        continue

        elif HAS_COLORSYNC:
            # platform is Mac OS
            main_d_id = Quartz.CGMainDisplayID()
            device_info = ColorSync.ColorSyncDeviceCopyDeviceInfo(ColorSync.kColorSyncDisplayDeviceClass,
                                                                  ColorSync.CGDisplayCreateUUIDFromDisplayID(main_d_id)
                                                                  )
            url = device_info['FactoryProfiles']['1']['DeviceProfileURL']
            profile_path = Foundation.CFURLCopyFileSystemPath(url, Foundation.kCFURLPOSIXPathStyle)

            if enumicmprofiles:
                for k1 in ['FactoryProfiles', 'CustomProfiles']:
                    if k1 not in device_info:
                        continue
                    for k2 in device_info[k1]:
                        if not k2.isdigit():
                            continue
                        url = device_info[k1][k2]['DeviceProfileURL']
                        path = Foundation.CFURLCopyFileSystemPath(url, Foundation.kCFURLPOSIXPathStyle)
                        profile_path_list.append(path)

        if profile_path_list:
            dlg = bLUeDialogCombo(parent=Gui.window)
            dlg.cb.addItems(profile_path_list)
            dlg.setWindowTitle('Available Display Profiles')
            if dlg.exec():
                profile_path = dlg.cb.currentText()

        return profile_path

    @staticmethod
    def B_get_display_profile(handle=None, device_id=None, enumicmprofiles=False):
        """
        Returns the display CmsProfile instance

        :param handle: screen handle (Windows)
        :type handle: int
        :param device_id: name of display
        :type device_id: str
        :param enumicmprofiles: force usage of EnumICMProfiles
        :type enumicmprofiles: bool
        :return: monitor profile or None
        :rtype: Union[CmsProfile, None]
        """

        profile_path = icc.B_get_display_profilePath(handle=handle,
                                                     device_id=device_id,
                                                     enumicmprofiles=enumicmprofiles
                                                     )
        try:
            profile = getProfile(profile_path)
        except (IOError, ValueError, ImageCms.PyCMSError):
            profile = icc.defaultMonitorProfile

        return profile

    @classmethod
    def getMonitorProfile(cls, qscreen=None, enumicmprofiles=False):
        """
        Tries to retrieve the current color profile
        associated with the monitor specified by QScreen
        (the system main display if qscreen is None).
        THe parameter mode determines the type of the returned
        value (CmsProfile or QColorSpace).
        The method returns None if no valid profile can be found.

        :param qscreen: QScreen instance
        :type qscreen: QScreen
        :param enumicmprofiles: force usage of EnumICMProfiles
        :type enumicmprofiles: bool
        :return: monitor profile or None
        :rtype: Union[CmsProfile, None]
        """

        monitorProfile = None

        # detect profile
        if qscreen is not None:
            try:
                if sys.platform == 'win32':
                    dc = win32gui.CreateDC('DISPLAY', str(qscreen.name()), None)
                    monitorProfile = cls.B_get_display_profile(handle=dc,
                                                               enumicmprofiles=enumicmprofiles
                                                               )
                    win32gui.DeleteDC(dc)
                else:
                    monitorProfile = cls.B_get_display_profile(device_id=qscreen.name(),
                                                               enumicmprofiles=enumicmprofiles
                                                               )
            except (RuntimeError, OSError, TypeError, ImageCms.PyCMSError):
                pass
            if isinstance(monitorProfile, QColorSpace):
                if not monitorProfile.isValid():
                    monitorProfile = None
        else:
            raise ValueError

        return monitorProfile

    @classmethod
    def configure(cls, qscreen=None, colorSpace=-1, workingProfile=None, softproofingwp=None, enumicmprofiles=False):
        """
        Try to configure color management for the monitor
        specified by QScreen, and build a color transformation
        from the working profile (default sRGB) to the monitor profile.
        Parameter softproofingwp toggles soft proofing mode off (if it is None) and
        on (if it is a valid ImageCmsProfile).
        Default parameter values do not modify the current state.

        :param qscreen: QScreen instance
        :type qscreen: QScreen
        :param colorSpace:
        :type colorSpace
        :param workingProfile:
        :type workingProfile: ImageCmsProfile
        :param softproofingwp: profile for the device to simulate
        :type softproofingwp:
        :param enumicmprofiles: force usage of EnumICMProfiles
        :type enumicmprofiles: bool
        """

        cls.HAS_COLOR_MANAGE = False

        if not COLOR_MANAGE_OPT:
            return

        try:
            # get monitor profile
            if qscreen is not None:
                # get monitor profile as CmsProfile object.
                cls.monitorProfile = cls.getMonitorProfile(qscreen=qscreen, enumicmprofiles=enumicmprofiles)
                cls.monitorProfileInfo = ImageCms.getProfileInfo(cls.monitorProfile)
            else:
                pass  # default keep current state

            # get working profile as CmsProfile object (priority to color space tag)
            if colorSpace == 1:
                cls.workingProfile = cls.defaultWorkingProfile
            elif colorSpace == 2:
                cls.workingProfile = getProfile(ADOBE_RGB_PROFILE_PATH)
            elif isinstance(workingProfile, ImageCms.ImageCmsProfile):
                cls.workingProfile = workingProfile
            else:
                pass  # default keep current state

            cls.workingProfileInfo = ImageCms.getProfileInfo(cls.workingProfile)

            # QColorTransform is faster than CmsTransform. So
            # we try to get valid Qt profiles from Cms profiles.
            useqcs = False
            try:
                cls.workingProfile_QCS = QColorSpace.fromIccProfile(cls.workingProfile.tobytes())
                cls.monitorProfile_QCS = QColorSpace.fromIccProfile(cls.monitorProfile.tobytes())
                useqcs = cls.workingProfile_QCS.isValid() and cls.monitorProfile_QCS.isValid()
            except(ImageCms.PyCMSError, ValueError, TypeError):
                pass

            # build color transformation according to current state
            if type(softproofingwp) is ImageCms.ImageCmsProfile:
                cls.softProofingProfile = softproofingwp
                # QColorSpace does not implement soft proofing yet
                # cls.softProofingProfile = softproofingwp
                cls.workToMonTransform = ImageCms.buildProofTransformFromOpenProfiles(
                    cls.workingProfile,
                    cls.monitorProfile,
                    cls.softProofingProfile,
                    "RGB",
                    "RGB",
                    renderingIntent=ImageCms.Intent.PERCEPTUAL,
                    proofRenderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    flags=ImageCms.FLAGS['SOFTPROOFING'] |
                          ImageCms.FLAGS['BLACKPOINTCOMPENSATION']
                )
            else:
                if useqcs:
                    cls.workToMonTransform = cls.workingProfile_QCS.transformationToColorSpace(cls.monitorProfile_QCS)
                else:
                    cls.workToMonTransform = ImageCms.buildTransformFromOpenProfiles(
                        cls.workingProfile,
                        cls.monitorProfile,
                        "RGB",
                        "RGB",
                        renderingIntent=ImageCms.Intent.PERCEPTUAL
                    )

            cls.HAS_COLOR_MANAGE = (cls.monitorProfile is not None) and \
                                   (cls.workingProfile is not None) and \
                                   (cls.workToMonTransform is not None)
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
        PIL_img = Image.frombuffer('RGB',
                                   (image.width(), image.height()),
                                   bufC,
                                   'raw',
                                   'RGB',
                                   0,
                                   1
                                   )  # these weird parameters are recommended by a runtime warning !!!
        ImageCms.applyTransform(PIL_img, cmsTransformation, inPlace=True)
        # back to the image buffer
        buf[...] = np.frombuffer(PIL_img.tobytes(), dtype=np.uint8).reshape(buf.shape)
        return image

    @staticmethod
    def qcsConvertQImage(image, transformation_QCS=None, inPlace=False):
        """
        Applies a color transformation to a QImage and
        returns the transformed image.
        If transformation_QCS is None, the input image is returned.

        :param image: image to transform
        :type image: QImage
        :param transformation_QCS : Cms transformation
        :type transformation_QCS: ImageCmsTransform
        :return: The converted QImage
        :rtype: QImage
        """
        if not inPlace:
            image = image.copy()
        if transformation_QCS.isIdentity():
            return image
        image.applyColorTransform(transformation_QCS)
        return image

    @classmethod
    def convertQImage(cls, image, transformation=None, inPlace=False):
        if type(transformation) is ImageCms.ImageCmsTransform:
            image = cls.cmsConvertQImage(image, cmsTransformation=transformation, inPlace=inPlace)
        elif type(transformation) is QColorTransform:
            image = cls.qcsConvertQImage(image, transformation_QCS=transformation, inPlace=inPlace)
        return image
