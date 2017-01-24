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

ADOBE_RGB_PROFILE_PATH = "C:\Windows\System32\spool\drivers\color\AdobeRGB1998.icc"
SRGB_PROFILE_PATH = "C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm"
MONITOR_PROFILE_PATH = "C:\Windows\System32\spool\drivers\color\CS240(34381075)00000002.icc"  #photography CS240 calibration

dp=get_display_profile()
WORKING_PROFILE_INFO=getProfileInfo(dp)
ip = getOpenProfile(SRGB_PROFILE_PATH)
t = buildTransformFromOpenProfiles(ip, dp, "RGB", "RGB")

"""
def convert(f, fromProfile=SRGB_PROFILE, toProfile=MONITOR_PROFILE):

    Load file f and convert image to new profile.
    icc profile data will be removed from image.
    :param f: string path to f
    :return: ImageQt converted image (ImageQt is a subclass of QImage)

    image_path = f
    original_image = Image.open(image_path)
    #icc = Image.open(path).info.get('icc_profile')
    #print original_image.info['exif']
    if True:#'icc_profile' in original_image.info:

        # This part is important. If the photo already has an
        # ICC profile, you should skip this step. If you don't
        # it can lead to color distortions in the image.
        # Therefore we always check whether the image
        # has an ICC profile first.
        converted_image = profileToProfile(original_image, fromProfile, toProfile)
        qim = ImageQt(converted_image).copy()#######################"

        #a=qim.bits()
        return qim
"""

def convertQImage(image, fromProfile=SRGB_PROFILE_PATH, toProfile=SRGB_PROFILE_PATH):
    """
    Convert a QImage from profile fromProfile to profile toProfile.
    :param image: source QImage
    :param fromProfile: the source profile, default sRGB
    :param toProfile: the destination profile, default sRGB
    :return: The converted QImage
    """

    if fromProfile == toProfile:
        return image

    p = fromProfile
    if len(image.meta.profile) >0 :
        # Imbedded profile
        try:
            p=getOpenProfile(StringIO(image.profile))
        except:
            pass
    # convert
    #converted_image = profileToProfile(QImageToPilImage(image), p, toProfile)
    converted_image = applyTransform(QImageToPilImage(image), t, 0)
    return PilImageToQImage(converted_image)