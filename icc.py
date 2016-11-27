from PIL import Image
from PIL.ImageCms import profileToProfile
from PIL.ImageQt import ImageQt
import numpy as np
from PyQt4.QtGui import QPixmap, QImage
from imgconvert import QImageToNdarray, PilImageToQImage, QImageToPilImage

#help(PIL.ImageCms)

ADOBE_RGB_PROFILE = "C:\Windows\System32\spool\drivers\color\AdobeRGB1998.icc"
SRGB_PROFILE = "C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm"
MONITOR_PROFILE = "C:\Windows\System32\spool\drivers\color\CS240(34381075)00000002.icc"

def convert(f, fromProfile=SRGB_PROFILE, toProfile=MONITOR_PROFILE):
    """
    Load file f and convert image to new profile.
    icc profile data will be removed from image.
    :param f: string path to f
    :return: ImageQt converted image (ImageQt is a subclass of QImage)
    """
    image_path = f
    original_image = Image.open(image_path)
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

def convertQImage(image, fromProfile=SRGB_PROFILE, toProfile=MONITOR_PROFILE):
    """
    Convert a QImage from profile fromProfile to profile toProfile.
    :param image: source QImage
    :param fromProfile: the source profile
    :param toProfile: the destination profile
    :return: The converted QImage
    """
    converted_image = profileToProfile(QImageToPilImage(image), fromProfile, toProfile)

    return PilImageToQImage(converted_image)