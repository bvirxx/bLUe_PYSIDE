"""
Copyright (C) 2017  Bernard Virot

bLUe - Photo editing software.

With Blue you can enhance and correct the colors of your photos in a few clicks.
No need for complex tools such as lasso, magic wand or masks.
bLUe interactively constructs 3D LUTs (Look Up Tables), adjusting the exact set
of colors you want.

3D LUTs are widely used by professional film makers, but the lack of
interactive tools maked them poorly useful for photo enhancement, as the shooting conditions
can vary widely from an image to another. With bLUe, in a few clicks, you select the set of
colors to modify, the corresponding 3D LUT is automatically built and applied to the image.
You can then fine tune it as you want.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
from os.path import isfile

###########################
# Path to the exiftool executable
import sys
if getattr( sys, 'frozen', False ) :
    # running in a bundle
    EXIFTOOL_PATH = 'exiftool(-k)'
else :
    # running live
    EXIFTOOL_PATH = "C:\standalone\exiftool\exiftool(-k)"
###########################

###########################
# Paths to installed profiles
SYSTEM_PROFILE_PATH = "C:\Windows\System32\spool\drivers\color"
ADOBE_RGB_PROFILE_PATH = SYSTEM_PROFILE_PATH + "\AdobeRGB1998.icc"
SRGB_PROFILE_PATH = SYSTEM_PROFILE_PATH + "\sRGB Color Space Profile.icm"
#############################