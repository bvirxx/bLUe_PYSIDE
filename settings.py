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