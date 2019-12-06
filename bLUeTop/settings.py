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
from json import load

########################
# read configuration file
########################
with open("config.json", "r") as fd:
    CONFIG = load(fd)

############
# exiftool path
############
if getattr( sys, 'frozen', False):
    # running in a bundle
    EXIFTOOL_PATH = CONFIG["PATHS"]["EXIFTOOL_PATH_BUNDLED"]  # "bin\exiftool.exe"
else:
    EXIFTOOL_PATH = CONFIG["PATHS"]["EXIFTOOL_PATH"]  # C:\standalone\exiftool\exiftool.exe"

##############
# Paths to system profiles
##############
SYSTEM_PROFILE_DIR = CONFIG["PATHS"]["SYSTEM_PROFILE_DIR"]  # "C:\Windows\System32\spool\drivers\color"
ADOBE_RGB_PROFILE_PATH = SYSTEM_PROFILE_DIR + CONFIG["PROFILES"]["ADOBE_RGB_PROFILE"]  # "\AdobeRGB1998.icc"
SRGB_PROFILE_PATH = SYSTEM_PROFILE_DIR + CONFIG["PROFILES"]["SRGB_PROFILE"]  # "\sRGB Color Space Profile.icm"
DEFAULT_MONITOR_PROFILE_PATH = SYSTEM_PROFILE_DIR + CONFIG["PROFILES"]["DEFAULT_MONITOR_PROFILE_NAME"]

DNG_PROFILES_DIR1 = CONFIG["DNG_PROFILES"]["DIR1"]
DNG_PROFILES_DIR2 = CONFIG["DNG_PROFILES"]["DIR2"]

#############
# 3D LUT
############
# use tetrahedral interpolation instead of trilinear; trilinear is faster
USE_TETRA = CONFIG["ENV"]["USE_TETRA"]  # False

######################
# parallel interpolation
#######################
USE_POOL = CONFIG["ENV"]["USE_POOL"]  # True
POOL_SIZE = CONFIG["ENV"]["POOL_SIZE"]  # 4

########
# Theme
########
THEME = CONFIG["LOOK"]["THEME"]

##########
# Params
#########
MAX_ZOOM = CONFIG["PARAMS"]["MAX_ZOOM"]
TABBING = CONFIG["PARAMS"]["TABBING"]
