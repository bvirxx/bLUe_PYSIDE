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
from os.path import expanduser
from json import load

#########################
# is Torch installed ?
#########################
import importlib

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None

########################
# read configuration file
########################
if sys.platform == 'win32':
    with open("config_win.json", "r") as fd:
        CONFIG = load(fd)
else:
    with open("config.json", "r") as fd:
        CONFIG = load(fd)

############
# General
############
COLOR_MANAGE_OPT = CONFIG["GENERAL"]["COLOR_MANAGE"]

############
# exiftool path
############
if getattr(sys, 'frozen', False):
    # running in a bundle
    EXIFTOOL_PATH = CONFIG["PATHS"]["EXIFTOOL_PATH_BUNDLED"]  # "bin\exiftool.exe"
else:
    EXIFTOOL_PATH = CONFIG["PATHS"]["EXIFTOOL_PATH"]  # C:\standalone\exiftool\exiftool.exe"

##############
# Paths to system profiles
##############
if sys.platform == 'win32':
    SYSTEM_PROFILE_DIR = CONFIG["PATHS"]["SYSTEM_PROFILE_DIR"]  # "C:\Windows\System32\spool\drivers\color"
    DNG_PROFILES_DIR1 = CONFIG["DNG_PROFILES"]["DIR1"]
    DNG_PROFILES_DIR2 = CONFIG["DNG_PROFILES"]["DIR2"]
else:
    SYSTEM_PROFILE_DIR = expanduser(CONFIG["PATHS"]["SYSTEM_PROFILE_DIR"])
    DNG_PROFILES_DIR1 = expanduser(CONFIG["DNG_PROFILES"]["DIR1"])
    DNG_PROFILES_DIR2 = expanduser(CONFIG["DNG_PROFILES"]["DIR2"])

ADOBE_RGB_PROFILE_PATH = SYSTEM_PROFILE_DIR + CONFIG["PROFILES"]["ADOBE_RGB_PROFILE_NAME"]  # "\AdobeRGB1998.icc"
SRGB_PROFILE_PATH = SYSTEM_PROFILE_DIR + CONFIG["PROFILES"]["SRGB_PROFILE_NAME"]  # "\sRGB Color Space Profile.icm"
DEFAULT_MONITOR_PROFILE_PATH = SYSTEM_PROFILE_DIR + CONFIG["PROFILES"]["DEFAULT_MONITOR_PROFILE_NAME"]

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

##############
# Brush folder
#############
BRUSHES_PATH = CONFIG["BRUSHES"]["DIR"]

########
# Theme
########
THEME = CONFIG["LOOK"]["THEME"]

##########
# Params
#########
MAX_ZOOM = CONFIG["PARAMS"]["MAX_ZOOM"]
TABBING = CONFIG["PARAMS"]["TABBING"]
