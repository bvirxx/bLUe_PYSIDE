This File is part of bLUe software.

Copyright (C) 2017-2018 Bernard Virot <bernard.virot@libertysurf.fr>

## DESCRIPTION

 bLUe proposes a modular and comprehensive GUI for photo edition, featuring a large set of controls as adjustment layers, including
 3D LUTs, 2.5D LUTs, 2D curves and 1D curves for various color models.

A 3D LUT is a table representing a 3D cube of color nodes. Image pixels are associated
to nodes, based on their color, and changes are applied to each node individually,
giving full control over colors. When changes in color do not depend on the pixel brightnesses,
the table is called a 2.5D LUT.
bLUe creates interactive 2.5D LUTs which can be edited by selecting, grouping and moving nodes as control points over
a (hue, saturation) color wheel. Nodes are bound to an elastic grid and a grid smoothing algorithm is provided
to even the changes in color. Full 3D LUTs are created and edited by combining 2.5 LUTs with curve layers.

Alternatively 3D LUTs can be created from curves defining changes (additive or multiplicative shifts)
for a color channel, depending on the value of another channel. bLUe can create and edit interactive 3D LUTs for
HSV shifts.

curves to selectively modify the brightness of pixels, depending of their hue.

Masks are automatically created from 2.5D LUTs, based on the node selection. They can be freely edited.

In addition to color images in formats jpg, tif, png, bLUe develops raw files in all usual formats : nef, cr2, dng, ...
It supports dng/dcp dual illuminant camera profiles. Camera model dcp profiles can be applied to any format
of raw file, eliminating the burden of a pre-conversion to the dng format.

The program is fully modular : controls are implemented as independent
adjustment layers using a common GUI. New functionality can be added very quickly and
any imaging library exposing Python bindings can take advantage of the GUI.

bLUe is aware of multi-screen environments and color profiles : it uses image and
monitor profiles in conjunction to display accurate colors.

The program is written in Python.

## FUNCTIONALITY

* Edition of images in formats jpg, png, tif, nef, cr2, dng,...
* Adjustment layers : exposure, brightness, saturation, contrast, channel mixer, color temperature, inversion, filters, noise reduction, cloning,
segmentation, geometric transformations, curves, 2.5D LUTs, 3D LUTs.
* RGB, HSV, CMYK, Lab color models
* Fast skin tone correction
* Automatic contrast enhancement (histogram warping and CLAHE)
* Fast Laplace Solver (cloning)
* Crop tool
* Multiple blending modes; adjustable layer opacity
* Import and export of 3D LUTs in .cube format
* Adjustable masks
* History
* Automatic import of camera specific dual illuminant dng/dcp profiles
* Adjustable profile tone curve
* Library viewer
* Slide show
* Context help

## REQUIREMENTS

* OpenCV-Python
* NumPy >= 1.15.0
* PySide2
* PIL
* RawPy
* PyWavelets

ExifTool must be installed.

Under Windows,  pywin32 is needed for multi-screen management.

Binary packages containing all dependencies are available for Windows.
Make sure to download the latest release.

## LICENSE

 This project is licensed under the LGPL V 3.

