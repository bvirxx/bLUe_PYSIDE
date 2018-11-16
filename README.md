This File is part of bLUe software.

Copyright (C) 2017-2018 Bernard Virot <bernard.virot@libertysurf.fr>

## DESCRIPTION

 bLUe is a complete GUI environment for photo edition, featuring a large set of controls as adjustment layers, including
 3D LUTs and 2.5D LUTs.

A 3D LUT is a table representing a 3D cube of color nodes. Image pixels are associated
to nodes, based on their color, and changes are applied to each node individually,
giving full control over colors. When changes in color do not depend on the pixel brightnesses,
the table is called a 2.5D LUT.
bLUe creates interactive 2.5D LUTs which can be edited by selecting, grouping and moving nodes as control points over
a (hue, saturation) color wheel. Nodes are bound to an elastic grid and a grid smoothing algorithm is provided
to even the changes in color. Full 3D LUTs are created and edited by combining 2.5 LUTs with curve layers.

In addition to color images in formats jpg, tif, png, bLUe develops raw files in all usual formats : nef, cr2, dng, ...
It supports dng/dcp dual illuminant camera profiles.

The program is fully modular : controls are implemented as independent
adjustment layers using a common GUI. Any imaging library exposing Python
bindings can take advantage of the GUI.

bLUe is aware of multi-screen environments and color profiles : it uses image and
monitor profiles in conjunction to display accurate colors.

The program is written in Python.

## FUNCTIONALITY

* Edition of images in formats jpg, png, tif, nef, cr2, dng,...
* Adjustment layers : exposure, brightness, saturation, contrast, color temperature, inversion, filters, noise reduction, cloning,
segmentation, geometric transformations, curves (1D LUTs), 2.5D LUTs, 3D LUTs.
* Automatic contrast enhancement (histogram warping and CLAHE)
* Crop tool
* Import and export of 3D LUTs in .cube format
* Masks
* Automatic import of camera specific dng and dcp profiles
* Editable profile tone curve
* Library viewer
* Slide Show

## REQUIREMENTS

* OpenCV-Python
* NumPy
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

