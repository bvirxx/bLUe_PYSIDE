This File is part of bLUe software.

Copyright (C) 2017-2018 Bernard Virot <bernard.virot@libertysurf.fr>

## DESCRIPTION

 bLUe is a complete GUI environment for photo edition, featuring a large set of controls as adjustment layers, including
 3D LUTs and 2.5D LUTs.

A 3D LUT is represented by a 3D cube of color nodes. Image pixels are associated
to nodes, based on their color, and modifications (change in output colors) are applied to each node individually,
giving full control over colors. However, for photo editing, aesthetic considerations show that
modifications should not depend on the pixel brightnesses. A 3D LUT respecting this constraint
is called 2.5D LUT.
bLUe creates interactive 2.5D LUTs which can be edited by selecting, grouping and moving nodes over
a (hue, saturation) color wheel.

Regarding raw file development, bLue provides full support for dng/dcp camera profiles : the same profile
can be applied to several images, to obtain uniform results with a significant saving in time.

The program is fully modular : controls are implemented as independent
adjustment layers using a common GUI. Any imaging library exposing Python
bindings can take advantage of the GUI.

bLUe is aware of multi-screen environments and color profiles : it uses image and
monitor profiles in conjunction to display accurate colors.

bLUe is written in Python.

## FUNCTIONALITY

* Edition of files in formats jpg, png, tif, nef, cr2, dng.
* Adjustment layers : exposure, brightness, saturation, contrast, color temperature, inversion, filters, noise reduction, cloning,
segmentation, geometric transformations, curves (1D LUTs), 2.5D LUTs, 3D LUTs.
* Automatic contrast enhancement (histogram warping and CLAHE)
* Import and export of 3D LUTs in .cube format
* Mask edition
* Automatic import of camera specific dng and dcp profiles
* editable tone curve
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

