This File is part of bLUe software.

Copyright (C) 2017-2019 Bernard Virot

## WEB SITE

See the [bLUe web site](http://bernard.virot.free.fr/) for images and tutorials.

## DESCRIPTION

 bLUe proposes a modular and comprehensive GUI for photo edition, featuring a large set of controls as adjustment layers, including
3D LUTs, 2.5D LUTs and curves in various color spaces.

A 3D LUT is a table representing a 3D cube of color nodes. Image pixels are associated
with nodes, based on their color. Changes are applied to each node individually,
giving full control over colors. When changes in color do not depend on the pixel brightnesses,
the table is called a 2.5D LUT. bLUe can create and edit 3D and 2.5D LUTs.

  * 2.5D LUTs are edited by selecting, grouping and moving nodes as control points over
a (hue, saturation) color wheel. Nodes are bound to an elastic grid and a grid smoothing algorithm is provided
to even the changes in color. Selection masks are automatically created from 2.5D LUTs, based on the node selection. 
Masks can be freely edited.

 * 3D LUTs are controlled by curves defining changes (additive or multiplicative shifts)
in a color channel, depending on the value of another channel. 

Multiple adjustment layers can be exported as a single 3D LUT in .cube format.

In addition to color images in formats jpg, tif, png, bLUe can develop raw files in all usual formats : nef, cr2, dng, ...
It supports dng/dcp dual illuminant camera profiles. 

The program is fully modular : controls are implemented as independent
adjustment layers using a common GUI. New functionality can be added very quickly and
any imaging library exposing Python bindings can take advantage of the GUI.

bLUe is aware of multi-screen environments and color profiles : it uses image and
monitor profiles in conjunction to display accurate colors.

The program is written in Python.

## FUNCTIONALITY

* Edition of images in formats jpg, png, tif, nef, cr2, dng,...
* Adjustment layers : exposure, brightness, saturation, contrast, channel mixer, color temperature, inversion, filters, noise reduction,
seamless cloning, segmentation, geometric transformations, merging, curves, 2.5D LUTs, 3D LUTs.
* RGB, HSV, CMYK, Lab color models
* Fast skin tone correction
* Automatic contrast enhancement (histogram warping and CLAHE)
* Fast Laplace Solver for seamless cloning
* Exposure fusion of (bracketed) images
* Crop tool
* Multiple blending modes; adjustable layer opacity
* Import and export of 3D LUTs in .cube format
* Adjustable masks
* Luminosity masks
* History
* Automatic import of camera specific dual illuminant dng/dcp profiles
* Adjustable profile tone curve
* Library viewer
* Slide show
* Context sensitive help

## REQUIREMENTS

* OpenCV-Python
* NumPy >= 1.15.0
* Qt for Python (PySide2)
* PIL
* RawPy
* PyWavelets

ExifTool must be installed.

Under Windows,  pywin32 is needed for multi-screen management.

Binary packages containing all dependencies are available for Windows.
Make sure to download the latest release.

## LICENSE

 This project is licensed under the LGPL V 3.
