This File is part of bLUe software.

Copyright (C) 2017-2020 Bernard Virot

## WEB SITE

See the [bLUe web site](http://bernard.virot.free.fr/) for images, tutorials and user manual.

## DESCRIPTION

bLUe is a layer-based image editor. Its goal is to integrate a 3D LUT editor with more traditional tools to propose a modular and 
powerful GUI for photo editing. The program is fully modular : tools are implemented as independent
adjustment layers using a common GUI. New functionality can be added very quickly and
any imaging library exposing Python bindings can take advantage of the GUI.


A 3D LUT is a table representing a 3D cube of color nodes. Image pixels are associated
with nodes, based on their color. Changes are applied to each node individually,
giving full control over colors. When changes in color do not depend on the pixel brightnesses,
the table is called a 2.5D LUT. bLUe can create and edit 3D LUTs and 2.5D LUTs.

  * 2.5D LUTs are edited in a *perceptual editor* by selecting, grouping and moving color nodes on
a hue-saturation color wheel. Nodes are bound to an elastic grid and a grid smoothing algorithm is provided
to even the changes in color. Grid transformations are supported to interactively modify the overall gamut.

 * 3D LUTs are controlled by curves defining changes (additive or multiplicative shifts)
in a color channel, depending on the value of another channel. 

 Selection masks are automatically created from 2.5D LUTs, based on the node selection. 
Masks can be freely edited.

A stack of adjustment layers can be exported as a single 3D LUT in .cube format.

bLUe can develop raw images in all usual formats : nef, cr2, dng, ...
It supports dng/dcp dual illuminant camera profiles. 

bLUe provides drawing layers and versatile paint brushes.

bLUe is aware of multi-screen environments and color profiles : it uses image and
monitor profiles in conjunction to display accurate colors.

The program is written in Python.

## FUNCTIONALITY

* Simultaneous edition of multiple images in formats jpg, png, tif, nef, cr2, dng,...
* Color profile management
* Adjustment layers : exposure, brightness, saturation, contrast, channel mixer, color temperature, inversion, filters, noise reduction,
seamless cloning, segmentation, exposure fusion, curves, 2.5D LUTs, 3D LUTs.
* Drawing layers
* Automatic contrast enhancement (histogram warping and CLAHE)
* Seamless cloning
* Exposure fusion
* Multiple blending modes; adjustable layer opacity
* Import and export of 3D LUTs in .cube format
* Editable masks
* Automatic import of camera specific profiles for development of raw images
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

ExifTool should be installed.

Under Windows,  pywin32 is needed for multi-screen management.

Binary packages containing all dependencies are available for Windows.
Make sure to download the latest release.

## LICENSE

 This project is licensed under the LGPL V 3.
