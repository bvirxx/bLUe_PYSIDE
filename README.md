This File is part of bLUe software.

Copyright (C) 2017-2018 Bernard Virot <bernard.virot@libertysurf.fr>

## DESCRIPTION

 bLUe is a complete GUI environment for photo edition. It can edit images in all
usual formats : jpg, png, tif, including raw format : nef, cr2, dng.
bLUe proposes a large set of adjustment layers:
exposure, brightness, saturation, contrast, color temperature, inversion, filters, noise reduction, cloning,
segmentation, 1D LUTs and 3D LUTs.

A 3D LUT is represented by a 3D cube of color nodes. Image pixels are associated
to nodes, based on their color and modifications are applied to each node individually.
bLUe proposes interactive 3D LUTs which can be edited by grouping and moving nodes over
a (hue, saturation) color wheel, making bLUe a powerful 3D LUT editor.

bLUe imports and exports 3D LUTs in .cube format.

Layer stacks can be saved as 3D LUTs.

 bLUe is aware of multi-screen environments and color profiles : it uses image and
 monitor profiles in conjunction to display accurate colors.

The program is fully modular : functionalities are implemented as independent
adjustment layers using a common GUI. Any imaging library exposing Python
bindings can take advantage of the GUI.

bLUe is written in Python.

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

