This File is part of bLUe software.

Copyright (C) 2017-2018 Bernard Virot <bernard.virot@libertysurf.fr>

DESCRIPTION

 bLUe is a complete GUI environment for photo edition. It can edit images in all
usual formats : jpg, png, tif, including raw files: nef, cr2, dng.
bLUe proposes a wide set of adjustment layers:
exposure, brightness, saturation, contrast, color temperature, inversion, filters, noise reduction, cloning,
segmentation, 1D LUTs and 3D LUTs editors.

Layer stacks can be saved as 3D LUTs, making bLUe a powerful 3D LUT builder.
It imports and exports 3D LUTs in .cube format.

 1D LUT are represented by curves. Curves are applied to each color channel individually.
As a consequence, the same correction is applied to all pixels regardless their color,
unless using selection tools to choose the region of the image that needs a correction.
In contrast, a 3D LUT is represented by a 3D cube of color nodes. Image pixels are associated
to nodes, based on their color and modifications are applied to each node individually.
With bLUe, 3D LUTs are interactively created by grouping and moving nodes over a (hue, sat.) color wheel.

 bLUe is aware of multi-screen environments and color profiles : it uses image and
 monitor profiles in conjunction to display accurate colors.

The program is fully modular : tools are implemented as independent
adjustment layers using a common GUI. Any imaging library exposing Python
bindings can take advantage of the GUI.

bLUe is written in Python.

REQUIREMENTS

   numpy

   PySide2

   PIL

   RawPy
   
   PyWavelets.

Binary packages containing all dependencies are available for Windows.
Make sure to download the latest release.

LICENSE

 This project is licensed under the LGPL V 3.

