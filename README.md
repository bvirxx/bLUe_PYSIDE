This File is part of bLUe software.

Copyright (C) 2017-2020 Bernard Virot

## WEB SITE

See the [bLUe site](http://bernard.virot.free.fr/) for images, tutorials and user manual.

Download an [example 3D LUT](http://bernard.virot.free.fr/sunrise.cube) and test it with your own images ( menu Layer > Load 3D LUT)

## DESCRIPTION

bLUe is a layer-based image editor. Its goal is to integrate a 3D LUT editor with more traditional tools to propose a powerful GUI for photo editing. The program is fully modular : tools are implemented as independent
adjustment layers using a common GUI. New features can be added easily:
any imaging library exposing Python bindings can take advantage of the GUI.

bLUe can develop raw images in all usual formats : nef, cr2, dng, ...
It supports dng/dcp dual illuminant camera profiles. 

bLUe provides drawing layers and paint brushes with adjustable parameters: size, flow, hardness, opacity.

bLUe is aware of multi-screen environments and color profiles : it uses image and
monitor profiles in conjunction to display accurate colors.

The program is written in Python.

### THE bLUe 3D LUT PERCEPTUAL EDITOR

A 3D LUT is a table representing a 3D cube of color nodes. Image pixels are associated
with nodes, according to their color. Changes are applied to each node individually,
giving full control over colors. 

3D LUTs are edited in a *perceptual editor* by selecting, grouping and moving color nodes over
a hue-saturation color wheel. Nodes are bound to an elastic grid and a grid smoothing algorithm is provided
to even the changes in color.

 ##### GROUPING NODES
A group of nodes represents a *significant* region of an image,
for example : tree leaves, skin, hairs, water, blue sky, ...), and all nodes in a group should be edited 
in a similar way. 

The editor's workflow is based on the grouping of nodes.

  *  Nodes are selected and grouped from the image;
  
  * To edit the colors of all pixels in a group while keeping their brightnesses, simply move the group over 
  the color wheel;
  
  * To control the brightnesses of pixels in a group, edit the brightness curve of the group.

##### CHANGING THE GAMUT
The grid of nodes can be warped to produce a particular look (orange-teal, moonlight, ...)

##### COMPLETING THE 3D LUT
Color transformations such as RGB curves, channel mixing, temperature,... can be easily integrated in the 3D LUT, in any order, 
by adding the corresponding adjustment layers to the bLUe layer stack and recording the stack as a single 3D LUT. 

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
