This File is part of bLUe software.

Copyright (C) 2017-2023 Bernard Virot

## DESCRIPTION

bLUe is a layer-based image editor. It aims to integrate a new *perceptual 3D LUT editor* with more traditional tools in
order to provide a powerful GUI for photo editing. The program is fully modular : tools are implemented as independent
adjustment layers using a common GUI. New features can be added easily:
any imaging library exposing Python bindings can take advantage of the GUI.

bLUe can develop raw images in all usual formats : nef, cr2, dng, ... It supports dng/dcp dual illuminant camera
profiles : they are essential for rendering colors similar to that produced by camera software.

bLUe provides drawing layers and paint brushes with adjustable parameters: size, flow, hardness, opacity.

bLUe is aware of multi-screen environments and color profiles : it uses image and monitor profiles in conjunction to
display accurate colors.

bLUe uses a pretrained neural network to provide automatic 3D LUTs for fast enhancement of images. The pretrained model
and the code are taken from the [recent work](https://github.com/HuiZeng/Image-Adaptive-3DLUT) of Hui Zeng, Jianrui Cai,
Lida Li, Zisheng Cao, and Lei Zhang.

Images being edited and their layer stack can be saved to .blu file format.

The program is written in Python.

### THE PERCEPTUAL 3D LUT EDITOR

A 3D LUT is a table representing a 3D cube of color nodes. Image pixels are associated with nodes, according to their
color. Changes are applied to each node individually, giving full control over colors.

3D LUTs are edited in a *perceptual editor* by selecting, grouping and moving color nodes over a hue-saturation color
wheel. Nodes are bound to an elastic grid and a grid smoothing algorithm is provided to even changes in color.

##### GROUPING NODES

A group of nodes represents a *meaning* region of an image, for example : tree leaves, skin, hairs, water, blue sky,
...), and all nodes in a group should be edited in a similar way.

The editor's workflow is based on the grouping of nodes.

* Nodes are selected and grouped from the image;

* To edit the colors of all pixels in a group while keeping their brightnesses, simply move the group on the color
  wheel;

* To control the brightnesses of pixels in a group, edit the brightness curve of the group.

##### CHANGING THE GAMUT

The whole grid can be warped to produce a particular look (orange-teal, moonlight, ...)

##### COMPLETING THE 3D LUT

Color transformations such as RGB curves, channel mixing, temperature,... can be easily integrated in the 3D LUT, in any
order, by adding the corresponding adjustment layers to the stack and recording the whole stack as a single 3D LUT.
An [example 3D LUT](http://bernard.virot.free.fr/sunrise.cube) can be found on the web site.

## WEB SITE

See the [bLUe site](http://bernard.virot.free.fr/) for screenshots, tutorials and user manual.

## FUNCTIONALITY

* Neural network based automatic 3D LUT for image enhancement
* Soft proofing
* Simultaneous edition of multiple images in formats jpg, png, tif, nef, cr2, dng,...
* Color profile management
* Adjustment layers : exposure, brightness, saturation, contrast, channel mixer, color temperature, inversion, filters,
* noise reduction, seamless cloning, segmentation, exposure fusion, curves, 2.5D LUTs, 3D LUTs.
* Drawing and painting layers
* Extensible set of brushes and patterns; import of abr files
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

* Python >= 3.9
* Qt6 for Python (PySide6)
* opencv-python
* numpy
* Pillow
* RawPy
* PyWavelets
* PyTorch >= 1.4 and torchvision for auto adaptive 3D LUT
* tifffile

ExifTool should be installed.

On Windows, pywin32 is needed for multi-screen management.

### The Qt5 (PySide2) version (branch master) is deprecated.

## LICENSE

 This project is licensed under the LGPL V 3.
