This File is part of bLUe software.

Copyright (C) 2017-2018 Bernard Virot <bernard.virot@libertysurf.fr>

DESCRIPTION
 bLUe is a complete GUI environment for photo edition. It is implemented in Python 3,
using Pyside 2, PIL and RawPy. It provides graphic tools to enhance color images in
usual formats : jpg, png, tif, including raw files : nef, cr2, dng.
bLUe proposes a wide set of adjustment layers : exposure, contrast,
color temperature, filters, noise reduction, cloning, segmentation, 1D LUTs and 3D LUTs.

 1D LUT are represented by curves. Curves are applied to each color channel individually.
As a consequence, the same correction is applied to all pixels regardless their color,
unless using selection tools to choose the region of the image that needs a correction.
In contrast, a 3D LUT is represented by a 3D cube of color nodes. Image pixels are associated
to nodes, based on their color and modifications can be applied to each node individually.

 The program provides 1D LUts (curves) and 3D LUT editors as adjustment layers. Moreover,
 bLUe can import existing 3D LUTs in .cube format.

 bLUe is aware of multi-screen environments and color profiles : it uses image and
 monitor profiles in conjunction to display accurate colors.

 The program is fully modular : tools are implemented as independent
adjustment layers using a common GUI. Any imaging python library, or C++ library exposing Python
bindings, can take advantage of the GUI.

LICENSE
 This project is licensed under the LGPL 3

 DISCLAIMER
  This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
