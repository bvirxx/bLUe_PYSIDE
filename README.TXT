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

THIRD PARTY NOTICES CONDITIONS and DISCLAIMERS
 Qt5 is licensed under the LGPL version 3,
Pyside2 is licensed under the LGPL version 2.1
exiftool Copyright © 2013-2016, Phil Harvey.exiftool is licensed under thePerl Artistic License
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh

Pillow is the friendly PIL fork. It is Copyright © 2010-2018 by Alex Clark and contributors

libRaw Copyright (C) 2008-2018 LibRaw LLC (http://www.libraw.org, info@libraw.org)
rawpy is licensed under the MIT license Copyright (c) 2014 Maik Riechert

The QtHelp module uses the CLucene indexing library
Copyright (C) 2003-2006 Ben van Klinken and the CLucene Team

Changes are Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

opencv copyright
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

pywin32 License

Unless stated in the specfic source file, this work is
Copyright (c) 1996-2008, Greg Stein and Mark Hammond.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the distribution.

Neither names of Greg Stein, Mark Hammond nor the name of contributors may be used
to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

