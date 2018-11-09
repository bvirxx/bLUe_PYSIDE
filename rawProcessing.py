"""
This File is part of bLUe software.

Copyright (C) 2017  Bernard Virot <bernard.virot@libertysurf.fr>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Lesser Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import cv2
import itertools

import numpy as np
import rawpy
from PySide2.QtGui import QImage

from bLUeCore.multi import interpMulti
from bLUeGui.bLUeImage import QImageBuffer, bImage
from bLUeGui.colorCIE import rgbLinear2rgbVec
from bLUeGui.graphicsSpline import channelValues
from bLUeGui.histogramWarping import warpHistogram
from debug import tdec
from dng import dngProfileLookTable, dngProfileToneCurve
from settings import USE_TETRA

def rawPostProcess(rImg, pool=None):
    """
    Develop raw image.
    Processing order is the following:
         1 - process raw image
         2 - contrast correction
         3 - saturation correction
    An Exception AttributeError is raised if rawImage
    is not an attribute of self.parentImage.
    """
    # postprocess output bits per channel
    output_bpc = 8
    max_ouput = 255 if output_bpc == 8 else 65535
    if rImg.parentImage.isHald:
        raise ValueError('Cannot build a 3D LUT from raw stack')

    # get adjustment form and rawImage
    adjustForm = rImg.getGraphicsForm()  # self.view.widget()
    options = adjustForm.options

    # show the Tone Curve form
    if options['cpToneCurve']:
        toneCurveShowFirst = adjustForm.setToneSpline()
    else:
        toneCurveShowFirst = False

    rawImage = getattr(rImg.parentImage, 'rawImage', None)
    if rawImage is None:
        raise ValueError("rawPostProcessing : not a raw image")
    currentImage = rImg.getCurrentImage()

    ##################
    # Control flags
    # postProcessCache is invalidated (reset to None) to by graphicsRaw.updateLayer (graphicsRaw.dataChanged event handler).
    # bufCache_HSV_CV32 is invalidated (reset to None) by camera profile related events.
    doALL = rImg.postProcessCache is None
    doCameraLookTable = options['cpLookTable'] and (doALL or rImg.bufCache_HSV_CV32 is None)
    #################

    ######################################################################################################################
    # process raw image (16 bits mode)                        RGB ------diag(multipliers)-----------> RGB
    # post processing pipeline (from libraw):                   |                                      |
    # - black substraction                                      | rawpyObj.rgb_xyz_matrix[:3,:]        | sRGB_lin2XYZ
    # - exposure correction                                     |                                      |
    # - white balance                                           |                                      |
    # - demosaic                                               XYZ                                    XYZ
    # - data scaling to use full range
    # - conversion to output color space
    # - gamma curve and brightness correction : gamma(imax) = 1, imax = 8*white/brightness
    # ouput is CV_16UC3
    ######################################################################################################################

    if doALL:
        ##############################
        # get postprocessing parameters
        ##############################
        # no_auto_scale = False  don't use : green shift
        #output_bpc = 16
        gamma = (2.222, 4.5)  # default REC BT 709 (exponent, slope)
        #gamma = (2.4, 12.92)  # sRGB (exponent, slope) cf. https://en.wikipedia.org/wiki/SRGB#The_sRGB_transfer_function_("gamma")
        exp_shift = adjustForm.expCorrection if not options['Auto Brightness'] else 0
        no_auto_bright = (not options['Auto Brightness'])
        use_auto_wb = options['Auto WB']
        use_camera_wb = options['Camera WB']
        exp_preserve_highlights = 0.99 if options['Preserve Highlights'] else 0.6  # range 0.0..1.0
        bright = adjustForm.brCorrection  # default 1, should be > 0
        hv = adjustForm.overexpValue
        highlightmode = rawpy.HighlightMode.Clip if hv == 0 \
            else rawpy.HighlightMode.Ignore if hv == 1 \
            else rawpy.HighlightMode.Blend if hv == 2 \
            else rawpy.HighlightMode.ReconstructDefault
        dv = adjustForm.denoiseValue
        fbdd_noise_reduction = rawpy.FBDDNoiseReductionMode.Off if dv == 0 \
            else rawpy.FBDDNoiseReductionMode.Light if dv == 1 \
            else rawpy.FBDDNoiseReductionMode.Full
        #############################################
        # build sample images for a set of multipliers
        if adjustForm.sampleMultipliers:
            bufpost16 = np.empty((rImg.height(), rImg.width(), 3), dtype=np.uint16)
            m = adjustForm.rawMultipliers
            co = np.array([0.85, 1.0, 1.2])
            mults = itertools.product(m[0] * co, [m[1]], m[2] * co)
            adjustForm.samples = []
            for i, mult in enumerate(mults):
                adjustForm.samples.append(mult)
                mult = (mult[0], mult[1], mult[2], mult[1])
                print(mult, '   ', m)
                bufpost_temp = rawImage.postprocess(
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=output_bpc,
                    exp_shift=exp_shift,
                    no_auto_bright=no_auto_bright,
                    use_auto_wb=use_auto_wb,
                    use_camera_wb=False,  # options['Camera WB'],
                    user_wb=mult,
                    gamma=gamma,
                    exp_preserve_highlights=exp_preserve_highlights,
                    bright=bright,
                    hightlightmode=highlightmode,
                    fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off
                )
                row = i // 3
                col = i % 3
                w, h = int(bufpost_temp.shape[1] / 3), int(bufpost_temp.shape[0] / 3)
                bufpost_temp = cv2.resize(bufpost_temp, (w, h))
                bufpost16[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = bufpost_temp
        # develop
        else:
            # highlight_mode : restoration of overexposed highlights. 0: clip, 1:unclip, 2:blend, 3...: rebuild
            # bufpost16 = rawImage.postprocess(use_camera_wb=True, output_bps=output_bps, gamma=(2.222,4.5))#, gamma=(1,1))
            bufpost16 = rawImage.postprocess(
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=output_bpc,
                exp_shift=exp_shift,
                no_auto_bright=no_auto_bright,
                use_auto_wb=use_auto_wb,
                use_camera_wb=use_camera_wb,
                user_wb=adjustForm.rawMultipliers,
                gamma=(1, 1),  # gamma,
                exp_preserve_highlights=exp_preserve_highlights,
                bright=bright,
                highlight_mode=highlightmode,
                fbdd_noise_reduction=fbdd_noise_reduction,
                median_filter_passes=1
            )
            # end of the post processing phase : save post processing cache
            rImg.postProcessCache = cv2.cvtColor(((bufpost16.astype(np.float32)) / max_ouput).astype(np.float32),
                                                    cv2.COLOR_RGB2HSV)  # TODO 29/10/18 change 65536 to 65535 validate
    else:
        pass

    if getattr(adjustForm, "toneForm", None) is not None:
        if doALL or toneCurveShowFirst:
            # update histogram
            s = rImg.postProcessCache.shape
            tmp = bImage(s[1], s[0], QImage.Format_RGB32)
            buf = QImageBuffer(tmp)
            buf[:, :, :] = (rImg.postProcessCache[:, :, 2, np.newaxis] * 255).astype(np.uint8)
            rImg.histImg = tmp.histogram(size=adjustForm.toneForm.scene().axeSize,
                                         bgColor=adjustForm.toneForm.scene().bgColor,
                                         range=(0, 255), chans=channelValues.Br,
                                         mode='Luminosity')
        adjustForm.toneForm.scene().quadricB.histImg = rImg.histImg
        adjustForm.toneForm.scene().update()

    # beginning of the camera profile phase : update buffers from the last post processed image
    bufHSV_CV32 = rImg.postProcessCache.copy()
    rImg.bufCache_HSV_CV32 = bufHSV_CV32.copy()


    ##########################
    # apply profile look table
    # it must be applied to the linear buffer and
    # before tone curve (cf. Adobe dng spec. p. 65)
    ##########################
    if doCameraLookTable:
        hsvLUT = dngProfileLookTable(adjustForm.dngDict)
        if hsvLUT.isValid:
            steps = tuple([360 / hsvLUT.dims[0], 1.0 / (hsvLUT.dims[1] - 1), 1.0 / (hsvLUT.dims[2] - 1)])
            coeffs = interpMulti(hsvLUT.data, steps, bufHSV_CV32, pool=pool, use_tetra=USE_TETRA, convert=False)
            bufHSV_CV32[:, :, 0] = np.mod(bufHSV_CV32[:, :, 0] + coeffs[:, :, 0], 360)
            bufHSV_CV32[:, :, 1:] = bufHSV_CV32[:, :, 1:] * coeffs[:, :, 1:]
            np.clip(bufHSV_CV32, (0, 0, 0), (360, 1, 1), out=bufHSV_CV32)
            rImg.bufCache_HSV_CV32 = bufHSV_CV32.copy()
    else:
        pass
    #############
    # tone curve
    ############
    buf = adjustForm.dngDict.get('ProfileToneCurve', [])
    # apply profile tone curve, if any
    if buf : # non empty list
        LUTXY = dngProfileToneCurve(buf).toLUTXY(maxrange=255)
        bufHSV_CV32[:, :, 2] = LUTXY[(bufHSV_CV32[:, :, 2] * 255).astype(np.uint16)] / 255  # self.postProcessCache]
    # apply user tone curbe
    toneForm = adjustForm.toneForm
    if toneForm is not None:
        if toneForm.isVisible():
            # apply user tone curve
            userLUTXY = toneForm.scene().quadricB.LUTXY
            # guiLUTXY = np.interp(np.arange(256), np.arange(256) * 256, guiLUTXY)
            bufHSV_CV32[:, :, 2] = userLUTXY[(bufHSV_CV32[:, :, 2] * 255).astype(np.uint16)] / 255  # TODO watch conversion ???

    rImg.bufCache_HSV_CV32 = bufHSV_CV32.copy() # CAUTION : must be outside of if adjusFormToneForm...

    # beginning of the contrast-saturation phase : update buffer from the last camera profile applcation
    bufHSV_CV32 = rImg.bufCache_HSV_CV32.copy()
    ###########
    # contrast and saturation correction (V channel).
    # We apply an automatic histogram equalization
    # algorithm, well suited for multimodal histograms.
    ###########

    if adjustForm.contCorrection > 0:
        # warp should be in range 0..1.
        # warp = 0 means that no additional warping is done, but
        # the histogram is always stretched.
        warp = max(0, (adjustForm.contCorrection - 1)) / 10
        bufHSV_CV32[:, :, 2], a, b, d, T = warpHistogram(bufHSV_CV32[:, :, 2], valleyAperture=0.05, warp=warp,
                                                         preserveHigh=options['Preserve Highlights'],
                                                         spline=None if rImg.autoSpline else rImg.getMmcSpline())  # preserveHigh=options['Preserve Highlights'])
        # show the spline
        if rImg.autoSpline and options['manualCurve']:
            rImg.getGraphicsForm().setContrastSpline(a, b, d, T)
            rImg.autoSpline = False  # mmcSpline = self.getGraphicsForm().scene().cubicItem # caution : misleading name for a quadratic s
    if adjustForm.satCorrection != 0:
        satCorr = adjustForm.satCorrection / 100  # range -0.5..0.5
        alpha = 1.0 / (0.501 + satCorr) - 1.0  # approx. map -0.5...0.0...0.5 --> +inf...1.0...0.0
        # tabulate x**alpha
        LUT = np.power(np.arange(256) / 255, alpha)
        # convert saturation s to s**alpha
        bufHSV_CV32[:, :, 1] = LUT[(bufHSV_CV32[:, :, 1] * 255).astype(int)]
    """proof of program assert : {postProcessCache contains the current post-processed image 
                                     and bufHSV_CV32 contains the current HSV image
                                     and bufCache_HSV_CV32 is a copy of bufHSV_CV32 after profile tone curve
                                     }"""
    # back to RGB
    bufpostF32_1 = cv2.cvtColor(bufHSV_CV32, cv2.COLOR_HSV2RGB) #* 65535 # .astype(np.uint16)  TODO 5/11/18 removed conversion removed * 65535validate
    #np.clip(bufpostF32_1, 0, 1, out=bufpostF32_1) # TODO 8/11/18 removed

    ###################
    # apply gamma curve
    ###################
    bufpostF32_255 = rgbLinear2rgbVec(bufpostF32_1)
    # np.clip(bufpostF32_255, 0, 255, out=bufpostF32_255)  # clip not needed after rgbLinear2rgbVec thresholds correction 8/11/18
    #############################
    # Conversion to 8 bits/channel
    #############################
    bufpostUI8 = bufpostF32_255.astype(np.uint8) # (bufpost16.astype(np.float32) / 256).astype(np.uint8) TODO 5/11/18 changed
    ###################################################
    #bufpostUI8 = (bufpost16/256).astype(np.uint8)
    #################################################
    if rImg.parentImage.useThumb:
        bufpostUI8 = cv2.resize(bufpostUI8, (currentImage.width(), currentImage.height()))

    bufOut = QImageBuffer(currentImage)
    bufOut[:, :, :3][:, :, ::-1] = bufpostUI8
    # base layer : no need to forward the alpha channel
    rImg.updatePixmap()
