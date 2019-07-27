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

from bLUeCore.multi import chosenInterp
from bLUeGui.bLUeImage import QImageBuffer, bImage
from bLUeGui.colorCIE import rgbLinear2rgb, sRGB_lin2XYZInverse, bradfordAdaptationMatrix
from bLUeGui.graphicsSpline import channelValues
from bLUeGui.histogramWarping import warpHistogram
from bLUeTop.dng import dngProfileLookTable, dngProfileToneCurve, interpolatedForwardMatrix


def rawPostProcess(rawLayer, pool=None):
    """
    raw layer development.
    Processing order is the following:
         1 - postprocessing
         2 - profile look table
         3 - profile and user tone curve
         2 - contrast correction
         3 - saturation correction
    A pool of workers is used to apply the
    profile look table.
    An Exception AttributeError is raised if rawImage
    is not an attribute of rawLayer.parentImage.
    @param rawLayer: development layer
    @type rawLayer: Qlayer
    @param pool: multi processing pool
    @type pool: multiprocessing.pool
    """
    # postprocess output bits per channel
    output_bpc = 8
    max_ouput = 255 if output_bpc == 8 else 65535
    if rawLayer.parentImage.isHald:
        raise ValueError('Cannot build a 3D LUT from raw stack')

    # get adjustment form and rawImage
    adjustForm = rawLayer.getGraphicsForm()  # self.view.widget()
    options = adjustForm.options

    # show the Tone Curve form
    if options['cpToneCurve']:
        toneCurveShowFirst = adjustForm.showToneSpline()
    else:
        toneCurveShowFirst = False

    # get RawPy instance
    rawImage = getattr(rawLayer.parentImage, 'rawImage', None)
    if rawImage is None:
        raise ValueError("rawPostProcessing : not a raw image")
    currentImage = rawLayer.getCurrentImage()

    ##################
    # Control flags
    # postProcessCache is invalidated (reset to None) to by graphicsRaw.updateLayer (graphicsRaw.dataChanged event handler).
    # bufCache_HSV_CV32 is invalidated (reset to None) by camera profile related events.
    doALL = rawLayer.postProcessCache is None
    if not doALL:
        parentImage = rawLayer.parentImage
        if (rawLayer.half and not parentImage.useThumb) or (not rawLayer.half and parentImage.useThumb):
            rawLayer.postProcessCache, rawLayer.bufCache_HSV_CV32 = (None,) * 2
            doALL = True
    doCameraLookTable = options['cpLookTable'] and (doALL or rawLayer.bufCache_HSV_CV32 is None)
    half_size = rawLayer.parentImage.useThumb
    #################

    ######################################################################################################################
    # process raw image (16 bits mode)                        camera ------diag(multipliers)----> camera
    # post processing pipeline (from libraw):                   ^
    # - black substraction                                      | CM=rawpyObj.rgb_xyz_matrix
    # - exposure correction                                     |
    # - white balance                                           |
    # - demosaic                                               XYZ
    # - data scaling to use full range
    # - conversion to output color space
    # - gamma curve and brightness correction : gamma(imax) = 1, imax = 8*white/brightness
    ######################################################################################################################

    use_auto_wb = options['Auto WB']
    use_camera_wb = options['Camera WB']
    exp_preserve_highlights = 0.99 if options['Preserve Highlights'] else 0.2  # 0.6  # range 0.0..1.0 (1.0 = full preservation)
    if doALL:
        ##############################
        # get postprocessing parameters
        ##############################
        # no_auto_scale = False  don't use : green shift
        gamma = (2.222, 4.5)  # default REC BT 709 (exponent, slope)
        # gamma = (2.4, 12.92)  # sRGB (exponent, slope) cf. https://en.wikipedia.org/wiki/SRGB#The_sRGB_transfer_function_("gamma")
        exp_shift = adjustForm.expCorrection if not options['Auto Brightness'] else 0
        no_auto_bright = (not options['Auto Brightness'])

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
            bufpost16 = np.empty((rawLayer.height(), rawLayer.width(), 3), dtype=np.uint16)
            m = adjustForm.rawMultipliers
            co = np.array([0.85, 1.0, 1.2])
            mults = itertools.product(m[0] * co, [m[1]], m[2] * co)
            adjustForm.samples = []
            for i, mult in enumerate(mults):
                adjustForm.samples.append(mult)
                mult = (mult[0], mult[1], mult[2], mult[1])
                print(mult, '   ', m)
                bufpost_temp = rawImage.postprocess(
                    half_size=half_size,
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
            bufpost16 = rawImage.postprocess(
                half_size=half_size,
                output_color=rawpy.ColorSpace.raw,  # XYZ
                output_bps=output_bpc,
                exp_shift=exp_shift,
                no_auto_bright=no_auto_bright,
                use_auto_wb=use_auto_wb,
                use_camera_wb=use_camera_wb,
                user_wb=adjustForm.rawMultipliers,
                gamma=(1, 1),
                exp_preserve_highlights=exp_preserve_highlights,
                bright=bright,
                highlight_mode=highlightmode,
                fbdd_noise_reduction=fbdd_noise_reduction,
                median_filter_passes=1
            )
            # save image into post processing cache
            rawLayer.postProcessCache = cv2.cvtColor(((bufpost16.astype(np.float32)) / max_ouput).astype(np.float32), cv2.COLOR_RGB2HSV)
            rawLayer.half = half_size
            rawLayer.bufpost16 = bufpost16
    else:
        pass

    # postProcessCache is in raw color space.
    # and must be converted to linear RGB. We follow
    # the guidelines of Adobe dng spec. (chapter 6).
    # If we have a valid dng profile and valid ForwardMatrix1
    # and ForwardMatrix2 matrices, we first convert to XYZ_D50 using the interpolated
    # ForwardMatrix for T and next from XYZ_D50 to RGB.
    # If we have no valid dng profile, we reinit the multipliers and
    # apply a Bradford chromatic adaptation matrix.
    m1, m2, m3 = adjustForm.asShotMultipliers[:3] if use_camera_wb else adjustForm.rawMultipliers[:3]
    D = np.diag((1/m1,1/m2,1/m3))
    tempCorrection = adjustForm.asShotTemp if use_camera_wb else adjustForm.tempCorrection
    MM = bradfordAdaptationMatrix(6500, tempCorrection)
    MM1 = bradfordAdaptationMatrix(6500, 5000)
    FM = None
    myHighlightPreservation = 0.8 if exp_preserve_highlights > 0.9 else 1.0
    if adjustForm.dngDict:
        try:
            FM = interpolatedForwardMatrix(adjustForm.tempCorrection, adjustForm.dngDict)
        except ValueError:
            pass
    raw2sRGBMatrix = sRGB_lin2XYZInverse @ MM1 @ FM * myHighlightPreservation if FM is not None else\
                     sRGB_lin2XYZInverse @ MM @ adjustForm.XYZ2CameraInverseMatrix @ D
    bufpost16 = np.tensordot(rawLayer.bufpost16, raw2sRGBMatrix, axes=(-1, -1))
    M = np.max(bufpost16) / 255.0
    bufpost16 = bufpost16 / M
    np.clip(bufpost16, 0, 255, out=bufpost16)
    rawLayer.postProcessCache = cv2.cvtColor(((bufpost16.astype(np.float32)) / max_ouput).astype(np.float32), cv2.COLOR_RGB2HSV)

    # update histogram
    s = rawLayer.postProcessCache.shape
    tmp = bImage(s[1], s[0], QImage.Format_RGB32)
    buf = QImageBuffer(tmp)
    buf[:, :, :] = (rawLayer.postProcessCache[:, :, 2, np.newaxis] * 255).astype(np.uint8)
    rawLayer.linearImg = tmp

    if getattr(adjustForm, "toneForm", None) is not None:
        rawLayer.histImg = tmp.histogram(size=adjustForm.toneForm.scene().axeSize,
                                         bgColor=adjustForm.toneForm.scene().bgColor,
                                         range=(0, 255), chans=channelValues.Br)  # mode='Luminosity')
        adjustForm.toneForm.scene().quadricB.histImg = rawLayer.histImg
        adjustForm.toneForm.scene().update()

    # beginning of the camera profile phase : update buffers from the last post processed image
    bufHSV_CV32 = rawLayer.postProcessCache.copy()
    rawLayer.bufCache_HSV_CV32 = bufHSV_CV32.copy()

    ##########################
    # Profile look table
    # it must be applied to the linear buffer and
    # before tone curve (cf. Adobe dng spec. p. 65)
    ##########################
    if doCameraLookTable:
        hsvLUT = dngProfileLookTable(adjustForm.dngDict)
        if hsvLUT.isValid:
            divs = hsvLUT.divs
            steps = tuple([360 / divs[0], 1.0 / (divs[1] - 1), 1.0 / (divs[2] - 1)]) # TODO -1 added 16/01/18 validate
            interp = chosenInterp(pool, currentImage.width() * currentImage.height())
            coeffs = interp(hsvLUT.data, steps, bufHSV_CV32, convert=False)
            bufHSV_CV32[:, :, 0] = np.mod(bufHSV_CV32[:, :, 0] + coeffs[:, :, 0], 360)
            bufHSV_CV32[:, :, 1:] = bufHSV_CV32[:, :, 1:] * coeffs[:, :, 1:]
            np.clip(bufHSV_CV32, (0, 0, 0), (360, 1, 1), out=bufHSV_CV32)
            rawLayer.bufCache_HSV_CV32 = bufHSV_CV32.copy()
    else:
        pass
    #############
    # tone curve
    ############
    buf = adjustForm.dngDict.get('ProfileToneCurve', [])
    # apply profile tone curve, if any
    if buf:  # non empty list
        LUTXY = dngProfileToneCurve(buf).toLUTXY(maxrange=255)
        bufHSV_CV32[:, :, 2] = LUTXY[(bufHSV_CV32[:, :, 2] * 255).astype(np.uint16)] / 255.0
    # apply user tone curve
    toneForm = adjustForm.toneForm
    if toneForm is not None:
        if toneForm.isVisible():
            userLUTXY = toneForm.scene().quadricB.LUTXY
            bufHSV_CV32[:, :, 2] = userLUTXY[(bufHSV_CV32[:, :, 2] * 255).astype(np.uint16)] / 255
    rawLayer.bufCache_HSV_CV32 = bufHSV_CV32.copy()  # CAUTION : must be outside of if toneForm.

    # beginning of the contrast-saturation phase : update buffer from the last camera profile applcation
    bufHSV_CV32 = rawLayer.bufCache_HSV_CV32.copy()
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
                                                         spline=None if rawLayer.autoSpline else rawLayer.getMmcSpline())
        # show the spline
        if rawLayer.autoSpline and options['manualCurve']:
            rawLayer.getGraphicsForm().setContrastSpline(a, b, d, T)
            rawLayer.autoSpline = False
    if adjustForm.satCorrection != 0:
        satCorr = adjustForm.satCorrection / 100  # range -0.5..0.5
        alpha = 1.0 / (0.501 + satCorr) - 1.0  # approx. map -0.5...0.0...0.5 --> +inf...1.0...0.0
        # tabulate x**alpha
        LUT = np.power(np.arange(256) / 255, alpha)
        # convert saturation s to s**alpha
        bufHSV_CV32[:, :, 1] = LUT[(bufHSV_CV32[:, :, 1] * 255).astype(int)]
    # back to RGB
    bufpostF32_1 = cv2.cvtColor(bufHSV_CV32, cv2.COLOR_HSV2RGB)  #* 65535 # .astype(np.uint16)
    # np.clip(bufpostF32_1, 0, 1, out=bufpostF32_1) # TODO 8/11/18 removed

    ###################
    # apply gamma curve
    ###################
    bufpostF32_255 = rgbLinear2rgb(bufpostF32_1)  # TODO modified 19/07 19 validate rgbLinear2rgbVec(bufpostF32_1)
    # np.clip(bufpostF32_255, 0, 255, out=bufpostF32_255)  # clip not needed after rgbLinear2rgbVec thresholds correction 8/11/18
    #############################
    # Conversion to 8 bits/channel
    #############################
    bufpostUI8 = bufpostF32_255.astype(np.uint8)  # (bufpost16.astype(np.float32) / 256).astype(np.uint8) TODO 5/11/18 changed
    ###################################################
    # bufpostUI8 = (bufpost16/256).astype(np.uint8)
    #################################################
    if rawLayer.parentImage.useThumb:
        bufpostUI8 = cv2.resize(bufpostUI8, (currentImage.width(), currentImage.height()))

    bufOut = QImageBuffer(currentImage)
    bufOut[:, :, :3][:, :, ::-1] = bufpostUI8
    # base layer : no need to forward the alpha channel
    rawLayer.updatePixmap()
