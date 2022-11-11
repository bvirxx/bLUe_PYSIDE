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

import numpy as np
import rawpy
from PySide2.QtGui import QImage
from rawpy._rawpy import LibRawFatalError

from bLUeCore.multi import chosenInterp
from bLUeGui.bLUeImage import QImageBuffer, bImage
from bLUeGui.colorCIE import rgbLinear2rgb, sRGB_lin2XYZInverse, bradfordAdaptationMatrix
from bLUeGui.dialog import dlgWarn
from bLUeGui.const import channelValues
from bLUeGui.histogramWarping import warpHistogram
from bLUeTop.dng import dngProfileLookTable, dngProfileToneCurve, interpolatedForwardMatrix


def rawRead(file):
    """
    Loads a raw image file or a buffer into a RawPy instance.
    The image file is closed after reading.
    Note. rawpy.imread keeps file open. Calling raw.close() deletes the RawPy instance.
    As a workaround we use low-level file buffer and unpack().

    :param file:
    :type file: str or file-like object
    :return:
    :rtype: RawPy instance
    """
    rawpyInst = rawpy.RawPy()
    try:
        if type(file) is str:
            with open(file, "rb") as bufio:
                rawpyInst.open_buffer(bufio)
        else:  # should be BytesIO
            rawpyInst.open_buffer(file)
        rawpyInst.unpack()
    except IOError as e:
        dlgWarn('rawRead : IO error', str(e))
        raise
    except LibRawFatalError as e:
        dlgWarn('rawRead : LibRaw Fatal Error', str(e))
        raise
    return rawpyInst


def rawPostProcess(rawLayer, pool=None):
    """
    raw layer development.
    Processing order is the following:
         1 - postprocessing
         2 - profile look up table
         3 - profile and user tone curve
         2 - contrast correction
         3 - saturation correction
    A pool of workers is used to apply the
    profile look up table.

    :raises AttributeError if rawImage
    is not an attribute of rawLayer.parentImage.

    :param rawLayer: development layer
    :type rawLayer: Qlayer
    :param pool: multiprocessing pool
    :type pool: multiprocessing.pool
    """
    # postprocess output bits per channel
    output_bpc = 8
    max_output = 255 if output_bpc == 8 else 65535
    if rawLayer.parentImage.isHald:
        raise ValueError('Cannot build a 3D LUT from raw stack')

    # get adjustment form and rawImage
    adjustForm = rawLayer.getGraphicsForm()  # self.view.widget()
    options = adjustForm.options

    # get RawPy instance
    # Relevant RawPy instance attributes are black_level_per_channel, camera_whitebalance, color_desc, color_matrix,
    # daylight_whitebalance, num_colors, raw_colors_visible, raw_image, raw_image_visible, raw_pattern,
    # raw_type, rgb_xyz_matrix, sizes, tone_curve.
    # raw_image and raw_image_visible are sensor data
    rawImage = getattr(rawLayer.parentImage, 'rawImage', None)
    if rawImage is None:
        raise ValueError("rawPostProcessing : not a raw image")
    currentImage = rawLayer.getCurrentImage()

    ##################
    # Control flags
    # postProcessCache is invalidated (reset to None) by
    # graphicsRaw.updateLayer (graphicsRaw.dataChanged event handler).
    # bufCache_HSV_CV32 is invalidated (reset to None) by camera profile related events.
    doALL = rawLayer.postProcessCache is None
    if not doALL:
        parentImage = rawLayer.parentImage
        if (rawLayer.half and not parentImage.useThumb) or (not rawLayer.half and parentImage.useThumb):
            rawLayer.postProcessCache, rawLayer.bufCache_HSV_CV32 = (None,) * 2
            doALL = True
    doCameraLookTable = options['cpLookTable']  # and (doALL or rawLayer.bufCache_HSV_CV32 is None)
    half_size = rawLayer.parentImage.useThumb
    #################

    #############################################################################################################
    # process raw image (16 bits mode)                        camera ------diag(multipliers)----> camera
    # post processing pipeline (from libraw):                   ^
    # - black substraction                                      | CM=rawpyObj.rgb_xyz_matrix
    # - exposure correction                                     |
    # - white balance                                           |
    # - demosaic                                               XYZ
    # - data scaling to use full range
    # - conversion to output color space
    # - gamma curve and brightness correction : gamma(imax) = 1, imax = 8*white/brightness
    ###########################################################################################################

    use_auto_wb = options['Auto WB']
    use_camera_wb = options['Camera WB']
    exp_preserve_highlights = 0.99 if options[
        'Preserve Highlights'] else 0.2  # 0.6  # range 0.0..1.0 (1.0 = full preservation)
    if doALL:
        ##############################
        # set postprocessing parameters
        ##############################
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

        ##########
        # develop
        ##########
        bufpost16 = rawImage.postprocess(
            half_size=half_size,
            output_color=rawpy.ColorSpace.raw,  # camera color space
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
        # save image to post processing cache
        rawLayer.postProcessCache = cv2.cvtColor(((bufpost16.astype(np.float32)) / max_output).astype(np.float32),
                                                 cv2.COLOR_RGB2HSV)
        rawLayer.half = half_size
        rawLayer.bufpost16 = bufpost16
    else:
        pass

    # postProcessCache is in raw color space
    # and must be converted to linear RGB. We follow
    # the guidelines of Adobe dng spec. (chapter 6).
    # If we have a valid dng profile and valid ForwardMatrix1
    # and ForwardMatrix2 matrices, we first convert to XYZ_D50 using the interpolated
    # ForwardMatrix for T and next from XYZ_D50 to RGB.
    # If we have no valid dng profile, we reinit the multipliers and
    # apply a Bradford chromatic adaptation matrix.

    # calculate the multipliers reinitialization matrix
    if use_camera_wb:
        m1, m2, m3 = adjustForm.asShotMultipliers[:3]
        m1, m2, m3 = m1 / m2, 1.0, m3 / m2
    else:
        m1, m2, m3 = adjustForm.rawMultipliers[:3]
    D = np.diag((1 / m1, 1 / m2, 1 / m3))

    # calculate the raw to sRGB transformation matrix
    tempCorrection = adjustForm.asShotTemp if use_camera_wb else adjustForm.tempCorrection
    MM = bradfordAdaptationMatrix(6500, tempCorrection)
    MM1 = bradfordAdaptationMatrix(6500, 5000)
    FM = None
    if adjustForm.dngDict:
        try:
            FM = interpolatedForwardMatrix(adjustForm.tempCorrection, adjustForm.dngDict)
        except ValueError:
            pass
    if FM is None:
        raw2sRGBMatrix = sRGB_lin2XYZInverse @ MM @ adjustForm.XYZ2CameraInverseMatrix @ D
    else:
        myHighlightPreservation = 0.8 if exp_preserve_highlights > 0.9 else 1.0
        raw2sRGBMatrix = sRGB_lin2XYZInverse @ MM1 @ FM * myHighlightPreservation

    # apply the raw to sRGB matrix to image buffer
    bufpost16 = np.tensordot(rawLayer.bufpost16, raw2sRGBMatrix, axes=(-1, -1))
    np.clip(bufpost16, 0, 255, out=bufpost16)
    rawLayer.postProcessCache = cv2.cvtColor(bufpost16.astype(np.float32) / max_output,
                                             cv2.COLOR_RGB2HSV
                                             )

    # update the background histogram of tone curve
    s = rawLayer.postProcessCache.shape
    tmp = bImage(s[1], s[0], QImage.Format_RGB32)
    buf = QImageBuffer(tmp)
    buf[:, :, :] = (rawLayer.postProcessCache[:, :, 2, np.newaxis] * 255).astype(np.uint8)
    rawLayer.linearImg = tmp  # attribute used by graphicsToneForm.colorPickedSlot()
    if getattr(adjustForm, "toneForm", None) is not None:
        histImg = tmp.histogram(size=adjustForm.toneForm.scene().axeSize,
                                bgColor=adjustForm.toneForm.scene().bgColor,
                                range=(0, 255),
                                chans=channelValues.Br
                                )
        adjustForm.toneForm.scene().quadricB.histImg = histImg
        adjustForm.toneForm.scene().update()

    # beginning of the camera profile phase : update buffers from the last post processed image
    bufHSV_CV32 = rawLayer.postProcessCache.copy()
    rawLayer.bufCache_HSV_CV32 = bufHSV_CV32.copy()

    ##########################
    # apply profile look table
    # it must be applied to the linear buffer and
    # before tone curve (cf. Adobe dng spec. p. 65)
    ##########################
    if doCameraLookTable:
        hsvLUT = dngProfileLookTable(adjustForm.dngDict)
        if hsvLUT.isValid:
            divs = hsvLUT.divs
            steps = tuple([360 / divs[0],
                           1.0 / (divs[1] - 1),
                           1.0 / (divs[2] - 1)]
                          )
            interp = chosenInterp(pool, currentImage.width() * currentImage.height())
            coeffs = interp(hsvLUT.data, steps, bufHSV_CV32, convert=False)
            bufHSV_CV32[:, :, 0] = np.mod(bufHSV_CV32[:, :, 0] + coeffs[:, :, 0], 360)
            bufHSV_CV32[:, :, 1:] *= coeffs[:, :, 1:]
            np.clip(bufHSV_CV32, (0, 0, 0), (360, 1, 1), out=bufHSV_CV32)
            rawLayer.bufCache_HSV_CV32 = bufHSV_CV32.copy()

    #############
    # apply tone curve
    ############
    buf = adjustForm.dngDict.get('ProfileToneCurve', [])
    # apply profile tone curve, if any
    if buf:  # non empty list
        LUTXY = dngProfileToneCurve(buf).toLUTXY(maxrange=255)
        # bufHSV_CV32[:, :, 2] = LUTXY[(bufHSV_CV32[:, :, 2] * 255).astype(np.uint16)]
        bufHSV_CV32[:, :, 2] = np.take(LUTXY, (bufHSV_CV32[:, :, 2] * 255).astype(np.uint16))
        bufHSV_CV32[:, :, 2] /= 255.0
    # apply user tone curve
    toneForm = adjustForm.toneForm
    if toneForm is not None:
        if toneForm.isVisible():
            userLUTXY = toneForm.scene().quadricB.LUTXY
            # bufHSV_CV32[:, :, 2] = userLUTXY[(bufHSV_CV32[:, :, 2] * 255).astype(np.uint16)]
            bufHSV_CV32[:, :, 2] = np.take(userLUTXY, (bufHSV_CV32[:, :, 2] * 255).astype(np.uint16))
            bufHSV_CV32[:, :, 2] /= 255

    rawLayer.bufCache_HSV_CV32 = bufHSV_CV32.copy()  # CAUTION : must be outside of if toneForm.

    # beginning of the contrast-saturation phase : update buffer from the last camera profile application
    bufHSV_CV32 = rawLayer.bufCache_HSV_CV32.copy()

    ###########
    # contrast (V channel) and saturation correction.
    # We apply an automatic histogram equalization
    # algorithm, well suited for multimodal histograms.
    ###########
    if adjustForm.contCorrection > 0:
        # warp should be in range 0..1.
        # warp = 0 means that no additional warping is done, but
        # the histogram is always stretched.
        warp = max(0, (adjustForm.contCorrection - 1)) / 10
        bufHSV_CV32[:, :, 2], a, b, d, T = warpHistogram(bufHSV_CV32[:, :, 2],
                                                         valleyAperture=0.05,
                                                         warp=warp,
                                                         preserveHigh=options['Preserve Highlights'],
                                                         spline=None if rawLayer.autoSpline else rawLayer.getMmcSpline()
                                                         )
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
        # bufHSV_CV32[:, :, 1] = LUT[(bufHSV_CV32[:, :, 1] * 255).astype(int)]
        bufHSV_CV32[:, :, 1] = np.take(LUT, (bufHSV_CV32[:, :, 1] * 255).astype(int))

    # back to RGB
    bufpostF32_1 = cv2.cvtColor(bufHSV_CV32, cv2.COLOR_HSV2RGB)

    # apply gamma curve
    # Gamma constants are defined in colorCIE.py
    bufpostF32_255 = rgbLinear2rgb(bufpostF32_1)

    # Conversion to 8 bits/channel
    bufpostUI8 = bufpostF32_255.astype(np.uint8)

    if rawLayer.parentImage.useThumb:
        bufpostUI8 = cv2.resize(bufpostUI8, (currentImage.width(), currentImage.height()))

    bufOut = QImageBuffer(currentImage)
    bufOut[:, :, :3][:, :, ::-1] = bufpostUI8
    # base layer : no need to forward the alpha channel
    rawLayer.updatePixmap()
