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
import argparse

from bLUeGui.bLUeImage import QImageBuffer

from bLUeNN.models_x import *
import torchvision.transforms.functional as TF

"""
import trilinear

trilinear_ = TrilinearInterpolation()


parser = argparse.ArgumentParser()
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--model_dir", type=str, default="bLUeNN/pretrained_models", help="directory of pretrained models")

opt = parser.parse_args()
opt.model_dir = opt.model_dir + '/' + opt.input_color_space
"""

model_dir = "bLUeNN/pretrained_models/sRGB"

# Tensor type
Tensor = torch.FloatTensor

# criterion_pixelwise = torch.nn.MSELoss()
LUT0 = Generator3DLUT(instance='identity')
LUT1 = Generator3DLUT()
LUT2 = Generator3DLUT()

classifier = Classifier()

# Load pretrained models
LUTs = torch.load("%s/LUTs.pth" % model_dir, map_location=torch.device('cpu'))
LUT0.load_state_dict(LUTs["0"])
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])

LUT0.eval()
LUT1.eval()
LUT2.eval()

classifier.load_state_dict(torch.load("%s/classifier.pth" % model_dir, map_location=torch.device('cpu')))
classifier.eval()


def generatePred(img):
    """
    Uses the pretrained model to classify img
    @param img: tensor image. Channels order is RGB and values are in range [0,1]
    @type img:
    @return: LUT tensor. Axes and channels are in RGB order and values should be in range [0,1]
    @rtype:
    """

    pred = classifier(img).squeeze()

    return pred


def generateLUTfromQImage(img, coeffs):
    """
    Computes 3D LUT from QImage
    @param img:
    @type img: QImage
    @param coeffs:
    @type coeffs: 3-uple
    @return: 3D LUT and pred
    @rtype:
    """

    buf = QImageBuffer(img)[:, :, :3][:, :, ::-1].copy()  # convert to RGB. Tensor does not support negative strides
    img1 = TF.to_tensor(buf).type(Tensor)  # convert ndarray with dtype np.uint8 to tensor in range [0,1]
    img1 = img1.unsqueeze(0)

    pred = generatePred(img1)

    scale = 1.0 / 500.0

    pred1 = pred[0] * (1 + coeffs[0] * scale), pred[1] * (1 + coeffs[1] * scale), pred[2] * (1 + coeffs[2] * scale)

    LUT = pred1[0] * LUT0.LUT + pred1[1] * LUT1.LUT + pred1[2] * LUT2.LUT

    return np.ascontiguousarray(LUT.detach().numpy().transpose(1, 2, 3, 0)[:, :, :, ::-1] * 255), pred1
