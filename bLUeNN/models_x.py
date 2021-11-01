##############################################################################################################
#This file is a modified version of https://github.com/HuiZeng/Image-Adaptive-3DLUT/blob/master/models_x.py  #
#The original file can be distributed and/or modified under the terms of the Apache-2.0 License.             #
##############################################################################################################

import torch.nn as nn
import torch

import numpy as np

from bLUeCore.cartesian import cartesianProduct

def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers


class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            #*discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)

class Classifier_unpaired(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier_unpaired, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            #*discriminator_block(128, 128),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Generator3DLUT(nn.Module):
    def __init__(self, dim=33, instance='zeros'):
        super().__init__()

        # sanity check
        if ((dim - 1) & (dim - 2)) != 0:
            raise ValueError("LUT3D : size should be 2**n+1, found %d" % dim)

        if instance == 'identity':
            maxrange = 256

            # interpolation step
            step = maxrange / (dim - 1)
            if not step.is_integer():
                raise ValueError('LUT3D : wrong size')

            a = np.arange(dim, dtype=np.int16) * np.float32(step)
            buffer = cartesianProduct((a, a, a))
            buffer = np.moveaxis(buffer, [0, 1, 2, 3], [1, 2, 3, 0])
        else:
            buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        # self.TrilinearInterpolation = TrilinearInterpolation()  not needed to apply a pretrained classifier

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output


