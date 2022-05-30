# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

class ConvLayer(nn.Sequential):
    '''

        Convolution Layer that applies a spectral normalization as it 
    is suggested and turned out to be improved the performance. Modified Conv
    Layer has `downsample` option that will be set True when ConvBlocks are created.
    
    
        As activation function, we try LeakyReLU, and its variants:
            
            - Standard LeakyReLU with 0.2 negative slope.
            - FusedLeakyReLU
            - ScaledLeakyReLU
    '''
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        bias=True,
        activate=True,
    ):
        layers = []

        # Downsample set to True : set Stride 2, Padding 0
        if downsample:
            stride = 2
            self.padding = 0
        
        # Downsample set to False : Add 0 paddings to preserve spatial dimensions
        else:
            stride = 1
            self.padding = kernel_size // 2
        
        # Apply spectral normalization along with convolution
        layers.append(
            spectral_norm(nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias= bias and not activate,
            ))
        )
        
        # In this configuration try LeakyReLU but might be changed to another variant of
        # LeakyReLU as well. TODO: Try ScaledLeakyReLU
        if activate:
            layers.append(nn.LeakyReLU(0.2))
            

        super().__init__(*layers)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, sn=True):
        super().__init__()

        # First convolution: Preserve spatial dimensionality
        self.conv1 = ConvLayer(in_channel, in_channel, kernel_size = 3, sn=sn)
        # Second convolution: Downsample by the factor of 2
        self.conv2 = ConvLayer(in_channel, out_channel, kernel_size = 3, downsample=True, sn=sn)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out



class Discriminator(nn.Module):
    '''
    Quote From the paper:
    >   "... Since the discriminator severely affects the stability of adversarial training, we opt to use a Conv-based discriminator..."
    
    In discriminator, in order to reduce the artifacts effect authors proposed to use Wavelet discriminator, which applies a Inverse
    Haar Transformation and Haar transformation back to the image. 

    In this implementation we do not concern the articat effects in the image, furthermore we, indeed, want to observe them clearly.
    For this reason, discriminator will consists of only conv layers. 
    '''
    def __init__(self, size, channel_multiplier=2):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

    
    def forward(self, input):
        pass
