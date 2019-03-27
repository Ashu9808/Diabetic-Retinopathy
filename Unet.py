from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import os
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from PIL import Image
import glob

from torch.autograd import Variable
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
from scipy.special import expit
import matplotlib.image as mpimg

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
#     def _get_padding(size, kernel_size=3, stride=1, dilation=1): 
#         padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) //2 
#         return padding

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StackEncoder, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout=nn.Dropout2d(.5,inplace=False)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace


class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size):
        super(StackDecoder, self).__init__()

        self.upSample = nn.ConvTranspose2d(in_channels,out_channels, kernel_size=(3,3),padding = 1,output_padding=1,stride=2)
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        # Crop + concat step between these 2
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)

    def _crop_concat(self, upsampled, bypass):
        """
         Crop y to the (h, w) of x and concat them.
         Used for the expansive path.
        Returns:
            The concatenated tensor
        """
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        #print(x.shape)
        x = self.upSample(x)
        #print(x.shape)
        
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)
        return x


class UNetOriginal(nn.Module):
    def __init__(self, in_channels=3):
        super(UNetOriginal, self).__init__()
#         channels, height, width = in_shape

        self.down1 = StackEncoder(in_channels, 16)
        self.down2 = StackEncoder(16, 32)
        self.down3 = StackEncoder(32, 64)
        self.down4 = StackEncoder(64, 128)
        self.down5 = StackEncoder(128, 256)
        self.down6 = StackEncoder(256, 512)
        

        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=1),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=1)
        )

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, upsample_size=(16, 16))
        self.up2 = StackDecoder(in_channels=512, out_channels=256, upsample_size=(32, 32))
        self.up3 = StackDecoder(in_channels=256, out_channels=128, upsample_size=(64, 64))
        self.up4 = StackDecoder(in_channels=128, out_channels=64, upsample_size=(128, 128))
        self.up5 = StackDecoder(in_channels=64, out_channels=32, upsample_size=(256, 256))
        self.up6 = StackDecoder(in_channels=32, out_channels=16, upsample_size=(512, 512))
        # 1x1 convolution at the last layer
        # Different from the paper is the output size here
        self.output_seg_map = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.sig = nn.Sigmoid()
#         self.rel = nn.ReLU()

    def forward(self, x):
        x, x_trace1 = self.down1(x)  
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)
        x, x_trace5 = self.down5(x)
        x, x_trace6 = self.down6(x)
        

        x = self.center(x)

        x = self.up1(x, x_trace6)
        
        x = self.up2(x, x_trace5)
        x = self.up3(x, x_trace4)
        x = self.up4(x, x_trace3)
        x = self.up5(x, x_trace2)
        x = self.up6(x, x_trace1)

        out = self.output_seg_map(x)
        #out = torch.squeeze(out, dim=1)
        return out
