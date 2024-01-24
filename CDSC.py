
"""
@FileName: CDSC.py
@Author: Chenghong Xiao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ComplexConv1d

class CDSC1d(nn.Module):
    """
    The Complex-Valued Depthwise Separable Convolution (CDSC)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        """ Initialize a CDSC.

            Description of the Structure:
            A CDSC factorizes one regular convolution into one depthwise convolution (DWC) in the spatial dimension
            and one pointwise convolution (PWC) in the channel dimension.
            We perform the real-valued DWC in the spatial dimension and the complex-valued PWC in the channel dimension.
        """

        super(CDSC1d, self).__init__()

        self.DWC = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.PWC = ComplexConv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.DWC(x)
        x = self.PWC(x)
        return x
